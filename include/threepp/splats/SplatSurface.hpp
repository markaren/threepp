// A Gaussian-splat scan becomes a triangle SURFACE, by depth fusion.
//
// A SplatCloud renders and reflects, but it is in no acceleration structure and
// has no geometry: nothing can stand on it, no lidar beam returns from it. This
// bake renders the cloud from a set of poses, reads the MEDIAN-depth AOV
// (VulkanRenderer::SplatDepthMode::Median — the transmittance-0.5 crossing, the
// only exported statistic that estimates a surface rather than an opacity
// centroid), fuses the depth maps into a block-sparse truncated signed distance
// field, and extracts an isosurface. One bake, three consumers: a PhysX static
// trimesh (PhysxWorld::addStaticTrimesh), a lidar target, a depth-sensor target.
// See plans/splat-surface-bake.md.
//
// DETERMINISM is the contract, not a nicety: the sensor goldens downstream of
// this are byte-exact, and the splat rasterizer went to some trouble to be
// (SplatPass.hpp). Integration is sequential in POSE-LIST ORDER, the marching
// cubes walks blocks in sorted key order, and the same cloud baked twice gives
// the same vertices and the same indices, bit for bit.
//
// Vulkan only — the depth AOV is a Vulkan G-buffer attachment.

#ifndef THREEPP_SPLATSURFACE_HPP
#define THREEPP_SPLATSURFACE_HPP

#include "threepp/math/Vector3.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace threepp {

    class Mesh;
    class SplatCloud;
    class VulkanRenderer;

}// namespace threepp

namespace threepp::splats {

    // One capture viewpoint. `fov` is vertical, in degrees, as PerspectiveCamera
    // takes it; the bake renders at the renderer's own framebuffer extent, so
    // the horizontal field follows from that aspect.
    struct BakePose {

        Vector3 position;
        Vector3 target;
        Vector3 up{0.f, 1.f, 0.f};
        float fov{55.f};
    };

    struct SurfaceBakeOptions {

        // 0 = derive from the cloud's robust fit: radius / 256, clamped to
        // [5 mm, 10 cm]. The same robust fit (component-wise median centre,
        // 90th-percentile radius) that examples/objects/gaussian_splats.cpp
        // frames with — a bounding box centres on a photogrammetry outlier.
        float voxelSize{0.f};

        // Truncation half-width. 0 = truncationVoxels * voxelSize.
        float truncation{0.f};
        float truncationVoxels{4.f};

        // Running-average weight cap (KinectFusion's), and the weight below
        // which marching cubes refuses to trust a voxel. One pose contributes
        // weight 1, so the FLOOR IS COUNTED IN POSES: 2 means two viewpoints had
        // to agree. Measured on the synthetic plane (16 poses, 5 cm voxels):
        // floor 1 gives 13652 triangles at 4.8 mm RMS, floor 4 gives 13006 at
        // 2.8 mm, floor 8 gives 12424 at 1.8 mm — the shaky triangles are the
        // ones a single pose saw. The floor stays at 2 anyway, because the same
        // 4 that flattens a plane costs a SPHERE SHELL 88 % of its vertices
        // (8369 -> 1000): poses on a sphere each see a patch once. A caller with
        // dense overlapping poses can raise it; it can never be met by fewer
        // poses than its own value.
        float maxWeight{32.f};
        float weightFloor{2.f};

        // Empty = generated: a Fibonacci sphere of `poseCount` viewpoints around
        // the fit sphere for a compact scan, a ring plus a top-down grid for one
        // wider than it is tall. Supplying poses is the point for interiors and
        // for replaying a real capture trajectory — a sphere of poses looking
        // inward at a room sees the outside of its walls.
        std::vector<BakePose> poses;
        int poseCount{26};
        // 0 = 2.2 * fit radius.
        float poseDistance{0.f};

        // Islands smaller than this many surface CELLS are dropped: what
        // survives free-space carving of photogrammetry floaters.
        //
        // A cell is voxelSize^2 of SURFACE, so this is an area in disguise and
        // it does not mean what a first guess says. Measured: a stray clump 5 cm
        // across, fused at 5 cm voxels, comes out as a closed blob ~30 cm across
        // — truncation inflates it — which is ~115 cells. 256 cells (0.64 m^2 at
        // 5 cm voxels, a sphere ~45 cm across) drops that and keeps anything a
        // scan resolves as an object. Scale it with the voxel, not with the
        // scene.
        int minComponentVoxels{256};

        // Defense against the median's near-gate failure (see the AOV's own
        // documentation and plans/splat-surface-bake.md): where coverage falls
        // toward 0.5 the crossing degenerates to the LAST contributing splat and
        // lands far BEHIND the surface. The AOV exports no coverage channel, so
        // coverage is inferred from the mask: low coverage lives at the cloud's
        // silhouette fringe.
        //
        // fringeErode: drop samples within this many pixels of an uncovered
        // pixel (0 disables). outlierTolerance: drop a sample sitting more than
        // this many truncations BEHIND its 3x3 neighbourhood median (0
        // disables) — one-signed, because the failure is.
        int fringeErode{1};
        float outlierTolerance{1.f};
    };

    struct SurfaceMesh {

        // World space, xyz triples; indices are triangles.
        std::vector<float> positions;
        std::vector<uint32_t> indices;

        // What the bake dropped, because a bounded thing says what it dropped.
        struct Stats {

            int poses{0};
            float voxelSize{0.f};
            float truncation{0.f};

            uint64_t depthSamples{0};    // covered pixels offered to fusion
            uint64_t skippedFringe{0};   // ... dropped by the fringe erode
            uint64_t skippedOutlier{0};  // ... dropped as behind-neighbourhood
            uint64_t blocks{0};          // 8^3 blocks allocated
            uint64_t observedVoxels{0};  // voxels with weight >= weightFloor

            uint32_t components{0};
            uint32_t culledComponents{0};
            uint64_t culledTriangles{0};

            Vector3 aabbMin;
            Vector3 aabbMax;

            double renderMs{0.0};
            double fuseMs{0.0};
            double meshMs{0.0};
        };
        Stats stats;

        [[nodiscard]] bool empty() const { return indices.empty(); }
        [[nodiscard]] size_t vertexCount() const { return positions.size() / 3; }
        [[nodiscard]] size_t triangleCount() const { return indices.size() / 3; }
    };

    // Bakes whatever the renderer DRAWS for this cloud: for a multi-level LOD
    // asset that is the level the LOD policy picks at the bake distance, not
    // necessarily level 0. A caller who needs a specific level builds a
    // SplatCloud from it and bakes that.
    //
    // The cloud is temporarily reparented under a private scene (its world
    // transform is preserved) so that no other geometry can occlude it during
    // the capture, and put back — under the same parent, at the end of its child
    // list — before returning. The renderer's splat-depth AOV mode is restored
    // too. Returns an empty mesh if the cloud has no splats or the AOV never
    // read back.
    SurfaceMesh bakeSurface(VulkanRenderer& renderer, SplatCloud& cloud,
                            const SurfaceBakeOptions& options = {});

    // The baked surface as a scene object that ONLY THE SENSORS perceive: an
    // ordinary Mesh (world-space vertices, so add it at the scene root, not
    // under the cloud) marked with VulkanRenderer::kSensorOnlyLayer. The
    // primary camera never draws it — the real splats render there — and no
    // radiance trace can see it; the scene's lidar and its depth views can,
    // once the scene calls VulkanRenderer::setSensorOnlySurfaces(true). Until
    // it does, the mesh is inert: nothing renders it and nothing senses it.
    //
    // The layer is set EXCLUSIVELY (Layers::set), which is also what keeps the
    // GL renderer — which honours camera-vs-object layers — from drawing it to
    // a default camera. Empty mesh in, nullptr out.
    std::shared_ptr<Mesh> makeSensorMesh(const SurfaceMesh& surface);

}// namespace threepp::splats

#endif//THREEPP_SPLATSURFACE_HPP
