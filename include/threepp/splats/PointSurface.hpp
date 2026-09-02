// A point cloud becomes a triangle surface, directly from its points.
//
// The twin of SplatSurface.hpp's depth-fusion bake, for the clouds that do
// not need one: a laser scan or a photogrammetry point cloud imported through
// SplatLoader::loadPointCloudPly is already a set of surface samples, with no
// fog splats to carve away and no opacity to weigh. Fusing it through a
// rendered pose loop would only add a resolution floor and a Vulkan
// dependency. This builds the surface on the CPU, from the means alone:
//
//   1. the points go into a voxel hash at `voxelSize` (default 2 x the
//      cloud's median nearest-neighbour distance, so a sheet of points fills
//      its voxels rather than leaving gaps);
//   2. every grid node within `radiusVoxels` voxels of a point gets the
//      union-of-balls value max(0, 1 - d / radius), d being the distance to
//      its nearest point — the same field extras/pointcloud/MarchingCubes.hpp
//      builds densely, here sparse, so a town-scale scan costs its surface
//      area and not its bounding volume;
//   3. marching cubes at `isolevel` over the occupied cells and their
//      neighbours, vertices welded on shared edges, triangles wound toward
//      the free side;
//   4. islands under `minComponentVoxels` cells are dropped.
//
// The result is an OFFSET surface: it sits (1 - isolevel) * radius outside
// the points, on both sides of a sheet, so a floor scanned as one layer of
// points comes back as a slab about one voxel thick with its top half a voxel
// above the samples. That offset is the price of a method with no normals
// and no free-space information; it is stated in Stats so a consumer can
// subtract it, and it is small next to a lidar's own noise at the defaults.
//
// DETERMINISTIC: cells are walked in sorted key order and the vertex and
// index arrays are a pure function of the points, the transform and the
// options. No renderer is involved, so it runs on every backend and on a
// worker thread. Output is world space, like bakeSurface's, and feeds the
// same consumers: PhysxWorld::addStaticTrimesh, splats::makeSensorMesh.

#ifndef THREEPP_POINTSURFACE_HPP
#define THREEPP_POINTSURFACE_HPP

#include "threepp/splats/SplatSurface.hpp"

#include <cstdint>

namespace threepp {

    class Matrix4;
    struct SplatData;

}// namespace threepp

namespace threepp::splats {

    struct PointSurfaceOptions {

        // 0 = 2 x the median nearest-neighbour distance of the transformed
        // points (splats::medianNeighbourSpacing), clamped to [5 mm, 0.5 m].
        // World units.
        float voxelSize{0.f};

        // Radius of the union-of-balls field, in voxels. 1 reaches the eight
        // voxels around a node and closes a regular sheet; raise it for a
        // cloud with gaps wider than its typical spacing (the offset above
        // grows with it).
        float radiusVoxels{1.f};

        // Where in [0, 1) the surface is cut. The surface sits
        // (1 - isolevel) * radius from the nearest point; 0.5 is half a voxel
        // at the default radius. Higher hugs the points tighter and breaks up
        // sooner where they thin out.
        float isolevel{0.5f};

        // Points with opacity below this are ignored. 0 keeps every point,
        // which is right for a loaded point cloud (opacity 1 throughout); a
        // Gaussian scan routed here wants a floor, and even then its fog
        // splats' centres are still points — that is what the depth-fusion
        // bake is for.
        float opacityFloor{0.f};

        // Islands smaller than this many surface CELLS are dropped. A cell is
        // voxelSize^2 of surface, so at a 10 cm voxel 64 cells is 0.64 m^2 —
        // a stray return or two, not an object.
        int minComponentVoxels{64};

        // Hard ceiling on occupied voxels. A cloud that exceeds it at the
        // chosen voxel size returns an EMPTY mesh with Stats::refusedBlocks
        // set to the count, rather than allocating without bound; the fix is
        // a larger voxel. 2^24 voxels is ~2 GB of field cache at the widest.
        uint64_t maxVoxels{1ull << 24};
    };

    // `toWorld` is applied to every mean first, so the mesh comes back in the
    // frame the physics and the sensors live in. Empty data, or data that
    // leaves no point after the opacity floor, gives an empty mesh.
    //
    // Stats fields used: voxelSize, truncation (= the field radius),
    // observedVoxels (occupied voxels), blocks (cells evaluated),
    // refusedBlocks (see maxVoxels), components, culledComponents,
    // culledTriangles, aabbMin/Max, fuseMs (the field), meshMs (marching
    // cubes and the island filter). poses is 0: nothing was rendered.
    [[nodiscard]] SurfaceMesh buildPointSurface(const SplatData& data, const Matrix4& toWorld,
                                                const PointSurfaceOptions& options = {});

    // The cloud's own data at its current world matrix (the caller keeps it
    // current, as with any Object3D).
    [[nodiscard]] SurfaceMesh buildPointSurface(const SplatCloud& cloud,
                                                const PointSurfaceOptions& options = {});

}// namespace threepp::splats

#endif//THREEPP_POINTSURFACE_HPP
