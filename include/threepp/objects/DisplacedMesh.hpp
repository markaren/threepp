// A mesh whose vertex positions are GPU-displaced every frame from
// height/displacement textures. Designed for use with the Vulkan path
// tracer: the renderer recognises the type, runs the displacement
// compute pass, and rebuilds the BLAS in-place every frame.
//
// Typical usage (see examples/projects/Ocean):
//
//   auto plane = PlaneGeometry::create(40.f, 40.f, 255, 255);
//   plane->rotateX(-math::PI / 2.f);
//   auto mat   = MeshPhysicalMaterial::create();
//   mat->setIor(1.33f);
//   mat->transmission = 1.f;
//   mat->roughness    = 0.05f;
//   auto ocean = DisplacedMesh::create(plane, mat);
//   scene->add(ocean);
//
// The renderer looks up the FFT cascade textures by (currently) a single
// known channel — this header exposes only the public-facing knobs;
// renderer-side state lives in VulkanRenderer.cpp's DisplacedMeshState.

#ifndef THREEPP_DISPLACEDMESH_HPP
#define THREEPP_DISPLACEDMESH_HPP

#include "threepp/objects/Mesh.hpp"

#include <cstdint>
#include <vector>

namespace threepp {

    class DisplacedMesh : public Mesh {

    public:
        struct Params {
            // World-space tile sizes for each cascade (the FFT texture wraps
            // once per tile). 0 disables that cascade; cascade 0 must be > 0.
            float tileSize0 = 40.0f;
            float tileSize1 =  0.0f;
            float tileSize2 =  0.0f;

            // Wind direction (radians, 0 = +X) and strength (m/s). Affects
            // wave anisotropy and overall amplitude via the Phillips spectrum.
            float windTheta = 0.0f;
            float windSpeed = 12.0f;

            // Global Y-displacement multiplier. 1.0 is physical; higher values
            // exaggerate wave height.
            float waveScale = 1.0f;

            // Horizontal-displacement multiplier ("choppiness"). 0 is a pure
            // height-field, 1 is full Tessendorf horizontal pull (sharp crests).
            // Real ocean falls around 0.4 — anything ≥ 0.8 starts producing
            // wave-folding artefacts (white spike crests).
            float choppiness = 0.45f;

            // FFT texture resolution (must be power of two; 256 is the
            // WebTide default; 128 is a good budget tradeoff).
            uint32_t textureSize = 256;
        };

        Params params;

        // Hull exclusion zone: suppresses wave displacement inside a
        // world-space oriented rectangle so the ocean doesn't clip through
        // a vessel's deck. Set each frame before render().
        struct HullExclusion {
            float centerX    = 0.f;
            float centerZ    = 0.f;
            float halfLength = 0.f;   // 0 = disabled
            float halfBeam   = 0.f;
            float sinYaw     = 0.f;
            float cosYaw     = 1.f;
        };
        HullExclusion hullExclusion;

        // Vessel wake: per-frame state used by water_displace.comp to inject a
        // Kelvin V-wake (geometric height), a bow bump, and a foam trail behind
        // the vessel. Reuses the HullExclusion pose (centerX/Z, sin/cosYaw,
        // halfLength, halfBeam) — only forward speed is wake-specific. Set
        // forwardSpeed = 0 (or enabled = false) to disable on stationary
        // vessels (buoys, anchored ships) that still need a hull exclusion.
        struct VesselWake {
            float forwardSpeed = 0.f;   // m/s along +heading; 0 disables wake
            bool  enabled      = true;
        };
        VesselWake wake;

        DisplacedMesh(const std::shared_ptr<BufferGeometry>& geometry,
                      const std::shared_ptr<Material>& material);

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<DisplacedMesh> create(
                const std::shared_ptr<BufferGeometry>& geometry,
                const std::shared_ptr<Material>& material) {
            return std::make_shared<DisplacedMesh>(geometry, material);
        }

        // Bumped each render() so the renderer's dirty-detect always treats
        // this mesh as moved (it is — FFT textures advance every frame).
        // Internal; do not mutate from user code.
        uint64_t frameTick = 0;

        // Renderer-filled spatial-domain height fields, one per cascade,
        // copied back from GPU each frame after the IFFT pass. Layout:
        // row-major RG32F (R = vertical displacement, G = unused).
        // Cell (ix, iz) lives at index `(iz*dim + ix)*2`.
        // mutable so const-method `sampleHeight` can be called on a const
        // DisplacedMesh while the renderer keeps the data fresh.
        struct CascadeField {
            std::vector<float> data;
            uint32_t dim      = 0;
            float    tileSize = 0.f;
        };
        mutable CascadeField heightFields[3];

        // Bilinear-sample the combined wave height at (worldX, worldZ).
        // Matches the GPU's sampleDisplacement().y exactly: each enabled
        // cascade contributes height * (1/tileSize) * waveScale.
        //
        // `cascadeMask` is a bitmask (bit i selects cascade i); the default
        // 0b111 sums all three. Use a narrower mask for hull-scale buoyancy
        // queries on long vessels — sampling the fine cascade at the bow
        // and stern of a 28 m hull picks up 4-8 m chop that wouldn't
        // physically induce pitch on a hull that long ("car on rocks"
        // feel). Masking out cascade 2 for the buoyancy sample (visual
        // remains unchanged — GPU still renders all cascades) restores
        // the integrating behaviour of a long hull.
        float sampleHeight(float worldX, float worldZ,
                           uint32_t cascadeMask = 0b111u) const;
    };

}// namespace threepp

#endif//THREEPP_DISPLACEDMESH_HPP
