// A ready-to-use FFT-displaced ocean surface for the Vulkan renderer.
//
// Ocean is a thin convenience layer over DisplacedMesh: its create() factory
// builds the plane geometry, a tuned transmissive water material, and a sensible
// three-cascade Phillips spectrum, so a caller gets photoreal water in one line:
//
//   scene->add(Ocean::create());
//
// The surface only animates under the Vulkan backend, which recognises any
// DisplacedMesh (Ocean derives from it) and runs the FFT / displace / foam /
// BLAS pipeline every frame in both deferred and path-traced modes. On other
// renderers it is an inert flat plane wearing the water material.
//
// It is NOT tied to any "hero" object: the adaptive-density warp, foam
// disturbances, and the (inherited) vessel wake all take world coordinates.
// Point the warp at whatever you like — a vessel, the camera, or nothing — via
// warpToward() or the inherited DisplacedMesh::warp field. The full ocean
// showcase in examples/vulkan/vulkan_ocean.cpp drives those same fields from a
// boat; a stripped-down demo lives in examples/vulkan/vulkan_ocean_minimal.cpp.
//
// ROTATED / TRANSLATED oceans (e.g. a Z-up robotics world with
// rotation.x = π/2): geometry, waves and the shallow-bottom shading all work,
// but the deferred per-pixel fine-chop normal and the foam accumulator are
// sampled in world XZ (Y-up assumption), so set tileSize2 = 0 there or the
// finest cascade reads as 1-D stripes (see spot_slam.py's pond). The warp /
// hull / wake fields additionally assume an identity transform.
//
// Distinct from threepp::Water (objects/Water.hpp), which is the classic
// three.js flat reflective plane, not an FFT-displaced surface.

#ifndef THREEPP_OCEAN_HPP
#define THREEPP_OCEAN_HPP

#include "threepp/objects/DisplacedMesh.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace threepp {

    class Ocean: public DisplacedMesh {

    public:
        // Water-body material recipe. Auto keys on the larger extent: bodies
        // under 100 m get the natural-pond water (green-brown, ~2.5 m
        // visibility), larger ones the deep-ocean recipe (teal Beer-Lambert
        // absorption). Set explicitly to pin either look regardless of scale —
        // a small harbour basin that is really sea, or a big lake that should
        // read as freshwater.
        // Fjord is the SCATTERING recipe: glacial/silty water whose turquoise
        // comes from rock flour throwing skylight back at the eye, not from
        // absorption. It is the only look that sets scatterColor/
        // scatterDistance, so it is also the only one whose body arithmetic
        // differs from before this enumerator existed (Vulkan deferred only —
        // the GL water ignores the scatter fields).
        enum class Look { Auto,
                          Ocean,
                          Pond,
                          Fjord };

        struct Options {
            // Mesh extent in metres along local X. The LARGER of size/sizeZ
            // doubles as the cascade-0 FFT tile, i.e. the wavelength of the
            // largest swell.
            float size = 1000.0f;

            // Mesh extent along local Z. 0 (default) = square (`size`). A
            // rectangle spends vertices only where the water actually is —
            // e.g. a fjord channel wants ~900 × 3200 instead of 3200² (the
            // wave field is unaffected: FFT tiles are world-anchored and
            // decoupled from the mesh extent).
            float sizeZ = 0.0f;

            // Vertex grid resolution along local X (columns). Decoupled from
            // the FFT size — the wave field stays at the FFT resolutions
            // regardless, so the mesh can be coarser than the spectrum.
            uint32_t resolution = 512;

            // Vertex rows along local Z. 0 (default) = derive from resolution
            // so grid cells stay square-ish (resolution · sizeZ / size).
            uint32_t resolutionZ = 0;

            // Wind direction (radians, 0 = +X) and speed (m/s). Speed is the
            // dominant "how big is the sea" lever (Phillips amplitude ∝ V⁴):
            // ~8–10 = a moderate Beaufort 4–5 sea, ~20 = a gale.
            float windTheta = 0.6f;
            float windSpeed = 10.0f;

            // Fetch (m of open water upwind); 0 = fully developed sea. See
            // DisplacedMesh::Params::fetch — 20–40 km trades the long PM
            // swell for the steeper 10–40 m chop of a coastal sea.
            float fetch = 0.0f;

            // Horizontal "choppiness" (sharper crests). ~0.45 is realistic;
            // ≥ 0.8 starts to fold wave crests into white spikes.
            float choppiness = 0.55f;

            // Global wave-height multiplier; 1.0 is physical.
            float waveScale = 1.0f;

            // Mid + fine cascade tile sizes (metres). Cascade-0 uses `size`.
            //
            // Default −1 = AUTO: derived from `size` as size/7.87 and
            // size/107.5, each capped at the 1000 m reference trio (127 m /
            // 9.3 m). A 1000 m ocean gets exactly {127, 9.3}; anything larger
            // keeps those classic bands (chop doesn't grow with the domain);
            // a small pond gets proportionally finer ones — size = 16 →
            // {2.0, 0.15}, i.e. dm-scale ripples instead of the ocean bands
            // that used to leave cascade 0 with ZERO representable modes
            // below size ≈ 100. Set 0 to disable a cascade, > 0 to pin it.
            //
            // The resolved trio is deliberately NON-COMMENSURATE
            // (1000/127 ≈ 7.9, 127/9.3 ≈ 13.7): integer tile ratios keep every
            // repeat of a cascade phase-aligned with the one above it, which
            // reads as a visible plaid in the far field. Combined with the
            // cascade-1 domain rotation (vulkan_shared.h) the repeats stop
            // lining up with anything.
            float tileSize1 = -1.0f;
            float tileSize2 = -1.0f;

            // FFT resolution CAP (power of two). Cascade 0 and 1 are band-
            // passed by the renderer (cascade 0 carries only λ ≥ tileSize1,
            // cascade 1 only λ ≥ tileSize2), so create() derives their actual
            // resolutions from the band (~10 samples per shortest wavelength,
            // clamped to this cap / half of it) — with the default tiles that
            // is {128, 256} instead of {1024, 512}, at identical wave content.
            // The open-band fine cascade always runs at fftSize / 2.
            uint32_t fftSize = 1024;

            // Material recipe — see Look. Auto = pond under 100 m, ocean above.
            Look look = Look::Auto;
        };

        static std::shared_ptr<Ocean> create(const Options& options);

        // Default-configured ocean. Kept as a separate overload (rather than a
        // `= {}` / `= Options{}` default argument) because GCC rejects a nested
        // aggregate's default member initializers when they are evaluated inside
        // the enclosing class's complete-class context.
        static std::shared_ptr<Ocean> create();

        [[nodiscard]] std::string type() const override;

        // Pack adaptive vertex density toward a world-space focus point. The
        // mesh keeps fixed topology but clusters vertices near (worldX, worldZ)
        // — generalises the vessel-follow warp to any focus, e.g. the camera.
        // coefA: 1 = uniform, ~0.1 ≈ 10 cm centre / 2.7 m edge spacing. Call
        // each frame before render(); halfRange defaults to size/2.
        void warpToward(float worldX, float worldZ, float coefA = 0.1f);

        // Convenience setter that forwards to `params`. LIVE: the Vulkan
        // renderer detects the change and regenerates the Phillips spectra
        // next frame; the persistent noise keeps the regenerations phase-
        // correlated, so the sea morphs smoothly into the new state instead
        // of popping. Safe to drive from a UI slider every frame.
        void setWind(float speed, float theta);

        Ocean(const std::shared_ptr<BufferGeometry>& geometry,
              const std::shared_ptr<Material>& material);

    private:
        float halfExtent_ = 500.0f;// size/2 — default warp halfRange
    };

}// namespace threepp

#endif//THREEPP_OCEAN_HPP
