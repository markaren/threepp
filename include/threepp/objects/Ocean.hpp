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
        struct Options {
            // Mesh extent in metres (square tile). Doubles as the cascade-0 FFT
            // tile, i.e. the wavelength of the largest swell.
            float size = 1000.0f;

            // Vertex grid resolution per side (total vertices = resolution²).
            // Decoupled from the FFT size — the wave field stays at fftSize²
            // regardless, so the mesh can be coarser than the spectrum.
            uint32_t resolution = 512;

            // Wind direction (radians, 0 = +X) and speed (m/s). Speed is the
            // dominant "how big is the sea" lever (Phillips amplitude ∝ V⁴):
            // ~8–10 = a moderate Beaufort 4–5 sea, ~20 = a gale.
            float windTheta = 0.6f;
            float windSpeed = 10.0f;

            // Horizontal "choppiness" (sharper crests). ~0.45 is realistic;
            // ≥ 0.8 starts to fold wave crests into white spikes.
            float choppiness = 0.55f;

            // Global wave-height multiplier; 1.0 is physical.
            float waveScale = 1.0f;

            // Mid + fine cascade tile sizes (metres). Cascade-0 uses `size`.
            float tileSize1 = 100.0f;
            float tileSize2 = 8.0f;

            // Cascade-0 FFT resolution (power of two). Cascades 1 & 2 use half.
            uint32_t fftSize = 1024;
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

        // Convenience setters that forward to `params`.
        void setWind(float speed, float theta);

        Ocean(const std::shared_ptr<BufferGeometry>& geometry,
              const std::shared_ptr<Material>& material);

    private:
        float halfExtent_ = 500.0f;// size/2 — default warp halfRange
    };

}// namespace threepp

#endif//THREEPP_OCEAN_HPP
