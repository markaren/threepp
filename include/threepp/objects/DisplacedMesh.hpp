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
    };

}// namespace threepp

#endif//THREEPP_DISPLACEDMESH_HPP
