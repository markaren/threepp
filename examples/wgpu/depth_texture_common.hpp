// Shared scene + post-process shader for the GL/WGPU DepthTexture comparison.
// Deterministic (fixed-seed RNG) so both backends render the exact same scene.

#ifndef THREEPP_DEPTH_TEXTURE_COMMON_HPP
#define THREEPP_DEPTH_TEXTURE_COMMON_HPP

#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/threepp.hpp"

#include <threepp/geometries/TorusKnotGeometry.hpp>

#include <cmath>

namespace depthdemo {

    constexpr int WIDTH = 512;
    constexpr int HEIGHT = 512;

    // Tiny deterministic LCG so GL and WGPU place identical geometry.
    struct Rng {
        unsigned int s = 1234567u;
        float next() {
            s = s * 1664525u + 1013904223u;
            return static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
        }
    };

    inline void setupScene(threepp::Scene& scene) {
        using namespace threepp;

        const auto geometry = TorusKnotGeometry::create(1, 0.3, 128, 64);
        const auto material = MeshBasicMaterial::create(MeshBasicMaterial::Params{}.color(Color::blue));

        Rng rng;
        const int count = 50;
        const float scale = 5;

        for (int i = 0; i < count; i++) {
            const float r = rng.next() * 2.0f * math::PI;
            const float z = rng.next() * 2.0f - 1.0f;
            const float zScale = std::sqrt(1.0f - z * z) * scale;

            const auto mesh = Mesh::create(geometry, material);
            mesh->position.set(std::cos(r) * zScale, std::sin(r) * zScale, z * scale);
            mesh->rotation.set(rng.next(), rng.next(), rng.next());
            scene.add(mesh);
        }
    }

    // Vertex shader: a flipUv uniform lets a backend that samples render targets
    // with a flipped Y origin (WebGPU) match one that doesn't (OpenGL), so the
    // same scene produces the same image. Driven by Renderer::renderTargetFlipY().
    inline const char* postVertexShader() {
        return R"(
        varying vec2 vUv;
        uniform float flipUv;
        void main() {
            vUv = vec2(uv.x, mix(uv.y, 1.0 - uv.y, flipUv));
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
    )";
    }

    inline const char* postFragmentShader() {
        return R"(
        varying vec2 vUv;
        uniform sampler2D tDepth;
        uniform float cameraNear;
        uniform float cameraFar;

        float perspectiveDepthToViewZ(float invClipZ, float near, float far) {
            return (near * far) / ((far - near) * invClipZ - far);
        }
        float viewZToOrthographicDepth(float viewZ, float near, float far) {
            return (viewZ + near) / (near - far);
        }

        void main() {
            // Sample at point of use: the Wgpu GLSL translator rebuilds a combined
            // sampler2D via a macro, which can't be passed as a function argument.
            float fragCoordZ = texture2D(tDepth, vUv).x;
            float viewZ = perspectiveDepthToViewZ(fragCoordZ, cameraNear, cameraFar);
            float depth = viewZToOrthographicDepth(viewZ, cameraNear, cameraFar);
            gl_FragColor.rgb = 1.0 - vec3(depth);
            gl_FragColor.a = 1.0;
        }
    )";
    }

}// namespace depthdemo

#endif// THREEPP_DEPTH_TEXTURE_COMMON_HPP
