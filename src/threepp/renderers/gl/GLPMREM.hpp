// Prefiltered Mipmapped Radiance Environment Map for GL.
//
// Produces a CubeUV-packed 2D texture (768x768 RGBA16F) compatible with
// three.js `cube_uv_reflection_fragment.glsl`. The 11 LODs are arranged so
// roughness -> mip -> packed-uv works without texture mipmap chains.
//
// Input: equirectangular HDR 2D texture (e.g. from RGBELoader).
// Output: RenderTarget whose Texture has Mapping::CubeUVReflection.
//
// Convolution is GGX importance sampling per LOD (simpler than three.js's
// separable Gaussian blur; the end visual is equivalent for typical HDR envs).

#ifndef THREEPP_GLPMREM_HPP
#define THREEPP_GLPMREM_HPP

#include "threepp/renderers/GLRenderTarget.hpp"

#include <memory>

namespace threepp {

    class GLRenderer;
    class Texture;

    namespace gl {

        class GLPMREM {

        public:
            explicit GLPMREM(GLRenderer& renderer);
            ~GLPMREM();

            // Build a PMREM from an equirectangular 2D HDR texture.
            // Returns a RenderTarget owning a 2D texture with mapping CubeUVReflection.
            std::unique_ptr<RenderTarget> fromEquirectangular(Texture& equirect);

        private:
            GLRenderer& renderer;

            struct Impl;
            std::unique_ptr<Impl> impl;
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLPMREM_HPP
