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
            //
            // `roughSource`, when non-null, is prefiltered into strips 1..N-1 in
            // place of `equirect` (strip 0 is always the sharp copy of
            // `equirect` itself). That is how the one-sun policy keeps the HDRI
            // sun disc visible in mirrors and the sky background while removing
            // it from every diffuse/glossy env lookup — see
            // common/EnvSunExtract.hpp.
            std::unique_ptr<RenderTarget> fromEquirectangular(Texture& equirect,
                                                              Texture* roughSource = nullptr);

            // Resample a cube texture (loaded or rendered) into an
            // equirectangular HalfFloat target of 4*faceSize x 2*faceSize, in
            // this atlas's equirect convention, with a full mip chain so it can
            // feed fromEquirectangular. `flipX` is the loaded-cube-map mirror
            // (CubeTexture::_needsFlipEnvMap); an sRGB cube is decoded to
            // linear. sigma > 0 blurs with a Gaussian of that many radians
            // (three.js PMREMGenerator.fromScene's sigma). The mapping is
            // EquirectangularReflection.
            std::unique_ptr<RenderTarget> cubeToEquirect(Texture& cube, int faceSize, float sigma = 0,
                                                         bool flipX = false);

        private:
            GLRenderer& renderer;

            struct Impl;
            std::unique_ptr<Impl> impl;
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLPMREM_HPP
