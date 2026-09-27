// https://github.com/mrdoob/three.js/blob/r186/src/textures/CubeDepthTexture.js

#ifndef THREEPP_CUBEDEPTHTEXTURE_HPP
#define THREEPP_CUBEDEPTHTEXTURE_HPP

#include "threepp/textures/DepthTexture.hpp"

namespace threepp {

    // A depth texture with six faces, attached as the depth buffer of a cube
    // render target. Point-light shadows render into one and sample it through
    // a samplerCubeShadow (or a plain samplerCube for ShadowMap::Basic).
    class CubeDepthTexture: public DepthTexture {

    public:
        static std::shared_ptr<CubeDepthTexture> create(unsigned int size, Type type = Type::UnsignedInt) {
            return std::shared_ptr<CubeDepthTexture>(new CubeDepthTexture(size, type));
        }

    private:
        CubeDepthTexture(unsigned int size, Type type)
            : DepthTexture(type, Format::Depth) {
            image() = Image(std::vector<unsigned char>{}, size, size);
        }
    };

}// namespace threepp

#endif//THREEPP_CUBEDEPTHTEXTURE_HPP
