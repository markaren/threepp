// https://github.com/mrdoob/three.js/blob/r129/src/textures/DepthTexture.js

#ifndef THREEPP_DEPTHTEXTURE_HPP
#define THREEPP_DEPTHTEXTURE_HPP

#include "threepp/textures/Texture.hpp"

namespace threepp {

    class DepthTexture: public Texture {

    protected:
        DepthTexture(int width, int height, int type, int mapping, int wrapS, int wrapT, int magFilter, int minFilter, int anisotropy, int format)
            : Texture(std::nullopt) {
        }
    };

}// namespace threepp

#endif//THREEPP_DEPTHTEXTURE_HPP
