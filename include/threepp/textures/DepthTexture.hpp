// https://github.com/mrdoob/three.js/tree/r129/src/textures

#ifndef THREEPP_DEPTHTEXTURE_HPP
#define THREEPP_DEPTHTEXTURE_HPP

#include "Texture.hpp"

namespace threepp {

    class DepthTexture: public Texture {

    public:
        static std::shared_ptr<DepthTexture> create(std::optional<Type> type = std::nullopt, Format format = Format::Depth);

    private:
        DepthTexture(std::optional<Type> type, Format format);
    };

}// namespace threepp

#endif//DEPTHTEXTURE_HPP
