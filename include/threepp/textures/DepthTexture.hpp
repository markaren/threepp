// https://github.com/mrdoob/three.js/tree/r129/src/textures

#ifndef THREEPP_DEPTHTEXTURE_HPP
#define THREEPP_DEPTHTEXTURE_HPP

#include "Texture.hpp"

namespace threepp {

    class DepthTexture: public Texture {

    public:
        // Depth comparison for sampling through a shadow sampler
        // (sampler2DShadow / samplerCubeShadow). When set, the texture is
        // uploaded with GL_TEXTURE_COMPARE_MODE = COMPARE_REF_TO_TEXTURE and this
        // function, so a Linear-filtered lookup returns hardware 2x2 PCF. Unset
        // (the default) samples the raw depth, as a plain sampler2D needs.
        // As three.js r186's DepthTexture.compareFunction.
        std::optional<DepthFunc> compareFunction;

        static std::shared_ptr<DepthTexture> create(std::optional<Type> type = std::nullopt, Format format = Format::Depth);

    protected:
        DepthTexture(std::optional<Type> type, Format format);
    };

}// namespace threepp

#endif//DEPTHTEXTURE_HPP
