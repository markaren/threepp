// https://github.com/mrdoob/three.js/blob/r129/src/renderers/shaders/ShaderLib.js

#ifndef THREEPP_SHADERLIB_HPP
#define THREEPP_SHADERLIB_HPP

#include "threepp/renderers/shaders/Shader.hpp"
#include "threepp/renderers/shaders/ShaderChunk.hpp"

#include "threepp/renderers/shaders/UniformsLib.hpp"
#include "threepp/renderers/shaders/UniformsUtil.hpp"

namespace threepp::shaders {

    class ShaderLib {

    public:
        Shader basic{
                mergeUniforms({// clang-format off
                            UniformsLib::instance().common,
                            UniformsLib::instance().specularmap,
                            UniformsLib::instance().envmap,
                            UniformsLib::instance().aomap,
                            UniformsLib::instance().lightmap,
                            UniformsLib::instance().fog
                        }),// clang-format on

                ShaderChunk::instance().meshbasic_vert(),
                ShaderChunk::instance().meshbasic_frag(),
        };

        Shader lambert;

        Shader phong;

        Shader standard;

        Shader matcap;

        Shader points;

        Shader dashed;

        Shader depth;

        Shader normal;

        Shader sprite;

        Shader background;

        Shader cube;

        Shader equirect;

        Shader distanceRGBA;

        Shader shadow;

        Shader physical;


        ShaderLib(const ShaderLib &) = delete;
        void operator=(const ShaderLib &) = delete;

        static ShaderLib &instance() {
            static ShaderLib instance;
            return instance;
        }

    private:
        ShaderLib() = default;
    };

}// namespace threepp::shaders

#endif//THREEPP_SHADERLIB_HPP
