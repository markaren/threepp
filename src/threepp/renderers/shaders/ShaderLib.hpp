// https://github.com/mrdoob/three.js/blob/r129/src/renderers/shaders/ShaderLib.js

#ifndef THREEPP_SHADERLIB_HPP
#define THREEPP_SHADERLIB_HPP

#include "threepp/renderers/shaders/Shader.hpp"
#include "threepp/renderers/shaders/ShaderChunk.hpp"

#include "threepp/renderers/shaders/UniformsLib.hpp"
#include "threepp/renderers/shaders/UniformsUtil.hpp"

namespace threepp::shaders {

    using UniformsLib = UniformsLib::getInstance;
    using ShaderChunk = ShaderChunk::getInstance;

    class ShaderLib {

        Shader basic{
                mergeUniforms({// clang-format off
                            UniformsLib()->common,
                            UniformsLib()->specularmap,
                            UniformsLib().envmap,
                            UniformsLib().aomap,
                            UniformsLib().lightmap,
                            UniformsLib()->fog
                        }),// clang-format on

                ShaderChunk().get("meshbasic_vert"),
                ShaderChunk().get("meshbasic_frag")};

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


    public:
        static ShaderLib *getInstance() {
            if (!instance_) {
                instance_ = new ShaderLib();
            }
            return instance_;
        }

    private:
        static ShaderLib *instance_;

        ShaderLib() = default;
    };

    ShaderLib *ShaderLib::instance_ = nullptr;

}// namespace threepp::shaders

#endif//THREEPP_SHADERLIB_HPP
