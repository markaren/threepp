// https://github.com/mrdoob/three.js/blob/r129/src/renderers/shaders/ShaderLib.js

#ifndef THREEPP_SHADERLIB_HPP
#define THREEPP_SHADERLIB_HPP

#include "threepp/renderers/shaders/Shader.hpp"
#include "threepp/renderers/shaders/ShaderChunk.hpp"

#include "threepp/renderers/shaders/mergeuniforms.hpp"

namespace threepp::shaders {

    class ShaderLib {

        Shader basic;

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
