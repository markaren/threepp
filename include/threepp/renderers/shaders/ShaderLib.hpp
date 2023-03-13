// https://github.com/mrdoob/three.js/blob/r129/src/renderers/shaders/ShaderLib.js

#ifndef THREEPP_SHADERLIB_HPP
#define THREEPP_SHADERLIB_HPP

#include "threepp/core/Shader.hpp"

namespace threepp::shaders {

    class ShaderLib {

    public:
        Shader basic;
        Shader lambert;
        Shader phong;
        Shader standard;
        Shader toon;
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

        [[nodiscard]] Shader& get(const std::string& name) {

            if (name == "basic") {
                return basic;
            } else if (name == "lambert") {
                return lambert;
            } else if (name == "phong") {
                return phong;
            } else if (name == "standard") {
                return standard;
            } else if (name == "toon") {
                return toon;
            } else if (name == "matcap") {
                return matcap;
            } else if (name == "points") {
                return points;
            } else if (name == "dashed") {
                return dashed;
            } else if (name == "depth") {
                return depth;
            } else if (name == "normal") {
                return normal;
            } else if (name == "sprite") {
                return sprite;
            } else if (name == "background") {
                return background;
            } else if (name == "cube") {
                return cube;
            } else if (name == "equirect") {
                return equirect;
            } else if (name == "distanceRGBA") {
                return distanceRGBA;
            } else if (name == "shadow") {
                return shadow;
            } else if (name == "physical") {
                return physical;
            } else {
                throw std::runtime_error("No shader with name: " + name);
            }
        }

        ShaderLib(const ShaderLib&) = delete;
        void operator=(const ShaderLib&) = delete;

        static ShaderLib& instance() {
            static ShaderLib instance;
            return instance;
        }

    private:
        ShaderLib();
    };

}// namespace threepp::shaders

#endif//THREEPP_SHADERLIB_HPP
