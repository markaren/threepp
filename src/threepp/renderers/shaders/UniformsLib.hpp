// https://github.com/mrdoob/three.js/blob/r129/src/renderers/shaders/UniformsLib.js

#ifndef THREEPP_UNIFORMSLIB_HPP
#define THREEPP_UNIFORMSLIB_HPP


#include "threepp/core/Uniform.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"

#include <optional>
#include <unordered_map>

namespace threepp::shaders {

    class UniformsLib {

    public:
        std::unordered_map<std::string, Uniform> common = {
                {"diffuse", Uniform(Color(0xffffff))},
                {"opacity", Uniform(1.f)},
                {"map", Uniform()},
                {"uvTransform", Uniform(Matrix3())},
                {"alphaMap", Uniform()}};

        std::unordered_map<std::string, Uniform> specularmap = {
                {"specularMap", Uniform()}};

        std::unordered_map<std::string, Uniform> envmap = {
                {"envmap", Uniform()},
                {"flipEnvMap", Uniform(-1.f)},
                {"reflectivity", Uniform()},
                {"refractionRatio", Uniform(0.98f)},
                {"maxMipMapLevel", Uniform(0)}};

        std::unordered_map<std::string, Uniform> aomap = {
                {"aomap", Uniform()},
                {"aoMapIntensity", Uniform(1)}};

        std::unordered_map<std::string, Uniform> lightmap = {
                {"lightMap", Uniform()},
                {"lightMapIntesity", Uniform(1)}};

        std::unordered_map<std::string, Uniform> emissivemap = {
                {"emissiveMap", Uniform()}};

        std::unordered_map<std::string, Uniform> bumpmap = {
                {"bumpMap", Uniform()},
                {"bumpScale", Uniform(1)}};

        std::unordered_map<std::string, Uniform> normalmap = {
                {"normalmap", Uniform()},
                {"normalScale", Uniform(Vector2(1, 1))}};

        std::unordered_map<std::string, Uniform> displacementmap = {
                {"displacementMap",Uniform()},
                {"displacementScale", Uniform(1)},
                {"displacementBias", Uniform(0)}
        };

        std::unordered_map<std::string, Uniform> roughnessmap = {
                {"roughnessMap",Uniform()}
        };

        std::unordered_map<std::string, Uniform> metalnessmap = {
                {"metalnessMap",Uniform()}
        };

        std::unordered_map<std::string, Uniform> gradientmap = {
                {"gradientMap",Uniform()}
        };

        std::unordered_map<std::string, Uniform> fog = {
                {"fogDensity",Uniform(0.00025f)},
                {"fogNear", Uniform(1.f)},
                {"fogFar", Uniform(2000.f)},
                {"fogColor", Uniform(Color(0xffffff))}
        };

        std::unordered_map<std::string, Uniform> points = {
                {"diffuse",Uniform(Color(0xffffff))},
                {"opacity", Uniform(1.f)},
                {"size", Uniform(1.f)},
                {"scale", Uniform(1.f)},
                {"map", Uniform()},
                {"alphaMap", Uniform()},
                {"uvTransform", Uniform(Matrix3())}
        };

        std::unordered_map<std::string, Uniform> sprite = {
                {"diffuse",Uniform(Color(0xffffff))},
                {"opacity", Uniform(1.f)},
                {"center", Uniform(Vector2(0.5f, 0.5f))},
                {"rotation", Uniform(0.f)},
                {"map", Uniform()},
                {"alphaMap", Uniform()},
                {"uvTransform", Uniform(Matrix3())}
        };

        static UniformsLib *getInstance() {
            if (!instance_) {
                instance_ = new UniformsLib();
            }
            return instance_;
        }

    private:
        static UniformsLib *instance_;

        UniformsLib() = default;
    };

    UniformsLib *UniformsLib::instance_ = nullptr;

}// namespace threepp::shaders

#endif//THREEPP_UNIFORMSLIB_HPP
