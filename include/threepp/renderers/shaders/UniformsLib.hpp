// https://github.com/mrdoob/three.js/blob/r129/src/renderers/shaders/UniformsLib.js

#ifndef THREEPP_UNIFORMSLIB_HPP
#define THREEPP_UNIFORMSLIB_HPP

#include "threepp/core/Uniform.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"

#include <unordered_map>

namespace threepp::shaders {

    class UniformsLib {

    public:
        UniformMap common{
                {"diffuse", Uniform(Color(0xffffff))},
                {"opacity", Uniform(1.f)},
                {"map", Uniform()},
                {"uvTransform", Uniform(Matrix3())},
                {"uv2Transform", Uniform(Matrix3())},
                {"alphaMap", Uniform()}};

        UniformMap specularmap{
                {"specularMap", Uniform()}};

        UniformMap envmap{
                {"envmap", Uniform()},
                {"flipEnvMap", Uniform(-1)},
                {"reflectivity", Uniform(1.f)},
                {"refractionRatio", Uniform(0.98f)},
                {"maxMipMapLevel", Uniform(0)}};

        UniformMap aomap{
                {"aomap", Uniform()},
                {"aoMapIntensity", Uniform(1.f)}};

        UniformMap lightmap{
                {"lightMap", Uniform()},
                {"lightMapIntesity", Uniform(1.f)}};

        UniformMap emissivemap{
                {"emissiveMap", Uniform()}};

        UniformMap bumpmap{
                {"bumpMap", Uniform()},
                {"bumpScale", Uniform(1.f)}};

        UniformMap normalmap{
                {"normalMap", Uniform()},
                {"normalScale", Uniform(Vector2(1, 1))}};

        UniformMap displacementmap{
                {"displacementMap", Uniform()},
                {"displacementScale", Uniform(1.f)},
                {"displacementBias", Uniform(0.f)}};

        UniformMap roughnessmap{
                {"roughnessMap", Uniform()}};

        UniformMap metalnessmap{
                {"metalnessMap", Uniform()}};

        UniformMap gradientmap{
                {"gradientMap", Uniform()}};

        UniformMap fog{
                {"fogDensity", Uniform(0.00025f)},
                {"fogNear", Uniform(1.f)},
                {"fogFar", Uniform(2000.f)},
                {"fogColor", Uniform(Color(0xffffff))}};

        UniformMap lights{
                {"ambientLightColor", Uniform()},
                {"lightProbe", Uniform()},
                {"directionalLights", Uniform()},
                {"directionalLightShadows", Uniform()},
                {"directionalShadowMap", Uniform()},
                {"directionalShadowMatrix", Uniform()},
                {"spotLights", Uniform()},
                {"spotLightShadows", Uniform()},
                {"spotShadowMap", Uniform()},
                {"spotShadowMatrix", Uniform()},
                {"pointLights", Uniform()},
                {"pointLightShadows", Uniform()},
                {"pointShadowMap", Uniform()},
                {"pointShadowMatrix", Uniform()},
                {"hemisphereLights", Uniform()},
                {"rectAreaLights", Uniform()},
                {"ltc_1", Uniform()},
                {"ltc_2", Uniform()},
        };

        UniformMap points{
                {"diffuse", Uniform(Color(0xffffff))},
                {"opacity", Uniform(1.f)},
                {"size", Uniform(1.f)},
                {"scale", Uniform(1.f)},
                {"map", Uniform()},
                {"alphaMap", Uniform()},
                {"uvTransform", Uniform(Matrix3())}};

        UniformMap sprite{
                {"diffuse", Uniform(Color(0xffffff))},
                {"opacity", Uniform(1.f)},
                {"center", Uniform(Vector2(0.5f, 0.5f))},
                {"rotation", Uniform(0.f)},
                {"map", Uniform()},
                {"alphaMap", Uniform()},
                {"uvTransform", Uniform(Matrix3())}};

        UniformsLib(const UniformsLib&) = delete;
        void operator=(const UniformsLib&) = delete;

        static UniformsLib& instance() {
            static UniformsLib instance;
            return instance;
        }

    private:
        UniformsLib() = default;
    };

}// namespace threepp::shaders

#endif//THREEPP_UNIFORMSLIB_HPP
