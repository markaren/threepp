// https://github.com/mrdoob/three.js/blob/dev/src/renderers/webgl/WebGLLights.js

#ifndef THREEPP_GLLIGHTS_HPP
#define THREEPP_GLLIGHTS_HPP

#include "threepp/lights/lights.hpp"

#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"

#include "threepp/utils/InstanceOf.hpp"

#include <array>
#include <unordered_map>
#include <vector>


namespace threepp::gl {

    namespace {

        bool shadowCastingLightsFirst(const Light *lightA, const Light *lightB) {

            return (lightB->castShadow ? 1 : 0) > (lightA->castShadow ? 1 : 0);
        }

    }// namespace

    typedef std::unordered_map<std::string, NestedUniformValue> LightUniforms;

    struct UniformsCache {

        LightUniforms &get(const Light &light) {

            if (lights.count(light.id)) {

                return lights.at(light.id);
            }

            auto type = light.type();
            LightUniforms uniforms;
            if (type == "DirectionalLight") {

                uniforms = {
                        {"direction", std::make_shared<Vector3>()},
                        {"color", std::make_shared<Color>()}};

            } else if (type == "SpotLight") {

                uniforms = {
                        {"position", std::make_shared<Vector3>()},
                        {"direction", std::make_shared<Vector3>()},
                        {"color", std::make_shared<Color>()},
                        {"distance", 0.f},
                        {"coneCos", 0.f},
                        {"penumbraCos", 0.f},
                        {"decay", 0.f}};

            } else if (type == "PointLight") {

                uniforms = {
                        {"position", std::make_shared<Vector3>()},
                        {"color", std::make_shared<Color>()},
                        {"distance", 0.f},
                        {"decay", 0.f}};

            } else if (type == "HemisphereLight") {

                uniforms = {
                        {"direction", std::make_shared<Vector3>()},
                        {"skyColor", std::make_shared<Color>()},
                        {"groundColor", std::make_shared<Color>()}};

            } else if (type == "RectAreaLight") {

                uniforms = {
                        {"color", std::make_shared<Color>()},
                        {"position", std::make_shared<Vector3>()},
                        {"halfWidth", std::make_shared<Vector3>()},
                        {"halfHeight", std::make_shared<Vector3>()}};
            }

            lights[light.id] = uniforms;

            return lights.at(light.id);
        }

    private:
        std::unordered_map<unsigned int, LightUniforms> lights;
    };

    struct ShadowUniformsCache {

        LightUniforms &get(const Light &light) {

            if (lights.count(light.id)) {

                return lights.at(light.id);
            }

            auto type = light.type();
            LightUniforms uniforms;
            if (type == "DirectionalLight") {

                uniforms = {
                        {"shadowBias", 0.f},
                        {"shadowNormalBias", 0.f},
                        {"shadowRadius", 1.f},
                        {"shadowMapSize", Vector2()}};

            } else if (type == "SpotLight") {

                uniforms = {
                        {"shadowBias", 0.f},
                        {"shadowNormalBias", 0.f},
                        {"shadowRadius", 1.f},
                        {"shadowMapSize", Vector2()}};

            } else if (type == "PointLight") {

                uniforms = {
                        {"shadowBias", 0.f},
                        {"shadowNormalBias", 0.f},
                        {"shadowRadius", 1.f},
                        {"shadowMapSize", Vector2()},
                        {"shadowCameraNear", 1.f},
                        {"shadowCameraFar", 1000.f}};
            }

            lights[light.id] = uniforms;

            return lights.at(light.id);
        }

    private:
        std::unordered_map<unsigned int, LightUniforms> lights;
    };


    struct GLLights {

        struct LightState {

            struct Hash {

                int directionalLength = -1;
                int pointLength = -1;
                int spotLength = -1;

                int numDirectionalShadows = -1;
                int numPointShadows = -1;
                int numSpotShadows = -1;
            };

            unsigned int version = 0;

            Hash hash{};

            Color ambient{0,0,0};
            std::vector<Vector3> probe{9};
            std::vector<LightUniforms> directional;
            std::vector<LightUniforms> directionalShadow;
            std::vector<Texture> directionalShadowMap;
            std::vector<Matrix4> directionalShadowMatrix;
            std::vector<LightUniforms> spot;
            std::vector<LightUniforms> spotShadow;
            std::vector<Texture> spotShadowMap;
            std::vector<Matrix4> spotShadowMatrix;
            std::vector<LightUniforms> rectArea;
            std::vector<LightUniforms> point;
            std::vector<LightUniforms> pointShadow;
            std::vector<Texture> pointShadowMap;
            std::vector<Matrix4> pointShadowMatrix;
            std::vector<LightUniforms> hemi;
        };

        LightState state{};

        void setup(std::vector<Light *> &lights);

        void setupView(std::vector<Light *> &lights, Camera *camera);

    private:
        UniformsCache cache_;
        ShadowUniformsCache shadowCache_;

        unsigned int nextVersion = 0;
    };

}// namespace threepp::gl

#endif//THREEPP_GLLIGHTS_HPP
