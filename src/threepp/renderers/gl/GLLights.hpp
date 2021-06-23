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

        bool shadowCastingLightsFirst(const std::shared_ptr<Light> &lightA, const std::shared_ptr<Light> &lightB) {

            return (lightB->castShadow ? true : false) - (lightA->castShadow ? 1 : 0);
        }

    }// namespace

    typedef std::unordered_map<std::string, std::any> LightUniforms;

    struct UniformCache {

        std::unordered_map<unsigned int, LightUniforms> lights;

        LightUniforms &get(const Light &light) {

            if (!lights.count(light.id)) {
                return lights[light.id];
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
                        {"shadowBias", 0},
                        {"shadowNormalBias", 0},
                        {"shadowRadius", 1},
                        {"shadowMapSize", Vector2()},
                        {"shadowCameraNear", 1},
                        {"shadowCameraFar", 1000}};
            }

            lights[light.id] = uniforms;
            return uniforms;
        }
    };

    struct LightState {

        unsigned int version = 0;

        std::array<float, 3> ambient;
        std::array<Vector3, 3> probe;
    };

    struct GLLights {

        int directionalLength = 0;
        int pointLength = 0;
        int spotLength = 0;
        int rectAreaLength = 0;
        int hemiLength = 0;

        int numDirectionalShadows = 0;
        int numPointShadows = 0;
        int numSpotShadows = 0;

        UniformCache cache;

        void setup(std::vector<std::shared_ptr<Light>> &lights) {

            int r, g, b = 0;

            std::sort(lights.begin(), lights.end(), &shadowCastingLightsFirst);

            for (int i = 0, l = lights.size(); i < l; i++) {

                auto &light = lights[i];

                auto &color = light->color;
                auto intensity = light->intensity;


                if (instanceof <AmbientLight>(light.get())) {

                    r += color.r * intensity;
                    g += color.g * intensity;
                    b += color.b * intensity;

                } else if (instanceof <LightProbe>(light.get())) {



                } else if (instanceof <DirectionalLight>(light.get())) {

                    auto& uniforms = cache.get(*light);


                }
            }
        }

    private:
    };

}// namespace threepp::gl

#endif//THREEPP_GLLIGHTS_HPP
