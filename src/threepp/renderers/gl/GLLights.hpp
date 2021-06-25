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

            return (lightB->castShadow ? 1 : 0) - (lightA->castShadow ? 1 : 0);
        }

    }// namespace

    typedef std::unordered_map<std::string, std::any> LightUniforms;

    struct UniformCache {

        LightUniforms &get(const Light &light) {

            if (!lights.count(light.id)) {
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
                        {"shadowBias", 0},
                        {"shadowNormalBias", 0},
                        {"shadowRadius", 1},
                        {"shadowMapSize", Vector2()},
                        {"shadowCameraNear", 1},
                        {"shadowCameraFar", 1000}};
            }

            lights[light.id] = uniforms;

            return lights.at(light.id);
        }

    private:
        std::unordered_map<unsigned int, LightUniforms> lights;
    };

    struct ShadowUniformsCache {
    };


    struct GLLights {

        struct LightState {

            struct Hash {

                int directionalLength = -1;
                int pointLength = -1;
                int spotLength = -1;
                int rectAreaLength = -1;
                int hemiLength = -1;

                int numDirectionalShadows = -1;
                int numPointShadows = -1;
                int numSpotShadows = -1;

            };

            unsigned int version = 0;

            Hash hash;

            std::array<float, 3> ambient;
            std::array<Vector3, 9> probe;
        };

        unsigned int nextVersion = 0;

        int directionalLength = 0;
        int pointLength = 0;
        int spotLength = 0;
        int rectAreaLength = 0;
        int hemiLength = 0;

        int numDirectionalShadows = 0;
        int numPointShadows = 0;
        int numSpotShadows = 0;

        void setup(std::vector<Light *> &lights) {

            float r = 0, g = 0, b = 0;

            std::sort(lights.begin(), lights.end(), &shadowCastingLightsFirst);

            for (auto &light : lights) {

                auto &color = light->color;
                auto intensity = light->intensity;

                if (instanceof <AmbientLight>(light)) {

                    r += color.r * intensity;
                    g += color.g * intensity;
                    b += color.b * intensity;

                } else if (instanceof <LightProbe>(light)) {

                    const auto l = dynamic_cast<LightProbe *>(light);

                    for (int j = 0; j < 9; j++) {

                        state_.probe[j].addScaledVector(l->sh.getCoefficients()[j], intensity);
                    }

                } else if (instanceof <DirectionalLight>(light)) {

                    auto &uniforms = cache_.get(*light);

                    std::any_cast<Color>(uniforms["color"]).copy(light->color).multiplyScalar(light->intensity);

                    if (light->castShadow) {

                        auto l = dynamic_cast<DirectionalLight *>(light);

                        // TODO
                    }
                }
            }

            state_.ambient[0] = r;
            state_.ambient[1] = g;
            state_.ambient[2] = b;

            auto &hash = state_.hash;

            if (hash.directionalLength != directionalLength ||
                hash.pointLength != pointLength ||
                hash.spotLength != spotLength ||
                hash.rectAreaLength != rectAreaLength ||
                hash.hemiLength != hemiLength ||
                hash.numDirectionalShadows != numDirectionalShadows ||
                hash.numPointShadows != numPointShadows ||
                hash.numSpotShadows != numSpotShadows) {

//                state_.directional.length = directionalLength;
//                state_.spot.length = spotLength;
//                state_.rectArea.length = rectAreaLength;
//                state_.point.length = pointLength;
//                state_.hemi.length = hemiLength;
//
//                state_.directionalShadow.length = numDirectionalShadows;
//                state_.directionalShadowMap.length = numDirectionalShadows;
//                state_.pointShadow.length = numPointShadows;
//                state_.pointShadowMap.length = numPointShadows;
//                state_.spotShadow.length = numSpotShadows;
//                state_.spotShadowMap.length = numSpotShadows;
//                state_.directionalShadowMatrix.length = numDirectionalShadows;
//                state_.pointShadowMatrix.length = numPointShadows;
//                state_.spotShadowMatrix.length = numSpotShadows;

                hash.directionalLength = directionalLength;
                hash.pointLength = pointLength;
                hash.spotLength = spotLength;
                hash.rectAreaLength = rectAreaLength;
                hash.hemiLength = hemiLength;

                hash.numDirectionalShadows = numDirectionalShadows;
                hash.numPointShadows = numPointShadows;
                hash.numSpotShadows = numSpotShadows;

                state_.version = nextVersion++;
            }
        }

        void setupView(std::vector<Light *> &lights, Camera *camera) {

            int directionalLength = 0;
            int pointLength = 0;
            int spotLength = 0;
            int rectAreaLength = 0;
            int hemiLength = 0;

            const auto &viewMatrix = camera->matrixWorldInverse;

            for (auto &light : lights) {

                if (instanceof <DirectionalLight>(light)) {

                    // TODO

                } else if (instanceof <SpotLight>(light)) {

                    // TODO

                } else if (instanceof <PointLight>(light)) {

                    // TODO

                }
            }
        }

    private:
        UniformCache cache_;

        LightState state_;
    };

}// namespace threepp::gl

#endif//THREEPP_GLLIGHTS_HPP
