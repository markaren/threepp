
#include "threepp/renderers/gl/GLLights.hpp"

#include <algorithm>
#include <unordered_map>

using namespace threepp;
using namespace threepp::gl;

void GLLights::setup(std::vector<Light *> &lights) {

    float r = 0, g = 0, b = 0;

    for (int i = 0; i < 9; i++) state.probe[i].set(0, 0, 0);

    int directionalLength = 0;
    int pointLength = 0;
    int spotLength = 0;

    int numDirectionalShadows = 0;
    int numPointShadows = 0;
    int numSpotShadows = 0;

    std::stable_sort(lights.begin(), lights.end(), shadowCastingLightsFirst);

    for (auto &light : lights) {

        auto &color = light->color;
        auto intensity = light->intensity;

        if (light->as<AmbientLight>()) {

            r += color.r * intensity;
            g += color.g * intensity;
            b += color.b * intensity;

        } else if (light->as<LightProbe>()) {

            const auto l = light->as<LightProbe>();

            for (int j = 0; j < 9; ++j) {

                state.probe[j].addScaledVector(l->sh.getCoefficients()[j], intensity);
            }

        } else if (light->as<DirectionalLight>()) {

            auto uniforms = cache_.get(*light);

            std::get<Color>(uniforms->at("color")).copy(light->color).multiplyScalar(light->intensity);

            if (light->castShadow) {

                auto l = light->as<DirectionalLight>();
                auto shadow = l->shadow;

                auto &shadowUniforms = shadowCache_.get(*light);

                shadowUniforms.at("shadowBias") = shadow->bias;
                shadowUniforms.at("shadowNormalBias") = shadow->normalBias;
                shadowUniforms.at("shadowRadius") = shadow->radius;
                shadowUniforms.at("shadowMapSize") = shadow->mapSize;

                state.directionalShadow.resize(directionalLength + 1);
                state.directionalShadow[directionalLength] = shadowUniforms;
                state.directionalShadowMap[directionalLength] = shadow->map->texture;
                state.directionalShadowMatrix[directionalLength] = shadow->matrix;

                numDirectionalShadows++;
            }

            state.directional.emplace_back(uniforms);

            directionalLength++;

        } else if (light->is<SpotLight>()) {

            auto l = light->as<SpotLight>();
            auto uniforms = cache_.get(*light);

            std::get<Vector3>(uniforms->at("position")).setFromMatrixPosition(l->matrixWorld);

            std::get<Color>(uniforms->at("color")).copy(color).multiplyScalar(l->intensity);
            std::get<float>(uniforms->at("distance")) = l->distance;

            std::get<float>(uniforms->at("coneCos")) = std::cos(l->angle);
            std::get<float>(uniforms->at("penumbraCos")) = std::cos(l->angle * (1 - l->penumbra));
            std::get<float>(uniforms->at("decay")) = l->decay;

            if (light->castShadow) {

                auto shadow = l->shadow;
                auto &shadowUniforms = shadowCache_.get(*light);

                shadowUniforms.at("shadowBias") = shadow->bias;
                shadowUniforms.at("shadowNormalBias") = shadow->normalBias;
                shadowUniforms.at("shadowRadius") = shadow->radius;
                shadowUniforms.at("shadowMapSize") = shadow->mapSize;

                state.spotShadow.resize(spotLength + 1);
                state.spotShadow[spotLength] = shadowUniforms;
                state.spotShadowMap[spotLength] = shadow->map->texture;
                state.spotShadowMatrix[spotLength] = shadow->matrix;

                numSpotShadows++;
            }

            state.spot.emplace_back(uniforms);

            spotLength++;

        } else if (light->as<PointLight>()) {

            auto l = dynamic_cast<PointLight *>(light);
            auto &shadow = l->shadow;

            auto uniforms = cache_.get(*light);

            std::get<Color>(uniforms->at("color")).copy(light->color).multiplyScalar(l->intensity);
            uniforms->at("distance") = l->distance;
            uniforms->at("decay") = l->decay;

            if (light->castShadow) {

                auto &shadowUniforms = shadowCache_.get(*light);

                shadowUniforms.at("shadowBias") = shadow->bias;
                shadowUniforms.at("shadowNormalBias") = shadow->normalBias;
                shadowUniforms.at("shadowRadius") = shadow->radius;
                shadowUniforms.at("shadowMapSize") = shadow->mapSize;
                shadowUniforms.at("shadowCameraNear") = shadow->camera->near;
                shadowUniforms.at("shadowCameraFar") = shadow->camera->far;

                state.pointShadow.resize(pointLength + 1);
                state.pointShadow[pointLength] = shadowUniforms;
                state.pointShadowMap[pointLength] = shadow->map->texture;
                state.pointShadowMatrix[pointLength] = shadow->matrix;

                numPointShadows++;
            }

            state.point.emplace_back(uniforms);

            pointLength++;
        }
    }

    state.ambient.setRGB(r, g, b);

    auto &hash = state.hash;

    if (hash.directionalLength != directionalLength ||
        hash.pointLength != pointLength ||
        hash.spotLength != spotLength ||
        hash.numDirectionalShadows != numDirectionalShadows ||
        hash.numPointShadows != numPointShadows ||
        hash.numSpotShadows != numSpotShadows) {

        state.directional.resize(directionalLength);
        state.spot.resize(spotLength);
        state.point.resize(pointLength);

        state.directionalShadow.resize(numDirectionalShadows);
        state.directionalShadowMap.resize(numDirectionalShadows);
        state.pointShadow.resize(numPointShadows);
        state.pointShadowMap.resize(numPointShadows);
        state.spotShadow.resize(numSpotShadows);
        state.spotShadowMap.resize(numSpotShadows);
        state.directionalShadowMatrix.resize(numDirectionalShadows);
        state.pointShadowMatrix.resize(numPointShadows);
        state.spotShadowMatrix.resize(numSpotShadows);

        hash.directionalLength = directionalLength;
        hash.pointLength = pointLength;
        hash.spotLength = spotLength;

        hash.numDirectionalShadows = numDirectionalShadows;
        hash.numPointShadows = numPointShadows;
        hash.numSpotShadows = numSpotShadows;

        state.version = nextVersion++;
    }
}

void GLLights::setupView(std::vector<Light *> &lights, Camera *camera) {

    int directionalLength = 0;
    int pointLength = 0;
    int spotLength = 0;

    const auto viewMatrix = camera->matrixWorldInverse;

    for (auto light : lights) {

        if (light->as<DirectionalLight>()) {

            auto l = light->as<DirectionalLight>();
            auto &uniforms = state.directional.at(directionalLength);

            auto &direction = std::get<Vector3>(uniforms->at("direction"));

            direction.setFromMatrixPosition(light->matrixWorld);

            Vector3 vector3;
            vector3.setFromMatrixPosition(l->target->matrixWorld);
            direction.sub(vector3);
            direction.transformDirection(viewMatrix);

            directionalLength++;

        } else if (light->as<SpotLight>()) {

            auto l = dynamic_cast<SpotLight *>(light);
            auto &uniforms = state.spot.at(spotLength);

            auto &position = std::get<Vector3>(uniforms->at("position"));
            auto &direction = std::get<Vector3>(uniforms->at("direction"));

            position.setFromMatrixPosition(l->matrixWorld);
            position.applyMatrix4(viewMatrix);

            direction.setFromMatrixPosition(l->matrixWorld);

            Vector3 vector3;
            vector3.setFromMatrixPosition(l->target->matrixWorld);
            direction.sub(vector3);
            direction.transformDirection(viewMatrix);

            spotLength++;

        } else if (light->as<PointLight>()) {

            auto &uniforms = state.point.at(pointLength);

            auto &position = std::get<Vector3>(uniforms->at("position"));

            position.setFromMatrixPosition(light->matrixWorld);
            position.applyMatrix4(viewMatrix);

            pointLength++;
        }
    }
}
