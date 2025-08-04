
#include "threepp/renderers/gl/GLLights.hpp"

#include "threepp/renderers/GLRenderTarget.hpp"

#include "threepp/lights/LightProbe.hpp"
#include "threepp/lights/LightShadow.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

using namespace threepp;
using namespace threepp::gl;

namespace {

    bool shadowCastingLightsFirst(const Light* lightA, const Light* lightB) {

        return (lightB->castShadow ? 1 : 0) < (lightA->castShadow ? 1 : 0);
    }

    template<class T>
    void ensureCapacity(T& container, size_t capacity) {

        while (container.size() < capacity) {
            container.emplace_back();
        }
    }

}// namespace


void GLLights::setup(std::vector<Light*>& lights) {

    float r = 0, g = 0, b = 0;

    for (unsigned i = 0; i < 9; i++) state.probe[i].set(0, 0, 0);

    int directionalLength = 0;
    int pointLength = 0;
    int spotLength = 0;
    int hemiLength = 0;

    int numDirectionalShadows = 0;
    int numPointShadows = 0;
    int numSpotShadows = 0;

    std::ranges::stable_sort(lights, shadowCastingLightsFirst);

    for (auto light : lights) {

        const auto& color = light->color;
        const auto intensity = light->intensity;

        if (light->type() == "AmbientLight") {

            r += color.r * intensity;
            g += color.g * intensity;
            b += color.b * intensity;

        } else if (light->type() == "LightProbe") {

            const auto l = light->as<LightProbe>();

            for (unsigned j = 0; j < 9; ++j) {

                state.probe[j].addScaledVector(l->sh.getCoefficients()[j], intensity);
            }

        } else if (auto directionalLight = light->as<DirectionalLight>()) {

            const auto uniforms = cache_.get(*light);

            std::get<Color>(uniforms->at("color")).copy(light->color).multiplyScalar(light->intensity);

            if (light->castShadow) {

                auto& shadow = directionalLight->shadow;

                auto shadowUniforms = shadowCache_.get(*light);

                shadowUniforms->at("shadowBias") = shadow->bias;
                shadowUniforms->at("shadowNormalBias") = shadow->normalBias;
                shadowUniforms->at("shadowRadius") = shadow->radius;
                std::get<Vector2>(shadowUniforms->at("shadowMapSize")).copy(shadow->mapSize);

                ensureCapacity(state.directionalShadow, directionalLength + 1);
                ensureCapacity(state.directionalShadowMap, directionalLength + 1);
                ensureCapacity(state.directionalShadowMatrix, directionalLength + 1);
                state.directionalShadow[directionalLength] = shadowUniforms;
                state.directionalShadowMap[directionalLength] = shadow->map ? shadow->map->texture.get() : nullptr;
                state.directionalShadowMatrix[directionalLength] = &shadow->matrix;

                ++numDirectionalShadows;
            }

            ensureCapacity(state.directional, directionalLength + 1);
            state.directional[directionalLength] = uniforms;

            ++directionalLength;

        } else if (auto spotLight = light->as<SpotLight>()) {

            const auto uniforms = cache_.get(*light);

            std::get<Vector3>(uniforms->at("position")).setFromMatrixPosition(*spotLight->matrixWorld);

            std::get<Color>(uniforms->at("color")).copy(color).multiplyScalar(spotLight->intensity);
            std::get<float>(uniforms->at("distance")) = spotLight->distance;

            std::get<float>(uniforms->at("coneCos")) = std::cos(spotLight->angle);
            std::get<float>(uniforms->at("penumbraCos")) = std::cos(spotLight->angle * (1 - spotLight->penumbra));
            std::get<float>(uniforms->at("decay")) = spotLight->decay;

            if (light->castShadow) {

                const auto& shadow = spotLight->shadow;
                const auto shadowUniforms = shadowCache_.get(*light);

                shadowUniforms->at("shadowBias") = shadow->bias;
                shadowUniforms->at("shadowNormalBias") = shadow->normalBias;
                shadowUniforms->at("shadowRadius") = shadow->radius;
                std::get<Vector2>(shadowUniforms->at("shadowMapSize")).copy(shadow->mapSize);

                ensureCapacity(state.spotShadow, spotLength + 1);
                ensureCapacity(state.spotShadowMap, spotLength + 1);
                ensureCapacity(state.spotShadowMatrix, spotLength + 1);
                state.spotShadow[spotLength] = shadowUniforms;
                state.spotShadowMap[spotLength] = shadow->map ? shadow->map->texture.get() : nullptr;
                state.spotShadowMatrix[spotLength] = &shadow->matrix;

                ++numSpotShadows;
            }

            ensureCapacity(state.spot, spotLength + 1);
            state.spot[spotLength] = uniforms;

            ++spotLength;

        } else if (auto pointLight = light->as<PointLight>()) {

            const auto uniforms = cache_.get(*light);

            std::get<Color>(uniforms->at("color")).copy(light->color).multiplyScalar(pointLight->intensity);
            uniforms->at("distance") = pointLight->distance;
            uniforms->at("decay") = pointLight->decay;

            if (light->castShadow) {

                const auto& shadow = pointLight->shadow;
                LightUniforms* shadowUniforms = shadowCache_.get(*light);

                shadowUniforms->at("shadowBias") = shadow->bias;
                shadowUniforms->at("shadowNormalBias") = shadow->normalBias;
                shadowUniforms->at("shadowRadius") = shadow->radius;
                std::get<Vector2>(shadowUniforms->at("shadowMapSize")).copy(shadow->mapSize);
                shadowUniforms->at("shadowCameraNear") = shadow->camera->nearPlane;
                shadowUniforms->at("shadowCameraFar") = shadow->camera->farPlane;

                ensureCapacity(state.pointShadow, pointLength + 1);
                ensureCapacity(state.pointShadowMap, pointLength + 1);
                ensureCapacity(state.pointShadowMatrix, pointLength + 1);
                state.pointShadow[pointLength] = shadowUniforms;
                state.pointShadowMap[pointLength] = shadow->map ? shadow->map->texture.get() : nullptr;
                state.pointShadowMatrix[pointLength] = &shadow->matrix;

                ++numPointShadows;
            }

            ensureCapacity(state.point, pointLength + 1);
            state.point[pointLength] = uniforms;

            ++pointLength;

        } else if (auto hemisphereLight = light->as<HemisphereLight>()) {

            const auto uniforms = cache_.get(*light);

            std::get<Color>(uniforms->at("skyColor")).copy(hemisphereLight->color).multiplyScalar(intensity);
            std::get<Color>(uniforms->at("groundColor")).copy(hemisphereLight->groundColor).multiplyScalar(intensity);

            ensureCapacity(state.hemi, hemiLength + 1);
            state.hemi[hemiLength] = uniforms;

            ++hemiLength;
        }
    }

    state.ambient.setRGB(r, g, b);

    auto& hash = state.hash;

    if (hash.directionalLength != directionalLength ||
        hash.pointLength != pointLength ||
        hash.spotLength != spotLength ||
        hash.hemiLength != hemiLength ||
        hash.numDirectionalShadows != numDirectionalShadows ||
        hash.numPointShadows != numPointShadows ||
        hash.numSpotShadows != numSpotShadows) {

        state.directional.resize(directionalLength);
        state.spot.resize(spotLength);
        state.point.resize(pointLength);
        state.hemi.resize(hemiLength);

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
        hash.hemiLength = hemiLength;

        hash.numDirectionalShadows = numDirectionalShadows;
        hash.numPointShadows = numPointShadows;
        hash.numSpotShadows = numSpotShadows;

        state.version = nextVersion++;
    }
}

void GLLights::setupView(std::vector<Light*>& lights, Camera* camera) {

    int directionalLength = 0;
    int pointLength = 0;
    int spotLength = 0;
    int hemiLength = 0;

    const auto& viewMatrix = camera->matrixWorldInverse;

    for (auto light : lights) {

        if (light->as<DirectionalLight>()) {

            auto l = light->as<DirectionalLight>();
            auto& uniforms = state.directional.at(directionalLength);

            auto& direction = std::get<Vector3>(uniforms->at("direction"));

            direction.setFromMatrixPosition(*light->matrixWorld);

            Vector3 vector3;
            vector3.setFromMatrixPosition(*l->target().matrixWorld);
            direction.sub(vector3);
            direction.transformDirection(viewMatrix);

            ++directionalLength;

        } else if (light->as<SpotLight>()) {

            auto l = dynamic_cast<SpotLight*>(light);
            auto& uniforms = state.spot.at(spotLength);

            auto& position = std::get<Vector3>(uniforms->at("position"));
            auto& direction = std::get<Vector3>(uniforms->at("direction"));

            position.setFromMatrixPosition(*l->matrixWorld);
            position.applyMatrix4(viewMatrix);

            direction.setFromMatrixPosition(*l->matrixWorld);

            Vector3 vector3;
            vector3.setFromMatrixPosition(*l->target().matrixWorld);
            direction.sub(vector3);
            direction.transformDirection(viewMatrix);

            ++spotLength;

        } else if (light->as<PointLight>()) {

            auto& uniforms = state.point.at(pointLength);

            auto& position = std::get<Vector3>(uniforms->at("position"));

            position.setFromMatrixPosition(*light->matrixWorld);
            position.applyMatrix4(viewMatrix);

            ++pointLength;

        } else if (light->as<HemisphereLight>()) {

            auto& uniforms = state.hemi.at(hemiLength);

            auto& direction = std::get<Vector3>(uniforms->at("direction"));

            direction.setFromMatrixPosition(*light->matrixWorld);
            direction.transformDirection(viewMatrix);
            direction.normalize();

            ++hemiLength;
        }
    }
}
