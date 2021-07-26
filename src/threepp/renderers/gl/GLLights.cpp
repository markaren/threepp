
#include "GLLights.hpp"

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

    std::stable_sort(lights.begin(), lights.end(), &shadowCastingLightsFirst);

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

                state.probe[j].addScaledVector(l->sh.getCoefficients()[j], intensity);
            }

        } else if (instanceof <DirectionalLight>(light)) {

            auto &uniforms = cache_.get(*light);

            std::get<std::shared_ptr<Color>>(uniforms["color"])->copy(light->color).multiplyScalar(light->intensity);

            if (light->castShadow) {

                auto l = dynamic_cast<DirectionalLight *>(light);
                auto shadow = l->shadow;

                auto &shadowUniforms = shadowCache_.get(*light);

                shadowUniforms["shadowBias"] = shadow->bias;
                shadowUniforms["shadowNormalBias"] = shadow->normalBias;
                shadowUniforms["shadowRadius"] = shadow->radius;
                shadowUniforms["shadowMapSize"] = shadow->mapSize;

                state.directionalShadow.resize(directionalLength + 1);
                state.directionalShadow[directionalLength] = shadowUniforms;
                state.directionalShadowMap[directionalLength] = shadow->map->texture;
                state.directionalShadowMatrix[directionalLength] = shadow->matrix;

                numDirectionalShadows++;
            }

            state.directional.resize(directionalLength + 1);
            state.directional[directionalLength] = uniforms;

            directionalLength++;
        } else if (instanceof <SpotLight>(light)) {

            auto &uniforms = cache_.get(*light);

            std::get<std::shared_ptr<Color>>(uniforms["color"])->copy(light->color).multiplyScalar(light->intensity);

            if (light->castShadow) {

                auto l = dynamic_cast<SpotLight *>(light);
                auto shadow = l->shadow;

                auto &shadowUniforms = shadowCache_.get(*light);

                shadowUniforms["shadowBias"] = shadow->bias;
                shadowUniforms["shadowNormalBias"] = shadow->normalBias;
                shadowUniforms["shadowRadius"] = shadow->radius;
                shadowUniforms["shadowMapSize"] = shadow->mapSize;

                state.spotShadow.resize(spotLength + 1);
                state.spotShadow[spotLength] = shadowUniforms;
                state.spotShadowMap[spotLength] = shadow->map->texture;
                state.spotShadowMatrix[spotLength] = shadow->matrix;

                numSpotShadows++;
            }

            state.spot.resize(spotLength + 1);
            state.spot[spotLength] = uniforms;

            spotLength++;

        } else if (instanceof <PointLight>(light)) {

            auto l = dynamic_cast<PointLight *>(light);
            auto &shadow = l->shadow;

            auto &uniforms = cache_.get(*light);

            std::get<std::shared_ptr<Color>>(uniforms["color"])->copy(light->color).multiplyScalar(light->intensity);
            uniforms["distance"] = l->distance;
            uniforms["decay"] = l->decay;

            if (light->castShadow) {

                auto &shadowUniforms = shadowCache_.get(*light);

                shadowUniforms["shadowBias"] = shadow->bias;
                shadowUniforms["shadowNormalBias"] = shadow->normalBias;
                shadowUniforms["shadowRadius"] = shadow->radius;
                shadowUniforms["shadowMapSize"] = shadow->mapSize;
                shadowUniforms["shadowCameraNear"] = shadow->camera->near;
                shadowUniforms["shadowCameraFar"] = shadow->camera->far;

                state.pointShadow.resize(pointLength + 1);
                state.pointShadow[pointLength] = shadowUniforms;
                state.pointShadowMap[pointLength] = shadow->map->texture;
                state.pointShadowMatrix[pointLength] = shadow->matrix;

                numPointShadows++;
            }

            state.point.resize(pointLength + 1);
            state.point[pointLength] = uniforms;

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

    const auto &viewMatrix = camera->matrixWorldInverse;

    for (auto &light : lights) {

        if (instanceof <DirectionalLight>(light)) {

            auto l = dynamic_cast<DirectionalLight *>(light);
            auto &uniforms = state.directional.at(directionalLength);

            auto direction = std::get<std::shared_ptr<Vector3>>(uniforms.at("direction"));

            direction->setFromMatrixPosition(light->matrixWorld);

            Vector3 vector3;
            vector3.setFromMatrixPosition(l->target->matrixWorld);
            direction->sub(vector3);
            direction->transformDirection(viewMatrix);

            directionalLength++;

        } else if (instanceof <SpotLight>(light)) {

            auto l = dynamic_cast<SpotLight *>(light);
            auto &uniforms = state.spot.at(spotLength);

            auto position = std::get<std::shared_ptr<Vector3>>(uniforms.at("position"));
            auto direction = std::get<std::shared_ptr<Vector3>>(uniforms.at("direction"));

            position->setFromMatrixPosition(light->matrixWorld);
            position->applyMatrix4(viewMatrix);

            direction->setFromMatrixPosition(light->matrixWorld);

            Vector3 vector3;
            vector3.setFromMatrixPosition(l->target->matrixWorld);
            direction->sub(vector3);
            direction->transformDirection(viewMatrix);

            spotLength++;

        } else if (instanceof <PointLight>(light)) {

            auto &uniforms = state.point.at(pointLength);

            auto position = std::get<std::shared_ptr<Vector3>>(uniforms.at("position"));

            position->setFromMatrixPosition(light->matrixWorld);
            position->applyMatrix4(viewMatrix);

            pointLength++;
        }
    }
}
