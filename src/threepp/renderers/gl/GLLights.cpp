
#include "threepp/renderers/gl/GLLights.hpp"

#include "threepp/math/Vector2.hpp"

#include <any>
#include <unordered_map>

using namespace threepp;
using namespace threepp::gl;


void GLLights::setup(std::vector<Light *> &lights) {

    float r = 0, g = 0, b = 0;

    for (int i = 0; i < 9; i++) state_.probe[i].set(0, 0, 0);

    int directionalLength = 0;
    int pointLength = 0;
    int spotLength = 0;

    int numDirectionalShadows = 0;
    int numPointShadows = 0;
    int numSpotShadows = 0;

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
                auto shadow = l->shadow;

                // TODO

                numDirectionalShadows++;
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
        hash.numDirectionalShadows != numDirectionalShadows ||
        hash.numPointShadows != numPointShadows ||
        hash.numSpotShadows != numSpotShadows) {

        state_.directional.resize(directionalLength);
        state_.spot.resize(spotLength);
        state_.point.resize(pointLength);

        state_.directionalShadow.resize(numDirectionalShadows);
        state_.directionalShadowMap.resize(numDirectionalShadows);
        state_.pointShadow.resize(numPointShadows);
        state_.pointShadowMap.resize(numPointShadows);
        state_.spotShadow.resize(numSpotShadows);
        state_.spotShadowMap.resize(numSpotShadows);
        state_.directionalShadowMatrix.resize(numDirectionalShadows);
        state_.pointShadowMatrix.resize(numPointShadows);
        state_.spotShadowMatrix.resize(numSpotShadows);

        hash.directionalLength = directionalLength;
        hash.pointLength = pointLength;
        hash.spotLength = spotLength;

        hash.numDirectionalShadows = numDirectionalShadows;
        hash.numPointShadows = numPointShadows;
        hash.numSpotShadows = numSpotShadows;

        state_.version = nextVersion++;
    }
}

void GLLights::setupView(std::vector<Light *> &lights, Camera *camera) {

    int directionalLength = 0;
    int pointLength = 0;
    int spotLength = 0;

    const auto &viewMatrix = camera->matrixWorldInverse;

    for (auto &light : lights) {

        if (instanceof <DirectionalLight>(light)) {

            auto l = dynamic_cast<DirectionalLight*>(light);
            auto& uniforms = state_.directional.at(directionalLength);

            auto& direction = std::any_cast<Vector3&>(uniforms.at("direction"));

            direction.setFromMatrixPosition(light->matrixWorld);
            vector3.setFromMatrixPosition(l->target->matrixWorld);
            direction.sub(vector3);
            direction.transformDirection(viewMatrix);

            directionalLength ++;

        } else if (instanceof <SpotLight>(light)) {

            auto l = dynamic_cast<SpotLight*>(light);
            auto& uniforms = state_.spot.at(spotLength);

            auto& position = std::any_cast<Vector3&>(uniforms.at("position"));
            auto& direction = std::any_cast<Vector3&>(uniforms.at("direction"));

            position.setFromMatrixPosition(light->matrixWorld);
            position.applyMatrix4(viewMatrix);

            direction.setFromMatrixPosition(light->matrixWorld);
            vector3.setFromMatrixPosition(l->target->matrixWorld);
            direction.sub(vector3);
            direction.transformDirection(viewMatrix);

            spotLength ++;

        } else if (instanceof <PointLight>(light)) {

            auto& uniforms = state_.spot.at(pointLength);

            auto& position = std::any_cast<Vector3&>(uniforms.at("position"));

            position.setFromMatrixPosition(light->matrixWorld);
            position.applyMatrix4(viewMatrix);

            pointLength++;
        }
    }
}
