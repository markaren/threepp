
#include "Lights.hpp"

#include "threepp/renderers/RenderTarget.hpp"

#include "threepp/lights/LightProbe.hpp"
#include "threepp/lights/LightShadow.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <unordered_map>

using namespace threepp;

namespace {

    // One counter for every Lights there is, as the module-level nextVersion
    // of three.js's WebGLLights. A renderer keeps a Lights per scene and per
    // nesting depth (a Reflector draws the scene again from inside the frame),
    // and a material remembers the version of the one it was last wired to.
    // Counted per instance, two of them both said "0": a material wired to the
    // mirror pass's lights passed for up to date in the main pass, and went on
    // being lit from the mirror camera's side until something else rebuilt it.
    std::atomic<unsigned int> nextVersion{0};

    // What a receiver samples: the native depth texture the shadow pass wrote
    // (PCF and Basic, as three.js r186), or the colour target holding the
    // blurred moments (VSM, which has no depth texture attached).
    Texture* shadowMapTexture(const LightShadow& shadow) {
        if (!shadow.map) return nullptr;
        if (shadow.map->depthTexture) return shadow.map->depthTexture.get();
        return shadow.map->texture.get();
    }

    bool shadowCastingLightsFirst(const Light* lightA, const Light* lightB) {

        return (lightB->castShadow ? 1 : 0) < (lightA->castShadow ? 1 : 0);
    }

    template<class T>
    void ensureCapacity(T& container, size_t capacity) {

        while (container.size() < capacity) {
            container.emplace_back();
        }
    }

    // Writes one slot of the light state and reports whether it changed. A
    // material copies these vectors when it is wired to the lights (GLRenderer
    // getProgram) where three.js shares the arrays by reference, so a slot
    // that comes to hold a different pointer has to bump the version or every
    // wired material keeps reading the old one: a light swapped for another
    // of the same type within one frame kept lighting from the removed light,
    // and a shadow map rebuilt at a new size left a freed texture behind the
    // sampler. Counting only the lights, as the hash below does, sees neither.
    template<class T>
    void setSlot(std::vector<T>& slots, size_t index, T value, bool& changed) {

        ensureCapacity(slots, index + 1);
        if (slots[index] != value) {
            slots[index] = value;
            changed = true;
        }
    }

}// namespace


void Lights::setup(std::vector<Light*>& lights) {

    float r = 0, g = 0, b = 0;

    for (unsigned i = 0; i < 9; i++) state.probe[i].set(0, 0, 0);

    int directionalLength = 0;
    int pointLength = 0;
    int spotLength = 0;
    int rectAreaLength = 0;
    int hemiLength = 0;

    int numDirectionalShadows = 0;
    int numPointShadows = 0;
    int numSpotShadows = 0;

    bool slotsChanged = false;

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

                shadowUniforms->at("shadowIntensity") = shadow->intensity;
                shadowUniforms->at("shadowBias") = shadow->bias;
                shadowUniforms->at("shadowNormalBias") = shadow->normalBias;
                shadowUniforms->at("shadowRadius") = shadow->radius;
                std::get<Vector2>(shadowUniforms->at("shadowMapSize")).copy(shadow->mapSize);

                setSlot(state.directionalShadow, directionalLength, shadowUniforms, slotsChanged);
                setSlot(state.directionalShadowMap, directionalLength, shadowMapTexture(*shadow), slotsChanged);
                setSlot(state.directionalShadowMatrix, directionalLength, &shadow->matrix, slotsChanged);

                ++numDirectionalShadows;
            }

            setSlot(state.directional, directionalLength, uniforms, slotsChanged);

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

                shadowUniforms->at("shadowIntensity") = shadow->intensity;
                shadowUniforms->at("shadowBias") = shadow->bias;
                shadowUniforms->at("shadowNormalBias") = shadow->normalBias;
                shadowUniforms->at("shadowRadius") = shadow->radius;
                std::get<Vector2>(shadowUniforms->at("shadowMapSize")).copy(shadow->mapSize);

                setSlot(state.spotShadow, spotLength, shadowUniforms, slotsChanged);
                setSlot(state.spotShadowMap, spotLength, shadowMapTexture(*shadow), slotsChanged);
                setSlot(state.spotShadowMatrix, spotLength, &shadow->matrix, slotsChanged);

                ++numSpotShadows;
            }

            setSlot(state.spot, spotLength, uniforms, slotsChanged);

            ++spotLength;

        } else if (auto pointLight = light->as<PointLight>()) {

            const auto uniforms = cache_.get(*light);

            std::get<Color>(uniforms->at("color")).copy(light->color).multiplyScalar(pointLight->intensity);
            uniforms->at("distance") = pointLight->distance;
            uniforms->at("decay") = pointLight->decay;

            if (light->castShadow) {

                const auto& shadow = pointLight->shadow;
                LightUniforms* shadowUniforms = shadowCache_.get(*light);

                shadowUniforms->at("shadowIntensity") = shadow->intensity;
                shadowUniforms->at("shadowBias") = shadow->bias;
                shadowUniforms->at("shadowNormalBias") = shadow->normalBias;
                shadowUniforms->at("shadowRadius") = shadow->radius;
                std::get<Vector2>(shadowUniforms->at("shadowMapSize")).copy(shadow->mapSize);
                shadowUniforms->at("shadowCameraNear") = shadow->camera->nearPlane;
                shadowUniforms->at("shadowCameraFar") = shadow->camera->farPlane;

                setSlot(state.pointShadow, pointLength, shadowUniforms, slotsChanged);
                setSlot(state.pointShadowMap, pointLength, shadowMapTexture(*shadow), slotsChanged);
                setSlot(state.pointShadowMatrix, pointLength, &shadow->matrix, slotsChanged);

                ++numPointShadows;
            }

            setSlot(state.point, pointLength, uniforms, slotsChanged);

            ++pointLength;

        } else if (auto rectAreaLight = light->as<RectAreaLight>()) {

            const auto uniforms = cache_.get(*light);

            std::get<Color>(uniforms->at("color")).copy(light->color).multiplyScalar(light->intensity);

            setSlot(state.rectArea, rectAreaLength, uniforms, slotsChanged);

            ++rectAreaLength;

        } else if (auto hemisphereLight = light->as<HemisphereLight>()) {

            const auto uniforms = cache_.get(*light);

            std::get<Color>(uniforms->at("skyColor")).copy(hemisphereLight->color).multiplyScalar(intensity);
            std::get<Color>(uniforms->at("groundColor")).copy(hemisphereLight->groundColor).multiplyScalar(intensity);

            setSlot(state.hemi, hemiLength, uniforms, slotsChanged);

            ++hemiLength;
        }
    }

    state.ambient.setRGB(r, g, b);

    auto& hash = state.hash;

    if (hash.directionalLength != directionalLength ||
        hash.pointLength != pointLength ||
        hash.spotLength != spotLength ||
        hash.rectAreaLength != rectAreaLength ||
        hash.hemiLength != hemiLength ||
        hash.numDirectionalShadows != numDirectionalShadows ||
        hash.numPointShadows != numPointShadows ||
        hash.numSpotShadows != numSpotShadows ||
        hash.ambient != state.ambient ||
        slotsChanged) {

        state.directional.resize(directionalLength);
        state.spot.resize(spotLength);
        state.point.resize(pointLength);
        state.rectArea.resize(rectAreaLength);
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
        hash.rectAreaLength = rectAreaLength;
        hash.hemiLength = hemiLength;

        hash.numDirectionalShadows = numDirectionalShadows;
        hash.numPointShadows = numPointShadows;
        hash.numSpotShadows = numSpotShadows;
        hash.ambient = state.ambient;

        state.version = nextVersion++;
    }
}

void Lights::setupView(std::vector<Light*>& lights, Camera* camera) {

    int directionalLength = 0;
    int pointLength = 0;
    int spotLength = 0;
    int rectAreaLength = 0;
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

        } else if (auto rectAreaLight = light->as<RectAreaLight>()) {

            auto& uniforms = state.rectArea.at(rectAreaLength);

            auto& position = std::get<Vector3>(uniforms->at("position"));
            auto& halfWidth = std::get<Vector3>(uniforms->at("halfWidth"));
            auto& halfHeight = std::get<Vector3>(uniforms->at("halfHeight"));

            position.setFromMatrixPosition(*rectAreaLight->matrixWorld);
            position.applyMatrix4(viewMatrix);

            // Local quad lies in the XY plane (emissive face toward -Z);
            // width spans ±X, height spans ±Y. Transform those half-axes into
            // view space using only the rotation of the light's world * view
            // matrix — applyMatrix4 on a pure-rotation matrix leaves vectors
            // without picking up translation.
            Matrix4 rotation;
            Matrix4 lightToView;
            lightToView.multiplyMatrices(viewMatrix, *rectAreaLight->matrixWorld);
            rotation.extractRotation(lightToView);

            halfWidth.set(rectAreaLight->width * 0.5f, 0.f, 0.f);
            halfHeight.set(0.f, rectAreaLight->height * 0.5f, 0.f);
            halfWidth.applyMatrix4(rotation);
            halfHeight.applyMatrix4(rotation);

            ++rectAreaLength;

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
