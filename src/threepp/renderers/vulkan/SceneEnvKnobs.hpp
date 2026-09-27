// SceneEnvKnobs — the Vulkan renderer's reading of three.js's five scene
// environment/background controls (Scene::environmentIntensity,
// environmentRotation, backgroundIntensity, backgroundRotation,
// backgroundBlurriness), resolved once per frame with GLRenderer's
// conventions (commit 9804c236):
//
//  - a rotation R = makeRotationFromEuler(euler) takes ENV directions to WORLD
//    ones; every lookup uses its transpose (world -> env), and the extracted
//    env sun is turned by R itself, so it is reported in world space;
//  - environmentIntensity MULTIPLIES material.envMapIntensity (as on GL);
//  - the background knobs apply only to a TEXTURE background (GL leaves a
//    Color background alone).
//
// Vulkan has one environment image: scene.environment, or the background
// texture when there is none. In that fallback the image IS the background,
// so its lookups turn with backgroundRotation, keeping a mirror's reflection
// on the sky it reflects (GL has no counterpart: with no scene.environment a
// GL material gets no IBL at all).

#ifndef THREEPP_VULKAN_SCENE_ENV_KNOBS_HPP
#define THREEPP_VULKAN_SCENE_ENV_KNOBS_HPP

#include "threepp/math/Euler.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>

namespace threepp::vulkan {

    struct SceneEnvKnobs {
        // Env -> world rotation, column-major 3x3 (the Euler's own rotation).
        float envToWorld[9]{1, 0, 0, 0, 1, 0, 0, 0, 1};
        bool  envRotActive = false;
        float envIntensity = 1.f;
        float bgToWorld[9]{1, 0, 0, 0, 1, 0, 0, 0, 1};
        bool  bgRotActive = false;
        float bgIntensity  = 1.f;
        float bgBlurriness = 0.f;

        // Column-major world -> env (the transpose), as the GLSL mat3 columns.
        static void worldToEnvColumns(const float envToWorld[9], float out[9]) {
            for (int c = 0; c < 3; ++c)
                for (int r = 0; r < 3; ++r) out[c * 3 + r] = envToWorld[r * 3 + c];
        }

        // Rotate an env-space direction to world space (the env sun).
        void envDirToWorld(const float in[3], float out[3]) const {
            for (int r = 0; r < 3; ++r)
                out[r] = envToWorld[0 * 3 + r] * in[0] + envToWorld[1 * 3 + r] * in[1] + envToWorld[2 * 3 + r] * in[2];
        }

        bool operator==(const SceneEnvKnobs& o) const {
            return std::equal(envToWorld, envToWorld + 9, o.envToWorld) && envRotActive == o.envRotActive &&
                   envIntensity == o.envIntensity && std::equal(bgToWorld, bgToWorld + 9, o.bgToWorld) &&
                   bgRotActive == o.bgRotActive && bgIntensity == o.bgIntensity && bgBlurriness == o.bgBlurriness;
        }
        bool operator!=(const SceneEnvKnobs& o) const { return !(*this == o); }

        static SceneEnvKnobs fromScene(const Object3D& scene) {
            SceneEnvKnobs k;
            const auto* sc = dynamic_cast<const Scene*>(&scene);
            if (!sc) return k;

            const auto rotation = [](const Euler& e, float out[9]) {
                if (static_cast<float>(e.x) == 0 && static_cast<float>(e.y) == 0 && static_cast<float>(e.z) == 0)
                    return false;
                Matrix4 m;
                m.makeRotationFromEuler(e);
                for (int c = 0; c < 3; ++c)
                    for (int r = 0; r < 3; ++r) out[c * 3 + r] = m.elements[c * 4 + r];
                return true;
            };

            const bool textureBackground = sc->background.isTexture();
            if (textureBackground) {
                k.bgRotActive  = rotation(sc->backgroundRotation, k.bgToWorld);
                k.bgIntensity  = sc->backgroundIntensity;
                k.bgBlurriness = std::clamp(sc->backgroundBlurriness, 0.f, 1.f);
            }
            const bool envIsBackground = !sc->environment && textureBackground;
            if (envIsBackground) {
                k.envRotActive = k.bgRotActive;
                std::copy(k.bgToWorld, k.bgToWorld + 9, k.envToWorld);
            } else {
                k.envRotActive = rotation(sc->environmentRotation, k.envToWorld);
            }
            k.envIntensity = sc->environmentIntensity;
            return k;
        }
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_SCENE_ENV_KNOBS_HPP
