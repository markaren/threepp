// https://stemkoski.github.io/Three.js/Particle-Engine.html

#ifndef THREEPP_PARTICLESYSTEM_HPP
#define THREEPP_PARTICLESYSTEM_HPP

#include "threepp/constants.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/textures/Texture.hpp"

namespace threepp {

    class ParticleSystem: public Object3D {

    public:
        enum class Type {
            BOX,
            SPHERE
        };

        struct Settings {

            Type positionStyle;
            Vector3 positionBase;
            Vector3 positionSpread;
            float positionRadius{};// distance from base at which particles start

            Type velocityStyle;
            // cube movement data
            Vector3 velocityBase;
            Vector3 velocitySpread;
            // sphere movement data
            // direction vector calculated using initial position
            float speedBase{};
            float speedSpread{};

            Vector3 accelerationBase;
            Vector3 accelerationSpread;

            float angleBase{};
            float angleSpread{};
            float angleVelocityBase{};
            float angleVelocitySpread{};
            float angleAccelerationBase{};
            float angleAccelerationSpread{};

            float sizeBase{};
            float sizeSpread{};

            // store colors in HSL format in a THREE.Vector3 object
            // http://en.wikipedia.org/wiki/HSL_and_HSV
            Vector3 colorBase;
            Vector3 colorSpread;

            float opacityBase{};
            float opacitySpread{};

            Blending blendStyle;

            int particlesPerSecond{};
            float particleDeathAge{};

            ////////////////////////
            // EMITTER PROPERTIES //
            ////////////////////////
            float emitterDeathAge{};// time (seconds) at which to stop creating particles.

            std::shared_ptr<Texture> texture;

            Settings& setSizeTween(const std::vector<float>& times, const std::vector<float>& values);
            Settings& setColorTween(const std::vector<float>& times, const std::vector<Vector3>& values);
            Settings& setOpacityTween(const std::vector<float>& times, const std::vector<float>& values);

            void makeDefault();

        private:
            friend class ParticleSystem;

            Settings();

            std::pair<std::vector<float>, std::vector<float>> size;
            std::pair<std::vector<float>, std::vector<float>> opacity;
            std::pair<std::vector<float>, std::vector<Vector3>> color;
        };

        ParticleSystem();

        Settings& settings();

        void initialize();

        void update(float dt);

        ~ParticleSystem() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_PARTICLESYSTEM_HPP
