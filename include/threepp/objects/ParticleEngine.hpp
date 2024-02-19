// https://stemkoski.github.io/Three.js/Particle-Engine.html

#ifndef THREEPP_PARTICLEENGINE_HPP
#define THREEPP_PARTICLEENGINE_HPP

#include "threepp/constants.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/math/Vector3.hpp"

namespace threepp {

    class ParticleEngine : public Object3D {

    public:
        enum class Type {
            BOX, SPHERE
        };

        Type positionStyle = Type::BOX;
        Vector3 positionBase;
        Vector3 positionSpread;
        float positionRadius = 0;// distance from base at which particles start

        Type velocityStyle = Type::BOX;
        // cube movement data
        Vector3 velocityBase;
        Vector3 velocitySpread;
        // sphere movement data
        // direction vector calculated using initial position
        float speedBase = 0;
        float speedSpread = 0;

        Vector3 accelerationBase;
        Vector3 accelerationSpread;

        float angleBase = 0;
        float angleSpread = 0;
        float angleVelocityBase = 0;
        float angleVelocitySpread = 0;
        float angleAccelerationBase = 0;
        float angleAccelerationSpread = 0;

        float sizeBase = 0.0;
        float sizeSpread = 0.0;

        // store colors in HSL format in a THREE.Vector3 object
        // http://en.wikipedia.org/wiki/HSL_and_HSV
        Vector3 colorBase = Vector3(0.0f, 1.0f, 0.5f);
        Vector3 colorSpread = Vector3(0.0f, 0.0f, 0.0f);

        float opacityBase = 1.0;
        float opacitySpread = 0.0;

        Blending blendStyle = Blending::Normal;// false;

        int particlesPerSecond = 100;
        float particleDeathAge = 1.0;

        ////////////////////////
        // EMITTER PROPERTIES //
        ////////////////////////
        float emitterDeathAge = 60;// time (seconds) at which to stop creating particles.

        ParticleEngine();

        void initialize();

        void update(float dt);

        void setValues();

        ~ParticleEngine();

    private:

        struct Impl;
        std::unique_ptr<Impl> pimpl_;

    };

}

#endif//THREEPP_PARTICLEENGINE_HPP
