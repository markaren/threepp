
#include "threepp/objects/ParticleSystem.hpp"

#include "threepp/constants.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/textures/Texture.hpp"

#include <cmath>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

using namespace threepp;

namespace {

    /**
    * Original three.js code by Lee Stemkoski   http://www.adelphi.edu/~stemkoski/
    */

    ///////////////////////////////////////////////////////////////////////////////

    /////////////
    // SHADERS //
    /////////////

    std::string particleVertexShader =

            R"(
                in vec3  customColor;
                in float customOpacity;
                in float customSize;
                in float customAngle;
                in float customVisible;  // float used as boolean (0 = false, 1 = true)
                out vec4  vColor;
                out float vAngle;
                void main()
                {
                    if ( customVisible > 0.5) 				// true
                    vColor = vec4( customColor, customOpacity ); //     set color associated to vertex; use later in fragment shader.
                    else							// false
                    vColor = vec4(0.0, 0.0, 0.0, 0.0); 		//     make particle invisible.

                    vAngle = customAngle;

                    vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );
                    gl_PointSize = customSize * ( 300.0 / length( mvPosition.xyz ) );     // scale particles as objects in 3D space
                    gl_Position = projectionMatrix * mvPosition;
                })";

    std::string particleFragmentShader =
            R"(
                uniform sampler2D tex;
                in vec4 vColor;
                in float vAngle;
                void main()
                {
                    gl_FragColor = vColor;

                    float c = cos(vAngle);
                    float s = sin(vAngle);
                    vec2 rotatedUV = vec2(c * (gl_PointCoord.x - 0.5) + s * (gl_PointCoord.y - 0.5) + 0.5,
                    c * (gl_PointCoord.y - 0.5) - s * (gl_PointCoord.x - 0.5) + 0.5);  // rotate UV coordinates to rotate texture
                    vec4 rotatedTexture = texture2D( tex,  rotatedUV );
                    gl_FragColor = gl_FragColor * rotatedTexture;    // sets an otherwise white particle texture to desired color
                })";

    // helper functions for randomization
    float randomValue(float base, float spread) {
        return base + spread * (math::randFloat() - 0.5f);
    }

    Vector3 randomVector3(const Vector3& base, const Vector3& spread) {
        Vector3 rand3(math::randFloat() - 0.5f, math::randFloat() - 0.5f, math::randFloat() - 0.5f);
        Vector3 rand = Vector3().addVectors(base, Vector3().multiplyVectors(spread, rand3));
        return rand;
    }

    template<class T>
    struct Tween {

        explicit Tween(const std::vector<float>& vectimeArray = {}, const std::vector<T>& valueArray = {})
            : times(vectimeArray), values(valueArray) {}

        [[nodiscard]] T lerp(float t) const {
            int i = 0;
            auto n = this->times.size();
            while (i < n && t > this->times[i]) {
                i++;
            }
            if (i == 0) return this->values[0];
            if (i == n) return this->values[n - 1];
            auto p = (t - this->times[i - 1]) / (this->times[i] - this->times[i - 1]);
            if constexpr (std::is_floating_point_v<T>) {
                return this->values[i - 1] + p * (this->values[i] - this->values[i - 1]);
            } else {
                return this->values[i - 1].clone().lerp(this->values[i], p);
            }
        }

    private:
        friend class Particle;

        std::vector<float> times;
        std::vector<T> values;
    };

    struct Particle {

        Vector3 position;
        Vector3 velocity;
        Vector3 acceleration;

        float angle{};
        float angleVelocity{};
        float angleAcceleration{};

        float size = 16;

        float age{};
        bool alive{};

        Tween<float>* sizeTween = nullptr;
        Tween<Vector3>* colorTween = nullptr;
        Tween<float>* opacityTween = nullptr;

        std::optional<Color> color;
        std::optional<float> opacity;

        void update(float dt) {
            this->position.addScaledVector(this->velocity, dt);
            this->velocity.addScaledVector(this->acceleration, dt);

            // convert from degrees to radians: 0.01745329251 = Math.PI/180
            this->angle += angleVelocity * 0.01745329251f * dt;
            this->angleVelocity += angleAcceleration * 0.01745329251f * dt;

            this->age += dt;

            // if the tween for a given attribute is nonempty,
            //  then use it to update the attribute's value

            if (!this->sizeTween->times.empty())
                this->size = this->sizeTween->lerp(this->age);

            if (!this->colorTween->times.empty()) {
                const auto colorHSL = this->colorTween->lerp(this->age);
                this->color = Color().setHSL(colorHSL.x, colorHSL.y, colorHSL.z);
            }

            if (!this->opacityTween->times.empty()) {
                this->opacity = this->opacityTween->lerp(this->age);
            }
        }
    };

}// namespace


struct ParticleSystem::Impl {

    Tween<float> sizeTween;
    Tween<Vector3> colorTween;
    Tween<float> opacityTween;
    std::vector<Particle> particleArray;

    float emitterAge = 0.0;
    bool emitterAlive = true;

    // How many particles could be active at any time?
    size_t particleCount{};

    std::shared_ptr<BufferGeometry> particleGeometry = nullptr;
    std::shared_ptr<ShaderMaterial> particleMaterial = nullptr;
    std::shared_ptr<Object3D> particleMesh;

    ParticleSystem& scope;

    explicit Impl(ParticleSystem& scope)
        : scope(scope) {}

    [[nodiscard]] Particle createParticle() {
        Particle particle;

        particle.sizeTween = &sizeTween;
        particle.colorTween = &colorTween;
        particle.opacityTween = &opacityTween;

        if (scope.positionStyle == Type::BOX)
            particle.position = randomVector3(scope.positionBase, scope.positionSpread);
        if (scope.positionStyle == Type::SPHERE) {
            auto z = 2 * math::randFloat() - 1;
            auto t = 6.2832f * math::randFloat();
            auto r = std::sqrt(1 - z * z);
            Vector3 vec3(r * std::cos(t), r * std::sin(t), z);
            particle.position = Vector3().addVectors(scope.positionBase, vec3.multiplyScalar(scope.positionRadius));
        }

        if (scope.velocityStyle == Type::BOX) {
            particle.velocity = randomVector3(scope.velocityBase, scope.velocitySpread);
        }
        if (scope.velocityStyle == Type::SPHERE) {
            auto direction = Vector3().subVectors(particle.position, scope.positionBase);
            auto speed = randomValue(scope.speedBase, scope.speedSpread);
            particle.velocity = direction.normalize().multiplyScalar(speed);
        }

        particle.acceleration = randomVector3(scope.accelerationBase, scope.accelerationSpread);

        particle.angle = randomValue(scope.angleBase, scope.angleSpread);
        particle.angleVelocity = randomValue(scope.angleVelocityBase, scope.angleVelocitySpread);
        particle.angleAcceleration = randomValue(scope.angleAccelerationBase, scope.angleAccelerationSpread);

        particle.size = randomValue(scope.sizeBase, scope.sizeSpread);

        auto color = randomVector3(scope.colorBase, scope.colorSpread);
        particle.color = Color().setHSL(color.x, color.y, color.z);

        particle.opacity = randomValue(scope.opacityBase, scope.opacitySpread);

        particle.age = 0;
        particle.alive = false;// particles initialize as inactive

        return particle;
    }

    void reset() {
        particleArray = {};
        emitterAge = 0.0;
        emitterAlive = true;

        if (particleMesh) {
            scope.remove(*particleMesh);
        }
    }

    void initialize() {

        reset();

        particleCount = scope.particlesPerSecond * std::min(scope.particleDeathAge, scope.emitterDeathAge);

        particleMaterial = ShaderMaterial::create();
        particleMaterial->vertexShader = particleVertexShader;
        particleMaterial->fragmentShader = particleFragmentShader;
        particleMaterial->transparent = true;

        particleGeometry = BufferGeometry::create();
        particleGeometry->setAttribute("position", FloatBufferAttribute::create(std::vector<float>(particleCount * 3), 3));
        particleGeometry->setAttribute("customVisible", FloatBufferAttribute::create(std::vector<float>(particleCount), 1));
        particleGeometry->setAttribute("customAngle", FloatBufferAttribute::create(std::vector<float>(particleCount), 1));
        particleGeometry->setAttribute("customSize", FloatBufferAttribute::create(std::vector<float>(particleCount), 1));
        particleGeometry->setAttribute("customColor", FloatBufferAttribute::create(std::vector<float>(particleCount * 3), 3));
        particleGeometry->setAttribute("customOpacity", FloatBufferAttribute::create(std::vector<float>(particleCount), 1));

        if (scope.texture) {
            particleMaterial->uniforms->operator[]("tex").setValue(scope.texture.get());
            particleMaterial->depthWrite = false;
        }

        // link particle data with geometry/material data
        for (auto i = 0; i < particleCount; i++) {
            // remove duplicate code somehow, here and in update function below.
            this->particleArray.emplace_back(createParticle());

            auto position = particleGeometry->getAttribute<float>("position");
            position->setXYZ(i, this->particleArray[i].position.x, this->particleArray[i].position.y, this->particleArray[i].position.z);

            auto customVisible = particleGeometry->getAttribute<float>("customVisible");
            customVisible->setX(i, this->particleArray[i].alive);

            auto customColor = particleGeometry->getAttribute<float>("customColor");
            customColor->setXYZ(i, this->particleArray[i].color->r, this->particleArray[i].color->g, this->particleArray[i].color->b);

            auto customOpacity = particleGeometry->getAttribute<float>("customOpacity");
            customOpacity->setX(i, *this->particleArray[i].opacity);

            auto customSize = particleGeometry->getAttribute<float>("customSize");
            customSize->setX(i, this->particleArray[i].size);

            auto customAngle = particleGeometry->getAttribute<float>("customAngle");
            customAngle->setX(i, this->particleArray[i].angle);
        }

        this->particleMaterial->blending = scope.blendStyle;
        if (scope.blendStyle != Blending::Normal) {
            this->particleMaterial->depthTest = false;
        }

        this->particleMesh = Points::create(this->particleGeometry, this->particleMaterial);
        scope.add(this->particleMesh);
    }

    void update(float dt) {
        std::vector<unsigned int> recycleIndices;

        auto position = particleGeometry->getAttribute<float>("position");
        auto customVisible = this->particleGeometry->getAttribute<float>("customVisible");
        auto customOpacity = this->particleGeometry->getAttribute<float>("customOpacity");
        auto customSize = this->particleGeometry->getAttribute<float>("customSize");
        auto customAngle = this->particleGeometry->getAttribute<float>("customAngle");
        auto customColor = this->particleGeometry->getAttribute<float>("customColor");

        // update particle data
        for (auto i = 0; i < this->particleCount; i++) {
            if (this->particleArray[i].alive) {
                this->particleArray[i].update(dt);

                // check if particle should expire
                // could also use: death by size<0 or alpha<0.
                if (this->particleArray[i].age > scope.particleDeathAge) {
                    this->particleArray[i].alive = false;
                    recycleIndices.emplace_back(i);
                }
                // update particle properties in shader

                position->setXYZ(i, this->particleArray[i].position.x, this->particleArray[i].position.y, this->particleArray[i].position.z);
                customVisible->setX(i, this->particleArray[i].alive);
                customOpacity->setX(i, *this->particleArray[i].opacity);
                customSize->setX(i, this->particleArray[i].size);
                customAngle->setX(i, this->particleArray[i].angle);
                customColor->setXYZ(i, this->particleArray[i].color->r, this->particleArray[i].color->g, this->particleArray[i].color->b);
            }
        }

        customVisible->needsUpdate();
        customOpacity->needsUpdate();
        customSize->needsUpdate();
        customAngle->needsUpdate();
        customColor->needsUpdate();

        // check if particle emitter is still running
        if (!this->emitterAlive) return;

        // if no particles have died yet, then there are still particles to activate
        if (this->emitterAge < scope.particleDeathAge) {
            // determine indices of particles to activate
            const auto startIndex = static_cast<int>(std::round(scope.particlesPerSecond * (this->emitterAge + 0)));
            auto endIndex = static_cast<int>(std::round(scope.particlesPerSecond * (this->emitterAge + dt)));
            if (endIndex > this->particleCount) {
                endIndex = this->particleCount;
            }

            for (auto i = startIndex; i < endIndex; i++)
                this->particleArray[i].alive = true;
        }

        // if any particles have died while the emitter is still running, we imediately recycle them
        for (unsigned int i : recycleIndices) {
            this->particleArray[i] = this->createParticle();
            this->particleArray[i].alive = true;// activate right away
            position->setXYZ(i, this->particleArray[i].position.x, this->particleArray[i].position.y, this->particleArray[i].position.z);
        }
        position->needsUpdate();

        // stop emitter?
        this->emitterAge += dt;
        if (this->emitterAge > scope.emitterDeathAge) {
            this->emitterAlive = false;
        }
    }
};

ParticleSystem::ParticleSystem()
    : pimpl_(std::make_unique<Impl>(*this)) {}

void ParticleSystem::initialize() {
    pimpl_->initialize();
}

void ParticleSystem::update(float dt) {
    pimpl_->update(dt);
}

ParticleSystem& ParticleSystem::setSizeTween(const std::vector<float>& times, const std::vector<float>& values) {
    pimpl_->sizeTween = Tween<float>(times, values);
    return *this;
}

ParticleSystem& ParticleSystem::setOpacityTween(const std::vector<float>& times, const std::vector<float>& values) {
    pimpl_->opacityTween = Tween<float>(times, values);
    return *this;
}

ParticleSystem& ParticleSystem::setColorTween(const std::vector<float>& times, const std::vector<Vector3>& values) {
    pimpl_->colorTween = Tween<Vector3>(times, values);
    return *this;
}


ParticleSystem::~ParticleSystem() = default;
