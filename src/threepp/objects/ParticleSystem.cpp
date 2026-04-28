
#include "threepp/objects/ParticleSystem.hpp"

#include "threepp/constants.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"
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
    * Rewritten to use billboard quads instead of GL points for WebGPU compatibility.
    */

    /////////////
    // SHADERS //
    /////////////

    // Billboard quad vertex shader — works on both GL and WebGPU (no gl_PointSize).
    // Uses only the 4 standard vertex attributes (position, normal, uv, color):
    //   position = particle center (world space)
    //   normal   = {size, angle, opacity}  (opacity=0 for invisible)
    //   uv       = quad corner offset (-0.5 to 0.5)
    //   color    = particle RGB color
    std::string particleVertexShader =
            R"(
                varying vec4  vColor;
                varying vec2  vUv;

                void main()
                {
                    float pSize    = normal.x;
                    float pAngle   = normal.y;
                    float pOpacity = normal.z;

                    vColor = vec4(color, pOpacity);

                    // Rotate corner offset by particle angle
                    float c = cos(pAngle);
                    float s = sin(pAngle);
                    vec2 rotated = vec2(
                        c * uv.x - s * uv.y,
                        s * uv.x + c * uv.y
                    );

                    // Billboard: transform particle center to clip space
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    vec4 clipPos = projectionMatrix * mvPosition;

                    // Scale by size with distance attenuation, offset in clip space.
                    // projectionMatrix[1][1] = 2/(top-bottom) ≈ 1/tan(fov/2),
                    // so multiplying by it converts from world-like units to NDC.
                    float scale = pSize * projectionMatrix[1][1] / length(mvPosition.xyz);
                    clipPos.xy += rotated * scale * clipPos.w;

                    gl_Position = clipPos;

                    // Derive texture UV from corner offset
                    vUv = uv + vec2(0.5);
                })";

    // Fragment shader — uses UV derived from quad corner offset.
    std::string particleFragmentShader =
            R"(
                uniform sampler2D tex;
                varying vec4 vColor;
                varying vec2 vUv;

                void main()
                {
                    gl_FragColor = vColor * texture2D(tex, vUv);
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
            unsigned i = 0;
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

    class Particle {

    public:
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

    // Helper: set per-particle attribute data for all 4 quad vertices
    inline void setQuadXYZ(FloatBufferAttribute* attr, unsigned particleIdx, float x, float y, float z) {
        unsigned vi = particleIdx * 4;
        attr->setXYZ(vi, x, y, z);
        attr->setXYZ(vi + 1, x, y, z);
        attr->setXYZ(vi + 2, x, y, z);
        attr->setXYZ(vi + 3, x, y, z);
    }

}// namespace


struct ParticleSystem::Impl {

    Settings settings;

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

        if (settings.positionStyle == Type::BOX)
            particle.position = randomVector3(settings.positionBase, settings.positionSpread);
        if (settings.positionStyle == Type::SPHERE) {
            auto z = 2 * math::randFloat() - 1;
            auto t = 6.2832f * math::randFloat();
            auto r = std::sqrt(1 - z * z);
            Vector3 vec3(r * std::cos(t), r * std::sin(t), z);
            particle.position = Vector3().addVectors(settings.positionBase, vec3.multiplyScalar(settings.positionRadius));
        }

        if (settings.velocityStyle == Type::BOX) {
            particle.velocity = randomVector3(settings.velocityBase, settings.velocitySpread);
        }
        if (settings.velocityStyle == Type::SPHERE) {
            auto direction = Vector3().subVectors(particle.position, settings.positionBase);
            auto speed = randomValue(settings.speedBase, settings.speedSpread);
            particle.velocity = direction.normalize().multiplyScalar(speed);
        }

        particle.acceleration = randomVector3(settings.accelerationBase, settings.accelerationSpread);

        particle.angle = randomValue(settings.angleBase, settings.angleSpread);
        particle.angleVelocity = randomValue(settings.angleVelocityBase, settings.angleVelocitySpread);
        particle.angleAcceleration = randomValue(settings.angleAccelerationBase, settings.angleAccelerationSpread);

        particle.size = randomValue(settings.sizeBase, settings.sizeSpread);

        auto color = randomVector3(settings.colorBase, settings.colorSpread);
        particle.color = Color().setHSL(color.x, color.y, color.z);

        particle.opacity = randomValue(settings.opacityBase, settings.opacitySpread);

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

        sizeTween = Tween<float>(settings.size.first, settings.size.second);
        opacityTween = Tween<float>(settings.opacity.first, settings.opacity.second);
        colorTween = Tween<Vector3>(settings.color.first, settings.color.second);

        particleCount = settings.particlesPerSecond * std::min(settings.particleDeathAge, settings.emitterDeathAge);

        particleMaterial = ShaderMaterial::create();
        particleMaterial->vertexShader = particleVertexShader;
        particleMaterial->fragmentShader = particleFragmentShader;
        particleMaterial->transparent = true;
        particleMaterial->side = Side::Double;
        particleMaterial->vertexColors = true;

        // Build quad geometry: 4 vertices and 6 indices per particle.
        // Standard vertex attributes are repurposed:
        //   position(3) = particle center (world space)
        //   normal(3)   = {size, angle, opacity}
        //   uv(2)       = quad corner offset (-0.5 to 0.5)
        //   color(3)    = particle RGB color
        size_t vertCount = particleCount * 4;
        std::vector<float> positions(vertCount * 3, 0.0f);
        std::vector<float> normals(vertCount * 3, 0.0f);
        std::vector<float> uvs(vertCount * 2);
        std::vector<float> colors(vertCount * 3, 0.0f);
        std::vector<int> indices(particleCount * 6);

        // Corner offsets for each quad vertex
        const float cu[] = {-0.5f, 0.5f, 0.5f, -0.5f};
        const float cv[] = {-0.5f, -0.5f, 0.5f, 0.5f};

        for (size_t i = 0; i < particleCount; i++) {
            size_t vi = i * 4;
            for (int j = 0; j < 4; j++) {
                uvs[(vi + j) * 2 + 0] = cu[j];
                uvs[(vi + j) * 2 + 1] = cv[j];
            }
            size_t ii = i * 6;
            auto v = static_cast<int>(vi);
            indices[ii + 0] = v;     indices[ii + 1] = v + 1; indices[ii + 2] = v + 2;
            indices[ii + 3] = v;     indices[ii + 4] = v + 2; indices[ii + 5] = v + 3;
        }

        particleGeometry = BufferGeometry::create();
        particleGeometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        particleGeometry->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
        particleGeometry->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
        particleGeometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));
        particleGeometry->setIndex(indices);

        if (settings.texture) {
            particleMaterial->uniforms["tex"].setValue(settings.texture.get());
            particleMaterial->depthWrite = false;
        }

        // link particle data with geometry/material data
        auto posAttr = particleGeometry->getAttribute<float>("position");
        auto normalAttr = particleGeometry->getAttribute<float>("normal");
        auto colorAttr = particleGeometry->getAttribute<float>("color");
        for (unsigned i = 0; i < particleCount; i++) {
            this->particleArray.emplace_back(createParticle());
            auto& p = this->particleArray[i];

            float opacity = p.alive ? *p.opacity : 0.0f;
            setQuadXYZ(posAttr, i, p.position.x, p.position.y, p.position.z);
            setQuadXYZ(normalAttr, i, p.size, p.angle, opacity);
            setQuadXYZ(colorAttr, i, p.color->r, p.color->g, p.color->b);
        }

        this->particleMaterial->blending = settings.blendStyle;
        if (settings.blendStyle != Blending::Normal) {
            this->particleMaterial->depthTest = false;
        }

        this->particleMesh = Mesh::create(this->particleGeometry, this->particleMaterial);
        scope.add(this->particleMesh);
    }

    void update(float dt) {
        std::vector<unsigned int> recycleIndices;

        auto posAttr = particleGeometry->getAttribute<float>("position");
        auto normalAttr = particleGeometry->getAttribute<float>("normal");
        auto colorAttr = particleGeometry->getAttribute<float>("color");

        // update particle data
        for (unsigned i = 0; i < this->particleCount; i++) {
            if (this->particleArray[i].alive) {
                this->particleArray[i].update(dt);

                // check if particle should expire
                if (this->particleArray[i].age > settings.particleDeathAge) {
                    this->particleArray[i].alive = false;
                    recycleIndices.emplace_back(i);
                }

                // update particle properties in shader (all 4 quad vertices)
                auto& p = this->particleArray[i];
                float opacity = p.alive ? *p.opacity : 0.0f;
                setQuadXYZ(posAttr, i, p.position.x, p.position.y, p.position.z);
                setQuadXYZ(normalAttr, i, p.size, p.angle, opacity);
                setQuadXYZ(colorAttr, i, p.color->r, p.color->g, p.color->b);
            }
        }

        posAttr->needsUpdate();
        normalAttr->needsUpdate();
        colorAttr->needsUpdate();

        // check if particle emitter is still running
        if (!this->emitterAlive) return;

        // if no particles have died yet, then there are still particles to activate
        if (this->emitterAge < settings.particleDeathAge) {
            // determine indices of particles to activate
            const auto startIndex = static_cast<int>(std::round(settings.particlesPerSecond * (this->emitterAge + 0)));
            auto endIndex = static_cast<size_t>(std::round(settings.particlesPerSecond * (this->emitterAge + dt)));
            if (endIndex > this->particleCount) {
                endIndex = this->particleCount;
            }

            for (unsigned i = startIndex; i < endIndex; i++)
                this->particleArray[i].alive = true;
        }

        // if any particles have died while the emitter is still running, we immediately recycle them
        for (unsigned int i : recycleIndices) {
            this->particleArray[i] = this->createParticle();
            this->particleArray[i].alive = true;// activate right away
            auto& p = this->particleArray[i];
            setQuadXYZ(posAttr, i, p.position.x, p.position.y, p.position.z);
            setQuadXYZ(normalAttr, i, p.size, p.angle, *p.opacity);
            setQuadXYZ(colorAttr, i, p.color->r, p.color->g, p.color->b);
        }
        posAttr->needsUpdate();

        // stop emitter?
        this->emitterAge += dt;
        if (this->emitterAge > settings.emitterDeathAge) {
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

ParticleSystem::Settings& threepp::ParticleSystem::settings() {
    return pimpl_->settings;
}

ParticleSystem::~ParticleSystem() = default;


ParticleSystem::Settings::Settings() {
    makeDefault();
}

ParticleSystem::Settings& ParticleSystem::Settings::setSizeTween(const std::vector<float>& times, const std::vector<float>& values) {
    size = {times, values};
    return *this;
}

ParticleSystem::Settings& ParticleSystem::Settings::setOpacityTween(const std::vector<float>& times, const std::vector<float>& values) {
    opacity = {times, values};
    return *this;
}

ParticleSystem::Settings& ParticleSystem::Settings::setColorTween(const std::vector<float>& times, const std::vector<Vector3>& values) {
    color = {times, values};
    return *this;
}

void ParticleSystem::Settings::makeDefault() {
    positionStyle = Type::BOX;
    positionBase = {};
    positionSpread = {};
    positionRadius = 0;// distance from base at which particles start

    velocityStyle = Type::BOX;

    velocityBase = {};
    velocitySpread = {};

    speedBase = 0;
    speedSpread = 0;

    accelerationBase = {};
    accelerationSpread = {};

    angleBase = 0;
    angleSpread = 0;
    angleVelocityBase = 0;
    angleVelocitySpread = 0;
    angleAccelerationBase = 0;
    angleAccelerationSpread = 0;

    sizeBase = 0.0;
    sizeSpread = 0.0;

    // store colors in HSL format in a THREE.Vector3 object
    // http://en.wikipedia.org/wiki/HSL_and_HSV
    colorBase = Vector3(0.0f, 1.0f, 0.5f);
    colorSpread = Vector3(0.0f, 0.0f, 0.0f);

    opacityBase = 1.0;
    opacitySpread = 0.0;

    blendStyle = Blending::Normal;// false;

    particlesPerSecond = 100;
    particleDeathAge = 1.0;

    ////////////////////////
    // EMITTER PROPERTIES //
    ////////////////////////
    emitterDeathAge = 60;// time (seconds) at which to stop creating particles.

    texture = nullptr;

    size = {{}, {}};
    opacity = {{}, {}};
    color = {{}, {}};
}
