#include "threepp/renderers/common/PreethamSky.hpp"

#include "threepp/core/Uniform.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Sky.hpp"

#include <algorithm>
#include <cmath>
#include <variant>

using namespace threepp;

namespace {

    // The constants of Sky.cpp's GLSL, verbatim.
    constexpr float kE = 2.71828182845904523536028747135266249775724709369995957f;
    constexpr float kPi = 3.141592653589793238462643383279502884197169f;
    constexpr float kTotalRayleigh[3] = {5.804542996261093E-6f, 1.3562911419845635E-5f, 3.0265902468824876E-5f};
    constexpr float kMieConst[3] = {1.8399918514433978E14f, 2.7798023919660528E14f, 4.0790479543861094E14f};
    constexpr float kCutoffAngle = 1.6110731556870734f;// pi / 1.95
    constexpr float kSteepness = 1.5f;
    constexpr float kEE = 1000.0f;
    constexpr float kRayleighZenithLength = 8.4E3f;
    constexpr float kMieZenithLength = 1.25E3f;
    constexpr float kSunAngularDiameterCos = 0.999956676946448443553574619906976478926848692873900859324f;
    constexpr float kThreeOverSixteenPi = 0.05968310365946075f;
    constexpr float kOneOverFourPi = 0.07957747154594767f;

    float dot3(const float a[3], const float b[3]) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }

    float sunIntensity(float zenithAngleCos) {
        zenithAngleCos = std::clamp(zenithAngleCos, -1.0f, 1.0f);
        return kEE * std::max(0.0f, 1.0f - std::pow(kE, -((kCutoffAngle - std::acos(zenithAngleCos)) / kSteepness)));
    }

    float smoothstep(float e0, float e1, float x) {
        const float t = std::clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    }

    float rayleighPhase(float cosTheta) {
        return kThreeOverSixteenPi * (1.0f + cosTheta * cosTheta);
    }

    float hgPhase(float cosTheta, float g) {
        const float g2 = g * g;
        const float inverse = 1.0f / std::pow(1.0f - 2.0f * g * cosTheta + g2, 1.5f);
        return kOneOverFourPi * ((1.0f - g2) * inverse);
    }

    template<typename T>
    bool readUniform(const UniformMap& uniforms, const char* name, T& out) {
        const auto it = uniforms.find(name);
        if (it == uniforms.end() || !it->second.hasValue()) return false;
        // Uniform::value() is non-const; nothing is written through it.
        auto& u = const_cast<Uniform&>(it->second);
        if (const T* v = std::get_if<T>(&u.value())) {
            out = *v;
            return true;
        }
        return false;
    }

}// namespace

bool PreethamSkyParams::operator==(const PreethamSkyParams& o) const {
    return sunDir[0] == o.sunDir[0] && sunDir[1] == o.sunDir[1] && sunDir[2] == o.sunDir[2] &&
           sunPosY == o.sunPosY && up[0] == o.up[0] && up[1] == o.up[1] && up[2] == o.up[2] &&
           turbidity == o.turbidity && rayleigh == o.rayleigh && mieCoefficient == o.mieCoefficient &&
           mieDirectionalG == o.mieDirectionalG;
}

PreethamSkyParams threepp::preethamParamsFromSky(const Sky& sky) {
    PreethamSkyParams p;
    const auto* mat = dynamic_cast<const ShaderMaterial*>(sky.material().get());
    if (!mat) return p;
    const UniformMap& u = mat->uniforms;
    Vector3 sunPosition(0, 1, 0);
    if (readUniform(u, "sunPosition", sunPosition)) {
        p.sunPosY = sunPosition.y;
        const float len = sunPosition.length();
        if (len > 0.f) {
            p.sunDir[0] = sunPosition.x / len;
            p.sunDir[1] = sunPosition.y / len;
            p.sunDir[2] = sunPosition.z / len;
        }
    }
    Vector3 up(0, 1, 0);
    if (readUniform(u, "up", up)) {
        const float len = up.length();
        if (len > 0.f) {
            p.up[0] = up.x / len;
            p.up[1] = up.y / len;
            p.up[2] = up.z / len;
        }
    }
    readUniform(u, "turbidity", p.turbidity);
    readUniform(u, "rayleigh", p.rayleigh);
    readUniform(u, "mieCoefficient", p.mieCoefficient);
    readUniform(u, "mieDirectionalG", p.mieDirectionalG);
    return p;
}

void threepp::preethamRadiance(const PreethamSkyParams& p, const float dir[3], bool withDisc, float outRgb[3]) {
    // Sky.cpp's vertex stage: the per-sun constants.
    const float sunE = sunIntensity(dot3(p.sunDir, p.up));
    const float sunfade = 1.0f - std::clamp(1.0f - std::exp(p.sunPosY / 450000.0f), 0.0f, 1.0f);
    const float rayleighCoefficient = p.rayleigh - (1.0f * (1.0f - sunfade));
    float betaR[3], betaM[3];
    {
        const float c = (0.2f * p.turbidity) * 10E-18f;
        for (int i = 0; i < 3; ++i) {
            betaR[i] = kTotalRayleigh[i] * rayleighCoefficient;
            betaM[i] = 0.434f * c * kMieConst[i] * p.mieCoefficient;
        }
    }
    // Its fragment stage.
    const float zenithAngle = std::acos(std::max(0.0f, dot3(p.up, dir)));
    const float inverse = 1.0f / (std::cos(zenithAngle) + 0.15f * std::pow(93.885f - ((zenithAngle * 180.0f) / kPi), -1.253f));
    const float sR = kRayleighZenithLength * inverse;
    const float sM = kMieZenithLength * inverse;
    const float cosTheta = dot3(dir, p.sunDir);
    const float rPhase = rayleighPhase(cosTheta * 0.5f + 0.5f);
    const float mPhase = hgPhase(cosTheta, p.mieDirectionalG);
    const float sunfadeMix = std::clamp(std::pow(1.0f - dot3(p.up, p.sunDir), 5.0f), 0.0f, 1.0f);
    const float sundisk = withDisc ? smoothstep(kSunAngularDiameterCos, kSunAngularDiameterCos + 0.00002f, cosTheta) : 0.0f;
    const float bias[3] = {0.0f, 0.0003f, 0.00075f};
    const float gammaInv = 1.0f / (1.2f + (1.2f * sunfade));
    for (int i = 0; i < 3; ++i) {
        const float Fex = std::exp(-(betaR[i] * sR + betaM[i] * sM));
        const float betaTheta = betaR[i] * rPhase + betaM[i] * mPhase;
        const float ratio = sunE * (betaTheta / (betaR[i] + betaM[i]));
        float Lin = std::pow(ratio * (1.0f - Fex), 1.5f);
        // mix(1, pow(ratio * Fex, 0.5), sunfadeMix)
        Lin *= 1.0f + (std::pow(ratio * Fex, 0.5f) - 1.0f) * sunfadeMix;
        float L0 = 0.1f * Fex;
        L0 += (sunE * 19000.0f * Fex) * sundisk;
        const float texColor = (Lin + L0) * 0.04f + bias[i];
        outRgb[i] = std::pow(texColor, gammaInv);
    }
}

EnvSunExtract threepp::preethamSun(const PreethamSkyParams& p) {
    EnvSunExtract s;
    float withDisc[3], without[3];
    preethamRadiance(p, p.sunDir, true, withDisc);
    preethamRadiance(p, p.sunDir, false, without);
    // The disc is the cap cosTheta >= sunAngularDiameterCos: solid angle
    // 2 pi (1 - cos r).
    const float omega = 2.0f * kPi * (1.0f - kSunAngularDiameterCos);
    s.found = true;
    for (int i = 0; i < 3; ++i) {
        s.dir[i] = p.sunDir[i];
        s.colorE[i] = std::max(0.0f, withDisc[i] - without[i]) * omega;
    }
    s.angularRadiusDeg = std::acos(kSunAngularDiameterCos) * 180.0f / kPi;
    return s;
}
