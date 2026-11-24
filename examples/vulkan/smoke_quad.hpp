// smoke_quad.hpp: what the smoke demos share: the race quad's rotor layout (racer_spec.json), a figure-eight flight
// through the plume whose attitude is what the path asks of a quad, and a procedural sunset sky.

#ifndef THREEPP_EXAMPLES_VULKAN_SMOKE_QUAD_HPP
#define THREEPP_EXAMPLES_VULKAN_SMOKE_QUAD_HPP

#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace smokedemo {

    using namespace threepp;


    // racer_spec.json: the rotors' places on the drawn model (m, body frame: x forward, y up, z right).
    constexpr float kMotor = 0.07955f, kMotorY = 0.01562f;
    constexpr float kDrawn = 0.45f;     // the .glb is modelled at 500 mm and flown at 225 mm
    constexpr float kPropR = 0.05715f;  // m
    constexpr float kG     = 9.81f;
    inline constexpr const char* kMotorName[4] = {"fr", "fl", "rl", "rr"};
    inline constexpr float kMotorSign[4][2] = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}};// (x, z) signs

    struct Pose {
        Vector3    p, v;
        Quaternion q;
        Vector3    up;        // the thrust axis
        float      thrust = 1;// in g
        Matrix4    basis;
    };

    // A figure-eight through the plume at the origin, about a metre up, and the attitude it asks of a quad.
    inline Pose flightAt(double t) {
        const double w = 3.2, A = 1.25, B = 0.55, H = 1.2, HA = 0.28, ph = 0.8;
        Pose         k;
        k.p.set(float(A * std::sin(w * t)), float(H + HA * std::sin(w * t + ph)), float(B * std::sin(2.0 * w * t)));
        k.v.set(float(A * w * std::cos(w * t)), float(HA * w * std::cos(w * t + ph)), float(2.0 * B * w * std::cos(2.0 * w * t)));
        const Vector3 a(float(-A * w * w * std::sin(w * t)), float(-HA * w * w * std::sin(w * t + ph)),
                        float(-4.0 * B * w * w * std::sin(2.0 * w * t)));
        Vector3       T = a;
        T.y += kG;
        k.thrust = T.length() / kG;
        k.up     = T.clone().normalize();
        Vector3 f(k.v.x, 0.f, k.v.z);
        if (f.lengthSq() < 1e-6f) f.set(1.f, 0.f, 0.f);
        f.normalize();
        Vector3 x = f - k.up * f.dot(k.up);
        x.normalize();
        const Vector3 z = Vector3(x).cross(k.up);// x forward, y up, z right: z = x cross y
        k.basis.makeBasis(x, k.up, z);
        k.q.setFromRotationMatrix(k.basis);
        return k;
    }

    // The attitude a quad takes to fly a path with velocity v and acceleration a: its thrust is a + g, its nose along
    // the ground track.
    inline Pose poseFrom(const Vector3& p, const Vector3& v, const Vector3& a) {
        Pose k;
        k.p = p;
        k.v = v;
        Vector3 T = a;
        T.y += kG;
        k.thrust = T.length() / kG;
        k.up     = T.clone().normalize();
        Vector3 f(v.x, 0.f, v.z);
        if (f.lengthSq() < 1e-6f) f.set(1.f, 0.f, 0.f);
        f.normalize();
        Vector3 x = f - k.up * f.dot(k.up);
        x.normalize();
        const Vector3 z = Vector3(x).cross(k.up);
        k.basis.makeBasis(x, k.up, z);
        k.q.setFromRotationMatrix(k.basis);
        return k;
    }

    // A strafing line along x through the cores of two columns at x = +-0.9 (z = 0), a little off the line and bobbing so
    // no two passes retrace each other: the quad slices each column twice a period, from alternate directions, and
    // turns over at the ends. Velocity and acceleration come from differences of the position.
    inline Vector3 linePos(double t) {
        const double w = 3.0;
        return Vector3(float(1.5 * std::sin(w * t)), float(1.15 + 0.20 * std::sin(0.9 * w * t + 0.6)), float(0.12 * std::sin(1.37 * w * t)));
    }
    inline Pose flightLine(double t) {
        const double  h  = 1.0 / 1200.0;
        const Vector3 p0 = linePos(t - h), p1 = linePos(t), p2 = linePos(t + h);
        const Vector3 v  = (p2 - p0) * float(1.0 / (2.0 * h));
        const Vector3 a  = (p2 - p1 * 2.f + p0) * float(1.0 / (h * h));
        return poseFrom(p1, v, a);
    }

    // The motor `i` of a pose, in the world.
    inline Vector3 motorWorld(const Pose& k, int i) {
        const Vector3 off(kMotorSign[i][0] * kMotor, kMotorY, kMotorSign[i][1] * kMotor);
        return k.p + off.clone().applyQuaternion(k.q);
    }

    // A sunset, as a float equirect: a deep blue zenith, a warm horizon, the glow round a low sun and the sun as a
    // small bright disc the renderer's HDRI extraction finds and turns into the scene's one directional light.
    // `elevDeg` is the sun's height above the horizon; `energy` its irradiance (the sky's own is about 1).
    inline std::shared_ptr<DataTexture> makeSunset(float elevDeg, float energy, float glowScale = 1.f, float azDeg = 0.f) {
        constexpr int W = 2048, H = 1024;
        const float   el = elevDeg * math::PI / 180.f;
        const float   az = azDeg * math::PI / 180.f;
        const Vector3 sun(std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az));
        const float   sigma = 0.0075f;
        const float   amp   = energy / (math::PI * sigma * sigma);
        std::vector<float> px(size_t(W) * H * 4);
        for (int y = 0; y < H; ++y) {
            const float lat = (float(y) + 0.5f) / float(H) * math::PI - 0.5f * math::PI;// row 0 = the nadir
            for (int x = 0; x < W; ++x) {
                const float lon = (float(x) + 0.5f) / float(W) * 2.f * math::PI - math::PI;
                const Vector3 d(std::cos(lat) * std::cos(lon), std::sin(lat), std::cos(lat) * std::sin(lon));
                const float   cg = std::clamp(d.dot(sun), -1.f, 1.f);
                const float   g  = std::acos(cg);
                float r, gr, b;
                if (lat >= 0.f) {
                    // Away from the sun the horizon is a dusky mauve and the zenith a deep blue; near it the horizon burns.
                    const float t     = std::pow(1.f - lat / (0.5f * math::PI), 3.2f);// 1 at the horizon, 0 overhead
                    const float near  = std::exp(-g / 0.85f);
                    const float hr = 0.30f + near * 1.00f, hg = 0.26f + near * 0.34f, hb = 0.34f - near * 0.16f;
                    r  = 0.025f + t * hr;
                    gr = 0.055f + t * hg;
                    b  = 0.16f + t * hb;
                    const float glow = glowScale * (0.55f * std::exp(-g / 0.07f) + 0.22f * std::exp(-g / 0.40f));
                    r += glow * 1.25f;
                    gr += glow * 0.60f;
                    b += glow * 0.20f;
                    const float disc = amp * std::exp(-(g * g) / (sigma * sigma));
                    r += disc * 1.00f;
                    gr += disc * 0.80f;
                    b += disc * 0.55f;
                } else {
                    r = 0.020f; gr = 0.016f; b = 0.014f;// the ground, below the horizon
                }
                const size_t i = (size_t(y) * W + x) * 4;
                px[i] = r; px[i + 1] = gr; px[i + 2] = b; px[i + 3] = 1.f;
            }
        }
        auto env = DataTexture::create(ImageData(std::move(px)), W, H);
        env->format  = Format::RGBA;
        env->type    = Type::Float;
        env->mapping = Mapping::EquirectangularReflection;
        env->needsUpdate();
        return env;
    }

}// namespace smokedemo

#endif
