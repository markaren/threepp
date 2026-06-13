// AcrobotEnv -- the classic 2-link underactuated swing-up (torque on the elbow only).
// Standard Sutton & Barto / Gym acrobot dynamics (RK4), as a plain-C++ Env. This is
// the "switch out the pendulum" proof: a genuinely different task (obs 6 vs 3, two
// links, harder) drops into the SAME swarm + learner by changing one typedef.
//
// Env concept (what swarm.cpp / the trainer expect of an Env struct):
//   static constexpr int kObsDim, kActDim, kEpisodeSteps, kRods;
//   static constexpr float kReach;  static constexpr const char* kName;
//   struct State;
//   template<RNG> static State sampleInitial(RNG&);
//   static void  observe(const State&, float* obs);     // obs[kObsDim]
//   static State step(const State&, const float* act);  // act[kActDim], in [-1,1]
//   static float reward(const State&, const float* act);
//   static void  rod(const State&, int r, float& x0,&y0,&x1,&y1); // link endpoints, local XY
//   static bool  upright(const State&);                 // "solved" test for the learning curve

#ifndef THREEPP_ENV_ACROBOT_HPP
#define THREEPP_ENV_ACROBOT_HPP

#include <cmath>
#include <random>

namespace rldemo {

    struct AcrobotEnv {
        static constexpr int kObsDim = 6;
        static constexpr int kActDim = 1;
        static constexpr int kEpisodeSteps = 200;
        static constexpr int kRods = 2;
        static constexpr float kReach = 2.0f;
        static constexpr const char* kName = "Acrobot";

        // physical constants (Gym acrobot)
        static constexpr float kL1 = 1.f, kL2 = 1.f, kLc1 = 0.5f, kLc2 = 0.5f;
        static constexpr float kM1 = 1.f, kM2 = 1.f, kI1 = 1.f, kI2 = 1.f;
        static constexpr float kG = 9.8f, kDt = 0.2f;
        static constexpr float kMaxV1 = 4.f * 3.14159265f, kMaxV2 = 9.f * 3.14159265f;

        struct State {
            float t1 = 0.f, t2 = 0.f, d1 = 0.f, d2 = 0.f;// theta1, theta2, dtheta1, dtheta2
        };

        template<class RNG>
        static State sampleInitial(RNG& rng) {
            std::uniform_real_distribution<float> u(-0.1f, 0.1f);
            return State{u(rng), u(rng), u(rng), u(rng)};
        }

        static void observe(const State& s, float* o) {
            o[0] = std::cos(s.t1);
            o[1] = std::sin(s.t1);
            o[2] = std::cos(s.t2);
            o[3] = std::sin(s.t2);
            o[4] = s.d1;
            o[5] = s.d2;
        }

        // Shaped reward: height of the end-effector (maximize -> swing up). At rest
        // (hanging) = -2; fully up = +2; Gym's goal line is tip height > 1.
        static float tipHeight(const State& s) {
            return -kL1 * std::cos(s.t1) - kL2 * std::cos(s.t1 + s.t2);
        }
        static float reward(const State& s, const float*) { return tipHeight(s); }
        static bool upright(const State& s) { return tipHeight(s) > 1.0f; }

        // equations of motion: returns the 4 state derivatives for torque `tau`
        static void dsdt(const State& s, float tau, float& dt1, float& dt2, float& ddt1, float& ddt2) {
            const float c2 = std::cos(s.t2), s2 = std::sin(s.t2);
            const float d1 = kM1 * kLc1 * kLc1 + kM2 * (kL1 * kL1 + kLc2 * kLc2 + 2 * kL1 * kLc2 * c2) + kI1 + kI2;
            const float d2 = kM2 * (kLc2 * kLc2 + kL1 * kLc2 * c2) + kI2;
            const float phi2 = kM2 * kLc2 * kG * std::cos(s.t1 + s.t2 - 1.57079633f);
            const float phi1 = -kM2 * kL1 * kLc2 * s.d2 * s.d2 * s2
                             - 2 * kM2 * kL1 * kLc2 * s.d2 * s.d1 * s2
                             + (kM1 * kLc1 + kM2 * kL1) * kG * std::cos(s.t1 - 1.57079633f) + phi2;
            ddt2 = (tau + d2 / d1 * phi1 - kM2 * kL1 * kLc2 * s.d1 * s.d1 * s2 - phi2)
                 / (kM2 * kLc2 * kLc2 + kI2 - d2 * d2 / d1);
            ddt1 = -(d2 * ddt2 + phi1) / d1;
            dt1 = s.d1;
            dt2 = s.d2;
        }

        static float wrap(float x) {
            const float twoPi = 6.2831853f;
            float m = std::fmod(x + 3.14159265f, twoPi);
            if (m < 0) m += twoPi;
            return m - 3.14159265f;
        }
        static float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

        static State step(const State& s, const float* act) {
            const float tau = clampf(act[0], -1.f, 1.f);// elbow torque in [-1,1]
            // RK4 over kDt
            auto add = [](const State& a, const State& k, float h) {
                return State{a.t1 + h * k.t1, a.t2 + h * k.t2, a.d1 + h * k.d1, a.d2 + h * k.d2};
            };
            auto deriv = [&](const State& st) {
                State k;
                dsdt(st, tau, k.t1, k.t2, k.d1, k.d2);
                return k;
            };
            State k1 = deriv(s);
            State k2 = deriv(add(s, k1, kDt / 2));
            State k3 = deriv(add(s, k2, kDt / 2));
            State k4 = deriv(add(s, k3, kDt));
            State n;
            n.t1 = s.t1 + kDt / 6 * (k1.t1 + 2 * k2.t1 + 2 * k3.t1 + k4.t1);
            n.t2 = s.t2 + kDt / 6 * (k1.t2 + 2 * k2.t2 + 2 * k3.t2 + k4.t2);
            n.d1 = s.d1 + kDt / 6 * (k1.d1 + 2 * k2.d1 + 2 * k3.d1 + k4.d1);
            n.d2 = s.d2 + kDt / 6 * (k1.d2 + 2 * k2.d2 + 2 * k3.d2 + k4.d2);
            n.t1 = wrap(n.t1);
            n.t2 = wrap(n.t2);
            n.d1 = clampf(n.d1, -kMaxV1, kMaxV1);
            n.d2 = clampf(n.d2, -kMaxV2, kMaxV2);
            return n;
        }

        // Link endpoints in local XY (pivot at origin). t1=0 -> link 1 hangs down.
        static void rod(const State& s, int r, float& x0, float& y0, float& x1, float& y1) {
            const float jx = kL1 * std::sin(s.t1), jy = -kL1 * std::cos(s.t1);// elbow joint
            if (r == 0) {
                x0 = 0.f;
                y0 = 0.f;
                x1 = jx;
                y1 = jy;
            } else {
                x0 = jx;
                y0 = jy;
                x1 = jx + kL2 * std::sin(s.t1 + s.t2);
                y1 = jy - kL2 * std::cos(s.t1 + s.t2);
            }
        }
    };

}// namespace rldemo

#endif//THREEPP_ENV_ACROBOT_HPP
