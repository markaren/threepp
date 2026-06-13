// env_dyn_compat.hpp -- the C++ side of the single-source env dialect.
//
// An ".envdyn" file holds ONE environment's dynamics written once in a small subset
// that is valid as BOTH C++ and GLSL, so the CPU learner/viz and the GPU rollout
// shader share the exact same source (no hand-duplication, no drift). This header
// supplies the GLSL builtins and the two qualifier macros that differ between the
// languages, so a .envdyn compiles unchanged as C++.
//
// Usage (C++): include this, then include the .envdyn inside a per-env namespace
// that pulls in envmath, e.g.
//     #include "envdyn/env_dyn_compat.hpp"
//     namespace rldemo { namespace pendulum_dyn {
//         using namespace rldemo::envmath;
//     #include "envdyn/pendulum.envdyn"
//     }}
// Each env gets its own namespace, so several .envdyn can coexist in one TU
// (the swarm includes both pendulum and acrobot to switch tasks with one typedef).
//
// Usage (GLSL): before the include, `#define ENVDYN_FN` (empty) and
//     #define ENVDYN_OUT_ARR(T, name, N) out T name[N]
// and enable `#extension GL_GOOGLE_include_directive : require`.
//
// Dialect rules a .envdyn must follow (the portable C++ <-> GLSL intersection):
//   - by-value struct returns; structs hold named floats / fixed float[N]
//   - `const int`/`const float` for dims+constants (NOT #define -> stays namespaced
//     in C++, so two envs never collide); f-suffixed float literals
//   - read-only array params spelled `const T x[N]`
//   - writable array params spelled ENVDYN_OUT_ARR(T, name, N)
//   - functions prefixed ENVDYN_FN; math via clamp/mod/sin/cos/... only
#pragma once
#include <cmath>

namespace rldemo {
    namespace envmath {
        inline float sin(float x) { return std::sin(x); }
        inline float cos(float x) { return std::cos(x); }
        inline float sqrt(float x) { return std::sqrt(x); }
        inline float exp(float x) { return std::exp(x); }
        inline float tanh(float x) { return std::tanh(x); }
        inline float clamp(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
        // GLSL mod(): a - b*floor(a/b)  (differs from fmodf for negative a)
        inline float mod(float a, float b) { return a - b * std::floor(a / b); }
    }// namespace envmath
}// namespace rldemo

#define ENVDYN_FN inline
// Writable array parameter: GLSL needs the `out` qualifier; in C++ the array decays
// to a (writable) pointer, so the qualifier is simply omitted.
#define ENVDYN_OUT_ARR(T, name, N) T name[N]
