// Shared layout for the path-traced LIDAR scanner. Included by both the C++
// host (LidarScanner.cpp) and the GLSL shaders (lidar.rgen / lidar.rchit /
// lidar.rmiss) so the LidarBeam / LidarResult / LidarPushConstants structs
// have a single source of truth. The static_asserts in the C++ branch catch
// any silent layout drift between the two languages.
//
// All structs are designed for scalar block layout (GL_EXT_scalar_block_layout
// in the shader) so the C-style `float[3]` mirror in C++ matches the GLSL
// `vec3` layout exactly. The explicit pad fields keep stride alignment at
// 16 bytes for the per-beam SSBOs — not strictly required under scalar
// layout, but lets the GPU coalesce loads across adjacent beams.

#ifndef THREEPP_VULKAN_LIDAR_SHARED_H
#define THREEPP_VULKAN_LIDAR_SHARED_H

#ifdef __cplusplus

#include <cstdint>

namespace threepp::vulkan_lidar {

    // One beam = (origin, direction). Direction must be unit length; the
    // shader does not re-normalise.
    struct LidarBeam {
        float origin[3];
        float _pad0;
        float direction[3];
        float _pad1;
    };
    static_assert(sizeof(LidarBeam) == 32,
                  "LidarBeam layout drifted — update the GLSL mirror below.");

    // One per-beam result. instanceId = -1 indicates miss / dropped (below
    // detector threshold). returnNo currently always 1 (single return); the
    // field is reserved for future multi-return through transmissive layers.
    struct LidarResult {
        float position[3];
        float distance;
        float normal[3];
        float intensity;
        int32_t instanceId;
        int32_t returnNo;
        float _pad[2];
    };
    static_assert(sizeof(LidarResult) == 48,
                  "LidarResult layout drifted — update the GLSL mirror below.");

    // 40-byte push constant block. Fits well within the 128-byte minimum
    // pushConstants size that every Vulkan implementation guarantees.
    struct LidarPushConstants {
        uint32_t numBeams;
        float maxRange;
        float laserPower;
        float invReferenceIntensity;
        float atmosphericExtinction;
        float detectorThreshold;
        uint32_t rngSeed;
        // Maximum returns the rgen will emit per beam. Beam continues
        // through transmissive surfaces and records subsequent hits
        // until either: maxReturns reached, ray misses, or remaining
        // throughput falls below detectorThreshold. 1 = legacy single-
        // return behaviour.
        uint32_t maxReturns;
        // Stochastic samples fired per logical beam. >1 jitters direction
        // within `beamDivergenceTan`-half-angle cone and runs an
        // independent multi-return + delta-tracking chain per sample.
        uint32_t samplesPerBeam;
        // tan(beamDivergenceMrad · 0.001 · 0.5) precomputed on the host —
        // the cone half-angle as a tangent so the GLSL jitter is one
        // multiply + sqrt.
        float beamDivergenceTan;
    };
    static_assert(sizeof(LidarPushConstants) == 40,
                  "LidarPushConstants layout drifted — update the GLSL mirror below.");

}// namespace threepp::vulkan_lidar

#else  // GLSL

struct LidarBeam {
    vec3  origin;
    float _pad0;
    vec3  direction;
    float _pad1;
};

struct LidarResult {
    vec3  position;
    float distance;
    vec3  normal;
    float intensity;
    int   instanceId;
    int   returnNo;
    vec2  _pad;
};

#endif  // __cplusplus

#endif  // THREEPP_VULKAN_LIDAR_SHARED_H
