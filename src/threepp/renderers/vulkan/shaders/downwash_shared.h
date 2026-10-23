// The rotor downwash's per-frame record (threepp::Downwash, set with
// VulkanRenderer::setDownwash): one host-visible block per frame in flight,
// reached by DEVICE ADDRESS from the G-buffer fragment stage (the address
// rides RasterCameraUbo.downwash), the grass wind compute and the particle
// emitter (their push block / aux record). No descriptor anywhere, so the
// pipelines' layouts are what they were before it existed, and a frame
// without a downwash pushes address 0 and runs the old arithmetic.
//
// Every member is a 4-byte scalar, so the C++ layout and the GLSL scalar
// block layout agree byte for byte (static_asserts below). The shader-side
// profile is downwash.glsl; its CPU twin is Downwash.cpp.

#ifndef THREEPP_DOWNWASH_SHARED_H
#define THREEPP_DOWNWASH_SHARED_H

#define kDownwashMaxSources 8
#define kDownwashKindJet    1u
#define kDownwashKindRings  2u

#ifdef __cplusplus

#include <cstdint>

namespace threepp::vulkan {

    struct DownwashHeaderGpu {
        uint32_t count;           // sources in use (enabled ones only, packed)
        float    time;            // renderer time, s (sim_time when pinned)
        float    rippleAmp;       // m, the jet's rings at share 1
        float    ringLam0;        // m, the jet's shortest ring
        float    ringLamGrowth;   // m of wavelength per m of radius
        float    ringHz;          // wavelengths a second the jet's rings travel
        float    roughBoost;      // roughness added in the struck disc
        float    leanMax;         // rad
        float    pressMax;        // share of a blade's height
        float    flutter;         // share
        float    hollow;          // m, the particle hollow at share 1
        float    rimLift;         // share of the hollow
    };

    struct DownwashSourceGpu {
        float    center[3];       // jet: rotor plane centre; rings: centre (world)
        uint32_t kind;            // kDownwashKind*
        float    prevCenter[3];   // last frame's centre (the particles' motion vectors)
        float    share;           // jet: v_i / v_ref; rings: amplitude, m
        float    radius;          // jet: disc radius, m; rings: start radius, m
        float    surfaceY;        // jet: struck surface for particles; -FLT_MAX = their own y
        float    lobes;           // jet: swell lobes (0 = round)
        float    lobePhase;       // jet: rad
        float    t0;              // rings: time the first ring leaves
        float    speed;           // rings: m/s
        float    interval;        // rings: s between rings
        float    count;           // rings: how many
        float    wavelength;      // rings: m
        float    _pad0;
        float    _pad1;
        float    _pad2;
    };

    struct DownwashGpu {
        DownwashHeaderGpu header;
        DownwashSourceGpu sources[kDownwashMaxSources];
    };

    static_assert(sizeof(DownwashHeaderGpu) == 48, "DownwashHeaderGpu drifted from downwash_shared.h's GLSL mirror");
    static_assert(sizeof(DownwashSourceGpu) == 80, "DownwashSourceGpu drifted from downwash_shared.h's GLSL mirror");
    static_assert(sizeof(DownwashGpu) == 48 + 80 * kDownwashMaxSources, "DownwashGpu layout");
}// namespace threepp::vulkan

#else// GLSL

struct DownwashHeaderGpu {
    uint  count;
    float time;
    float rippleAmp;
    float ringLam0;
    float ringLamGrowth;
    float ringHz;
    float roughBoost;
    float leanMax;
    float pressMax;
    float flutter;
    float hollow;
    float rimLift;
};

struct DownwashSourceGpu {
    vec3  center;
    uint  kind;
    vec3  prevCenter;
    float share;
    float radius;
    float surfaceY;
    float lobes;
    float lobePhase;
    float t0;
    float speed;
    float interval;
    float count;
    float wavelength;
    float _pad0;
    float _pad1;
    float _pad2;
};

#endif// __cplusplus

#endif// THREEPP_DOWNWASH_SHARED_H
