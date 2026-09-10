// Backend-neutral HDRI sun extraction.
//
// An equirect HDR sky carries its sun as a handful of texels at ~10^4x the
// mean sky radiance. No Monte-Carlo GGX prefilter integrates that smoothly, so
// both backends prefilter their glossy/rough env levels from a SUN-CLAMPED copy
// (the disc replaced by its own surround) and re-inject the removed energy as
// an analytic directional light. Level 0 keeps the original pixels either way:
// the sky background and true mirrors still show the real disc.
//
// This lived inside vulkan/EnvPrefilter.cpp. The GL raster path needs exactly
// the same detector and the same numbers (a different detector would light the
// two backends differently, which is the colour-temperature mismatch this
// helper exists to remove), so it moved here unchanged.

#ifndef THREEPP_COMMON_ENVSUNEXTRACT_HPP
#define THREEPP_COMMON_ENVSUNEXTRACT_HPP

#include <cstdint>
#include <vector>

namespace threepp {

    // What the detector measured about the environment's sun disc.
    struct EnvSunExtract {
        bool found = false;
        float dir[3] = {0.f, 1.f, 0.f};   // unit world dir TOWARD the sun (env mapping)
        float colorE[3] = {0.f, 0.f, 0.f};// disc irradiance E = Σ L·dΩ (linear RGB) — a
                                          // directional light of this color replaces the
                                          // disc's energy exactly
        float angularRadiusDeg = 0.f;     // from the disc solid angle (Ω ≈ π·r²)
    };

    // Detect the dominant compact bright source in an equirect float RGBA image
    // and build the sun-clamped prefilter source.
    //
    // `rgba` is w*h*4 floats, row 0 = south pole (v = 0), with
    // u = 0.5 + atan2(z, x)/2π and v = 0.5 + asin(y)/π — the mapping shared by
    // GLPMREM.cpp's dirFromUv/equirectUv, ShaderChunk/common.glsl's equirectUv
    // and prefilter_env.comp. A different mapping here would aim the re-injected
    // analytic sun away from the disc it replaced.
    //
    // Returns true when a sun was found: `clamped` is then filled with w*h*4
    // floats (the disc replaced by its surround, plus a soft circumsolar cap)
    // and `out` describes the removed energy. Returns false otherwise, leaving
    // `clamped` empty and `out` default-constructed.
    bool extractEnvSun(uint32_t w, uint32_t h, const float* rgba,
                       std::vector<float>& clamped, EnvSunExtract& out);

}// namespace threepp

#endif//THREEPP_COMMON_ENVSUNEXTRACT_HPP
