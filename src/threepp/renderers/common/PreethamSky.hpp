// The Preetham daylight model as threepp::Sky carries it (three.js's Sky:
// Simon Wallner's implementation, Martin Upitis's HDR improvements), evaluated
// on the CPU.
//
// The GL renderer shades the Sky's dome per pixel with the GLSL in Sky.cpp.
// The deferred (Vulkan) renderer keeps one environment image for the sky it
// shows and the light it lights with, so it bakes the same model into that
// image (vulkan/SkyBake, sky_preetham.comp). This header is the model's one
// CPU form: the Sky's six uniforms as a plain record, the radiance along a
// direction (the reference the bake is held to, texel for texel), and the
// model's sun as EnvSunExtract reports an HDRI's, so the one-sun policy
// treats a baked sky and a photographed one alike.

#ifndef THREEPP_COMMON_PREETHAMSKY_HPP
#define THREEPP_COMMON_PREETHAMSKY_HPP

#include "threepp/renderers/common/EnvSunExtract.hpp"

namespace threepp {

    class Sky;

    struct PreethamSkyParams {
        float sunDir[3] = {0.f, 1.f, 0.f};// unit, TOWARD the sun: normalize(sunPosition)
        float sunPosY = 1.f;              // sunPosition.y as given (the model's sunfade reads it raw)
        float up[3] = {0.f, 1.f, 0.f};
        float turbidity = 2.f;
        float rayleigh = 1.f;
        float mieCoefficient = 0.005f;
        float mieDirectionalG = 0.8f;

        bool operator==(const PreethamSkyParams& o) const;
        bool operator!=(const PreethamSkyParams& o) const { return !(*this == o); }
    };

    // The Sky's uniforms, read off its ShaderMaterial. A uniform that is
    // missing or of another type keeps the default above.
    PreethamSkyParams preethamParamsFromSky(const Sky& sky);

    // Linear radiance of the sky along `dir` (unit, world space): the GL
    // fragment shader's retColor before the renderer's tone mapping.
    // withDisc adds the solar disc.
    void preethamRadiance(const PreethamSkyParams& p, const float dir[3], bool withDisc, float outRgb[3]);

    // The model's sun: the disc's irradiance (its radiance over its solid
    // angle) and direction. A directional light of colorE replaces the disc.
    EnvSunExtract preethamSun(const PreethamSkyParams& p);

}// namespace threepp

#endif//THREEPP_COMMON_PREETHAMSKY_HPP
