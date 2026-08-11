#ifndef OVERLAY_DISPLAY_GLSL
#define OVERLAY_DISPLAY_GLSL

// The display transform an OVERLAY-pass shader has to apply by hand.
//
// Everything drawn after recordUpscaleAndPost composites onto a swapchain that
// post_composite.comp already exposed, tone-mapped and sRGB-ENCODED. A pass
// that arrives afterwards with linear HDR radiance in its hands — a lit
// billboard, an emissive spark — must therefore run the same curve on its own
// values before blending, or it lands in a different value domain and reads
// too bright and too saturated against the frame behind it.
//
// KEEP IN SYNC with post_composite.comp, which is the authority, and with the
// inline copy in particle.frag. That third copy is deliberate for now: the
// legacy ParticleSystem path is required by plans/particle-field.md to stay
// untouched by the ParticleField work, and folding it onto this header would
// recompile it. Fold them together the next time that path is opened for its
// own reasons.
//
// White balance and the grading LUT are intentionally omitted — the accepted
// approximation for small translucent/additive overlay content, same as
// particle.frag's lit path.

vec3 odLinearToSRGB(vec3 x) {
    const vec3 cutoff = vec3(lessThan(x, vec3(0.0031308)));
    const vec3 lower  = 12.92 * x;
    const vec3 higher = 1.055 * pow(max(x, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(higher, lower, cutoff);
}

vec3 odReinhard(vec3 c) { return clamp(c / (vec3(1.0) + c), 0.0, 1.0); }

vec3 odCineon(vec3 c) {
    c = max(vec3(0.0), c - 0.004);
    return pow((c * (6.2 * c + 0.5)) / (c * (6.2 * c + 1.7) + 0.06), vec3(2.2));
}

vec3 odAcesFilmic(vec3 c) {
    const mat3 inMat = mat3(
            vec3(0.59719, 0.07600, 0.02840),
            vec3(0.35458, 0.90834, 0.13383),
            vec3(0.04823, 0.01566, 0.83777));
    const mat3 outMat = mat3(
            vec3( 1.60475, -0.10208, -0.00327),
            vec3(-0.53108,  1.10813, -0.07276),
            vec3(-0.07367, -0.00605,  1.07602));
    c = inMat * c;
    const vec3 a = c * (c + 0.0245786) - 0.000090537;
    const vec3 b = c * (0.983729 * c + 0.4329510) + 0.238081;
    return clamp(outMat * (a / b), 0.0, 1.0);
}

vec3 odNeutral(vec3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;
    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;
    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;
    float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);
    color *= newPeak / peak;
    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return mix(color, vec3(newPeak), g);
}

vec3 odAgxContrast(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return + 15.5     * x4 * x2
           - 40.14    * x4 * x
           + 31.96    * x4
           - 6.868    * x2 * x
           + 0.4298   * x2
           + 0.1191   * x
           - 0.00232;
}

vec3 odAgx(vec3 color) {
    const mat3 AgXInsetMatrix = mat3(
            vec3(0.856627153315983, 0.137318972929847, 0.11189821299995),
            vec3(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
            vec3(0.0482516061458583, 0.101439036467562, 0.811302368396859));
    const mat3 AgXOutsetMatrix = mat3(
            vec3(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
            vec3(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
            vec3(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405));
    const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
            vec3(0.6274, 0.0691, 0.0164),
            vec3(0.3293, 0.9195, 0.0880),
            vec3(0.0433, 0.0113, 0.8956));
    const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
            vec3(1.6605, -0.1246, -0.0182),
            vec3(-0.5876, 1.1329, -0.1006),
            vec3(-0.0728, -0.0083, 1.1187));
    const float AgxMinEv = -12.47393;
    const float AgxMaxEv = 4.026069;

    color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
    color = AgXInsetMatrix * color;
    color = max(color, vec3(1e-10));
    color = log2(color);
    color = (color - AgxMinEv) / (AgxMaxEv - AgxMinEv);
    color = clamp(color, 0.0, 1.0);
    color = odAgxContrast(color);
    color = AgXOutsetMatrix * color;
    color = pow(max(vec3(0.0), color), vec3(2.2));
    color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
    return clamp(color, 0.0, 1.0);
}

vec3 odToneMap(vec3 c, uint mode, float exposure) {
    if (mode == 1u) return c * exposure;                      // Linear
    if (mode == 2u) return odReinhard(c * exposure);          // Reinhard
    if (mode == 3u) return odCineon(c * exposure);            // OptimizedCineon
    if (mode == 4u) return odAcesFilmic(c * (exposure / 0.6));// ACESFilmic
    if (mode == 6u) return odNeutral(c * exposure);           // Khronos PBR Neutral
    if (mode == 7u) return odAgx(c * exposure);               // AgX
    return c;// None / Custom — pass-through
}

// Linear HDR radiance -> the swapchain's display-referred, sRGB-encoded domain.
vec3 odDisplay(vec3 hdr, uint mode, float exposure) {
    return odLinearToSRGB(odToneMap(hdr, mode, exposure));
}

#endif// OVERLAY_DISPLAY_GLSL
