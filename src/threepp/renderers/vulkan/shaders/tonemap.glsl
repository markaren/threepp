// The display transform's curves: the linear -> sRGB encode and the tone-map
// operators behind threepp::ToneMapping. One copy for the two shaders that
// have to land in the swapchain's display domain:
//   post_composite.comp  the scene (exposure -> white balance -> toneMap ->
//                        linearToSRGB -> grade LUT); the authority
//   particle.frag        the lit particle path, which composites over that
//                        already-encoded frame (white balance and the grade
//                        LUT skipped)
// overlay_display.glsl holds an od-prefixed copy for the overlay passes:
// KEEP IN SYNC with it.

#ifndef THREEPP_TONEMAP_GLSL
#define THREEPP_TONEMAP_GLSL

vec3 linearToSRGB(vec3 x) {
    const vec3 cutoff = vec3(lessThan(x, vec3(0.0031308)));
    const vec3 lower  = 12.92 * x;
    const vec3 higher = 1.055 * pow(max(x, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(higher, lower, cutoff);
}
vec3 reinhard(vec3 c) { return clamp(c / (vec3(1.0) + c), 0.0, 1.0); }
vec3 cineon(vec3 c) {
    c = max(vec3(0.0), c - 0.004);
    return pow((c * (6.2 * c + 0.5)) / (c * (6.2 * c + 1.7) + 0.06), vec3(2.2));
}
vec3 acesFilmic(vec3 c) {
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
vec3 neutral(vec3 color) {
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
// AgX (three.js AgXToneMapping port — Sobotka's AgX via Missing Deadlines'
// matrices): log2 encode into the AgX inset gamut, 6th-order sigmoid
// contrast, outset. Compresses toward white far more gently than ACES and
// holds saturated emitters without hue skew.
vec3 agxDefaultContrastApprox(vec3 x) {
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
vec3 agx(vec3 color) {
    // GLSL mat3 constructors take COLUMNS.
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
    color = agxDefaultContrastApprox(color);
    color = AgXOutsetMatrix * color;
    // AgX's sigmoid lives in a 2.2-gamma encoding; linearize back for the
    // caller's linearToSRGB encode.
    color = pow(max(vec3(0.0), color), vec3(2.2));
    color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
    return clamp(color, 0.0, 1.0);
}
vec3 toneMap(vec3 c, uint mode, float exposure) {
    if (mode == 1u) return c * exposure;                    // Linear
    if (mode == 2u) return reinhard(c * exposure);          // Reinhard
    if (mode == 3u) return cineon(c * exposure);            // OptimizedCineon
    if (mode == 4u) return acesFilmic(c * (exposure / 0.6));// ACESFilmic
    if (mode == 6u) return neutral(c * exposure);           // Khronos PBR Neutral
    if (mode == 7u) return agx(c * exposure);               // AgX
    return c;// None / Custom — pass-through
}

#endif// THREEPP_TONEMAP_GLSL
