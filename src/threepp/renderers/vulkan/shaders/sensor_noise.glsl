// sensor_noise.glsl — an image-sensor noise model, applied as the very last
// thing before the swapchain.
//
// WHY IT LIVES HERE. Rendered frames are clean in a way no real sensor's are,
// and that gap is a known sim-to-real problem for anything trained on
// synthetic images. Modelling ISO as pure exposure gain (which is all the
// physical camera did before this) is only half the story: on a real camera,
// pushing ISO also pushes noise, and the whole reason to shoot at ISO 100 is
// that it does not.
//
// WHY IT RUNS POST-RESOLVE. TAA (and DLSS/FSR) average successive frames.
// Noise injected before the resolve is therefore filtered back out over a
// handful of frames, and the sim ends up reporting a noise model it is not
// actually producing. This is the last stage before the swapchain, so what is
// added here survives.
//
// THE MODEL. Working backwards from the displayed pixel to the photosite:
//
//   linear   = sRGB^-1(pixel)                       display-referred -> linear
//   e        = linear · fullWell / isoGain          electrons collected
//   e       *= 1 + prnu·N_fixed(px)                 photo-response non-uniformity
//   e       += sqrt(e)·N()                          shot (Poisson) noise
//   e       += sqrt(dark)·N()                       dark-current shot noise
//   e       += readNoise·N()                        read noise
//   pixel'   = sRGB(e · isoGain / fullWell)
//
// The isoGain division is what makes ISO behave: at ISO 1600 a full-scale
// pixel represents 1/16 as many electrons as at ISO 100, so sqrt(e)/e — the
// relative shot noise — is 4× larger. That falls out of the physics rather
// than being dialled in.
//
// Dark current contributes its SHOT NOISE but not its pedestal: real cameras
// subtract the black level, so the bias is calibrated away and the noise it
// carries is not. Adding the pedestal here would just fog the blacks.
//
// PRNU is drawn from the pixel position ALONE — no frame seed. It is a
// fixed-pattern defect, frozen into the silicon; if it shimmered frame to
// frame it would be indistinguishable from read noise and would average away
// in exactly the situations (long stacks, temporal filters) where real
// fixed-pattern noise stubbornly does not.
//
// Shot noise uses a Gaussian approximation to the Poisson distribution. That
// is excellent above ~10 electrons and progressively coarser below, where it
// also has to be clamped against going negative — deep-shadow statistics are
// approximate here, and honestly so.

#ifndef SENSOR_NOISE_GLSL
#define SENSOR_NOISE_GLSL

// PCG hash — cheap, and decorrelates neighbouring pixel indices well enough
// that the fixed pattern looks like a defect map rather than a grid.
uint sensorHash(uint v) {
    v = v * 747796405u + 2891336453u;
    uint w = ((v >> ((v >> 28) + 4u)) ^ v) * 277803737u;
    return (w >> 22) ^ w;
}

uint sensorHash3(uvec2 px, uint seed) {
    return sensorHash(px.x ^ sensorHash(px.y ^ sensorHash(seed)));
}

float sensorUnit(uint h) {
    // (0,1) — never exactly 0, so log() in Box-Muller stays finite.
    return (float(h & 0x00FFFFFFu) + 0.5) * (1.0 / 16777216.0);
}

// Two independent standard normals via Box-Muller.
vec2 sensorGauss2(uint h0, uint h1) {
    float u1 = sensorUnit(h0);
    float u2 = sensorUnit(h1);
    float r  = sqrt(-2.0 * log(u1));
    float th = 6.28318530718 * u2;
    return vec2(r * cos(th), r * sin(th));
}

float srgbToLinear1(float c) {
    return c <= 0.04045 ? c * (1.0 / 12.92) : pow((c + 0.055) * (1.0 / 1.055), 2.4);
}
float linearToSrgb1(float c) {
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

// `color` is display-referred sRGB-encoded in [0,1]; the result is too.
vec3 applySensorNoise(vec3 color, uvec2 px, uint frameSeed,
                      float fullWell, float readNoise, float darkElectrons,
                      float prnu, float isoGain) {
    fullWell = max(fullWell, 1.0);
    isoGain  = max(isoGain, 1e-3);

    // Fixed-pattern gain: frame-independent by construction (see header).
    float prnuGain = 1.0;
    if (prnu > 0.0) {
        vec2 gf = sensorGauss2(sensorHash3(px, 0x9E3779B9u), sensorHash3(px, 0x85EBCA6Bu));
        prnuGain = max(0.0, 1.0 + prnu * gf.x);
    }

    // Two frame-varying normals per channel, drawn from disjoint streams so
    // the channels do not share a noise pattern (that would read as luma-only
    // grain rather than sensor noise).
    vec3 outColor;
    for (int ch = 0; ch < 3; ++ch) {
        float lin = srgbToLinear1(clamp(color[ch], 0.0, 1.0));

        float e = lin * fullWell / isoGain;
        e *= prnuGain;

        uint  s  = frameSeed * 3u + uint(ch);
        vec2  g0 = sensorGauss2(sensorHash3(px, s), sensorHash3(px, s ^ 0xC2B2AE35u));
        vec2  g1 = sensorGauss2(sensorHash3(px, s ^ 0x27D4EB2Fu), sensorHash3(px, s ^ 0x165667B1u));

        e += sqrt(max(e, 0.0)) * g0.x;                    // shot
        e += sqrt(max(darkElectrons, 0.0)) * g0.y;        // dark-current shot
        e += readNoise * g1.x;                            // read
        e = max(e, 0.0);

        outColor[ch] = linearToSrgb1(clamp(e * isoGain / fullWell, 0.0, 1.0));
    }
    return outColor;
}

#endif// SENSOR_NOISE_GLSL
