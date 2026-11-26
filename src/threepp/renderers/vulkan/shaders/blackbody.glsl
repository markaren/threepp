// A fire's colour: the blackbody radiance of a flame, and where a solved
// volume's soot stops glowing. Its own header because two places evaluate a
// fire's emission and must not drift apart: particle_density.glsl
// (pdEmissionAt, the emission every march integrates) and smoke_glow.comp (the
// same emission summed into the light a burning grid smoke throws on its
// surroundings). Declares no binding, so any shader can include it.

#ifndef THREEPP_BLACKBODY_GLSL
#define THREEPP_BLACKBODY_GLSL

// ── Analytic blackbody radiance, linear Rec.709, ~10 ALU, no LUT asset ───────
// Derived rather than tabulated (reuse-first: a 1D LUT here would be a new
// image, a new binding and a new upload for a curve that fits in two mads).
// Planck's law integrated against the CIE 1931 observer and converted to linear
// sRGB has, over 700-6500 K, a RED channel that is always the maximum — so
// normalising by it leaves only two curves to fit, and both are near-linear in
// kilokelvin:
//
//   T (K)      exact (r,g,b), max-normalised      this fit
//    1000        1.000 0.029 0.000                1.000 0.036 0.000
//    1900        1.000 0.243 0.000                1.000 0.237 0.000
//    2600        1.000 0.402 0.083                1.000 0.394 0.094
//    3000        1.000 0.484 0.155                1.000 0.484 0.171
//
// Worst component error is 0.017 over 700-3000 K, which is the whole range a
// sooty flame occupies. Above ~4000 K the green fit saturates early and the
// result reads slightly yellow rather than white (error 0.15 at 5500 K); that
// is out of scope on purpose — this is a fire ramp, not a white-point tool.
//
// MAGNITUDE is Stefan-Boltzmann, (T / 2000 K)^4: a grey body's radiance goes as
// T^4, which is what makes a flame's base blindingly brighter than its tips
// with no second authored curve. The 2000 K normalisation gives
// DensityRepr::emissiveIntensity the plain meaning "radiance of a 2000 K
// flame". T is clamped before the power so a nonsense authored temperature can
// neither produce a negative colour nor overflow to Inf.
vec3 blackbodyChroma(float T) {// the fit above: (1, g, b), red-normalised
    const float s = clamp(T, 500.0, 12000.0) * 0.001;// kilokelvin
    const float g = clamp(0.22394 * s - 0.18818, 0.0, 1.0);
    const float u = max(s - 2.09, 0.0);
    const float b = clamp(u * (0.17759 + 0.01179 * u), 0.0, 1.0);
    return vec3(1.0, g, b);
}

vec3 blackbodyRGB(float T) {
    const float h  = clamp(T, 500.0, 12000.0) * 0.0005;// T / 2000 K
    const float h2 = h * h;
    return blackbodyChroma(T) * (h2 * h2);
}

// ── THE VISIBLE GLOW OF SIMULATED HEAT: PLANCK AT THE RED PRIMARY ───────────
// T^4 is a hot body's TOTAL radiance, at a flame's temperatures almost all of it
// infrared. What an eye or a camera sees is the visible part, and that rises far
// more steeply: below ~3000 K Planck's law at a wavelength lambda goes as
// exp(-c2 / (lambda T)) (Wien; c2 = 14388 um K), at the red primary (610 nm)
// exp(-23587 K / T), locally about T^16 at 1500 K. So in a real fire the hottest
// gas carries both what is seen and the light thrown: a 1300 K tip is about a
// hundredth of a 1700 K core in the red, not the third T^4 gives, and firelight
// is the core's orange, not the red of the many cooler cells around it.
// A solved volume's temperatures are simulated, so it glows by this curve: the
// colour of the fit above, the magnitude Planck's red radiance normalised to
// 2000 K, which keeps DensityRepr::emissiveIntensity "the radiance of a 2000 K
// flame". Above 2000 K it is brighter than T^4 says (x51 at 3000 K, where T^4
// gives x5): that is the physics too. Particle fields keep blackbodyRGB: their
// authored ramp was tuned against T^4 and reaches down to 800 K tips that this
// curve would put out.
//
// The curve stops at kFlameMaxK. A solver's temperature is a model: a source
// that heats faster than its gas cools settles where heating and cooling
// balance, which can be thousands of kelvin, and this curve would make such a
// cell thousands of times brighter than a flame, and white: a few overheated
// cells by a burner then swamp the whole fire's glow and the light it throws.
// Real flames stay below ~2500 K, so the glow never goes past 3000 K's (x51 of
// a 2000 K flame).
const float kFlameRedK = 23587.0;// c2 / 610 nm
const float kFlameMaxK = 3000.0;
vec3 flameRGB(float T) {
    const float Tc = clamp(T, 500.0, kFlameMaxK);
    return blackbodyChroma(Tc) * exp(kFlameRedK * (0.0005 - 1.0 / Tc));
}

// ── THE GLOW OF A SOLVED VOLUME'S SOOT, AND WHERE IT STOPS ──────────────────
// blackbodyRGB's magnitude is Stefan-Boltzmann, T^4: the TOTAL radiance, almost
// all of it infrared below ~1000 K. The authored ramp wants exactly that, since
// it keeps a flame's tips (800-1250 K in every field in this tree) visible
// beside its core. A solver's soot does not stop at the tips: it goes on
// cooling, through 800 and 600 K to the ambient, and T^4 (which blackbodyRGB
// holds at its 500 K floor) still lit it at a sixteenth of a 1000 K tip or
// more all the way down, so a whole cooled smoke column glowed a dull red.
// Real soot stops glowing VISIBLY near the Draper point (~800 K, where solids
// first glow dull red in the dark): below it the visible radiance falls by
// orders of magnitude for every hundred kelvin. So a solved volume's glow fades
// out over the band around that point, and soot that has cooled below it is
// dark. Now that the glow is flameRGB's, whose red at 1000 K is already 1e-5 of
// a 2000 K flame's, the fade changes nothing visible; it stays because it is
// the line below which smoke_glow.comp skips a cell after one 2-byte read.
const float kPdGlowDarkK = 700.0;
const float kPdGlowFullK = 1000.0;

#endif// THREEPP_BLACKBODY_GLSL
