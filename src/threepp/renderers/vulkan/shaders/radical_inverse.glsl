// The base-2 radical inverse the deferred shade's GI gather
// (deferred_shade_30_gi_gather.glsl) and the RTAO pass (rtao.comp) build
// their per-pixel Hammersley directions from.

#ifndef RADICAL_INVERSE_GLSL
#define RADICAL_INVERSE_GLSL

// Van der Corput radical inverse (base 2) — the second dimension of the
// per-pixel Hammersley point set.
float radicalInverse2(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;// × 1/2^32
}

#endif// RADICAL_INVERSE_GLSL
