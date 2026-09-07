// Accessors for the per-instance flag word (gbuffer IDs .z — bit layout in
// vulkan_shared.h). Consumers ask for MEANING, not bit positions, so a new
// source of an existing meaning (a new flag, a material hint) touches only
// this file.
//
// TEMPORAL REACTIVITY — the consolidated "how little should temporal
// accumulation trust history on this surface" signal. Two channels, because
// shading and irradiance decorrelate from motion vectors differently:
//
//   ifShadingReactivity — the shaded APPEARANCE changes every frame in a way
//     motion vectors cannot express: a persistent deformer's normals /
//     stretch / contact shadowing swing in place (near-zero screen motion,
//     full shading change), and animated textures move their pattern with
//     ZERO geometric motion. The TAA resolve floors its blend α toward the
//     current frame on this signal. Scoping the deformer term to persistent
//     deformers rather than all skinned meshes is deliberate: the old
//     is_skinned force-high-α hack paid every walking character's temporal
//     AA for a per-frame-shading problem skinned meshes don't inherently
//     have.
//
//   ifIrradianceReactivity — the demodulated GI INTEGRAND itself is
//     non-stationary: a deformer re-orients the gather hemisphere every
//     frame, so a long history displays a lagged average. The GI accumulator
//     pins a short CONSTANT history cap on this signal. Animated textures
//     deliberately contribute 0 here — GI accumulates DEMODULATED
//     irradiance, which a scrolling albedo does not change.
//
// Both channels are binary today (flag-driven) but typed float [0..1] and
// consumed via mix()/scale, so graded sources (material reactivity hints,
// shading-change detection) plug in here without touching consumers.

#ifndef THREEPP_INSTANCE_FLAGS_GLSL
#define THREEPP_INSTANCE_FLAGS_GLSL

#include "vulkan_shared.h"

bool ifWater(uint z)       { return (z & kInstFlagWater) != 0u; }
bool ifSkinned(uint z)     { return (z & kInstFlagSkinned) != 0u; }
bool ifDoubleSided(uint z) { return (z & kInstFlagDoubleSided) != 0u; }
bool ifMoving(uint z)      { return (z & kInstFlagMoving) != 0u; }
uint ifClassId(uint z)     { return (z >> 8) & 0xFFu; }

float ifShadingReactivity(uint z) {
    // kInstFlagSplat is the third source of the same meaning: a Gaussian-splat
    // pixel carries a motion vector built from the alpha-weighted EXPECTED
    // depth of a semi-transparent stack, which is exact for an opaque cloud and
    // approximate at every silhouette — so history is worth less there than a
    // rasterized surface's is. Deliberately NOT in ifIrradianceReactivity:
    // splats contribute no demodulated GI integrand at all.
    return ((z & (kInstFlagDeformer | kInstFlagTexAnim | kInstFlagSplat)) != 0u) ? 1.0 : 0.0;
}

float ifIrradianceReactivity(uint z) {
    return ((z & kInstFlagDeformer) != 0u) ? 1.0 : 0.0;
}

#endif// THREEPP_INSTANCE_FLAGS_GLSL
