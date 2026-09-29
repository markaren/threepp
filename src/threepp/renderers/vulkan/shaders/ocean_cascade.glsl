// Ocean cascade sample-domain helpers, shared by every GLSL consumer of the
// FFT cascade textures (water_displace.comp, foam_world.comp). The rotation
// constants live in vulkan_shared.h so the C++ mirror
// (DisplacedMesh::sampleHeight) and the host-side windTheta compensation use
// the exact same literals — see the comment block there for the why.

#ifndef THREEPP_OCEAN_CASCADE_GLSL
#define THREEPP_OCEAN_CASCADE_GLSL

#include "vulkan_shared.h"

// World XZ → cascade-1 sample domain (q = R(θ)·p).
vec2 oceanC1Domain(vec2 p) {
    return vec2(kOceanCascade1RotCos * p.x - kOceanCascade1RotSin * p.y,
                kOceanCascade1RotSin * p.x + kOceanCascade1RotCos * p.y);
}

// Cascade-1 domain vector → world (w = Rᵀ·v). Apply to the sampled
// horizontal-displacement (dx,dz) so the pull stays a world-space vector.
vec2 oceanC1World(vec2 v) {
    return vec2( kOceanCascade1RotCos * v.x + kOceanCascade1RotSin * v.y,
                -kOceanCascade1RotSin * v.x + kOceanCascade1RotCos * v.y);
}

// Width (m) of the band outside a hull's footprint over which the sea blends from
// the vessel's waterline plane back to the wave field, and inside which the wake
// ridges fade in. It was a fixed 2 m, sized for the 28 m Gunnerus; on a 6 m USV
// that flattened a glassy apron a third of her length ahead of the bow. Now it
// scales with the hull, 0.6 x half-beam clamped to [0.3, 2] m, so Gunnerus
// (4.5 m half-beam) keeps exactly 2 m. `scaled` false = the old fixed 2 m (the
// THREEPP_OCEAN_HULL_MARGIN=0 A/B). DisplacedMesh::sampleWakeHeight mirrors this.
float oceanHullMargin(float halfBeam, bool scaled) {
    return scaled ? clamp(0.6 * halfBeam, 0.3, 2.0) : 2.0;
}

#endif// THREEPP_OCEAN_CASCADE_GLSL
