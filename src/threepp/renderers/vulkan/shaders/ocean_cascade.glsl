// Ocean cascade sampling helpers (the sample domain, a C1 bicubic fetch),
// shared by every GLSL consumer of the FFT cascade textures
// (water_displace.comp, foam_world.comp). The rotation
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

// Bicubic B-spline RG fetch (4 bilinear taps, Sigg & Hadwiger). The kernel is
// C1, so neither a derivative of the fetched field nor a mesh finer than its
// texels finds a kink at a texel boundary, as both do in a bilinear fetch:
// foam_world's Jacobian and water_displace's displacement say what that broke.
vec2 sampleBicubicRG(sampler2D tex, vec2 uv) {
    const vec2 res = vec2(textureSize(tex, 0));
    const vec2 st  = uv * res - 0.5;
    const vec2 i   = floor(st);
    const vec2 f   = st - i;
    const vec2 f2  = f * f;
    const vec2 f3  = f2 * f;
    const vec2 w0  = (-f3 + 3.0 * f2 - 3.0 * f + 1.0) * (1.0 / 6.0);
    const vec2 w1  = (3.0 * f3 - 6.0 * f2 + 4.0)       * (1.0 / 6.0);
    const vec2 w2  = (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) * (1.0 / 6.0);
    const vec2 w3  = f3 * (1.0 / 6.0);
    const vec2 g0  = w0 + w1;
    const vec2 g1  = w2 + w3;
    const vec2 c0  = (i - 0.5 + w1 / g0) / res;
    const vec2 c1  = (i + 1.5 + w3 / g1) / res;
    const vec2 t00 = texture(tex, vec2(c0.x, c0.y)).rg;
    const vec2 t10 = texture(tex, vec2(c1.x, c0.y)).rg;
    const vec2 t01 = texture(tex, vec2(c0.x, c1.y)).rg;
    const vec2 t11 = texture(tex, vec2(c1.x, c1.y)).rg;
    return g0.y * (g0.x * t00 + g1.x * t10) + g1.y * (g0.x * t01 + g1.x * t11);
}

// Width (m) of the band outside a hull's footprint over which the sea blends from
// the vessel's waterline plane back to the wave field, and inside which the wake
// ridges fade in. It scales with the hull, 0.6 x half-beam clamped to [0.3, 2] m:
// a fixed 2 m, sized for the 28 m Gunnerus (4.5 m half-beam, which keeps exactly
// 2 m), flattened a glassy apron a third of a 6 m USV's length ahead of her bow.
// DisplacedMesh::sampleWakeHeight mirrors this.
float oceanHullMargin(float halfBeam) {
    return clamp(0.6 * halfBeam, 0.3, 2.0);
}

// One vessel as the ocean passes read it: DisplacedMesh::HullExclusion, the
// wake's forward speed and the vessel's slice of the wake-trail buffer. The host
// packs only the active vessels (halfLength > 0). 48 bytes, mirrored field for
// field by vulkan::OceanHullGpu (WaterDisplacePipeline.hpp).
struct OceanHull {
    float centerX;
    float centerZ;
    float halfLength;
    float halfBeam;
    float sinYaw;
    float cosYaw;
    float forwardSpeed; // m/s along +heading; 0 = no wake (or the wake is disabled)
    float centerY;      // waterline plane: world y at (centerX, centerZ),
    float pitch;        // +bow up (+localZ),
    float roll;         // +starboard up (+localX)
    uint  trailFirst;   // this vessel's samples in the wake-trail buffer
    uint  trailCount;
};

#endif// THREEPP_OCEAN_CASCADE_GLSL
