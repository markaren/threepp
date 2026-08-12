// Shared declarations for the splat VOLUME BAKE — the two dispatches
// SplatPass::uploadCloud records inside its staging one-shot
// (plans/splat-volume-reflections.md, Part 1).
//
// The bake voxelizes a cloud ONCE, in cloud-local space, into an rgba16f 3D
// image (rgb = linear radiance, a = sigma_t per local metre) that rays the tile
// rasterizer cannot serve — reflection legs first — march as a participating
// medium. Clouds are rigid, so this is paid per upload rather than per frame.
//
// ITS OWN DESCRIPTOR SET LAYOUT, not splat_common.glsl's. The bake binds
// nothing the frame binds: no sceneHdr, no G-buffer, no fog/cloud/lights UBOs,
// no environment. Including splat_common.glsl to reuse its SH_C0 and
// splatSrgbToLinear would drag its 24 bindings into a pipeline layout that has
// four, so the two constants the bake needs are mirrored at their use sites
// with KEEP IN SYNC notes instead.

#ifndef THREEPP_SPLAT_BAKE_COMMON_GLSL
#define THREEPP_SPLAT_BAKE_COMMON_GLSL

// ── Fixed point: Q20.12, and the whole determinism argument ──────────────────
// The scatter accumulates with integer atomicAdd ONLY. Integer adds are
// associative, so a voxel's four counters end up with the same bits however the
// GPU scheduled the atomics — VERBATIM the argument particle_density.glsl makes
// for its r32ui density volume, and the reason a massively parallel bake can be
// asserted byte-for-byte (VulkanSplat_test's volume hash). imageAtomicAdd on a
// float format would make the result a per-run coin flip even where the
// extension exists, because float addition is not associative.
//
// 12 fractional bits: quantum 2.44e-4 /m, saturation 2^32/4096 = 1.05e6 /m.
// The colour counters carry sigma*radiance in the same units, so a linear
// radiance of ~5 costs a fifth of that headroom. Reaching either needs ~10^5
// splats in ONE voxel, which is a volume whose resolution is grossly mismatched
// to its contents rather than a scene — the same trade, with the same numbers,
// particle_density.glsl's header records for the dust volume.
const float kSplatVolFixedScale    = 4096.0;
const float kSplatVolInvFixedScale = 1.0 / 4096.0;

// Per-bake scalars. A push constant rather than a UBO because the bake runs
// once per upload from a one-shot command buffer — there is no frame to key a
// buffer slot off, and nothing here outlives the submission.
//
// KEEP IN SYNC with SplatBakePc in SplatPass.cpp, which static_asserts its own
// size. scalar layout, so vec3 packs to 12 bytes and the block is 60.
layout(push_constant, scalar) uniform SplatBakePc {
    vec3  boxMin;      // cloud-local min corner of the bake box
    vec3  boxSize;     // its extent, local metres (strictly positive)
    vec3  voxelSize;   // boxSize / res, per axis
    uvec3 res;         // voxels per axis
    uint  splatCount;
    uint  shCoeffs;    // (degree+1)^2; coefficient 0 is the DC term
    float voxelVolume; // voxelSize.x * .y * .z, local m^3
} pc;

// Transient accumulator, four counters per voxel: sigma, sigma*r, sigma*g,
// sigma*b, all Q20.12. Created for the bake and destroyed once the one-shot's
// wait returns — it is 16 B/voxel against the image's 8, and nothing outside
// the submission has any use for it.
layout(set = 0, binding = 2, scalar) coherent buffer SplatBakeScratch { uint scratch[]; };

// x fastest, z slowest — the order the resolve's 3D dispatch walks and the
// order the host sizes the scratch in.
uint svVoxelIndex(uvec3 c) {
    return (c.z * pc.res.y + c.y) * pc.res.x + c.x;
}

// Value -> Q20.12 counter. NON-FINITE IN, ZERO OUT: `q > 0.0` is false for NaN,
// so a corrupt splat deposits nothing rather than executing an undefined
// float->uint conversion. The min keeps a single absurd deposit inside the
// counter's range; the SUM is bounded by the resolve's soft cap instead.
uint svFix(float v) {
    const float q = v * kSplatVolFixedScale + 0.5;
    return (q > 0.0) ? uint(min(q, 4.29e9)) : 0u;
}

#endif// THREEPP_SPLAT_BAKE_COMMON_GLSL
