// Shared declarations for the Gaussian-splat compute tile rasterizer.
//
// Every splat_*.comp includes this and binds the SAME descriptor set layout
// (see vulkan/SplatPass.hpp) — one layout for the whole pipeline means one
// descriptor write per (cloud, frame-in-flight) instead of one per dispatch.
// Bindings a given stage doesn't touch are still declared; an unused binding
// costs nothing and keeps the layout literally identical across stages.
//
// GEOMETRY IS BACKEND-NEUTRAL. means / opacity / the six distinct entries of
// the 3D covariance come from SplatData exactly as the GL path uploads them
// (SplatCloud.cpp buildTextures) — same precompute, same fp32/fp16 split, so
// the two backends read the same numbers and a difference between them is a
// difference in the RASTERIZER, which is what the GL-oracle comparison is for.

#ifndef THREEPP_SPLAT_COMMON_GLSL
#define THREEPP_SPLAT_COMMON_GLSL

// The per-instance flag bit layout the G-buffer IDs channel uses — the splat
// raster ORs kInstFlagSplat into it so the temporal resolve knows a pixel's
// motion came from an expected depth rather than a rasterized surface.
#include "vulkan_shared.h"

// ── Tiling ──────────────────────────────────────────────────────────────────
// 16x16 is the 3DGS reference tile and it is also exactly one raster
// workgroup (256 threads, one thread per pixel), so the tile loop needs no
// pixel-to-thread mapping beyond gl_LocalInvocationID.
// Submission ranges per cloud (per-chunk LOD). The Sanctuaire scan is ~10
// chunks per level across 4 levels, so 64 covers "every chunk at some level"
// with room to spare. KEEP IN SYNC with kMaxRanges in SplatPass.cpp.
const uint kMaxRanges = 64u;

const uint kTileW = 16u;
const uint kTileH = 16u;
const uint kTilePixels = kTileW * kTileH;

// ── Scan / radix block sizes ────────────────────────────────────────────────
// Both are (threads x elements-per-thread). The scan block is deliberately
// 1024 so a single 256-thread workgroup can scan a whole level of block sums
// for any cloud up to 1024^3 elements — see SplatPass::recordScan.
const uint kScanThreads  = 256u;
const uint kScanPerThread = 4u;
const uint kScanBlock    = kScanThreads * kScanPerThread;// 1024

const uint kRadixThreads  = 128u;
const uint kRadixPerThread = 4u;
const uint kRadixBlock    = kRadixThreads * kRadixPerThread;// 512
const uint kRadixBits     = 4u;
const uint kRadixBins     = 1u << kRadixBits;// 16

// The blend is a no-op below this alpha, and transmittance below it can no
// longer change a stored 8-bit-ish result. Kept identical to the GL fragment
// shader's constant.
const float kMinAlpha = 0.00392156862;// 1/255
const float kMaxAlpha = 0.99;

// EWA low-pass on the projected covariance. IDENTICAL to SCREEN_DILATION in
// SplatCloud.cpp's vertex shader — without the same dilation the two backends
// disagree on every sub-pixel splat, which is most of a scan.
const float kScreenDilation = 0.3;

// A splat straddling the near plane projects to an unbounded footprint; the
// GL path clamps its quad radius to the viewport's long side and so does this.
// Without it one bad Gaussian expands into every tile on screen and the
// expansion budget is gone.
const float kMaxRadiusFactor = 1.0;

// ── Per-splat projection record ─────────────────────────────────────────────
// Written by splat_project.comp, read by expand + raster.
//
// EXACTLY 64 BYTES with scalar layout, and SplatPass.cpp's kProjStride says so
// in C++. Nothing checks that the two agree, and they only have to disagree by
// four bytes for the host to under-allocate the array: the shader keeps
// striding by its own size, the tail of the cloud writes past the end, and
// whatever fraction of the splats that is simply vanishes — as a CONTIGUOUS
// region of the scan, because a PLY's splat order is spatially coherent. It
// looks exactly like a culling bug. (It was one, for a while.)
//
// scalar layout packs to component alignment, so: 8 + 12 + 4 + 12 + 4 + 20 + 4.
struct SplatProj {
    vec2  center;  // 0   pixel coords of the mean (framebuffer space, y down)
    vec3  conic;   // 8   inverse 2D covariance (a, b, c)
    float opacity; // 20  [0, 1]
    vec3  color;   // 24  display-referred, un-pre-exposed (see the domain note)
    float viewDist;// 36  positive distance along the view axis; the key's source
    uint  tileX0;  // 40  inclusive tile rect
    uint  tileY0;  // 44
    uint  tileX1;  // 48
    uint  tileY1;  // 52
    uint  tileCount;// 56 (tileX1-tileX0+1) * (tileY1-tileY0+1), 0 = culled
    uint  pad0;    // 60  rounds the record to 64
};

// ── Globals (one record, GPU-owned) ─────────────────────────────────────────
// minDistBits / maxDistBits are atomic min/max over the visible splats'
// viewDist, monotone-encoded: viewDist is strictly positive here, and the IEEE
// bit pattern of a positive float is monotone in the float, so a plain uint
// atomicMin/atomicMax IS a float min/max. Both are exact and order-independent,
// so they cost the frame nothing in determinism.
struct SplatGlobals {
    uint minDistBits;
    uint maxDistBits;
    uint entryCount;  // total (splat, tile) pairs the expansion wants
    uint overflow;    // pairs that did not fit the budget (0 = healthy)
    uint visibleCount;// splats that survived the cull
    // Order-independent hashes, all built with uint atomicAdd — commutative
    // and associative, so the value does not depend on scheduling and a
    // DIFFERENCE between two runs is a real difference, never a race.
    uint hashGeom;    // uploaded means/opacity/covariance (V0 upload proof)
    uint hashSh;      // uploaded spherical harmonics
    uint hashKey;     // sorted key array
    uint hashVal;     // sorted payload array
    uint hashColor;   // composited pixels (determinism gate)
    uint pad0;
    uint pad1;
};

// ── Descriptor set 0 — identical in every stage ─────────────────────────────

layout(set = 0, binding = 0, scalar) uniform SplatUbo {
    mat4  modelView;  // cloud local -> view
    mat4  proj;       // view -> clip, GL convention (NDC z in [-1,1], y UP), JITTERED
    mat4  projInv;    // clip -> view, REVERSE-Z — the one gbufDepth was written with
    mat4  model;      // cloud local -> world (SH view direction)
    // View space -> PREVIOUS frame's unjittered clip, in one matrix (the host
    // composes TaaResolve's own sky-reprojection with this frame's projection,
    // so the splat motion vectors are built from exactly the matrices the
    // raster's are).
    mat4  prevVPfromView;
    mat4  camWorld;   // view -> world (fog needs a world-space height)
    vec4  camPosWs;   // world-space camera position, .w unused
    vec4  camFwdWs;   // world-space camera forward (orthographic view axis)
    vec2  viewport;   // render extent in pixels
    vec2  focal;      // fx, fy in pixels
    vec2  percentile; // p1 / p99 of the view distances (CPU sample)
    vec2  jitterClip; // this frame's raster jitter, already inside `proj`
    float nearPlane;
    float preExposure;// the factor already baked into sceneHdr's stores
    // Point rendering (SplatCloud::setPointMix). 0 = Gaussians, the pre-
    // existing path bit for bit; 1 = an opaque disc of 3 * pointSigma pixels
    // radius (one-pixel feather at the edge) per splat. project lerps the
    // projected covariance and the opacity, raster lerps the falloff.
    float pointMix;
    float pointSigma;// pixels; see SplatCloud::pointSigmaPixels

    uint  splatCount;
    uint  shCoeffs;   // (degree+1)^2
    uint  shDegree;
    uint  tilesX;
    uint  tilesY;
    uint  tileBits;   // bits the tile id occupies at the TOP of the sort key
    uint  depthBits;  // 32 - tileBits
    uint  budget;     // expanded-entry capacity
    uint  flags;      // see kSplatFlag* below
    uint  envMipCount;// mip levels in the prefiltered env (top mip = sky ambient)
    // ── Partial submission (per-chunk LOD) ──────────────────────────────────
    // The cloud's buffers hold every chunk at every detail level, uploaded once;
    // a frame submits a SUBSET as a list of source ranges. splatCount above is
    // the submitted TOTAL, so every stage after project — the scan, the sort,
    // the raster — works on a compact [0, splatCount) index and needs no
    // knowledge of this at all. Only project translates.
    //
    // Why ranges rather than one cloud per chunk: a second SplatCloud is a
    // second run of the ENTIRE pass (clears, sizing, 8 sort rounds, a
    // full-screen tile walk) and measures ~1.3 ms — ten chunks would be ~12 ms
    // before drawing a splat. And why ranges rather than rebuilding a merged
    // buffer per selection: the rebuild is a re-upload of up to 1.2 GB.
    //
    // rangeCount 0 = submit the whole cloud, the identity path, bit-for-bit
    // what this pass did before ranges existed.
    uint  rangeCount;
    uvec2 ranges[kMaxRanges];// .x = first compact index, .y = source base
} ubo;

// Compact destination index -> source splat index. Binary search over the
// range table's compact starts, which are ascending by construction (the host
// builds them by accumulating counts). ~6 UBO reads at kMaxRanges = 64.
uint splatSourceIndex(uint dst) {
    if (ubo.rangeCount == 0u) return dst;
    uint lo = 0u, hi = ubo.rangeCount - 1u;
    while (lo < hi) {
        const uint mid = (lo + hi + 1u) / 2u;
        if (ubo.ranges[mid].x <= dst) lo = mid;
        else                          hi = mid - 1u;
    }
    return ubo.ranges[lo].y + (dst - ubo.ranges[lo].x);
}

const uint kSplatFlagOrtho      = 1u;// parallel projection
const uint kSplatFlagDebugNaN   = 2u;// paint non-finite results magenta
const uint kSplatFlagDepthTest  = 4u;// terminate behind opaque geometry
const uint kSplatFlagMotion     = 8u;// write gbufMotion + reactivity (V2)
const uint kSplatFlagFog        = 16u;// apply height fog per splat (V2)
const uint kSplatFlagChecksum   = 32u;// hash the composited result (determinism test)
const uint kSplatFlagBgSolid    = 64u;// scene background is a solid colour
const uint kSplatFlagDepthAov   = 128u;// export a view distance to splatDepth
const uint kSplatFlagDepthMed   = 256u;// that AOV carries the MEDIAN, not the expected, distance

// Per-dispatch scalars. One block, shared by every stage — the scan and the
// radix run the same shader many times over different regions, and a push
// constant is the only per-dispatch channel that costs nothing.
layout(push_constant) uniform SplatPc {
    uint count;  // elements this dispatch covers
    uint srcOff; // scanScratch read base  (scan levels > 0)
    uint dstOff; // scanScratch write base
    uint sumOff; // block-sum write base
    uint arg0;   // scan: 0 = counts->offsets, 1 = scratch in place | radix: shift
    uint arg1;   // radix: block count | checksum: mode
    uint arg2;
    uint arg3;
} pc;

// Per-splat static geometry, uploaded once, device-local. 40 bytes, scalar.
struct SplatGeom {
    vec3  mean;
    float opacity;
    float cov[6];// xx, xy, xz, yy, yz, zz
};
layout(set = 0, binding = 1, scalar) readonly buffer SplatGeomBuf { SplatGeom geom[]; };

// SH, half-packed: TWO uints per coefficient — packHalf2x16(r, g) and
// packHalf2x16(b, 0). That is the same 8 bytes per coefficient the GL path
// spends on an RGBA16F texel (SplatCloud.hpp explains why the alpha lane is
// worth the 25%: one fetch per coefficient, no straddling).
layout(set = 0, binding = 2, scalar) readonly buffer SplatShBuf { uint sh[]; };

layout(set = 0, binding = 3, scalar) buffer ProjBuf     { SplatProj proj[]; };
layout(set = 0, binding = 4, scalar) buffer CountBuf    { uint counts[]; };
layout(set = 0, binding = 5, scalar) buffer OffsetBuf   { uint offsets[]; };
layout(set = 0, binding = 6, scalar) buffer KeyABuf     { uint keysA[]; };
layout(set = 0, binding = 7, scalar) buffer ValABuf     { uint valsA[]; };
layout(set = 0, binding = 8, scalar) buffer KeyBBuf     { uint keysB[]; };
layout(set = 0, binding = 9, scalar) buffer ValBBuf     { uint valsB[]; };
layout(set = 0, binding = 10, scalar) buffer RangeBuf   { uvec2 tileRange[]; };
layout(set = 0, binding = 11, scalar) buffer GlobalBuf  { SplatGlobals g; };
layout(set = 0, binding = 12, scalar) buffer HistBuf    { uint hist[]; };
layout(set = 0, binding = 13, scalar) buffer ScanBuf    { uint scanScratch[]; };

layout(set = 0, binding = 14, rgba16f) uniform image2D sceneHdr;
layout(set = 0, binding = 15) uniform sampler2D gbufDepth;// reversed-Z D32
layout(set = 0, binding = 16, rgba16f) uniform image2D gbufMotion;
// READ-ONLY, and sampled rather than storage on purpose: the ids attachment is
// created without STORAGE usage (it is a render target the whole rest of the
// frame samples), and the one thing this pass needs from it — is this pixel the
// solid-colour background the shade left un-pre-exposed — is a read.
layout(set = 0, binding = 17) uniform usampler2D gbufIds;

// ── Fog inputs ──────────────────────────────────────────────────────────────
// The same UBOs the deferred shade and particle_light.comp read, bound here
// because deferred_shade_60_fog_volumetrics.glsl bakes fog into sceneHdr
// DURING the shade — anything composited afterwards gets exactly none of it,
// and a splat cloud punching a clear hole through a foggy scene is the most
// obvious tell there is. KEEP IN SYNC with particle_light.comp's copies.
layout(set = 0, binding = 18, scalar) uniform FogUbo {
    vec3  sigmaT;
    float enabled;
    vec3  color;
    float anisotropy;
    float waterSurfaceY;
    vec3  worldUp;
    float hfDensity;
    float hfBaseY;
    float hfFalloff;
    float murkDensity;
    vec3  murkColor;
} fog;

layout(set = 0, binding = 19, scalar) uniform CloudUbo {
    float enabled;
    float coverage;
    float density;
    float bottomY;
    float topY;
    float evolveSpeed;
    float timeSec;
    float heteroActive;
    vec3  wind;
    float hfDensity;
    float hfBaseY;
    float hfFalloff;
    float hfNoiseAmount;
    float shadowActive;
} clouds;

// Only the leading `ambient` and the sun list are read; a uniform block may be
// shorter than the buffer bound to it, and the offsets of what it does declare
// still match — so the point/spot/rect tail stays undeclared. The fields that
// ARE declared must keep the exact order and types deferred_shade.comp and
// particle_light.comp give them, or every offset after the first mismatch is
// garbage.
struct DirLight { vec3 direction; vec3 color; };
layout(set = 0, binding = 20, scalar) uniform LightsUbo {
    vec3     ambient;
    uint     dirCount;
    DirLight dirLights[8];
} lights;

layout(set = 0, binding = 21) uniform sampler2D envTex;// prefiltered PMREM chain

// VkDispatchIndirectCommand pairs written by splat_indirect.comp: [0..2] the
// radix histogram/scatter extent, [3..5] the tile-range extent. Written by one
// thread after the expansion, consumed by vkCmdDispatchIndirect in the same
// submission — so it is one buffer, not one per frame in flight.
layout(set = 0, binding = 22, scalar) buffer IndirectBuf { uint indirect[]; };

// Depth AOV: a per-pixel view distance in WORLD UNITS (positive, view-space)
// for the pixels a cloud owns — the alpha-weighted EXPECTED distance the
// accumulation loop already produces, or the MEDIAN one (the transmittance-0.5
// crossing) under kSplatFlagDepthMed. Written only under kSplatFlagDepthAov; a 1x1
// image is bound when the AOV is off, because every binding in this set needs
// a real object either way (writeSets' opening comment). See splat_raster.comp
// for the coverage gate and the nearest-wins rule.
layout(set = 0, binding = 23, r32f) uniform image2D splatDepth;

// ── Spherical harmonics ─────────────────────────────────────────────────────
// The SAME decimal literals as threepp/splats/SplatSH.hpp and as the GL
// vertex shader. SplatSH_test asserts the GL text still carries every one of
// them; this file is pinned the same way by SplatSH_test's Vulkan clause.
const float SH_C0 = 0.28209479177387814;
const float SH_C1 = 0.4886025119029199;
const float SH_C2_0 = 1.0925484305920792;
const float SH_C2_1 = -1.0925484305920792;
const float SH_C2_2 = 0.31539156525252005;
const float SH_C2_3 = -1.0925484305920792;
const float SH_C2_4 = 0.5462742152960396;
const float SH_C3_0 = -0.5900435899266435;
const float SH_C3_1 = 2.890611442640554;
const float SH_C3_2 = -0.4570457994644658;
const float SH_C3_3 = 0.3731763325901154;
const float SH_C3_4 = -0.4570457994644658;
const float SH_C3_5 = 1.445305721320277;
const float SH_C3_6 = -0.5900435899266435;

vec3 splatShCoeff(uint base, uint c) {
    const uint w = (base + c) * 2u;
    const vec2 rg = unpackHalf2x16(sh[w]);
    const vec2 b_ = unpackHalf2x16(sh[w + 1u]);
    return vec3(rg.x, rg.y, b_.x);
}

// dir points FROM the camera TO the splat, world space. Mirrors shColor() in
// SplatCloud.cpp line for line — including the deliberate absence of a clamp
// (max(NaN, 0) is 0 on most hardware, which would launder a corrupt
// coefficient into a plausible colour; the caller checks isnan first).
vec3 splatShColor(uint splat, vec3 dir) {
    const uint base = splat * ubo.shCoeffs;
    vec3 c = SH_C0 * splatShCoeff(base, 0u);

    if (ubo.shDegree >= 1u) {
        const float x = dir.x, y = dir.y, z = dir.z;
        c += -SH_C1 * y * splatShCoeff(base, 1u)
           +  SH_C1 * z * splatShCoeff(base, 2u)
           -  SH_C1 * x * splatShCoeff(base, 3u);

        if (ubo.shDegree >= 2u) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, yz = y * z, xz = x * z;
            c += SH_C2_0 * xy * splatShCoeff(base, 4u)
               + SH_C2_1 * yz * splatShCoeff(base, 5u)
               + SH_C2_2 * (2.0 * zz - xx - yy) * splatShCoeff(base, 6u)
               + SH_C2_3 * xz * splatShCoeff(base, 7u)
               + SH_C2_4 * (xx - yy) * splatShCoeff(base, 8u);

            if (ubo.shDegree >= 3u) {
                c += SH_C3_0 * y * (3.0 * xx - yy) * splatShCoeff(base, 9u)
                   + SH_C3_1 * xy * z * splatShCoeff(base, 10u)
                   + SH_C3_2 * y * (4.0 * zz - xx - yy) * splatShCoeff(base, 11u)
                   + SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * splatShCoeff(base, 12u)
                   + SH_C3_4 * x * (4.0 * zz - xx - yy) * splatShCoeff(base, 13u)
                   + SH_C3_5 * z * (xx - yy) * splatShCoeff(base, 14u)
                   + SH_C3_6 * x * (xx - 3.0 * yy) * splatShCoeff(base, 15u);
            }
        }
    }
    // +0.5 belongs to the evaluation, not the data (splats::SH_COLOR_OFFSET).
    return c + 0.5;
}

// ── Colour domain: WHERE the alpha blend happens ────────────────────────────
//
// A 3DGS optimiser fits its spherical harmonics against sRGB-ENCODED training
// images, by minimising |sum(c_i * a_i * T_i) - I_sRGB|. The coefficients are
// therefore NOT radiances; they are display-referred values chosen so that a
// DISPLAY-SPACE alpha blend of them reproduces the photograph. Where the blend
// happens is not a detail — it is the difference between reproducing the scan
// and reproducing something close to it:
//
//   decode per splat, blend in linear   +15% mean on the procedural cloud, and
//                                       a scan's near-invisible sky halo (a
//                                       shell at ~2% coverage) lifts 7x into a
//                                       visible grey DOME. Physically the
//                                       correct operator for radiances; these
//                                       are not radiances.
//   blend display, decode the composite exact where the cloud is opaque, same
//                                       halo lift where it is not (the lift is
//                                       inherent to linear compositing over
//                                       black, not to where the decode sits).
//   blend display, background included  what this does. Bit-for-bit the GL
//                                       answer at every coverage.
//
// So the whole composite — splats AND the scene behind them — happens in the
// display-referred domain, and the result re-enters sceneHdr through the
// decode. The background round-trips EXACTLY (encode then decode is the
// identity, including above 1.0 — the sRGB curve is monotone on all of
// [0, inf) and nothing clamps), so a pixel no splat touched is unchanged, and
// an HDR sky behind a translucent splat keeps its range. Everything downstream
// still sees a linear value: fog, DoF, bloom, tone mapping and TAA act on
// splats without any of it being re-derived here, which is why this pass
// composites pre-post at all.
//
// The trade-off: partial-coverage compositing is not linear-light. It is the
// operator the asset was fitted with, which for a scan — where the ground
// truth is "what the capture looks like" — is the one that matters.
vec3 splatSrgbToLinear(vec3 c) {
    const vec3 lo = c / 12.92;
    const vec3 hi = pow(max(c + 0.055, vec3(0.0)) / 1.055, vec3(2.4));
    return mix(hi, lo, lessThanEqual(c, vec3(0.04045)));
}

// KEEP IN SYNC with linearToSRGB in post_composite.comp — this is its exact
// inverse, and the pair has to round-trip or a pixel with no splats over it
// would come out changed.
vec3 splatLinearToSrgb(vec3 x) {
    const vec3 lo = 12.92 * x;
    const vec3 hi = 1.055 * pow(max(x, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(hi, lo, lessThan(x, vec3(0.0031308)));
}

// ── Sort key ────────────────────────────────────────────────────────────────
// (tileId << depthBits) | depthBucket, Open3D's scheme. Ascending key sorts
// primarily by tile and, within a tile, FRONT to back — which is the order the
// raster loop wants so it can early-out on transmittance.
//
// The depth quantisation carries the GL path's tail-band doctrine over intact
// (SplatCloud.cpp, and SplatCloudSort_test pins the CPU version): the content
// interval [p1, p99] gets almost every bucket, and each tail keeps a small
// MONOTONE band of its own instead of collapsing into one end bucket. A
// collapsed tail is what repainted a scan's sky in file order, and the sky is
// part of the acceptance set now.
uint splatDepthBucket(float dist, float dMin, float dMax, float pLo, float pHi) {
    const uint buckets = 1u << ubo.depthBits;
    // Same 1/32 of the range per tail as the GL sort's 2048 of 65536.
    const uint tail    = max(buckets >> 5u, 1u);
    const uint content = buckets - 2u * tail;

    // Every branch is written so a non-finite distance lands on a plain
    // integer (the near end) rather than an undefined float-to-int convert.
    if (!(dist >= pLo)) {
        const float span = pLo - dMin;
        const float k = (span > 0.0) ? (dist - dMin) * (float(tail - 1u) / span) : 0.0;
        return !(k > 0.0) ? 0u : (k >= float(tail - 1u) ? tail - 1u : uint(k));
    }
    if (dist <= pHi) {
        const float span = pHi - pLo;
        const float k = (span > 0.0) ? (dist - pLo) * (float(content - 1u) / span) : 0.0;
        return tail + (k >= float(content - 1u) ? content - 1u : uint(max(k, 0.0)));
    }
    const float span = dMax - pHi;
    const float k = (span > 0.0) ? (dist - pHi) * (float(tail - 1u) / span) : 0.0;
    return (buckets - tail) + (!(k > 0.0) ? 0u : (k >= float(tail - 1u) ? tail - 1u : uint(k)));
}

// ── Fog over the camera -> splat leg ────────────────────────────────────────
// The ANALYTIC form of every fog term that puts light back INTO this leg:
// exponential height-fog extinction with ambient in-scatter, the murk below a
// water surface, and the sun's single-scattering glow. KEEP IN SYNC with
// heightFogOpticalDepth / fogPathLength / volumetricDirScatter in
// deferred_shade_60_fog_volumetrics.glsl and particle_light.comp — same
// clamps, same Taylor guard, same HG phase.
//
// Still NOT mirrored, both additive: the froxel LUT's integrated point-light
// glow (needs the froxel volume + cluster grid this pass does not bind), and
// the SHADOWING of the sun term — the shafts the surface march carves with an
// RT ray per step, which needs the TLAS. So a splat cloud is a little less lit
// by nearby lamps than a mesh beside it, and its sun haze is the unoccluded
// upper bound rather than a shafted one: both leave it too BRIGHT by a bounded
// amount. That direction matters. The sun term itself used to be missing too,
// and because a sun-only scene zeroes the ambient AND env terms that were the
// only ones here, the leg kept its extinction and got nothing back — a cloud
// in lit air faded to black, the one failure a "slightly dimmer" gap cannot
// produce.

float splatHeightFogOd(vec3 a, vec3 b) {
    if (clouds.hfDensity <= 0.0) return 0.0;
    const float H   = max(clouds.hfFalloff, 1e-3);
    const float ya  = max(a.y - clouds.hfBaseY, 0.0);
    const float yb  = max(b.y - clouds.hfBaseY, 0.0);
    const float len = min(distance(a, b), 1.0e7);
    const float ea = exp(-ya / H);
    const float eb = exp(-yb / H);
    const float x  = (yb - ya) / H;
    const float f  = (abs(x) < 1e-3) ? (ea * (1.0 - 0.5 * x + x * x * (1.0 / 6.0)))
                                     : ((ea - eb) / x);
    return min(clouds.hfDensity * len * f, 80.0);
}

float splatMurkPathLength(vec3 a, vec3 b) {
    const float full = distance(a, b);
    if (fog.waterSurfaceY > 1e29) return full;
    const float ya = a.y - fog.waterSurfaceY;
    const float yb = b.y - fog.waterSurfaceY;
    if (ya >= 0.0 && yb >= 0.0) return 0.0;
    if (ya < 0.0 && yb < 0.0) return full;
    const float t = ya / (ya - yb);
    return (ya < 0.0) ? full * t : full * (1.0 - t);
}

vec3 splatEnvTop() {
    const float lod = float(max(ubo.envMipCount, 1u) - 1u);
    return textureLod(envTex, vec2(0.5, 1.0), lod).rgb;
}

// Henyey-Greenstein phase — the same expression as hgPhase in
// deferred_shade_60_fog_volumetrics.glsl / particle_light.comp / froxel_inject.comp,
// with PI written out because splat_common carries no PI of its own.
float splatHgPhase(float mu, float g) {
    const float g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159265358979 * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5));
}

// Returns the leg transmittance; `inScatter` receives what to ADD after it.
float splatFog(vec3 camPos, vec3 P, out vec3 inScatter) {
    inScatter = vec3(0.0);
    const float odAir  = splatHeightFogOd(camPos, P);
    const float odMurk = (fog.murkDensity > 0.0)
                                 ? fog.murkDensity * splatMurkPathLength(camPos, P)
                                 : 0.0;
    if (odAir <= 0.0 && odMurk <= 0.0) return 1.0;

    const float Ta = exp(-odAir);
    const float Tm = exp(-min(odMurk, 80.0));
    const vec3  fogLight  = lights.ambient + splatEnvTop();
    const vec3  airAlbedo = (fog.enabled > 0.5) ? fog.color : vec3(1.0);
    // Sequential air-then-murk composition, mirroring applyHeteroSurfaceFog +
    // applyMurk: lit*(Ta*Tm) + [airIn*(1-Ta)]*Tm + murkIn*(1-Tm).
    inScatter = airAlbedo * fogLight * (1.0 - Ta) * Tm
              + fog.murkColor * fogLight * (1.0 - Tm);
    // Sun in-scatter over the same AIR leg, closed form. The surface path
    // MARCHES this (volumetricDirScatter) because its per-step shadow ray makes
    // the integrand vary along the ray; with no TLAS here nothing varies — for a
    // directional light L and rd are both fixed, so the HG phase is constant
    // over the whole leg, and the weight the march is left carrying integrates
    // exactly: the integral of sigma(x)*exp(-od(cam,x)) over the leg IS 1 - Ta,
    // over the very profile splatHeightFogOd already integrates. So this is that
    // march with vis == 1 and no cloud shadow, not an approximation of its shape.
    // Murk gets no sun term, matching applyMurk (ambient in-scatter only), and
    // the air term is attenuated by Tm to keep the same air-then-murk order.
    // Gated as the march is (an air medium + at least one sun) plus a leg long
    // enough to matter, which also keeps normalize(P - camPos) off zero length.
    if (odAir > 1e-4 && clouds.heteroActive > 0.5 && lights.dirCount > 0u) {

        const vec3 rd = ((ubo.flags & kSplatFlagOrtho) != 0u)
                                ? normalize(ubo.camFwdWs.xyz)
                                : normalize(P - camPos);
        vec3 sun = vec3(0.0);
        for (uint i = 0u; i < lights.dirCount; ++i) {

            const vec3 L = normalize(lights.dirLights[i].direction);
            sun += lights.dirLights[i].color * splatHgPhase(dot(L, rd), fog.anisotropy);
        }
        inScatter += airAlbedo * sun * (1.0 - Ta) * Tm;
    }
    return Ta * Tm;
}

#endif// THREEPP_SPLAT_COMMON_GLSL
