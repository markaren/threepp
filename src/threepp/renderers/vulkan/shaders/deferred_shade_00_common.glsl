// Everything the deferred shade's kernels share, ahead of their main(): the
// descriptor set (set 0, the same layout for every kernel), the push and
// specialization constants, the per-pixel globals and the stage includes
// _10 to _60. deferred_shade.comp and shade_reflect.comp each include it
// once, right after their extensions, so the functions a stage calls are
// the same text in both and evaluate the same way.

// This pass owns the per-pixel dust march (applyParticleFog), so it carries
// the r16f volume array (binding 69) the froxel pipelines do not. Must precede
// every include that can reach particle_density.glsl.
#define PD_LINEAR 1

#include "vulkan_shared.h"  // MaterialDesc
#include "instance_flags.glsl"// per-instance flag accessors + reactivity channels

const float PI         = 3.14159265358979;
const float TWO_PI     = 6.28318530717959;
// Surface offset along N (avoids self-shadow / self-reflection acne). NOT a
// const: main() rescales it per-pixel from the shading point's world-coordinate
// magnitude (see rtSelfEps rationale). A float32 ULP grows with |coord|, so the
// fixed 1mm offset is swallowed by rounding at ~1e3-1e4 units and the sun's
// tmin=0.0 shadow ray (shadowVis) self-intersects the surface → the plane
// flickers under the directional light in large outdoor scenes. The per-pixel
// rescale floors at 1e-3, so geometry within ~100 units of origin is unchanged.
float SHADOW_EPS = 1e-3;

// Foliage translucency (MaterialWithTranslucency) tuning. The direct term is a
// wrapped back-light plus a view-dependent forward-scatter lobe; the ambient
// term samples env irradiance at -N. Everything is gated behind translucency>0
// (deferred_shade only, raster-primary hits) → zero cost/change when off.
const float kTransScatterPower  = 4.0; // forward-scatter lobe tightness
const float kTransScatterWeight = 0.5; // forward-scatter strength vs back-wrap
const float kTransAmbient       = 0.4; // back-face env fill strength

// Shadow-ratio history floor when the σ-step cut fires on a moving-caster edge.
// Low enough to track the sweeping shadow (~half the old smudge), high enough
// that no single 1-2 ray (binary) sample dominates → no salt-and-pepper dust.
const float kShadowStepFloor    = 5.0;

#include "camera_ubo.glsl"
#include "camera_ray.glsl"// camRayOrigin / camRayDir — perspective AND ortho

// This pixel's primary ray, set once at the top of main() and read by the
// included _NN stages below (they shade the point this ray landed on and have
// no ndc of their own to rebuild it from). Under a perspective camera
// gPrimaryOrigin is the eye — the value those sites read out of viewInverse[3]
// before parallel projections existed.
vec3 gPrimaryOrigin = vec3(0.0);
vec3 gPrimaryDir    = vec3(0.0, 0.0, -1.0);
// Height of this pixel's NEAR-PLANE point (camNearPoint) — the camera's port,
// not its eye. Only the underwater MEDIUM test reads it: a lens parked at the
// sea surface is submerged for the pixels whose port point is under it and dry
// for the rest, which is what splits the frame along the waterline instead of
// flipping the whole thing at once. Every other camera-relative term keeps
// using gPrimaryOrigin. Set beside it, so the same is true per pixel.
float gNearPointY = 0.0;
// This pixel's integer coordinate, in the same global-invocation space every
// other per-pixel term uses. A GLOBAL rather than a parameter because
// applyParticleFog needs it for its spatial dither and is reached through
// skyBackground(), which the MSAA sky-minority blend calls with no px of its
// own; threading it down would grow four signatures to hand one value to one
// hash. Set beside gPrimaryOrigin, so every reader below sees this pixel's.
ivec2 gPixelCoord = ivec2(0);

#include "lights_ubo.glsl"

layout(set = 0, binding = 2) uniform sampler2D  envTex;        // prefiltered PMREM mip chain
layout(set = 0, binding = 3) uniform sampler2D  gbufNormalTex; // .xyz = n*0.5+0.5, .w = roughness (<0 = unlit)
layout(set = 0, binding = 4) uniform sampler2D  gbufDepthTex;  // .x = Vulkan NDC z [0,1]
layout(set = 0, binding = 5) uniform usampler2D gbufIdsTex;    // .x = instanceId+1 (0 == sky)
layout(set = 0, binding = 6) uniform sampler2D  gbufAlbedoTex; // .rgb = albedo, .a = metalness
// Not writeonly: dispatch B (shadeMode==1, per-sample edge shading) does a
// read-modify-write accumulate (imageLoad + imageStore) into pixels dispatch
// A already wrote. Dispatch A's own stores are still plain writes.
layout(set = 0, binding = 7, rgba16f) uniform image2D outImage;
#ifndef THREEPP_NO_RAY_QUERY
layout(set = 0, binding = 8) uniform accelerationStructureEXT topAS;
#else
#include "ray_query_stub.glsl"
#endif
layout(set = 0, binding = 9, scalar) readonly buffer MatBuf  { MaterialDesc mats[]; };

#include "geometry_fetch.glsl"// GeometryDesc, the vertex/index buffer references, gfetch*
layout(set = 0, binding = 10, scalar) readonly buffer GeomDescBuf { GeometryDesc geoms[]; };
// mats[] is the material TABLE, one record per material, not one per entry: an
// entry, a G-buffer id less one or a traced hit's instance custom index,
// reaches its record through its GeometryDesc, and entryMat folds in the
// entry's instance colour. `precise`: the one rounded multiply the host made
// when the colour lived in a per-entry desc, never fused into what follows.
uint matIdxOf(uint entry) { return geoms[entry].flags >> kGeomFlagMaterialShift; }
uint matIdxOf(int entry) { return geoms[entry].flags >> kGeomFlagMaterialShift; }
MaterialDesc entryMat(uint entry) {
    MaterialDesc m = mats[matIdxOf(entry)];
    precise vec3 tinted = m.albedo * geoms[entry].instanceColor;
    m.albedo = tinted;
    return m;
}
MaterialDesc entryMat(int entry) { return entryMat(uint(entry)); }

// Bindless material textures (reflection hits sample these). Indexed by the
// material's per-map texIndex.
layout(set = 0, binding = 11) uniform sampler2D albedoMaps[kMaxMaterialTextures];
#include "hit_fetch.glsl"// fetchUvAt / fetchHit / hitTex at a traced hit
// Emissive triangles (area-light NEE), packed as EmTri:
// v0.xyz/area, v1.xyz/cumPower(CDF), v2.xyz/power, emission.rgb (per-frame
// emissiveTriBuffers).
struct EmTri {
    vec4 v0;       // xyz = pos, w = area
    vec4 v1;       // xyz = pos, w = cumPower (running power CDF)
    vec4 v2;       // xyz = pos, w = power (lum × area)
    vec4 emission; // xyz = emissive × intensity
};
layout(set = 0, binding = 12, scalar) readonly buffer EmissiveTriBuf { EmTri emissiveTris[]; };

// Ocean FFT fine-cascade height (sub-mesh chop) + world-space foam accumulator.
// Both fall back to 1×1 dummies (and the
// tile-size push constants to 0) when no DisplacedMesh is in the scene, so the
// thin-shell water branch's `tileSize > 0` gates skip sampling them entirely.
layout(set = 0, binding = 13) uniform sampler2D oceanFineHeight;
layout(set = 0, binding = 14) uniform sampler2D oceanFoamWorld;
// The ocean's wake field (DisplacedMesh::WakeField, WakeFieldPipeline): what
// propellers, jets and paddles left on the water, as one 2D array of
// world-anchored patches (R foam, G aeration, B turbulence m/s, A lane). A 1x1
// dummy when no ocean has one. It is read only through the patch table that an
// ocean's GeometryDesc::foamAddress names (1 = an ocean without a wake field).
layout(set = 0, binding = 78) uniform sampler2DArray oceanWakeField;
#define OCEAN_WAKE_SAMPLER oceanWakeField
// The patches' RIPPLES, the waves their producers make: one layer a patch, a
// period of the patch's own small sea (slope x, slope z, their squares' sum),
// mipped, REPEAT. The same 1x1 dummy when no ocean has any.
layout(set = 0, binding = 79) uniform sampler2DArray oceanWakeRipple;
#define OCEAN_WAKE_RIPPLE_SAMPLER oceanWakeRipple
#include "ocean_wake.glsl"
// This pixel's read of it, taken once in main() where the ocean's fine normal
// is built and used again by shadeWater (bubbles, film). Zero off a wake.
vec4 gWake = vec4(0.0);
// The field's own scale on what its film and bubbles turn back (WakeField::albedo).
float gWakeAlbedo = 1.0;
// Baked tileable foam detail (RG8, mipped): R = micro bubble brightness
// (designed for a 4 m world mapping), G = ridged lace/filament pattern
// (12 m mapping). Replaces the per-pixel procedural noise octaves in the
// foam block — a sampled texture is band-limited at distance via its mip
// chain (no shimmer under TAA) and 2 fetches replace ~20 hash evaluations.
layout(set = 0, binding = 34) uniform sampler2D foamDetailTex;
// 64×64 R8 void-and-cluster blue-noise tile. Used to dither the stochastic GI hemisphere directions
// so the 1-spp residual is high-frequency (à-trous-removable) instead of the
// low-frequency clumps white noise leaves behind — the AO speckle.
layout(set = 0, binding = 35) uniform sampler2D blueNoiseTex;
layout(set = 0, binding = 15) uniform sampler2D gbufUvTex;     // .rg = primary surface UV (emissive-map sample)
// Demodulated diffuse-indirect irradiance (gatherEnv AO/GI, PRE-albedo). Written
// here, spatially filtered + recombined by deferred_gi_filter.comp. vec4(0) where
// the indirect is kept inline (sky/unlit/water/glass) so the filter adds
// nothing there.
// Write-target for THIS frame's accumulated demod GI (the denoise reads it; NEXT
// frame reads it as the reproject history via binding 17).
layout(set = 0, binding = 16, rgba16f) uniform image2D indirectImage;
// GI reproject: prev accumulated GI (the OTHER frame-in-flight's indirect
// = 1-frame history), the motion vector, and the prev world NORMALS for the
// geometric disocclusion (NOT IDs — those are draw-order, see binding setup).
layout(set = 0, binding = 17) uniform sampler2D giPrevTex;      // 1-frame GI history (reprojected)
layout(set = 0, binding = 18) uniform sampler2D gbufMotionTex;  // .rg = NDC motion
layout(set = 0, binding = 19) uniform sampler2D gbufNormalPrevTex;// prev world normals (geometric disocclusion)
layout(set = 0, binding = 20, rg32f) uniform image2D momentsImage; // .r = SVGF E[L²] accumulator (r32 — the square overflows fp16 at lum≈256); .g = GI content-change trend (antilag)
layout(set = 0, binding = 21) uniform sampler2D momentsPrevTex;   // PREV E[L²] (reprojected SVGF history)
layout(set = 0, binding = 24) uniform sampler2D gbufDepthPrevTex; // PREV depth (depth-discontinuity disocclusion)
layout(set = 0, binding = 55) uniform usampler2D gbufIdsPrevTex;  // PREV ids (.x = instanceId+1) — moving-mesh trailing-edge GI disocclusion
layout(set = 0, binding = 25, rgba16f) uniform image2D reflectImage;// sharp mirror-ray reflection radiance (.rgb pre-weighted, .a=roughness); denoise blurs by roughness
layout(set = 0, binding = 26) uniform sampler2D reflectPrevTex;   // PREV reflect (other fif) — temporal AA history
// ReSTIR DI reservoir ping-pong (STORAGE images, GENERAL layout).
// 27/28 = lightPos.xyz + lightType.w (rgba32f) write/read; 29/30
// = W_sum/M/W/p_hat (rgba16f) write/read. READ = last frame's reservoir (temporal
// reuse); WRITE = this frame's, persisted for next frame's read.
layout(set = 0, binding = 27, rgba32f) uniform writeonly image2D reservoirPosWrite;
layout(set = 0, binding = 28, rgba32f) uniform readonly  image2D reservoirPosRead;
layout(set = 0, binding = 29, rgba16f) uniform writeonly image2D reservoirWWrite;
layout(set = 0, binding = 30, rgba16f) uniform readonly  image2D reservoirWRead;
// Reflection denoiser AUX (ReBLUR-style), ping-pong (mirrors reflect 25/26). 31 = CUR
// write, 32 = PREV read. .x = history length (accumulated frames, →fast convergence +
// noise proxy), .y = E[L²] of the reflection luma (→ temporal variance), .z = reflection
// hit distance (→ distance-correct blur width). The reflection denoise (channel 1) reads
// CUR to drive a variance/history/hit-distance blur: wide where noisy/unconverged/moving,
// narrow where converged (preserves the lobe detail).
layout(set = 0, binding = 31, rgba16f) uniform writeonly image2D reflAuxWrite;
layout(set = 0, binding = 32) uniform sampler2D reflAuxPrevTex;
// Scene fog (homogeneous medium) — the SAME per-frame GpuFogUbo LidarScanner
// binds, so scene.fog drives both consumers from one source of truth.
// waterSurfaceY caps the medium (underwater fog stops at the surface).
layout(set = 0, binding = 33, scalar) uniform FogUbo {
    vec3  sigmaT;        // per-channel extinction (1/world unit)
    float enabled;       // 1.0 = fog active
    vec3  color;         // inscatter tint = single-scattering albedo
    float anisotropy;    // HG g (drives the volumetric beam phase when fog is on)
    float waterSurfaceY; // world-Y of the water surface — the MURK clip; 1e30 = unbounded
    vec3  worldUp;       // world up axis (= camera.up) for sky aerial perspective
    // Air-medium profile — declared for offset alignment with GpuFogUbo; the
    // shade pass reads the SAME values from the CloudUbo (clouds.hf*), which the
    // froxel passes also bind. Present here so the murk fields below sit at their
    // real buffer offsets.
    float hfDensity;
    float hfBaseY;
    float hfFalloff;
    // Underwater murk (setUnderwaterMurk) — a separate homogeneous medium clipped
    // to BELOW waterSurfaceY (applyMurk). 0 density = off.
    float murkDensity;   // the clearest channel of murkSigma: every gate and optical-depth budget reads this
    vec3  murkColor;
    vec3  murkSigma;     // σ_t per channel; all three equal murkDensity for the scalar murk
    float murkAmbientK;  // setUnderwaterAmbientFalloff; 0 = the ambient terms ignore depth
} fog;

// ── World-space irradiance probe grid (multi-bounce GI, on by default) ──────
// SH-L1 probes updated by probe_update.comp (ProbeGI pass). gatherEnv's GI
// rays tap this at their HIT points, adding the hit's INDIRECT irradiance —
// the bounce 2..∞ + through-the-opening sky term the 1-bounce giRadiance
// deliberately omits (enclosed interiors otherwise stay near-black). The
// grid UBO's `enabled` gates all sampling; buffers are dummies when off.
// See probe_common.glsl for the storage layout + energy accounting.
layout(set = 0, binding = 36, scalar) readonly buffer ProbeShBuf { vec4 probeSh[]; };
#define PROBE_GRID_UBO_BINDING 37
#include "probe_grid_ubo.glsl"// probeGrid: the scene-fit grid or the camera-centred cascades
// Octahedral depth store (Chebyshev visibility — see probe_common.glsl).
layout(set = 0, binding = 54, scalar) readonly buffer ProbeDepthBuf { uint probeDepth[]; };

// ── Specialization constants (DeferredShade picks the pipeline per dispatch) ──
// kSpecMask names the pc.flags bits that are compile-time in this pipeline and
// kSpecFlags carries their values; effFlags() reads every other bit from
// pc.flags, so each permutation computes exactly what the runtime kernel does.
// The G-buffer MSAA sample count (bits 5-6) is always in the mask: it is
// renderer configuration, not per-frame state, so DeferredShade builds one
// pipeline per sample count in use. That is for registers. This kernel sits on
// a register-allocation cliff, landing at 128 or 168 depending on what is live
// at its peak, and the runtime sample-count code alone puts the 1x kernel on
// the far side; after an edit to what the kernel holds live, check the default
// permutation's register count with THREEPP_VULKAN_PIPELINE_STATS=1. The rest
// are diagnostics (DeferredShade's THREEPP_VK_SHADE_* knobs): more mask bits
// make more scene toggles compile-time, so the driver can drop the code they
// turn off, and kProbeGi does the same for the probe GI gate (0 runtime, the
// UBO; 1 forced off; 2 forced on).
layout(constant_id = 0) const uint kSpecFlags = 0u;
layout(constant_id = 1) const uint kSpecMask  = 0u;
layout(constant_id = 2) const uint kProbeGi   = 0u;
// 1 = this view's shadow maps are live (ShadowMode::ShadowMaps): the sun and
// every slotted lamp take their PRIMARY visibility from the maps
// (shadow_maps.glsl), and a point no map covers traces as before (a miss
// in the no-ray-query variant). Compile-time so a ray-traced frame runs the
// kernel it always ran.
layout(constant_id = 3) const uint kShadowMaps = 0u;
const bool kSM = kShadowMaps != 0u;
bool probesEnabled() { return kProbeGi == 0u ? probeGrid.enabled > 0.5 : kProbeGi == 2u; }
// probe_common.glsl's own gate, made the same test here (its default is the UBO).
#define PROBE_GRID_OFF (!probesEnabled())

// (Binding 56 retired — it carried a removed hybrid-SSR input; 55 lives on
// above as gbufIdsPrevTex. Binding 57, the raster camera UBO, lives on for
// cloud_march.comp's temporal reprojection — declared there, unused by this
// shader.)

// ── MSAA per-sample edge shading (dispatch B, shadeMode==1 only) ───────────
// Raw MS raster attachments — bound only when setGbufferMsaa(2|4) is active;
// 1x1 dummy MS views otherwise (same "always bound, harmlessly unused"
// pattern as the ocean/foam dummy textures elsewhere in this shader).
// texelFetch(tex, px, sampleIdx) reads ONE live sample directly, bypassing
// the dominant-sample resolve gbuf_resolve.comp already did for dispatch A.
layout(set = 0, binding = 38) uniform sampler2DMS  gbufNormalMS;
layout(set = 0, binding = 39) uniform sampler2DMS  gbufDepthMS;
layout(set = 0, binding = 40) uniform usampler2DMS gbufIdsMS;
layout(set = 0, binding = 41) uniform sampler2DMS  gbufAlbedoMS;
layout(set = 0, binding = 42) uniform sampler2DMS  gbufUvMS;

// ── Denoised direct-shadow channel (ratio estimator, Heitz et al. 2018) ─────
// The analytic dir/point/spot direct lighting is SPLIT: the UNSHADOWED sum U is
// computed exactly (noise-free, → directUImage) while the shadowing is a 1-2 ray
// stochastic estimate of the luminance-weighted visibility ratio
//   R = Σ lum(c_i)·V_i / Σ lum(c_i)   ∈ [0,1]
// accumulated temporally here (shadowVisImage, reprojected like the GI) and
// à-trous-filtered by deferred_gi_filter, whose recombine adds U × R̃. R is
// bounded and firefly-free — ideal denoiser food — so the penumbra converges
// smooth instead of the static-dither grain the old always-inline path had,
// and point/spot lights gain radius-driven soft shadows at the same flat cost.
// .x = accumulated R, .y = E[R²] (temporal variance), .z = histLen, .w = trend
// (content-change antilag — a sweeping shadow edge must cut history, because
// the receiver's motion vectors can't see the OCCLUDER moving).
layout(set = 0, binding = 43, rgba16f) uniform image2D shadowVisImage;
layout(set = 0, binding = 44) uniform sampler2D shadowVisPrevTex;// prev fif (reproject history)
layout(set = 0, binding = 45, rgba16f) uniform writeonly image2D directUImage;
// The GI recombine's per-pixel demodulation colour: what deferred_gi_filter
// multiplies the filtered diffuse irradiance by (base diffuse under the sheen
// share, plus the sheen lobe, all times diffBaseShare). vec3(0) off the demod path.
layout(set = 0, binding = 77, rgba16f) uniform writeonly image2D demodColorImage;

// The view's shadow maps (bindings 80-83), read only where kSM is set.
#include "shadow_maps.glsl"

// ── Clustered lights (bindings 48/49, built by cluster_build.comp) ─────────
// ALL scene point/spot lights (power-sorted, no 8-per-type cap) + per-cell
// index lists over a 16×8×24 screen-tile × exponential-view-Z grid. The
// analytic direct split below loops its own cell's lights instead of the UBO
// arrays — the UBO's 8-per-type slots (the STRONGEST 8, same sort) remain the
// source for every other consumer (reflection hits, GI rays, volumetric
// beams, probes), where screen-space clusters can't apply.
#include "cluster_lights.glsl"// ClusterLight, the grid shape, the Z ladder
layout(set = 0, binding = 48, scalar) readonly buffer ClusterGridBuf { uint clusterGrid[]; };
layout(set = 0, binding = 49, scalar) readonly buffer ClusterLightsBuf { ClusterLight clusterLights[]; };

// ── Froxel volumetric lighting (binding 53, built by froxel_inject/integrate) ─
// Front-to-back-integrated volumetric in-scatter LUT: sun shafts, spot beams
// and point-light glow for the camera→(uv, viewDist) leg, in ONE trilinear
// sample — replaces the per-pixel volumetricSpot/DirScatter marches. flags
// bit 8 says the froxel passes ran this frame (medium present); the LUT holds
// stale data otherwise. Surface extinction stays applySceneFog's analytic
// Beer-Lambert — this LUT carries in-scatter only.
layout(set = 0, binding = 53) uniform sampler3D froxelLutTex;

#define FROXEL_GRID_LADDER_ONLY
#include "froxel_grid.glsl"// kFroxelZMin/Max: the LUT's slice ladder
// (froxelInscatter() lives just below the push-constant block — it gates on pc.flags.)

// ── Volumetric clouds (binding 58) ─────────────────────────────────────────
// Far-field raymarched cloud-layer parameters (VulkanRenderer::setClouds). A
// tiny always-bound UBO — clouds.enabled == 0 short-circuits cloudMarch(), so
// the layer costs nothing and is image-identical to today when disabled. See
// cloudMarch()/cloudDensity() in deferred_shade_60_fog_volumetrics.glsl.
#include "cloud_ubo.glsl"

// Half-res cloud march result (cloud_march.comp, binding 59). LINEAR sampler —
// the full-res shade composites it with ONE bilinear upsample tap. rgb =
// in-scatter, a = transmittance. Gated by clouds.enabled (off = col*1 + 0).
layout(set = 0, binding = 61) uniform sampler2D cloudColorTex;// composited by compositeClouds() below the push block
// This frame's cloud aux (binding 66): r=mean cloud depth, g=histLen,
// b=scene distance the march clipped against, a=epoch. The depth-aware
// upsample in cloudGather() needs r to reject taps whose cloud lies behind
// the compositing pixel's own surface.
layout(set = 0, binding = 66) uniform sampler2D cloudAuxTex;
// Half-res RTAO (rtao.comp, bindings 72-76). rgb = bent normal (n*0.5+0.5),
// a = ambient occlusion in [0,1] (1 = fully open). The full-res shade
// depth-aware bilateral-upsamples it (rtaoGather() below) and feeds ao/bentN
// into the ambient + specular-occlusion + bent-normal probe/env diffuse
// indirect. texelFetch (the depth reject reconstructs each tap's own surface).
layout(set = 0, binding = 74) uniform sampler2D rtaoTex;
// RTAO aux (the SAME storage descriptor the rtao pass writes — this pass only
// reads it via imageLoad). .b = NEAR-FIELD ao (distance-weighted: contact hits
// darken fully, occluders near AO_RADIUS barely) — the AO applied to the probe
// indirect, whose own Chebyshev depth already carries far-field visibility.
// .a = far-field sky visibility (enclosure).
layout(set = 0, binding = 75, rgba16f) readonly uniform image2D rtaoAuxImage;
// Cloud shadow map (cloud_shadow.comp, binding 65) — top-down cloud
// transmittance over an 8 km camera-centred square. Sampled by cloudShadowSample()
// (deferred_shade_10_lighting_utils.glsl) to attenuate the sun on the ground.
layout(set = 0, binding = 65) uniform sampler2D cloudShadowTex;

// ParticleField density volumes (bindings 67/68 + the r16f mirrors at 69,
// declared because PD_LINEAR is defined above). applyParticleFog marches the
// r16f volumes per pixel — world-anchored, hardware trilinear — which is what
// keeps dust view-independent and sharper than the froxel grid.
#include "particle_density.glsl"

// Splat reflection volumes (bindings 70/71) — the SECOND density table, baked
// once per SplatCloud by SplatPass and marched by svLeg on the traced
// reflection legs only. Never on the primary leg: SplatPass composites the real
// thing there (see the header for the double-count this avoids).
#include "splat_volume.glsl"

#include "probe_common.glsl"// probeIrradiance()

// Shared ShadePush block (vulkan_shared.h) — field docs + the flags bit
// table live there. shadeMode == 1 is dispatch B (MSAA only): per-sample
// shading at COMPLEX (edge) pixels — see main()'s top (tap selection) and
// bottom (imageStore gating) for the shadeMode == 1 branches.
layout(push_constant) uniform Pc { ShadePush pc; };
// Every pc.flags read in the shade goes through these (see kSpecMask above).
uint effFlags() { return (pc.flags & ~kSpecMask) | (kSpecFlags & kSpecMask); }
bool flagOn(uint bits) { return (effFlags() & bits) != 0u; }

// The geometric normal for the probe lookups (depth_normal.glsl): the MSAA
// sample the centre came from, or the resolved G-buffer.
float dnDepth(ivec2 q, int s) { return s >= 0 ? texelFetch(gbufDepthMS, q, s).x : texelFetch(gbufDepthTex, q, 0).x; }
uint  dnId(ivec2 q, int s) { return s >= 0 ? texelFetch(gbufIdsMS, q, s).x : texelFetch(gbufIdsTex, q, 0).x; }
#include "depth_normal.glsl"

// DEPTH-AWARE 2×2 upsample of the half-res cloud layer at this full-res pixel.
// A plain bilinear tap bleeds across depth discontinuities: a full-res pixel
// on a thin foreground object (a ship's mast) blends half-res texels whose
// centres missed it and marched cloud kilometres BEHIND it — painting cloud
// onto the object and haloing its silhouette. Each tap therefore only counts
// when its cloud is actually IN FRONT of this pixel's surface (aux.r mean
// cloud depth < sceneDist); taps carrying no cloud are the identity and are
// always safe. All four rejected → identity (cloud fully behind the surface).
// Sky pixels pass sceneDist=1e30 → nothing rejected → plain bilinear.
// `meanDist` returns the coverage-weighted mean cloud DISTANCE (aux.r, ray units)
// over the accepted cloud-bearing taps — sceneDist when no tap carries cloud —
// so compositeClouds can fog the cloud in-scatter over the camera→cloud leg.
vec4 cloudGather(ivec2 px, float sceneDist, out float meanDist) {
    // Cloud march runs one thread per half-PANE pixel ((pc.width+1)/2 x ...),
    // so the valid content spans the pane's half extent at the image origin —
    // NOT the physical image extent (fullscreen the two coincide). Mapping by
    // textureSize here would stretch the lookup across the whole image under a
    // scissored pane render (see paneToPhys below).
    const ivec2 hsz = ivec2(int(pc.width + 1u) / 2, int(pc.height + 1u) / 2);
    const vec2  uv  = (vec2(px) + 0.5) / vec2(float(pc.width), float(pc.height));
    const vec2  pos = uv * vec2(hsz) - 0.5;
    const ivec2 b0  = ivec2(floor(pos));
    const vec2  f   = pos - vec2(b0);
    const vec4  wb  = vec4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y),
                           (1.0 - f.x) * f.y,         f.x * f.y);
    vec4  acc  = vec4(0.0);
    float wSum = 0.0;
    float dAcc = 0.0;// coverage-weighted mean cloud distance over cloud-bearing taps
    float dW   = 0.0;
    for (int i = 0; i < 4; ++i) {
        const ivec2 tp = clamp(b0 + ivec2(i & 1, i >> 1), ivec2(0), hsz - 1);
        const vec4  c  = texelFetch(cloudColorTex, tp, 0);
        const bool hasCloud = (c.a < 0.999) || any(greaterThan(c.rgb, vec3(1e-5)));
        float cd = 0.0;
        if (hasCloud) {
            cd = texelFetch(cloudAuxTex, tp, 0).r;// aux.r = mean cloud depth (ray distance)
            if (cd >= sceneDist) continue;        // cloud behind THIS pixel's surface → reject
        }
        acc  += wb[i] * c;
        wSum += wb[i];
        if (hasCloud) { dAcc += wb[i] * cd; dW += wb[i]; }
    }
    meanDist = (dW > 1e-4) ? dAcc / dW : sceneDist;
    return (wSum > 1e-4) ? acc / wSum : vec4(0.0, 0.0, 0.0, 1.0);
}
// Forward decls — DEFINED later via the deferred_shade_*.glsl includes below.
// compositeClouds is declared here (before those includes) but only CALLED from
// main(), which is after them, so the definitions link in the same translation
// unit. cam/clouds/fog/lights UBOs are already declared above.
float heightFogOpticalDepth(vec3 a, vec3 b);// deferred_shade_60_fog_volumetrics.glsl
vec3  sampleEnvLod(vec3 dir, float lod);
vec3  sampleEnvDiffuse(vec3 n);    // deferred_shade_10_lighting_utils.glsl
vec3  sampleBackground(vec3 dir);           // deferred_shade_10_lighting_utils.glsl
// Composite the upsampled half-res cloud layer (binding 61) over `col` at this
// pixel. sceneDist = this pixel's own camera→surface distance (1e30 for sky).
// A no-op when clouds are off, so the whole system is free/image-identical
// when unused. rgb = in-scatter, a = transmittance.
vec3 compositeClouds(vec3 col, ivec2 px, float sceneDist) {
    if (clouds.enabled < 0.5) return col;
    float meanDist;
    const vec4 cl = cloudGather(px, sceneDist, meanDist);
    // Composite the cloud exactly as before (byte-identical) when EITHER:
    //   • no air-fog medium is present (nothing to fog), OR
    //   • no cloud tap contributed here, so cloudGather returned meanDist ==
    //     sceneDist (the sentinel). On a clear-SKY pixel sceneDist is 1e30, and
    //     building camP+dir·1e30 would feed a 1e30 leg into heightFogOpticalDepth:
    //     distance()² = 1e60 overflows fp32 → Inf, and Inf·0 → NaN when the closed
    //     form's f underflows for a grazing sky ray ⇒ exp(-NaN)=NaN ⇒ the whole SKY
    //     goes BLACK (angle-dependent, clouds-gated — the observed regression). With
    //     no cloud gathered there is also no cloud radiance to fog, so this early
    //     out is the exact result, not an approximation.
    if (clouds.hfDensity <= 0.0 || meanDist >= sceneDist) return col * cl.a + cl.rgb;
    // ── Fog the cloud IN-SCATTER over the camera→cloud leg ───────────────────
    // The background `col` already carries its OWN full-path fog (applySkyFog for
    // sky pixels, applyHeteroSurfaceFog for terrain/water) and composites through
    // the cloud transmittance cl.a untouched. Only the cloud's own radiance is
    // attenuated here — by the closed-form height-fog transmittance camera→cloud
    // (the froxel LUT is NOT valid at the SKY pixels where clouds most often
    // composite, so the analytic optical depth — the SAME one applyHeteroSurfaceFog
    // integrates over the whole leg — stands in). The mist in front of the cloud
    // also ADDS its haze, but only over the cloud-COVERED fraction (1-cl.a); the
    // transparent fraction's haze already lives in `col`. This mirrors
    // applyHeteroSurfaceFog's col*T + medAlbedo*fogLight*(1-T), so cloud, distant
    // terrain and the fogged sky converge to ONE haze. fogT==1 when the medium is
    // off (short-circuited above) → byte-identical col*cl.a + cl.rgb.
    const vec2  uvp   = (vec2(px) + 0.5) / vec2(float(pc.width), float(pc.height));
    const vec2  ndcp  = vec2(uvp.x * 2.0 - 1.0, -(uvp.y * 2.0 - 1.0)) + cam.jitter.xy;
    const vec3  dirWS = camRayDir(ndcp);
    const vec3  camP  = camRayOrigin(ndcp);
    // Clamp the camera→cloud leg defensively: cloud_march caps mean cloud depth at
    // 42 km, but a finite bound here means no future far-layer depth can overflow
    // the closed form regardless (heightFogOpticalDepth also self-clamps len + od).
    const vec3  cloudP = camP + dirWS * min(meanDist, 6.0e4);
    const float fogT  = exp(-heightFogOpticalDepth(camP, cloudP));
    const vec3  fogLight  = lights.ambient
                          + sampleEnvLod(vec3(0.0, 1.0, 0.0), float(max(pc.envMipCount, 1u) - 1u));
    const vec3  medAlbedo = (fog.enabled > 0.5) ? fog.color : vec3(1.0);
    return col * cl.a
         + cl.rgb * fogT
         + medAlbedo * fogLight * ((1.0 - fogT) * (1.0 - cl.a));
}

// Set by shadowVis (in _10) when a committed shadow hit is a currently-MOVING
// mesh (sticky GeometryDesc._pad): the receiver is under a SWEEPING shadow, so
// the denoised shadow channel keeps its history short (statistics-free antilag —
// see the sCap composition in the shadow temporal). Reset before, and captured
// right after, the analyticDirectSplit visibility rays; later inline shadowVis
// calls (reflection hits, translucency) don't pollute the captured value.
bool gShadowMovingOccluder = false;
// True when analyticDirectSplit replaced this pixel's visEst with the 16-ray
// static-dithered sunShadowVis estimate (dwell/release top-up). The temporal
// may then drop to cap 1 (pure per-frame — the denoise-off estimator, per-
// frame stable by construction): with only 1-2 binary rays a cap-1 mean is
// salt-and-pepper, so without the top-up the dwell floor stays at 2.
bool gShadowSunTopUp = false;
// analyticDirectSplit: every light of the ratio came from a shadow map (kSM),
// so visEst is exact, not an estimate.
bool gShadowMapOnly = false;
// Set per-ray by giRadiance when the 1-bounce hit lands on a currently-MOVING
// mesh (sticky GeometryDesc._pad); gatherEnv folds the RAY-COUNT fraction into
// gGiMovFrac. Ray-count, not luminance-weighted — a dark occlusion hit (the
// car blocking a road pixel's sky = the soft AO blob around the car)
// contributes no luminance but fully stale history. The GI temporal cuts its
// cap while the fraction is significant (dwell), which also fixes the
// departure side by construction: the deposit never gets deep, so the ground
// the car just left recovers in ~3 frames instead of dragging the contact
// darkening as a trailing smear (the artifact the RT-AO/GI toggle isolates;
// the luminance-trend antilag alone is too weak here — the 1-spp gather's
// noise drowns the trend signal that works for the low-variance channels).
bool  gGiRayHitMoved = false;
float gGiMovFrac     = 0.0;

// ── Pane-UV → physical-UV rescale (split-screen) ────────────────────────────
// Under a scissored pane render (setScissorTest + a sub-rect viewport) the
// deferred pipeline renders PANE-SIZED content at the ORIGIN of the same
// full-render-extent images used fullscreen. Every screen-space normalized
// coordinate in this shader family ([0,1] over pc.width/height) must therefore
// be rescaled by pane/physical before it touches a sampler, or the tap lands
// at a stretched position — the visible symptom was shadow/GI history ghosts
// offset from their casters in lidar_slam's split view. Fullscreen the ratio
// is exactly 1.0 (w/w), so this is a no-op there. Applies ONLY to samplers
// backed by render-extent images (gbuf*, *Prev, history); env/noise/froxel/
// cloud-shadow textures live in their own spaces and take their uv raw.
vec2 paneToPhys(sampler2D t, vec2 paneUv) {
    return paneUv * vec2(float(pc.width), float(pc.height)) / vec2(textureSize(t, 0));
}

#include "emissive_lights.glsl"// emitter light table + coverage sample plan (the NEE loops below)
// DEPTH-AWARE 2×2 bilateral upsample of the half-res RTAO at this full-res
// pixel (mirrors cloudGather). A plain bilinear tap bleeds occlusion across
// depth discontinuities — a dark halo hugging every silhouette. Each tap is
// reconstructed to its OWN surface view-Z from the full-res depth and rejected
// when it disagrees with this pixel's surface; all four rejected (thin
// geometry between the half-res taps) → fully open (ao=1), the safe identity.
void rtaoGather(ivec2 px, float centerViewZ, vec3 N,
                out float ao, out float aoNear, out float skyVis, out vec3 bentN) {
    const ivec2 hsz = textureSize(rtaoTex, 0);
    const vec2  size = vec2(float(pc.width), float(pc.height));
    const vec2  uv  = (vec2(px) + 0.5) / size;
    const vec2  pos = uv * vec2(hsz) - 0.5;
    const ivec2 b0  = ivec2(floor(pos));
    const vec2  f   = pos - vec2(b0);
    const vec4  wb  = vec4((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y),
                           (1.0 - f.x) * f.y,         f.x * f.y);
    float aoAcc = 0.0, nearAcc = 0.0, skyAcc = 0.0; vec3 bnAcc = vec3(0.0); float wSum = 0.0;
    for (int i = 0; i < 4; ++i) {
        const ivec2 tp   = clamp(b0 + ivec2(i & 1, i >> 1), ivec2(0), hsz - 1);
        // Each half-res tap is ANCHORED to full-res texel tp*2 (rtao.comp's
        // convention) — reconstruct the tap's surface from THAT texel's own
        // depth + NDC so the reject compares the surface the AO was actually
        // traced from (a half-pixel mismatch reads grazing floors as
        // depth-inconsistent and rejects everything).
        const ivec2 tf   = min(tp * 2, ivec2(int(pc.width) - 1, int(pc.height) - 1));
        const vec2  tuv  = (vec2(tf) + 0.5) / size;
        const vec2  tndc = vec2(tuv.x * 2.0 - 1.0, -(tuv.y * 2.0 - 1.0)) + cam.jitter.xy;
        const float td   = texelFetch(gbufDepthTex, tf, 0).x;
        const vec4  tvh  = cam.projInverse * vec4(tndc, td, 1.0);
        const float tvz  = tvh.z / tvh.w;
        if (abs(tvz - centerViewZ) > 0.1 * max(abs(centerViewZ), 1e-3)) continue;
        const vec4 r   = texelFetch(rtaoTex, tp, 0);
        const vec4 aux = imageLoad(rtaoAuxImage, tp);// (histLen, E[ao^2], aoNear, skyVis)
        aoAcc   += wb[i] * r.a;
        nearAcc += wb[i] * clamp(aux.b, 0.0, 1.0);
        skyAcc  += wb[i] * clamp(aux.a, 0.0, 1.0);
        bnAcc   += wb[i] * (r.xyz * 2.0 - 1.0);
        wSum    += wb[i];
    }
    if (wSum > 1e-4) {
        ao     = aoAcc / wSum;
        aoNear = nearAcc / wSum;
        skyVis = skyAcc / wSum;
        bentN  = (dot(bnAcc, bnAcc) > 1e-8) ? normalize(bnAcc) : N;
    } else {
        ao = 1.0; aoNear = 1.0; skyVis = 1.0; bentN = N;
    }
}

// Jiménez et al. (GTAO 2016) multi-bounce approximation: a per-channel curve
// that lifts the AO term in bright-albedo regions (light bounces more in a
// white room) so corners don't over-darken. Returns a colour ∈ [ao..1] used in
// place of the scalar ao on the diffuse indirect.
vec3 gtaoMultiBounce(float visibility, vec3 albedo) {
    const vec3 a =  2.0404 * albedo - 0.3324;
    const vec3 b = -4.7951 * albedo + 0.6417;
    const vec3 c =  2.7552 * albedo + 0.6903;
    return max(vec3(visibility), ((visibility * a + b) * visibility + c) * visibility);
}

// Frostbite specular occlusion (Lagarde & de Rousiers 2014): derive a specular
// AO from the diffuse AO + view/roughness so glossy IBL is occluded in
// cavities without a separate bent-cone trace. Rough surfaces keep more, mirror
// surfaces almost none (they sample a tight cone).
float specularOcclusion(float NdotV, float ao, float roughness) {
    return clamp(pow(max(NdotV + ao, 0.0), exp2(-16.0 * roughness - 1.0)) - 1.0 + ao, 0.0, 1.0);
}

// FORWARD DECLARATION, and the include order is the whole reason for it. The
// dapple (caustics on submerged SURFACES) belongs with the shafts' own crest
// proxy in _60 — same constants, same refraction, one place documenting that it
// is a proxy and not a focusing determinant — but the two direct-light loops it
// has to multiply live in _20 (analyticDirectSplit, the denoised-shadow path)
// and _30 (shadeDiffuseDirect, the inline path). A prototype here is cheaper
// than splitting the murk constants across three files to satisfy an ordering.
// Returns EXACTLY 1.0 with no murk, no ocean, or a point above the surface.
float murkSunCaustic(vec3 P, vec3 L);
// Its sibling: bends L to the in-water direction for a submerged P and returns
// the in-water leg transmittance (1.0, L untouched, everywhere else).
// Per channel: a murk with setUnderwaterMurk's per-channel extinction takes the
// red out of the sun first.
vec3 murkSunLeg(vec3 P, inout vec3 L);
// gi_bounce.glsl's directional-light leg at a GI hit: the murk leg above.
vec3 giSunLeg(vec3 P, inout vec3 L) { return murkSunLeg(P, L); }
// The same for the light that has no direction: `amb` is an environment,
// AmbientLight or HemisphereLight term arriving at a submerged P about the
// direction `dir` (a normal for a diffuse term, the mirror direction for a
// specular one). Returns it as the water column over P leaves it, plus the
// water's own light for what the column took. `amb` itself, untouched, with
// setUnderwaterAmbientFalloff at 0 (the default), no murk, or P at or above
// the surface.
vec3 murkAmbient(vec3 amb, vec3 P, vec3 dir);

#include "pcg_rng.glsl"
#include "deferred_shade_10_lighting_utils.glsl"
#include "deferred_shade_20_restir_di.glsl"
#include "deferred_shade_30_gi_gather.glsl"
#include "deferred_shade_40_reflections.glsl"
#include "deferred_shade_50_water_glass.glsl"
#include "deferred_shade_60_fog_volumetrics.glsl"
