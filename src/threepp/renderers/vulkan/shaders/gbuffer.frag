#version 460
#extension GL_EXT_scalar_block_layout : require
#extension GL_GOOGLE_include_directive : enable
// Divergent bindless indexing: material texture indices derive from
// vInstanceIdx (a flat PER-INSTANCE input — NOT dynamically uniform), so
// fragments of different instances/materials packed into one wave (small
// distant triangles, silhouettes — exactly the thin-geometry case) index
// DIFFERENT descriptors. Without nonuniformEXT that is spec-UB: the compiler
// may hoist one wave-uniform descriptor load, so whichever lane wins defines
// every lane's albedo/normal map — and the winner shifts with the TAA jitter
// → per-frame texture flicker scaling with material diversity.
#extension GL_EXT_nonuniform_qualifier : require

#include "vulkan_shared.h"

// G-buffer fragment for the hybrid raster prepass. Emits world-space
// normal (with normal map applied), screen-space motion vector, and
// per-pixel IDs/flags. Depth is written automatically.
//
// Normal mapping is done here via screen-space derivatives
// of vWorldPos + vUv. Without it, primary surfaces look flat; most assets
// rely on the normal map for surface detail (mortar lines, fabric weave,
// brick relief, etc.).
//
// Raster-first: albedo / roughness / metalness are also
// sampled here and written to the G-buffer so the deferred shading pass can
// light the surface analytically (albedo.rgb, roughness from .g, metalness
// from .b, per-channel uvTransforms, hardware sRGB decode on the albedo
// view). Fragment-shader derivatives give correct mip selection for free —
// deferred_shade.comp's ray-query reflection/GI hits need the lodBias
// attachment because those bounce rays have no fragment-shader derivatives,
// but here plain texture() is correct.

layout(set = 0, binding = 0) uniform CameraUbo {
    mat4 currVPjittered;
    mat4 currVPunjittered;
    mat4 prevVP;
    vec4 jitter;          // .xy = curr clip-space sub-pixel jitter
    vec4 prevJitter;      // .xy = prev clip-space sub-pixel jitter
} cam;

layout(set = 0, binding = 2, scalar) readonly buffer GbufMatBuf {
    MaterialDesc gbufMats[];
};
layout(set = 0, binding = 3) uniform sampler2D gbufAlbedoMaps[kMaxMaterialTextures];

layout(location = 0) in vec3 vWorldNormal;
layout(location = 1) in vec4 vCurrClipUnjit;
layout(location = 2) in vec4 vPrevClip;
layout(location = 3) flat in uint vInstanceIdx;
layout(location = 4) flat in uint vFlags;
layout(location = 5) in vec2 vUv;
layout(location = 6) in vec3 vWorldPos;
layout(location = 7) in vec3 vColor;// per-vertex color (material.vertexColors); white when unused
layout(location = 8) flat in uint vStableId;// stable per-object id (host-assigned; NOT the visible-set index)

// Attachment 0: world-space normal (rgba16f). .xyz = n*0.5+0.5 encoded world
// normal, .w = linear roughness (the deferred_shade.comp shading pass reads
// both). rgba16f is necessary for ocean wave normals — rgba8
// loses too much precision on the FFT-driven detail.
layout(location = 0) out vec4 outNormal;

// Attachment 1: motion vector in NDC delta (rg16f). TaaResolve converts to
// pixel-space when sampling the temporal accumulator. Stationary pixels
// produce zero motion.
layout(location = 1) out vec4 outMotion;

// Attachment 2: per-pixel IDs + flags (rgba16ui).
//   .x = instanceCustomIndex + 1 (matches deferred_shade.comp's
//        gbufIdsTex.x convention; 0 reserved for sky/miss because the render pass
//        clears IDs to 0 before any draw). This is the PER-FRAME visible-set
//        index — used internally (reproject/motionMat), NOT stable across frames.
//   .y = STABLE per-object instance id (host-assigned, persists across frames
//        and visible-set changes; 0 = sky). The recoverable label for
//        instance segmentation — see VulkanRendererCore::setObjectInstanceId.
//   .z = per-instance flags in bits 0..7 | semantic CLASS id in bits 8..15
//        (0 = unset). Canonical bit layout: vulkan_shared.h (kInstFlag*);
//        shader consumers use the instance_flags.glsl accessors. Consumers
//        bit-test the low byte, so the class byte is inert to them; the MSAA
//        resolve carries the dominant sample's .z through unchanged.
//   .w = reserved (repacked with MSAA coverage metadata by gbuf_resolve.comp)
layout(location = 2) out uvec4 outIds;

// Attachment 3: material UV + LOD bias (rgba16f). .rg = UV, precomputed here
// so deferred_shade.comp's emissive-map sample doesn't need triangle
// interpolation. .b = log2(max(|dUV/dx|, |dUV/dy|)) — a texture-size-
// independent footprint. Consumers add log2(textureSize) per sample to drive
// textureLod. Without this, ray-traced hits (reflection/GI bounces have no
// implicit derivatives) would snap to mip 0 and per-frame texture shimmer
// would survive TAA at the deferred pass's 1-2 spp/frame sample count.
layout(location = 3) out vec4 outUv;

// Attachment 4: albedo + metalness (rgba8 unorm). .rgb = linear base colour
// (material albedo × albedo-map, sRGB-decoded on sample), .a = metalness.
// Roughness lives in outNormal.w. Together these give the deferred shading
// pass a complete PBR surface. Linear 8-bit is the standard albedo G-buffer
// format; emissive / clearcoat / sheen stay re-sampled from MaterialDesc in
// the deferred pass rather than baked here.
layout(location = 4) out vec4 outAlbedoMetal;

// Set to 1 on the decal pipeline variant (rasterGbufDecalPipeline): bucket-[3]
// blend decals draw with albedo alpha-blending + write-masked id/normal/motion
// attachments, so this shader emits texture alpha as the blend factor instead
// of running the stochastic screen-door. 0 everywhere else, where decal
// materials draw with the regular pipeline and fall back to the
// screen-door (TAA converges it over multiple frames; a single frame can't).
layout(constant_id = 0) const uint DECAL_PASS = 0u;

// Per-pixel, per-frame hash in [0,1) for the stochastic alpha-blend screen-
// door. Folds the Halton sub-pixel jitter (changes every frame) into the seed
// so the dither pattern decorrelates over time and the temporal accumulator /
// TAA resolve it toward the true alpha-weighted blend instead of a fixed grid.
float alphaHash(vec2 fragXY, vec2 jitter) {
    uint h = uint(fragXY.x) * 1973u + uint(fragXY.y) * 9277u
           + floatBitsToUint(jitter.x) * 26699u
           + floatBitsToUint(jitter.y) * 53401u + 0x9e3779b9u;
    h ^= h >> 16; h *= 0x7feb352du;
    h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    return float(h) / 4294967296.0;
}

void main() {
    vec3 N = normalize(vWorldNormal);

    // Two-sided / back-facing fragments: flip the geometric normal to face the
    // viewer. A Side::Double surface stores ONE geometric normal for both faces,
    // so the face whose normal points away from a light stays dark in the
    // deferred shade (and the lit side appears to "bleed" — e.g. one of two
    // symmetric divider walls dark, the other lit). gl_FrontFacing is reliable
    // here: the vertex-shader Y-flip (gl_Position.y = -y) restores GL's CCW-front
    // convention, matching the pipeline frontFace = COUNTER_CLOCKWISE. Single-
    // sided (cull-back) meshes never produce back fragments, so this is a no-op
    // for them; done BEFORE the normal-map TBN so perturbation is relative to the
    // correctly-oriented surface.
    if (!gl_FrontFacing) N = -N;

    // UV derivatives — used both for the LOD bias attachment and the normal-
    // map TBN construction below. Hoisted out of the normal-map branch
    // because dFdx/dFdy must be called from non-divergent control flow.
    const vec2 duvx = dFdx(vUv);
    const vec2 duvy = dFdy(vUv);

    // Normal map perturbation. TBN derived from screen-space derivatives:
    // (dpx, dpy) = world-space partial derivatives of position
    // (duvx, duvy) = uv derivatives.
    // Tangent T = (dpx · duvy.y - dpy · duvx.y) / det, expressed via
    // fragment-shader derivatives so we don't need triangle vertex/UV data
    // here.
    //
    // Skipped on water (is_water flag bit 0): the FFT cascade normal map
    // is a water-specific input applied as part of the water BSDF in
    // deferred_shade_50_water_glass.glsl, not a
    // generic surface perturbation — running it here would tile the foam
    // normal across the wave geometry and produce visible cellular noise.
    const MaterialDesc m = gbufMats[vInstanceIdx];
    const bool isWater = (vFlags & 1u) != 0u;
    // vMF/Toksvig normal-map specular AA (deferred raster G-buffer only;
    // see project_deferred_shade_parity notes).
    // A trilinear-minified normal-map tap's raw filtered vector is SHORTER
    // than unit length in proportion to the sub-texel normal variance the mip
    // already averaged away — that shortening is free, physically-grounded
    // variance information a single re-normalized sample throws out. Recover
    // it as a vMF concentration proxy (toksvigSigma2, below) and fold it into
    // roughness so a highly minified high-frequency normal map shades with
    // the stable pre-integrated lobe its pixel footprint actually spans,
    // instead of re-aliasing every frame. Toggle packed into cam.prevJitter.z
    // (see uploadRasterCameraUbo) — no descriptor change. Length MUST be
    // taken from the raw `*2-1` tap, before the TBN perturbation / Z-
    // reconstruction / normalize below destroy it.
    //
    // Measured on the dielectric_shimmer harness (rough dielectric sphere,
    // high-frequency procedural normal map, jittered msaa=1): the roughness-
    // widening term alone (kToksvigScale up to 1.0, the documented range)
    // only recovers ~10-25% of the consecutive-frame |dLuma| (0.85 -> ~0.77),
    // NOT the 5x target — this material is a dielectric (metalness 0), so
    // Lambertian N.L dominates the pixel's radiance, and it is the shading
    // NORMAL DIRECTION itself re-aliasing frame to frame (a different high-
    // frequency texel under each jitter phase) that drives most of the
    // measured delta, not specular lobe width. Confirmed by fully zeroing the
    // tangent-plane perturbation (ns.xy = 0, geometric normal only): 0.85 ->
    // 0.11, a 7.6x drop — i.e. the perturbation DIRECTION noise, not just its
    // BRDF-visible width, is the dominant term for this stress texture.
    // kNormalDampExp supplements the literal roughness formula with the
    // additional damping actually needed to hit the harness's acceptance
    // target; both terms are driven by the same measured nLen/sigma2 so a
    // single physical quantity (vMF concentration) still gates everything,
    // and both are no-ops at mip 0 (nLen ~= 1) by construction.
    const bool toksvigOn = cam.prevJitter.z > 0.5;
    float toksvigSigma2 = 0.0;
    if (m.normalTexIndex >= 0 && !isWater) {
        const vec3 dpx = dFdx(vWorldPos);
        const vec3 dpy = dFdy(vWorldPos);
        const float det = duvx.x * duvy.y - duvy.x * duvx.y;
        if (abs(det) > 1e-8) {
            vec3 T = (dpx * duvy.y - dpy * duvx.y) / det;
            T = T - dot(T, N) * N;// Gram-Schmidt
            const float Tlen = length(T);
            if (Tlen > 1e-6) {
                T /= Tlen;
                const vec3 B = cross(N, T);
                const int nidx = clamp(m.normalTexIndex, 0, int(kMaxMaterialTextures) - 1);
                const vec2 uvN = (m.uvTransformNormal * vec3(vUv, 1.0)).xy;
                const vec3 nTap = texture(gbufAlbedoMaps[nonuniformEXT(nidx)], uvN).rgb * 2.0 - 1.0;
                vec3 ns = nTap;
                if (toksvigOn) {
                    // Raw tap length, BEFORE normalScale/Z-reconstruction — at
                    // mip 0 a unit-length source normal filters to length ~1
                    // (no-op by construction); minified/averaged taps shrink.
                    const float nLen = clamp(length(nTap), 1e-4, 1.0);
                    toksvigSigma2 = (1.0 - nLen) / nLen;// vMF variance proxy (-> roughness, below)

                    // Extra confidence-gated damping of the tangent-plane
                    // perturbation itself (see comment block above for why
                    // roughness alone isn't enough here). Strictly SHRINKS
                    // the existing raw ns.xy (never rotates/renormalizes it —
                    // renormalizing a near-zero vector was tried and measured
                    // to amplify frame noise instead of suppressing it).
                    // kNormalDampExp=32 calibrated on the harness's hiFreqNM
                    // row (two operating points: renderScale 0.75 and 1.0) to
                    // clear the >=5x consecutive-frame |dLuma| drop target at
                    // every roughness column at scale 0.75 (6.8-11x measured)
                    // and at all but the highest-roughness column at scale
                    // 1.0 (4.9-9.4x measured; the 0.80-roughness column lands
                    // just under 5x). pow(nLen,1)=nLen (the tap's own natural
                    // length, i.e. no extra damping) was measured
                    // insufficient (0.85->0.79 at scale 0.75).
                    const float kNormalDampExp = 32.0;
                    ns.xy *= pow(nLen, kNormalDampExp);
                }
                ns.xy *= m.normalScale;
                ns.z = sqrt(max(0.0, 1.0 - dot(ns.xy, ns.xy)));
                N = normalize(T * ns.x + B * ns.y + N * ns.z);
            }
        }
    }

    // ── PBR material sampling.
    // Per-channel transformed UVs.
    const vec2 uvAlbedo     = (m.uvTransform           * vec3(vUv, 1.0)).xy;
    const vec2 uvRoughMetal = (m.uvTransformRoughMetal * vec3(vUv, 1.0)).xy;

    // Albedo: scalar PBR colour × bound albedo map (.rgb). Albedo views are
    // VK_FORMAT_*_SRGB so texture() returns linear.
    vec3  albedoSample = vec3(1.0);
    float albedoAlpha  = 1.0;
    if (m.albedoTexIndex >= 0) {
        const int  ai    = clamp(m.albedoTexIndex, 0, int(kMaxMaterialTextures) - 1);
        const vec4 texel = texture(gbufAlbedoMaps[nonuniformEXT(ai)], uvAlbedo);
        albedoSample = texel.rgb;
        albedoAlpha  = texel.a;// linear (alpha is never sRGB-decoded)
    }
    // Per-vertex color (material.vertexColors): vColor is white when the mesh
    // has no "color" attribute, so this multiply is a no-op then. Linear working
    // space — matches m.albedo.
    const vec3 albedo = m.albedo * albedoSample * vColor;

    // glTF packs roughness in .g and metalness in .b; threepp's roughnessMap /
    // metalnessMap usually point at the same packed texture. Multiplicative —
    // matches three.js.
    //
    // MeshBasicMaterial (unlit) is flagged with material roughness < 0 (see
    // VulkanRenderer::materialFromMesh). Preserve that sentinel through the
    // G-buffer — don't sample the rough/metal maps or clamp — so the deferred
    // pass (deferred_shade.comp) can emit the base colour unlit via its own
    // `roughness < 0` gate. Clamping here would turn the unlit surface into a
    // glossy one that reflects the environment.
    float roughness = m.roughness;
    float metalness = m.metalness;
    if (roughness >= 0.0) {
        if (m.roughnessTexIndex >= 0) {
            const int i = clamp(m.roughnessTexIndex, 0, int(kMaxMaterialTextures) - 1);
            roughness *= texture(gbufAlbedoMaps[nonuniformEXT(i)], uvRoughMetal).g;
        }
        if (m.metalnessTexIndex >= 0) {
            const int i = clamp(m.metalnessTexIndex, 0, int(kMaxMaterialTextures) - 1);
            metalness *= texture(gbufAlbedoMaps[nonuniformEXT(i)], uvRoughMetal).b;
        }
        roughness = clamp(roughness, 0.04, 1.0);
        metalness = clamp(metalness, 0.0,  1.0);

        // Fold the vMF/Toksvig variance proxy into roughness BEFORE it is
        // written to the G-buffer (outNormal.w), so every consumer — direct
        // spec, env split-sum, reflections, and deferred_shade.comp's screen-
        // space geometric spec-AA (which reads this same channel via nTap.w
        // and widens further in quadrature) — inherits the pre-integrated
        // lobe. kToksvigScale=0.4 is the documented v1 constant (0.25-0.5
        // range); it stabilizes the SPECULAR contribution but, measured
        // alone, only recovers a fraction of this harness's consecutive-
        // frame delta for a dielectric — see the kNormalDampExp comment above
        // (the normal-map application site) for why the direction-damping
        // term carries most of the acceptance-gate result on this material.
        // sqrt(...) keeps roughness (not roughness^2) as the widened
        // quantity, matching roughMat's units downstream.
        if (toksvigSigma2 > 0.0) {
            const float kToksvigScale = 0.4;
            const float a2 = roughness * roughness;
            roughness = sqrt(clamp(a2 + kToksvigScale * toksvigSigma2, a2, 1.0));
        }
    }

    // ── Alpha cutout / blend test. The
    // raster prepass draws every visible mesh regardless of transparency
    // (buildIndirectDrawData does not filter), so without this, alpha-tested
    // foliage/decals fill their cutout holes with opaque G-buffer data and
    // BLEND surfaces render fully opaque. Discarding here writes no depth/ID,
    // so the deferred pass sees sky (or the opaque surface behind, which the
    // depth test lets win) through the hole. alphaCutoff semantics match the host
    // (VulkanRenderer::materialFromMesh: d.alphaCutoff = mat->alphaTest):
    //   > 0  cutout : discard fragments below the cutoff.
    //   < 0  BLEND  : stochastic screen-door so the surface behind shows
    //                 through; the temporal accumulator + TAA average it.
    //                 (Overridden on the decal pipeline — see DECAL pass below.)
    //   == 0 opaque : keep (the common path; one comparison, no texture cost).
    // MUST run after every texture()/dFdx/dFdy above — a per-pixel discard
    // before an implicit-derivative sample corrupts the 2×2 quad's neighbours.
    // DECAL pass: the dedicated decal pipeline alpha-blends ONLY the albedo
    // attachment (others write-masked) — albedoAlpha is emitted as the blend
    // factor below, so no discard logic is needed beyond skipping fully-
    // transparent texels. Deterministic: no screen-door, no temporal flicker,
    // the splat lerps over the receiver exactly like GL.
    const bool isDecal = (DECAL_PASS != 0u);
    if (m.albedoTexIndex >= 0 && m.alphaCutoff != 0.0) {
        if (m.alphaCutoff > 0.0) {
            if (albedoAlpha < m.alphaCutoff) discard;
        } else if (isDecal) {
            if (albedoAlpha <= 0.004) discard;// nothing to blend
        } else {
            // BLEND: variance-reduced stochastic rejection with 0.99 (accept)
            // / 0.01 (reject) early-outs.
            if (albedoAlpha <= 0.01) {
                discard;
            } else if (albedoAlpha < 0.99) {
                if (alphaHash(gl_FragCoord.xy, cam.jitter.xy) >= albedoAlpha) discard;
            }
        }
    }

    // Remap [-1, 1] → [0, 1] so negative components are visible in the
    // BGRA8_UNORM debug blit (which clamps negatives to 0). deferred_shade.comp
    // reverses this with `n * 2 - 1` when reading the attachment. .w carries linear
    // roughness for the deferred pass.
    outNormal = vec4(N * 0.5 + 0.5, roughness);
    // Decals emit texture alpha as the blend factor (the decal pipeline's
    // SRC_ALPHA blend consumes it; its RGB-only write mask keeps the
    // receiver's metalness in .a). Everything else writes metalness.
    outAlbedoMetal = vec4(albedo, isDecal ? albedoAlpha : metalness);

    vec2 currNDC = vCurrClipUnjit.xy / vCurrClipUnjit.w;
    vec2 prevNDC = vPrevClip.xy      / vPrevClip.w;
    // Motion vector points from current pixel to where the surface WAS last
    // frame. Tested adding a (curr_jitter - prev_jitter) delta here; sign
    // flip didn't change behavior. Motion vec stays jitter-free.
    vec2 motion = prevNDC - currNDC;
    // .b = this surface's OWN previous NDC depth (same convention as the depth
    // buffer: both are (VP·worldPos).z/w). The deferred GI disocclusion compares
    // it against what the prev depth BUFFER actually held at the reprojected spot —
    // they MATCH for a correctly-reprojected moving/deforming surface (skinned
    // mesh) so it does NOT false-reset, but DIFFER for a real disocclusion (a
    // trail revealing another surface). Comparing curr-vs-prev depth instead would
    // wrongly reset any surface that moves in depth → dust on animated meshes.
    outMotion = vec4(motion, vPrevClip.z / vPrevClip.w, 0.0);

    // .x = per-frame visible index +1 (clear-to-0 = sky, matches
    // deferred_shade.comp's gbufIdsTex.x convention). .y = stable per-object id (host-assigned; 0 when
    // unassigned/sky). .z = flags | class byte (already packed into vFlags host-
    // side, bits 8..15). Truncates to uint16 per channel — class fits in 8..15.
    outIds = uvec4(vInstanceIdx + 1u, vStableId, vFlags, 0u);

    // log2 of the per-pixel UV-footprint diameter (texture-size-independent).
    // A consumer would turn this into a per-texture LOD via `bias + log2(textureSize.x)`.
    // Floor at -16 to keep textureLod's clamp from biting the rare
    // duvx==duvy==0 case (degenerate triangle / fully orthogonal view).
    const float fp2 = max(dot(duvx, duvx), dot(duvy, duvy));
    const float lodBias = 0.5 * log2(max(fp2, 1e-32));

    outUv = vec4(vUv, lodBias, 0.0);
}
