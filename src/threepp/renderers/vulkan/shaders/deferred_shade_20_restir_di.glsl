// Split from deferred_shade.comp: ReSTIR DI (emissive-triangle reservoir
// resampling), emissive irradiance for the GI gather, forward declarations
// for traceRadiance/giRadiance, soft sun shadow visibility, cluster-light
// cell lookup, and the analytic direct-light split (Heitz ratio estimator).

// ───────────────────────────────────────────────────────────────────────────
// ReSTIR DI (deferred) — emissive area-light direct lighting.
// The coherent emissiveNEE
// above is noise-free but BIASED (a fixed sample set lights some emitter faces
// more than others); per-pixel dithering it just trades the bias for noise that
// EXPLODES with light count (Bistro = 1000s of emissive tris → un-denoisable).
// ReSTIR resolves both: importance-RESAMPLE M emissive candidates into ONE
// reservoir (weighted by unshadowed contribution), cast ONE shadow ray, and
// (Stage B) reuse the reservoir across frames so it converges to smooth.
// Stage A here = init RIS + visibility, NO temporal/spatial reuse
// yet → 1 shadow ray (vs emissiveNEE's 16 = faster) but still single-sample
// noisy until Stage B adds the reservoir ping-pong. lightType ≥ 1000 ⇒ emissive
// triangle (analytic dir/point/spot keep their own noise-free analytic loops).
float lum3(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

struct Reservoir { vec3 lightPos; float lightType; float W_sum; float M; float W; float p_hat; };
void updateReservoir(inout Reservoir r, vec3 pos, float ltype, float w, float p_hat_new, inout uint seed) {
    r.W_sum += w;
    r.M     += 1.0;
    if (rnd(seed) < w / max(r.W_sum, 1e-20)) { r.lightPos = pos; r.lightType = ltype; r.p_hat = p_hat_new; }
}
void finalizeReservoir(inout Reservoir r) { r.W = r.W_sum / max(r.M * r.p_hat, 1e-20); }

// Reconstruct (dir, maxDist, Le) for an emissive-triangle reservoir sample.
struct LightInfoD { vec3 dir; float maxDist; vec3 Le; };
LightInfoD evalLightInfoEm(int typeCode, vec3 lightPos, vec3 hitPos) {
    LightInfoD o; o.dir = vec3(0.0); o.maxDist = 0.0; o.Le = vec3(0.0);
    const int eTi = typeCode - 1000;
    if (eTi < 0 || eTi >= int(pc.emissiveCount)) return o;// bounds: stale/garbage reservoir guard
    const vec3 toL = lightPos - hitPos;
    const float dist = length(toL);
    if (dist < 1e-12) return o;
    o.dir = toL / dist; o.maxDist = dist - 1e-2; o.Le = emissiveTris[eTi].emission.rgb;
    return o;
}

// One non-opaque shadow ray toward an emitter sample; SKIPS emitters (a light
// can't shadow itself) — same ray-query the emissiveNEE shadow path uses.
bool restirOccluded(vec3 shadowOrig, vec3 dir, float maxDist) {
    rayQueryEXT rq;
    // kRayMaskOpaque: decals/glass never block emissive light (matches
    // emissiveNEE's shadow query).
    rayQueryInitializeEXT(rq, topAS, gl_RayFlagsTerminateOnFirstHitEXT, kRayMaskOpaque,
                          shadowOrig, 1e-3, dir, maxDist);
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
            const MaterialDesc hm = mats[rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false)];
            const vec3 hem = hm.emissive * hm.emissiveIntensity;
            if (max(max(hem.r, hem.g), hem.b) < 0.05) rayQueryConfirmIntersectionEXT(rq);
        }
    }
    return rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

// Returns DEMOD diffuse irradiance (NdotL·Le·W/π, no albedo/metalness — the
// recombine applies albedo·(1-metalness)). The caller folds this into diffInd so it
// rides the SVGF temporal + à-trous denoise: the reservoir converges light SELECTION,
// the denoiser smooths the 1-sample RADIANCE (ReSTIR DI is always denoiser-paired).
vec3 restirEmissiveDI(vec3 P, vec3 N, float NdotV, vec3 F0, bool doShadows) {
    if (pc.emissiveCount == 0u || pc.emissiveTotalPower <= 0.0) return vec3(0.0);
    const vec3 shadowOrig = P + N * SHADOW_EPS;
    const int  emIters    = findMSB(max(pc.emissiveCount - 1u, 1u)) + 1;
    // Per-(pixel,frame) seed → candidates vary each frame so Stage B's temporal
    // reuse accumulates them. Stable spatial key (pixel index) + clean per-frame
    // increment so the temporal sequence converges (a jittered-worldPos key did not).
    uint seed = pcgHash(gl_GlobalInvocationID.x
              + pcgHash(gl_GlobalInvocationID.y + pcgHash(pc.frame * 9781u + 1u)));

    Reservoir r;
    r.lightPos = vec3(0.0); r.lightType = -2.0;
    r.W_sum = 0.0; r.M = 0.0; r.W = 0.0; r.p_hat = 0.0;

    const int M_CANDIDATES = 8;// RIS candidates resampled to 1 (→ 1 shadow ray)
    for (int s = 0; s < M_CANDIDATES; ++s) {
        const float xi = rnd(seed) * pc.emissiveTotalPower;
        uint lo = 0u, hi = pc.emissiveCount - 1u;
        for (int it = 0; it < emIters; ++it) {
            if (lo >= hi) break;
            const uint mid = (lo + hi) >> 1u;
            if (emissiveTris[mid].v1.w < xi) lo = mid + 1u; else hi = mid;
        }
        const EmTri t = emissiveTris[lo];
        const float r1 = rnd(seed), r2 = rnd(seed);
        const float su1 = sqrt(r1);
        const vec3 lp = (1.0 - su1) * t.v0.xyz + (su1 * (1.0 - r2)) * t.v1.xyz + (su1 * r2) * t.v2.xyz;
        const vec3 toL = lp - P;
        const float dist2 = dot(toL, toL);
        const float dist  = sqrt(max(dist2, 1e-20));
        if (dist <= 1e-4) { r.M += 1.0; continue; }
        const vec3 L = toL / dist;
        const float NdotL = dot(N, L);
        const vec3 lnRaw = cross(t.v1.xyz - t.v0.xyz, t.v2.xyz - t.v0.xyz);
        const float lnLen = length(lnRaw);
        if (NdotL <= 0.01 || lnLen < 1e-20 || t.v0.w <= 1e-20 || t.v2.w <= 0.0) { r.M += 1.0; continue; }
        const float cosLight = abs(dot(-L, lnRaw / lnLen));
        if (cosLight <= 0.01) { r.M += 1.0; continue; }
        const float pickPdf = t.v2.w / pc.emissiveTotalPower;
        const float p_omega = pickPdf * dist2 / (t.v0.w * cosLight);
        const float p_hat   = NdotL * lum3(t.emission.rgb);
        updateReservoir(r, lp, 1000.0 + float(lo), p_hat / max(p_omega, 1e-20), p_hat, seed);
    }
    finalizeReservoir(r);

    // ── Visibility reuse (Bitterli 2020 §5) ── the RIS target p_hat is UNSHADOWED, so
    // the chosen candidate may be occluded (the emitter's own far face, a wall, the
    // sphere's shadow). Shadow-test it NOW — before temporal reuse — and DISCARD the
    // reservoir if occluded (W←0, W_sum←0; keep M for the merge normalisation). Without
    // this, occluded samples persist into temporal history with positive weight and
    // inflate W on a later VISIBLE pick → the penumbra / self-occlusion OVER-COUNT that
    // read as too bright (floor brighter than wall). +1 shadow ray (still ≪ the 16 the
    // old emissiveNEE cast); introduces the standard small conservative bias.
    if (r.W > 0.0 && r.lightType >= 999.5) {
        const LightInfoD liV = evalLightInfoEm(int(r.lightType), r.lightPos, P);
        if (max(dot(N, liV.dir), 0.0) <= 0.0
                || (doShadows && restirOccluded(shadowOrig, liV.dir, liV.maxDist))) {
            r.W = 0.0; r.W_sum = 0.0;// occluded — let temporal reuse recover a visible sample
        }
    }

    // ── Stage B: temporal reuse ── reproject via the motion vector, validate the
    // surface (prev-normal match), read LAST frame's reservoir, re-evaluate ITS
    // target p_hat at THIS pixel (the point of ReSTIR reuse), and merge. The single
    // per-frame sample thus accumulates into an effective many-sample estimate that
    // converges to SMOOTH via this temporal accumulation.
    const ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    {
        const vec2 uv   = (vec2(px) + 0.5) / vec2(float(pc.width), float(pc.height));
        const vec2 mv   = texture(gbufMotionTex, paneToPhys(gbufMotionTex, uv)).rg;
        const vec2 cNDC = vec2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));
        const vec2 pNDC = cNDC + mv;
        const vec2 pUv  = vec2(pNDC.x * 0.5 + 0.5, 0.5 - pNDC.y * 0.5);
        bool tvalid = all(greaterThanEqual(pUv, vec2(0.0))) && all(lessThanEqual(pUv, vec2(1.0)));
        if (tvalid) {
            const vec3 pn = texture(gbufNormalPrevTex, paneToPhys(gbufNormalPrevTex, pUv)).xyz * 2.0 - 1.0;
            tvalid = dot(pn, pn) > 1e-6 && dot(N, normalize(pn)) > 0.7;// surface match (rejects sky/disocclusion)
        }
        if (tvalid) {
            const ivec2 ppx = ivec2(pUv * vec2(float(pc.width), float(pc.height)));
            const vec4 pPos = imageLoad(reservoirPosRead, ppx);
            const vec4 pWd  = imageLoad(reservoirWRead,   ppx);
            // M-clamp: short history in motion (flush stale fast), long when static.
            const float mClamp = mix(20.0, 5.0, smoothstep(0.002, 0.03, length(mv)));
            const float pM = min(pWd.y, mClamp);
            if (pWd.z > 0.0 && pM > 0.0 && pPos.w >= 999.5) {// valid emissive reservoir
                const LightInfoD pli = evalLightInfoEm(int(pPos.w), pPos.xyz, P);
                const float pPhat = max(dot(N, pli.dir), 0.0) * lum3(pli.Le);// target re-eval at THIS pixel
                if (pPhat > 0.0) {
                    const float wPrev = pPhat * pM * pWd.z;
                    r.W_sum += wPrev;
                    r.M     += pM;
                    if (rnd(seed) < wPrev / max(r.W_sum, 1e-20)) {
                        r.lightPos = pPos.xyz; r.lightType = pPos.w; r.p_hat = pPhat;
                    }
                    finalizeReservoir(r);
                }
            }
        }
    }
    // ── Snapshot the PRE-SPATIAL reservoir ── this (NOT the post-spatial r) is what
    // gets persisted for next frame's temporal merge. Persisting the post-spatial
    // reservoir would feed spatial
    // contributions — which don't generalise across reprojection — back into the
    // temporal M, biasing the weights and producing visible temporal instability.
    Reservoir rPreSpatial = r;
    // Cap the persisted W (firefly bound) so a one-off high-W sample can't perpetuate
    // via the multiplicative w_prev temporal feedback (= static bright specks that
    // never average out). Shading below uses the unclamped post-spatial r.
    float persistW = rPreSpatial.W;
    if (pc.fireflyClamp < 1e20) persistW = min(persistW, 5.0);

    // ── Stage 1c: spatial reuse ──
    // Tap a few random neighbours from the PREV-frame reservoir buffer (bindings
    // 28/30), validate each against THIS pixel's surface (prev-normal cone + world
    // distance — the same prev gbuffer the temporal stage validates against), and
    // RIS-merge it after re-evaluating its target p_hat at OUR shading point. The
    // M>=mTarget early-out means converged static pixels (high temporal M) skip it;
    // the cost is spent where temporal reuse is weakest — freshly DISOCCLUDED pixels
    // with little/no history (the frame-after-disocclusion noise the temporal-only
    // path couldn't fix). M-cap 4/neighbour; spMax 5 static / 2 in motion (fewer
    // stale taps under motion). Reads prev-frame neighbours so there's no second
    // dispatch / barrier.
    {
        const vec2  size   = vec2(float(pc.width), float(pc.height));
        const float distC  = length(P - gPrimaryOrigin);// view leg of THIS pixel's ray
        const float mvLen  = length(texture(gbufMotionTex, paneToPhys(gbufMotionTex, (vec2(px) + 0.5) / size)).rg);
        const uint  spMax   = (mvLen > 0.01) ? 2u : 5u;
        const float mTarget = 20.0;
        const ivec2 maxPx   = ivec2(int(pc.width) - 1, int(pc.height) - 1);
        for (uint sp = 0u; sp < spMax; ++sp) {
            if (r.M >= mTarget) break;
            const float ang = rnd(seed) * TWO_PI;
            const float rad = sqrt(rnd(seed)) * 20.0;// 20px tap radius
            const ivec2 off = ivec2(int(rad * cos(ang)), int(rad * sin(ang)));
            if (off.x == 0 && off.y == 0) continue;
            const ivec2 spPx = clamp(px + off, ivec2(0), maxPx);
            // Validate the neighbour's PREV surface vs ours: normal cone (rejects
            // sky / differently-facing) + world distance (rejects depth jumps that
            // share a normal, e.g. parallel floors). Reconstructs the neighbour's
            // prev world pos with the current cam — a near-exact gate for the static
            // camera where spatial reuse matters most; loosens harmlessly otherwise
            // (a bad tap's p_hat re-eval keeps the merge unbiased).
            const vec3 pn = texelFetch(gbufNormalPrevTex, spPx, 0).xyz * 2.0 - 1.0;
            if (dot(pn, pn) < 1e-6 || dot(N, normalize(pn)) <= 0.7) continue;
            const vec2  uvN  = (vec2(spPx) + 0.5) / size;
            const vec2  ndcN = vec2(uvN.x * 2.0 - 1.0, -(uvN.y * 2.0 - 1.0));
            const float dN   = texelFetch(gbufDepthPrevTex, spPx, 0).x;
            const vec4  vhN  = cam.projInverse * vec4(ndcN, dN, 1.0);
            const vec3  wpN  = (cam.viewInverse * vec4(vhN.xyz / vhN.w, 1.0)).xyz;
            if (length(wpN - P) > 0.1 * distC) continue;
            // Merge the neighbour's prev-frame reservoir (target p_hat re-eval'd here).
            const vec4 spPos = imageLoad(reservoirPosRead, spPx);
            const vec4 spWd  = imageLoad(reservoirWRead,   spPx);
            if (spWd.z <= 0.0 || spWd.y <= 0.0 || spPos.w < 999.5) continue;// valid emissive only
            const float spM    = min(spWd.y, 4.0);// per-neighbour M-cap
            const LightInfoD spli = evalLightInfoEm(int(spPos.w), spPos.xyz, P);
            const float spPhat = max(dot(N, spli.dir), 0.0) * lum3(spli.Le);
            if (spPhat <= 0.0) continue;
            const float wSp = spPhat * spM * spWd.z;
            r.W_sum += wSp;
            r.M     += spM;
            if (rnd(seed) < wSp / max(r.W_sum, 1e-20)) {
                r.lightPos = spPos.xyz; r.lightType = spPos.w; r.p_hat = spPhat;
            }
        }
        finalizeReservoir(r);
    }

    // Persist the PRE-SPATIAL reservoir for next frame's temporal read.
    imageStore(reservoirPosWrite, px, vec4(rPreSpatial.lightPos, rPreSpatial.lightType));
    imageStore(reservoirWWrite,   px, vec4(rPreSpatial.W_sum, rPreSpatial.M, persistW, rPreSpatial.p_hat));

    // Shading uses the POST-SPATIAL reservoir r (the improved current-frame estimate).
    if (r.W <= 0.0 || r.p_hat <= 0.0 || r.lightType < 999.5) return vec3(0.0);
    const LightInfoD li = evalLightInfoEm(int(r.lightType), r.lightPos, P);
    const float NdotL = max(dot(N, li.dir), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);
    if (doShadows && restirOccluded(shadowOrig, li.dir, li.maxDist)) return vec3(0.0);

    // DEMOD irradiance: (1-F)·NdotL·Le·W/π. The (1-F) diffuse Fresnel (specular takes F,
    // the diffuse lobe gets the rest) restores the term emissiveNEE's kd=(1-Fr)(1-metal)
    // carried — dropping it over-brightened, WORST at grazing (the floor sees the emitter
    // at a shallow angle). Keyed on the surface NdotV (not per-light H) so it stays a
    // per-surface demod quantity. albedo·(1-metal) is reapplied by the denoise recombine.
    vec3 c = (NdotL * (1.0 / PI) * r.W) * li.Le * (vec3(1.0) - fresnelSchlick(NdotV, F0));
    const float clum = max(max(c.r, c.g), c.b);
    if (clum > pc.fireflyClamp) c *= pc.fireflyClamp / clum;
    return c;
}

// Cheap diffuse IRRADIANCE from the emissive area lights — like emissiveNEE but
// returns ∫Li·cosθ/pdf (no BRDF; the caller applies albedo/π) with a small,
// caller-chosen sample count. Used by the 1-bounce GI gather, which tolerates the
// extra noise (it's temporally accumulated + à-trous denoised). Shares the
// emitter sample plan of emissiveNEE (emissive_lights.glsl).
vec3 emissiveIrradiance(vec3 P, vec3 N, int EM_SAMPLES, bool doShadows) {
    if (pc.emissiveCount == 0u || pc.emissiveTotalPower <= 0.0) return vec3(0.0);
    // The pick (global strata, or per-light COVERAGE with point proxies for small
    // lights) is emissive_lights.glsl's; this loop owns the cosine term and the plain
    // opaque shadow ray. twoSided: a GI bounce accepts an emitter's back face.
    const vec3   shadowOrig = P + N * SHADOW_EPS;
    const EmPlan plan = emPlanBuild(P, EM_SAMPLES);
    vec3 sum = vec3(0.0);
    for (int s = 0; s < plan.count; ++s) {
        EmSample es;
        if (!emPlanSample(plan, s, P, /*twoSided=*/true, es)) continue;
        const float ndl = dot(N, es.L);
        if (ndl <= 0.01) continue;
        if (doShadows) {
            const float tMax = es.dist - es.backoff - 1e-2;// a point proxy stops at the light's radius
            if (tMax > 1e-3) {
                rayQueryEXT rq;
                // kRayMaskOpaque: blend/transmissive surfaces don't block (force-
                // opaque flag would otherwise commit a decal's transparent quad).
                rayQueryInitializeEXT(rq, topAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT,
                                      kRayMaskOpaque, shadowOrig, 1e-3, es.L, tMax);
                while (rayQueryProceedEXT(rq)) {}
                if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT)
                    continue;
            }
        }
        vec3 c = ndl * es.Le * es.gw;
        const float lum = max(max(c.r, c.g), c.b);
        if (lum > pc.fireflyClamp) c *= pc.fireflyClamp / lum;
        sum += c * es.w;
    }
    return sum;
}

vec3 traceRadiance(vec3 origin, vec3 dir, bool doShadows, float maxLod, float missLod, inout uint seed, bool cheapHits, bool probeHitFill);// fwd decl
vec3 giRadiance(vec3 origin, vec3 dir, bool doShadows, float maxLod, inout uint seed, out bool missed);// fwd decl (cheap 1-bounce GI)

// Scales the DIRECT emission of traceRadiance's hits — the emitter-specular
// OWNERSHIP split for the opaque reflection. 1 = the ray owns emitter spec
// (near-mirror: the tight lobe hits the emitter every frame → no variance).
// 0 = emissiveSpecNEE owns it (rough lobes: a stochastic ray hits a small
// bright emitter with p≪1 → a clamped binomial spike train = constant boiling;
// dielectrics: F0≈0.04 → the full-radiance hit was a "probe speckle"). Blended
// in between so nothing is double-counted at the crossover.
float gReflEmitterScale = 1.0;

// Diffuse indirect. TWO paths, selected by the denoiser flag (pc.flags bit 2):
//  • DENOISER ON → REAL stochastic 1-bounce GI: a few cosine rays via traceRadiance
//    (hit → the surface's LIT colour = colour BLEED; miss → sky; enclosure → its
//    own dark shade). No AO/sky hacks — occlusion, sky-visibility, colour bleed all
//    emerge from the bounce. Noisy at low spp → demodulated + spatially denoised +
//    temporally accumulated (TAA) by the caller.
//  • DENOISER OFF → the deterministic Fibonacci AO + gated "far≈sky" approximation
//    (noise-free WITHOUT a denoiser; the clean fallback for non-denoised scenes).
// Blue-noise tile lookup — dithers the deferred GI off the 64×64 void-and-cluster
// tile (blueNoiseTex). Two `salt` values step the lookup along different irrational
// rates → decorrelated x/y samples.
float blueNoiseDef(uvec2 px, uint frame, uint salt) {
    const ivec2 step = ivec2(47, 17) * int(salt + 1u);
    const ivec2 cell = ivec2(px) + step * int(frame);
    const vec2  uv   = (vec2(cell & 63) + 0.5) / 64.0;
    return texture(blueNoiseTex, uv).r;
}
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

// Soft directional-light (sun) visibility: average shadowVis over a stratified
// disc of directions inside the sun cone (tanR = tan of the angular radius;
// the real sun subtends ~0.27°). MULTI-RAY with a STATIC per-pixel dither,
// because direct lighting is never denoised and the primary RNG stream is
// temporally stable: a SINGLE static-jittered ray renders the penumbra as a
// fixed binary dither ("pixelated" shadows), and a per-frame-jittered one
// flickers (the property the static stream protects). The dither is the
// BLUE-NOISE tile, not IGN: IGN is a gradient noise, smooth along diagonals,
// so with a small sample set the penumbra's quantization error correlates
// into a diagonal "weave" — the same failure family as the emissive "fence
// weave". Blue noise has no directional structure; the error becomes fine
// isotropic grain (the same reasoning as the GI hemisphere dither above).
// ADAPTIVE: 3 probes (exact centre + two opposite rim points, per-pixel
// rotated) — agreement means fully lit / fully occluded (the vast majority of
// pixels, 3 rays; the centre probe keeps sub-cone thin occluders like chair
// slats from being straddled). Disagreement = penumbra → 13 more rays complete
// a golden-angle spiral (16 total → 17 visibility levels, the rect lights'
// smoothness; 8 read too grainy), with a blue-noise Cranley-Patterson offset
// on the RADIAL strata too so the ring quantization decorrelates per pixel.
// Only penumbra pixels pay the full count, and radius 0 disables entirely.
// Reflection/denoised hits (cheapHit) instead take ONE per-frame-ANIMATED
// jittered ray — they ride the reflection SVGF temporal, which integrates it
// (the rect-light SIDE=1 policy, verbatim).
float sunShadowVis(vec3 orig, vec3 L, float tanR, bool cheapHit) {
    if (tanR <= 0.0) return shadowVis(orig, L, 1e30);
    const vec3 up = abs(L.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    const vec3 t  = normalize(cross(up, L));
    const vec3 b  = cross(L, t);
    const uvec2 px = uvec2(gl_GlobalInvocationID.xy);
    const uint  fr = cheapHit ? pc.frame : 0u;// static at primaries, animated at cheapHit
    const float bnRot = blueNoiseDef(px, fr, 0u);
    const float bnRad = blueNoiseDef(px, fr, 1u);
    const float rot   = TWO_PI * bnRot;
    if (cheapHit) {
        const vec3 d = t * cos(rot) + b * sin(rot);
        return shadowVis(orig, normalize(L + d * (sqrt(bnRad) * tanR)), 1e30);
    }
    const vec3  rim = t * cos(rot) + b * sin(rot);
    const float v0  = shadowVis(orig, L, 1e30);
    const float v1  = shadowVis(orig, normalize(L + rim * (0.9 * tanR)), 1e30);
    const float v2  = shadowVis(orig, normalize(L - rim * (0.9 * tanR)), 1e30);
    if (v0 == v1 && v1 == v2) return v0;// binary 0/1 (TerminateOnFirstHit) → exact
    float sum = v0 + v1 + v2;
    const float GA = 2.39996323;
    for (int i = 0; i < 13; ++i) {
        const float a = rot + (float(i) + 1.0) * GA;
        const float r = sqrt((float(i) + bnRad) / 13.0) * tanR;
        sum += shadowVis(orig, normalize(L + (t * cos(a) + b * sin(a)) * r), 1e30);
    }
    return sum * (1.0 / 16.0);
}

// Cell record base for this pixel (count at [base], indices after). viewDist
// = view-space distance of the shading point. Fixed exponential ladder — must
// match cluster_build.comp; beyond kClusterZMax clamps into the last slice.
uint clusterCellBase(ivec2 px, float viewDist) {
    const uint cx = min(uint(px.x) * kClusterX / max(pc.width, 1u),  kClusterX - 1u);
    const uint cy = min(uint(px.y) * kClusterY / max(pc.height, 1u), kClusterY - 1u);
    const float t = log(max(viewDist, kClusterZMin) / kClusterZMin)
                  / log(kClusterZMax / kClusterZMin);
    const uint cz = min(uint(max(t, 0.0) * float(kClusterZ)), kClusterZ - 1u);
    return (cx + cy * kClusterX + cz * kClusterX * kClusterY) * (kClusterMaxPerCell + 1u);
}

// ── Denoised direct-shadow channel (primary surfaces, denoise ON) ───────────
// One soft shadow ray toward the light picked by index `gi`: gi < dirCount →
// UBO directional (jittered within the sun cone, pc.sunTanHalfAngle);
// otherwise cluster slot (gi - dirCount) of this pixel's cell — point/spot
// sample a disc of the light's PHYSICAL source radius perpendicular to the
// shading direction, which is what turns their hard 1-ray shadows into real
// penumbras that widen with distance (radius 0 keeps them exact-hard). xi is
// frame-ANIMATED blue noise: unlike the old always-inline path this estimate
// feeds a temporal accumulator, so fresh directions every frame integrate
// instead of flickering.
float shadowRayVisTo(vec3 orig, vec3 P, uint gi, vec2 xi, uint cellBase) {
    if (gi < lights.dirCount) {
        const vec3  L    = normalize(lights.dirLights[gi].direction);
        const float tanR = pc.sunTanHalfAngle;
        if (tanR <= 0.0) return shadowVis(orig, L, 1e30);
        const vec3 up = abs(L.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        const vec3 t  = normalize(cross(up, L));
        const vec3 b  = cross(L, t);
        const float a = TWO_PI * xi.x;
        return shadowVis(orig, normalize(L + (t * cos(a) + b * sin(a)) * (sqrt(xi.y) * tanR)), 1e30);
    }
    const uint li = min(clusterGrid[cellBase + 1u + (gi - lights.dirCount)],
                        pc.clusterLightCount - 1u);
    const vec3  lp   = clusterLights[li].position;
    const float srcR = clusterLights[li].radius;
    vec3  toL  = lp - P;
    float dist = length(toL);
    if (dist < 1e-4) return 1.0;
    toL /= dist;
    if (srcR > 0.0) {
        const vec3 up = abs(toL.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        const vec3 t  = normalize(cross(up, toL));
        const vec3 b  = cross(toL, t);
        const float a = TWO_PI * xi.x;
        const vec3 lp2 = lp + (t * cos(a) + b * sin(a)) * (sqrt(xi.y) * srcR);
        toL  = lp2 - P;
        dist = length(toL);
        if (dist < 1e-4) return 1.0;
        toL /= dist;
    }
    return shadowVis(orig, toL, dist - 1e-2);
}

// Analytic/stochastic direct-light split (Heitz et al. 2018 ratio estimator).
// Returns the exact UNSHADOWED direct sum U — dir lights from the UBO plus
// THIS PIXEL'S CLUSTER LIGHTS (all scene point/spot lights, no 8-per-type
// cap; per-light BRDF × color × attenuation, kept in sync with
// shadeDiffuseDirect's loops) — and writes a 1-2 ray estimate of the
// luminance visibility ratio R to visEst: each ray picks one light ∝
// lum(c_i) and returns its soft visibility, so
// E[visEst] = Σ lum(c_i)·V_i / Σ lum(c_i). The recombine multiplies U × R̃
// (denoised) — color/BRDF stay exact, only the bounded scalar is stochastic.
// This pairing is what makes many lights affordable: the cluster bounds the
// per-pixel eval count and the ratio estimator holds shadow cost at 1-2 rays
// TOTAL regardless of how many lights the cell holds.
// (Known Heitz-2018 approximation: differently-colored lights with different
// shadowing share one scalar ratio — a slight tint in mixed penumbras.)
vec3 analyticDirectSplit(vec3 P, vec3 N, vec3 V, vec3 albedo, float roughness,
                         float metalness, vec3 F0, vec3 sheenColor, float sheenRoughness,
                         ivec2 px, float clusterDist, int nRays, out float visEst) {
    const vec3  LUM   = vec3(0.2126, 0.7152, 0.0722);
    const float NdotV = max(dot(N, V), 1e-4);
    const float k     = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float lw[8u + kClusterMaxPerCell];// pick weights (lum of the unshadowed contribution)
    uint  nL = 0u;
    vec3  U    = vec3(0.0);
    float wSum = 0.0;
    // Moving cloud shadows: the directional SUN on the ground is attenuated by
    // the cloud transmittance overhead (cloudShadowSample; 1.0 when clouds off).
    const float cloudShadow = cloudShadowSample(P);
    for (uint i = 0u; i < lights.dirCount; ++i) {
        const vec3 L = normalize(lights.dirLights[i].direction);
        vec3 c = vec3(0.0);
        // murkSunCaustic is cloudShadow's underwater twin and sits beside it for
        // that reason: both are an attenuation this sun picked up on its way down
        // from something the shading point cannot see, one a cloud deck and one a
        // rippled surface. Both are exactly 1.0 when their feature is off.
        if (dot(N, L) > 0.0)
            c = evalLight(N, V, L, NdotV, F0, albedo, roughness, metalness, k, sheenColor, sheenRoughness)
              * lights.dirLights[i].color * cloudShadow * murkSunCaustic(P, L);
        U += c;
        lw[nL] = dot(c, LUM); wSum += lw[nL]; ++nL;
    }
    // Cluster lights: unified point/spot records — a point light carries the
    // -1.1/-1.05 cone sentinel, so its spotAtten evaluates to exactly 1.
    const uint cellBase = clusterCellBase(px, clusterDist);
    const uint cCnt = (pc.clusterLightCount == 0u)
                          ? 0u
                          : min(clusterGrid[cellBase], kClusterMaxPerCell);
    for (uint ci = 0u; ci < cCnt; ++ci) {
        const uint li = min(clusterGrid[cellBase + 1u + ci], pc.clusterLightCount - 1u);
        vec3 c = vec3(0.0);
        vec3        toL  = clusterLights[li].position - P;
        const float dist = length(toL);
        if (dist >= 1e-4) {
            toL /= dist;
            if (dot(N, toL) > 0.0) {
                const float spotCos   = dot(-toL, clusterLights[li].direction);
                const float spotAtten = smoothstep(clusterLights[li].cosAngleOuter,
                                                   clusterLights[li].cosAngleInner, spotCos);
                if (spotAtten > 0.0) {
                    const float decay = clusterLights[li].decay;
                    float atten = 1.0 / max(pow(dist, decay), 0.01);
                    const float range = clusterLights[li].range;
                    if (range > 0.0) {
                        const float t  = dist / range;
                        const float t4 = t * t * t * t;
                        const float wnd = max(1.0 - t4, 0.0);
                        atten *= wnd * wnd;
                    }
                    c = evalLight(N, V, toL, NdotV, F0, albedo, roughness, metalness, k, sheenColor, sheenRoughness)
                      * clusterLights[li].color * atten * spotAtten;
                }
            }
        }
        U += c;
        lw[nL] = dot(c, LUM); wSum += lw[nL]; ++nL;
    }
    visEst = 1.0;
    if (wSum <= 1e-8) return U;// nothing lit → ratio irrelevant (U×R = 0)
    const vec3 orig = P + N * SHADOW_EPS;
    float vAcc = 0.0;
    for (int s = 0; s < nRays; ++s) {
        // Distinct blue-noise salts per ray + dimension (GI uses 0/1, the
        // inline sun path 0/1 static — these are frame-animated and distinct).
        const float xiPick = blueNoiseDef(uvec2(px), pc.frame, uint(2 + s * 3));
        const vec2  xiDisc = vec2(blueNoiseDef(uvec2(px), pc.frame, uint(3 + s * 3)),
                                  blueNoiseDef(uvec2(px), pc.frame, uint(4 + s * 3)));
        const float xi = xiPick * wSum;
        float acc = 0.0;
        uint  pick = 0xFFFFFFFFu;
        for (uint i = 0u; i < nL; ++i) {
            acc += lw[i];
            if (xi <= acc && lw[i] > 1e-8) { pick = i; break; }
        }
        vAcc += (pick == 0xFFFFFFFFu) ? 1.0 : shadowRayVisTo(orig, P, pick, xiDisc, cellBase);
    }
    visEst = vAcc / float(nRays);
    // SUBGROUP DILATION of the moving-occluder flag. The per-pixel signal fires
    // only when one of THIS pixel's 1-2 rays commits a hit on the mover — in the
    // OUTER penumbra (visEst 0.8..1.0) the hit probability is low, so a band of
    // partially-darkened pixels sweeps by unflagged, never gets dwell-pinned,
    // never becomes release-eligible, and its faint stale darkening keeps the
    // long history = the residual ghost ring the per-pixel cluster left behind
    // ("less but not gone"). Inner-penumbra/umbra pixels detect near-certainly
    // (and, below, with 16 rays), so an OR over the subgroup (≤8 px of the same
    // 8×8 tile) carries their verdict outward across the penumbra — same-frame,
    // no extra rays or storage. Vote semantics over ACTIVE lanes only; worst-
    // case divergence degrades to the per-pixel flag, never worse. The cost of
    // over-dilation is ≤8 px of briefly-short history beside a moving shadow.
    gShadowMovingOccluder = subgroupAny(gShadowMovingOccluder);
    // MOVING-CASTER SHADOW: a ray above (or a subgroup neighbour's) hit a MOVING
    // occluder (car sweeping its shadow over this pixel — gShadowMovingOccluder,
    // set in shadowVis). A sweeping shadow cannot be temporally accumulated —
    // ANY history trails it as a smear (the reason shadow-map engines regenerate
    // shadows per frame, and NRD SIGMA keeps moving-caster shadows near-
    // frameless). So these pixels run at cap≈2 in the shadow temporal, and the
    // penumbra SMOOTHNESS must come from THIS frame: re-estimate the SUN's
    // visibility with sunShadowVis — the inline (denoise-off) path's adaptive
    // 3→16-ray, STATIC-dithered estimator, whose per-frame result is already
    // smooth and stable (the user-validated moving-shadow reference) — and fold
    // it in by the sun's share of the unshadowed luminance (dir light 0 = the
    // sun, EnvSunPolicy). Umbra interior costs 3 agreeing rays; only the
    // penumbra ring pays 16. Static scenes never fire the flag and keep the
    // converged 1-2 ray economy.
    // The RELEASE ring needs the same quality: pixels a moving shadow JUST LEFT
    // carry stale history and a short (pinned ≤2) histLen — read the pixel's own
    // prev aux (un-reprojected: a chase cam moves it a few px, fine for a
    // ray-count hint) and treat "was dwell-pinned" like the dwell itself, so the
    // release comparison in the temporal is made against a CONFIDENT estimate,
    // not a 1-ray binary.
    // flags bit 9 (512u) = THREEPP_VK_SHADOW_DWELL=0 kill switch (perf triage:
    // separates the dwell machinery's ray cost from kernel-occupancy cost).
    // SUN-DOMINANCE gate (> 0.5 share, not > 0): the top-up re-estimates ONLY
    // the sun, so the folded visEst is confident only where the sun carries the
    // pixel's light mix. Where analytic point/spot lights dominate (night
    // bistro lamps), a MULTI-light pixel's 1-2 ray visEst flips 0/1 with the
    // per-frame stochastic light pick — an estimate the 16 sun rays cannot
    // stabilise — and the temporal's release compare (|visEst-mean| ≤ 0.1)
    // can then NEVER be satisfied: the pixel is TRAPPED at cap ≤2 in permanent
    // binary boil (seen on the Bistro awning wall). No confidence → no top-up
    // (poor ROI on a minority term) → gShadowSunTopUp stays false → the
    // temporal's release path stays disabled and history grows normally.
    gShadowSunTopUp = false;
    if ((pc.flags & 512u) == 0u && lights.dirCount > 0u && lw[0] > 0.5 * wSum) {
        bool topUp = gShadowMovingOccluder;
        // Prev-aux fetch deferred behind the cheaper predicates: most pixels
        // (sun-minor, or flagged already) never pay it.
        if (!topUp) topUp = texelFetch(shadowVisPrevTex, px, 0).z <= 2.5;
        if (topUp) {
            const vec3  Ls     = normalize(lights.dirLights[0].direction);
            const float sunVis = sunShadowVis(orig, Ls, pc.sunTanHalfAngle, /*cheapHit=*/false);
            visEst = mix(visEst, sunVis, lw[0] / wSum);
            gShadowSunTopUp = true;
        }
    }
    return U;
}
