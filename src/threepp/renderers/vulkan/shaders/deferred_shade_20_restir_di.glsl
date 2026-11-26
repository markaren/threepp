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
// RIS resolves both: importance-RESAMPLE M emissive candidates into ONE
// reservoir (weighted by unshadowed contribution) and cast ONE shadow ray
// (vs emissiveNEE's 16 = faster); the single-sample noise is the denoiser's to
// integrate. ReSTIR's temporal and spatial reuse of the reservoir are gone (see
// restirEmissiveDI). lightType ≥ 1000 ⇒ emissive triangle (analytic
// dir/point/spot keep their own noise-free analytic loops).
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
            const MaterialDesc hm = entryMat(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false));
            const vec3 hem = hm.emissive * hm.emissiveIntensity;
            if (max(max(hem.r, hem.g), hem.b) < 0.05) rayQueryConfirmIntersectionEXT(rq);
        }
    }
    return rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

// Returns DEMOD diffuse irradiance (NdotL·Le·W/π, no albedo/metalness — the
// recombine applies albedo·(1-metalness)). The caller folds this into diffInd so it
// rides the denoiser (SVGF's temporal + à-trous, or Ray Reconstruction): RIS picks one
// of this frame's candidates at each pixel, and the denoiser integrates the picks.
//
// NO REUSE. ReSTIR's temporal and spatial reuse ran here on the SVGF path; Ray
// Reconstruction dropped it 2026-11-03 (a reservoir carried across frames or shared
// with its neighbours is noise RR keeps as a static grain). On SVGF it measured as no
// gain and one fault (2026-11-26; gltf-research-scenes kitchen, bathroom2 and
// living-room, and a room lit by 48 small coloured strips): the converged level
// against the path tracer, the noise of a still frame and of a fast pan and the
// share of spikes came out the same within a few tenths of a per cent, and the shade
// 0.02-0.12 ms cheaper without it. The fault: a camera cut (so every load) restarted
// the reservoirs, the reuse carried the first frames' estimate on in them for up to 20
// frames each, and the indirect accumulation stretched that into an overshoot of
// the kitchen's bounce-lit areas (+5 %) that took ~4 s to settle. Without it the cut
// settles in ~30 frames.
vec3 restirEmissiveDI(vec3 P, vec3 N, float NdotV, vec3 F0, bool doShadows) {
    if (pc.emissiveCount == 0u || pc.emissiveTotalPower <= 0.0) return vec3(0.0);
    if (emGateBound(P) < 1.0) return vec3(0.0);// out of every emitter's range: no candidates, no rays
    const vec3 shadowOrig = P + N * SHADOW_EPS;
    const int  emIters    = findMSB(max(pc.emissiveCount - 1u, 1u)) + 1;
    // Per-(pixel,frame) seed → the candidates vary each frame, and the denoiser's
    // accumulation integrates them. Stable spatial key (pixel index) + a clean
    // per-frame increment (a jittered-worldPos key did not converge).
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

    // The pick's own visibility: one shadow ray. The RIS target is UNSHADOWED, so
    // the pick may be occluded (the emitter's far face, a wall, the sphere's own
    // shadow); occluded, it contributes nothing this frame.
    if (r.W <= 0.0 || r.p_hat <= 0.0 || r.lightType < 999.5) return vec3(0.0);
    const LightInfoD li = evalLightInfoEm(int(r.lightType), r.lightPos, P);
    const float NdotL = max(dot(N, li.dir), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);
    if (doShadows && restirOccluded(shadowOrig, li.dir, li.maxDist)) return vec3(0.0);

    // DEMOD irradiance: NdotL·Le·W/π; albedo·(1-metal) is reapplied by the denoise
    // recombine. No Fresnel weight, as evalLight's note says for every direct diffuse
    // term. It used to take (1 - F(N·V)), the smooth-surface Fresnel at the view angle:
    // 4 % off a floor seen from 41 degrees above, a third off one seen at grazing, on
    // a rough surface whose real specular albedo stays a few percent at every angle.
    vec3 c = (NdotL * (1.0 / PI) * r.W) * li.Le;
    const float clum = max(max(c.r, c.g), c.b);
    if (clum > pc.fireflyClamp) c *= pc.fireflyClamp / clum;
    return c;
}

// Cheap diffuse IRRADIANCE from the emissive area lights — like emissiveNEE but
// returns ∫Li·cosθ/pdf (no BRDF; the caller applies albedo/π) with a small,
// caller-chosen sample count. Used by the 1-bounce GI gather, which tolerates the
// extra noise (it's temporally accumulated + à-trous denoised). Shares the
// emitter sample plan of emissiveNEE (emissive_lights.glsl).
vec3 emissiveIrradiance(vec3 P, vec3 N, int EM_SAMPLES, bool doShadows, inout uint seed) {
    if (pc.emissiveCount == 0u || pc.emissiveTotalPower <= 0.0) return vec3(0.0);
    if (emGateBound(P) < 1.0) return vec3(0.0);// out of every emitter's range
    // The pick (global strata, or per-light COVERAGE with point proxies for small
    // lights) is emissive_lights.glsl's; this loop owns the cosine term and the plain
    // opaque shadow ray. twoSided: a GI bounce accepts an emitter's back face.
    const vec3   shadowOrig = P + N * SHADOW_EPS;
    const EmPlan plan = emPlanBuild(P, EM_SAMPLES);
    const vec3   shift = vec3(rnd(seed), rnd(seed), rnd(seed));// a random turn per call (emPlanSample)
    vec3 sum = vec3(0.0);
    for (int s = 0; s < plan.count; ++s) {
        EmSample es;
        if (!emPlanSample(plan, s, P, /*twoSided=*/true, shift, es)) continue;
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

vec3 traceRadiance(vec3 origin, vec3 dir, bool doShadows, float maxLod, float missLod, inout uint seed, bool cheapHits, bool probeHitFill, float envInt);// fwd decl
vec3 giRadiance(vec3 origin, vec3 orgN, vec3 dir, bool doShadows, float maxLod, inout uint seed, out bool missed, float envInt);// fwd decl (cheap 1-bounce GI)

// Scales the DIRECT emission of traceRadiance's hits — the emitter-specular
// OWNERSHIP split for the opaque reflection. 1 = the ray owns emitter spec
// (near-mirror: the tight lobe hits the emitter every frame → no variance).
// 0 = emissiveSpecNEE owns it (rough lobes: a stochastic ray hits a small
// bright emitter with p≪1 → a clamped binomial spike train = constant boiling;
// dielectrics: F0≈0.04 → the full-radiance hit was a "probe speckle"). Blended
// in between so nothing is double-counted at the crossover.
float gReflEmitterScale = 1.0;
// Per light, the share of its specular that the ray keeps on top of that scale
// (emissive_lights.glsl, kEmRaySpacing0): a hit on light l is scaled by
// mix(gReflEmitterScale, 1, share of l). 0 = no light has one.
uvec2 gReflEmitterRayShare = uvec2(0u);

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
// radicalInverse2, the Hammersley second dimension, shared with rtao.comp.
#include "radical_inverse.glsl"

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
//
// gSunFixedPattern (set by traceRadiance for the hits of a trace along the
// view ray: what a THIN pane and a blend sheet show) takes the per-pixel turn
// out of the adaptive estimate: the same probes and the same sixteen
// directions for every pixel, fixed about the sun's direction. Solid glass
// has the same trouble and keeps the turned pattern all the same: with it
// included, tests/renderers/VulkanGolden_test.cpp's glass scene (a solid
// sphere over a floor, its own penumbra seen through it) left its baseline
// (46.0 dB and maxD 33 where 32 is allowed; 55.9 dB and 16 without), and a
// baseline is not moved to make room for a fix. The turned pattern is right for a
// primary, whose pixels the TAA resolves where they stand. Behind clear glass
// nothing resolves it (no spatial filter, and reflReproject drops the history
// of a near-mirror surface whenever the camera moves), and the turn does two
// things there. The three probes, on a diameter that differs from pixel to
// pixel, agree here and disagree next door, so a pixel that returns an exact
// 0 or 1 stands among neighbours that return eleven sixteenths: black and
// white dots through every penumbra. And sixteen rays turned differently per
// pixel through anything fine (a canopy) are a different sixteen samples per
// pixel: a stipple over everything under a tree.
//
// Measured from the car at Trollstigen through a clear pane. A sun of 3
// degrees, the pixels of the penumbra on the road (where the direct view's
// own sun term is 15 to 85 % of its lit level): 15.1 % of them at or under
// zero with the turned pattern, 0.1 % with the fixed one; mean visibility
// 0.611 and 0.616, the direct view's 0.618. The scene's own sun (0.6
// degrees), fine grain as a share of the mean: asphalt in full sun 9.7 % ->
// 4.2 % (seen directly 3.8 %), the shadow's edge 13.2 % -> 8.3 % (7.3 %), a
// concrete barrier under birches 79 % -> 29 % (19 %). Under the birches the
// level against the direct view's was 0.74 and is 0.79 (the pane passes
// about 0.82): what is left there is the sixteen directions' own pattern,
// flat patches with straight edges where the direct view's denoiser has a
// soft dapple.
//
// The probes are five where the turned estimate has three: the centre and
// four points on the rim, on two fixed axes. With no turn to move the
// diameter about, two rim points would miss every edge that runs along it.
// They only DECIDE: where they disagree the even spiral is the whole
// estimate (the five are not an even sample of the disc). Against sixteen
// rays at every hit the five-probe rule gave the same picture in each of the
// measures above (the bins of the penumbra's histogram within 0.2 points).
//
// Cost (RTX 4070, 1280x720, switched in one process), five rays where the
// turned estimate has three and sixteen more wherever the five disagree: the
// windscreen on 27 % of the frame +0.7 ms on open road and +1.8 ms under
// birches, over 78 % of it +2.9 ms in both places.
bool gSunFixedPattern = false;
float sunShadowVisOpaque(vec3 orig, vec3 L, float tanR, bool cheapHit);
// The soft sun's visibility, times what the glass on the way lets through
// (one glass ray along the centre; a cheap hit goes without).
float sunShadowVis(vec3 orig, vec3 L, float tanR, bool cheapHit) {
    const float v = sunShadowVisOpaque(orig, L, tanR, cheapHit);
    return (v > 0.0 && !cheapHit) ? v * glassTransmittance(orig, L, 1e30) : v;
}
float sunShadowVisOpaque(vec3 orig, vec3 L, float tanR, bool cheapHit) {
    // ONE call site of shadowVis, in a loop over this estimate's rays: each call
    // site is a whole inlined copy of the shadow test, and this function, with
    // its sixteen sites, is inlined into every lit hit and primary there is; the
    // copies are what the driver's compile time is made of. Every ray is
    // shadowVis(orig, <direction>, 1e30), so nothing turns from a literal into
    // a runtime value in it. The rays are the ones the four estimates below
    // cast, in the same order, each direction by its own expression as before,
    // and each estimate stops where it stopped (the fixed pattern's probes stop
    // casting at the first that disagrees); shadowVis returns exactly 0 or 1,
    // so every sum is exact. The ray categories bracket the same rays.
    //   hard sun (tanR <= 0): the one ray along L;
    //   cheapHit (incl. under kRR): one jittered ray;
    //   fixed pattern (gSunFixedPattern, not cheapHit): the centre, four probes
    //     (all four, averaged with the centre, when coarse), then sixteen;
    //   turned: the centre and two rim rays, then thirteen (none when coarse).
    const int kSunHard = 0, kSunCheap = 1, kSunFixed = 2, kSunTurned = 3;
    int   mode = kSunHard;
    vec3  t = vec3(0.0), b = vec3(0.0), d = vec3(0.0), rim = vec3(0.0);
    bool  coarse = false;
    float rot = 0.0, bnRad = 0.0;
    uint  rayCatWas = 0u;
    if (tanR > 0.0) {
        // Under DLSS Ray Reconstruction (kRR) every caller's sun is the cheapHit
        // one: a single jittered ray, animated with the frame, at every pixel.
        // RR integrates it over frames; the static patterns and the 3 + 13 (or 5
        // + 16) ray tiers exist to look settled without a denoiser.
        cheapHit = cheapHit || kRR;
        const vec3 up = abs(L.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        t = normalize(cross(up, L));
        b = cross(L, t);
        // setSunSoftDistance (the lights UBO's sunSoftDistance, 0 = off): past it
        // from the eye a probe that disagrees returns its own mean, and the
        // denoiser smooths the coarse penumbra instead of 13 or 16 more rays.
        const vec3 eyeToOrig = orig - gPrimaryOrigin;
        coarse = (kRayBudget & kRayBudgetSoftSun) != 0u && lights.sunSoftDistance > 0.0 &&
                 dot(eyeToOrig, eyeToOrig) > lights.sunSoftDistance * lights.sunSoftDistance;
        if (gSunFixedPattern && !cheapHit) {
            mode      = kSunFixed;
            rayCatWas = rayCatSet(kRayCatSoftSun);
        } else {
            const uvec2 px = uvec2(gl_GlobalInvocationID.xy);
            const uint  fr = cheapHit ? pc.frame : 0u;// static at primaries, animated at cheapHit
            const float bnRot = blueNoiseDef(px, fr, 0u);
            bnRad = blueNoiseDef(px, fr, 1u);
            rot   = TWO_PI * bnRot;
            if (cheapHit) {
                mode = kSunCheap;
                d    = t * cos(rot) + b * sin(rot);
            } else {
                mode      = kSunTurned;
                rayCatWas = rayCatSet(kRayCatSoftSun);
                rim       = t * cos(rot) + b * sin(rot);
            }
        }
    }
    const float GA = 2.39996323;
    float c = 0.0, v1 = 0.0, sum = 0.0;
    for (int k = 0; k < 21; ++k) {
        vec3 dir;
        if (mode == kSunCheap) {
            dir = normalize(L + d * (sqrt(bnRad) * tanR));
        } else if (k == 0) {
            dir = L;// the hard sun's ray, and the centre of both patterns
        } else if (mode == kSunFixed) {
            if (k == 1) dir = normalize(L + t * tanR);
            else if (k == 2) dir = normalize(L - t * tanR);
            else if (k == 3) dir = normalize(L + b * tanR);
            else if (k == 4) dir = normalize(L - b * tanR);
            else {
                const int   i = k - 5;
                const float a = (float(i) + 0.5) * 2.39996323;
                const float r = sqrt((float(i) + 0.5) / 16.0) * tanR;
                dir = normalize(L + (t * cos(a) + b * sin(a)) * r);
            }
        } else {
            if (k == 1) dir = normalize(L + rim * (0.9 * tanR));
            else if (k == 2) dir = normalize(L - rim * (0.9 * tanR));
            else {
                const int   i = k - 3;
                const float a = rot + (float(i) + 1.0) * GA;
                const float r = sqrt((float(i) + bnRad) / 13.0) * tanR;
                dir = normalize(L + (t * cos(a) + b * sin(a)) * r);
            }
        }
        const float v = shadowVis(orig, dir, 1e30);
        if (mode == kSunHard || mode == kSunCheap) return v;
        if (mode == kSunFixed) {
            if (k == 0) {
                c = v;
            } else if (coarse) {
                sum = (k == 1) ? c + v : sum + v;
                if (k == 4) {
                    rayCatSet(rayCatWas);
                    return sum * 0.2;
                }
            } else if (k <= 4) {
                if (c != v) {
                    k = 4;// the remaining probes are not cast: on to the sixteen
                } else if (k == 4) {
                    rayCatSet(rayCatWas);
                    return c;// binary 0/1 (TerminateOnFirstHit) -> exact
                }
            } else {
                sum += v;
                if (k == 20) {
                    rayCatSet(rayCatWas);
                    return sum * (1.0 / 16.0);
                }
            }
        } else {
            if (k == 0) {
                c = v;// v0
            } else if (k == 1) {
                v1 = v;
            } else if (k == 2) {
                if (c == v1 && v1 == v) {
                    rayCatSet(rayCatWas);
                    return c;// binary 0/1 (TerminateOnFirstHit) → exact
                }
                sum = c + v1 + v;
                if (coarse) {
                    rayCatSet(rayCatWas);
                    return sum * (1.0 / 3.0);
                }
            } else {
                sum += v;
                if (k == 15) {
                    rayCatSet(rayCatWas);
                    return sum * (1.0 / 16.0);
                }
            }
        }
    }
    return sum;// not reached: every estimate returns inside the loop
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
        vec3        L    = normalize(lights.dirLights[gi].direction);
        murkSunLeg(P, L);// submerged: the occluder test follows the refracted leg
        const float tanR = pc.sunTanHalfAngle;
        if (tanR > 0.0) {
            const vec3 up = abs(L.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
            const vec3 t  = normalize(cross(up, L));
            const vec3 b  = cross(L, t);
            const float a = TWO_PI * xi.x;
            L = normalize(L + (t * cos(a) + b * sin(a)) * (sqrt(xi.y) * tanR));
        }
        const float v = shadowVis(orig, L, 1e30);
        return v > 0.0 ? v * glassTransmittance(orig, L, 1e30) : 0.0;// the glass the sun comes through
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
    const float v = shadowVis(orig, toL, dist - 1e-2);
    return v > 0.0 ? v * glassTransmittance(orig, toL, dist - 1e-2) : 0.0;// the glass the lamp shines through
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
    float mv[8u + kClusterMaxPerCell];// kSM: each light's shadow-map visibility, -1 = no map
    uint  nL = 0u;
    vec3  U    = vec3(0.0);
    float wSum = 0.0;
    // Moving cloud shadows: the directional SUN on the ground is attenuated by
    // the cloud transmittance overhead (cloudShadowSample; 1.0 when clouds off).
    const float cloudShadow = cloudShadowSample(P);
    for (uint i = 0u; i < lights.dirCount; ++i) {
        vec3 L = normalize(lights.dirLights[i].direction);
        // murkSunCaustic is cloudShadow's underwater twin and sits beside it for
        // that reason: both are an attenuation this sun picked up on its way down
        // from something the shading point cannot see, one a cloud deck and one a
        // rippled surface. Both are exactly 1.0 when their feature is off.
        // murkSunLeg takes the ORIGINAL L (the caustic walks back along it) and
        // hands the loop the in-water direction + the leg's transmittance.
        const float caus = murkSunCaustic(P, L);
        const vec3  leg  = murkSunLeg(P, L);
        vec3 c = vec3(0.0);
        if (dot(N, L) > 0.0)
            c = evalLight(N, V, L, NdotV, F0, albedo, roughness, metalness, k, sheenColor, sheenRoughness)
              * lights.dirLights[i].color * (cloudShadow * pdSurfaceLightTransmittance(P, L) * caus * leg);
        U += c;
        mv[nL] = (kSM && i == 0u && dot(c, c) > 0.0) ? smSunVis(P, N, vec2(px)) : -1.0;
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
        mv[nL] = (kSM && dot(c, c) > 0.0)
                ? smLampVis(clusterLights[li].shadowSlot, clusterLights[li].type > 0.5, P, N,
                            clusterLights[li].position, vec2(px)) : -1.0;
        lw[nL] = dot(c, LUM); wSum += lw[nL]; ++nL;
    }
    // NOTHING LIT (every N.L <= 0, every cluster light out of range or cone):
    // this pixel HAS NO shadow ratio. It used to return 1.0 here on the
    // grounds that U x R = 0 makes the value irrelevant. But the ratio plane
    // is filtered SPATIALLY (deferred_gi_filter.comp) and TEMPORALLY (the
    // shadowVis gather in deferred_shade.comp), and a lit normal-map bevel or
    // cutout-leaf edge whose neighbours are such pixels averaged their
    // placeholder 1.0 into its own 0 and lit up with UNSHADOWED sun: the
    // Bistro "bright edges with the denoiser on" (sun-coloured outlines on
    // every stone-block joint and hedge leaf inside a shadowed wall; those
    // pixels' G-buffer normal flips with the jitter, so their history is
    // reset every frame and the filter sits in its wide spatial-variance
    // regime). -1 is the no-ratio marker: both filters skip such a pixel as
    // a tap, and the recombine's clamp() makes it 0 (U is 0 there anyway).
    visEst = -1.0;
    if (wSum <= 1e-8) return U;
    visEst = 1.0;
    const vec3 orig = P + N * SHADOW_EPS;
    // kSM: a light a shadow map answers enters the ratio exactly, by its map
    // visibility, and leaves the ray pick (its weight zeroed); the rays then
    // estimate only the share of the lights no map covers. All mapped (the
    // no-ray-query case): no ray, a deterministic ratio.
    float smMapped = 0.0;
    float smRest   = wSum;
    if (kSM) {
        smRest = 0.0;
        for (uint i = 0u; i < nL; ++i) {
            if (mv[i] >= 0.0) {
                smMapped += lw[i] * mv[i];
                lw[i] = 0.0;
            } else {
                smRest += lw[i];
            }
        }
    }
    float vAcc = 0.0;
    for (int s = 0; s < nRays; ++s) {
        // Distinct blue-noise salts per ray + dimension (GI uses 0/1, the
        // inline sun path 0/1 static — these are frame-animated and distinct).
        const float xiPick = blueNoiseDef(uvec2(px), pc.frame, uint(2 + s * 3));
        const vec2  xiDisc = vec2(blueNoiseDef(uvec2(px), pc.frame, uint(3 + s * 3)),
                                  blueNoiseDef(uvec2(px), pc.frame, uint(4 + s * 3)));
        const float xi = xiPick * smRest;
        float acc = 0.0;
        uint  pick = 0xFFFFFFFFu;
        for (uint i = 0u; i < nL; ++i) {
            acc += lw[i];
            if (xi <= acc && lw[i] > 1e-8) { pick = i; break; }
        }
        vAcc += (pick == 0xFFFFFFFFu) ? 1.0 : shadowRayVisTo(orig, P, pick, xiDisc, cellBase);
    }
    visEst = kSM ? (smMapped + smRest * (vAcc / float(nRays))) / wSum : vAcc / float(nRays);
    // Every light answered by a map: the ratio is exact this frame, and the
    // channel stores it without history (gShadowMapOnly, deferred_shade.comp):
    // a map cannot say what moved, and a swept shadow accumulated over frames
    // smears. A mapped sun also drops out of the top-up below (lw[0] = 0).
    gShadowMapOnly = kSM && smRest <= 1e-8 * wSum;
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
    // Not under Ray Reconstruction (kRR): RR denoises the one-ray estimate
    // itself and has no history to protect from a sweeping shadow.
    gShadowSunTopUp = false;
    if (!kRR && !flagOn(512u) && lights.dirCount > 0u && lw[0] > 0.5 * wSum) {
        bool topUp = gShadowMovingOccluder;
        // Prev-aux fetch deferred behind the cheaper predicates: most pixels
        // (sun-minor, or flagged already) never pay it.
        if (!topUp) topUp = texelFetch(shadowVisPrevTex, px, 0).z <= 2.5;
        if (topUp) {
            vec3        Ls     = normalize(lights.dirLights[0].direction);
            murkSunLeg(orig, Ls);// submerged: refracted leg
            const float sunVis = sunShadowVis(orig, Ls, pc.sunTanHalfAngle, /*cheapHit=*/false);
            visEst = mix(visEst, sunVis, lw[0] / wSum);
            gShadowSunTopUp = true;
        }
    }
    return U;
}
