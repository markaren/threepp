// Split from deferred_shade.comp: reflection-hit geometry/UV fetch,
// bindless material texture sampling helpers, GGX half-vector sampling
// (random + deterministic Hammersley/Fibonacci variants), and the
// multi-bounce specular reflection + 1-bounce diffuse GI ray functions
// (traceRadiance, giRadiance).

// Interpolated world-space normal + UV at a reflection hit. Mirrors
// closest_hit's GeometryDesc + barycentric interpolation + world transform.
void fetchHit(int instIdx, int primId, vec2 bary, mat4x3 worldToObj,
              out vec3 Nworld, out vec2 uvOut) {
    const GeometryDesc g = geoms[instIdx];
    const uvec3 idx = gfetchTri(g, primId);
    const float w = 1.0 - bary.x - bary.y;

    // gfetch* honour GeometryDesc.packedAttrs (oct-snorm16 normals / unorm16
    // uvs on packed static geometry, tightly-packed float otherwise).
    const vec3 n0 = gfetchNormal(g, idx.x);
    const vec3 n1 = gfetchNormal(g, idx.y);
    const vec3 n2 = gfetchNormal(g, idx.z);
    const vec3 nObj = normalize(w * n0 + bary.x * n1 + bary.y * n2);
    Nworld = normalize(transpose(mat3(worldToObj)) * nObj);

    uvOut = vec2(0.0);
    if (g.uvAddress != 0ul) {
        const vec2 u0 = gfetchUv(g, idx.x);
        const vec2 u1 = gfetchUv(g, idx.y);
        const vec2 u2 = gfetchUv(g, idx.z);
        uvOut = w * u0 + bary.x * u1 + bary.y * u2;
    }
}

// Sample a bindless material texture at a reflection-hit UV (compute has no
// derivatives → explicit mip 0). texIndex < 0 → return `fallback` unchanged.
vec3 hitTex(int texIndex, mat3 uvXform, vec2 uv, vec3 fallback) {
    if (texIndex < 0) return fallback;
    const int i = clamp(texIndex, 0, int(kMaxMaterialTextures) - 1);
    return fallback * textureLod(albedoMaps[nonuniformEXT(i)], (uvXform * vec3(uv, 1.0)).xy, 0.0).rgb;
}

// Albedo-texture ALPHA at a hit — for alpha-testing cutouts and weighting
// blend surfaces along secondary rays (the queries run force-opaque, so no
// any-hit ever evaluates alpha for us). 1.0 when untextured, matching the
// gbuffer screen-door (factor-only opacity routes through transmission).
float hitTexAlpha(int texIndex, mat3 uvXform, vec2 uv) {
    if (texIndex < 0) return 1.0;
    const int i = clamp(texIndex, 0, int(kMaxMaterialTextures) - 1);
    return textureLod(albedoMaps[nonuniformEXT(i)], (uvXform * vec3(uv, 1.0)).xy, 0.0).a;
}

// GGX NDF importance-sampled microfacet half-vector around N (alpha=roughness²).
// Tight cone at r→0, wide at high r. Plain NDF (not VNDF) sampling — cheap and
// fine across the moderate roughness range we use. The caller frame-jitters the
// seed so TAA averages successive samples into a clean lobe on static surfaces.
vec3 ggxHalfVector(vec3 N, float roughness, inout uint seed) {
    const float a    = roughness * roughness;
    const float u1   = rnd(seed);
    const float u2   = rnd(seed);
    const float phi  = TWO_PI * u1;
    const float cosT = sqrt((1.0 - u2) / (1.0 + (a * a - 1.0) * u2));
    const float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    const vec3  Ht   = vec3(sinT * cos(phi), sinT * sin(phi), cosT);// tangent space
    const vec3  up   = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    const vec3  T    = normalize(cross(up, N));
    const vec3  B    = cross(N, T);
    return normalize(T * Ht.x + B * Ht.y + N * Ht.z);
}
// Perturbed reflection direction (mirror reflect about a GGX-sampled half-vec) —
// blurs glossy reflections instead of staying mirror-sharp.
vec3 sampleGGXReflection(vec3 V, vec3 N, float roughness, inout uint seed) {
    return reflect(-V, ggxHalfVector(N, roughness, seed));
}

// DETERMINISTIC, well-distributed (Hammersley) GGX half-vector — NO per-pixel RNG.
// The random version left a FROZEN per-pixel speckle once the seed went
// frame-stable → "static noise on glass/reflective". Hammersley is coherent across
// pixels (smooth, no speckle) and converges with sample count, like the AO gather.
float radicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}
vec3 ggxHalfVectorFib(vec3 N, float roughness, int s, int count) {
    if (count <= 1) return N;// single sample = exact mirror (deterministic bounce)
    const float a    = roughness * roughness;
    // Elevation: stratified GGX-importance uniform → correct lobe distribution.
    const float u1   = (float(s) + 0.5) / float(count);
    // Azimuth: GOLDEN-ANGLE SPIRAL (s*GA) instead of a count-spaced ring. The old
    // ring (phi = 2π·(s+0.5)/count) put the samples at `count` discrete azimuths,
    // so a flat glossy surface reflected the scene as `count` coherent COPIES at
    // those angles. The spiral spreads them evenly with no banding. Coherent
    // across pixels (no per-pixel jitter) → NO grain; the reflection lobe is
    // clamped narrow at the call site so the residual copies cluster tightly near
    // the mirror direction and read as a clean soft reflection, not ghost images.
    const float phi  = float(s) * 2.39996323;
    const float cosT = sqrt((1.0 - u1) / (1.0 + (a * a - 1.0) * u1));
    const float sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    const vec3  Ht   = vec3(sinT * cos(phi), sinT * sin(phi), cosT);
    const vec3  up   = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    const vec3  T    = normalize(cross(up, N));
    const vec3  B    = cross(N, T);
    return normalize(T * Ht.x + B * Ht.y + N * Ht.z);
}
vec3 sampleGGXReflectionFib(vec3 V, vec3 N, float roughness, int s, int count) {
    return reflect(-V, ggxHalfVectorFib(N, roughness, s, count));
}

// MULTI-BOUNCE specular reflection (or refraction-continuation). Follows the ray
// through up to REFL_MAX_BOUNCES reflective hits: at each hit it adds the
// surface's diffuse + direct (shadowed) shading weighted by the running specular
// throughput, then — if the surface is smooth/metallic enough — CONTINUES the
// reflection one bounce deeper. Rough surfaces, the final bounce, or negligible
// throughput cap off with prefiltered-env specular. A miss terminates into the
// environment at `missLod` (then the running roughness mip on deeper bounces).
//
// This is what gives reflections DEPTH — a reflected mirror/metal now shows the
// real scene (and its further reflections), not just the environment as the old
// single-bounce version did. Non-recursive: GLSL has no recursion, so it's an
// explicit loop. Most rays hit a rough surface and stop after one bounce, so the
// extra cost is paid only on actual mirror-on-mirror chains. Same signature as
// before, so every caller (primary glossy reflection, water + glass reflect,
// glass behind-view) gets the depth. Compute has no derivatives → mip 0 on tex.
// cheapHits: shade hits with the stochastic 1-picked-light NEE (cheap, noisy —
// ONLY for callers whose result flows through a blur + temporal channel that
// absorbs it: the opaque reflection buffer, ocean water). Callers whose result
// reaches the screen UNBLURRED — clear glass (its denoise path is deliberately
// sharp), additive/alpha-blend behind-traces straight into `lit` — must pass
// false: the stochastic pick showed as per-pixel noise in the glass sphere.
// probeHitFill: hit diffuse fill source in probe-GI mode. REFLECTION callers
// pass true — the probe field is the occlusion-correct interior irradiance
// (the mirror-in-a-corridor fix). TRANSMISSION continuations (thin-shell /
// refraction / behind-view — content directly under glass) pass false and
// keep the env fill's SHAPE: a world-scale probe grid cannot resolve a
// glass-covered cavity (watch dial, goggles, vitrines — probes there sit
// inside the case). The fill's MAGNITUDE is scaled by gEnvFillVis, the sky
// visibility measured at the transmitting surface — "seen through clear
// glass" sees the sky the glass surface sees, which in an enclosed interior
// is none of it (the raw fill lit everything behind glass sky-bright while
// probe GI darkened the opaque surround).
// First-bounce hit distance of the last traceRadiance call, for the reflection
// denoiser's gloss-blur footprint (carried in reflAux .w): -1 = no committed
// hit (pure env miss — the radiance is ALREADY lobe-filtered via missLod, so
// the spatial gloss blur must not double-blur it), > 0 = distance to the first
// shaded geometry hit (the mirror ray returns it sharp; the spatial blur IS
// the gloss there, sized by this distance's projected footprint).
float gTraceHitT = -1.0;
// True when the first shaded reflected hit is a MOVING mesh (GeometryDesc.flags bit 0,
// stamped from meshMovedBits each frame). The reflection/GI temporal accumulator
// resets on it: a moving reflected object can't be temporally integrated (the
// surface reproject tracks the surface, not the moving content → ghost trail).
bool gTraceHitMoved = false;

// Set by shadeWater around its own reflection trace: PASS THROUGH other water
// surfaces instead of shading them. A wave-perturbed reflection ray leaves the
// surface nearly grazing and re-enters a neighbouring crest — geometrically
// legitimate on a curved surface, not merely a precision artefact — and
// shading that hit as opaque water paints a DARK DOT. Scattered across the
// distance band where the slope distribution straddles the horizon, those dots
// are the long-standing "patch of breakup" on calm fjord/norway water.
// Skipping is also the physically better answer: what a grazing ray would see
// past the crest is the near-horizon sky it is already about to sample.
// Water reflected in OTHER surfaces is unaffected — only water's own
// reflection sets this.
bool gTraceSkipWater = false;

// Sky-visibility scale for the env hit fill on probeHitFill=false traces.
// Stamped by the transmission callers (shadeGlass, the additive/alpha-blend
// behind-views) with probeEnvFillVis at the transmitting surface; 1.0 for
// every other caller (reflection traces read the probe fill and ignore this).
float gEnvFillVis = 1.0;

// Measured probe/env irradiance ratio at P — the envSpecVis construction from
// the opaque primary shade: 1 under open sky, → 0 deep inside an enclosure.
// Conf-gated the same way: a starved probe neighbourhood (all 8 probes inside
// geometry) means "unmeasured", not "dark" — return 1.0 so the env fill
// survives untouched (the watch-dial/goggles cavity case).
float probeEnvFillVis(vec3 P, vec3 N, float maxLod) {
    if (probeGrid.enabled <= 0.5) return 1.0;
    const vec3  LUMW   = vec3(0.2126, 0.7152, 0.0722);
    float conf;
    const float actE   = dot(probeIrradianceConf(P, N, conf), LUMW) * (1.0 / PI);
    const float unoccE = dot(sampleEnvLod(N, maxLod), LUMW);
    const float vis    = clamp(actE / max(unoccE, 1e-5), 0.0, 1.0);
    return mix(1.0, vis, smoothstep(0.0, 0.1, conf));
}

// envInt = the PRIMARY surface's MaterialDesc.envMapIntensity, passed down
// rather than read off a global so the two nested callers (a glass retrace
// inside a water shade) can never disagree about whose material is in force.
// It scales ONLY the terms where this ray ESCAPES to the environment; a
// geometry hit is lit by the scene, not by the primary surface's IBL knob, and
// stays untouched at every bounce.
vec3 traceRadiance(vec3 origin, vec3 dir, bool doShadows, float maxLod, float missLod, inout uint seed, bool cheapHits, bool probeHitFill, float envInt) {
    const int REFL_MAX_BOUNCES = 3;
    // A PASS-THROUGH is not a bounce: the water-crest skip, a sub-cutoff cutout
    // texel and a blend layer all leave the ray travelling in the SAME direction
    // through a surface that did not stop it. Charging those against the bounce
    // budget made three leaf-card texels exhaust the trace, and then whatever
    // the terminal is, is wrong — black (the original behaviour) hid inside a
    // dark tree reflection, while the environment (which briefly replaced it)
    // painted bright white speckle over every canopy reflected in the fjord
    // water: a canopy IS a stack of alpha-cutout cards, so most rays into one
    // thread several. Give the pass-throughs their own, larger budget so the ray
    // reaches a real surface (or the open sky) on its own, and let
    // REFL_MAX_BOUNCES count only genuine reflective bounces, unchanged.
    const int REFL_MAX_STEPS = 12;
    vec3  radiance   = vec3(0.0);
    vec3  tput       = vec3(1.0);
    vec3  o          = origin;
    vec3  d          = dir;
    float curMissLod = missLod;
    gTraceHitT = -1.0;
    gTraceHitMoved = false;
    // Reset the moving-occluder shadow flag so it observes ONLY this trace's
    // shadow rays (safe: the shadow channel captured its value right after
    // analyticDirectSplit, before any reflection/glass tracing).
    gShadowMovingOccluder = false;

    int b    = 0;// REFLECTIVE bounces only — incremented at the continuation below
    int step = 0;// every ray segment (bounces + pass-throughs); the loop guard
    for (; step < REFL_MAX_STEPS; ++step) {
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, topAS, gl_RayFlagsOpaqueEXT, kRayMaskAll, o, 1e-3, d, 1e30);
        while (rayQueryProceedEXT(rq)) {}
        if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
            radiance += tput * sampleEnvLod(d, curMissLod) * envInt;// escaped → environment
            break;
        }
        const int          hitId  = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
        const MaterialDesc hm     = mats[hitId];
        const int    primId = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
        const vec2   bary   = rayQueryGetIntersectionBarycentricsEXT(rq, true);
        const float  tHit   = rayQueryGetIntersectionTEXT(rq, true);
        const mat4x3 w2o    = rayQueryGetIntersectionWorldToObjectEXT(rq, true);
        vec3 hitN; vec2 hitUv;
        fetchHit(hitId, primId, bary, w2o, hitN, hitUv);

        // Water self-hit pass-through (see gTraceSkipWater). Advance past the
        // crest and keep tracing; the offset scales with the hit distance so a
        // far grazing hit doesn't re-hit the same triangle from float error.
        if (gTraceSkipWater && geoms[hitId].foamAddress != 0ul) {
            o = o + d * (tHit + max(1e-3, tHit * 1e-4));
            continue;
        }

        // Cutout (MASK): the query runs force-opaque, so alpha-test here —
        // sub-cutoff texels pass through (mirrors gbuffer.frag / shadow_anyhit).
        // Without this a masked plane behind glass shaded fully opaque,
        // painting its transparent regions as a solid wall.
        if (hm.alphaCutoff > 0.0 &&
            hitTexAlpha(hm.albedoTexIndex, hm.uvTransform, hitUv) < hm.alphaCutoff) {
            o = o + d * (tHit + 1e-3);
            continue;
        }

        // Blend surfaces (alphaCutoff < 0, no transmission — alpha quads, text
        // decals): COMPOSITE them src-over along the ray instead of skipping.
        // Skipping made every blend surface invisible through glass/reflections
        // (the TransmissionOrderTest "missing alpha blend" failure); shading
        // them fully opaque painted the whole quad as a dark plate. The texel
        // alpha weights a diffuse shade, the remainder carries on through.
        const bool blendThrough = (hm.alphaCutoff < 0.0 && hm.transmission <= 0.0);
        float hitAlpha = 1.0;
        if (blendThrough) {
            hitAlpha = hitTexAlpha(hm.albedoTexIndex, hm.uvTransform, hitUv);
            if (hitAlpha <= 0.01) {// nothing to composite
                o = o + d * (tHit + 1e-3);
                continue;
            }
        }

        const vec3 hAlbedo = hitTex(hm.albedoTexIndex, hm.uvTransform, hitUv, hm.albedo);
        if (hm.roughness < 0.0) { radiance += tput * hAlbedo; break; }// unlit hit

        vec3 hitV = -d;// view dir at the hit = back along the ray
        if (dot(hitN, hitV) < 0.0) hitN = -hitN;
        float hRough = hm.roughness;
        float hMetal = hm.metalness;
        if (hm.roughnessTexIndex >= 0)
            hRough *= textureLod(albedoMaps[nonuniformEXT(clamp(hm.roughnessTexIndex, 0, int(kMaxMaterialTextures) - 1))],
                                 (hm.uvTransformRoughMetal * vec3(hitUv, 1.0)).xy, 0.0).g;
        if (hm.metalnessTexIndex >= 0)
            hMetal *= textureLod(albedoMaps[nonuniformEXT(clamp(hm.metalnessTexIndex, 0, int(kMaxMaterialTextures) - 1))],
                                 (hm.uvTransformRoughMetal * vec3(hitUv, 1.0)).xy, 0.0).b;
        hRough = clamp(hRough, 0.04, 1.0);
        hMetal = clamp(hMetal, 0.0, 1.0);
        const vec3 hEmissive = hitTex(hm.emissiveTexIndex, hm.uvTransformEmissive, hitUv,
                                      hm.emissive * hm.emissiveIntensity);
        const vec3 hitP = o + d * tHit;
        // First shaded (non-cutout, non-transparent) hit = the visible reflected
        // content; its distance sizes the denoiser's gloss-blur footprint, and its
        // MOVED flag tells the temporal accumulator to reset (moving reflection).
        if (gTraceHitT < 0.0) {
            gTraceHitT = length(hitP - origin);
            gTraceHitMoved = (geoms[hitId].flags & 1u) != 0u;
        }
        // Reflected-hit diffuse fill. PROBE-GI mode: the probe field IS the
        // occlusion-correct local irradiance (÷π for the diffInd convention) —
        // the crude env+ambient fill leaked full sky onto reflected interior
        // walls (a mirror in a corridor showed sky-lit walls). Blend by the
        // neighbourhood's VALIDITY confidence, not a hard switch: on surfaces
        // whose surrounding probes sit inside geometry (a watch dial recessed
        // behind its crystal, thin shells, models smaller than a grid cell)
        // the interpolation is black because nothing MEASURED it, not because
        // the point is dark — a hard switch rendered every glass-covered dial
        // near-black. Low confidence falls back to the legacy env fill.
        // Probes off keeps the original approximation.
        vec3 hitDiffInd = sampleEnvLod(hitN, maxLod) + lights.ambient + hemiAmbient(hitN);
        if (probeGrid.enabled > 0.5) {
            if (probeHitFill) {
                float probeConf;
                const vec3 probeFill = probeIrradianceConf(hitP, hitN, probeConf) * (1.0 / PI);
                // SATURATED confidence: interior walls have partially-valid probe
                // neighbourhoods (some of the 8 sit inside the wall), and a linear
                // blend re-admitted (1−conf) of the full-sky fill — re-brightening
                // the enclosed corridors the probe fill exists to fix. Any
                // reasonably-measured neighbourhood (conf ≥ 0.25) is trusted
                // outright; only a truly starved one (all 8 probes inside
                // geometry) falls back to the env fill instead of black.
                hitDiffInd = mix(hitDiffInd, probeFill, smoothstep(0.0, 0.25, probeConf));
            } else {
                // Transmission retrace: env-fill shape (probes cannot resolve
                // the cavity) at the sky level that actually reaches the
                // transmitting surface. See gEnvFillVis.
                hitDiffInd *= gEnvFillVis;
            }
        }
        radiance += tput * hitAlpha * shadeDiffuseDirect(hitP, hitN, hitV, hAlbedo, hRough, hMetal,
                                              hEmissive * gReflEmitterScale,
                                              doShadows, hitDiffInd,
                                              hm.sheenColor, hm.sheenRoughness,
                                              hm.specularIntensity, hm.specularColor,
                                              hm.iridescence, hm.iridescenceIOR, hm.iridescenceThicknessNm, seed,
                                              /*addEmissive=*/true,// reflected hit: no per-pixel reservoir → coherent emissiveNEE
                                              /*cheapHit=*/cheapHits);// caller decides (see header comment)

        // Blend hit composited — carry the remaining (1−α) through along the
        // same ray. No specular continuation for the blend layer itself (its
        // alpha-weighted diffuse is the visually dominant term).
        if (blendThrough) {
            tput *= (1.0 - hitAlpha);
            if (max(max(tput.r, tput.g), tput.b) < 0.02) break;
            o = o + d * (tHit + 1e-3);
            continue;
        }

        const float hitNdotV = max(dot(hitN, hitV), 1e-4);
        const vec3  hitF0 = mix(vec3(0.04) * hm.specularIntensity * hm.specularColor, hAlbedo, hMetal);
        const vec2  hitAB = envBRDFApprox(hitNdotV, hRough);
        const vec3  specW = hitF0 * hitAB.x + hitAB.y;// specular throughput for the next bounce
        const vec3  hitR  = reflect(-hitV, hitN);
        // Stop bouncing on rough surfaces, the last bounce, or when the specular
        // throughput is negligible — cap off with prefiltered-env specular.
        if (b >= REFL_MAX_BOUNCES - 1 || hRough > 0.35 ||
            max(max(specW.r, specW.g), specW.b) < 0.02) {
            // Transmission retrace: the split-sum env cap-off takes the same
            // surface sky-visibility scale as the diffuse fill (a rough metal
            // behind glass has no diffuse — this term is its whole shade).
            radiance += tput * specW * sampleEnvLod(hitR, hRough * maxLod)
                      * (probeHitFill ? 1.0 : gEnvFillVis);
            break;
        }
        // Continue the reflection one bounce deeper (GGX-jittered when glossy).
        ++b;// the ONLY place a reflective bounce is counted
        tput      *= specW;
        d          = (hRough < 0.08) ? hitR : sampleGGXReflectionFib(hitV, hitN, hRough, 0, 1);
        if (dot(d, hitN) <= 0.0) d = hitR;
        o          = hitP + hitN * SHADOW_EPS;
        curMissLod = hRough * maxLod;
    }
    // STEP budget exhausted — the ray threaded REFL_MAX_STEPS surfaces without
    // one of them ever stopping it. d and curMissLod still describe the live
    // ray, so terminate on the environment exactly as the miss branch would
    // (returning black here is what put dark dots in grazing ocean
    // reflections). At 12 steps this fires rarely enough not to be a visible
    // answer in its own right; at 3 it fired constantly, and a frequent wrong
    // answer is a visible artifact whichever colour it is.
    // Every shading exit break's first, so this can never double-count.
    if (step >= REFL_MAX_STEPS) radiance += tput * sampleEnvLod(d, curMissLod) * envInt;
    // A moving-caster SHADOW inside the reflected content is moving content just
    // like the mover itself: the shadow rays at the reflected hits committed on
    // a currently-MOVING mesh, so the reflected radiance carries a shadow that
    // slides frame-to-frame while the reflecting surface's reproject tracks the
    // surface — without this the swept reflected shadow ACCUMULATES as ghost
    // echoes on glossy floors (hitMoved alone can't see it: the hit is the
    // static surface CARRYING the shadow, not the mover). Fold it into the same
    // hard-reset; when the caster stops the flag drops and accumulation resumes.
    if (gShadowMovingOccluder) gTraceHitMoved = true;
    return radiance;
}

// CHEAP 1-bounce diffuse GI — replaces the recursive traceRadiance on the GI
// path. One ray; on a hit, shade with ONLY direct sun/point/spot (one shadow ray
// each) + the hit's own emissive + a small-sample emitter NEE. NO specular lobe
// and NO further bounces. That removes both the COST (the 3-bounce recursion +
// 16-sample emissiveNEE per bounce, which dominate in emitter-heavy scenes) and
// the FIREFLIES (the specular continuation catching the sun via a glossy hit).
// Diffuse GI is what the cosine gather integrates anyway. The hit gets NO env
// fill: the 2nd-bounce sky comes from the gather's OWN rays that MISS — so
// enclosed scenes don't leak sky while open scenes still get it.
vec3 giRadiance(vec3 origin, vec3 dir, bool doShadows, float maxLod, inout uint seed, out bool missed, float envInt) {
    missed = false;
    gGiRayHitMoved = false;// set below on a 1-bounce hit on a MOVING mesh (GI dwell cut)
    rayQueryEXT rq;
    // kRayMaskOpaque: the cheap GI bounce passes through blend decals/glass and
    // shades the opaque surface behind them (shading a decal quad as an opaque
    // dark wall painted indirect darkening onto its receiver).
    rayQueryInitializeEXT(rq, topAS, gl_RayFlagsOpaqueEXT, kRayMaskOpaque, origin, 1e-3, dir, 1e30);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        missed = true;// sky visible along this ray — feeds the openness estimate
        return sampleEnvLod(dir, maxLod) * envInt;// escaped → sky (the env 1st-bounce term)
    }

    const int          hitId  = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
    const MaterialDesc hm     = mats[hitId];
    const int    primId = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    const vec2   bary   = rayQueryGetIntersectionBarycentricsEXT(rq, true);
    const float  tHit   = rayQueryGetIntersectionTEXT(rq, true);
    const mat4x3 w2o    = rayQueryGetIntersectionWorldToObjectEXT(rq, true);
    vec3 hitN; vec2 hitUv;
    fetchHit(hitId, primId, bary, w2o, hitN, hitUv);

    const vec3 hAlbedo = hitTex(hm.albedoTexIndex, hm.uvTransform, hitUv, hm.albedo);
    gGiRayHitMoved = (geoms[hitId].flags & 1u) != 0u;// moving 1-bounce hit (before the unlit early-out — those count too)
    if (hm.roughness < 0.0) return hAlbedo;// unlit hit (e.g. a pure-emissive proxy)

    if (dot(hitN, -dir) < 0.0) hitN = -hitN;
    const vec3 hitP       = origin + dir * tHit;
    const vec3 shadowOrig = hitP + hitN * SHADOW_EPS;
    // Metalness = factor × TEXTURE (blue channel), like traceRadiance's hits.
    // glTF defaults metallicFactor to 1.0 and masks dielectrics in the map
    // (Khronos Sponza: 24/25 materials) — the raw factor alone made
    // (1 − metalness) zero the diffuse albedo, silently killing the whole
    // 1-bounce gather on such assets (the "corridor renders near-black" bug).
    float hMetal = clamp(hm.metalness, 0.0, 1.0);
    if (hm.metalnessTexIndex >= 0)
        hMetal *= textureLod(albedoMaps[nonuniformEXT(clamp(hm.metalnessTexIndex, 0, int(kMaxMaterialTextures) - 1))],
                             (hm.uvTransformRoughMetal * vec3(hitUv, 1.0)).xy, 0.0).b;
    const vec3 diff = hAlbedo * (1.0 - hMetal) * (1.0 / PI);

    // Emitter EMISSION is SUPPRESSED in the GI bounce. The DIRECT term (emitter → the
    // shaded surface) is owned by ReSTIR DI / emissiveNEE at the primary, so re-adding
    // the emitter's glow when a GI ray lands on it DOUBLE-COUNTS the direct light — a
    // path tracer's MIS prevents exactly this; the cheap GI has no MIS. The hit's
    // DIFFUSE bounce (loops below) + its own emitter-NEE (emissiveIrradiance) stay —
    // that's legitimate INDIRECT. Same principle as gSuppressReflEmitter for reflections.
    // Was `lit = hEmissive`, which over-brightened every emitter-lit surface (worst where
    // GI rays point at the emitter — e.g. a floor under it = the floor>wall over-count).
    // Analytic direct at the GI hit. With MANY lights, sample ONE (×wL
    // compensation, unbiased) — the GI channel is temporally accumulated +
    // SVGF-denoised, which absorbs the selection noise, and this keeps lamp-
    // heavy interiors cheap (1 shadow ray vs nLights). But for a SMALL light
    // count that 1-of-N pick is pure variance the denoiser only SMEARS — worst
    // for concentrated lights like SPOTLIGHTS, whose tight cone makes "is this
    // bounce point in the cone?" a high-contrast coin-flip per gather ray. So
    // for nLights<=8 loop them ALL (deterministic, ~nLights shadow rays/hit) —
    // mirrors the cheapHit reflection path; the 1-pick stays only where looping
    // all would be the real cost.
    vec3 lit = vec3(0.0);
    const uint nLights = lights.dirCount + lights.pointCount + lights.spotCount;
    const bool pickOne = nLights > 8u;
    float wL = 1.0;
    const uint pick = pickOne ? pickAnalyticLight(hitP, seed, wL) : 0xFFFFFFFFu;

    for (uint i = 0u; i < lights.dirCount; ++i) {
        if (pickOne && i != pick) continue;
        const vec3  L   = normalize(lights.dirLights[i].direction);
        const float ndl = dot(hitN, L);
        if (ndl <= 0.0) continue;
        const float vis = doShadows ? shadowVis(shadowOrig, L, 1e30) : 1.0;
        lit += diff * ndl * lights.dirLights[i].color * (vis * wL);
    }
    for (uint i = 0u; i < lights.pointCount; ++i) {
        if (pickOne && (lights.dirCount + i) != pick) continue;
        vec3        toL  = lights.pointLights[i].position - hitP;
        const float dist = length(toL);
        if (dist < 1e-4) continue;
        toL /= dist;
        const float ndl = dot(hitN, toL);
        if (ndl <= 0.0) continue;
        float atten = 1.0 / max(pow(dist, lights.pointLights[i].decay), 0.01);
        const float range = lights.pointLights[i].range;
        if (range > 0.0) { const float tt = dist / range; const float t4 = tt*tt*tt*tt; const float wnd = max(1.0 - t4, 0.0); atten *= wnd * wnd; }
        if (atten <= 1e-6) continue;
        const float vis = doShadows ? shadowVis(shadowOrig, toL, dist - 1e-2) : 1.0;
        lit += diff * ndl * lights.pointLights[i].color * (atten * vis * wL);
    }
    for (uint i = 0u; i < lights.spotCount; ++i) {
        if (pickOne && (lights.dirCount + lights.pointCount + i) != pick) continue;
        vec3        toL  = lights.spotLights[i].position - hitP;
        const float dist = length(toL);
        if (dist < 1e-4) continue;
        toL /= dist;
        const float ndl = dot(hitN, toL);
        if (ndl <= 0.0) continue;
        const float spotCos   = dot(-toL, lights.spotLights[i].direction);
        const float spotAtten = smoothstep(lights.spotLights[i].cosAngleOuter,
                                           lights.spotLights[i].cosAngleInner, spotCos);
        if (spotAtten <= 0.0) continue;
        float atten = spotAtten / max(pow(dist, lights.spotLights[i].decay), 0.01);
        const float range = lights.spotLights[i].range;
        if (range > 0.0) { const float tt = dist / range; const float t4 = tt*tt*tt*tt; const float wnd = max(1.0 - t4, 0.0); atten *= wnd * wnd; }
        if (atten <= 1e-6) continue;
        const float vis = doShadows ? shadowVis(shadowOrig, toL, dist - 1e-2) : 1.0;
        lit += diff * ndl * lights.spotLights[i].color * (atten * vis * wL);
    }
    // Emitter 1-bounce (e.g. enclosed scene lit only by an emissive sphere) —
    // cheap small-sample diffuse NEE so the colour bleed survives the denoiser.
    // 2 samples (was 4): rides the same accumulation as the rays above.
    lit += diff * emissiveIrradiance(hitP, hitN, 2, doShadows);
    // PROBE GRID (opt-in, multi-bounce): the hit's INDIRECT irradiance from
    // the world-space SH-L1 cache — the bounce 2..∞ light plus the sky the
    // hit actually sees (probe rays that escape through real openings). The
    // direct terms above stay authoritative; the probe stores neither the
    // analytic direct at the hit nor emitter emission, so nothing double-
    // counts (see probe_common.glsl). vec3(0) when probeGrid.enabled == 0.
    lit += diff * probeIrradiance(hitP, hitN);
    return lit;
}
