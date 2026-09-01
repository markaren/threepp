// Split from deferred_shade.comp: gatherEnv (diffuse GI/AO gather), exact
// rect-light form factor, analytic-light pick weighting, and
// shadeDiffuseDirect (emissive + direct analytic + diffuse IBL).

// envInt = the shaded surface's MaterialDesc.envMapIntensity. It scales the
// SKY the gather sees -- the rays that miss, and (deterministic path) the
// near-hit env fill -- and nothing else: a ray that lands on geometry brings
// back that geometry's own shade, which this surface's IBL knob has no say in.
vec3 gatherEnv(vec3 P, vec3 N, ivec2 px, uint frame, bool doShadows, bool stochastic, int nGI, float envInt, out float openness) {
    const float maxLod  = float(max(pc.envMipCount, 1u) - 1u);
    const vec3 up = abs(N.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    const vec3 T  = normalize(cross(up, N));
    const vec3 B  = cross(N, T);
    const vec3 orig = P + N * SHADOW_EPS;
    openness = 1.0;

    if (stochastic) {
        // Real stochastic 1-bounce GI (accumulated + denoised downstream).
        uint seed = uint(px.x) * 1973u + uint(px.y) * 9277u + frame * 26699u + 0x9e3779b9u;
        const int N_GI = nGI;// adaptive: fresh/disoccluded pixels get more rays so they
                             // converge with SAMPLES (crisp) instead of a wide blur (cloud)
        // BLUE-NOISE DITHERED cosine hemisphere sampling (Heitz & Belcour 2019).
        // The hemisphere (u1,u2) is a per-sample Hammersley base (stratified WITHIN
        // the pixel → clean integration) rotated by a per-pixel/per-frame blue-noise
        // offset (Cranley-Patterson). The result is decorrelated ACROSS neighbouring
        // pixels in a BLUE-NOISE pattern, so the SVGF à-trous averages the 1-spp
        // residual cleanly. Plain white noise (the old rnd(seed)×2) leaves
        // low-frequency clumps the edge-stopping filter can't separate from signal —
        // the AO "speckle". The offset animates with `frame`, so the temporal
        // accumulator still integrates fresh directions every frame.
        const vec2 bnOff = vec2(blueNoiseDef(uvec2(px), frame, 0u),
                                blueNoiseDef(uvec2(px), frame, 1u));
        vec3  acc   = vec3(0.0);
        float missN = 0.0;
        float movN  = 0.0;// rays whose 1-bounce hit a MOVING mesh (→ gGiMovFrac dwell cut)
        for (int s = 0; s < N_GI; ++s) {
            const float u1 = fract((float(s) + 0.5) / float(N_GI) + bnOff.x);
            const float u2 = fract(radicalInverse2(uint(s))         + bnOff.y);
            const float r   = sqrt(u1);
            const float phi = TWO_PI * u2;
            const vec3  hemi = vec3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
            const vec3  dir  = normalize(T * hemi.x + B * hemi.y + N * hemi.z);
            bool missed;
            vec3 gi = giRadiance(orig, dir, doShadows, maxLod, seed, missed, envInt);// cheap 1-bounce; cosine pdf cancels
            if (missed) missN += 1.0;
            // FIREFLY CLAMP (safety) — giRadiance has no specular sun-catch and its
            // emitter NEE is internally clamped, so the diffuse GI is already
            // bounded; this just caps any residual emitter spike.
            const float gl = max(max(gi.r, gi.g), gi.b);
            if (gl > 6.0) gi *= 6.0 / gl;
            acc += gi;
            if (gGiRayHitMoved) movN += 1.0;
        }
        gGiMovFrac = movN / float(N_GI);// ray-count fraction (dark occlusion hits count fully)
        // PROBE-GI mode: ambient is a sky-fill term, so gate it by the gather's
        // REAL sky visibility (the deterministic path's openness semantics) —
        // an enclosed corridor must not receive scene ambient it can't see; the
        // probe field supplies the light that ACTUALLY reaches it. The 1-spp
        // miss-fraction noise rides diffInd's temporal + SVGF like the rest of
        // the gather. Probes OFF keeps the cosmetic openness = 1 fill (there is
        // nothing to replace the ambient with — enclosures would just go black).
        if (probeGrid.enabled > 0.5) openness = missN / float(N_GI);
        return acc / float(N_GI);
    }

    // Deterministic AO + gated far≈sky (denoiser off).
    const int   GI_RAYS = 64;
    const float GA = 2.39996323;
    const float AO_RADIUS = 2.0;
    const float SKY_DIST  = 1000.0;
    vec3  acc  = vec3(0.0);
    float open = 0.0;
    vec3  hitHack = vec3(0.0);
    for (int s = 0; s < GI_RAYS; ++s) {
        const float u1  = (float(s) + 0.5) / float(GI_RAYS);
        const float phi = float(s) * GA;
        const float r   = sqrt(u1);
        const vec3  hemi = vec3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
        const vec3  dir  = normalize(T * hemi.x + B * hemi.y + N * hemi.z);
        rayQueryEXT rq;
        // kRayMaskOpaque: blend decals / glass must not block sky visibility —
        // with 0xFF + force-opaque, a transparent text quad on a hull painted
        // a dark env-occlusion rectangle onto the surface behind it.
        rayQueryInitializeEXT(rq, topAS, gl_RayFlagsOpaqueEXT, kRayMaskOpaque, orig, 1e-3, dir, SKY_DIST);
        while (rayQueryProceedEXT(rq)) {}
        if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
            acc  += sampleEnvLod(dir, maxLod) * envInt;
            open += 1.0;
        } else {
            const float t = rayQueryGetIntersectionTEXT(rq, true);
            hitHack += clamp(t / AO_RADIUS, 0.0, 1.0) * sampleEnvLod(dir, maxLod) * envInt;
        }
    }
    openness = open / float(GI_RAYS);
    acc += hitHack * smoothstep(0.0, 0.03, openness);
    return acc / float(GI_RAYS);
}

// Exact diffuse irradiance from a quad area light — Lambert's polygon form factor (the
// identity-transform case of LTC). Smooth + correct (no closest-point hotspots; a
// physically-exact soft area-light falloff), closed-form (no LUT). Returns the Lambertian factor
// Φ so that diffuse_out = diffuseColor · Φ · lightRadiance. Corners c0..c3 in winding
// order; max(.,0) clamps the back/under-horizon result.
float rectFormFactor(vec3 N, vec3 P, vec3 c0, vec3 c1, vec3 c2, vec3 c3) {
    const vec3 v0 = normalize(c0 - P);
    const vec3 v1 = normalize(c1 - P);
    const vec3 v2 = normalize(c2 - P);
    const vec3 v3 = normalize(c3 - P);
    float f = 0.0;
    f += acos(clamp(dot(v0, v1), -1.0, 1.0)) * dot(N, normalize(cross(v0, v1)));
    f += acos(clamp(dot(v1, v2), -1.0, 1.0)) * dot(N, normalize(cross(v1, v2)));
    f += acos(clamp(dot(v2, v3), -1.0, 1.0)) * dot(N, normalize(cross(v2, v3)));
    f += acos(clamp(dot(v3, v0), -1.0, 1.0)) * dot(N, normalize(cross(v3, v0)));
    return max(f * (1.0 / TWO_PI), 0.0);
}

// Emissive + direct analytic lights (optionally shadowed) + approximate
// diffuse IBL (× ambient occlusion). Shared by the primary surface and the
// reflection hit (so
// reflected geometry is lit + shadowed). Does NOT add the specular lobe — the
// caller adds it (RT reflection for the primary, env IBL for the hit) to avoid
// recursion.
// Pick weight for the single-light estimators (cheapHit / giRadiance):
// premultiplied colour luminance (= intensity), distance-attenuated for
// point/spot. ZERO-power lights (e.g. the ocean's day-mode moon + lighthouse
// beam at intensity 0 — uploaded regardless) get ZERO pick probability; a
// uniform pick wasted most samples on them and flickered the water
// reflections dark. The pick pdf w/W is compensated exactly (×W/w) → any
// positive weight set is unbiased; proportional weights just cut variance.
float lightPickWeight(uint gi, vec3 P) {
    const vec3 LUM = vec3(0.2126, 0.7152, 0.0722);
    if (gi < lights.dirCount) {
        return dot(lights.dirLights[gi].color, LUM);
    } else if (gi < lights.dirCount + lights.pointCount) {
        const uint i = gi - lights.dirCount;
        const vec3 d = lights.pointLights[i].position - P;
        return dot(lights.pointLights[i].color, LUM) / (1.0 + dot(d, d));
    }
    const uint i = gi - lights.dirCount - lights.pointCount;
    const vec3 d = lights.spotLights[i].position - P;
    return dot(lights.spotLights[i].color, LUM) / (1.0 + dot(d, d));
}

// Power-proportional single-light pick. Returns the global light index (or
// 0xFFFFFFFF when no light has power) and writes the exact compensation
// factor W/w_pick.
uint pickAnalyticLight(vec3 P, inout uint seed, out float wPick) {
    wPick = 1.0;
    const uint nL = lights.dirCount + lights.pointCount + lights.spotCount;
    if (nL == 0u) return 0xFFFFFFFFu;
    float wSum = 0.0;
    for (uint i = 0u; i < nL; ++i) wSum += lightPickWeight(i, P);
    if (wSum <= 1e-8) return 0xFFFFFFFFu;
    const float xi = rnd(seed) * wSum;
    float acc = 0.0;
    for (uint i = 0u; i < nL; ++i) {
        const float w = lightPickWeight(i, P);
        acc += w;
        if (xi <= acc && w > 1e-8) {
            wPick = wSum / w;
            return i;
        }
    }
    return 0xFFFFFFFFu;// numeric edge: treat as no pick
}

// Set by the PRIMARY demod path only (never at reflection/GI hits): the
// dir/point/spot loops below are skipped because the denoised-shadow channel
// owns them — analyticDirectSplit computes the exact unshadowed sum and the
// denoise recombine adds U × R̃. Rect lights + emissive NEE stay inline.
bool gSkipAnalyticDirect = false;

vec3 shadeDiffuseDirect(vec3 P, vec3 N, vec3 V, vec3 albedo, float roughness,
                        float metalness, vec3 emissive, bool doShadows, vec3 diffuseIndirect,
                        vec3 sheenColor, float sheenRoughness,
                        float specularIntensity, vec3 specularColor,
                        float iridescence, float iridescenceIOR, float iridescenceThicknessNm,
                        inout uint seed, bool addEmissive, bool cheapHit) {
    const float NdotV = max(dot(N, V), 1e-4);
    vec3        F0           = mix(vec3(0.04) * specularIntensity * specularColor, albedo, metalness);
    // Thin-film iridescence (KHR_materials_iridescence) — shift the Fresnel base
    // before any lobe uses it so the direct analytic lighting matches the path
    // tracer. Skipped when factor == 0 (non-iridescent surfaces pay only the branch).
    if (iridescence > 0.0) {
        const vec3 irid = evalIridescence(1.0, iridescenceIOR, NdotV, iridescenceThicknessNm, F0);
        F0 = mix(F0, irid, iridescence);
    }
    const vec3  diffuseColor = albedo * (1.0 - metalness);
    const float k            = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    const vec3  shadowOrig   = P + N * SHADOW_EPS;

    vec3 lit = emissive;

    // REFLECTION-HIT cheap mode (cheapHit): ONE stochastically-picked analytic
    // light (×nL compensation), a 1-ray jittered rect shadow, and a small
    // emitter NEE. The reflection channel rides the ReBLUR temporal
    // accumulation + the roughness-driven spatial blur, which absorb the
    // estimator noise; the PRIMARY path keeps the exhaustive loops.
    uint  pickIdx = 0xFFFFFFFFu;
    float wPick   = 1.0;
    // cheapHit (reflection / denoised hits) normally samples ONE analytic light
    // to keep reflected-hit cost down, leaning on SVGF to average the pick. But
    // with only a handful of analytic lights that stochastic 1-of-N pick is pure
    // variance for no real saving — and in VIEW-DEPENDENT reflections SVGF can't
    // fully average it (the reflection history resets under camera motion), so it
    // shows up as non-deterministic point/spot-light shimmer on metals. (The
    // DIFFUSE/primary path is clean precisely because it always loops every
    // light.) For small light counts, loop them all here too: deterministic,
    // noise-free, at ~nLights shadow rays per reflected hit — no more than the
    // exhaustive eval the primary path already runs. Keep the 1-pick only when
    // there are many lights, where looping all would be the real cost.
    const uint nLights = lights.dirCount + lights.pointCount + lights.spotCount;
    const bool pickOne = cheapHit && (nLights > 8u);
    if (pickOne) pickIdx = pickAnalyticLight(P, seed, wPick);

    for (uint i = 0u; !gSkipAnalyticDirect && i < lights.dirCount; ++i) {
        if (pickOne && i != pickIdx) continue;
        const vec3 L = normalize(lights.dirLights[i].direction);
        if (dot(N, L) <= 0.0) continue;
        // Soft sun shadow: adaptive multi-ray disc visibility (see sunShadowVis;
        // pc.sunTanHalfAngle = 0 → exact hard single-ray shadow). The BRDF
        // response keeps the exact L — only the occlusion test softens.
        const float vis = (doShadows
                ? sunShadowVis(shadowOrig, L, pc.sunTanHalfAngle, cheapHit)
                : 1.0) * wPick;
        if (vis <= 0.0) continue;
        // The dapple, on the INLINE path (denoise off, dispatch B, glass/sheen)
        // and at every reflection/GI hit — the same multiplier analyticDirectSplit
        // applies on the demod path, so a submerged plate wears the same net
        // whichever of the two computed it, and so does its reflection. 1.0 with
        // no ocean / no murk / above the surface, which is every scene that has
        // never called setUnderwaterMurk.
        lit += evalLight(N, V, L, NdotV, F0, albedo, roughness, metalness, k, sheenColor, sheenRoughness)
               * lights.dirLights[i].color * vis * murkSunCaustic(P, L);
    }
    for (uint i = 0u; !gSkipAnalyticDirect && i < lights.pointCount; ++i) {
        if (pickOne && (lights.dirCount + i) != pickIdx) continue;
        vec3        toL  = lights.pointLights[i].position - P;
        const float dist = length(toL);
        if (dist < 1e-4) continue;
        toL /= dist;
        if (dot(N, toL) <= 0.0) continue;
        const float vis = (doShadows ? shadowVis(shadowOrig, toL, dist - 1e-2) : 1.0) * wPick;
        if (vis <= 0.0) continue;
        const float decay = lights.pointLights[i].decay;
        float atten = 1.0 / max(pow(dist, decay), 0.01);
        const float range = lights.pointLights[i].range;
        if (range > 0.0) {
            const float t  = dist / range;
            const float t4 = t * t * t * t;
            const float wnd = max(1.0 - t4, 0.0);
            atten *= wnd * wnd;
        }
        lit += evalLight(N, V, toL, NdotV, F0, albedo, roughness, metalness, k, sheenColor, sheenRoughness)
               * lights.pointLights[i].color * atten * vis;
    }
    for (uint i = 0u; !gSkipAnalyticDirect && i < lights.spotCount; ++i) {
        if (pickOne && (lights.dirCount + lights.pointCount + i) != pickIdx) continue;
        vec3        toL  = lights.spotLights[i].position - P;
        const float dist = length(toL);
        if (dist < 1e-4) continue;
        toL /= dist;
        if (dot(N, toL) <= 0.0) continue;
        const float spotCos   = dot(-toL, lights.spotLights[i].direction);
        const float spotAtten = smoothstep(lights.spotLights[i].cosAngleOuter,
                                           lights.spotLights[i].cosAngleInner, spotCos);
        if (spotAtten <= 0.0) continue;
        const float vis = (doShadows ? shadowVis(shadowOrig, toL, dist - 1e-2) : 1.0) * wPick;
        if (vis <= 0.0) continue;
        const float decay = lights.spotLights[i].decay;
        float atten = 1.0 / max(pow(dist, decay), 0.01);
        const float range = lights.spotLights[i].range;
        if (range > 0.0) {
            const float t  = dist / range;
            const float t4 = t * t * t * t;
            const float wnd = max(1.0 - t4, 0.0);
            atten *= wnd * wnd;
        }
        atten *= spotAtten;
        lit += evalLight(N, V, toL, NdotV, F0, albedo, roughness, metalness, k, sheenColor, sheenRoughness)
               * lights.spotLights[i].color * atten * vis;
    }

    // Rect area lights (analytic, representative-point — Karis 2013). RectAreaLights are
    // NOT geometry in the BVH, so reflection rays can't hit them: BOTH the diffuse fill
    // AND the specular highlight must be computed HERE (in shadeDiffuseDirect, so PRIMARY
    // and REFLECTED hits get them — a metal reflecting a rect-lit wall stays correct, and
    // a metal facing the rect shows its highlight). Noise-free (no sampling): the rect is
    // approximated by its closest point to P for diffuse and its closest point to the
    // reflection ray (the most-representative-point) for specular.
    for (uint i = 0u; i < lights.rectCount; ++i) {
        const RectLight rl = lights.rectLights[i];
        const float uLen = length(rl.halfU);
        const float vLen = length(rl.halfV);
        if (uLen < 1e-6 || vLen < 1e-6) continue;
        const vec3  uN   = rl.halfU / uLen;
        const vec3  vN   = rl.halfV / vLen;
        const float area = 4.0 * uLen * vLen;

        // DIFFUSE — exact polygon form factor (smooth, physically exact; no hotspots),
        // gated by a SOFT shadow: average visibility over a jittered 4×4 grid of points
        // on the rect (one shadow ray each) → a real penumbra instead of the hard single-
        // ray shaft (the window-frame occlusion was projecting sharp light cones). The
        // per-pixel jitter (stable PCG hash → no flicker) turns the grid's banding into
        // fine high-frequency grain. The smooth form factor carries the falloff.
        {
            const vec3  toC   = rl.position - P;
            const float distC = length(toC);
            if (distC > 1e-4 && dot(toC, rl.normal) < 0.0) {// P in front of the lit face
                const vec3  c0 = rl.position - rl.halfU - rl.halfV;
                const vec3  c1 = rl.position + rl.halfU - rl.halfV;
                const vec3  c2 = rl.position + rl.halfU + rl.halfV;
                const vec3  c3 = rl.position - rl.halfU + rl.halfV;
                const float ff = rectFormFactor(N, P, c0, c1, c2, c3);
                if (ff > 0.0) {
                    float vis = 1.0;
                    if (doShadows) {
                        // Blue-noise-like jitter via Interleaved Gradient Noise (Jiménez
                        // 2014), replacing the old white-noise PCG hash so the per-pixel
                        // penumbra error is spectrally blue (reads smooth, not clumpy).
                        //
                        // PRIMARY hits (SIDE=4, the clean raster base — NOT temporally
                        // accumulated) keep a STATIC per-pixel pattern: deterministic, no
                        // flicker (the property the static jitter was protecting).
                        // REFLECTION / denoised hits (cheapHit, SIDE=1) are fed through the
                        // reflection SVGF temporal pass, so ANIMATE the pattern per frame:
                        // a static 1-sample shadow is a fixed pattern the temporal denoiser
                        // cannot average — that's what baked in as a static weave + black
                        // speckle on the reflections of distant metals. Per-frame rotation
                        // lets SVGF integrate the single sample into a smooth soft shadow
                        // (it joins the GGX ray + light pick, already animated via reflSeed).
                        vec2 fpx = vec2(gl_GlobalInvocationID.xy);
                        if (cheapHit) fpx += float(pc.frame & 0xFFFFu) * vec2(0.7548776662, 0.5698402910);
                        const float jx  = fract(52.9829189 * fract(dot(fpx,    vec2(0.06711056, 0.00583715))));
                        const float jy  = fract(52.9829189 * fract(dot(fpx.yx, vec2(0.06711056, 0.00583715)) + 0.5));
                        const uint SIDE = cheapHit ? 1u : 4u;// 1-ray jittered at reflection hits
                        float vsum = 0.0;
                        for (uint sj = 0u; sj < SIDE; ++sj)
                        for (uint si = 0u; si < SIDE; ++si) {
                            const float fu = (float(si) + jx) / float(SIDE) * 2.0 - 1.0;
                            const float fv = (float(sj) + jy) / float(SIDE) * 2.0 - 1.0;
                            const vec3  sp = rl.position + fu * rl.halfU + fv * rl.halfV;
                            const vec3  sl = sp - shadowOrig; const float sd = length(sl);
                            vsum += (sd > 1e-4) ? shadowVis(shadowOrig, sl / sd, sd - 1e-2) : 1.0;
                        }
                        vis = vsum / float(SIDE * SIDE);
                    }
                    lit += diffuseColor * rl.color * (ff * vis);
                }
            }
        }

        // SPECULAR — most-representative-point: closest point on the rect to the reflection
        // ray, then a GGX lobe WIDENED by the rect's angular size so the highlight spreads
        // to ~the light's apparent size (a glossy/metal surface shows the rect).
        {
            const vec3  R   = reflect(-V, N);
            const float rdn = dot(R, rl.normal);
            vec3 plane = rl.position;
            if (abs(rdn) > 1e-4) {
                const float t = dot(rl.position - P, rl.normal) / rdn;
                if (t > 0.0) plane = P + R * t;
            }
            const vec3 ds  = plane - rl.position;
            const vec3 mrp = rl.position + clamp(dot(ds, uN), -uLen, uLen) * uN
                                         + clamp(dot(ds, vN), -vLen, vLen) * vN;
            vec3 L = mrp - P; const float dist = length(L);
            if (dist > 1e-4) {
                L /= dist;
                const float ndl  = dot(N, L);
                const float cosE = dot(-L, rl.normal);
                if (ndl > 0.0 && cosE > 0.0) {
                    const float vis = doShadows ? shadowVis(shadowOrig, L, dist - 1e-2) : 1.0;
                    if (vis > 0.0) {
                        const vec3  H     = normalize(V + L);
                        const float NdotH = max(dot(N, H), 0.0);
                        const float angR  = 0.5 * sqrt(area) / dist;               // ~angular radius
                        const float ar    = clamp(roughness + angR, roughness, 1.0);// widen the lobe
                        const float D     = distGGX(NdotH, ar);
                        const float G     = geomSmithG1(NdotV, k) * geomSmithG1(ndl, k);
                        const vec3  F     = fresnelSchlick(max(dot(V, H), 0.0), F0);
                        // Bound the multiplier by the WIDENED lobe's own solid
                        // angle (≈ 4π·α² with α = ar²): D grew as the lobe
                        // narrowed but still swept the light's FULL solid angle,
                        // so a near-mirror surface returned ~8× the light's own
                        // radiance (a mirror cannot out-shine what it reflects —
                        // the ceiling is F·L). The α⁴ cap lands the mirror limit
                        // exactly there and is a no-op on rough surfaces, where
                        // the lobe is already wider than the light.
                        const float solid = min(min(area * cosE / (dist * dist), TWO_PI),
                                                4.0 * PI * ar * ar * ar * ar);
                        lit += (D * G) * F / max(4.0 * NdotV * ndl, 1e-4) * rl.color * (solid * ndl * vis);
                    }
                }
            }
        }
    }

    // Emissive area lights via the coherent emissiveNEE. The CALLER controls when to
    // add it (addEmissive): the PRIMARY surface skips it when ReSTIR DI is on (main()
    // folds the reservoir result into the denoised diffuse-indirect channel instead),
    // but REFLECTED hits (traceRadiance) ALWAYS add it — reservoirs are per-screen-pixel
    // and don't exist for reflected surfaces, so without this metals reflecting an
    // emissive-lit room lose their reflected lighting (= "metals lost their color").
    if (addEmissive)
        lit += emissiveNEE(cheapHit ? 4 : 16, P, N, V, NdotV, F0, albedo, roughness, metalness, k, seed, doShadows);

    // Diffuse indirect (caller-supplied): visibility-gated env irradiance for
    // the primary surface, crude env IBL for reflected hits. Only modulates the
    // indirect term — direct lights have their own shadows.
    lit += diffuseIndirect * diffuseColor;
    // Env/IBL sheen — grazing-rim fabric glow under image-based light. This is
    // what actually shows on an env-lit fabric (e.g. the satin cushion), where
    // there's no analytic light for the per-light Charlie lobe above.
    if (dot(sheenColor, sheenColor) > 0.0)
        lit += sheenColor * IBLSheenBRDF(NdotV, sheenRoughness) * diffuseIndirect;
    return lit;
}
