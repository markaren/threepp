// 1-bounce diffuse GI shared by deferred_shade.comp (giRadiance, and the light
// pick in shadeDiffuseDirect) and probe_update.comp (probeRayRadiance): the
// point/spot distance falloff, the power-proportional analytic-light pick, and
// the direct light at a GI ray's hit. One copy, so the probe grid and the
// per-pixel gather shade a bounce the same way.
//
// The includer declares, BEFORE including this file:
//   lights  — the LightsUbo block (at least the dir/point/spot prefix)
//   float rnd(inout uint seed)
//   float shadowVis(vec3 origin, vec3 dir, float tMax)
//   vec3  emissiveIrradiance(vec3 P, vec3 N, int samples, bool doShadows)
//   vec3  giSunLeg(vec3 P, inout vec3 L) — a directional light's leg to P:
//         deferred_shade bends L into the water and returns the in-water
//         transmittance per channel (murkSunLeg); probe_update has no murk and
//         returns 1.0.

// Distance term of the point/spot falloff, d^decay. decay is a per-light uniform
// copied verbatim from the host, so it round-trips through the UBO bit-exact and
// the two values that occur in practice — PointLight's default 1 and the physical
// inverse-square 2 — can be matched with == and evaluated without pow()'s log2/exp2
// pair. Avoiding pow() also removes a source of cross-vendor drift: GLSL specifies
// pow() only to 16 ULP and every driver lowers it differently, while d and d*d are
// exact everywhere. decay 0 (and any authored fractional decay) still takes pow.
float distFalloff(float d, float decay) {
    if (decay == 1.0) return d;
    if (decay == 2.0) return d * d;
    return pow(d, decay);
}

// Pick weight for the single-light estimators (cheapHit / giRadiance / probe
// rays): premultiplied colour luminance (= intensity), distance-attenuated for
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

// Direct light leaving a GI ray's hit toward the ray origin: the analytic
// lights and a 2-sample emitter NEE, times the hit's Lambert term
// diff = albedo·(1 − metalness)/π. No emission of the hit itself, no
// specular, no indirect — the callers add what their energy accounting owns.
// shadowOrig = hitP offset along hitN by the caller's shadow epsilon.
//
// With MANY lights, sample ONE (×wL compensation, unbiased) — the GI channel
// and the probe EMA both average the selection noise, and this keeps lamp-heavy
// interiors cheap (1 shadow ray vs nLights). But for a SMALL light count that
// 1-of-N pick is pure variance the denoiser only SMEARS — worst for
// concentrated lights like SPOTLIGHTS, whose tight cone makes "is this bounce
// point in the cone?" a high-contrast coin-flip per gather ray. So for
// nLights<=8 loop them ALL (deterministic, ~nLights shadow rays/hit) — mirrors
// the cheapHit reflection path; the 1-pick stays only where looping all would
// be the real cost.
vec3 giHitDirect(vec3 hitP, vec3 hitN, vec3 shadowOrig, vec3 diff, bool doShadows, inout uint seed) {
    vec3 lit = vec3(0.0);
    const uint nLights = lights.dirCount + lights.pointCount + lights.spotCount;
    const bool pickOne = nLights > 8u;
    float wL = 1.0;
    const uint pick = pickOne ? pickAnalyticLight(hitP, seed, wL) : 0xFFFFFFFFu;

    for (uint i = 0u; i < lights.dirCount; ++i) {
        if (pickOne && i != pick) continue;
        vec3        L   = normalize(lights.dirLights[i].direction);
        const vec3  leg = giSunLeg(hitP, L);
        const float ndl = dot(hitN, L);
        if (ndl <= 0.0) continue;
        const float vis = doShadows ? shadowVis(shadowOrig, L, 1e30) : 1.0;
        lit += diff * ndl * lights.dirLights[i].color * (vis * wL * leg);
    }
    for (uint i = 0u; i < lights.pointCount; ++i) {
        if (pickOne && (lights.dirCount + i) != pick) continue;
        vec3        toL  = lights.pointLights[i].position - hitP;
        const float dist = length(toL);
        if (dist < 1e-4) continue;
        toL /= dist;
        const float ndl = dot(hitN, toL);
        if (ndl <= 0.0) continue;
        float atten = 1.0 / max(distFalloff(dist, lights.pointLights[i].decay), 0.01);
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
        float atten = spotAtten / max(distFalloff(dist, lights.spotLights[i].decay), 0.01);
        const float range = lights.spotLights[i].range;
        if (range > 0.0) { const float tt = dist / range; const float t4 = tt*tt*tt*tt; const float wnd = max(1.0 - t4, 0.0); atten *= wnd * wnd; }
        if (atten <= 1e-6) continue;
        const float vis = doShadows ? shadowVis(shadowOrig, toL, dist - 1e-2) : 1.0;
        lit += diff * ndl * lights.spotLights[i].color * (atten * vis * wL);
    }
    // Emitter 1-bounce (e.g. enclosed scene lit only by an emissive sphere) —
    // cheap small-sample diffuse NEE so the colour bleed survives the denoiser.
    lit += diff * emissiveIrradiance(hitP, hitN, 2, doShadows);
    return lit;
}
