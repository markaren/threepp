// Split from deferred_shade.comp: thin-shell water shading (RasterFirst),
// in-glass interior transit marching, and refractive glass shading.

// Term isolation for water-artefact triage (-DSCOUT_WATER=N):
//   1 = reflection only, 2 = transmission only, 3 = Fresnel F as grey,
//   4 = foam mask as grey, 5 = |N - up| amplified, 6 = the reflection
//   hit's distance class, 7 = sun visibility, 8 = how the reflection ray ended (see there).
//   0 = normal shading. deferred_shade_40 is included first and sets the
//   default, so a define given here alone does not take.
#ifndef SCOUT_WATER
#define SCOUT_WATER 0
#endif
// Crest foam triage (-DCREST_FOAM_SCOUT=N, kOceanCrestFoam frames only):
//   1 = a 4 m checker on the rest coordinates q under the raw foam field F,
//   2 = the foam cover as grey. 0 = normal shading.
#ifndef CREST_FOAM_SCOUT
#define CREST_FOAM_SCOUT 0
#endif

// Forward declarations — the underwater medium lives in deferred_shade_60, which
// is included AFTER this file (the split is ordered by what the shade's main()
// needs first, not by call graph). shadeWater's from-below branch needs both:
// the gate, and applyMurk to composite the murk onto its traced reflection leg.
bool camUnderwater();
bool murkLive();
vec3 applyMurkSky(vec3 dir);
bool camPortWetDryEye();
vec3 applyMurk(vec3 col, vec3 ro, vec3 hit);
// The air medium on shadeWater's reflected leg (also deferred_shade_60).
vec3 applyHeteroSurfaceFog(vec3 col, vec2 fuv, float viewDist, vec3 ro, vec3 hit);
vec3 applySkyFog(vec3 sky, vec3 dir);
vec3 applyMurkWaterLeg(vec3 col, vec3 ro, vec3 hit);
// The Henyey-Greenstein phase the fog/particle/cloud volumetrics all use — the
// water column gets the SAME phase function, not a second one (hg_phase.glsl,
// included by deferred_shade_60).
float hgPhase(float mu, float g);
vec3 applySkyFog(vec3 sky, vec3 dir);

// ── Water-body IN-SCATTER calibration ───────────────────────────────────────
// kSky ties the in-scatter's skylight source to the SAME scale the analytic
// deep body already uses (`skyLum * 0.35` below), so the two terms are one
// calibration, not two. A fully scattering column (single-scattering albedo
// -> 1, leg >> 1/sigma_t) then in-scatters exactly the skylight the old body
// showed at tint == 1 — it reproduces that brightness rather than adding a
// second, independently scaled copy of it.
const float kWaterInScatterSky = 1.7;
// Forward-scattering water: suspended matter throws light along the ray far
// more than back at it. 0.75 is the fog/cloud family's mid-forward value.
const float kWaterInScatterG = 0.75;
// The HG lobe peaks at 28x isotropic for g = 0.75 (mu -> 1), i.e. looking down
// the sun's own refracted direction. Single scattering has no multiple-bounce
// budget to spend that on, so the spike is capped — the same reasoning as the
// glint loop's fireflyClamp, at the one place it can bite.
const float kWaterInScatterPhaseMax = 1.6;
// Shadow samples of the sun's in-scatter down the water column (see its use).
const int kWaterColumnShadowSamples = 3;
const float kWaterMeanFresnelCos = 0.18;

// The murk is only the WATER BODY once it is dense enough to BE water. Below
// this the scene is using setUnderwaterMurk as a haze/absorption stand-in, and
// the one-medium branch (the `else if` in shadeWater's transmission, below)
// would then paint the whole column with a near-transparent medium's in-scatter:
// applyMurkSky is the t→∞ limit, so a σ_t of 0.0004 /m still returns
// murkColor·fogLight at full strength and the water goes milk-grey from above.
// 0.01 /m is an extinction length of 100 m — 600 m of visibility at the 6
// optical depths the branch budgets. No sea is that clear: the clearest open
// ocean (Jerlov I) runs ~0.03-0.04 /m and coastal water an order more, while
// 0.01 and below is the range a demo picks when it wants a faint tint over a
// kilometre, i.e. haze. A scene under the threshold keeps the material's own
// Beer-Lambert body from above, which is what it was tuned against; from BELOW
// nothing changes, because that branch and every other murk path are still on
// murkLive() alone.
const float kMurkIsWater = 0.01;

// Foam after Crest's, sampled at the REST position xz so it rides the waves.
// Shape: the texture thresholded by the foam value, so foam thins into the texture's holes
// as it fades. Volume: a blurred layer of bubbles under the surface, shifted by parallax.
// Light: a bump normal from the same texture, less sky in the troughs (pm.sheenRoughness
// carries the sea level as (y + 50) / 100). Foam keeps its own lit colour, without reflected water colour.
float gFoamBoil = 0.0;// how much of a boiling wash this pixel's foam is
bool  gFoamCarried = false;// in the wash's flow map: its pattern's own animation runs at the wash's slower pace
// THREEPP_OCEAN_FOAM_KILLS_SPEC (oceanOptions2 bit 8): the mirror part of this pixel's water colour (F x reflection, the sun's and
// the lamps' glints) as it stands now, so foam and aeration can take it out instead of alpha-blending over it.
vec3 gWaterSpec = vec3(0.0);
// Foam and the air under it scatter: they do not mirror (white wallpaper is not a mirror). Over the local cover of foam
// or boiling wash the water's specular gives way to the sky seen diffusely (F x the fully blurred sky), gone by 25 %
// cover. Cover is aeration, not drawn foam: the wash's energy (gWake.r), its boil (gWake.g) and the foam field's raw
// amount, read once before any foam is drawn. The foam drawn on top afterwards is unchanged.
vec3 foamKillSpec(vec3 col, float cover, vec3 N, float F, float maxLod) {
    const float k = smoothstep(0.03, 0.25, cover);
    if (k <= 0.0) return col;
    const vec3 rough = F * sampleEnvLod(N, maxLod);
    col += k * (rough - gWaterSpec);
    gWaterSpec = mix(gWaterSpec, rough, k);
    return col;
}
// warp in metres BEFORE wind stretching, avoiding rectangular shoulders.
vec2 foamEdgePosition(vec2 xz) {
    const mat2 turn = mat2(0.8, 0.6, -0.6, 0.8);
    vec2 q = turn * xz * 0.18;
    return xz + 6.0 * (vec2(fbm4(q + vec2(11.7,83.1)),
                               fbm4(q + vec2(47.3,19.8))) - vec2(0.46875));
}
// the cap trail's patches, a noise of about 2 m fixed to the water (a slow drift), on a domain warped by
// a third of a metre over 4.5 m so that no two patches share a shape or a row (at a metre the trail was marbled, swirls
// where the wind draws streaks); a little longer along the wind than across.
// Returned in standard deviations about its mean (x), with the same noise at 0.75 m (y) to break a patch up inside.
vec2 capTrailPatches(vec2 xz, vec2 wd, float t) {
    const mat2 turn = mat2(0.8, 0.6, -0.6, 0.8);
    const vec2 q = turn * xz * 0.22;
    const vec2 w = xz + 3.0 * (vec2(fbm4(q + vec2(31.4, 7.9)), fbm4(q + vec2(3.3, 58.2))) - vec2(0.467));
    const vec2 p = vec2(dot(w, wd) * 0.4, dot(w, vec2(-wd.y, wd.x)) * 0.6) + vec2(0.05, -0.04) * t + 2.1;
    return (vec2(fbm4(p), fbm4(2.7 * p + 9.1)) - 0.467) / 0.124;
}
// fbm4 with each octave faded to its mean once its period is under two footprints (fp in p units)
float fbm4Lim(vec2 p, float fp) {
    const mat2 turn = mat2(0.8, -0.6, 0.6, 0.8) * 2.03;
    float a = 0.0, w = 0.5, f = 1.0;
    for (int i = 0; i < 4; ++i) {
        a += w * gnoise21(p) * (1.0 - smoothstep(0.25, 0.5, f * fp));
        p = turn * p;
        w *= 0.5;
        f *= 2.03;
    }
    return 0.467 + 1.218 * a;
}
float foamTexture(vec2 xz, float lod) {
    // soft clumps of 1.5-4 m on a warped, churning domain, broken by finer octaves and gathering along curving
    // ridges, with bubbles in it, at every distance; the finer octaves give way to their mean as they stop resolving. The
    // Voronoi veins this used past ~15 m (foamFilament, 72 % of the far texture) drew storm foam as a uniform net of thin
    // bright lines round dark holes, cell noise rather than foam; close up foamclose had already replaced them.
    // Churn: the warp evolves and the layers slide against each other, so a patch's outline keeps changing while its
    // position rides the water.
    const float t = pc.timeSec * (gFoamCarried ? 0.25 : 1.0);
    const vec2  q = xz * 0.22 + 1.3 * vec2(fbm4(xz * 0.13 + vec2(0.09, -0.05) * t), fbm4(xz * 0.13 + vec2(4.1, 2.7) + vec2(-0.06, 0.08) * t));
    const vec2  qv = q + vec2(0.04, 0.03) * t;
    // THREEPP_OCEAN_BAND_FOOTPRINT_FADE: band-limited octaves. Each fbm4 octave whose wavelength is under two pixels'
    // footprint goes to its mean instead of being point-sampled: a, b and c reach wavelengths of 0.42, 0.14 and 0.05 m,
    // and past ~0.1 m a pixel they aliased into the grain that far caps and the storm spill showed (the old fades of b and
    // c started at 0.5 and 0.12 m). q runs at about 0.22 per metre of xz.
    const bool limF = (pc.oceanOptions2 & 1024u) != 0u;
    const float fpQ = 0.0078 * exp2(lod) * 0.22;   // the footprint in q units
    const float a = limF ? fbm4Lim(qv * 1.3, fpQ * 1.3) : fbm4(qv * 1.3);
    const float b = limF ? fbm4Lim(qv * 3.9 + 7.1, fpQ * 3.9) : mix(fbm4(qv * 3.9 + 7.1), 0.467, smoothstep(6.0, 8.0, lod));
    const float c = limF ? fbm4Lim(qv * 11.0 - 2.3, fpQ * 11.0) : mix(fbm4(qv * 11.0 - 2.3), 0.467, smoothstep(4.0, 6.0, lod));
    const float ridge = 1.0 - abs(2.0 * b - 0.94);
    const float soft = mix(0.55 * a + 0.3 * b + 0.15 * c, 0.5 * ridge + 0.22, 0.35);
    // (the detail texture's texel is 1 / (0.6 x 512) = 3.3 mm, so its mip for this footprint is lod + 1.26; lod - 0.5
    // read it 3.4x too sharp, and far bubbles aliased)
    const float bub = textureLod(foamDetailTex, xz * 0.6 + vec2(-0.08, 0.05) * t, limF ? lod + 1.26 : max(lod - 0.5, 0.0)).r;
    // (contrast 3.2 about the mean: the share above the shade's whitening threshold the far veins had, so the cover the
    // foam field sets is drawn the same)
    return clamp(0.54 + 3.2 * (soft - 0.46), 0.0, 1.0) * 0.82 + 0.18 * bub;
}
// residual foam (Beaufort 8-9) lies in streaks along the wind: long, thin, wandering lines a metre or so wide
// and tens of metres long, a few metres apart, broken into segments, drifting downwind at the surface drift. The lines
// are where an anisotropic gradient noise crosses zero (no lattice, no cells); past the distance where a line is under a
// pixel they fade to their area share.
float gStormStreak = 0.0;   // the wind-streak amount at this pixel
float gStormRoller = 0.0;   // the in-place breakers' roller
float gCapBreaking = 0.0;   // THREEPP_OCEAN_CAP_TRAIL: an in-place breaker is breaking at this pixel
float gCapTrailFresh = 0.0; // how fresh the field's trail is here (1 just laid, 0 at about 6 s)
float gCapBreakLine = 0.0;  // the downwind rim of that breaker's lobe
float gCapTrail = 0.0;      // the trail's opacity at this pixel
float gCapTrailBright = 0.0; // how large and fresh the breaking event behind that trail is (its white)
vec2 gFoamWind = vec2(1.0, 0.0);   // set by shadeWater (the material's wind direction)
float foamStreaks(vec2 xz, float footprint) {
    const vec2 wd = gFoamWind, wc = vec2(-wd.y, wd.x);
    vec2 p = vec2(dot(xz, wd) - 0.55 * pc.timeSec, dot(xz, wc));
    p.y += 2.5 * (fbm4(p * vec2(0.012, 0.05)) - 0.467);   // wandering across the wind
    const float n1 = gnoise21(p * vec2(0.02, 0.17));
    const float w1 = 0.06 + 0.2 * footprint;
    const float seg = smoothstep(0.4, 0.6, fbm4(p * vec2(0.01, 0.07) + 3.1));   // broken into segments along the wind
    const float l = (1.0 - smoothstep(0.0, w1, abs(n1))) * seg;
    return mix(l, 0.0, smoothstep(0.3, 1.0, footprint));   // (sub-pixel lines go, not to a grey mean: it made far cover climb)
}
// THREEPP_OCEAN_FOAM_RESIDUE: stage B as filaments. Residue is drawn out by the surface's strain: the Stokes drift's
// shear and the windrows' convergence lay it in thin lines along the wind (the review's item 6), longer the older it is.
// Ridged gradient noise in coordinates stretched along the wind: a filament wherever |n| < w, w wider where the foam is
// denser, so a thin residue is a few hairlines and a fresh one nearly merges. Its coverage is pre-filtered over the
// pixel's footprint (Dupuy & Bruneton 2012's linear pre-filtering): the edge softens by the footprint's spread of |n|, and
// once a pixel spans several filaments it takes their mean cover (2w) rather than aliasing, so it mipmaps to the same cover.
// THREEPP_OCEAN_WATER_BUBBLES: the bubbles' backscatter (1/m) in a plume a breaker has just injected, at Beaufort 10; the water's
// own is about 0.008, so a fresh plume returns some six times the clear column (at the first 0.16 a plume was a flat
// bright dot until it was nearly gone; at a third of that its brightness follows its age)
const float kWaterBubbleBb = 0.05;

float foamFilamentsW(vec2 xz, float footprint, float stretch, float w) {
    const vec2 wd = gFoamWind, wc = vec2(-wd.y, wd.x);
    const float L = 0.7;   // the filaments' cross-wind spacing (m)
    vec2 p = vec2(dot(xz, wd) / (L * stretch), dot(xz, wc) / L);
    p.y += 0.9 * gnoise21(p * vec2(0.6, 0.12) + 7.3);   // they wander and braid
    const float r = abs(gnoise21(p) + 0.35 * gnoise21(2.03 * p + 3.7));
    const float sigma = 0.6 * footprint / L + 0.01;
    const float sharp = 1.0 - smoothstep(w - sigma, w + sigma, r);
    return mix(sharp, min(1.0, 2.0 * w), smoothstep(0.5, 2.0, sigma / w));
}
float foamFilaments(vec2 xz, float footprint, float stretch, float amount) {
    return foamFilamentsW(xz, footprint, stretch, mix(0.12, 0.42, amount));
}
// the cap trail's cover drawn as lace, opaque strands as wide as the foam is dense (their mean cover is the
// cover asked for), not as a translucent sheet. A pixel's footprint at a low look is many times longer along the view
// than across it, and filtered over the long one the strands went to their mean, a film on the wave from deck height:
// they are averaged along it at five points, each with the short footprint, so they stay resolved across the view.
// (soft, 0..1, is the foam's age: a fresh strand's edge is a fifth of its width, an aged one thins out from its middle
// over its whole width, a ramp of 0.9 w to both sides of its edge, at the same mean cover. foamFilamentsW with that
// ramp added. With a cut edge a fresh strand seen at a grazing angle was a white sliver, a glint on the wave.)
float capTrailStrands(vec2 xz, float footprint, float stretch, float w, float soft) {
    const vec2 wd = gFoamWind, wc = vec2(-wd.y, wd.x);
    const float L = 0.7;
    vec2 p = vec2(dot(xz, wd) / (L * stretch), dot(xz, wc) / L);
    p.y += 0.9 * gnoise21(p * vec2(0.6, 0.12) + 7.3);
    const float r = abs(gnoise21(p) + 0.35 * gnoise21(2.03 * p + 3.7));
    const float sigma = 0.6 * footprint / L + 0.01;
    const float e = sigma + mix(0.2, 0.9, soft) * w;
    const float sharp = 1.0 - smoothstep(w - e, w + e, r);
    return mix(sharp, min(1.0, 2.0 * w), smoothstep(0.5, 2.0, sigma / w));
}
// where the cap look hands back to the foam layer's own caps, by the pixel's size across the view (0.001 d):
// 0..1 over 0.08-0.16 m at a low look, over 0.035-0.07 m where the look is steeper than about 35 degrees.
float capTrailHandOver(float d, float h) {
    const float lo = mix(0.08, 0.035, smoothstep(0.25, 0.6, h / max(d, 1e-3)));
    return smoothstep(lo, 2.0 * lo, d * 0.001);
}
float capTrailLace(vec2 xz, vec2 along, float fpS, float fpL, float cover, float stretch, float soft) {
    const float fpT = max(1.5 * fpS, fpL / 6.0);
    float a = 0.0;
    for (int i = 0; i < 5; ++i) a += capTrailStrands(xz + (float(i) - 2.0) * 0.2 * fpL * along, fpT, stretch, 0.5 * cover, soft);
    return 0.2 * a;
}
#ifdef THREEPP_OCEAN_WATER_BODY_EXPLORATION
float debugWaterCover = 0.0;
#endif
vec3 foamLayer(vec3 col, vec3 P, vec3 N, vec3 V, float seaCode, float F, vec3 reflectColor, float foam, vec2 xz0, float footprint,
               bool below, bool doShadows, float maxLod, inout uint seed) {
    if ((pc.oceanOptions & 2097152u) != 0u) return col;
    // A wash boils: where the water is full of air its foam folds over in under a second, and patches die and are born inside the race

#ifdef THREEPP_OCEAN_INSPECT
    if (pc.inspectMode != 0u) {
        if (pc.inspectSolo == 2u) return col;
    }
#endif
    const float boil = smoothstep(0.04, 0.15, gWake.g);
    gFoamBoil = boil;
    const float tB = pc.timeSec;
    const vec2  xz = xz0;
    const float f   = clamp(foam, 0.0, 0.85);
    // below 0.04 and outside a wash this returns col untouched (the aerated tint starts at 0.08, the gate,
    // its shadow and the white at 0.04, the film needs a boil), so skip the textures: the faint tails of the foam field
    // cover much of a storm sea, and each read here is several Voronoi and fbm evaluations.
    if (f <= 0.04 && boil <= 0.0) return col;
    const float lod = log2(max(1.0, footprint / 0.0078));
    const float sky = mix(mix(0.5, 1.0, smoothstep(-1.1, 0.5, P.y - (seaCode * 100.0 - 50.0))), 0.9, boil);
    const vec3  amb = sampleEnvDiffuse(N) + lights.ambient + hemiAmbient(N);
    if (!below) {
        // On the veins too, seen a little deeper: never a flat sheet
        const vec2  par = -0.4 * V.xz / max(dot(N, V), 0.15);
        const float bub = 0.5 + 0.9 * (fbm4((xz + par) * 0.5 + vec2(0.05, -0.04) * pc.timeSec) - 0.47);
        // The aerated water under it: pale, bright with the sky, with no halo round thin foam
        col = mix(col, vec3(0.3, 0.4, 0.43) * amb * sky, smoothstep(0.08, 0.5, f) * smoothstep(0.2, 0.7, bub) * mix(0.25, 0.4, boil));
#ifdef THREEPP_OCEAN_INSPECT
    if (pc.inspectMode != 0u) {
        float retain=1.0-smoothstep(0.08,0.5,f)*smoothstep(0.2,0.7,bub)*mix(0.25,0.4,boil);
        gInspectReflection*=retain; gInspectTransmission*=retain;
    }
#endif

    }
    // A wash's pattern is carried, not animated: a flow map over a varying flow (a stream function's curl, eddies of 4 m and 1.3 m, up to
    // about 3 m/s), so it stretches, shears and curls as it goes. Two read positions reset out of step, each reset hidden while its weight
    // is zero, and the cycle's phase varies over the sea so the whole sheet doesn't breathe at the cycle's period.
    float ft, ero;
    if (boil > 0.0 && lod < 7.5) {
        // (smooth by construction: gradients of a noise were rough enough to shred the pattern into salt)
        vec2 uF = vec2(0.0);
        for (int i = 0; i < 6; ++i) {
            const float fi = float(i);
            const float an = 2.399963 * fi + 0.4;
            const vec2  kv = vec2(cos(an), sin(an)) * (0.45 + 0.42 * fi);// wavelengths from 14 m to 2.4 m
            uF += vec2(-kv.y, kv.x) / length(kv) * 0.19 * cos(dot(kv, xz0) + (0.5 + 0.23 * fi) * 0.25 * tB + 1.7 * fi);
        }
        // The race's stream under the eddies, surging as groups pass (the direction of the race's sources, this scene's heading)
        const float guF = 0.5 + 0.5 * sin(0.71 * tB + 1.0) * sin(0.29 * tB);
        uF += vec2(-0.65, -0.15) * (0.7 + 1.2 * guF * guF);
        const float cyc = tB / 2.8 + 2.0 * fbm4(xz0 * 0.17);
        const float p0 = fract(cyc), p1 = fract(cyc + 0.5);
        const float w0 = 1.0 - abs(2.0 * p0 - 1.0), w1 = 1.0 - w0;
        const vec2  xA = xz0 - boil * uF * p0 * 2.8 + vec2(7.3, 2.9) * floor(cyc);
        const vec2  xB = xz0 - boil * uF * p1 * 2.8 + vec2(-5.1, 8.7) * floor(cyc + 0.5);
        // (the mix of two patterns has less contrast than either: put it back)
        gFoamCarried = true;
        ft  = 0.6 + (w0 * foamTexture(xA, lod) + w1 * foamTexture(xB, lod) - 0.6) / max(sqrt(w0 * w0 + w1 * w1), 0.85);
        ero = 0.45 * (w0 * fbm4(xA * 2.6) + w1 * fbm4(xB * 2.6) - 0.47);
        gFoamCarried = false;
        if (boil < 1.0) {   // into the plain pattern at the wash's edge: no seam
            ft  = mix(foamTexture(xz, lod), ft, boil);
            ero = mix(0.3 * (fbm4(xz * 2.6) - 0.47), ero, boil);
        }
    } else {
        ft  = foamTexture(xz, lod);
        ero = 0.3 * (fbm4(xz * 2.6) - 0.47);
    }
    // (0.72: even the freshest foam keeps some of the texture's holes; the threshold falls as foam
    // thins into lace, and nothing where there is no foam: the texture alone is not foam)
    // Age is what thins it: fresh foam (f near 1) is a nearly solid core, and as f falls the threshold rises through the
    // veins, so a cap breaks into lace and holes before it fades. A slow noise wanders the outline by metres, or the
    // gaussian of a source shows as a rounded oval with a hard rim.
    const float thr   = min(0.95 - 1.1 * f, mix(0.56, 0.42, boil)) + 0.2 * boil + 0.14 * (mix(fbm4(xz * 1.2 + vec2(0.9, -0.6) * pc.timeSec), fbm4(xz * 1.2 + vec2(0.22, -0.15) * pc.timeSec), boil) - 0.47)
                                         + 0.42 * (fbm4(xz * 0.42 + vec2(-0.12, 0.09) * pc.timeSec) - 0.47)
                                         + ero;
    const float gateF = smoothstep(0.04, 0.2 + 0.2 * boil, f);
    const float edgeW = mix(0.18, 0.12, boil) * (1.0 + 0.8 * (1.0 - smoothstep(3.5, 5.5, lod)));   // foamclose: soft edges close up
    // and a film of lace past the edge of the thick foam, so a patch thins out instead of ending at a line
    const float film  = 0.25 * boil * smoothstep(0.68, 0.82, ft) * smoothstep(0.02, 0.12, f);
    // three parts. A fresh cap (f past ~0.5) is a dense bright core: the threshold drops through the texture
    // so it is nearly solid, broken only at its rim. Residual foam (f under ~0.3, the long tail of each cap's decay) is
    // drawn only along the wind streaks, thin and grey. Between them, the water.
    const float core   = smoothstep(0.45, 0.7, f);
    const float resid  = (pc.oceanOptions & 512u) != 0u
        ? smoothstep(0.04, 0.12, f) * (1.0 - core) * (1.0 - boil)
        : smoothstep(0.04, 0.1, f) * (1.0 - smoothstep(0.15, 0.3, f)) * (1.0 - boil);
    // (the cap's texture rises with f alone: masking it by the residue's own ramp left a bright rim round each patch)
    // Dense cores must retain the texture's holes. With f=0.85, subtracting
    // 0.3*core from the already falling threshold puts it below zero: even
    // an empty texel turns opaque, exposing the band's smooth contour as a
    // solid strip/shard. Keep the existing dense-foam texture threshold (0.42)
    // as a floor only at the core; source amount, lifetime and band parameters stay
    // untouched. The diagnostic bit permits a matched before/after capture.
    // Retain the existing outline erosion in the floor too. A constant floor
    // against a filtered, constant texture mean would itself leave an even
    // ribbon at grazing distance, losing the holes as the texture is filtered.
    const float capErosion = (pc.oceanOptions & 4096u) != 0u
        ? thr - (min(0.95 - 1.1 * f, mix(0.56, 0.42, boil)) + 0.2 * boil) : 0.0;
    const float capThreshold = (pc.oceanOptions & 128u) != 0u
        ? max(thr - 0.3 * core, (0.42 + capErosion) * core) : thr - 0.3 * core;
    // THREEPP_OCEAN_BAND_FOOTPRINT_FADE: a pre-filtered threshold (Dupuy & Bruneton 2012's linear pre-filtering). Past
    // lod ~4 foamTexture swaps its finer octaves for their mean; the variance they carried is still inside the pixel, so
    // the cut of the coarse remainder is softened by that lost spread. Without it a far cap was either all over the
    // threshold (a flat slab with a hard rim) or a few-pixel clump of it (speckle). fbm4's spread is ~0.124; the octaves
    // weigh 0.3 and 0.15, times the texture's 3.2 x 0.82 contrast.
    // (an estimate: ft spreads about 0.2-0.3 near the camera, a good part of it from the two finer octaves and the bubbles,
    // which go to their mean between lod 2 and 8; up to 0.2 of it is restored as the cut's softness)
    const float sigLost = (pc.oceanOptions2 & 1024u) != 0u
        ? 0.2 * (0.35 * smoothstep(2.0, 6.0, lod) + 0.35 * smoothstep(4.0, 6.0, lod) + 0.3 * smoothstep(6.0, 8.0, lod)) : 0.0;
    const float capW   = smoothstep(capThreshold - sigLost, capThreshold + edgeW + sigLost, ft) * gateF * max(smoothstep(0.12, 0.3, f), boil);
    // A residue gate selects an age range; it is not optical coverage. The
    // old unit-height gate could turn f=0.1 into alpha=0.9 along an entire
    // windrow, so fading the field exposed smooth white bars. Carry the
    // actual amount into opacity: the same windrow remains, faint and porous.
    const float residueOpacity = (pc.oceanOptions & 256u) != 0u ? f : 1.0;
    // THREEPP_OCEAN_FOAM_RESIDUE (oceanOptions2 bit 3): the residue's wind lines as the residue's own filaments, broken along their length.
    // foamStreaks laid parallel lines a constant 1/0.17 apart over the whole aged patch, which read as grey combed bands
    // behind the caps. Filaments wander and braid, are as wide as the foam is dense, and here are cut every few metres along
    // the wind (a hashed gradient noise stretched 3:1 along it), so the residue thins out as strands and gaps, not as a band.
    float residLines = foamStreaks(xz, footprint);
    // (not on the residue-off path. There the residue is the field's rate-built amount, smooth over metres and
    // over whole aged patches, and the filaments, far denser than these lines, drew it as a pale sheet edged by the cap
    // texture's gate: on Hs 1.5 aged foam, and faintly round the storm caps. The storm slabs were the spume veil.)
    if ((pc.oceanOptions2 & 8u) != 0u) {
        const vec2 wdR = gFoamWind, wcR = vec2(-wdR.y, wdR.x);
        const vec2 pr = vec2(dot(xz, wdR) / 4.5, dot(xz, wcR) / 1.5);
        const float cut = smoothstep(0.42, 0.62, 0.5 + 0.9 * gnoise21(pr + 11.7) + 0.45 * gnoise21(2.1 * pr + 3.3));
        residLines = foamFilaments(xz, footprint, 5.0, clamp(f / 0.3, 0.0, 1.0)) * mix(1.0, cut, 1.0 - smoothstep(0.3, 1.0, footprint));
    }
    // breakers in place (oceanOptions2 bit 13, storm look): the in-place breakers' decaying foam draws as lace,
    // so the wind lines over it are damped (they read as thin grey streaks where a cap decays)
    if ((pc.oceanOptions2 & 8192u) != 0u && (pc.oceanOptions & 2147483648u) != 0u) residLines *= 0.3;
    const float strW   = resid > 0.0 && !below ? residueOpacity * resid * 0.9 * residLines * smoothstep(0.25, 0.6, ft) : 0.0;
    // Minimal rollback of the earlier alpha expression, holding every source and
    // later lighting/reflection change fixed (the earlier version had no windrow alpha).
    const float white = (pc.oceanOptions & 2048u) != 0u
        ? max(smoothstep(thr, thr + edgeW, ft) * gateF, film)
        : max(max(capW, strW), film);
#ifdef THREEPP_OCEAN_INSPECT
    if (gInspectLiveBand) gInspectFreshAlpha = white * (below ? 0.12 : 1.0);
#endif
    // Foam has height and casts shadow, on itself and on the water beside it, by what the sun can do
    const vec3  sunL  = lights.dirCount > 0u ? normalize(lights.dirLights[0].direction) : vec3(0.0, 1.0, 0.0);
    const float sunS  = lights.dirCount > 0u ? clamp(dot(lights.dirLights[0].color, vec3(0.2126, 0.7152, 0.0722)) / 2.0, 0.0, 1.0) : 0.0;
    const float ftSun = foamTexture(xz + sunL.xz * 0.5, lod);
    if ((pc.oceanOptions & 65536u) == 0u)
        col *= 1.0 - 0.3 * sunS * smoothstep(thr, thr + 0.18, ftSun) * gateF * (1.0 - white) * (1.0 - resid);   // (residue casts none: it drew dark rings)
#ifdef THREEPP_OCEAN_WATER_BODY_EXPLORATION
    debugWaterCover = 1.0 - (1.0 - debugWaterCover) * (1.0 - white);
#endif
    if (white <= 0.0) return col;
    const float e  = max(0.05, 0.5 * footprint);
    // (a sheet of wash takes the sky evenly: lit by the tilted wave it brightened and dimmed with each swell, all of it at once)
    const vec3  fN = normalize(mix(N, vec3(0.0, 1.0, 0.0), 0.8 * boil) + vec3(ft - foamTexture(xz + vec2(e, 0.0), lod), 0.0, ft - foamTexture(xz + vec2(0.0, e), lod)) * 3.0);
    // Hollows and thin edges are grey, and only the thick of a patch is white, under any sky.
    const float thick = max(smoothstep(thr, thr + 0.4, ft) * (1.0 - resid), core);
    const float selfS = 1.0 - 0.45 * sunS * smoothstep(0.0, 0.15, ftSun - ft);
    const vec3  fCol = mix(vec3(0.42, 0.47, 0.5), vec3(0.8, 0.82, 0.84) * mix(1.0, 0.8, boil), thick * smoothstep(0.4, 0.85, f)) * sky * selfS;   // (0.4, was 0.15: aging foam is grey, only a fresh cap white)
    const vec3  fDiff = sampleEnvLod(fN, maxLod) + lights.ambient + hemiAmbient(fN);
    const vec3  lit = shadeDiffuseDirect(P, fN, V, fCol, 0.8, 0.0, vec3(0.0), doShadows, fDiff,
                                         vec3(0.0), 0.0, vec3(0.04), 0.0, 1.3, 0.0, seed,
                                         /*addEmissive=*/true, /*cheapHit=*/false);
#ifdef THREEPP_OCEAN_INSPECT
    if (pc.inspectMode != 0u) {
        float alpha=white*(below?0.12:1.0);
        gInspectReflection*=1.0-alpha;
        gInspectTransmission*=1.0-alpha;
    }
#endif
    return mix(col, lit, ((pc.oceanOptions & 131072u) != 0u ? 1.0 : white) * (below ? 0.12 : 1.0));
}

// How much a wave here holds foam: its front face, and where the fine waves fold. Foam shaped by this
// follows the crests, the way foam made by the waves does, instead of lying over them.
float foamWaveShape(vec3 N, vec2 xz, vec2 windDir) {
    const float front = smoothstep(-0.03, 0.12, dot(N.xz, windDir));
    float fold = 0.0;
    if (pc.oceanFineTileSize > 0.0) {
        const float inv = 1.0 / pc.oceanFineTileSize, e = 0.6;
        const vec2 dL = textureLod(oceanDisplace2, (xz - vec2(e, 0.0)) * inv, 0.0).rg, dR = textureLod(oceanDisplace2, (xz + vec2(e, 0.0)) * inv, 0.0).rg;
        const vec2 dD = textureLod(oceanDisplace2, (xz - vec2(0.0, e)) * inv, 0.0).rg, dU = textureLod(oceanDisplace2, (xz + vec2(0.0, e)) * inv, 0.0).rg;
        const float k = 1.0 * inv / (2.0 * e);// 1.0: the application's choppiness
        fold = 1.0 - ((1.0 + (dR.x - dL.x) * k) * (1.0 + (dU.y - dD.y) * k) - (dU.x - dD.x) * (dR.y - dL.y) * k * k);
    }
    return 0.4 + 0.9 * max(front, smoothstep(0.0, 0.25, fold));
}

// How much the surface here folds, the short waves' over a 1.2 m stencil and the long waves' over 12 m: the same read of the waves as their shape.
float foamLiveFold(vec2 xz) {
    float fold = 0.0;
    if (pc.oceanFineTileSize > 0.0) {
        // (oceanOptions2 bit 14, set by THREEPP_OCEAN_SLOPE_BUDGET=1: over 4 m, the waves of 2 m and up; foam_world.comp's too)
        const float inv = 1.0 / pc.oceanFineTileSize, e = (pc.oceanOptions2 & 16384u) != 0u ? 2.0 : 0.6, k = 1.0 * inv / (2.0 * e);
        const vec2 dL = textureLod(oceanDisplace2, (xz - vec2(e, 0.0)) * inv, 0.0).rg, dR = textureLod(oceanDisplace2, (xz + vec2(e, 0.0)) * inv, 0.0).rg;
        const vec2 dD = textureLod(oceanDisplace2, (xz - vec2(0.0, e)) * inv, 0.0).rg, dU = textureLod(oceanDisplace2, (xz + vec2(0.0, e)) * inv, 0.0).rg;
        fold = 1.0 - ((1.0 + (dR.x - dL.x) * k) * (1.0 + (dU.y - dD.y) * k) - (dU.x - dD.x) * (dR.y - dL.y) * k * k);
    }
    if (pc.oceanFoamTileSize > 0.0) {
        const float inv = 1.0 / pc.oceanFoamTileSize, e = 6.0, k = 1.0 * inv / (2.0 * e);
        // the swell's displacement B-spline filtered: differenced from the bilinear read the fold was piecewise
        // linear on the 2.93 m texel grid, and the band's threshold cut it into flat polygons with straight seams by her hull
        const vec2 dL = disp0Smooth((xz - vec2(e, 0.0)) * inv), dR = disp0Smooth((xz + vec2(e, 0.0)) * inv);
        const vec2 dD = disp0Smooth((xz - vec2(0.0, e)) * inv), dU = disp0Smooth((xz + vec2(0.0, e)) * inv);
        fold += 3.0 * (1.0 - ((1.0 + (dR.x - dL.x) * k) * (1.0 + (dU.y - dD.y) * k) - (dU.x - dD.x) * (dR.y - dL.y) * k * k));
    }
    return fold;
}

// The water surface's analytic response to ONE light direction: a GGX glint on
// the wave facet plus the clearcoat layer's tighter lobe. shadeWater's
// directional, spot and point loops all evaluate this one lobe, so the three
// light types cannot drift apart on the same water.
float waterGlint(vec3 N, vec3 V, vec3 L, float NdotV, float ndl, float r0,
                 float specRough, float kG, float clearcoat, float ccRough) {
    const vec3  H   = normalize(V + L);
    const float ndh = max(dot(N, H), 0.0);
    const float vdh = max(dot(V, H), 0.0);
    const float Gs  = geomSmithG1(NdotV, kG) * geomSmithG1(ndl, kG);
    const float Fb  = r0 + (1.0 - r0) * pow(1.0 - vdh, 5.0);
    float spec = distGGX(ndh, specRough) * Gs * Fb / max(4.0 * NdotV * ndl, 1e-4);
    if (clearcoat > 0.0) {
        const float Fc = 0.04 + 0.96 * pow(1.0 - vdh, 5.0);
        spec += clearcoat * distGGX(ndh, ccRough) * Gs * Fc / max(4.0 * NdotV * ndl, 1e-4);
    }
    return spec;
}

// Thin-shell water (RenderMode::RasterFirst). Replicates closest_hit's
// DETERMINISTIC thin-shell BSDF — noise-free because it's an analytic Fresnel
// split, not a stochastic reflect/refract pick:
//   colour = mix( (1-F)·transmit , F·reflect )
// • reflect  = ray-traced sky + scene (carries the env's sun → glittering
//              glints across the FFT-perturbed wave facets),
// • transmit = the refracted underwater view (sand floor / deep env) tinted by
//              Beer-Lambert absorption (thin-shell `thickness` proxy),
// • foam     = a diffuse whitewater layer where the Tessendorf Jacobian folded
//              the surface (ocean only; gated on the per-vertex foam attribute).
// N must already carry the FFT fine-cascade perturbation (applied in main()).
// slopeVarSq: the sub-pixel FFT-chop slope variance that main() FADED OUT of N
// (the LEAN/Toksvig "banked" term). At the altitudes this ocean is viewed from
// (100 m – 2 km) the fine cascade's metre-scale wave slope sits below the pixel
// footprint; keeping it in N and glinting a razor GGX lobe against it aliases the
// sun highlight into a per-pixel speckle field. Folding that removed variance
// into the specular roughness broadens every glint into a coherent glitter sheen
// instead — the missing half of the water spec-AA (the fade alone left the glint
// razor-sharp). σ_slope² adds to α² (≈roughness²) for a GGX lobe.
// The sun's shadow visibility at the water surface point, traced by the first
// of the in-scatter and glint loops to need it and reused by the second: both
// trace from P + N * SHADOW_EPS along the same L, so the ray is the same ray.
// -1 = not traced yet. Only the first directional light is cached (the one-sun
// policy); any further light traces as before.
float gWaterSunVis0 = -1.0;


#ifdef THREEPP_NO_RAY_QUERY
// ── The water's mirror on a device without ray query: a screen-space march ──
// traceRadiance has no scene to hit there (every ray misses, ray_query_stub),
// so the water mirrored the environment alone: no hull, no quay, no shore
// trees. This marches the water's one mirror ray against the G-buffer's depth
// instead: what stands on screen above the waterline mirrors; what is off
// screen or hidden stays the environment. A hit is lit from the G-buffer as a
// cheap traced hit is (shadeDiffuseDirect with the sun through the shadow map,
// the sky and ambient fills, the murk over a submerged hit), so the two paths
// agree where both can see. The march is perspective-correct (1/w is linear
// along the screen segment), jittered per pixel (TAA averages), passes
// through water pixels as the traced ray passes through crests, and fades at
// the screen's edge into the environment. Returns false on a miss: the caller
// keeps its environment path, byte for byte what it was. A multisampled frame
// (the overlay's MSAA) has no single-sample G-buffer to march: no mirror.
bool ssrWaterReflect(vec3 origin, vec3 dir, float maxLod, float missLod, float envInt, bool doShadows,
                     inout uint seed, out vec3 radiance, out float hitT, out bool hitMoved) {
    radiance = vec3(0.0);
    hitT     = -1.0;
    hitMoved = false;
    if (((effFlags() >> 5u) & 0x3u) != 0u) return false;
    const bool ortho = cam.camAux.x > 0.5;
    const mat4 view  = inverse(cam.viewInverse);
    const mat4 proj  = inverse(cam.projInverse);
    const vec3 o     = (view * vec4(origin, 1.0)).xyz;// view space, -z forward
    const vec3 d     = (view * vec4(dir, 0.0)).xyz;
    // The reach: kReach metres, cut short of the near plane for a ray that
    // comes toward the camera.
    const float kReach = 4000.0;
    float       tMax   = kReach;
    if (d.z > 1e-6) tMax = min(tMax, (-0.05 - o.z) / d.z);
    if (tMax <= 1e-3) return false;
    const vec4 c0 = proj * vec4(o, 1.0);
    const vec4 c1 = proj * vec4(o + d * tMax, 1.0);
    if (c0.w <= 1e-6 || c1.w <= 1e-6) return false;
    // Screen uv as the G-buffer is addressed: ndc.y up, uv.y down (main's ndc).
    const vec2 n0 = c0.xy / c0.w, n1 = c1.xy / c1.w;
    const vec2 s0 = vec2(n0.x * 0.5 + 0.5, 0.5 - n0.y * 0.5);
    const vec2 s1 = vec2(n1.x * 0.5 + 0.5, 0.5 - n1.y * 0.5);
    const vec2 ds = s1 - s0;
    // The segment clipped to the screen.
    float kEnd = 1.0;
    for (int a = 0; a < 2; ++a) {
        if (abs(ds[a]) > 1e-6) kEnd = min(kEnd, ((ds[a] > 0.0 ? 1.0 : 0.0) - s0[a]) / ds[a]);
    }
    if (kEnd <= 0.0) return false;
    const vec2  res   = vec2(float(pc.width), float(pc.height));
    const float lenPx = length(ds * kEnd * res);
    const int   steps = int(clamp(ceil(lenPx / 4.0), 4.0, 48.0));// ~4 px a step
    const float jit   = rnd(seed);// per pixel AND per frame: TAA averages the march's phase
    // Perspective-correct depth along the screen segment: 1/w is linear in k;
    // an orthographic view interpolates z itself.
    const float iw0 = 1.0 / c0.w, iw1 = 1.0 / c1.w;
    const float z0 = o.z, z1 = o.z + d.z * tMax;
    float kPrev = 0.0, kHit = -1.0, zRayPrev = z0;
    for (int i = 0; i < steps; ++i) {
        const float k     = kEnd * (float(i) + jit) / float(steps);
        const vec2  s     = s0 + ds * k;
        const ivec2 px    = ivec2(clamp(s * res, vec2(0.0), res - 1.0));
        const float depth = texelFetch(gbufDepthTex, px, 0).x;
        if (depth > 0.0) {// reverse-Z: 0 = sky
            const uint id = texelFetch(gbufIdsTex, px, 0).x;
            if (id != 0u && geoms[id - 1u].foamAddress == 0ul) {// not water: a crest is passed through
                const float zRay  = ortho ? mix(z0, z1, k) : -1.0 / mix(iw0, iw1, k);
                const vec2  ndcS  = vec2(s.x * 2.0 - 1.0, -(s.y * 2.0 - 1.0));
                const vec4  vh    = cam.projInverse * vec4(ndcS, depth, 1.0);
                const float zS    = vh.z / vh.w;// the surface's view z (negative)
                // Thick enough that a crossing between two samples registers
                // (the ray's own depth advance), never thinner than the surface.
                const float thick = max(max(0.3, 0.03 * -zS), 1.5 * abs(zRay - zRayPrev));
                if (zRay < zS && zRay > zS - thick) {
                    kHit = k;
                    break;
                }
                zRayPrev = zRay;
            }
        }
        kPrev = k;
    }
    if (kHit < 0.0) return false;
    // Refine between the last free sample and the hit (four bisections).
    float kLo = kPrev, kHi = kHit;
    for (int b = 0; b < 4; ++b) {
        const float km    = 0.5 * (kLo + kHi);
        const vec2  s     = s0 + ds * km;
        const ivec2 px    = ivec2(clamp(s * res, vec2(0.0), res - 1.0));
        const float depth = texelFetch(gbufDepthTex, px, 0).x;
        bool behind = false;
        if (depth > 0.0) {
            const float zRay = ortho ? mix(z0, z1, km) : -1.0 / mix(iw0, iw1, km);
            const vec2  ndcS = vec2(s.x * 2.0 - 1.0, -(s.y * 2.0 - 1.0));
            const vec4  vh   = cam.projInverse * vec4(ndcS, depth, 1.0);
            behind           = zRay < vh.z / vh.w;
        }
        if (behind) kHi = km;
        else kLo = km;
    }
    const vec2  sH  = s0 + ds * kHi;
    const ivec2 pxH = ivec2(clamp(sH * res, vec2(0.0), res - 1.0));
    const uint  idH = texelFetch(gbufIdsTex, pxH, 0).x;
    if (idH == 0u) return false;
    const float depthH = texelFetch(gbufDepthTex, pxH, 0).x;
    const vec2  ndcH   = vec2(sH.x * 2.0 - 1.0, -(sH.y * 2.0 - 1.0));
    const vec4  vhH    = cam.projInverse * vec4(ndcH, depthH, 1.0);
    const vec3  P      = (cam.viewInverse * vec4(vhH.xyz / vhH.w, 1.0)).xyz;
    const vec4  nTap   = texelFetch(gbufNormalTex, pxH, 0);
    const vec4  aTap   = texelFetch(gbufAlbedoTex, pxH, 0);
    const MaterialDesc hm = entryMat(idH - 1u);
    const vec3  V      = -dir;
    vec3        N      = normalize(nTap.xyz * 2.0 - 1.0);
    if (dot(N, V) < 0.0) N = -N;
    hitT     = length(P - origin);
    hitMoved = (geoms[idH - 1u].flags & kGeomFlagMoved) != 0u;
    vec3 shade;
    if (nTap.w < 0.0) {
        shade = aTap.rgb;// unlit
    } else {
        const vec3 diffInd = murkAmbient(sampleEnvDiffuse(N) + lights.ambient + hemiAmbient(N), P, N);
        shade = shadeDiffuseDirect(P, N, V, aTap.rgb, clamp(nTap.w, 0.04, 1.0), clamp(aTap.a, 0.0, 1.0),
                                   hm.emissive * hm.emissiveIntensity, doShadows, diffInd,
                                   hm.sheenColor, hm.sheenRoughness,
                                   dielectricF0(hm.ior, hm.specularIntensity, hm.specularColor),
                                   hm.iridescence, hm.iridescenceIOR, hm.iridescenceThicknessNm, seed,
                                   /*addEmissive=*/true,
                                   /*cheapHit=*/false);// the sun through the shadow map, not a stub ray
    }
    // The screen's edge: into the environment over the last 8 % of it.
    const vec2  eb   = min(sH, 1.0 - sH);
    const float edge = smoothstep(0.0, 0.08, min(eb.x, eb.y));
    radiance         = mix(sampleEnvLod(dir, missLod) * envInt, shade, edge);
    return true;
}
#endif// THREEPP_NO_RAY_QUERY

// DLSS Ray Reconstruction's guides for a water or glass pixel (kRR only):
// the Fresnel the reflection is weighted by, the roughness of the lobe it
// was traced with, and the reflection ray's own hit distance (< 0: it
// escaped), latched where the reflection is traced, before the refraction or
// bottom rays that follow overwrite gTraceHitT.
vec3  gRRSpecF     = vec3(0.0);
float gRRSpecRough = 1.0;
float gRRReflHitT  = -1.0;
// Reflectance of an unpolarised ray from air into water, the full Fresnel equations.
float waterFresnelExact(float c, float n) {
    const float g = sqrt(max(n * n - 1.0 + c * c, 0.0));
    const float a = (g - c) / (g + c), b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * a * a * (1.0 + b * b);
}

// Isotropic Gaussian tangent slopes, integrated in polar coordinates.
// r^2 / MSS is exponentially distributed: six-point Gauss-Laguerre radial
// quadrature and 24 uniform azimuths (144 sky samples). This is a numerical
// approximation, audited against dense 2D Gauss-Hermite, not a look gain.
// The old 3-point, view-plane-only rule can reverse the grazing-angle trend.
// sunL/sunF: the share of the result (radiance, Fresnel weight) mirrored from within ~8 deg of the sun, kept
// apart so the cloud deck can dim it with the SUN's transmittance (the glitter) rather than the deck along
// the macro mirror direction, which at a grazing view lies tens of km off and shadowed the glitter path in
// straight-edged blocks
// Speck fix (oceanOptions2 bit 16, on by default; THREEPP_OCEAN_REFL_DOWN_FIX=0 rolls back): the normal the reflection optics
// use. A shading normal that leans away until it is seen edge-on or from behind is a facet hidden by the ones in front
// of it: the optics ease to the macro normal as N.V goes to 0, and a normal still facing away is turned until it is
// seen just edge-on. With a back-facing normal the slope integral had no visible facet at all: F came out 0 and the
// pixel showed the bare water body, a black core.
vec3 waterVisibleNormal(vec3 N, vec3 Nmacro, vec3 V) {
    if ((pc.oceanOptions2 & 65536u) == 0u) return N;
    vec3 n;
    if ((pc.oceanOptions2 & 4194304u) != 0u) {
        // (THREEPP_OCEAN_NORMAL_NO_FLIP: a normal that faces away is a facet behind its own crest; what shows in its place is
        // the crest, edge-on: the normal turned until it is seen edge-on. The easing to the macro normal stops
        // half way, so the far side of a wave keeps its relief instead of going flat.)
        const float nv = dot(N, V);
        const vec3 Nt = nv < 0.0 ? normalize(N - V * nv) : N;
        n = normalize(mix(Nmacro, Nt, max(smoothstep(0.0, 0.12, nv), 0.5)));
    } else
        n = normalize(mix(Nmacro, N, smoothstep(0.0, 0.12, dot(N, V))));
    const float c = dot(n, V);
    if (c < 0.02) n = normalize(n + V * (0.02 - c));
    return n;
}

vec4 waterSlopeIntegral(vec3 N, vec3 V, vec3 Nsurf, float mss, float ior, float maxLod, out vec3 sunL, out float sunF) {
    const float rad[6] = float[6](0.2228466042, 1.188932102, 2.992736326, 5.775143569, 9.837467418, 15.98287398);
    const float wt[6] = float[6](0.4589646739, 0.4170008308, 0.1133733821, 0.01039919745, 0.0002610172028, 8.985479064e-07);
    const vec3 along = length(V-N*dot(N,V)) > 1e-4 ? normalize(V-N*dot(N,V)) : normalize(cross(N,vec3(0.0,0.0,1.0)));
    const vec3 across = cross(N,along);
    vec3 sumL=vec3(0.0), sumLs=vec3(0.0); float sumF=0.0, sumW=0.0, sumFs=0.0;
    const vec3 Ls = lights.dirCount > 0u ? normalize(lights.dirLights[0].direction) : vec3(0.0, -1.0, 0.0);
    [[dont_unroll]] for (int j=0; j<6; ++j) {
        const float r=sqrt(max(mss,0.0)*rad[j]);
        [[dont_unroll]] for (int k=0; k<24; ++k) {
            const float phi=TWO_PI*(float(k)+0.5)/24.0;
            const vec3 Nt=normalize(N+r*(cos(phi)*along+sin(phi)*across));
            const float c=max(dot(Nt,V),0.0);
            // PDF is per area of the base tangent plane; convert facet area
            // before weighting its projected area towards the viewer.
            const float weight=wt[j]*c/max(dot(Nt,N),1e-6);
#ifdef THREEPP_OCEAN_WATER_BODY_EXPLORATION
            float f=waterFresnelExact(c,ior);
#else
            const float f=waterFresnelExact(c,ior);
#endif
            vec3 R=reflect(-V,Nt);
#ifdef THREEPP_OCEAN_WATER_BODY_EXPLORATION
            if ((lights.envKnobFlags & 16u) != 0u) {
                // The projected visible-facet weights already supply outgoing
                // masking. Incoming light must also reach the facet: do not
                // replace occluded/below-plane directions with horizon sky.
                // Smith G1 for Gaussian (Beckmann) slopes, Walter et al. 2007
                // eq. 26. alpha^2=MSS; no fitted brightness scale.
                const float mu = dot(R,N);
                const float a = max(mu,0.0) / sqrt(max(mss*(1.0-mu*mu),1e-12));
                const float G1 = mu <= 0.0 ? 0.0 : a >= 1.6 ? 1.0 :
                    (3.535*a+2.181*a*a)/(1.0+2.276*a+2.577*a*a);
                f *= G1;
            }
#endif
            const float up=dot(R,Nsurf);
            // Retained approximation: lower reflected directions use horizon
            // sky. Resolved reflected-water masking/multiple bounces remain absent.
            if (up<0.02) R=normalize(R+Nsurf*(0.02-up));
            sumF+=weight*f; sumW+=weight;
            const vec3 E=sampleEnvRaw(R,0.12*maxLod);// the deck goes on once, in shadeWater
            sumL+=weight*f*E;// as before, so a cloudless frame is bit-identical
            if (clouds.enabled>0.5) {
                const float sw=smoothstep(0.985,0.995,dot(R,Ls));
                sumLs+=(weight*f*sw)*E; sumFs+=weight*f*sw;
            }
        }
    }
    sunL=sumLs/max(sumW,1e-12); sunF=sumFs/max(sumW,1e-12);
    return vec4(sumL/max(sumW,1e-12),sumF/max(sumW,1e-12));
}

vec3 waterSkyIrradiance() {
    // Ed = integral L(mu,phi) mu dmu dphi. A coarse GGX-prefiltered zenith
    // lookup is not this integral: its footprint and angular centres differ.
    // Three-point Gauss-Legendre in mu is exact for the CIE overcast sky
    // L=1+2*mu (Ed=7*pi/3); eight azimuths also sample nonuniform day skies.
    const float mu[3] = float[3](0.1127016654, 0.5, 0.8872983346);
    const float wt[3] = float[3](0.2777777778, 0.4444444444, 0.2777777778);
    vec3 Ed = vec3(0.0);
    for (int j = 0; j < 3; ++j) {
        const float r = sqrt(1.0 - mu[j] * mu[j]);
        for (int i = 0; i < 8; ++i) {
            const float phi = (float(i) + 0.5) * TWO_PI / 8.0;
            Ed += (TWO_PI / 8.0) * wt[j] * mu[j]
                * sampleEnvLod(vec3(r * cos(phi), mu[j], r * sin(phi)), 0.0);// an irradiance: the deck as its mean
        }
    }
    return Ed;
}

// Bréon & Henriot (2006), slope covariance in scene x/z. Subtract the
// ensemble covariance retained by the FFT normal filter at this footprint.
// The CPU LUT integrates the actual cascade modes and linear stencil/strip
// transfer functions. No per-pixel squared-slope estimate or artistic floor.
mat2 waterGlitterRemainder(MaterialDesc pm, vec3 P, vec3 V, vec3 Nmacro) {
    vec2 w=normalize(pm.sheenColor.rg*2.0-1.0);
    vec2 c=vec2(-w.y,w.x);
    float u=max(pm.dispersion,0.0);
    mat2 total=(0.001+0.00316*u)*outerProduct(w,w)
              +(0.003+0.00185*u)*outerProduct(c,c);
    float pixWorld=(cam.camAux.x>0.5 ? 1.0 : length(P-gPrimaryOrigin))*2.0*cam.projInverse[1][1]/max(float(pc.height),1.0);
    float pixAlong=pixWorld/max(dot(Nmacro,V),0.02);
    float eps=pc.oceanFineTileSize/32.0;
    float keep=1.0-smoothstep(0.5*eps,1.5*eps,pixWorld);
    vec3 fine=vec3(0.0);
    if(pm.emissiveTexIndex>=0) {
        float x=clamp(log2(max(pixAlong,0.005)/0.005)/14.0,0.0,1.0)*23.0;
        float y=mod(atan(V.z,V.x)/PI,1.0)*16.0;
        ivec2 q=ivec2(floor(x),floor(y));vec2 f=fract(vec2(x,y));
        vec3 a=texelFetch(albedoMaps[nonuniformEXT(pm.emissiveTexIndex)],ivec2(q.x+24*q.y,0),0).rgb;
        vec3 b=texelFetch(albedoMaps[nonuniformEXT(pm.emissiveTexIndex)],ivec2(min(q.x+1,23)+24*q.y,0),0).rgb;
        vec3 c0=texelFetch(albedoMaps[nonuniformEXT(pm.emissiveTexIndex)],ivec2(q.x+24*((q.y+1)%16),0),0).rgb;
        vec3 d=texelFetch(albedoMaps[nonuniformEXT(pm.emissiveTexIndex)],ivec2(min(q.x+1,23)+24*((q.y+1)%16),0),0).rgb;
        fine=mix(mix(a,b,f.x),mix(c0,d,f.x),f.y);
        fine.y=2.0*fine.y-1.0;
        fine*=pm.iridescenceThicknessNm*keep*keep;
    }
    float coarseKeep=1.0;
    if(pc.oceanSkirtD1>pc.oceanSkirtD0) {
        vec2 off=abs(P.xz-vec2(pc.oceanSkirtX,pc.oceanSkirtZ));
        coarseKeep=1.0-smoothstep(pc.oceanSkirtD0,pc.oceanSkirtD1,max(off.x,off.y));
    }
    vec3 resolved=pm.specularColor*coarseKeep*coarseKeep+fine;
    mat2 rem=total-mat2(resolved.x,resolved.y,resolved.y,resolved.z);
    // Project onto the PSD cone: a rough/swell-rich resolved direction may
    // already exceed the empirical wind-only distribution.
    float mean=0.5*(rem[0][0]+rem[1][1]);
    float rad=length(vec2(0.5*(rem[0][0]-rem[1][1]),rem[0][1]));
    float lo=max(mean-rad,0.0), hi=max(mean+rad,0.0);
    rem=rad>1e-8 ? mat2(lo)+(rem-mat2(mean-rad))*((hi-lo)/(2.0*rad)) : mat2(max(mean,0.0));
    if (pm.specularIntensity < -1.5) rem=mat2(0.5*(rem[0][0]+rem[1][1]));
    return rem;
}

// Gaussian (Beckmann) slopes have a finite covariance, unlike GGX. Slopes
// are added in world x/z to the resolved FFT slope, matching the covariance.
float waterGaussianGlint(vec3 N, vec3 V, vec3 L, mat2 cov, float r0) {
    vec3 H=normalize(V+L);
    if(H.y<=0.0 || N.y<=0.0) return 0.0;
    vec2 delta=-H.xz/H.y+N.xz/max(N.y,0.05);
    // Finite solar disc (~0.27 degree radius) supplies the zero-remainder floor.
    cov+=mat2(0.0000014);
    float det=max(determinant(cov),1e-14);
    float pdf=exp(-0.5*dot(delta,inverse(cov)*delta))/(TWO_PI*sqrt(det));
    float D=pdf/max(pow(H.y,4.0),1e-6);
    float nv=max(dot(N,V),0.001), nl=max(dot(N,L),0.001);
    float rough=pow(max(cov[0][0]+cov[1][1],1e-8),0.25);
    float k=(rough+1.0)*(rough+1.0)/8.0;
    float F=r0+(1.0-r0)*pow(1.0-max(dot(V,H),0.0),5.0);
    return D*geomSmithG1(nv,k)*geomSmithG1(nl,k)*F/(4.0*nv*nl);
}

vec2 waterNormalGaussian(ivec2 cell, uint sampleIndex) {
    uint s=pcgHash(pcgHash(uint(cell.x)^pcgHash(uint(cell.y)))^(sampleIndex*0x9e3779b9u));
    float u=(float(pcgHash(s)&0x00ffffffu)+0.5)/16777216.0;
    float v=(float(pcgHash(s^0x85ebca6bu)&0x00ffffffu)+0.5)/16777216.0;
    return sqrt(-2.0*log(u))*vec2(cos(TWO_PI*v),sin(TWO_PI*v));
}

// Pull the world-space hit back through the same horizontal displacement as
// the resolved surface. A rest-space integer cell follows the moving wave;
// there is no independent time drift or floating-point hash lattice.
vec2 waterFacetRest(vec2 xz) {
    vec2 q=xz;
    if(pc.oceanFoamTileSize>0.0 && pc.oceanFineTileSize>0.0) {
        float inv0=1.0/pc.oceanFoamTileSize, inv2=1.0/pc.oceanFineTileSize;
        for(int it=0;it<6;++it)
            q=xz-(it<4 ? textureLod(oceanDisplace0,q*inv0,0.0).rg : disp0Smooth(q*inv0))*inv0
                -textureLod(oceanDisplace2,q*inv2,0.0).rg*inv2;
    }
    return q;
}

// Cov(sampled facets) + Cov(pixel kernel) = remainder before face weighting.
// Weighting is an appearance constraint, not a covariance-preserving closure.
// The 8 cm scale and 15% minimum kernel are rendering parameters. The wider
// kernel gives rotating facets a finite angular dwell instead of needle peaks.
float waterFacetGlitter(vec3 P, vec3 N, vec3 Nmacro, vec3 V, vec3 L, mat2 cov, float r0, bool original) {
    float pixel=(cam.camAux.x>0.5?1.0:length(P-gPrimaryOrigin))*2.0*cam.projInverse[1][1]/max(float(pc.height),1.0);
    float fraction=clamp(pixel*pixel/max(dot(Nmacro,V),0.02)/(0.08*0.08),original ? 0.02 : 0.15,1.0);
    if(fraction>0.999) return waterGaussianGlint(N,V,L,cov,r0);
    float a=sqrt(max(cov[0][0],1e-10)),b=cov[0][1]/a;
    float d=sqrt(max(cov[1][1]-b*b,0.0));
    vec2 position=(original ? P.xz+vec2(0.6,-0.4)*pc.timeSec : waterFacetRest(P.xz))/0.08;
    ivec2 cell=ivec2(floor(position));
    vec2 f=fract(position);f=f*f*(3.0-2.0*f);
    vec4 weights=vec4((1.0-f.x)*(1.0-f.y),f.x*(1.0-f.y),(1.0-f.x)*f.y,f.x*f.y);
    // Variance-normalised interpolation removes square cell silhouettes without
    // reducing the target slope variance between lattice points.
    weights*=inversesqrt(dot(weights,weights));
    vec2 base=-N.xz/max(N.y,0.05);
    float sum=0.0;
    for(uint i=0u;i<4u;++i) {
        vec2 z=weights.x*waterNormalGaussian(cell,i)
              +weights.y*waterNormalGaussian(cell+ivec2(1,0),i)
              +weights.z*waterNormalGaussian(cell+ivec2(0,1),i)
              +weights.w*waterNormalGaussian(cell+ivec2(1,1),i);
        vec2 slope=base+sqrt(1.0-fraction)*vec2(a*z.x,b*z.x+d*z.y);
        vec3 facet=normalize(vec3(-slope.x,1.0,-slope.y));
        sum+=waterGaussianGlint(facet,V,L,cov*fraction,r0);
    }
    vec3 H=normalize(V+L);
    vec2 delta=-H.xz/max(H.y,0.05)+N.xz/max(N.y,0.05);
    float face=exp(-0.5*dot(delta,inverse(cov+mat2(0.0000014))*delta));
    // Fade the face gate out as a pixel integrates a full facet cell, avoiding
    // a discontinuity at the continuous far-field branch above.
    return 0.25*sum*(original ? 1.0 : mix(face,1.0,smoothstep(0.5,1.0,fraction)));
}

#ifdef THREEPP_OCEAN_REFL_LOBE_EXPERIMENT
// Width controls for the reflection audit. The bank contains slope removed
// from the resolved normal by its footprint filter. Material roughness is an
// empirical Cox-Munk *remainder*, not another copy of that normal's slope.
// Its missing spatial spectrum prevents a unique physical footprint split.
// "footprint" is therefore an explicit appearance hypothesis: introduce the
// empirical remainder over the existing fine-stencil resolution transition.
// It is not a variance-conserving reconstruction of unsimulated capillary waves.
float waterGeometryMss(MaterialDesc pm, float bankedMss, vec3 P, vec3 Nmacro, vec3 V) {
    const float residual = pow(pm.roughness, 4.0);
    if ((pc.oceanOptions & 16777216u) != 0u) return bankedMss;
    if ((pc.oceanOptions & 33554432u) != 0u) {
        const float pixel = (cam.camAux.x > 0.5 ? 1.0 : length(P - gPrimaryOrigin))
            * 2.0 * cam.projInverse[1][1] / max(float(pc.height), 1.0);
        const float along = pixel / max(dot(Nmacro, V), 0.02);
        const float eps = max(pc.oceanFineTileSize / 32.0, 1e-4);
        const float unresolved = smoothstep(0.5 * eps, 1.5 * eps, pixel)
            * smoothstep(eps, 7.0 * eps, along);
        return bankedMss + residual * unresolved;
    }
    return residual + bankedMss;
}
#endif

#ifdef THREEPP_OCEAN_INSPECT
// Defined by ocean_inspect.glsl after shadeWater; these evaluators mirror
// water_displace.comp's live Catmull-Rom fields and hull/contact transforms.
vec3 sampleDisplacement(vec2 worldXZ, uint mask);
vec3 hullLocal(OceanHull hu, vec2 p);
layout(buffer_reference, scalar) buffer HorizonHullBuf { OceanHull h[]; };

vec3 waterReflectionCoarseDisplacement(vec2 q) {
    // Mask 1 selects ONLY the long/swell cascade. In particular, neither
    // cascade 1's height nor the fine 21 m cascade participates.
    vec3 d = sampleDisplacement(q, 1u);
    HorizonHullBuf hulls = HorizonHullBuf(inspectData.hullAddress);
    for (uint k = 0u; k < inspectData.hullCount; ++k) {
        const OceanHull hu = hulls.h[k];
        const vec3 hl = hullLocal(hu, q);
        const float planeY = hu.centerY + tan(clamp(hu.pitch, -1.0, 1.0)) * hl.y
            + tan(clamp(hu.roll, -1.0, 1.0)) * hl.x;
        d.y = mix(planeY, d.y, hl.z);
        d.xz *= hl.z;
    }
    return d;
}

float waterReflectionCoarseHeight(vec2 x) {
    // Solve the horizontal map anew at EVERY world position. This is the
    // inspector's fixed-point inversion, restricted to the coarse surface;
    // no local/frozen normal-map warp and no mesh-triangle proxy.
    vec2 q = x;
    const float tolerance = 0.001 * inspectData.tile[0] / float(textureSize(inspectHeight0, 0).x);
    [[dont_unroll]] for (int k = 0; k < 12; ++k) {
        const vec2 next = x - waterReflectionCoarseDisplacement(q).xz;
        const float error = length(next - q);
        q = next;
        if (error <= tolerance) break;
    }
    const vec3 d = waterReflectionCoarseDisplacement(q);
    // A folded/nonconvergent map cannot establish an obstruction. Fail open
    // rather than invent a height from a remote material point.
    if (length(q + d.xz - x) > tolerance) return -1e30;
    return inspectData.seaY + d.y;
}

float waterReflectionHorizonClear(vec3 origin, vec3 direction, float hitT, float wavelength) {
    // Near hits do NO texture reads or inversion; misses never call us.
    if (hitT < wavelength || wavelength <= 0.0 || pc.inspectAddress == 0ul) return 1.0;
    if ((inspectData.cascadeMask & 1u) == 0u || inspectData.tile[0] <= 0.0) return 1.0;
    // Eight samples / dominant wavelength, bounded at 64. At the cap we
    // still cover the FULL hit segment: >=4/lambda up to hitT=16*lambda.
    // Longer paths can miss crests; the reflhits B channel exposes hitT/lambda.
    const int steps = int(clamp(ceil(8.0 * hitT / wavelength), 1.0, 64.0));
    [[dont_unroll]] for (int j = 0; j < steps; ++j) {
        const float t = hitT * (float(j) + 0.5) / float(steps);
        const vec3 ray = origin + t * direction;
        if (ray.y < waterReflectionCoarseHeight(ray.xz) - 0.001) return 0.0;
    }
    return 1.0;
}

// Renewal approximation: one independent crest encounter per
// half peak wavelength of HORIZONTAL travel. alpha^2 = the lobe's MSS.
// Walter 2007 Eq. 26 is a numerical approximation to Gaussian Smith G1;
// its published coefficients are not scene-fitted visibility controls.
// P(0)=1; P(d1+d2)=P(d1)P(d2). No endpoint-height/phase information is used.
float waterReflectionPathClear(float mss, float elevationSin, float hitT, float wavelength) {
    if (hitT <= 0.0 || wavelength <= 0.0 || mss <= 0.0) return 1.0;
    const float mu = clamp(elevationSin, -1.0, 1.0);
    const float horizontal = sqrt(max(1.0 - mu * mu, 0.0));
    if (mu <= 0.0) return 0.0;
    if (horizontal == 0.0) return 1.0;
    const float a = mu / (sqrt(mss) * horizontal);
    if (a >= 1.6) return 1.0;
    // The rational approximation overshoots 1 very slightly near its cutoff.
    // Clamp numerical error so compounding always remains a probability.
    const float G1 = min(1.0, (3.535*a + 2.181*a*a) / (1.0 + 2.276*a + 2.577*a*a));
    const float encounters = 2.0 * hitT * horizontal / wavelength;
    return pow(G1, encounters);
}
#endif

// Each lobe ray owns its hit distance: apply the upstream media before averaging.
vec3 waterReflectionLeg(vec3 col, vec3 origin, vec3 direction, bool below) {
    // ── THE AIR ON THE REFLECTED LEG ─────────────────────────────────────────
    // The caller fogs the camera->water leg and nothing else, so a calm fjord
    // under haze mirrored the far mountains clear and saturated while the same
    // mountains seen directly stood blue in the air: the reflection read
    // clearer than the view. The opaque mirror ray got its own leg's air in
    // deferred_shade.comp ("THE AIR ON THE REFLECTED LEG"); this is the water's
    // twin, with the same two closed forms: the height fog's extinction and
    // ambient in-scatter over the hit distance (applyHeteroSurfaceFog), and the
    // sky pixel's fog on an escape (applySkyFog).
    //
    // BEFORE the fire and the splats below, and that order is the point: this
    // fogs what the TRACE returned, a surface at gTraceHitT or the sky at
    // infinity. The media below are composited over it at their own, near
    // distance. Placed after them, an escape gave a campfire and a splat cloud
    // a few metres up the leg the sky's fog, saturated at a grazing ray, and
    // both vanished from the mirror (rg_ab outdoor).
    //
    // Two terms of the opaque block are left out on purpose. The sun's
    // in-scatter march (volumetricDirScatter) is 16 shadow rays a pixel, and an
    // ocean is most of a frame. The ParticleField media have their own leg
    // below (pdEmissiveLeg); applyParticleFog here would count them twice.
    //
    // AFTER the specularIntensity scale: that knob dims reflected geometry and
    // leaves an escape at full Fresnel, and the haze a far shore fades into is
    // the horizon sky's colour, so the leg's in-scatter belongs with the escape.
    //
    // Above water only (`below` takes the murk, further down), and only with
    // the air medium on: a scene without fog runs the arithmetic it ran before.
    // With heteroActive raised by a density volume alone hfDensity is 0, and
    // both forms return their input.
    if (!below && clouds.heteroActive > 0.5) {
        col = (gTraceHitT > 0.0)
                           ? applyHeteroSurfaceFog(col, vec2(0.0), gTraceHitT, origin,
                                                   origin + direction * gTraceHitT)
                           : applySkyFog(col, direction);
    }

    // ── FIRE IN THE MIRROR ───────────────────────────────────────────────────
    // The traced leg above sees geometry and the environment. It does NOT see
    // the participating media — a ParticleField flame is density with a
    // blackbody emission ramp, not a mesh — so a campfire beside a pond
    // reflected only its flicker PointLight's glint and never the flame body.
    // Match the primary leg: multiply the traced radiance by the medium's
    // transmittance and add the emission it produced, marched over the same
    // volumes with the same expression (pdEmissiveLeg, 16 steps, emission +
    // extinction only — see its header for what is deliberately left out).
    //
    // tMax is the reflected content's own distance so the march stops at the
    // reflected surface: gTraceHitT < 0 means the ray escaped to the sky, and
    // then the volume boxes are the only thing bounding it. The whole block is
    // behind the same two uniform gates the emission term uses — flags bit 11
    // ("a density volume is live") and pd.counts.y ("something is emissive") —
    // so a scene with no fields, and a dust-only scene as well, runs the
    // pre-change arithmetic textually.
#ifdef PD_LINEAR
    if (flagOn(2048u) && pd.counts.y != 0u) {
        vec3 legEmis;
        const float legT = pdEmissiveLeg(origin, direction, gTraceHitT > 0.0 ? gTraceHitT : 1e30,
                                         legEmis);
        col = col * legT + legEmis;
    }
#endif

    // ── SPLATS IN THE MIRROR ─────────────────────────────────────────────────
    // Same argument, same ray, different medium: a SplatCloud is composited by
    // SplatPass's tile rasterizer AFTER this shade, so the traced leg above
    // cannot see it even in principle — and a cloud behind the camera, the case
    // that motivates the whole design, is out of reach of any screen-space
    // scheme. svLeg marches the volume SplatPass baked for exactly this
    // (plans/splat-volume-reflections.md); flags bit 12 is the single uniform
    // gate, set only for the PRIMARY view, so a splat-free scene — and every
    // secondary/sensor view — runs the pre-change arithmetic textually.
    if (flagOn(4096u)) {
        vec3 svEmis;
        const float svT = svLeg(origin, direction, gTraceHitT > 0.0 ? gTraceHitT : 1e30, svEmis);
        col = col * svT + svEmis;
    }

    return col;
}

// Keep upstream's screen-space mirror on devices without ray queries for every
// feature lobe sample, with the traced path's hit/miss state for its consumers.
vec3 traceWaterReflection(vec3 origin, vec3 direction, bool doShadows,
                          float maxLod, float missLod, inout uint seed,
                          bool cheapHits, bool probeHitFill, float envInt) {
#ifdef THREEPP_NO_RAY_QUERY
    vec3 color;
    float hitT;
    bool hitMoved;
    if (ssrWaterReflect(origin, direction, maxLod, missLod, envInt, doShadows,
                        seed, color, hitT, hitMoved)) {
        gTraceHitT = hitT;
        gTraceHitMoved = hitMoved;
        gTraceMissTput = vec3(0.0);
        gTraceHitWater = false;
#ifdef THREEPP_OCEAN_REFL_HITS_DIAG
        gTraceHitHull = false;
#endif
        return color;
    }
#endif
    return traceRadiance(origin, direction, doShadows, maxLod, missLod, seed,
                         cheapHits, probeHitFill, envInt);
}

// ── the base renderer's shadeWater, verbatim (the extended ocean master switch off) ────────
// THREEPP_OCEAN_EXTENDED=0 shades the water with upstream's own function,
// copied unchanged from the base renderer f2343291 (lines 259-1278 of this file
// there: waterFacetFresnelMean and shadeWater) but for its name and the two in-scatter constants that the extended ocean
// retuned, which keep upstream's values under their own names. It is compiled
// only into the THREEPP_OCEAN_UPSTREAM_WATER variant of the shade (the one
// DeferredShade builds with the master switch off), and the extended ocean
// shadeWater below only into the others: one water function per module (two
// overflow spirv-opt's id bound in the inlined shade), and no runtime branch
// between them. Not a second water model to maintain: it is upstream's, and
// it follows upstream.
#ifdef THREEPP_OCEAN_UPSTREAM_WATER
// The mean Schlick Fresnel of a surface's facets seen at cosine c, its slope
// variance var (both axes) Gaussian, each facet weighted by its projected
// area toward the eye (max(c', 0)): the slope along the line of sight tilts a
// facet's cosine to (c + s sin) / sqrt(1 + s^2), the slope across it barely
// does. Five-point Gauss-Hermite quadrature over the along slope; at var = 0
// it is Schlick's own value.
float waterFacetFresnelMean(float c, float var, float r0) {
    const float sd = sqrt(max(var, 0.0));// sqrt(2) * the per-axis deviation sqrt(var / 2)
    const float sn = sqrt(max(1.0 - c * c, 0.0));
    const float x[5] = float[5](-2.02018287, -0.95857246, 0.0, 0.95857246, 2.02018287);
    const float w[5] = float[5](0.01995324, 0.39361932, 0.94530872, 0.39361932, 0.01995324);
    float num = 0.0, den = 0.0;
    for (int i = 0; i < 5; ++i) {
        const float s  = sd * x[i];
        const float cp = clamp((c + s * sn) * inversesqrt(1.0 + s * s), 0.0, 1.0);
        const float f  = r0 + (1.0 - r0) * pow(1.0 - cp, 5.0);
        num += w[i] * f * cp;
        den += w[i] * cp;
    }
    return den > 1e-6 ? num / den : r0 + (1.0 - r0) * pow(1.0 - c, 5.0);
}

const float kWaterInScatterSkyUpstream      = 0.35;
const float kWaterInScatterPhaseMaxUpstream = 8.0;
vec3 shadeWaterUpstream(vec3 P, vec3 N, vec3 V, MaterialDesc pm, int instIdx,
                bool doShadows, float maxLod, uint frame, inout uint seed,
                float slopeVarSq, vec3 Nmacro, ivec2 px, float clusterDist) {
    gWaterSunVis0 = -1.0;// per surface point: a second call must not reuse the first's ray
    // The surface the rays see may be the mesh's ray-tracing proxy
    // (DisplacedMesh::rtProxyCells, kGeomFlagRtProxy): a coarser grid
    // than the one the raster drew P from, above or below it by the proxy's
    // interpolation error. Every ray below starts a SHADOW_EPS off P, so
    // against the proxy it met the water at once: a crest's refraction ray
    // found a "bottom" at zero depth, and the bay lit up along every ridge.
    // Put P on the proxy: a short vertical probe through the surface (the
    // water is single-valued in y), kept only when what it finds is this
    // water; under a hull or a quay P stays where the raster put it.
    if ((geoms[instIdx].flags & kGeomFlagRtProxy) != 0u) {
        const float h = 0.5;
        rayQueryEXT pq;
        rayQueryInitializeEXT(pq, topAS, gl_RayFlagsOpaqueEXT, kRayMaskOpaque,
                              P + vec3(0.0, h, 0.0), 0.0, vec3(0.0, -1.0, 0.0), 2.0 * h);
        while (rayQueryProceedEXT(pq)) {}
        if (rayQueryGetIntersectionTypeEXT(pq, true) != gl_RayQueryCommittedIntersectionNoneEXT &&
            rayQueryGetIntersectionInstanceCustomIndexEXT(pq, true) == instIdx) {
            P.y += h - rayQueryGetIntersectionTEXT(pq, true);
        }
    }
    const float NdotV = max(dot(N, V), 1e-4);
    const float ior   = max(pm.ior, 1.0);
    // ── FROM BELOW ───────────────────────────────────────────────────────────
    // The surface rasterizes into the G-buffer from underneath already (the
    // ocean material is Side::Double thin-shell), and main() flips N toward the
    // camera before calling in, so N/Nmacro point DOWN here and every N-relative
    // term below is already in the right frame. What is NOT is the optics: the
    // ray leaves the DENSE medium, which the above-water Fresnel and the
    // sand-floor transmit have no expression for. Gated on the murk medium being
    // live, so a scene without one can never enter these branches.
    const bool below = camUnderwater();
    // Explicit square: the max() above makes (1-ior)/(1+ior) ≤ 0, and GLSL's
    // pow() is spec-undefined for a negative base (works only where the driver
    // folds pow(x,2) → x·x).
    const float f0n   = (1.0 - ior) / (1.0 + ior);
    const float r0    = f0n * f0n;
    // r0 is symmetric in the two indices, so only the ANGLE term changes going
    // the other way: Schlick's cosine is the one on the LESS dense side, and
    // past the critical angle (sin²θ_air ≥ 1) no transmitted ray exists at all —
    // F is exactly 1 and the surface is a perfect mirror. That discontinuity IS
    // Snell's window: a ~97° cone of sky overhead, a mirrored water column
    // everywhere outside it. eta here is n_water/n_air = ior, refract()'s
    // convention for a ray crossing water → air.
    float cosAir = 1.0;// air-side cosine; 0 = total internal reflection
    float F;
    if (below) {
        const float sinAir2 = ior * ior * (1.0 - NdotV * NdotV);
        cosAir = (sinAir2 < 1.0) ? sqrt(1.0 - sinAir2) : 0.0;
        F = (sinAir2 >= 1.0) ? 1.0 : r0 + (1.0 - r0) * pow(1.0 - cosAir, 5.0);
    } else {
        // How much is reflected is the mean Fresnel of the facets the pixel
        // cannot show, not the Fresnel of their mean: N has had those waves
        // filtered out of it, and a flat surface seen at a glancing angle is
        // nearly a mirror, where the facets of a rough sea that face the eye
        // (more of them visible, at a steeper cosine) reflect far less (Cox
        // and Munk 1954; Bruneton, Neyret and Holzschuch 2010). The variance
        // is the one the glitter's lobe already takes, alpha^2 with
        // alpha = effRough^2. Near water, where the drawn normal resolves the
        // waves, it is next to nothing and F is the mirror's.
        const float aF = pm.roughness * pm.roughness + slopeVarSq;
        F = waterFacetFresnelMean(NdotV, aF * aF, r0);
    }

    // Effective specular roughness = base water roughness ⊕ the sub-pixel chop
    // variance banked out of N (LEAN spec-AA). Distant water fattens its lobe →
    // the sun glint reads as a soft coherent sheen; near water (slopeVarSq→0)
    // keeps its crisp near-mirror sparkle.
    const float effRough = sqrt(pm.roughness * pm.roughness + slopeVarSq);

    // Reflected sky + scene. Env MISSES read a MILDLY blurred mip — point
    // features in the env (stars, a small bright moon) otherwise reflect off
    // the chop-perturbed normals as per-pixel white speckle; smooth day skies
    // are visually unchanged.
    //
    // HORIZON CLAMP (Nmacro): the wave-perturbed N can reflect the view ray
    // BELOW the water's own plane. Such a ray immediately re-hits the surface
    // (a neighbouring crest), and traceRadiance shades that hit as an opaque
    // water surface → a dark dot. Scattered over the distance band where the
    // slope distribution straddles the horizon, that is exactly the
    // long-standing "square patch of breakup" on calm fjord/norway water
    // (proven with a reflection-hit classification AOV: the speckles are
    // self-hits < 60 m). The transmission path documents the identical failure
    // mode — a downward ray self-intersecting the chop — and side-steps it by
    // staying analytic; the reflection ray had no such guard.
    //
    // Lifting R just above the macro plane is the physically-right repair: a
    // real surface cannot reflect below itself, and a grazing ray that would
    // have skimmed a crest reflects near-horizon light anyway. Clamped against
    // the MACRO normal (pre-tilt, pre-jitter), so a rotated ocean (Z-up pond)
    // needs no world-up assumption.
    //
    // The clamp needs NO from-below branch, and that is worth stating because
    // the physical rule inverts: a submerged surface cannot reflect ABOVE
    // itself. Nmacro is the CAMERA-FACING macro normal (main() flips it), so it
    // points down for a submerged view and `dot(R, Nmacro) >= 0.02` reads as
    // "keep R just below the plane" there and "just above" from the air — one
    // expression, both half-spaces, exactly as the world-up-free framing intends.
    //
    // ONE exception, and it is the whole waterline shot: when the split runs
    // through the PORT, a pixel in the wet half is looking into the water while
    // the eye is still in the air, so the ray reaches this surface from ABOVE
    // and "the camera's side" is the wrong side to keep the reflection on.
    // Clamping to the camera there mirrors the sky into the underwater half of
    // the frame — a bright grazing sheet exactly where the murk should start.
    // Flip the clamp to the WATER side and the same expression puts the traced
    // leg down the column, where applyMurk below turns it into the murk the
    // open-water pixels beside it already show.
    const vec3 Nsurf = camPortWetDryEye() ? -Nmacro : Nmacro;
    vec3 R = reflect(-V, N);
    const float rDotUp = dot(R, Nsurf);
    if (rDotUp < 0.02) R = normalize(R + Nsurf * (0.02 - rDotUp));
    // Env-miss reflection blur widens with the banked chop variance: distant
    // sub-pixel wave facets reflect a point-feature sky (sun disc / bright cloud
    // edge) as per-pixel speckle unless the reflection is filtered by the same
    // footprint the glint is. Floor at 0.12 (the near-water sheen).
    const float reflLod = max(0.12, effRough) * maxLod;
    // Offset along the MACRO normal, not the tilted N: on a coarse water mesh
    // the shading normal can lean far enough from the triangle plane that a
    // fixed offset along it fails to clear the neighbouring facet.
    // (Wave reflections, DisplacedMesh::Params::waveReflections, change what
    // the pass-through does when the ray never comes out of the wave it
    // entered: see traceRadiance's skip.)
#if SCOUT_WATER == 8
    gScoutKind = -1; gScoutWHits = 0;
#endif
    gTraceSkipWater = true;
    const vec3 reflOrig = P + Nmacro * SHADOW_EPS;// this water's own reflection: pass through crests
#ifdef THREEPP_NO_RAY_QUERY
    // No scene to trace here: the screen-space march (ssrWaterReflect above),
    // and where it misses the environment, exactly as traceRadiance leaves it.
    vec3  reflectColor;
    float ssrT;
    bool  ssrMoved;
    if (ssrWaterReflect(reflOrig, R, maxLod, reflLod, pm.envMapIntensity, doShadows, seed, reflectColor, ssrT,
                        ssrMoved)) {
        gTraceHitT     = ssrT;
        gTraceHitMoved = ssrMoved;
        gTraceMissTput = vec3(0.0);
    } else {
        reflectColor = traceRadiance(reflOrig, R, doShadows, maxLod, reflLod, seed, /*cheapHits=*/true,
                                     /*probeHitFill=*/true, /*envInt=*/pm.envMapIntensity);
    }
#else
    vec3 reflectColor = traceRadiance(reflOrig, R, doShadows, maxLod,
                                      reflLod, seed, /*cheapHits=*/true,// water: blur+temporal absorb
                                      /*probeHitFill=*/true,
                                      /*envInt=*/pm.envMapIntensity);
#endif
    gTraceSkipWater = false;
    if (kRR) {
        gRRSpecF     = vec3(F);
        gRRSpecRough = effRough;
        gRRReflHitT  = gTraceHitT;
    }
    // Reflected GEOMETRY strength, on specularIntensity (KHR_materials_specular).
    //
    // DELIBERATE DEVIATION, do not "correct" it: the glTF extension scales the whole
    // specular BRDF, environment included. Here it scales ONLY a reflection that landed
    // on geometry and leaves an escape to sky at full Fresnel. That asymmetry is the
    // point. Water's horizon sky-mirror band and its Snell's-window view from below are
    // what make water read as water; dimming those is never what is wanted, whereas a
    // shoreline mirrored at full radiance routinely is too strong.
    //
    // Why a new knob is needed at all: water's reflection is ONE hard mirror ray whose
    // energy is Schlick Fresnel from ior alone. effRough reaches only reflLod, the
    // environment-MISS mip, so material.roughness blurs a reflected sky and leaves a
    // reflected forest at full strength; water writes reflectImage = 0 so it never gets
    // the roughness-driven reflection denoiser opaque surfaces get; envMapIntensity
    // scales only the escape-to-environment terms, so lowering it dims the sky and makes
    // the ratio worse; and r0 = 0.02 against a grazing F of 0.25-0.67 leaves ior nearly
    // inert. Wave slope is not a lever either: the slope integral is dominated by the
    // finest cascade's cutoff, so a fully developed sea moves rms slope by -5% while
    // multiplying wave height by 2.85.
    //
    // gTraceHitT < 0 is the environment escape (deferred_shade_40_reflections.glsl:148).
    // Defaults to 1.0, so every existing scene runs the pre-change arithmetic textually.
    if (gTraceHitT > 0.0) reflectColor *= clamp(pm.specularIntensity, 0.0, 1.0);

    // ── THE AIR ON THE REFLECTED LEG ─────────────────────────────────────────
    // The caller fogs the camera->water leg and nothing else, so a calm fjord
    // under haze mirrored the far mountains clear and saturated while the same
    // mountains seen directly stood blue in the air: the reflection read
    // clearer than the view. The opaque mirror ray got its own leg's air in
    // deferred_shade.comp ("THE AIR ON THE REFLECTED LEG"); this is the water's
    // twin, with the same two closed forms: the height fog's extinction and
    // ambient in-scatter over the hit distance (applyHeteroSurfaceFog), and the
    // sky pixel's fog on an escape (applySkyFog).
    //
    // BEFORE the fire and the splats below, and that order is the point: this
    // fogs what the TRACE returned, a surface at gTraceHitT or the sky at
    // infinity. The media below are composited over it at their own, near
    // distance. Placed after them, an escape gave a campfire and a splat cloud
    // a few metres up the leg the sky's fog, saturated at a grazing ray, and
    // both vanished from the mirror (rg_ab outdoor).
    //
    // Two terms of the opaque block are left out on purpose. The sun's
    // in-scatter march (volumetricDirScatter) is 16 shadow rays a pixel, and an
    // ocean is most of a frame. The ParticleField media have their own leg
    // below (pdEmissiveLeg); applyParticleFog here would count them twice.
    //
    // AFTER the specularIntensity scale: that knob dims reflected geometry and
    // leaves an escape at full Fresnel, and the haze a far shore fades into is
    // the horizon sky's colour, so the leg's in-scatter belongs with the escape.
    //
    // Above water only (`below` takes the murk, further down), and only with
    // the air medium on: a scene without fog runs the arithmetic it ran before.
    // With heteroActive raised by a density volume alone hfDensity is 0, and
    // both forms return their input.
    if (!below && clouds.heteroActive > 0.5) {
        reflectColor = (gTraceHitT > 0.0)
                           ? applyHeteroSurfaceFog(reflectColor, vec2(0.0), gTraceHitT, reflOrig,
                                                   reflOrig + R * gTraceHitT)
                           : applySkyFog(reflectColor, R);
    }

    // ── FIRE IN THE MIRROR ───────────────────────────────────────────────────
    // The traced leg above sees geometry and the environment. It does NOT see
    // the participating media — a ParticleField flame is density with a
    // blackbody emission ramp, not a mesh — so a campfire beside a pond
    // reflected only its flicker PointLight's glint and never the flame body.
    // Match the primary leg: multiply the traced radiance by the medium's
    // transmittance and add the emission it produced, marched over the same
    // volumes with the same expression (pdEmissiveLeg, 16 steps, emission +
    // extinction only — see its header for what is deliberately left out).
    //
    // tMax is the reflected content's own distance so the march stops at the
    // reflected surface: gTraceHitT < 0 means the ray escaped to the sky, and
    // then the volume boxes are the only thing bounding it. The whole block is
    // behind the same two uniform gates the emission term uses — flags bit 11
    // ("a density volume is live") and pd.counts.y ("something is emissive") —
    // so a scene with no fields, and a dust-only scene as well, runs the
    // pre-change arithmetic textually.
#ifdef PD_LINEAR
    if (flagOn(2048u) && pd.counts.y != 0u) {
        vec3 legEmis;
        const float legT = pdEmissiveLeg(reflOrig, R, gTraceHitT > 0.0 ? gTraceHitT : 1e30,
                                         legEmis);
        reflectColor = reflectColor * legT + legEmis;
    }
#endif

    // ── SPLATS IN THE MIRROR ─────────────────────────────────────────────────
    // Same argument, same ray, different medium: a SplatCloud is composited by
    // SplatPass's tile rasterizer AFTER this shade, so the traced leg above
    // cannot see it even in principle — and a cloud behind the camera, the case
    // that motivates the whole design, is out of reach of any screen-space
    // scheme. svLeg marches the volume SplatPass baked for exactly this
    // (plans/splat-volume-reflections.md); flags bit 12 is the single uniform
    // gate, set only for the PRIMARY view, so a splat-free scene — and every
    // secondary/sensor view — runs the pre-change arithmetic textually.
    if (flagOn(4096u)) {
        vec3 svEmis;
        const float svT = svLeg(reflOrig, R, gTraceHitT > 0.0 ? gTraceHitT : 1e30, svEmis);
        reflectColor = reflectColor * svT + svEmis;
    }

    // ── THE MIRROR IS UNDER WATER ────────────────────────────────────────────
    // Third medium on the same leg, and the one that decides whether the from-
    // below surface reads at all: R points DOWN into the water column, so the
    // traced leg runs through the very murk the caller composites onto the
    // primary. Skip it and total internal reflection becomes an impossibly clean
    // mirror — a crisp hull at 40 m inside water whose direct view fades out by
    // 15. applyMurk is the same expression and the same medium the primary leg
    // gets, so the two cannot disagree. A ray that escaped (gTraceHitT < 0) is
    // heading down and never leaves the water, so it takes the saturating 10 km
    // leg, i.e. the infinite-murk limit applyMurkSky returns.
    if (below) {
        const float legT = (gTraceHitT > 0.0) ? min(gTraceHitT, 1.0e4) : 1.0e4;
        reflectColor = applyMurk(reflectColor, reflOrig, reflOrig + R * legT);
    }

    // Transmission. From BELOW it is Snell's window; from above, the analytic
    // deep-water body (+ an optional shallow bottom).
    vec3 transmitColor;
    if (below) {
        // ── SNELL'S WINDOW ───────────────────────────────────────────────────
        // Inside the critical cone the refracted ray exits upward into air, so
        // the transmitted radiance is simply the sky along it — the WHOLE upper
        // hemisphere squeezed into ~97°, which is what makes the window both
        // bright and compressed. Sampled at the reflection's own LOD so the sun
        // disc survives (it is the brightest thing an underwater shot has) while
        // a point-feature sky still does not speckle across the chop.
        //
        // NO SCENE TRACE, deliberately: rigging, topsides and gulls ABOVE the
        // surface are absent from the window. The sky outweighs them by orders
        // of magnitude and a trace here would add a second full radiance ray to
        // every water pixel. Accepted for v1 — named so a mast that fails to
        // appear overhead reads as a known approximation, not a bug.
        const vec3 wDir = refract(-V, N, ior);
        transmitColor = (dot(wDir, wDir) > 1e-6)
                      ? sampleEnvLod(normalize(wDir), reflLod)
                      : reflectColor;// outside the window there is no transmitted ray (F == 1)
    } else {
        // Two media for the column under this surface (the murk the scene
        // declared, or the material's own), one refracted probe for a bottom in
        // either, and ONE traceRadiance call that shades the bottom for both:
        // each call site is a whole inlined copy of traceRadiance, and the
        // copies are what the driver's compile time is made of. Each medium
        // keeps its own probe before the call and its own use of the result
        // after it.
        const bool murkColumn = murkLive() && fog.murkDensity >= kMurkIsWater;
        bool  bottomHit = false;// the probe met something that is not a wave crest
        vec3  uwOrigin  = vec3(0.0);
        vec3  dRef      = vec3(0.0);
        float dBot      = 0.0;
        float maxVis    = 0.0;
        // The material path's terms, used again after the bottom trace.
        float skyLum    = 0.0;
        vec3  deepBody  = vec3(0.0);
        vec3  sigmaT    = vec3(0.0);
        vec3  sigmaS    = vec3(0.0);
        float bodyLeg   = -1.0;
        if (murkColumn) {
            // ── ONE MEDIUM, FROM ABOVE TOO ──────────────────────────────────────
            // The scene declared a murk (setUnderwaterMurk + setFogWaterSurfaceY),
            // so the column under this surface IS that murk — the same medium the
            // primary leg gets from below, the from-below mirror above, and every
            // particle sprite on both sides. Until this branch existed the view
            // from above used the MATERIAL's Beer-Lambert (attenuationColor over
            // attenuationDistance, sun path doubled) plus the analytic deep body,
            // so a hull at 3 m kept ~(0.007, 0.18, 0.27) of its light while a
            // sprite beside it, on the murk, kept 0.66 achromatic: the prop
            // ghosted into teal and the cavitation ropes blazed through the waves.
            // Two water models for one column; this makes it one. Refract the view
            // ray for real, probe for what is under the surface, and hand the hit
            // to applyMurk over the actual in-water leg — attenuation and veil
            // from the same expression the primary leg uses, so the two cannot
            // disagree. A miss is the infinite column, which is applyMurkSky's
            // saturated limit — exactly what the env miss shows from below.
            // Gated on murkLive(), so every scene without a murk keeps the
            // material path below byte-for-byte.
            dRef = refract(-V, N, 1.0 / ior);
            // From the air side a transmitted ray always exists; the guard only
            // covers a degenerate normal.
            if (dot(dRef, dRef) < 1e-6) dRef = -N;
            dRef = normalize(dRef);
            transmitColor = applyMurkSky(dRef);
            // Beyond ~6 optical depths the hit term is < ~2% — invisible. The same
            // budget the material path uses, on the medium that is actually there.
            maxVis   = min(6.0 / fog.murkDensity, 1.0e4);
            uwOrigin = P - N * SHADOW_EPS;
            rayQueryEXT rq;
            rayQueryInitializeEXT(rq, topAS, gl_RayFlagsOpaqueEXT, kRayMaskAll,
                                  uwOrigin, 1e-3, dRef, maxVis);
            while (rayQueryProceedEXT(rq)) {}
            if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
                const int   hitId = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
                if (geoms[hitId].foamAddress == 0ul) {// not a wave-crest self-hit
                    bottomHit = true;
                    dBot      = rayQueryGetIntersectionTEXT(rq, true);
                }
            }
        } else {
        // Two terms:
        //
        // 1) ANALYTIC deep-water body — deliberately NOT a refraction ray for the
        // body itself. A stochastic downward continuation self-intersects this same
        // choppy surface (adjacent wave crests), which traceRadiance then shades as
        // opaque water → a chaotic black/white speckle that fills the screen when
        // viewed from ABOVE (F≈0.02 there, so the transmission term dominates).
        // Instead: the Beer-Lambert absorption tint lit by ambient skylight → a
        // smooth deep blue-green. (The path tracer can refract correctly because it
        // CONTINUES the path with medium tracking across the bounce.)
        vec3 tint = vec3(1.0);
        if (pm.attenuationDistance > 0.0)
            // 2× the crossing depth → an approximate down+up Beer-Lambert (2 crossings
            // over a dark seabed). The deeper absorption darkens it AND shifts it bluer
            // (red absorbs first), turning the old washed grey into rich deep-ocean blue.
            tint = pow(max(pm.attenuationColor, vec3(1e-6)), vec3(2.0 * pm.thickness / pm.attenuationDistance));
        // Lit by the sky's BRIGHTNESS (scalar luminance), NOT the coloured env — the
        // coloured sky washed the absorption hue to a dull grey. Scalar skylight keeps
        // the deep blue SATURATED. A view-depth term deepens it looking straight down
        // (longer water column) → a foreground-deep / horizon-bright gradient.
        // Dimmed (×0.45) because the deep body sits over a near-black seabed.
        //
        // BOTH terms keyed on SMOOTH per-view quantities, NOT the chop-perturbed N:
        // skyLum at the fixed zenith, depthFade on the view ELEVATION (V.y). Keying
        // them on N made every fine-cascade crest a 2.5×-brighter saturated-teal
        // BLOTCH (the deferred ocean's "cyan patches" — diagnosed by isolating this
        // term: the blotch pattern was entirely transmit). The analytic water BODY is
        // a smooth volumetric quantity; the chop belongs to Fresnel/reflection only.
        skyLum    = dot(sampleEnvLod(vec3(0.0, 1.0, 0.0), maxLod), vec3(0.2126, 0.7152, 0.0722));
        const float depthFade = mix(1.0, 0.4, clamp(V.y, 0.0, 1.0));// darker looking down into the column
        deepBody  = tint * (skyLum * 0.35 + lights.ambient) * depthFade;
        transmitColor = deepBody;

        // ── VOLUME EXTINCTION (absorption + scattering) ──────────────────────────
        // Everything from here to the in-scatter block is gated on
        // pm.scatterDistance > 0.0, so a material that never sets it takes the
        // byte-identical old arithmetic. sigma_a is read back out of the SAME
        // Beer-Lambert the tint above uses — attenuationColor is the per-
        // attenuationDistance transmittance, so sigma_a = -ln(att)/attDist [1/m]
        // — which keeps one medium description, not two.
        if (pm.scatterDistance > 0.0) {
            const vec3 sigmaA = (pm.attenuationDistance > 0.0)
                              ? -log(max(pm.attenuationColor, vec3(1e-4))) / pm.attenuationDistance
                              : vec3(0.0);
            sigmaS = pm.scatterColor / pm.scatterDistance;
            // Floored: sigmaT divides below, and a channel with neither absorption
            // nor scattering is a vacuum whose in-scatter is zero anyway.
            sigmaT = max(sigmaA + sigmaS, vec3(1e-4));
        }
        // bodyLeg (declared above; -1 means "not probed yet") is the length of
        // the in-water leg for the single-scattering integral; the shallow-
        // bottom probe below sets it to the real bottom distance when it finds one.

        // 2) SHALLOW BOTTOM — the pond/shore path. Refract the view ray for real
        // and probe for geometry within the Beer-Lambert visibility range; a hit
        // (sand, rocks, a hull below the waterline) is shaded deterministically and
        // ADDED with the absorption over the actual in-water path. Design notes:
        //  • The deterministic single refracted ray doesn't have the deep path's
        //    speckle problem: it only degenerates when it re-hits the WATER surface
        //    (steep chop relative to depth) — detected via the instance's water
        //    marker (foamAddress) and dropped, falling back to the body alone.
        //  • deepBody stays UNREDUCED: its ×0.35/×0.45 constants were calibrated
        //    over a finite dark seabed, so the finite-column energy loss is already
        //    baked in — cross-fading by transmittance would double-count it and
        //    darken every existing deep scene. Adding the attenuated bottom instead
        //    keeps deep water bit-identical (dark bottom × tint ≈ 0) and lets a
        //    bright shallow floor show through. Mild over-energy where both terms
        //    are bright; acceptable for the pond regime.
        //  • doShadows=false on the bottom shade: water is opaque-masked for
        //    occlusion queries (see vulkan_shared.h — the PT's caustic energy
        //    balance relies on it), so a shadow ray from the bottom would report
        //    the surface above as a full occluder and render the floor sun-black.
        //    Unshadowed direct + the vertical-depth absorption term is the
        //    workable raster approximation of sunlight penetrating water.
        if (pm.attenuationDistance > 0.0) {
            // Beyond ~6 attenuation lengths the bottom term is < ~2% — invisible.
            maxVis = 6.0 * pm.attenuationDistance;
            const vec3 dRefRaw = refract(-V, N, 1.0 / ior);
            if (dot(dRefRaw, dRefRaw) > 1e-6) {
                dRef     = normalize(dRefRaw);
                uwOrigin = P - N * SHADOW_EPS;
                rayQueryEXT rq;
                rayQueryInitializeEXT(rq, topAS, gl_RayFlagsOpaqueEXT, kRayMaskAll,
                                      uwOrigin, 1e-3, dRef, maxVis);
                while (rayQueryProceedEXT(rq)) {}
                if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
                    const int   hitId = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
                    if (geoms[hitId].foamAddress == 0ul) {// not a wave-crest self-hit
                        bottomHit = true;
                        dBot      = rayQueryGetIntersectionTEXT(rq, true);
                    }
                }
            }
        }
        }
        if (bottomHit) {
            // doShadows=false in both media: water is opaque-masked for occlusion
            // queries (see vulkan_shared.h), so a shadow ray from the bottom would
            // read the surface overhead as a full occluder and render the floor
            // sun-black. cheapHits=false: the floor IS the subject, keep it sharp;
            // probeHitFill=false: probes cannot resolve the underwater cavity.
            const vec3 bottom = traceRadiance(uwOrigin, dRef, /*doShadows=*/false,
                                              maxLod, 0.12 * maxLod, seed,
                                              /*cheapHits=*/false,
                                              /*probeHitFill=*/false,
                                              /*envInt=*/1.0);// transmitted, not IBL
            if (murkColumn) {
                // murkSunCaustic on the hit's direct light is what the from-below
                // primary gets too.
                transmitColor = applyMurk(bottom, uwOrigin, uwOrigin + dRef * dBot);
            } else {
                // View path + approximate sun path (vertical depth) through
                // the column, Beer-Lambert per channel.
                const float sunPath = dBot * (1.0 + clamp(-dRef.y, 0.0, 1.0));
                // With scattering live the column removes light by the FULL
                // extinction, not absorption alone: light scattered out of
                // the bottom ray is gone from it (it reappears in the
                // in-scatter term below, which is where that energy goes).
                // The else branch is the old expression, textually.
                const vec3  Tbot    = (pm.scatterDistance > 0.0)
                                    ? exp(-sigmaT * sunPath)
                                    : pow(max(pm.attenuationColor, vec3(1e-6)),
                                          vec3(sunPath / pm.attenuationDistance));
                // The body veil builds up over the water column: SCALAR
                // exponential saturation with a half-attenuation-length
                // constant. Deep water (d ≥ ~2 lengths) keeps ≥96% of the
                // calibrated deepBody — visually the old look — while a
                // metre-deep pond sheds most of the veil so its floor
                // reads instead of drowning in deep-ocean teal. Scalar on
                // purpose: a per-channel weight would hue-shift the body.
                const float bodyW = 1.0 - exp(-dBot / (0.5 * pm.attenuationDistance));
                // The probe stops at maxVis, so a floor that slopes away
                // crosses from "hit" to "no hit" along a depth isoline. Tbot
                // is small there (attenuationColor^6..12) but the floor it
                // multiplies is unshadowed sunlit albedo, so the residual
                // step printed as a hard band parallel to the shore in a
                // top-down view (netpen --terrain, 2026-09-05). Fade the
                // floor term out over the last 30% of the probe range so
                // the two branches meet continuously.
                const float edgeFade = 1.0 - smoothstep(0.7 * maxVis, maxVis, dBot);
                transmitColor = deepBody * bodyW + bottom * Tbot * edgeFade;
                bodyLeg = dBot;// the in-scatter integral stops at the floor
            }
        }
        if (!murkColumn) {

        // ── SINGLE SCATTERING IN THE WATER COLUMN ────────────────────────────────
        // The block above is pure Beer-Lambert: it can only ever REMOVE light from
        // the column, so the transmitted side is at best a darkened skylight. That
        // is why glacial and silty fjord water — bright turquoise precisely BECAUSE
        // rock flour scatters skylight sideways into the eye — could only be faked
        // by stretching attenuationDistance to non-physical values. This puts the
        // light back IN.
        //
        // ONE expression for the in-water leg: the deep body and the shallow-bottom
        // case differ only in the leg length handed to it (bodyLeg), never in the
        // integral.
        if (pm.scatterDistance > 0.0) {
            // Refracted view leg — the same refract() the bottom probe uses. From
            // the air side a transmitted ray always exists; the guard is for a
            // degenerate normal only.
            vec3 dRef = refract(-V, N, 1.0 / ior);
            if (dot(dRef, dRef) < 1e-6) dRef = -N;
            dRef = normalize(dRef);
            // No bottom within reach → integrate to the visibility limit, the same
            // 6-optical-depth budget the bottom probe budgets: past it the
            // exponential below is within 0.3% of its saturated value.
            const float legLen = (bodyLeg > 0.0)
                               ? bodyLeg
                               : 6.0 / max(max(sigmaT.r, sigmaT.g), sigmaT.b);

            // Light ENTERING the column at this surface point. Both terms are
            // Fresnel-transmitted (1-F) — what reflects off the surface is already
            // accounted for in reflectColor.
            //   • sky: the hemisphere is already integrated, so it is isotropic and
            //     takes no phase factor. Scaled by kWaterInScatterSkyUpstream against the
            //     SAME skyLum the deep body uses (see the constant's note).
            //   • sun: a delta light, so it gets the HG phase between the refracted
            //     view leg and the refracted sun leg. Shadowed exactly as the
            //     surface is — a cliff shadow on the water must darken the glow
            //     under it, or the column lights up inside the shadow.
            vec3 Ein = vec3(skyLum * kWaterInScatterSkyUpstream) + lights.ambient;
            const vec3 sunOrigIn = P + N * SHADOW_EPS;
            for (uint i = 0u; i < lights.dirCount; ++i) {
                const vec3  L   = normalize(lights.dirLights[i].direction);
                const float ndl = dot(N, L);
                if (ndl <= 0.0) continue;// sun below the surface plane: no transmitted beam
                float vis = 1.0;
                if (doShadows) {
                    vis = (i == 0u && gWaterSunVis0 >= 0.0) ? gWaterSunVis0
                                                            : primarySunVis(i, P, N, sunOrigIn, L, false);
                    if (i == 0u) gWaterSunVis0 = vis;
                }
                if (vis <= 0.0) continue;
                vec3 sunRefr = refract(-L, N, 1.0 / ior);
                if (dot(sunRefr, sunRefr) < 1e-6) continue;
                sunRefr = normalize(sunRefr);
                const float phase = min(hgPhase(dot(dRef, sunRefr), kWaterInScatterG) * 4.0 * PI,
                                        kWaterInScatterPhaseMaxUpstream);// 1 == isotropic
                Ein += lights.dirLights[i].color * (ndl * vis * cloudShadowSample(P) * phase);
            }
            Ein *= (1.0 - F);

            // Source sigma_s·E_in attenuated by exp(-sigma_t·t), integrated over
            // t in [0, legLen]:  (sigma_s/sigma_t)·E_in·(1 - exp(-sigma_t·legLen)).
            // SATURATING by construction — looking straight down lengthens the leg
            // but the term can never exceed the single-scattering albedo times the
            // incident light, so there is no blow-up down the column.
            const vec3 inScatter = (sigmaS / sigmaT) * Ein * (1.0 - exp(-sigmaT * legLen));

            // NO DOUBLE COUNT. The analytic deepBody is itself a stand-in for the
            // light that comes back out of a deep column — over a truly dark seabed
            // pure absorption returns nothing, and `skyLum * 0.35` is the hand-fit
            // that made deep water look lit at all. Now that the in-scatter is
            // computed for real, the stand-in is faded out by exactly the fraction
            // of extinction that IS scattering (the single-scattering albedo, as a
            // scalar so the body's hue does not shift). A non-scattering medium
            // keeps the old body whole; a strongly scattering one hands the job
            // over to the physical term instead of stacking two copies of it.
            const float ssAlbedo = clamp(dot(sigmaS / sigmaT, vec3(0.2126, 0.7152, 0.0722)),
                                         0.0, 1.0);
            transmitColor = transmitColor * (1.0 - ssAlbedo) + inScatter;
        }
        }
    }

    // ── WAKE FIELD: bubbles under the surface ────────────────────────────────
    // A propeller's race is full of air for seconds after the film on top has
    // gone. Seen from above that is not white: the bubbles turn back light
    // that has been through a few decimetres of this water, so it is the
    // water's own hue, pale and milky, and it takes the sun and the shadows a
    // lit surface takes. Shaded as a diffuse body under the surface (the macro
    // normal: it is a volume, the chop is not in it) and laid in place of the
    // deep body by its optical thickness. The hue is the material's own
    // attenuation colour brought most of the way to white, so it follows
    // whatever sea the scene has made.
    if (!below && gWake.g > 2e-3) {
        vec3 hue = max(pm.attenuationColor, vec3(1e-4));
        hue = mix(vec3(1.0), hue / max(max(hue.r, hue.g), hue.b), 0.7);
        const vec3 aerDiff = sampleEnvDiffuse(Nmacro) + lights.ambient + hemiAmbient(Nmacro);
        const vec3 aerLit  = shadeDiffuseDirect(P, Nmacro, V, 0.30 * gWakeAlbedo * hue, 1.0, 0.0,
                                                vec3(0.0), doShadows, aerDiff,
                                                vec3(0.0), 0.0,
                                                vec3(0.0),
                                                0.0, 1.3, 0.0, seed,
                                                /*addEmissive=*/true,
                                                /*cheapHit=*/false);
        transmitColor = mix(transmitColor, aerLit, 1.0 - exp(-1.8 * gWake.g));
    }

#if SCOUT_WATER == 6
    // Reflection hit classification: red = SELF-HIT on water (< 60 m),
    // green = other geometry, blue = escaped to sky.
    if (gTraceHitT < 0.0)      return vec3(0.0, 0.0, 1.0);
    if (gTraceHitT < 60.0)     return vec3(1.0, 0.0, 0.0);
    return vec3(0.0, 1.0, 0.0);
#elif SCOUT_WATER == 8
    // How the reflection ray ended (traceRadiance): blue escaped to the
    // sky, green the face of a wave it went into; on something that is not
    // water, red with no water crossed on the way, yellow after some.
    if (gScoutKind == 0) return vec3(0.0, 0.0, 4.0);
    if (gScoutKind == 1) return vec3(0.0, 4.0, 0.0);
    return gScoutWHits == 0 ? vec3(4.0, 0.0, 0.0) : vec3(4.0, 4.0, 0.0);
#elif SCOUT_WATER == 1
    return reflectColor;
#elif SCOUT_WATER == 2
    return transmitColor;
#elif SCOUT_WATER == 3
    return vec3(F * 8.0);
#elif SCOUT_WATER == 5
    return vec3(length(N - vec3(0.0, 1.0, 0.0)) * 20.0);
#endif

    vec3 col = mix(transmitColor, reflectColor, F);

    // ── Sun/light specular GLINTS ─────────────────────────────────────────────
    // Sharp analytic GGX highlights of the scene lights across the FFT-perturbed
    // facets — the glittering sparkle that sells "ocean". shadeWater never did this
    // (only reflect+transmit), so the deferred water was flat/dull; the env
    // reflection alone misses it (the scene's directional sun isn't baked into the
    // HDRI). The clearcoat layer (pm.clearcoat) adds a 2nd tighter lobe. This is the
    // bulk of the dull→epic difference, added here via analytic light NEE.
    {
        const float specRough = max(effRough, 0.03);
        const float kG        = (specRough + 1.0) * (specRough + 1.0) / 8.0;
        const float ccRough   = clamp(pm.clearcoatRoughness, 0.03, 0.3);
        const vec3  sunOrig   = P + N * SHADOW_EPS;
        for (uint i = 0u; i < lights.dirCount; ++i) {
            const vec3  L   = normalize(lights.dirLights[i].direction);
            const float ndl = dot(N, L);
            if (ndl <= 0.0) continue;
            float vis = 1.0;
            if (doShadows) {
                vis = (i == 0u && gWaterSunVis0 >= 0.0) ? gWaterSunVis0
                                                        : primarySunVis(i, P, N, sunOrig, L, false);
                if (i == 0u) gWaterSunVis0 = vis;
            }
            if (vis <= 0.0) continue;
            const float spec = waterGlint(N, V, L, NdotV, ndl, r0, specRough, kG, pm.clearcoat, ccRough);
            vec3 c = spec * ndl * lights.dirLights[i].color * (vis * cloudShadowSample(P));
            // Directional lights are deltas → GGX spikes; clamp so a glint blooms
            // bright but never fireflies. (×4 over the global clamp → bright suns.)
            const float cl = max(max(c.r, c.g), c.b);
            const float gMax = pc.fireflyClamp * 4.0;
            if (cl > gMax) c *= gMax / cl;
            col += c;
        }
        // POINT and SPOT lights — the same GGX glints with range and
        // inverse-square attenuation (and, for a spot, its cone), so a sweeping
        // beam (lighthouse) lights the waves it crosses instead of existing only
        // as the in-scattered air volume, and a lantern over the sea lights the
        // waves under it as it lights the hull beside it.
        // The lights are THIS PIXEL'S CLUSTER CELL, as the opaque path's are
        // (analyticDirectSplit), not the UBO's eight strongest of each type: in
        // a scene with more than eight (a town's street lamps, a road's head
        // lamps) a boat's flood light is rarely among those, and it lit her deck
        // and left the water under it black. A cell is culled by each light's
        // range, so what is looped here is what can reach the pixel.
        const uint cellBase = clusterCellBase(px, clusterDist);
        const uint cCnt = (pc.clusterLightCount == 0u)
                              ? 0u
                              : min(clusterGrid[cellBase], kClusterMaxPerCell);
        for (uint ci = 0u; ci < cCnt; ++ci) {
            const uint li = min(clusterGrid[cellBase + 1u + ci], pc.clusterLightCount - 1u);
            const bool isSpot = clusterLights[li].type > 0.5;
            vec3 toL = clusterLights[li].position - P;
            const float dist = length(toL);
            if (dist < 1e-4) continue;
            toL /= dist;
            const float ndl = dot(N, toL);
            if (ndl <= 0.0) continue;
            // Falloff as the opaque path's two loops (shadeDiffuseDirect), so the
            // two surfaces agree on how far a light reaches: a spot's cone over
            // dist^decay, a point's distFalloff.
            float atten;
            if (isSpot) {
                const float spotCos   = dot(-toL, clusterLights[li].direction);
                const float spotAtten = smoothstep(clusterLights[li].cosAngleOuter,
                                                   clusterLights[li].cosAngleInner, spotCos);
                if (spotAtten <= 0.0) continue;
                atten = spotAtten / max(pow(dist, clusterLights[li].decay), 0.01);
            } else {
                atten = 1.0 / max(distFalloff(dist, clusterLights[li].decay), 0.01);
            }
            const float range = clusterLights[li].range;
            if (range > 0.0) {
                const float tt = dist / range;
                const float t4 = tt * tt * tt * tt;
                const float wnd = max(1.0 - t4, 0.0);
                atten *= wnd * wnd;
            }
            if (atten <= 1e-6) continue;
            const float smV = (kSM && doShadows)
                    ? smLampVis(clusterLights[li].shadowSlot, clusterLights[li].type > 0.5, P, N,
                                clusterLights[li].position, vec2(gl_GlobalInvocationID.xy)) : -1.0;
            const float vis = smV >= 0.0 ? smV : doShadows ? shadowVis(sunOrig, toL, dist - 1e-2) : 1.0;
            if (vis <= 0.0) continue;
            float spec = waterGlint(N, V, toL, NdotV, ndl, r0, specRough, kG, pm.clearcoat, ccRough);
            // Beam POOL — a broad faint sheen where the beam strikes the
            // surface. Real water returns part of a grazing beam diffusely
            // (micro-facets beyond the GGX tail, spray, top-layer scatter);
            // at roughness 0.04 the pure specular lobe is so tight the lit
            // pool would otherwise be invisible unless perfectly aligned.
            spec += 0.035 / PI;
            vec3 c = spec * ndl * clusterLights[li].color * atten * vis;
            const float cl = max(max(c.r, c.g), c.b);
            const float gMax = pc.fireflyClamp * 4.0;
            if (cl > gMax) c *= gMax / cl;
            col += c;
        }
    }

    // ── CREST FOAM (DisplacedMesh::Params::crestFoam) ────────────────────────
    // Compiled in only for a frame that has it on (spec constant 8). The foam
    // is a field F on the water's REST coordinates q (foam_crest.comp: laid
    // down where the surface folds, decaying on the parcels that took it), and
    // a pixel's q is its position minus the displacement it was drawn with.
    // min(F, 1) is the fraction of the water covered. What covers it is read
    // off a photograph of sea foam (binding 92), laid on q too, so it rides
    // and stretches with the water and never slides over it: a point is foam
    // where photo > 1 - F. The photo is rank-equalised, so the covered
    // fraction is F, and what goes first as F falls is what was thinnest in
    // the real foam: holes open, then a lace, then threads.
    // Both fetches are over the pixel's footprint on the water, a strip along
    // the line of sight. Where the footprint has averaged the photo's
    // contrast away, the step widens by exactly what was lost (the values
    // under a pixel then span about that much around the fetched mean), so
    // far foam tends to F itself, a covered fraction, with no speckle.
    // Lit as foam is: a Lambertian reflector of albedo 0.55 where it is thick
    // (Whitlock, Bartlett and Gurganus 1982), under water's own interface
    // (F0 = 0.02); where it has only just closed it is a thin layer and lets
    // some of the water through. From below it is the dull raft the
    // accumulator's foam is.
    if (kOceanCrestFoam != 0u && pc.oceanFoamTileSize > 0.0 && geoms[instIdx].foamAddress != 0ul) {
        const vec2  q        = P.xz - gCrestDisp;
        const float distF    = length(P - gPrimaryOrigin);
        const float pixWorld = (cam.camAux.x > 0.5 ? 1.0 : distF) * 2.0 * cam.projInverse[1][1]
                             / max(float(pc.height), 1.0);// world m / pixel across the view
        const float pixAlong = pixWorld / max(dot(Nmacro, V), 0.05);
        const float vl       = length(V.xz);
        const vec2  along    = vl > 1e-4 ? V.xz / vl : vec2(1.0, 0.0);
        const vec2  gA       = along * pixAlong;
        const vec2  gX       = vec2(-along.y, along.x) * pixWorld;
        const float invTile  = 1.0 / pc.oceanFoamTileSize;
        float       foamF;
        float       foamFresh = 0.0;// foam_crest.comp's .g: 1 on a breaking crest, gone soon after
        if ((kOceanClimate & 2u) != 0u) {
            // The wave climate: the texture keeps the foam this water would
            // have with cascades 0 and 1 at 1, 2/3 and 1/3 of the sea
            // (foam_crest.comp, OCEAN_CLIMATE_LEVELS); take it at the map's
            // level here, cascade 1's (the waves the fold foam counts are
            // mostly its), linearly between levels and to none at 0.
            const vec4  lv = textureGrad(oceanCrestFoam, q * invTile, gA * invTile, gX * invTile);
            const float L  = clamp(oceanClimate(oceanClimateMap, q).y, 0.0, 1.0) * 3.0;
            foamF = L >= 2.0 ? mix(lv.b, lv.r, L - 2.0) : L >= 1.0 ? mix(lv.a, lv.b, L - 1.0) : lv.a * L;
            foamFresh = lv.g * clamp(L / 3.0, 0.0, 1.0);
        } else {
            const vec2 cf = textureGrad(oceanCrestFoam, q * invTile, gA * invTile, gX * invTile).rg;
            foamF     = cf.r;
            foamFresh = cf.g;
        }
        bool cfFilm = false;// the foam level is a vessel's film, drawn as it was
        // What the vessels left on the water is the same stuff, so it is
        // drawn the same way: the wake field's film (how much foam its flow
        // has carried here) reads as a foam level. It stops just short of a
        // closed sheet: a race is churned water, never a lid, and the last
        // tenth of the photo's range stays open so it shows as foam with
        // water in it rather than as a white shape. Its lit colour takes the
        // field's albedo scale.
        float cfAlbedo = kOceanFoamAlbedoWhitlock;
        if (geoms[instIdx].foamAddress > 1ul) {
            const float filmF = clamp((gWake.r - 0.05) / 0.35, 0.0, 0.9);
            if (filmF > foamF) {
                foamF    = filmF;
                cfAlbedo *= gWakeAlbedo;
                cfFilm   = true;
            }
        }
#if CREST_FOAM_SCOUT == 1
        // A 4 m checker on q under the raw F: the checker must ride the waves
        // (squeezed on crests, stretched in troughs) and never slide.
        return vec3(0.12 + 0.30 * mod(floor(q.x * 0.25) + floor(q.y * 0.25), 2.0)) +
               vec3(1.0, 0.35, 0.05) * min(foamF, 2.0);
#endif
        if (foamF > 1.0 / 512.0) {
            // The photo twice, at two sizes and angles; a slow selector on q
            // keeps one or the other, so neither placement's period shows
            // across a patch (their mean would not do: two foams averaged are
            // neither). kTile is metres across one tile: the photo's holes
            // are then a few decimetres to a metre, as they were.
            const float kTileA = 6.0;
            const float kTileB = 8.3;
            const vec2  rB  = vec2(0.6216100, 0.7833269);// cos, sin of 0.9 rad
            const vec2  qB  = vec2(rB.x * q.x - rB.y * q.y, rB.y * q.x + rB.x * q.y) + vec2(3.1, -1.7);
            const vec2  aB  = vec2(rB.x * gA.x - rB.y * gA.y, rB.y * gA.x + rB.x * gA.y);
            const vec2  xB  = vec2(rB.x * gX.x - rB.y * gX.y, rB.y * gX.x + rB.x * gX.y);
            const float nA  = textureGrad(oceanCrestFoamPattern, q / kTileA, gA / kTileA, gX / kTileA).r;
            const float nB  = textureGrad(oceanCrestFoamPattern, qB / kTileB, aB / kTileB, xB / kTileB).r;
            const float sel = smoothstep(-0.3, 0.3, sin(dot(q, vec2(0.17, 0.09)) +
                                                        1.7 * sin(dot(q, vec2(-0.07, 0.12)))));
            const float n   = mix(nA, nB, sel);
            // Contrast the footprint left in the photo: its structure runs
            // from about an eighth to a fiftieth of the tile, and each scale
            // keeps about 1 / sqrt(1 + (footprint / size)^2) along each axis.
            float kept = 0.0;
            {
                const vec3 sizes = vec3(1.0 / 8.0, 1.0 / 20.0, 1.0 / 50.0) * mix(kTileA, kTileB, sel);
                const vec3 kx = inversesqrt(1.0 + (pixWorld / sizes) * (pixWorld / sizes));
                const vec3 ka = inversesqrt(1.0 + (pixAlong / sizes) * (pixAlong / sizes));
                kept = dot(vec3(0.4, 0.35, 0.25), kx * ka);
            }
            // How far this point is inside the covered set, in the photo's own
            // units. Half-width of the edge: what the average took, never less
            // than a thin edge, which itself closes with F (no foam is no foam).
            const float excess = n - (1.0 - foamF);
            const float halfW  = max(0.5 * (1.0 - kept), min(0.04, foamF));
            // Foam on open water is a layer of bubbles a few deep, not a
            // lid: the water shows through it. LOOK constants (Lars,
            // 2026-11-10: "foam should have opacity and not so white"): a
            // third opaque where it is thinnest, three quarters where a
            // crest has just broken.
            float layer = mix(0.35, 0.75, clamp(foamF, 0.0, 1.0));
            // A crest's foam AGES. While the crest breaks it is a thick white
            // mass of bubbles on the crest; the crest then runs on and leaves
            // a patch on the water behind it that breaks up and clears, which
            // is most of the foam on the sea at any moment. Drawn alike, the
            // sea's white lay in the troughs. So the fresh part is nearly
            // closed, and what is left behind stays white but SHRINKS (the
            // photo's holes open as F falls, above) and goes TRANSPARENT with
            // it, as F squared: gone well before F is. LOOK constants (Lars,
            // 2026-11-11: "it needs to shrink and go transparent, not just
            // grey out").
            if (!cfFilm) {
                const float fresh = clamp(foamFresh, 0.0, 1.0);
                const float left  = clamp(foamF, 0.0, 1.0);
                layer    = 0.95 * max(fresh, left * left);
                cfAlbedo = 0.85;
            }
            const float cover  = clamp(excess / (2.0 * halfW) + 0.5, 0.0, 1.0) *
                                 layer * (below ? 0.12 : 1.0);
#if CREST_FOAM_SCOUT == 2
            return vec3(cover);
#endif
            if (cover > 0.0) {
                const vec3 cfAmb = sampleEnvDiffuse(N) + lights.ambient + hemiAmbient(N);
                const vec3 cfLit = shadeDiffuseDirect(P, N, V, vec3(cfAlbedo), 1.0, 0.0,
                                                      vec3(0.0), doShadows, cfAmb,
                                                      vec3(0.0), 0.0,
                                                      vec3(0.02),
                                                      0.0, 1.3, 0.0, seed,
                                                      /*addEmissive=*/true,
                                                      /*cheapHit=*/false);
                col = mix(col, cfLit, cover);
            }
        }
#if CREST_FOAM_SCOUT == 2
        return vec3(0.0);
#endif
    }

    // Foam: folded-surface coverage (foamAddress attribute + live world foam
    // texture) bleaches a diffuse whitewater layer over the glass. The world
    // accumulator decays exponentially, so COVERAGE IS FRESHNESS — the mask
    // uses it to run a whitewater lifecycle instead of uniformly dimming:
    //   fresh (≈1)  — near-solid sheet, micro-bubble texture
    //   aging (mid) — fBm pools bite harder as coverage drops, tearing the
    //                 sheet into patches
    //   residue     — a ridged filament web (the lace a collapsed whitecap
    //                 leaves behind) outlives the sheet, thinning to streaks
    // Matches closest_hit's foam shading.
    // Breaking crests (DisplacedMesh::Params::breakingCrests): white water
    // where the crest under this pixel is breaking NOW, from the surface's
    // own local steepness (water_displace.comp), so it sits on the crest top,
    // rides it and is gone when the crest relaxes: a step of half-width
    // kOceanBreakingRamp on the threshold, so cover > 0.5 exactly where it is
    // exceeded, and the steepness itself shapes it (largest on the crest,
    // falling down both faces). No texture, no noise, no accumulator. Thick
    // fresh foam is a Lambertian reflector of albedo 0.55, flat across the
    // visible (Whitlock, Bartlett and Gurganus 1982), with the specular of
    // water's own interface (n = 1.333, F0 = 0.02). From below the raft is
    // the dull patch the accumulator's foam is.
    if (gCrestEps > -1e29) {
        const float crestCover = smoothstep(kOceanBreakingSteepness - kOceanBreakingRamp,
                                            kOceanBreakingSteepness + kOceanBreakingRamp, gCrestEps) *
                                 (below ? 0.12 : 1.0);
        if (crestCover > 0.0) {
            const vec3 crestAmb = sampleEnvDiffuse(N) + lights.ambient + hemiAmbient(N);
            const vec3 crestLit = shadeDiffuseDirect(P, N, V, vec3(kOceanFoamAlbedoWhitlock), 1.0, 0.0,
                                                     vec3(0.0), doShadows, crestAmb,
                                                     vec3(0.0), 0.0,
                                                     vec3(0.02),
                                                     0.0, 1.3, 0.0, seed,
                                                     /*addEmissive=*/true,
                                                     /*cheapHit=*/false);
            col = mix(col, crestLit, crestCover);
        }
    }

    if (pc.oceanFoamTileSize > 0.0 && geoms[instIdx].foamAddress != 0ul) {
        // Bicubic + 1.3 gain: the B-spline kernel approximates rather than
        // interpolates, attenuating an isolated single-texel whitecap to
        // ~0.44 of its stored value — the gain restores it; patch interiors
        // already read ~1 and just clamp.
        const float foamCoverage = clamp(sampleFoamBicubic(P.xz / pc.oceanFoamTileSize) * 1.3, 0.0, 1.0);
        if (foamCoverage > 0.0) {
            const vec2  drift = vec2(0.42, 0.71) * pc.timeSec * 0.18;
            // Detail from the baked foam tile (binding 34). Manual LOD —
            // compute has no derivatives; a distance-based estimate (pixel
            // angle ≈ 0.001 rad) lands within ±1 mip of the true footprint,
            // indistinguishable for band-limited noise content.
            const float distF   = length(P - gPrimaryOrigin);// view leg of THIS pixel's ray
            const float lodMicro = log2(max(1.0, distF * 0.128)); // 512 texels / 4 m · 0.001
            const float lodLace  = log2(max(1.0, distF * 0.0427));// 512 texels / 12 m · 0.001
            const float lodEdge  = log2(max(1.0, distF * 0.064)); // 512 texels / 8 m · 0.001
            // Sheet: solid where coverage beats the pool noise; the noise
            // bite scales with age so fresh caps read unbroken. Calibrated
            // to foam_world.comp's GRADED soft-knee deposits (solid from
            // coverage ≈ 0.55) — retune the two together.
            //
            // The coverage accumulator only resolves ~1 m (cascade-0 texel
            // bumps survive the graded deposit), so a sheet edge cut purely
            // by coverage reads TEXELATED. A zero-mean mid-frequency sample
            // of the detail tile (R over an 8 m mapping → 0.3–1 m features)
            // erodes the threshold so the boundary detail comes from the
            // texture — same trick that fixed the lace streaks.
            const float pools = fbm4(P.xz * 0.18 + drift);
            const float edgeN = textureLod(foamDetailTex, P.xz * 0.125 + drift * 0.07, lodEdge).r;
            const float sheet = smoothstep(0.03, 0.42,
                    foamCoverage - pools * mix(0.50, 0.22, foamCoverage)
                                 - (edgeN - 0.5) * 0.30);
            // Lace: ridged filament pattern from the detail tile (~1.2 m
            // cells over the 12 m mapping). Fresh foam widens the band until
            // it merges with the sheet; old foam keeps only filament cores.
            const float lace = textureLod(foamDetailTex, P.xz * (1.0 / 12.0) + drift * 0.12, lodLace).g;
            const float web  = smoothstep(0.55, 0.80, lace * (0.55 + 0.45 * foamCoverage)) *
                               smoothstep(0.03, 0.22, foamCoverage);
            // From BELOW, foam is not paint on the surface — it is a raft of
            // bubbles seen from underneath, which reads as a dull patch in the
            // window rather than white. The lifecycle above still runs (the mask
            // is where the sheet IS), only its opacity collapses; killing it
            // outright would leave a breaking crest looking like clear water.
            const float foamMask = max(sheet, web * 0.8) * (below ? 0.12 : 1.0);
            if (foamMask > 0.0) {
                const float micro = textureLod(foamDetailTex, P.xz * 0.25 - drift * 0.1, lodMicro).r;
                vec3 foamCol = mix(vec3(0.62, 0.68, 0.72), vec3(0.97, 0.99, 1.00), micro);
                foamCol = mix(foamCol, vec3(0.97, 0.99, 1.00), 0.35 * foamCoverage);// fresh caps brighter
                const float foamRough= mix(1.0, 0.45, micro);
                const vec3  foamDiff = sampleEnvDiffuse(N) + lights.ambient + hemiAmbient(N);
                const vec3  foamLit  = shadeDiffuseDirect(P, N, V, foamCol, foamRough, 0.0,
                                                          vec3(0.0), doShadows, foamDiff,
                                                          vec3(0.0), 0.0,
                                                          vec3(0.04),
                                                          0.0, 1.3, 0.0, seed,
                                                          /*addEmissive=*/true,// foam: no iridescence
                                                          /*cheapHit=*/false);
                col = mix(col, foamLit, foamMask);
            }
        }
    }

    // ── WAKE FIELD: the film on top ──────────────────────────────────────────
    // The wake field's foam is a density that the water's own flow has carried
    // and torn (wake_field.comp), so its shape is already what motion made of
    // it and nothing is cut into it here: coverage is a soft step on the
    // density, thin film grey and wet, thick film white. Lit as the whitecaps
    // above are. From below it is the dull raft the whitecaps are too.
    if (gWake.r > 4e-3 && kOceanCrestFoam == 0u) {// crest foam draws the film itself (above)
        const float cover = smoothstep(0.10, 0.45, gWake.r) * (below ? 0.12 : 1.0);
        const vec3  filmCol  = gWakeAlbedo * mix(vec3(0.66, 0.72, 0.75), vec3(0.97, 0.99, 1.00), smoothstep(0.15, 0.85, gWake.r));
        const vec3  filmDiff = sampleEnvDiffuse(N) + lights.ambient + hemiAmbient(N);
        const vec3  filmLit  = shadeDiffuseDirect(P, N, V, filmCol, 0.8, 0.0,
                                                  vec3(0.0), doShadows, filmDiff,
                                                  vec3(0.0), 0.0,
                                                  vec3(0.04),
                                                  0.0, 1.3, 0.0, seed,
                                                  /*addEmissive=*/true,
                                                  /*cheapHit=*/false);
        col = mix(col, filmLit, cover);
    }
    return col;
}
#endif// THREEPP_OCEAN_UPSTREAM_WATER

vec3 shadeWater(vec3 P, vec3 N, vec3 V, MaterialDesc pm, int instIdx,
                bool doShadows, float maxLod, uint frame, inout uint seed,
                float slopeVarSq, float bankedMss, vec3 Nmacro, ivec2 px, float clusterDist) {
#ifdef THREEPP_OCEAN_UPSTREAM_WATER
    return shadeWaterUpstream(P, N, V, pm, instIdx, doShadows, maxLod, frame, seed, slopeVarSq, Nmacro, px,
                              clusterDist);
#else
    gWaterSunVis0 = -1.0;// per surface point: a second call must not reuse the first's ray
    // The surface the rays see may be the mesh's ray-tracing proxy
    // (DisplacedMesh::rtProxyCells, kGeomFlagRtProxy): a coarser grid
    // than the one the raster drew P from, above or below it by the proxy's
    // interpolation error. Every ray below starts a SHADOW_EPS off P, so
    // against the proxy it met the water at once: a crest's refraction ray
    // found a "bottom" at zero depth, and the bay lit up along every ridge.
    // Put P on the proxy: a short vertical probe through the surface (the
    // water is single-valued in y), kept only when what it finds is this
    // water; under a hull or a quay P stays where the raster put it.
    if ((geoms[instIdx].flags & kGeomFlagRtProxy) != 0u) {
        const float h = 0.5;
        rayQueryEXT pq;
        rayQueryInitializeEXT(pq, topAS, gl_RayFlagsOpaqueEXT, kRayMaskOpaque,
                              P + vec3(0.0, h, 0.0), 0.0, vec3(0.0, -1.0, 0.0), 2.0 * h);
        while (rayQueryProceedEXT(pq)) {}
        if (rayQueryGetIntersectionTypeEXT(pq, true) != gl_RayQueryCommittedIntersectionNoneEXT &&
            rayQueryGetIntersectionInstanceCustomIndexEXT(pq, true) == instIdx) {
            P.y += h - rayQueryGetIntersectionTEXT(pq, true);
        }
    }
    const float NdotV = max(dot(N, V), 1e-4);
    const float ior   = max(pm.ior, 1.0);
    vec3 debugColumnSun = vec3(0.0);// opt-in irradiance AOV, before any local lamps
#ifdef THREEPP_OCEAN_WATER_BODY_EXPLORATION
    vec3 debugBodySky = vec3(0.0), debugBodyDirect = vec3(0.0);
    debugWaterCover = clamp(gWake.g,0.0,1.0);
#endif
    // ── FROM BELOW ───────────────────────────────────────────────────────────
    // The surface rasterizes into the G-buffer from underneath already (the
    // ocean material is Side::Double thin-shell), and main() flips N toward the
    // camera before calling in, so N/Nmacro point DOWN here and every N-relative
    // term below is already in the right frame. What is NOT is the optics: the
    // ray leaves the DENSE medium, which the above-water Fresnel and the
    // sand-floor transmit have no expression for. Gated on the murk medium being
    // live, so a scene without one can never enter these branches.
    const bool below = camUnderwater();
    const bool physicalGlitter = !below && pm.specularIntensity < 0.0;
    const mat2 glitterCov = physicalGlitter ? waterGlitterRemainder(pm,P,V,Nmacro) : mat2(0.0);
    const float glitterMss = glitterCov[0][0]+glitterCov[1][1];
    // Explicit square: the max() above makes (1-ior)/(1+ior) ≤ 0, and GLSL's
    // pow() is spec-undefined for a negative base (works only where the driver
    // folds pow(x,2) → x·x).
    const float f0n   = (1.0 - ior) / (1.0 + ior);
    const float r0    = f0n * f0n;
    if(physicalGlitter && pm.iridescence<0.0) {
        // Sun-only, linear, ungraded and unclamped: no sky/body/foam contribution.
        vec3 sun=vec3(0.0);
        if(lights.dirCount>0u) {
            vec3 L=normalize(lights.dirLights[0].direction);
            float ndl=max(dot(N,L),0.0);
            float vis=doShadows ? shadowVis(P+N*SHADOW_EPS,L,1e30) : 1.0;
            sun=waterFacetGlitter(P,N,Nmacro,V,L,glitterCov,r0,pm.iridescence>-1.5)
                *ndl*lights.dirLights[0].color*vis*cloudShadowSample(P);
        }
        return sun;
    }
    // r0 is symmetric in the two indices, so only the ANGLE term changes going
    // the other way: Schlick's cosine is the one on the LESS dense side, and
    // past the critical angle (sin²θ_air ≥ 1) no transmitted ray exists at all —
    // F is exactly 1 and the surface is a perfect mirror. That discontinuity IS
    // Snell's window: a ~97° cone of sky overhead, a mirrored water column
    // everywhere outside it. eta here is n_water/n_air = ior, refract()'s
    // convention for a ray crossing water → air.
    float cosAir = 1.0;// air-side cosine; 0 = total internal reflection
    float F;
    vec3 Nvis = N;// speck fix: the normal the reflection optics use (waterVisibleNormal)
    vec3 slopeSky = vec3(0.0);
    if (below) {
        const float sinAir2 = ior * ior * (1.0 - NdotV * NdotV);
        cosAir = (sinAir2 < 1.0) ? sqrt(1.0 - sinAir2) : 0.0;
        F = (sinAir2 >= 1.0) ? 1.0 : r0 + (1.0 - r0) * pow(1.0 - cosAir, 5.0);
    } else {
        // Base roughness^4 is the captured Cox-Munk remainder. Add the fine
        // slopes actually filtered out of N. The old distance-only 1.8 factor
        // ignored the pixel footprint and the measured banked variance.
        const vec3 slopeUp=Nmacro;
        vec3 sunL; float sunF;
        // (THREEPP_OCEAN_NORMAL_NO_FLIP: everything below takes a normal that faces away as seen edge-on)
        if ((pc.oceanOptions2 & 4194304u) != 0u && !below && dot(N, V) < 0.0) N = normalize(N - V * (dot(N, V) - 1e-3));
        Nvis = waterVisibleNormal(N, Nmacro, V);
#ifdef THREEPP_OCEAN_REFL_LOBE_EXPERIMENT
        // Paired audit: use the selected conditional measure for both F/sky
        // and geometry. This tests the energy split independently of width.
        const float opticalMss = (pc.oceanOptions & 67108864u) != 0u
            ? waterGeometryMss(pm, bankedMss, P, Nmacro, V)
            : pm.roughness*pm.roughness*pm.roughness*pm.roughness+bankedMss;
        const vec4 integral=waterSlopeIntegral(Nvis,V,slopeUp,
            opticalMss,ior,maxLod,sunL,sunF);
#else
        const vec4 integral=waterSlopeIntegral(N,V,slopeUp,
            pm.roughness*pm.roughness*pm.roughness*pm.roughness+bankedMss,ior,maxLod,sunL,sunF);
#endif
        F=integral.a;
        slopeSky=integral.rgb;
        // the cloud deck once per pixel (the 144 slope directions read the bare sky): the sky along the macro
        // mirror direction from this point, the sun's neighbourhood through the sun's own transmittance here,
        // the one the direct glint takes
        if (clouds.enabled > 0.5) {
            vec3 Rm = reflect(-V, slopeUp);
            Rm.y = max(Rm.y, 0.02);
            const float skyF = max(F - sunF, 0.0);
            const vec3 skyPart = cloudOverSky(P, normalize(Rm), (slopeSky - sunL) / max(skyF, 1e-4), 0.12 * maxLod, 1.0) * skyF;
            slopeSky = skyPart + mix(cloudLight() * sunF, sunL, cloudShadowSample(P));
        }

    }

    // Effective specular roughness = base water roughness ⊕ the sub-pixel chop
    // variance banked out of N (LEAN spec-AA). Distant water fattens its lobe →
    // the sun glint reads as a soft coherent sheen; near water (slopeVarSq→0)
    // keeps its crisp near-mirror sparkle.
    // The bulk Cox-Munk remainder grew when the spectrum accounting changed
    // (0.251 -> 0.407 roughness in Hs 1.5), while resolved slope energy fell
    // from 0.0755 to 0.0216. Keep the facet highlight narrower than that bulk
    // distribution: 0.18 preserves the old width relative to resolved slopes
    // (0.251 * (0.0216/0.0755)^0.25 ~= 0.184). This is a render-only look
    // calibration. Actual footprint variance still widens distant highlights.
    // A neutral body reveals the wide tails more readily; retain sparkle contrast
    // with a slightly narrower clear-day appearance cap. Wave variance is unchanged.
    const float facetCap = !below && clouds.enabled < 0.5 && (lights.envKnobFlags & 4u) != 0u ? 0.16 : 0.18;
    const float facetRoughSq = min(pm.roughness * pm.roughness, facetCap * facetCap);
    const float effRough = physicalGlitter ? pow(max(glitterMss,1e-8),0.25) : sqrt(facetRoughSq + slopeVarSq);

    // Reflected sky + scene. Env MISSES read a MILDLY blurred mip — point
    // features in the env (stars, a small bright moon) otherwise reflect off
    // the chop-perturbed normals as per-pixel white speckle; smooth day skies
    // are visually unchanged.
    //
    // HORIZON CLAMP (Nmacro): the wave-perturbed N can reflect the view ray
    // BELOW the water's own plane. Such a ray immediately re-hits the surface
    // (a neighbouring crest), and traceRadiance shades that hit as an opaque
    // water surface → a dark dot. Scattered over the distance band where the
    // slope distribution straddles the horizon, that is exactly the
    // long-standing "square patch of breakup" on calm fjord/norway water
    // (proven with a reflection-hit classification AOV: the speckles are
    // self-hits < 60 m). The transmission path documents the identical failure
    // mode — a downward ray self-intersecting the chop — and side-steps it by
    // staying analytic; the reflection ray had no such guard.
    //
    // Lifting R above the macro plane is an approximation for this opaque-hit
    // reflection path. A real facet can reflect towards neighbouring water,
    // requiring secondary water bounces that this path does not solve. The
    // clamp substitutes near-horizon light; its grazing brightness is not
    // physically certified (not validated against a reference). Clamped against
    // the MACRO normal (pre-tilt, pre-jitter), so a rotated ocean (Z-up pond)
    // needs no world-up assumption.
    //
    // The clamp needs NO from-below branch, and that is worth stating because
    // the chosen clamp hemisphere inverts. Nmacro is the CAMERA-FACING macro
    // normal (main() flips it), so it
    // points down for a submerged view and `dot(R, Nmacro) >= 0.02` reads as
    // "keep R just below the plane" there and "just above" from the air — one
    // expression, both half-spaces, exactly as the world-up-free framing intends.
    //
    // main() orients Nmacro toward the incident MEDIUM, including the wet
    // half of a straddling port. Do not flip it again when the eye is dry:
    // that sent wet pixels' reflected rays upward, producing sky-coloured
    // patches bounded by the per-pixel eye/near-surface test.
    const vec3 Nsurf = Nmacro;
    // Geometry and sky reflect differently. The sky is smooth, so every facet's own mirror direction
    // reads it, and that is the sea's texture. A hull is a hard-edged shape: a ray per facet made a
    // mosaic of hull or sky beside her, the dark dots (bisect: gone with the sky alone). What the eye
    // gets from a ripple field is a stretched, averaged image of her: three rays round the direction
    // of the smoothed surface, and each ray that escapes counts as the facet's own sky.
    // Opt-in (pc.oceanOptions2 bit 12, THREEPP_WATER_REFL_FULL_SLOPE=1): geometry is reflected by the
    // wave slopes the glitter and the sky see. The default reads it off a normal tilted only 35 %
    // of the way to the wave's (Nsmooth below) with a fixed lobe, so a hull or a pile mirrored
    // in a rough sea stays a calm-water image under a broken, glittering sky. With it on, the
    // rays go round the full shading normal and the lobe is the slope the pixel banked.
    const bool  reflFullSlope = (pc.oceanOptions2 & (1u << 12u)) != 0u;
    const float reflLobe = reflFullSlope ? clamp(0.7 * sqrt(slopeVarSq), 0.02, 0.22)
                                         : clamp(0.09 + 0.5 * sqrt(slopeVarSq), 0.09, 0.22);
    vec3 R = reflect(-V, N);
    const float rDotUp = dot(R, Nsurf);
    if (rDotUp < 0.02) R = normalize(R + Nsurf * (0.02 - rDotUp));
    // Env-miss reflection blur widens with the banked chop variance: distant
    // sub-pixel wave facets reflect a point-feature sky (sun disc / bright cloud
    // edge) as per-pixel speckle unless the reflection is filtered by the same
    // footprint the glint is. Floor at 0.12 (the near-water sheen).
    // The mirror sky image follows the resolved facet and its pixel footprint.
    // Convolving it with the full solar slope distribution a second time makes
    // the calibrated water body pale. Sky-lit microglints remain future work.
    const float reflLod = max(0.12, physicalGlitter ? sqrt(slopeVarSq) : effRough) * maxLod;
    // Offset along the MACRO normal, not the tilted N: on a coarse water mesh
    // the shading normal can lean far enough from the triangle plane that a
    // fixed offset along it fails to clear the neighbouring facet.
    // (Wave reflections, DisplacedMesh::Params::waveReflections, change what
    // the pass-through does when the ray never comes out of the wave it
    // entered: see traceRadiance's skip.)
#if SCOUT_WATER == 8
    gScoutKind = -1; gScoutWHits = 0;
#endif
    gTraceSkipWater = true;
    const vec3 reflOrig = P + Nmacro * SHADOW_EPS;// this water's own reflection: pass through crests
    vec3 skyFacet = sampleEnvLod(R, reflLod) * pm.envMapIntensity;
    // Fresnel and incoming sky radiance are correlated over the unresolved
    // facets: use <F * L(R)> rather than <F> * PMREM(R(N)). In particular, a
    // grazing facet distribution sees sky well above the macro horizon.
    // The quadrature already integrates the physical slope variance and uses
    // only the small source-filter mip; do not convolve that result a second
    // time with the unrelated footprint/glint roughness. This deterministic
    // quadrature is for diffuse skies: compact solar images require importance
    // sampling, so retain the resolved mirror/glint path for clear/veiled skies.
    // F still owns the reflection/transmission split; clouds keep their existing
    // sun/deck-separated integral. Neither branch scales reflected energy.
    if (!below && (clouds.enabled > 0.5 || (lights.envKnobFlags & 8u) != 0u))
        skyFacet = slopeSky / max(F, 1e-3) * pm.envMapIntensity;
    const vec3 skyFacetBeforeBodyGrade = skyFacet;
    if (!below && clouds.enabled < 0.5 && (lights.envKnobFlags & 4u) != 0u) {
        // Remove the clear-day photographic blue cast from the water's sky
        // reflection. Normalize its energy before a small exposure trim that
        // preserves display-referred shadow/glint contrast after the hue change.
        // Geometry reflections, the visible sky and foam keep their own shading.
        const vec3 Y = vec3(0.2126, 0.7152, 0.0722);
        const vec3 tint = mix(vec3(0.55, 0.82, 1.0), vec3(0.17, 0.5, 0.95), smoothstep(0.0, 0.35, R.y));
        // Retain a calibrated red attenuation for the established coastal slate hue.
        const vec3 ungraded = skyFacet / tint * vec3(0.78, 1.0, 1.0);
        const vec3 body = ungraded * (0.90 * dot(skyFacet, Y) / max(dot(ungraded, Y), 1e-8));
        // The baked solar image is a highlight, not body colour: keep its white
        // glints instead of turning the sun path into warm/olive flecks.
        const float bodyWeight = 1.0 - smoothstep(1.0, 4.0, max(max(skyFacet.r, skyFacet.g), skyFacet.b));
        skyFacet = mix(skyFacet, body, bodyWeight);
    }
    const vec3 Nsmooth  = normalize(mix(Nmacro, N, reflFullSlope ? 1.0 : 0.35));
    vec3 reflectColor = vec3(0.0);
    float reflGeoShare = 0.0;// how much of the mirror direction ends on something
    // Speck fix (oceanOptions2 bit 16, on by default; THREEPP_OCEAN_REFL_DOWN_FIX=0 rolls back). The mirror lobe's rays that
    // leave a facet downward meet the next wave: they took only that water's surface reflection, F2 x the bake in
    // one mirror direction about the macro normal (under the world's horizon the bake's dark ground), and the rest
    // was dropped. The direction is kept over the horizon and (1 - F2) of this pixel's own water-leaving radiance
    // is added (below, once the body is known). The lobe is centred on the visible normal (waterVisibleNormal).
    const bool reflDownFix = (pc.oceanOptions2 & 65536u) != 0u;
    float reflDownBody = 0.0;// share of the lobe that ended in water below, times (1 - F2)
    float reflDownMasked = 0.0;
    if (reflDownFix) {
        const float pixelM = (cam.camAux.x > 0.5 ? 1.0 : length(P - gPrimaryOrigin)) * 2.0 * cam.projInverse[1][1] / max(float(pc.height), 1.0);
        reflDownMasked = smoothstep(0.1, 0.6, pixelM / max(dot(Nmacro, V), 0.02));
    }
#ifdef THREEPP_OCEAN_REFL_HITS_DIAG
    const bool debugReflHits = ((lights.envKnobFlags >> 8u) & 31u) == 17u;
#endif
    if (!below && (pc.oceanOptions2 & 32u) != 0u) {
        // Sigma = (bankedMss/2) I. This is ONLY the slope removed
        // from the resolved N, before the artificial footprint roughness floor.
        // Neither material roughness nor the retained FFT slope widens this lobe.
        // THREEPP_OCEAN_REFL_LOBE_SLOPE (oceanOptions2 bit 20, on by default; =0 rolls back): the lobe is as wide as the slope
        // the Fresnel and sky integral uses, banked + the material's unresolved remainder (roughness^4). With the
        // banked part alone it was a perfect mirror wherever the pixel resolves the fine waves (banked 0): near
        // water drew sharp mirror images of the tower and the hull, outlined, although 0.07 of mean-square slope
        // (the waves under the fine tile's cells) lies on it. What THREEPP_OCEAN_CAPILLARY draws is already off the banked.
        const bool lobeSlope = (pc.oceanOptions2 & 1048576u) != 0u;
        const float mss = max(bankedMss, 0.0) + (lobeSlope ? pow(pm.roughness, 4.0) : 0.0);
        // THREEPP_OCEAN_REFL_WATER_HIT (oceanOptions2 bit 21): a lobe ray that meets the sea ends there (traceRadiance); one
        // that leaves below the surface is traced like the rest and meets it at once. No closure of its own.
        const bool waterHit = (pc.oceanOptions2 & 2097152u) != 0u;
        gWaterHitMss = max(bankedMss, 0.0) + pow(pm.roughness, 4.0);
        const vec3 Nl = Nvis;
        const vec3 tangent = length(V - Nl * dot(Nl, V)) > 1e-4
            ? normalize(V - Nl * dot(Nl, V)) : normalize(cross(Nl, vec3(0.0, 0.0, 1.0)));
        const vec3 bitangent = cross(Nl, tangent);
        const uint pixelHash = pcgHash(uint(px.x) ^ pcgHash(uint(px.y)));
        const float pixelShift = float(pixelHash & 0x00ffffffu) / 16777216.0;
        float rotation = TWO_PI * fract(pixelShift + 0.618033989 * float(frame % 4096u));
        float radialShift = fract(pixelShift + 0.754877666 * float(frame % 4096u));
        if (lobeSlope) {
            // (four rays over a wide lobe: the spiral's turn and radial offset come from interleaved gradient noise
            // (Jimenez 2014) stepped each frame, so neighbours and successive frames take different, evenly spread
            // facets and the temporal filter averages them, instead of a white-noise grain)
            const vec2 ig = vec2(px) + 5.588238 * float(frame % 64u);
            rotation = TWO_PI * fract(52.9829189 * fract(dot(ig, vec2(0.06711056, 0.00583715))));
            radialShift = fract(52.9829189 * fract(dot(ig + vec2(47.0, 17.0), vec2(0.06711056, 0.00583715))));
        }
        uint selectionSeed = pcgHash(pixelHash ^ pcgHash(frame ^ 0x58e4a321u));
        float sumWeight = 0.0;
        // Four expensive rays; 32 cheap facets approximate the conditional
        // reflection distribution. Select within four radial strata with w/Ws.
        // E[Ws L_selected] = sum_j w_j L_j, with the SAME full denominator.
        // This avoids normalizing just four random Fresnel weights.
        [[dont_unroll]] for (int stratum = 0; stratum < 4; ++stratum) {
            float stratumWeight = 0.0;
            vec3 selectedFacet = Nl;
            [[dont_unroll]] for (int j = 0; j < 8; ++j) {
                const int index = stratum * 8 + j;
                const float u = (float(index) + radialShift) / 32.0;
                const float radius = sqrt(-mss * log(max(1.0 - u, 1e-7)));
                const float phi = rotation + 2.39996323 * float(index);
                const vec3 facet = normalize(Nl + radius * (cos(phi) * tangent + sin(phi) * bitangent));
                const float cosine = max(dot(facet, V), 0.0);
                const float weight = cosine / max(dot(facet, Nl), 1e-6) * waterFresnelExact(cosine, ior);
                stratumWeight += weight;
                if (weight > 0.0 && rnd(selectionSeed) * stratumWeight < weight) selectedFacet = facet;
            }
            if (stratumWeight <= 0.0) continue;
            const vec3 Rk = reflect(-V, selectedFacet);
            const float up = dot(Rk, Nmacro);
            bool hit = false;
            bool hullHit = false;
            vec3 leg;
            bool endedOnWater = false;
            if (up < 0.0 && !waterHit) {
                // No lift and no hull query. The ray stays in air, meets water,
                // and reflects again: Lww=F2(-R.Nmacro)*Lsky(reflect(R,Nmacro)).
                const vec3 secondRay = reflect(Rk, Nmacro);
                const float f2 = waterFresnelExact(clamp(-up, 0.0, 1.0), ior);
                leg = f2 * sampleEnvLod(secondRay, 0.12 * maxLod) * pm.envMapIntensity;
                // (speck fix: the second mirror direction is taken about the macro normal, which on a wave's back
                // face leans away: it often points under the world's horizon, into the bake's dark ground. It is
                // kept over the horizon; and what the water there does not reflect is its body light, added below)
                if (reflDownFix) {
                    vec3 skyDir = secondRay;
                    if (skyDir.y < 0.03) skyDir = normalize(vec3(skyDir.x, 0.03, skyDir.z));
                    leg = f2 * sampleEnvLod(skyDir, 0.12 * maxLod) * pm.envMapIntensity;
                    // (THREEPP_OCEAN_REFL_LOBE_SLOPE: the water the ray meets is this same rough water: it reflects the sky
                    // its facets see, the slope-integrated sky, as this pixel's rays that escape do. One blurred
                    // read of the bake just over the horizon takes in the bake's dark ground: 5 % of that sky.)
                    if (lobeSlope) leg = f2 * skyFacet;
                    // That is what a facet the pixel resolves shows. Where the pixel's footprint down range spans
                    // many facets (reflDownMasked), one that mirrors below the surface is a lee face hidden behind
                    // the crests in front of it, and the single normal the pixel carries stands for the visible
                    // ones: it takes the sky a visible facet reflects, as a ray that escapes does.
                    leg = mix(leg, skyFacet, reflDownMasked);
                    reflDownBody += stratumWeight * (1.0 - f2) * (1.0 - reflDownMasked);
                }
            } else {
                const vec3 rc = traceWaterReflection(reflOrig, Rk, doShadows, maxLod, reflLod, seed,
                                              /*cheapHits=*/true, /*probeHitFill=*/true, /*envInt=*/pm.envMapIntensity);
                hit = gTraceHitT > 0.0;
#ifdef THREEPP_OCEAN_REFL_HITS_DIAG
                hullHit = hit && gTraceHitHull;
#endif
                // Preserve the earlier sky closure and every hit-radiance operation.
                leg = hit ? rc * clamp(pm.specularIntensity, 0.0, 1.0) : skyFacet;
                if (gTraceHitWater) {// the sea's own radiance; what the hit does not reflect is its body, added below
                    endedOnWater = true;
                    leg = rc;
                    reflDownBody += stratumWeight * gTraceWaterBody;
                }
                leg = waterReflectionLeg(leg, reflOrig, Rk, below);
            }
#ifdef THREEPP_OCEAN_REFL_HITS_DIAG
            if (debugReflHits) {
                reflectColor += vec3(hullHit ? 0.25 : 0.0, (waterHit ? endedOnWater : up < 0.0) ? 0.25 : 0.0,
                                     0.25 * (0.5 + asin(clamp(up, -1.0, 1.0)) / PI));
            } else
#endif
            {
                reflectColor += stratumWeight * leg;
                reflGeoShare += hit ? stratumWeight : 0.0;
            }
            sumWeight += stratumWeight;
        }
#ifdef THREEPP_OCEAN_REFL_HITS_DIAG
        if (debugReflHits) { gTraceSkipWater = false; return reflectColor; }
#endif
        reflectColor /= max(sumWeight, 1e-12);
        reflGeoShare /= max(sumWeight, 1e-12);
        reflDownBody /= max(sumWeight, 1e-12);
    // Diagnostic rollback of the stratified reflection disk only; retain all other current shading.
    } else if ((pc.oceanOptions & 8u) != 0u) {
    const float jA = 0.0;
    // refine disagreeing geometry rays with a half-width ring turned 45 degrees.
    float lMin = 1e30, lMax = 0.0;
    for (int rk = 0; rk < 4; ++rk) {
        const float ra = jA + 1.5707963 * float(rk);
        vec3 Rk = reflect(-V, normalize(Nsmooth + vec3(cos(ra), 0.0, sin(ra)) * reflLobe));
        const float rkUp = dot(Rk, Nsurf);
        if (rkUp < 0.02) Rk = normalize(Rk + Nsurf * (0.02 - rkUp));
        const vec3 rc = traceWaterReflection(reflOrig, Rk, doShadows, maxLod, reflLod, seed, /*cheapHits=*/true,
                                      /*probeHitFill=*/true, /*envInt=*/pm.envMapIntensity);
        // (the specularIntensity below goes on each ray that landed on geometry: after the
        // loop gTraceHitT is the LAST ray's alone, so the whole average was scaled or not
        // by whether ray 3 hit, sky and geometry alike)
        vec3 reflectedLeg = gTraceHitT > 0.0 ? rc * clamp(pm.specularIntensity, 0.0, 1.0) : skyFacet;
        reflectedLeg = waterReflectionLeg(reflectedLeg, reflOrig, Rk, below);
        // Each lobe ray has its own endpoint. Applying the last ray's length
        // to the four-ray average made a hit/miss boundary dim all four rays.
        if (below) {
            const float legT = gTraceHitT > 0.0 ? min(gTraceHitT, 1.0e4) : 1.0e4;
            reflectedLeg = applyMurkWaterLeg(reflectedLeg, reflOrig, reflOrig + Rk * legT);
        }
        const float lk = dot(reflectedLeg, vec3(0.2126, 0.7152, 0.0722));
        lMin = min(lMin, lk), lMax = max(lMax, lk);
        reflectColor += reflectedLeg;
        if (gTraceHitT > 0.0) reflGeoShare += 0.25;
    }
    reflectColor /= 4.0;
    if (reflGeoShare > 0.0 && (reflGeoShare < 1.0 || lMax - lMin > 0.25 * lMax)) {
        vec3 refine = vec3(0.0);
        float refineGeo = 0.0;
        for (int rk = 0; rk < 4; ++rk) {
            const float ra = jA + 0.7853982 + 1.5707963 * float(rk);
            vec3 Rk = reflect(-V, normalize(Nsmooth + vec3(cos(ra), 0.0, sin(ra)) * (0.5 * reflLobe)));
            const float rkUp = dot(Rk, Nsurf);
            if (rkUp < 0.02) Rk = normalize(Rk + Nsurf * (0.02 - rkUp));
            const vec3 rc = traceWaterReflection(reflOrig, Rk, doShadows, maxLod, reflLod, seed, /*cheapHits=*/true,
                                          /*probeHitFill=*/true, /*envInt=*/pm.envMapIntensity);
            // (the specularIntensity below goes on each ray that landed on geometry: after the
            // loop gTraceHitT is the LAST ray's alone, so the whole average was scaled or not
            // by whether ray 3 hit, sky and geometry alike)
            vec3 reflectedLeg = gTraceHitT > 0.0 ? rc * clamp(pm.specularIntensity, 0.0, 1.0) : skyFacet;
            reflectedLeg = waterReflectionLeg(reflectedLeg, reflOrig, Rk, below);
            // Each lobe ray has its own endpoint. Applying the last ray's length
            // to the four-ray average made a hit/miss boundary dim all four rays.
            if (below) {
                const float legT = gTraceHitT > 0.0 ? min(gTraceHitT, 1.0e4) : 1.0e4;
                reflectedLeg = applyMurkWaterLeg(reflectedLeg, reflOrig, reflOrig + Rk * legT);
            }
            refine += reflectedLeg;
            if (gTraceHitT > 0.0) refineGeo += 1.0;
        }
        reflectColor = 0.5 * reflectColor + refine / 8.0;
        reflGeoShare = 0.5 * reflGeoShare + refineGeo / 8.0;
    }
    } else if (!below && (pc.oceanOptions & 1048576u) != 0u && (pc.oceanOptions & 402653184u) != 0u) {
        // Stratified weighted resampling of the *unchanged* full-slope rule.
        // Within each radial stratum, select i with probability w_i / W_s;
        // its contribution is W_s * L_i / sum(w). Thus E[result] equals the
        // full quadrature, including its projected-area/Fresnel normalization.
        // In particular, never normalize only the 2/4/8 selected weights:
        // that ratio estimator changes the lobe at low ray counts.
#ifdef THREEPP_OCEAN_REFL_LOBE_EXPERIMENT
        const float mss = waterGeometryMss(pm, bankedMss, P, Nmacro, V);
#else
        const float mss = pow(pm.roughness, 4.0) + bankedMss;
#endif
        const vec3 tangent = length(V - Nsmooth * dot(Nsmooth, V)) > 1e-4
            ? normalize(V - Nsmooth * dot(Nsmooth, V))
            : normalize(cross(Nsmooth, vec3(0.0, 0.0, 1.0)));
        const vec3 bitangent = cross(Nsmooth, tangent);
        const float rotation = TWO_PI * fract(0.754877666 * float(px.x) + 0.569840296 * float(px.y)
                                              + 0.618033989 * float(frame % 64u));
        const uint estimator = (pc.oceanOptions >> 27u) & 3u;
        const int rays = estimator == 1u ? 4 : estimator == 2u ? 8 : 2;
        const int candidates = (pc.oceanOptions & 262144u) != 0u ? 64 :
            (pc.oceanOptions & 4194304u) != 0u ? 16 : 32;
        const int perStratum = candidates / rays;
        // Independent of hit-light RNG and simulation time. pc.frame advances
        // during offline held-time settle draws as well as normal rendering.
        uint selectionSeed = pcgHash(uint(px.x) + pcgHash(uint(px.y) + pcgHash(frame ^ 0x50f4a321u)));
        float sumWeight = 0.0;
        [[dont_unroll]] for (int s = 0; s < rays; ++s) {
            float stratumWeight = 0.0;
            vec3 selectedFacet = Nsmooth;
            [[dont_unroll]] for (int j = 0; j < perStratum; ++j) {
                const int rk = s * perStratum + j;
                const float u = (float(rk) + 0.5) / float(candidates);
                const float radius = sqrt(-mss * log(1.0 - u));
                const float phi = rotation + 2.39996323 * float(rk);
                const vec3 facet = normalize(Nsmooth + radius * (cos(phi) * tangent + sin(phi) * bitangent));
                const float cosine = max(dot(facet, V), 0.0);
                const float weight = cosine / max(dot(facet, Nsmooth), 1e-6) * waterFresnelExact(cosine, ior);
                stratumWeight += weight;
                const float pick = rnd(selectionSeed);
                if (weight > 0.0 && pick * stratumWeight < weight) selectedFacet = facet;
            }
            if (stratumWeight <= 0.0) continue;
            vec3 Rk = reflect(-V, selectedFacet);
            const float rkUp = dot(Rk, Nsurf);
            if (rkUp < 0.02) Rk = normalize(Rk + Nsurf * (0.02 - rkUp));
            const vec3 rc = traceWaterReflection(reflOrig, Rk, doShadows, maxLod, reflLod, seed,
                                          /*cheapHits=*/true, /*probeHitFill=*/true, /*envInt=*/pm.envMapIntensity);
            const bool hit = gTraceHitT > 0.0;
            reflectColor += stratumWeight * waterReflectionLeg(hit ? rc * clamp(pm.specularIntensity, 0.0, 1.0) : skyFacet, reflOrig, Rk, below);
            reflGeoShare += hit ? stratumWeight : 0.0;
            sumWeight += stratumWeight;
        }
        reflectColor /= max(sumWeight, 1e-12);
        reflGeoShare /= max(sumWeight, 1e-12);
    } else if (!below && (pc.oceanOptions & 1048576u) != 0u) {
        // Use the SAME unresolved Gaussian tangent-slope measure
        // as waterSlopeIntegral. The old disk includes only banked FFT slope,
        // omitting the material's unresolved roughness^4. Weight incoming
        // radiance by visible projected facet area AND that facet's Fresnel.
        // Divide by the Fresnel-weight sum because F is applied once below.
#ifdef THREEPP_OCEAN_REFL_LOBE_EXPERIMENT
        const float mss = waterGeometryMss(pm, bankedMss, P, Nmacro, V);
#else
        const float mss = pow(pm.roughness, 4.0) + bankedMss;
#endif
        const vec3 tangent = length(V - Nsmooth * dot(Nsmooth, V)) > 1e-4
            ? normalize(V - Nsmooth * dot(Nsmooth, V))
            : normalize(cross(Nsmooth, vec3(0.0, 0.0, 1.0)));
        const vec3 bitangent = cross(Nsmooth, tangent);
        const float rotation = TWO_PI * fract(0.754877666 * float(px.x) + 0.569840296 * float(px.y)
                                              + 0.618033989 * float(frame % 64u));
        const int samples = (pc.oceanOptions & 262144u) != 0u ? 64 :
            (pc.oceanOptions & 4194304u) != 0u ? 16 : 32;
        float sumWeight = 0.0;
        [[dont_unroll]] for (int rk = 0; rk < samples; ++rk) {
            const float u = (float(rk) + 0.5) / float(samples);
            const float radius = sqrt(-mss * log(1.0 - u));
            const float phi = rotation + 2.39996323 * float(rk);
            const vec3 facet = normalize(Nsmooth + radius * (cos(phi) * tangent + sin(phi) * bitangent));
            const float cosine = max(dot(facet, V), 0.0);
            const float weight = cosine / max(dot(facet, Nsmooth), 1e-6) * waterFresnelExact(cosine, ior);
            if (weight <= 0.0) continue;
            vec3 Rk = reflect(-V, facet);
            const float rkUp = dot(Rk, Nsurf);
            if (rkUp < 0.02) Rk = normalize(Rk + Nsurf * (0.02 - rkUp));
            const vec3 rc = traceWaterReflection(reflOrig, Rk, doShadows, maxLod, reflLod, seed,
                                          /*cheapHits=*/true, /*probeHitFill=*/true, /*envInt=*/pm.envMapIntensity);
            const bool hit = gTraceHitT > 0.0;
#ifdef THREEPP_OCEAN_REFL_HITS_DIAG
            const bool pathVis = (pc.oceanOptions2 & 1u) != 0u && pm.oceanPeakWavelength > 0.0;
            const bool horizonVis = (pc.oceanOptions2 & 2u) != 0u && pm.oceanPeakWavelength > 0.0;
            const float clear = horizonVis && hit
                ? waterReflectionHorizonClear(reflOrig, Rk, gTraceHitT, pm.oceanPeakWavelength)
                : pathVis && hit ? waterReflectionPathClear(mss, Rk.y, gTraceHitT, pm.oceanPeakWavelength) : 1.0;
            if (debugReflHits) {
                // Reuse the production accumulators in this exclusive AOV
                // path instead of keeping four extra scalars live in normal
                // rendering. No optical weights or hit-only conditioning.
                if (horizonVis) {
                    // R=effective hull hits/16, G=blocked hull hits/16,
                    // B=MAX hull hitT/lambda_p (linear HDR, may exceed 1).
                    // The baseline AOV retains its direction/lift channels.
                    if (hit && gTraceHitHull) {
                        reflectColor.y += 1.0 - clear;
                        reflectColor.z = max(reflectColor.z, gTraceHitT / pm.oceanPeakWavelength);
                    }
                } else reflectColor += Rk;
                // R becomes effective hull hits / 16 with path visibility on.
                // G/B retain mean direction / centre-lift diagnostics.
                reflGeoShare += hit && gTraceHitHull ? clear : 0.0;
            } else
#endif
            {
#ifdef THREEPP_OCEAN_INSPECT
                if (pathVis || horizonVis) {
                    // Joint hull/sky estimator: every blocked-hit weight and
                    // every miss receives the same directional sky sampler
                    // used by water escapes; weights and F are untouched.
                    const float visibleHit = hit ? clear : 0.0;
                    const vec3 raySky = sampleEnvLod(Rk, reflLod) * pm.envMapIntensity;
                    reflectColor += weight * waterReflectionLeg(mix(raySky, rc * clamp(pm.specularIntensity, 0.0, 1.0), visibleHit), reflOrig, Rk, below);
                    reflGeoShare += weight * visibleHit;
                } else
#endif
                {
                reflectColor += weight * waterReflectionLeg(hit ? rc * clamp(pm.specularIntensity, 0.0, 1.0) : skyFacet, reflOrig, Rk, below);
                reflGeoShare += hit ? weight : 0.0;
                }
                sumWeight += weight;
            }
        }
#ifdef THREEPP_OCEAN_REFL_HITS_DIAG
        if (debugReflHits) {
            gTraceSkipWater = false;
            if ((pc.oceanOptions2 & 2u) != 0u)
                return vec3(reflGeoShare / 16.0, reflectColor.y / 16.0, reflectColor.z);
            const vec3 meanDirection = reflectColor / max(length(reflectColor), 1e-12);
            const float elevation = asin(clamp(meanDirection.y, -1.0, 1.0));
            const bool centreLifted = dot(reflect(-V, Nsmooth), Nsurf) < 0.02;
            return vec3(reflGeoShare / 16.0, 0.5 + elevation / PI, centreLifted ? 1.0 : 0.0);
        }
#endif
        reflectColor /= max(sumWeight, 1e-12);
        reflGeoShare /= max(sumWeight, 1e-12);
    } else {
    // Eight equal-weight samples of one disk. The previous conditional inner ring
    // changed the reflection kernel abruptly at hit-count/luminance thresholds,
    // producing hard red islands beside the hull. Rotate stratified samples over
    // the TAA cycle so silhouettes receive coverage instead of fixed ray-count bands.
    const float rotation = TWO_PI * fract(0.754877666 * float(px.x) + 0.569840296 * float(px.y)
                                          + 0.618033989 * float(frame % 64u));
    // Keep the stratified disk; opt-ins only reduce its sample count.
    const int reflectionSamples = (pc.oceanOptions & 262144u) != 0u ? 64 : (pc.oceanOptions & 1u) != 0u ? 2 :
        ((pc.oceanOptions & 2u) != 0u && length(P - gPrimaryOrigin) >= 60.0 ? 4 : 8);
    for (int rk = 0; rk < reflectionSamples; ++rk) {
        const float ra = rotation + 2.39996323 * float(rk);
        const float radius = reflectionSamples == 8
            ? sqrt((float(rk) + 0.5) / 8.0) * reflLobe
            : sqrt((float(rk) + 0.5) / float(reflectionSamples)) * reflLobe;
        vec3 Rk = reflect(-V, normalize(Nsmooth + vec3(cos(ra), 0.0, sin(ra)) * radius));
        const float rkUp = dot(Rk, Nsurf);
        if (rkUp < 0.02) Rk = normalize(Rk + Nsurf * (0.02 - rkUp));
        const vec3 rc = traceWaterReflection(reflOrig, Rk, doShadows, maxLod, reflLod, seed,
                                      /*cheapHits=*/true, /*probeHitFill=*/true, /*envInt=*/pm.envMapIntensity);
        vec3 reflectedLeg = gTraceHitT > 0.0 ? rc * clamp(pm.specularIntensity, 0.0, 1.0) : skyFacet;
        reflectedLeg = waterReflectionLeg(reflectedLeg, reflOrig, Rk, below);
        if (below) {
            const float legT = gTraceHitT > 0.0 ? min(gTraceHitT, 1.0e4) : 1.0e4;
            reflectedLeg = applyMurkWaterLeg(reflectedLeg, reflOrig, reflOrig + Rk * legT);
        }
        reflectColor += reflectionSamples == 8 ? reflectedLeg / 8.0 : reflectedLeg / float(reflectionSamples);
        if (gTraceHitT > 0.0) reflGeoShare += reflectionSamples == 8 ? 1.0 / 8.0 : 1.0 / float(reflectionSamples);
    }
    }
    if ((pc.oceanOptions & 16u) != 0u) { reflectColor = waterReflectionLeg(skyFacet, reflOrig, R, below); reflGeoShare = 0.0; }
    gTraceSkipWater = false;
    if (kRR) {
        gRRSpecF     = vec3(F);
        gRRSpecRough = effRough;
        gRRReflHitT  = gTraceHitT;
    }
#ifdef THREEPP_OCEAN_REFL_HITS_DIAG
    // reflhits is an above-water full-quadrature diagnostic. Unsupported
    // paths return a finite zero-count sentinel rather than normal radiance.
    if (debugReflHits) return vec3(0.0, 0.5, 0.0);
#endif
    // Reflected GEOMETRY strength, on specularIntensity (KHR_materials_specular).
    //
    // DELIBERATE DEVIATION, do not "correct" it: the glTF extension scales the whole
    // specular BRDF, environment included. Here it scales ONLY a reflection that landed
    // on geometry and leaves an escape to sky at full Fresnel. That asymmetry is the
    // point. Water's horizon sky-mirror band and its Snell's-window view from below are
    // what make water read as water; dimming those is never what is wanted, whereas a
    // shoreline mirrored at full radiance routinely is too strong.
    //
    // Why a new knob is needed at all: water's reflection is ONE hard mirror ray whose
    // energy is Schlick Fresnel from ior alone. effRough reaches only reflLod, the
    // environment-MISS mip, so material.roughness blurs a reflected sky and leaves a
    // reflected forest at full strength; water writes reflectImage = 0 so it never gets
    // the roughness-driven reflection denoiser opaque surfaces get; envMapIntensity
    // scales only the escape-to-environment terms, so lowering it dims the sky and makes
    // the ratio worse; and r0 = 0.02 against a grazing F of 0.25-0.67 leaves ior nearly
    // inert. Wave slope is not a lever either: the slope integral is dominated by the
    // finest cascade's cutoff, so a fully developed sea moves rms slope by -5% while
    // multiplying wave height by 2.85.
    //
    // gTraceHitT < 0 is the environment escape (deferred_shade_40_reflections.glsl:148).
    // Defaults to 1.0, so every existing scene runs the pre-change arithmetic textually.
    // Applied per ray in the loop above.

    // ── THE MIRROR IS UNDER WATER ────────────────────────────────────────────
    // Third medium on the same leg, and the one that decides whether the from-
    // below surface reads at all: R points DOWN into the water column, so the
    // traced leg runs through the very murk the caller composites onto the
    // primary. Skip it and total internal reflection becomes an impossibly clean
    // mirror — a crisp hull at 40 m inside water whose direct view fades out by
    // 15. applyMurk is the same expression and the same medium the primary leg
    // gets, so the two cannot disagree. A ray that escaped (gTraceHitT < 0) is
    // heading down and never leaves the water, so it takes the saturating 10 km
    // leg, i.e. the infinite-murk limit applyMurkSky returns.
    // Applied per reflected ray above, before averaging.

    // Transmission. From BELOW it is Snell's window; from above, the analytic
    // deep-water body (+ an optional shallow bottom).
    vec3 transmitColor;
    if (below) {
        // ── SNELL'S WINDOW ───────────────────────────────────────────────────
        // Inside the critical cone the refracted ray exits upward into air, so
        // the transmitted radiance is simply the sky along it — the WHOLE upper
        // hemisphere squeezed into ~97°, which is what makes the window both
        // bright and compressed. Sampled at the reflection's own LOD so the sun
        // disc survives (it is the brightest thing an underwater shot has) while
        // a point-feature sky still does not speckle across the chop.
        //
        // NO SCENE TRACE, deliberately: rigging, topsides and gulls ABOVE the
        // surface are absent from the window. The sky outweighs them by orders
        // of magnitude and a trace here would add a second full radiance ray to
        // every water pixel. Accepted for v1 — named so a mast that fails to
        // appear overhead reads as a known approximation, not a bug.
        const vec3 wDir = refract(-V, N, ior);
        transmitColor = (dot(wDir, wDir) > 1e-6)
                      ? sampleEnvLod(normalize(wDir), reflLod)
                      : reflectColor;// outside the window there is no transmitted ray (F == 1)
    } else {
        // Two media for the column under this surface (the murk the scene
        // declared, or the material's own), one refracted probe for a bottom in
        // either, and ONE traceRadiance call that shades the bottom for both:
        // each call site is a whole inlined copy of traceRadiance, and the
        // copies are what the driver's compile time is made of. Each medium
        // keeps its own probe before the call and its own use of the result
        // after it.
        const bool murkColumn = murkLive() && fog.murkDensity >= kMurkIsWater;
        bool  bottomHit = false;// the probe met something that is not a wave crest
        vec3  uwOrigin  = vec3(0.0);
        vec3  dRef      = vec3(0.0);
        float dBot      = 0.0;
        float maxVis    = 0.0;
        // The material path's terms, used again after the bottom trace.
        float skyLum    = 0.0;
        vec3  deepBody  = vec3(0.0);
        vec3  sigmaT    = vec3(0.0);
        vec3  sigmaS    = vec3(0.0);
        float bodyLeg   = -1.0;
        if (murkColumn) {
            // ── ONE MEDIUM, FROM ABOVE TOO ──────────────────────────────────────
            // The scene declared a murk (setUnderwaterMurk + setFogWaterSurfaceY),
            // so the column under this surface IS that murk — the same medium the
            // primary leg gets from below, the from-below mirror above, and every
            // particle sprite on both sides. Until this branch existed the view
            // from above used the MATERIAL's Beer-Lambert (attenuationColor over
            // attenuationDistance, sun path doubled) plus the analytic deep body,
            // so a hull at 3 m kept ~(0.007, 0.18, 0.27) of its light while a
            // sprite beside it, on the murk, kept 0.66 achromatic: the prop
            // ghosted into teal and the cavitation ropes blazed through the waves.
            // Two water models for one column; this makes it one. Refract the view
            // ray for real, probe for what is under the surface, and hand the hit
            // to applyMurk over the actual in-water leg — attenuation and veil
            // from the same expression the primary leg uses, so the two cannot
            // disagree. A miss is the infinite column, which is applyMurkSky's
            // saturated limit — exactly what the env miss shows from below.
            // Gated on murkLive(), so every scene without a murk keeps the
            // material path below byte-for-byte.
            dRef = refract(-V, N, 1.0 / ior);
            // From the air side a transmitted ray always exists; the guard only
            // covers a degenerate normal.
            if (dot(dRef, dRef) < 1e-6) dRef = -N;
            dRef = normalize(dRef);
            transmitColor = applyMurkSky(dRef);
            // Beyond ~6 optical depths the hit term is < ~2% — invisible. The same
            // budget the material path uses, on the medium that is actually there.
            maxVis   = min(6.0 / fog.murkDensity, 1.0e4);
            uwOrigin = P - N * SHADOW_EPS;
            rayQueryEXT rq;
            rayQueryInitializeEXT(rq, topAS, gl_RayFlagsOpaqueEXT, kRayMaskAll,
                                  uwOrigin, 1e-3, dRef, maxVis);
            while (rayQueryProceedEXT(rq)) {}
            if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
                const int   hitId = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
                if ((pc.oceanOptions & 524288u) == 0u && geoms[hitId].foamAddress == 0ul) {// not a wave-crest self-hit
                    bottomHit = true;
                    dBot      = rayQueryGetIntersectionTEXT(rq, true);
                }
            }
        } else {
        // Two terms:
        //
        // 1) ANALYTIC deep-water body — deliberately NOT a refraction ray for the
        // body itself. A stochastic downward continuation self-intersects this same
        // choppy surface (adjacent wave crests), which traceRadiance then shades as
        // opaque water → a chaotic black/white speckle that fills the screen when
        // viewed from ABOVE (F≈0.02 there, so the transmission term dominates).
        // Instead: the Beer-Lambert absorption tint lit by ambient skylight → a
        // smooth deep blue-green. (The path tracer can refract correctly because it
        // CONTINUES the path with medium tracking across the bounce.)
        vec3 tint = vec3(1.0);
        if (pm.attenuationDistance > 0.0)
            // 2× the crossing depth → an approximate down+up Beer-Lambert (2 crossings
            // over a dark seabed). The deeper absorption darkens it AND shifts it bluer
            // (red absorbs first), turning the old washed grey into rich deep-ocean blue.
            tint = pow(max(pm.attenuationColor, vec3(1e-6)), vec3(2.0 * pm.thickness / pm.attenuationDistance));
        // Lit by the sky's BRIGHTNESS (scalar luminance), NOT the coloured env — the
        // coloured sky washed the absorption hue to a dull grey. Scalar skylight keeps
        // the deep blue SATURATED. A view-depth term deepens it looking straight down
        // (longer water column) → a foreground-deep / horizon-bright gradient.
        // Dimmed (×0.45) because the deep body sits over a near-black seabed.
        //
        // BOTH terms keyed on SMOOTH per-view quantities, NOT the chop-perturbed N:
        // skyLum at the fixed zenith, depthFade on the view ELEVATION (V.y). Keying
        // them on N made every fine-cascade crest a 2.5×-brighter saturated-teal
        // BLOTCH (the deferred ocean's "cyan patches" — diagnosed by isolating this
        // term: the blotch pattern was entirely transmit). The analytic water BODY is
        // a smooth volumetric quantity; the chop belongs to Fresnel/reflection only.
        skyLum    = dot(sampleEnvLod(vec3(0.0, 1.0, 0.0), maxLod), vec3(0.2126, 0.7152, 0.0722));
        const float depthFade = mix(1.0, 0.4, clamp(V.y, 0.0, 1.0));// darker looking down into the column
        deepBody  = tint * (skyLum * 0.35 + lights.ambient) * depthFade;
        transmitColor = deepBody;

        // ── VOLUME EXTINCTION (absorption + scattering) ──────────────────────────
        // Everything from here to the in-scatter block is gated on
        // pm.scatterDistance > 0.0, so a material that never sets it takes the
        // byte-identical old arithmetic. sigma_a is read back out of the SAME
        // Beer-Lambert the tint above uses — attenuationColor is the per-
        // attenuationDistance transmittance, so sigma_a = -ln(att)/attDist [1/m]
        // — which keeps one medium description, not two.
        if (pm.scatterDistance > 0.0) {
            const vec3 sigmaA = (pm.attenuationDistance > 0.0)
                              ? -log(max(pm.attenuationColor, vec3(1e-4))) / pm.attenuationDistance
                              : vec3(0.0);
            sigmaS = pm.scatterColor / pm.scatterDistance * (1.0 + 2.0 * clamp(gWake.g, 0.0, 1.0) + 1.0 * clamp(gWake.r, 0.0, 1.0));
            // Floored: sigmaT divides below, and a channel with neither absorption
            // nor scattering is a vacuum whose in-scatter is zero anyway.
            sigmaT = max(sigmaA + sigmaS, vec3(1e-4));
        }
        // bodyLeg (declared above; -1 means "not probed yet") is the length of
        // the in-water leg for the single-scattering integral; the shallow-
        // bottom probe below sets it to the real bottom distance when it finds one.

        // 2) SHALLOW BOTTOM — the pond/shore path. Refract the view ray for real
        // and probe for geometry within the Beer-Lambert visibility range; a hit
        // (sand, rocks, a hull below the waterline) is shaded deterministically and
        // ADDED with the absorption over the actual in-water path. Design notes:
        //  • The deterministic single refracted ray doesn't have the deep path's
        //    speckle problem: it only degenerates when it re-hits the WATER surface
        //    (steep chop relative to depth) — detected via the instance's water
        //    marker (foamAddress) and dropped, falling back to the body alone.
        //  • deepBody stays UNREDUCED: its ×0.35/×0.45 constants were calibrated
        //    over a finite dark seabed, so the finite-column energy loss is already
        //    baked in — cross-fading by transmittance would double-count it and
        //    darken every existing deep scene. Adding the attenuated bottom instead
        //    keeps deep water bit-identical (dark bottom × tint ≈ 0) and lets a
        //    bright shallow floor show through. Mild over-energy where both terms
        //    are bright; acceptable for the pond regime.
        //  • doShadows=false on the bottom shade: water is opaque-masked for
        //    occlusion queries (see vulkan_shared.h — the PT's caustic energy
        //    balance relies on it), so a shadow ray from the bottom would report
        //    the surface above as a full occluder and render the floor sun-black.
        //    Unshadowed direct + the vertical-depth absorption term is the
        //    workable raster approximation of sunlight penetrating water.
        if (pm.attenuationDistance > 0.0) {
            // Beyond ~6 attenuation lengths the bottom term is < ~2% — invisible.
            maxVis = 6.0 * pm.attenuationDistance;
            const vec3 dRefRaw = refract(-V, N, 1.0 / ior);
            if (dot(dRefRaw, dRefRaw) > 1e-6) {
                dRef     = normalize(dRefRaw);
                uwOrigin = P - N * SHADOW_EPS;
                rayQueryEXT rq;
                rayQueryInitializeEXT(rq, topAS, gl_RayFlagsOpaqueEXT, kRayMaskAll,
                                      uwOrigin, 1e-3, dRef, maxVis);
                while (rayQueryProceedEXT(rq)) {}
                if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
                    const int   hitId = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
                    if ((pc.oceanOptions & 524288u) == 0u && geoms[hitId].foamAddress == 0ul) {// not a wave-crest self-hit
                        bottomHit = true;
                        dBot      = rayQueryGetIntersectionTEXT(rq, true);
                    }
                }
            }
        }
        }
        if (bottomHit) {
            // doShadows=false in both media: water is opaque-masked for occlusion
            // queries (see vulkan_shared.h), so a shadow ray from the bottom would
            // read the surface overhead as a full occluder and render the floor
            // sun-black. cheapHits=false: the floor IS the subject, keep it sharp;
            // probeHitFill=false: probes cannot resolve the underwater cavity.
            const vec3 bottom = traceRadiance(uwOrigin, dRef, /*doShadows=*/false,
                                              maxLod, 0.12 * maxLod, seed,
                                              /*cheapHits=*/false,
                                              /*probeHitFill=*/false,
                                              /*envInt=*/1.0);// transmitted, not IBL
            if (murkColumn) {
                // murkSunCaustic on the hit's direct light is what the from-below
                // primary gets too.
                transmitColor = applyMurk(bottom, uwOrigin, uwOrigin + dRef * dBot);
            } else {
                // View path + approximate sun path (vertical depth) through
                // the column, Beer-Lambert per channel.
                const float sunPath = dBot * (1.0 + clamp(-dRef.y, 0.0, 1.0));
                // With scattering live the column removes light by the FULL
                // extinction, not absorption alone: light scattered out of
                // the bottom ray is gone from it (it reappears in the
                // in-scatter term below, which is where that energy goes).
                // The else branch is the old expression, textually.
                const vec3  Tbot    = (pm.scatterDistance > 0.0)
                                    ? exp(-sigmaT * sunPath)
                                    : pow(max(pm.attenuationColor, vec3(1e-6)),
                                          vec3(sunPath / pm.attenuationDistance));
                // The body veil builds up over the water column: SCALAR
                // exponential saturation with a half-attenuation-length
                // constant. Deep water (d ≥ ~2 lengths) keeps ≥96% of the
                // calibrated deepBody — visually the old look — while a
                // metre-deep pond sheds most of the veil so its floor
                // reads instead of drowning in deep-ocean teal. Scalar on
                // purpose: a per-channel weight would hue-shift the body.
                const float bodyW = 1.0 - exp(-dBot / (0.5 * pm.attenuationDistance));
                // The probe stops at maxVis, so a floor that slopes away
                // crosses from "hit" to "no hit" along a depth isoline. Tbot
                // is small there (attenuationColor^6..12) but the floor it
                // multiplies is unshadowed sunlit albedo, so the residual
                // step printed as a hard band parallel to the shore in a
                // top-down view (netpen --terrain, 2026-09-05). Fade the
                // floor term out over the last 30% of the probe range so
                // the two branches meet continuously.
                const float edgeFade = 1.0 - smoothstep(0.7 * maxVis, maxVis, dBot);
                transmitColor = deepBody * bodyW + bottom * Tbot * edgeFade;
                bodyLeg = dBot;// the in-scatter integral stops at the floor
            }
        }
        if (!murkColumn) {

        // ── SINGLE SCATTERING IN THE WATER COLUMN ────────────────────────────────
        // The block above is pure Beer-Lambert: it can only ever REMOVE light from
        // the column, so the transmitted side is at best a darkened skylight. That
        // is why glacial and silty fjord water — bright turquoise precisely BECAUSE
        // rock flour scatters skylight sideways into the eye — could only be faked
        // by stretching attenuationDistance to non-physical values. This puts the
        // light back IN.
        //
        // ONE expression for the in-water leg: the deep body and the shallow-bottom
        // case differ only in the leg length handed to it (bodyLeg), never in the
        // integral.
        if (pm.scatterDistance > 0.0) {
            // Refracted view leg — the same refract() the bottom probe uses. From
            // the air side a transmitted ray always exists; the guard is for a
            // degenerate normal only.
            vec3 dRef = refract(-V, N, 1.0 / ior);
            if (dot(dRef, dRef) < 1e-6) dRef = -N;
            dRef = normalize(dRef);
            // No bottom within reach → integrate to the visibility limit, the same
            // 6-optical-depth budget the bottom probe budgets: past it the
            // exponential below is within 0.3% of its saturated value.
            const float legLen = (bodyLeg > 0.0)
                               ? bodyLeg
                               : 6.0 / max(max(sigmaT.r, sigmaT.g), sigmaT.b);
        const vec3 EinSky = (vec3(skyLum * kWaterInScatterSky) + lights.ambient) * (1.0 - F) * (1.0 + 2.6 * clamp(V.y, 0.0, 1.0));
        vec3 Ein = vec3(0.0);
        vec3 EdSun = vec3(0.0);// downwelling irradiance from direct lights, into the water
        const vec3 sunOrigIn = P + N * SHADOW_EPS;
        for (uint i = 0u; i < lights.dirCount; ++i) {
            const vec3  L   = normalize(lights.dirLights[i].direction);
            const float ndl = dot(N, L);
            if (ndl <= 0.0) continue;// sun below the surface plane: no transmitted beam
            vec3 sunRefr = refract(-L, N, 1.0 / ior);
            if (dot(sunRefr, sunRefr) < 1e-6) continue;
            sunRefr = normalize(sunRefr);
            const float phase = min(hgPhase(dot(dRef, sunRefr), kWaterInScatterG) * 4.0 * PI,
                                    kWaterInScatterPhaseMax);// 1 == isotropic
            // The light scattered back at this pixel entered the water elsewhere: follow
            // the view leg down a mean free path or so and the refracted sun leg back up
            // to the surface, and ask THERE whether the sun is blocked. The shadow edge
            // then blurs by the light's path through the column, as in real water,
            // instead of cutting the water to near black with a razor edge. Depths are
            // stratified and jittered per pixel and frame; the temporal passes resolve them.
            float visCol = 1.0;
            if (doShadows) {
                const float meanD = 4.0;// the green's path in coastal water, not the red's that sets legLen
                const float ign   = fract(52.9829189 * fract(0.06711056 * float(px.x) + 0.00583715 * float(px.y))
                                          + 0.61803399 * float(frame % 64u));
                const float rise  = 2.5 / max(L.y, 0.2);// clear of the wave crests before the ray starts
                visCol = 0.0;
                for (int k = 0; k < kWaterColumnShadowSamples; ++k) {
                    const float u = (float(k) + ign) / float(kWaterColumnShadowSamples);
                    const float d = -log(max(1.0 - u, 1e-3)) * meanD;
                    const float h = d * max(-dRef.y, 0.0);
                    const vec3  S = P + dRef * d - sunRefr * (h / max(-sunRefr.y, 0.2));
                    visCol += shadowVis(S + L * rise, L, 1e30);
                }
                visCol /= float(kWaterColumnShadowSamples);
            }
            Ein += lights.dirLights[i].color * (ndl * visCol * cloudShadowSample(P) * phase);
            EdSun += lights.dirLights[i].color * (ndl * visCol * cloudShadowSample(P));
        }
        debugColumnSun = EdSun;
        // Lamps light the column too, or floodlit water stays black at night however
        // bright the lamp: only their glints reached the eye. As the glint loop below:
        // this pixel's cluster cell, a spot's cone over dist^decay, no phase (a lamp's
        // pool is seen from all round it), and unshadowed for cost.
        {
            const uint cellBaseIn = clusterCellBase(px, clusterDist);
            const uint cCntIn = (pc.clusterLightCount == 0u) ? 0u : min(clusterGrid[cellBaseIn], kClusterMaxPerCell);
            for (uint ci = 0u; ci < cCntIn; ++ci) {
                const uint li = min(clusterGrid[cellBaseIn + 1u + ci], pc.clusterLightCount - 1u);
                vec3 toL = clusterLights[li].position - P;
                const float dist = length(toL);
                if (dist < 1e-4) continue;
                toL /= dist;
                const float ndl = dot(Nmacro, toL);
                if (ndl <= 0.0) continue;
                float atten;
                if (clusterLights[li].type > 0.5) {
                    const float spotAtten = smoothstep(clusterLights[li].cosAngleOuter, clusterLights[li].cosAngleInner,
                                                       dot(-toL, clusterLights[li].direction));
                    if (spotAtten <= 0.0) continue;
                    atten = spotAtten / max(pow(dist, clusterLights[li].decay), 0.01);
                } else {
                    atten = 1.0 / max(distFalloff(dist, clusterLights[li].decay), 0.01);
                }
                // lamps on water (oceanOptions2 bit 11, THREEPP_OCEAN_WATER_LAMPS=1): a lamp is a delta light like the sun, so
                // its in-scatter takes the same forward HG lobe. mu is the cosine between the light's refracted
                // propagation and the scattered light's, which travels back up the view leg (-dRef): a floodlight
                // shining down and away from a camera beside it is seen near backscatter, 0.1-0.2x isotropic.
                float lampPhase = 1.0;
                if ((pc.oceanOptions2 & 2048u) != 0u) {
                    const vec3 lRefr = refract(-toL, Nmacro, 1.0 / ior);
                    if (dot(lRefr, lRefr) > 1e-6)
                        lampPhase = min(hgPhase(dot(normalize(lRefr), -dRef), kWaterInScatterG) * 4.0 * PI,
                                        kWaterInScatterPhaseMax);
                }
                Ein += clusterLights[li].color * (atten * ndl * lampPhase);
                EdSun += clusterLights[li].color * (atten * ndl);
            }
        }
        // At a low view the eye gets the sky's reflection and little of the sunlit column:
        // teal is for looking down into the water. (A clear sea seen obliquely is sky blue.)
        Ein *= mix(0.3, 1.0, smoothstep(0.12, 0.65, clamp(dot(Nmacro, V), 0.0, 1.0)));
        Ein *= (1.0 - F); // Solar transmission into the water; final mix is the exit interface.
        // Source sigma_s·E_in attenuated by exp(-sigma_t·t), integrated over
        // t in [0, legLen]:  (sigma_s/sigma_t)·E_in·(1 - exp(-sigma_t·legLen)).
        // SATURATING by construction — looking straight down lengthens the leg
        // but the term can never exceed the single-scattering albedo times the
        // incident light, so there is no blow-up down the column.
        const vec3 ssa = sigmaS / sigmaT;
        const vec3 inScatter = ssa * Ein * (1.0 - exp(-sigmaT * legLen))
                             + vec3(0.98, 1.0, 1.06) * dot(ssa, vec3(0.2126, 0.7152, 0.0722)) * EinSky;

        // NO DOUBLE COUNT. The analytic deepBody is itself a stand-in for the
        // light that comes back out of a deep column — over a truly dark seabed
        // pure absorption returns nothing, and `skyLum * 0.35` is the hand-fit
        // that made deep water look lit at all. Now that the in-scatter is
        // computed for real, the stand-in is faded out by exactly the fraction
        // of extinction that IS scattering (the single-scattering albedo, as a
        // scalar so the body's hue does not shift). A non-scattering medium
        // keeps the old body whole; a strongly scattering one hands the job
        // over to the physical term instead of stacking two copies of it.
        const float ssAlbedo = clamp(dot(sigmaS / sigmaT, vec3(0.2126, 0.7152, 0.0722)),
                                     0.0, 1.0);
        // Diffuse upwelling sky plus a bounded directional return. The Rrs-only
        // closure discarded the view-dependent term that carries visible solar
        // shadows. Do not feed EdSun to both closures. The quarter-scale return
        // retains the established coastal-water calibration without the old
        // neutral sky veil; it is an appearance calibration, not measured radiometry.
        const vec3 Rrs = 0.33 * sigmaS / sigmaT;
        const vec3 skyEd = 0.94 * (waterSkyIrradiance() + PI * lights.ambient);
        const vec3 directScatter = 0.25 * ssa * Ein * (1.0 - exp(-sigmaT * legLen));
        transmitColor = Rrs * skyEd / (3.5 * ior * ior) + directScatter;
#ifdef THREEPP_OCEAN_WATER_BODY_EXPLORATION
        debugBodySky = Rrs * skyEd / (3.5 * ior * ior);
        debugBodyDirect = directScatter;
        // THREEPP_OCEAN_WATER_BODY_CLOSURE=rrs restores the earlier Rrs replacement. The
        // default retains the prior directional return and its calibration;
        // incoming sky Smith visibility remains enabled independently.
        if ((lights.envKnobFlags & 16u) != 0u && (pc.oceanOptions & 1073741824u) == 0u) {
            // sigmaS is bb (backscattering), NOT total scattering b. The old
            // directional HG source used bb as b, a fitted 0.25 gain, and no
            // water-to-air radiance conversion. Use one r_rs closure for both
            // irradiance sources. Gordon et al. 1988: r_rs=0.0949*u+0.0794*u^2,
            // u=bb/(a+bb). The final Fresnel mix supplies exit transmission;
            // 1/n^2 converts underwater radiance to air. No optical IOP changes.
            const vec3 rrs = (0.0949 + 0.0794 * ssa) * ssa;
            debugBodySky = rrs * skyEd / (ior * ior);
            debugBodyDirect = rrs * EdSun * 0.94 / (ior * ior);
            transmitColor = debugBodySky + debugBodyDirect;
        }
#endif
        // THREEPP_OCEAN_WATER_BUBBLES (oceanOptions2 bit 15): in a Beaufort 9-10 sea the breakers inject bubble plumes into the top
        // metre or two, and bubbles backscatter far more than clear water (Terrill et al. 2001; Zhang et al. 2002), flat
        // across the spectrum. A plume is a diffuse layer over the column: albedo Ab returns Ab*Ed/(pi n^2) and the
        // column below is seen through (1 - Ab). Its strength follows the whitecap fraction (void fraction ~ W,
        // Callaghan et al. 2008; W of Brumer et al. 2017, 1 at U10 26 m/s, nothing below Beaufort 6).
        // bubbles only at the crests: the load is the foam field's plume history and nothing else
        // (foam_world.comp's alpha: injected where an in-place breaker breaks, by its own trigger, cell and lobes, and
        // under the vessel's wash; gone in 18 s, spreading as it drifts). No sea-wide cloud, no background and no surface
        // foam term: where the history is 0 this adds nothing and the water is the switch-off water, bit for bit. The
        // layer's albedo is the water's own closure with the bubbles' backscatter added, 0.33 bbB / (a + bb + bbB) per
        // channel (Gordon et al. 1988), so its colour is this water's: a thin plume returns what the water absorbs
        // least (a dark teal in the North Sea preset, red gone) and a dense fresh one is turquoise. Works the same over
        // both closures (legacy and rrs).
        if ((pc.oceanOptions2 & THREEPP_OCEAN_OPT2_WATER_BUBBLES) != 0u && pc.oceanFoamTileSize > 0.0) {
            const float windB = pm.specularIntensity < 0.0 ? pm.dispersion : pm.iridescenceThicknessNm;
            if (windB < 100.0 && windB > 12.3) {
                const float plumeB = clamp(sampleFoamBicubic4(gFoamXZ / pc.oceanFoamTileSize).a, 0.0, 1.0);
                if (plumeB > 0.0) {
                    const float wN = 7.38e-4 * pow(windB - 4.23, 1.42) / 0.0586 * smoothstep(12.3, 19.0, windB);
                    // (a thin layer: its optical depth is seen along the refracted view, 1/mu_w, up to 1.5x at grazing)
                    const float cosB = clamp(dot(Nmacro, V), 0.0, 1.0);
                    const float muB = sqrt(1.0 - (1.0 - cosB * cosB) / (ior * ior));
                    // (calibration: THREEPP_OCEAN_WATER_BUBBLES_BB scales the backscatter by n/16, oceanOptions2 bits 24..29)
                    const float bbB = kWaterBubbleBb * (float((pc.oceanOptions2 >> 24u) & 63u) / 16.0) * wN * plumeB / muB;
                    const vec3 Ab = 0.33 * bbB / (sigmaT + bbB);
                    transmitColor = transmitColor * (1.0 - Ab) + Ab * (skyEd + 0.94 * EdSun) / (PI * ior * ior);
                }
            }
        }
    }
        }
    }

    // ── WAKE FIELD: bubbles under the surface ────────────────────────────────
    // A propeller's race is full of air for seconds after the film on top has
    // gone. Seen from above that is not white: the bubbles turn back light
    // that has been through a few decimetres of this water, so it is the
    // water's own hue, pale and milky, and it takes the sun and the shadows a
    // lit surface takes. Shaded as a diffuse body under the surface (the macro
    // normal: it is a volume, the chop is not in it) and laid in place of the
    // deep body by its optical thickness. The hue is the material's own
    // attenuation colour brought most of the way to white, so it follows
    // whatever sea the scene has made.
    if ((pc.oceanOptions & 32768u) == 0u && !below && gWake.g > 2e-3) {
        vec3 hue = max(pm.attenuationColor, vec3(1e-4));
        hue = mix(vec3(1.0), hue / max(max(hue.r, hue.g), hue.b), 0.25);// only a little of the water's hue: real DP wash has no turquoise glow
        const vec3 aerDiff = sampleEnvDiffuse(Nmacro) + lights.ambient + hemiAmbient(Nmacro);
        const vec3 aerLit  = shadeDiffuseDirect(P, Nmacro, V, 0.14 * gWakeAlbedo * hue, 1.0, 0.0,
                                                vec3(0.0), doShadows, aerDiff,
                                                vec3(0.0), 0.0,
                                                vec3(0.0),
                                                0.0, 1.3, 0.0, seed,
                                                /*addEmissive=*/true,
                                                /*cheapHit=*/false);
        // (broken by the foam's own veins: a uniform cloud read as a milky sheet of haze)
        const float aerBreak = smoothstep(0.35, 0.6, fbm4(gFoamXZ * 0.5));
        transmitColor = mix(transmitColor, aerLit, (1.0 - exp(-1.2 * gWake.g)) * mix(0.2, 0.8, aerBreak));
    }

#if SCOUT_WATER == 7
    return vec3(doShadows && lights.dirCount > 0u ? shadowVis(P + Nmacro * SHADOW_EPS, normalize(lights.dirLights[0].direction), 1e30) : 0.5);
#elif SCOUT_WATER == 6
    // Reflection hit classification: red = SELF-HIT on water (< 60 m),
    // green = other geometry, blue = escaped to sky.
    if (gTraceHitT < 0.0)      return vec3(0.0, 0.0, 1.0);
    if (gTraceHitT < 60.0)     return vec3(1.0, 0.0, 0.0);
    return vec3(0.0, 1.0, 0.0);
#elif SCOUT_WATER == 8
    // How the reflection ray ended (traceRadiance): blue escaped to the
    // sky, green the face of a wave it went into; on something that is not
    // water, red with no water crossed on the way, yellow after some.
    if (gScoutKind == 0) return vec3(0.0, 0.0, 4.0);
    if (gScoutKind == 1) return vec3(0.0, 4.0, 0.0);
    return gScoutWHits == 0 ? vec3(4.0, 0.0, 0.0) : vec3(4.0, 4.0, 0.0);
#elif SCOUT_WATER == 1
    return reflectColor;
#elif SCOUT_WATER == 2
    return transmitColor;
#elif SCOUT_WATER == 3
    return vec3(F * 8.0);
#elif SCOUT_WATER == 5
    return vec3(length(N - vec3(0.0, 1.0, 0.0)) * 20.0);
#endif

    if ((pc.oceanOptions & 16384u) != 0u) transmitColor = vec3(0.0);
    const vec3 transmitBeforeBodyGrade = transmitColor;
    if (!below && clouds.enabled < 0.5 && (lights.envKnobFlags & 4u) != 0u) {
        // The restored directional column return made the from-above body more
        // saturated. A luminance-preserving chroma calibration retains slate water.
        const float bodyY = dot(transmitColor, vec3(0.2126, 0.7152, 0.0722));
        transmitColor = mix(vec3(bodyY), transmitColor, 0.70);
    }
#ifdef THREEPP_OCEAN_INSPECT
    if (pc.inspectMode != 0u) {
        gInspectReflection=F*reflectColor; gInspectTransmission=(1.0-F)*transmitColor;
        gInspectGlitter=vec3(0); gInspectRayShare=reflGeoShare;
        gInspectFreshAlpha=0.0; gInspectLiveBand=false;
        if(pc.inspectSolo==4u) transmitColor=vec3(0);
    }
#endif
    if (reflDownBody > 0.0) reflectColor += reflDownBody * transmitColor;   // (speck fix: the water the lobe's downward rays met)
    vec3 col = mix(transmitColor, reflectColor, F);
    if ((pc.oceanOptions2 & 256u) != 0u) gWaterSpec = F * reflectColor;
    // Opt-in radiometric AOVs, before direct glints, foam and camera-leg media.
    // The caller bypasses those media and the application selects linear sRGB output.
    const uint waterDebug = (lights.envKnobFlags >> 8u) & 31u;
#ifdef THREEPP_OCEAN_WATER_BODY_EXPLORATION
    if (waterDebug == 13u) return 0.125 * (1.0 - F) * debugBodySky;
    if (waterDebug == 14u) return 0.125 * (1.0 - F) * debugBodyDirect;
#endif
    if (waterDebug == 1u) return 0.125 * F * skyFacet;
    if (waterDebug == 2u) return vec3(F);
    if (waterDebug == 3u) return 0.125 * F * reflectColor;
    if (waterDebug == 4u) return 0.125 * (1.0 - F) * transmitColor;
    if (waterDebug == 5u) {
        const vec3 Y = vec3(0.2126, 0.7152, 0.0722);
        const float bg = max(dot(applySkyFog(sampleBackground(R), R), Y), 1e-8);
        return 0.5 * vec3(dot(skyFacet, Y), dot(sampleEnvRaw(R, 0.0) * pm.envMapIntensity, Y),
                          dot(skyFacetBeforeBodyGrade, Y)) / bg;
    }
    if (waterDebug == 6u) return vec3(pow(pm.roughness, 4.0) + bankedMss, NdotV, max(dot(Nmacro, V), 0.0));
    if (waterDebug == 7u) return 0.125 * skyFacet;
    // lobe geometry (linear, 0.5 + 0.5 x): N.V, visible N.V, N.macro | mirror ray of the visible normal . macro, macro.y, V.macro
    if (waterDebug == 18u) return vec3(0.5 + 0.5 * dot(N, V), 0.5 + 0.5 * dot(Nvis, V), 0.5 + 0.5 * dot(N, Nmacro));
    if (waterDebug == 19u) return vec3(0.5 + 0.5 * dot(reflect(-V, Nvis), Nmacro), 0.5 + 0.5 * Nmacro.y, 0.5 + 0.5 * dot(V, Nmacro));
#ifdef THREEPP_OCEAN_REFL_LOBE_EXPERIMENT
    // Linear AOVs (sRGB encoded on readback). Lobe widths use x16 to
    // retain small variances in 8-bit captures; blue is geometry ray share.
    if (waterDebug == 11u) return vec3(reflGeoShare);
    if (waterDebug == 12u) return vec3(16.0 * bankedMss,
        16.0 * waterGeometryMss(pm, bankedMss, P, Nmacro, V), reflGeoShare);
#endif
    if (waterDebug == 10u) {
        const vec3 Y = vec3(0.2126, 0.7152, 0.0722);
        const float sunY = lights.dirCount > 0u ? dot(lights.dirLights[0].color, Y) : 0.0;
        return vec3(cloudShadowSample(P), 0.125 * sunY, 0.125 * dot(debugColumnSun, Y));
    }
    const vec3 bodyAfterGrade = col;
    const vec3 bodyBeforeGrade = mix(transmitBeforeBodyGrade,
        reflectColor + (skyFacetBeforeBodyGrade - skyFacet) * (1.0 - reflGeoShare), F);

    // ── Sun/light specular GLINTS ─────────────────────────────────────────────
    // Sharp analytic GGX highlights of the scene lights across the FFT-perturbed
    // facets — the glittering sparkle that sells "ocean". shadeWater never did this
    // (only reflect+transmit), so the deferred water was flat/dull; the env
    // reflection alone misses it (the scene's directional sun isn't baked into the
    // HDRI). The clearcoat layer (pm.clearcoat) adds a 2nd tighter lobe. This is the
    // bulk of the dull→epic difference, added here via analytic light NEE.
    {
        const float specRough = max(effRough, 0.03);
        const float kG        = (specRough + 1.0) * (specRough + 1.0) / 8.0;
        const float ccRough   = clamp(pm.clearcoatRoughness, 0.03, 0.3);
        const vec3  sunOrig   = P + N * SHADOW_EPS;
        for (uint i = 0u; i < lights.dirCount; ++i) {
            const vec3  L   = normalize(lights.dirLights[i].direction);
            const float ndl = dot(N, L);
            if (ndl <= 0.0) continue;
            float vis = 1.0;
            if (doShadows) {
                vis = (i == 0u && gWaterSunVis0 >= 0.0) ? gWaterSunVis0
                                                        : primarySunVis(i, P, N, sunOrig, L, false);
                if (i == 0u) gWaterSunVis0 = vis;
            }
            if (vis <= 0.0) continue;
            // Near, the sun's image is broken into sparkles by capillaries the wave spectrum doesn't reach: without them it was an oily smear
            const float glNear = smoothstep(70.0, 12.0, length(P - gPrimaryOrigin));
            const vec3  glN = !physicalGlitter && glNear > 0.0 ? normalize(N + 0.14 * glNear * vec3(fbm4(P.xz * 5.0 + vec2(1.1, -0.8) * pc.timeSec) - 0.47, 0.0,
                                                                                  fbm4(P.xz * 5.0 + vec2(31.0, 17.0) + vec2(-0.9, 1.2) * pc.timeSec) - 0.47)) : N;
            const float spec = physicalGlitter ? waterFacetGlitter(P,N,Nmacro,V,L,glitterCov,r0,false) :
                waterGlint(glN, V, L, NdotV, ndl, r0, specRough, kG, pm.clearcoat, ccRough);
            // (where the mirror direction ends on the hull, few facets can still reach round it to the sun)
            vec3 c = spec * ndl * lights.dirLights[i].color * (vis * cloudShadowSample(P)) * (1.0 - 0.85 * reflGeoShare);
            // Keep the white core and weak tails separate from the neutral body:
            // a bounded highlight contrast response avoids replacing blue with a veil.
            if (!below && clouds.enabled < 0.5 && (lights.envKnobFlags & 4u) != 0u)
                c *= mix(0.8, 1.15, smoothstep(0.1, 1.0, dot(c, vec3(0.2126, 0.7152, 0.0722))));
            // Directional lights are deltas → GGX spikes; clamp so a glint blooms
            // bright but never fireflies. (×4 over the global clamp → bright suns.)
            const float cl = max(max(c.r, c.g), c.b);
            const float gMax = pc.fireflyClamp * 4.0;
            if (cl > gMax) c *= gMax / cl;
            col += c;
            if ((pc.oceanOptions2 & 256u) != 0u) gWaterSpec += c;
#ifdef THREEPP_OCEAN_INSPECT
    if (pc.inspectMode != 0u) {
        gInspectGlitter+=c;
        if(pc.inspectSolo==3u) col-=c;
    }
#endif

        }
        if (waterDebug == 9u) return 0.125 * max(col - bodyAfterGrade, vec3(0.0));
        // POINT and SPOT lights — the same GGX glints with range and
        // inverse-square attenuation (and, for a spot, its cone), so a sweeping
        // beam (lighthouse) lights the waves it crosses instead of existing only
        // as the in-scattered air volume, and a lantern over the sea lights the
        // waves under it as it lights the hull beside it.
        // The lights are THIS PIXEL'S CLUSTER CELL, as the opaque path's are
        // (analyticDirectSplit), not the UBO's eight strongest of each type: in
        // a scene with more than eight (a town's street lamps, a road's head
        // lamps) a boat's flood light is rarely among those, and it lit her deck
        // and left the water under it black. A cell is culled by each light's
        // range, so what is looped here is what can reach the pixel.
        const uint cellBase = clusterCellBase(px, clusterDist);
        const uint cCnt = (pc.clusterLightCount == 0u)
                              ? 0u
                              : min(clusterGrid[cellBase], kClusterMaxPerCell);
        for (uint ci = 0u; ci < cCnt; ++ci) {
            const uint li = min(clusterGrid[cellBase + 1u + ci], pc.clusterLightCount - 1u);
            const bool isSpot = clusterLights[li].type > 0.5;
            vec3 toL = clusterLights[li].position - P;
            const float dist = length(toL);
            if (dist < 1e-4) continue;
            toL /= dist;
            const float ndl = dot(N, toL);
            if (ndl <= 0.0) continue;
            // Falloff as the opaque path's two loops (shadeDiffuseDirect), so the
            // two surfaces agree on how far a light reaches: a spot's cone over
            // dist^decay, a point's distFalloff.
            float atten;
            if (isSpot) {
                const float spotCos   = dot(-toL, clusterLights[li].direction);
                const float spotAtten = smoothstep(clusterLights[li].cosAngleOuter,
                                                   clusterLights[li].cosAngleInner, spotCos);
                if (spotAtten <= 0.0) continue;
                atten = spotAtten / max(pow(dist, clusterLights[li].decay), 0.01);
            } else {
                atten = 1.0 / max(distFalloff(dist, clusterLights[li].decay), 0.01);
            }
            const float range = clusterLights[li].range;
            if (range > 0.0) {
                const float tt = dist / range;
                const float t4 = tt * tt * tt * tt;
                const float wnd = max(1.0 - t4, 0.0);
                atten *= wnd * wnd;
            }
            if (atten <= 1e-6) continue;
            const float smV = (kSM && doShadows)
                    ? smLampVis(clusterLights[li].shadowSlot, clusterLights[li].type > 0.5, P, N,
                                clusterLights[li].position, vec2(gl_GlobalInvocationID.xy)) : -1.0;
            const float vis = smV >= 0.0 ? smV : doShadows ? shadowVis(sunOrig, toL, dist - 1e-2) : 1.0;
            if (vis <= 0.0) continue;
            // A lamp is a tiny source seen in a facet, not a blur: the lobe that integrates the
            // sun's glitter over a pixel dims it to nothing. A narrower one keeps the sparkle.
            float spec = waterGlint(N, V, toL, NdotV, ndl, r0, 0.7 * specRough, kG, pm.clearcoat, ccRough);
            // Beam POOL — a broad faint sheen where the beam strikes the
            // surface. Real water returns part of a grazing beam diffusely
            // (micro-facets beyond the GGX tail, spray, top-layer scatter);
            // at roughness 0.04 the pure specular lobe is so tight the lit
            // pool would otherwise be invisible unless perfectly aligned.
            // lamps on water (bit 11): not with the column's lamp in-scatter, which is that top-layer return computed
            // (sigma_s, the HG lobe, the leg): the flat 3.5 % Lambert pool on top of it read as a milky sheet.
            if ((pc.oceanOptions2 & 2048u) == 0u) spec += 0.035 / PI;
            vec3 c = spec * ndl * clusterLights[li].color * atten * vis;
            const float cl = max(max(c.r, c.g), c.b);
            const float gMax = pc.fireflyClamp * 4.0;
            if (cl > gMax) c *= gMax / cl;
            col += c;
            if ((pc.oceanOptions2 & 256u) != 0u) gWaterSpec += c;
        }
    }

    if (!below && clouds.enabled < 0.5 && (lights.envKnobFlags & 4u) != 0u) {
        // Protect the complete glint colour response, including the base under
        // an analytic highlight. Otherwise a neutral body exposes warm sun RGB
        // as olive flecks even when the baked solar image itself is protected.
        const vec3 Y = vec3(0.2126, 0.7152, 0.0722);
        const float glintShare = max(dot(col - bodyAfterGrade, Y), 0.0) / max(dot(col, Y), 1e-8);
        col += (bodyBeforeGrade - bodyAfterGrade) * smoothstep(0.10, 0.35, glintShare);
    }
    // The long waves in what the eye gets: a crest sees more sky and a trough less, by height, at the waves' own period
    if (!below && pm.sheenColor.b > 0.0) col *= 1.0 + clamp(0.04 * (P.y - (pm.sheenRoughness * 100.0 - 50.0)), -0.15, 0.15);
    if ((pc.oceanOptions2 & 256u) != 0u && !below && pm.sheenColor.b > 0.0) gWaterSpec *= 1.0 + clamp(0.04 * (P.y - (pm.sheenRoughness * 100.0 - 50.0)), -0.15, 0.15);
    // THREEPP_OCEAN_FOAM_KILLS_SPEC: aerated water does not mirror. One pass, before the foam layers.
    if ((pc.oceanOptions2 & 256u) != 0u && !below) {
        float aer = max(gWake.r, smoothstep(0.04, 0.15, gWake.g));
        if (kOceanCrestFoam == 0u && (pc.oceanOptions & 32u) == 0u && pc.oceanFoamTileSize > 0.0 && geoms[instIdx].foamAddress != 0ul) {
            const float nearR = floor(pm.sheenColor.b * 256.0);
            const float farFoam = nearR > 0.0 ? smoothstep(0.75 * nearR, nearR, length(P.xz)) : 1.0;
            aer = max(aer, sampleFoamBicubic(gFoamXZ / pc.oceanFoamTileSize) * farFoam);
        }
        col = foamKillSpec(col, aer, N, F, maxLod);
    }
#ifdef THREEPP_OCEAN_INSPECT
    if (pc.inspectMode != 0u && !below && pm.sheenColor.b > 0.0) {
        float k=1.0+clamp(0.04*(P.y-(pm.sheenRoughness*100.0-50.0)),-0.15,0.15);
        gInspectReflection*=k; gInspectTransmission*=k;
    }
#endif
    // ── CREST FOAM (foam_crest.comp) ──────────────────────────────────────────
    // Compiled in only for a frame that has it on (spec constant 8). The foam
    // is a field F on the water's REST coordinates q (foam_crest.comp: laid
    // down where the surface folds, decaying on the parcels that took it), and
    // a pixel's q is its position minus the displacement it was drawn with.
    // min(F, 1) is the fraction of the water covered. What covers it is read
    // off a photograph of sea foam (binding 92), laid on q too, so it rides
    // and stretches with the water and never slides over it: a point is foam
    // where photo > 1 - F. The photo is rank-equalised, so the covered
    // fraction is F, and what goes first as F falls is what was thinnest in
    // the real foam: holes open, then a lace, then threads.
    // Both fetches are over the pixel's footprint on the water, a strip along
    // the line of sight. Where the footprint has averaged the photo's
    // contrast away, the step widens by exactly what was lost (the values
    // under a pixel then span about that much around the fetched mean), so
    // far foam tends to F itself, a covered fraction, with no speckle.
    // Lit as foam is: a Lambertian reflector of albedo 0.55 where it is thick
    // (Whitlock, Bartlett and Gurganus 1982), under water's own interface
    // (F0 = 0.02); where it has only just closed it is a thin layer and lets
    // some of the water through. From below it is the dull raft the
    // accumulator's foam is.
    if (kOceanCrestFoam != 0u && pc.oceanFoamTileSize > 0.0 && geoms[instIdx].foamAddress != 0ul) {
        const vec2  q        = P.xz - gCrestDisp;
        const float distF    = length(P - gPrimaryOrigin);
        const float pixWorld = (cam.camAux.x > 0.5 ? 1.0 : distF) * 2.0 * cam.projInverse[1][1]
                             / max(float(pc.height), 1.0);// world m / pixel across the view
        const float pixAlong = pixWorld / max(dot(Nmacro, V), 0.05);
        const float vl       = length(V.xz);
        const vec2  along    = vl > 1e-4 ? V.xz / vl : vec2(1.0, 0.0);
        const vec2  gA       = along * pixAlong;
        const vec2  gX       = vec2(-along.y, along.x) * pixWorld;
        const float invTile  = 1.0 / pc.oceanFoamTileSize;
        float       foamF;
        float       foamFresh = 0.0;// foam_crest.comp's .g: 1 on a breaking crest, gone soon after
        if ((kOceanClimate & 2u) != 0u) {
            // The wave climate: the texture keeps the foam this water would
            // have with cascades 0 and 1 at 1, 2/3 and 1/3 of the sea
            // (foam_crest.comp, OCEAN_CLIMATE_LEVELS); take it at the map's
            // level here, cascade 1's (the waves the fold foam counts are
            // mostly its), linearly between levels and to none at 0.
            const vec4  lv = textureGrad(oceanCrestFoam, q * invTile, gA * invTile, gX * invTile);
            const float L  = clamp(oceanClimate(oceanClimateMap, q).y, 0.0, 1.0) * 3.0;
            foamF = L >= 2.0 ? mix(lv.b, lv.r, L - 2.0) : L >= 1.0 ? mix(lv.a, lv.b, L - 1.0) : lv.a * L;
            foamFresh = lv.g * clamp(L / 3.0, 0.0, 1.0);
        } else {
            const vec2 cf = textureGrad(oceanCrestFoam, q * invTile, gA * invTile, gX * invTile).rg;
            foamF     = cf.r;
            foamFresh = cf.g;
        }
        bool cfFilm = false;// the foam level is a vessel's film, drawn as it was
        // What the vessels left on the water is the same stuff, so it is
        // drawn the same way: the wake field's film (how much foam its flow
        // has carried here) reads as a foam level. It stops just short of a
        // closed sheet: a race is churned water, never a lid, and the last
        // tenth of the photo's range stays open so it shows as foam with
        // water in it rather than as a white shape. Its lit colour takes the
        // field's albedo scale.
        float cfAlbedo = kOceanFoamAlbedoWhitlock;
        if (geoms[instIdx].foamAddress > 1ul) {
            const float filmF = clamp((gWake.r - 0.05) / 0.35, 0.0, 0.9);
            if (filmF > foamF) {
                foamF    = filmF;
                cfAlbedo *= gWakeAlbedo;
                cfFilm   = true;
            }
        }
#if CREST_FOAM_SCOUT == 1
        // A 4 m checker on q under the raw F: the checker must ride the waves
        // (squeezed on crests, stretched in troughs) and never slide.
        return vec3(0.12 + 0.30 * mod(floor(q.x * 0.25) + floor(q.y * 0.25), 2.0)) +
               vec3(1.0, 0.35, 0.05) * min(foamF, 2.0);
#endif
        if (foamF > 1.0 / 512.0) {
            // The photo twice, at two sizes and angles; a slow selector on q
            // keeps one or the other, so neither placement's period shows
            // across a patch (their mean would not do: two foams averaged are
            // neither). kTile is metres across one tile: the photo's holes
            // are then a few decimetres to a metre, as they were.
            const float kTileA = 6.0;
            const float kTileB = 8.3;
            const vec2  rB  = vec2(0.6216100, 0.7833269);// cos, sin of 0.9 rad
            const vec2  qB  = vec2(rB.x * q.x - rB.y * q.y, rB.y * q.x + rB.x * q.y) + vec2(3.1, -1.7);
            const vec2  aB  = vec2(rB.x * gA.x - rB.y * gA.y, rB.y * gA.x + rB.x * gA.y);
            const vec2  xB  = vec2(rB.x * gX.x - rB.y * gX.y, rB.y * gX.x + rB.x * gX.y);
            const float nA  = textureGrad(oceanCrestFoamPattern, q / kTileA, gA / kTileA, gX / kTileA).r;
            const float nB  = textureGrad(oceanCrestFoamPattern, qB / kTileB, aB / kTileB, xB / kTileB).r;
            const float sel = smoothstep(-0.3, 0.3, sin(dot(q, vec2(0.17, 0.09)) +
                                                        1.7 * sin(dot(q, vec2(-0.07, 0.12)))));
            const float n   = mix(nA, nB, sel);
            // Contrast the footprint left in the photo: its structure runs
            // from about an eighth to a fiftieth of the tile, and each scale
            // keeps about 1 / sqrt(1 + (footprint / size)^2) along each axis.
            float kept = 0.0;
            {
                const vec3 sizes = vec3(1.0 / 8.0, 1.0 / 20.0, 1.0 / 50.0) * mix(kTileA, kTileB, sel);
                const vec3 kx = inversesqrt(1.0 + (pixWorld / sizes) * (pixWorld / sizes));
                const vec3 ka = inversesqrt(1.0 + (pixAlong / sizes) * (pixAlong / sizes));
                kept = dot(vec3(0.4, 0.35, 0.25), kx * ka);
            }
            // How far this point is inside the covered set, in the photo's own
            // units. Half-width of the edge: what the average took, never less
            // than a thin edge, which itself closes with F (no foam is no foam).
            const float excess = n - (1.0 - foamF);
            const float halfW  = max(0.5 * (1.0 - kept), min(0.04, foamF));
            // Foam on open water is a layer of bubbles a few deep, not a
            // lid: the water shows through it. LOOK constants (Lars,
            // 2026-11-10: "foam should have opacity and not so white"): a
            // third opaque where it is thinnest, three quarters where a
            // crest has just broken.
            float layer = mix(0.35, 0.75, clamp(foamF, 0.0, 1.0));
            // A crest's foam AGES. While the crest breaks it is a thick white
            // mass of bubbles on the crest; the crest then runs on and leaves
            // a patch on the water behind it that breaks up and clears, which
            // is most of the foam on the sea at any moment. Drawn alike, the
            // sea's white lay in the troughs. So the fresh part is nearly
            // closed, and what is left behind stays white but SHRINKS (the
            // photo's holes open as F falls, above) and goes TRANSPARENT with
            // it, as F squared: gone well before F is. LOOK constants
            // (upstream's).
            if (!cfFilm) {
                const float fresh = clamp(foamFresh, 0.0, 1.0);
                const float left  = clamp(foamF, 0.0, 1.0);
                layer    = 0.95 * max(fresh, left * left);
                cfAlbedo = 0.85;
            }
            const float cover  = clamp(excess / (2.0 * halfW) + 0.5, 0.0, 1.0) *
                                 layer * (below ? 0.12 : 1.0);
#if CREST_FOAM_SCOUT == 2
            return vec3(cover);
#endif
            if (cover > 0.0) {
                const vec3 cfAmb = sampleEnvDiffuse(N) + lights.ambient + hemiAmbient(N);
                const vec3 cfLit = shadeDiffuseDirect(P, N, V, vec3(cfAlbedo), 1.0, 0.0,
                                                      vec3(0.0), doShadows, cfAmb,
                                                      vec3(0.0), 0.0,
                                                      vec3(0.02),
                                                      0.0, 1.3, 0.0, seed,
                                                      /*addEmissive=*/true,
                                                      /*cheapHit=*/false);
                col = mix(col, cfLit, cover);
            }
        }
#if CREST_FOAM_SCOUT == 2
        return vec3(0.0);
#endif
    }

    // Foam: folded-surface coverage (foamAddress attribute + live world foam
    // texture) bleaches a diffuse whitewater layer over the glass. The world
    // accumulator decays exponentially, so COVERAGE IS FRESHNESS — the mask
    // uses it to run a whitewater lifecycle instead of uniformly dimming:
    //   fresh (≈1)  — near-solid sheet, micro-bubble texture
    //   aging (mid) — fBm pools bite harder as coverage drops, tearing the
    //                 sheet into patches
    //   residue     — a ridged filament web (the lace a collapsed whitecap
    //                 leaves behind) outlives the sheet, thinning to streaks
    // Matches closest_hit's foam shading.
    // Breaking crests (DisplacedMesh::Params::breakingCrests): white water
    // where the crest under this pixel is breaking NOW, from the surface's
    // own local steepness (water_displace.comp), so it sits on the crest top,
    // rides it and is gone when the crest relaxes: a step of half-width
    // kOceanBreakingRamp on the threshold, so cover > 0.5 exactly where it is
    // exceeded, and the steepness itself shapes it (largest on the crest,
    // falling down both faces). No texture, no noise, no accumulator. Thick
    // fresh foam is a Lambertian reflector of albedo 0.55, flat across the
    // visible (Whitlock, Bartlett and Gurganus 1982), with the specular of
    // water's own interface (n = 1.333, F0 = 0.02). From below the raft is
    // the dull patch the accumulator's foam is.
    if (gCrestEps > -1e29) {
        const float crestCover = smoothstep(kOceanBreakingSteepness - kOceanBreakingRamp,
                                            kOceanBreakingSteepness + kOceanBreakingRamp, gCrestEps) *
                                 (below ? 0.12 : 1.0);
        if (crestCover > 0.0) {
            const vec3 crestAmb = sampleEnvDiffuse(N) + lights.ambient + hemiAmbient(N);
            const vec3 crestLit = shadeDiffuseDirect(P, N, V, vec3(kOceanFoamAlbedoWhitlock), 1.0, 0.0,
                                                     vec3(0.0), doShadows, crestAmb,
                                                     vec3(0.0), 0.0,
                                                     vec3(0.02),
                                                     0.0, 1.3, 0.0, seed,
                                                     /*addEmissive=*/true,
                                                     /*cheapHit=*/false);
            col = mix(col, crestLit, crestCover);
        }
    }

    // Foam: the coarse field far off (near the vessel the wake field holds it, at 0.16 m a
    // texel; the material's sheenColor.b carries that near radius / 256 m). Both are sampled
    // at the rest position gFoamXZ. A cell noise stretched along the wind (sheenColor.rg, its
    // direction) breaks a coarse patch into several long caps.
    if (kOceanCrestFoam == 0u && (pc.oceanOptions & 32u) == 0u && pc.oceanFoamTileSize > 0.0 && geoms[instIdx].foamAddress != 0ul) {
        const float nearR = floor(pm.sheenColor.b * 256.0);
        const float farFoam = nearR > 0.0 ? smoothstep(0.75 * nearR, nearR, length(P.xz)) : 1.0;
#ifdef THREEPP_OCEAN_UNIFORM_CURRENT
        const vec2 foamMaterialXZ = gFoamXZ - vec2(currentX,currentZ)*pc.timeSec;
        const float cover = clamp(sampleFoamBicubic(foamMaterialXZ / pc.oceanFoamTileSize) * 2.5 / pow(gSeaRelief, 0.3), 0.0, 1.0) * farFoam;
#else
        const float cover = clamp(sampleFoamBicubic(gFoamXZ / pc.oceanFoamTileSize) * 2.5 / pow(gSeaRelief, 0.3), 0.0, 1.0) * farFoam;
#endif
        const bool stormLook = (pc.oceanOptions & 2147483648u) != 0u;   // THREEPP_OCEAN_STORM_LOOK, bit 31
        if (cover > 0.0 || stormLook) {
            const vec2  wv = pm.sheenColor.rg * 2.0 - 1.0;
            const float wl = length(wv);
            const vec2  wd = wl > 0.1 ? wv / wl : vec2(1.0, 0.0);
            gFoamWind = wd;
#ifdef THREEPP_OCEAN_UNIFORM_CURRENT
            const vec2 foamMaskXZ = foamMaterialXZ;
#else
            const vec2 foamMaskXZ = gFoamXZ;
#endif
            const vec2 maskXZ = foamEdgePosition(foamMaskXZ);
            const vec2 pW = vec2(dot(maskXZ, wd), dot(maskXZ, vec2(-wd.y, wd.x))) * vec2(0.18, 1.0);
            // foamlook: streaks that taper and fray, not cut: the cell's edge wanders with a finer noise, and the
            // streaking takes the fading foam (the trail) while a fresh cap's core stays whole
            const float fRaw = sampleFoamBicubic(foamMaskXZ / pc.oceanFoamTileSize);
            const float cell = wl > 0.1 ? smoothstep(0.26, 0.62, fbm4(pW * 0.32) + 0.2 * (fbm4(foamMaskXZ * 0.85) - 0.47)) : 1.0;
            const float fieldAmount = cover * mix(cell, 1.0, smoothstep(0.15, 0.45, fRaw));
            // THREEPP_OCEAN_FOAM_CARRY draws the field at full strength: with breaking cut to short events it holds the caps
            // where they fell, and is the foam the crests pass under (the live band only marks the breaking itself).
            // and only what the band broke there, and only while young (the age channel, ~1.4 s, which only the
            // band sets): older foam falls back to the residue's lace, as a cap's stage A gives way to stage B. The field
            // also holds the vessels' wake trail, which stays at the residue's look.
            const float residueAmount = (pc.oceanOptions & 512u) != 0u ? fieldAmount
                : (pc.oceanOptions & 536870912u) != 0u ? max(min(0.3, 0.45 * fieldAmount),
                      // (on a wide ramp of the raw field, 0.56 being the band's kept level, not the x 2.5 cover that
                      // saturates within a texel: the cap texture below then frays its edge instead of a crisp crest line)
                      // (THREEPP_OCEAN_FOAM_RESIDUE: stage B a few tenths of a second after its break is under the cap texture's
                      // solid knee, 0.45: residue is lace and grey (Koepke 1984: fresh foam albedo about 0.55, whitecap average 0.22),
                      // only a fresh deposit is white; at 0.7 a dense core lay as a smooth slab and a re-broken storm patch stayed white)
                      ((pc.oceanOptions2 & 8u) != 0u ? mix(0.45, 0.7, smoothstep(0.75, 0.95, textureLod(oceanFoamWorld, foamMaskXZ / pc.oceanFoamTileSize, 0.0).g)) : 0.7)
                      * farFoam * smoothstep(0.0, 0.56, fRaw / pow(gSeaRelief, 0.3))
                      // (THREEPP_OCEAN_FOAM_RESIDUE: stage B is drawn for its whole life, thinning and greying with its own amount;
                      // the age channel only marks it as foam a breaker laid, not a vessel's wake trail)
                      * ((pc.oceanOptions2 & 8u) != 0u ? smoothstep(0.02, 0.1, textureLod(oceanFoamWorld, foamMaskXZ / pc.oceanFoamTileSize, 0.0).g)
                                                       : smoothstep(0.1, 0.6, textureLod(oceanFoamWorld, foamMaskXZ / pc.oceanFoamTileSize, 0.0).g))
                      // broken by the cap texture's own clumps (the fresh path's pores, below): the field is smooth over
                      // metres, and drawn dense it lay as solid slabs edged by the crest line where the band last broke
                      * ((pc.oceanOptions2 & 8u) != 0u ? 1.0   // (the filaments below break stage B instead)
                         : smoothstep(0.55, 0.8, foamTexture(3.0 * foamMaskXZ, log2(max(1.0, 3.0 * length(P - gPrimaryOrigin) * 0.001 / max(NdotV, 0.05) / 0.0078))))))
                : min(0.3, 0.45 * fieldAmount);
            // under THREEPP_OCEAN_FOAM_RESIDUE all but the freshest deposit is drawn as filaments, stretched 2x when young to 8x
            // when old (the age channel: about 2.5 s to fall from 1 to 0.4)
            const float ageB = textureLod(oceanFoamWorld, foamMaskXZ / pc.oceanFoamTileSize, 0.0).g;
            const float residueDrawn = (pc.oceanOptions2 & 8u) != 0u
                ? residueAmount * mix(foamFilaments(foamMaskXZ, length(P - gPrimaryOrigin) * 0.001 / max(NdotV, 0.05), mix(8.0, 2.0, ageB), residueAmount / 0.7),
                                      1.0, smoothstep(0.85, 0.97, ageB))
                : residueAmount;
            // Fresh deposits share live caps' porous texture threshold and erosion
            // in foamLayer; never draw an untextured brightness overlay. Sample age
            // in the same material coordinates as amount, including uniform current.
            // (the storm look reads the freshness through the B-spline: a bilinear read drew the 2.93 m texels as
            // flat rectangles from a low camera; the default keeps the bilinear read)
            const vec4 fieldS = stormLook ? sampleFoamBicubic4(foamMaskXZ / pc.oceanFoamTileSize) : vec4(0.0);
            const float freshF = (pc.oceanOptions & 8192u) != 0u ? 0.0
                : smoothstep(0.25, 0.75, stormLook ? fieldS.g : textureLod(oceanFoamWorld, foamMaskXZ / pc.oceanFoamTileSize, 0.0).g);
            // Freshness raises only the input amount; foamLayer owns the live
            // cap's texture, erosion, opacity and lighting. Its metre-scale
            // clumps can cover a whole accumulator deposit, so break up the
            // boost with finer samples of that SAME texture before shading.
            // This mask can only remove fresh input, never fill the cap's holes.
            const float footprint = length(P - gPrimaryOrigin) * 0.001 / max(NdotV, 0.05);
            const float pores = smoothstep(0.65, 0.85,
                foamTexture(3.0 * foamMaskXZ, log2(max(1.0, 3.0 * footprint / 0.0078))));
            const float freshStrength = 10000.0;
            float amountEff = min(0.85, max(residueDrawn, freshF * freshStrength * pores));
            // breakers in place (oceanOptions2 bit 13): the field's breakers, white while fresh, greying as they age
            if ((pc.oceanOptions2 & 8192u) != 0u && stormLook)
                amountEff = max(amountEff, 0.9 * smoothstep(0.08, 0.7, fieldS.r) * mix(0.35, 1.0, smoothstep(0.1, 0.6, fieldS.g))
                                           * mix(0.45, 1.0, smoothstep(0.3, 0.65, fbm4(foamMaskXZ * 0.45))));   // lace, not a flat disc
            // the roller: where a breaker is fresh, the water above the crest's mid height is a dense, opaque white top, seen
            // from behind and at grazing angles (the lace alone was too faint against the grazing sky reflection)
            if ((pc.oceanOptions2 & 8192u) != 0u && stormLook) {
                const float irR = max(pm.iridescenceIOR - 1.0, 0.0), hsR = max(0.5, irR - 100.0 * floor(irR / 100.0));
                const float hRelR = (P.y - (pm.sheenRoughness * 100.0 - 50.0)) / hsR;
                const float roller = smoothstep(0.45, 0.85, fieldS.g) * smoothstep(0.15, 0.5, fieldS.r) * smoothstep(0.05, 0.3, hRelR)
                                   * mix(0.6, 1.0, smoothstep(0.35, 0.6, fbm4(foamMaskXZ * 0.8)));
                amountEff = max(amountEff, 0.97 * roller);
                gStormRoller = roller;
            }
            // THREEPP_OCEAN_CAP_TRAIL (oceanOptions2 bit 19; storm look): a cap is a thin bright crest line with a wider,
            // greyer, broken trail behind it that fades over a few seconds (deck footage; the NOAA Beaufort 9-10
            // plates), not a white patch. The white is the live band's, cut down to the line at its downwind edge (below).
            // What the field holds, the band's deposits and the in-place breakers' lobes, is the trail: foam drawn over the
            // water after the foam layer (through foamLayer even its grey, at 0.45 albedo under an overcast sky, read as
            // white against the water, and its texture's cut gave hard-edged flakes), near white while fresh and greying as
            // it ages (the age channel, 2.5 s to fall to 0.37), gone in about 6 s. The field is smooth over metres, so the
            // trail's breakup is a noise fixed to the water: patches of about 2 m with the residue's filaments as their
            // texture. Its region is the waves', its lace is not. The vessels' wake trail in the field has no age and
            // keeps its draw. Up close the breakers' white lobe and the roller are off (on the 2.93 m field they were the
            // solid blobs of 3-8 m); past 2 m of sea per pixel they stay, the far caps' white dashes.
            if ((pc.oceanOptions2 & 524288u) != 0u && stormLook) {
                const float trS = 1.5;
                const float amtT = smoothstep(0.02, 0.35, fieldS.r) * smoothstep(0.02, 0.12, fieldS.g);
                // (the event's size, its foam 5 m to both sides along the crest: a breaker of 10 m and up leaves a dense white
                // patch, a small one a thin grey trail. One strength for both gave the large one a pale film)
                float enF = 0.0;
                if (fieldS.r > 0.02) {
                    const vec2 wpT = vec2(-wd.y, wd.x);
                    enF = smoothstep(0.15, 0.45, min(sampleFoamBicubic4((foamMaskXZ + 5.0 * wpT) / pc.oceanFoamTileSize).r,
                                                     sampleFoamBicubic4((foamMaskXZ - 5.0 * wpT) / pc.oceanFoamTileSize).r));
                }
                if (amtT > 0.0 && !below) {
                    // (dense for the first 2 s behind the line, thinning to hairlines by 6 s: the age's e-folding is 2.5 s)
                    const float dens = smoothstep(0.03, 0.45, fieldS.g);
                    // the residue's own filaments carry it, as opaque strands (capTrailLace): braided and nearly
                    // merged while fresh, drawn out along the wind into hairlines as it ages, at their mean cover far off.
                    // (and the whole is cut into patches of about 2 m, near opaque: most of the trail while fresh, a fifth of it as it
                    // ages. From deck height a pixel's footprint on the back of a wave is drawn out past the fine filaments, they
                    // went to their mean cover, and the trail was an even translucent film with the ripples' glints through it, a
                    // glaze on the wave; patches this size hold at that angle. On a wind-stretched noise they were rows of dashes
                    // from the air, and as a coarse web of the filaments a sheet with round holes and rings: capTrailPatches)
                    // (an aged patch has a softer edge and is broken up inside: as one even grey piece it was a paper cut-out on the wave)
                    const float thrT = mix(0.9, -0.3, dens) - 0.8 * enF * dens;
                    // (white for its first half second, grey by 2.5 s: the age channel is 0.82 at 0.5 s and 0.37 at 2.5 s. dens
                    // holds at 1 for 2 s, and with the white on it the whole patch behind a front was one white)
                    const float freshT = smoothstep(0.35, 0.85, fieldS.g);
                    gCapTrailBright = enF * freshT;
                    const vec2  ptT = capTrailPatches(foamMaskXZ, wd, pc.timeSec);
                    gCapTrailFresh = freshT;
                    // (a large event's trail is denser, whiter and holds its patches longer)
                    const float cvT = min(mix(0.9, 0.97, enF), trS * amtT * smoothstep(thrT, thrT + mix(1.0, 0.6, dens), ptT.x) * mix(smoothstep(-0.7, 0.5, ptT.y), 1.0, 0.6 * dens)
                                          * mix(0.25, mix(0.6, 1.0, enF), dens) * mix(0.8, 1.0, 0.6 * enF * dens));
                    // (the strands age: foam a few seconds old is a thin layer of bubbles with the water showing through it, so
                    // an aged strand is soft-edged and at most 40 % opaque, the back of a patch greying into the water. Opaque
                    // to the end, the aged hairlines were hard pure-white ribbons, and some read as glints)
                    if (cvT > 0.005)
                        gCapTrail = mix(0.4, 1.0, freshT)
                                  * capTrailLace(foamMaskXZ, normalize((P - gPrimaryOrigin).xz + vec2(1e-5)), length(P - gPrimaryOrigin) * 0.001, footprint, cvT, mix(6.0, 1.6, dens), 1.0 - freshT);
                }
                // (past 2 m of sea per pixel a lobe is a few pixels and keeps its white top: the footage's far caps are white dashes)
                const float distT = length(P - gPrimaryOrigin);
                // (and past about 100 m, a pixel 0.08-0.16 m across, from any height: there the lace's strands and holes are
                // a few pixels each, and from the air a fresh patch read as a pale sheet with dark dots, a leopard skin. From
                // 35-70 m where the look is steep, where nothing foreshortens the holes. capTrailHandOver)
                const float farT = max(smoothstep(0.6, 2.0, distT * 0.001 / max(abs(gPrimaryOrigin.y - P.y) / distT, 0.02)),
                                       capTrailHandOver(distT, abs(gPrimaryOrigin.y - P.y)));
                amountEff *= 1.0 - (1.0 - farT) * smoothstep(0.0, 0.02, fieldS.g);
                gStormRoller *= farT;
                gCapTrail *= 1.0 - farT;
                // (a lobe is at 0.9 while it breaks and for about a second after; the band's deposits stay under 0.56)
                gCapBreaking = smoothstep(0.58, 0.8, fieldS.r) * smoothstep(0.6, 0.9, fieldS.g);
                const vec4 fieldD = sampleFoamBicubic4((foamMaskXZ + mix(1.5, 4.5, enF) * wd) / pc.oceanFoamTileSize);
                gCapBreakLine = smoothstep(0.25, 0.6, gCapBreaking) * (1.0 - smoothstep(0.1, 0.5, smoothstep(0.58, 0.8, fieldD.r) * smoothstep(0.6, 0.9, fieldD.g)));
            }
            // Beaufort 8-10 streaks, their own channel (b): laid by fresh caps, drawn out downwind by the
            // accumulator, living 8 s; drawn here as grey residue (foamLayer under 0.32, never cap white) broken into the
            // thin wind lines of foamStreaks, held off at a grazing look (where the texels' B-spline footprint draws out
            // into flat slabs), and on thresholds well up the B-spline's slope (near its base its iso-lines are its square support)
            const float stormWind = pm.specularIntensity < 0.0 ? pm.dispersion : pm.iridescenceThicknessNm;
            if (stormLook && stormWind < 100.0) {
                const float bfK = smoothstep(23.0, 27.0, stormWind);   // streaks from about 25 m/s (Holthuijsen 2012)
                // (no floor under the wind lines, and only where the deposit has aged: with the 0.3 floor the
                // fresh imprint of each crest, laid in one tick along the whole crest, drew as a thin white wire across the wind)
                // the earlier streak channel (fieldS.b) is retired; the wind streaks below replace it
                // wind streaks (Beaufort 9-10: "dense streaks of foam along the direction of the wind"). Old foam
                // gathered into Langmuir convergence lanes (Leibovich 1983): long, thin lines along the wind, lanes some 8-20 m
                // apart that wander and break, each lane its own width, strength and gaps (integer-hashed per lane), grey aged
                // foam (foamLayer under 0.4, never cap white). From about 25 m/s (bfK), seen from any height.
                if (bfK > 0.0) {
                    float streakL = 0.0;
                    // faint stranded lanes at the onset, growing to full strands by about 30 m/s
                    const float grow = smoothstep(25.0, 30.0, stormWind);
                    for (int ls = 0; ls < 2; ++ls) {   // two lane families, 7 deg either side of the wind, different spacing
                        const float aS = (ls == 0 ? 0.12 : -0.12);
                        const vec2  wdL = vec2(wd.x * cos(aS) - wd.y * sin(aS), wd.x * sin(aS) + wd.y * cos(aS)), wpL = vec2(-wdL.y, wdL.x);
                        const vec2  qS = vec2(dot(foamMaskXZ, wdL) - 0.03 * stormWind * pc.timeSec, dot(foamMaskXZ, wpL));
                        const float laneW = ls == 0 ? 11.0 : 19.0;
                        const float cS = (qS.y + 16.0 * (fbm4(qS * vec2(0.006, 0.02) + 13.0 * float(ls)) - 0.467)) / laneW;
                        const vec2  hS = hash22i(vec2(floor(cS), 17.0 + 31.0 * float(ls)));
                        const float dS = abs(fract(cS) - 0.5 + 0.35 * (hS.x - 0.5)) * laneW;   // metres from the lane's centre line
                        const float widthS = 1.0 + 2.6 * hS.y;
                        // (a lane under a few pixels goes to its mean cover, its share of the lane spacing, at lower contrast: drawn
                        // sharp it read as a scratch, widened as a field of grey strokes)
                        const float pxS = footprint / (0.25 * widthS);
                        const bool shortS = (pc.oceanOptions2 & 8192u) != 0u;
                        // the sub-pixel lane fade is under the breakers switch too, so bit 13 off restores the earlier lanes
                        const float lineS = mix(1.0 - smoothstep(0.35 * widthS, widthS, dS), 0.65 * widthS / laneW, shortS ? smoothstep(1.0, 4.0, pxS) : 0.0);
                        const float segS = smoothstep(mix(0.5, 0.44, grow) + (shortS ? 0.08 : 0.0), mix(0.64, 0.6, grow) + (shortS ? 0.08 : 0.0), fbm4(vec2(qS.x * (shortS ? 0.045 : 0.016) + 37.0 * hS.x, 11.0 * hS.y + 3.0 * float(ls))));
                        // strands: the lane is several thin threads of foam, marbled, that part and rejoin along the wind
                        const float frayS = smoothstep(mix(0.38, 0.3, grow), mix(0.72, 0.62, grow), fbm4(vec2(qS.x * 0.05, (qS.y - laneW * floor(cS)) * mix(2.4, 1.3, grow)) + 5.0 * hS)) * mix(0.5, 1.0, smoothstep(0.3, 0.7, fbm4(qS * vec2(0.2, 0.7) + 9.0 * hS)));
                        streakL = max(streakL, lineS * segS * frayS * mix(0.5, 1.0, hS.x));
                    }
                    streakL *= bfK * smoothstep(0.02, 0.08, NdotV);
                    gStormStreak = streakL * mix(0.55 / 0.8, 1.0, grow);
                    // (breakers in place, oceanOptions2 bit 13: the lanes as a faint texture, not bundles of grey lines; seen from
                    // contrast about a third, the lanes in short sparse segments)
                    if ((pc.oceanOptions2 & 8192u) != 0u) gStormStreak *= 0.35;
                }
            }
            col = foamLayer(col, P, N, V, pm.sheenRoughness, F, reflectColor, amountEff, foamMaskXZ,
                            length(P - gPrimaryOrigin) * 0.001 / max(NdotV, 0.05), below, doShadows, maxLod, seed);
        }
        // the wind streaks drawn as a thin layer of aged foam on top (through foamLayer they drowned in its cap
        // texture and erosion): matte, lit by the sky, about 55 % (25 m/s) to 80 % (30 m/s) opaque in a strand's core
        // (and the roller is opaque: the foam layer's Fresnel-weighted blend let the grazing sky through it)
        if (gStormRoller > 0.0)
            col = mix(col, vec3(0.92, 0.94, 0.96) * (0.6 * sampleEnvLod(vec3(0.0, 1.0, 0.0), maxLod) + 0.35 * max(dot(N, normalize(vec3(0.3, 1.0, 0.2))), 0.0) * vec3(1.0)), 0.75 * gStormRoller);
        if (gStormStreak > 0.0)
            col = mix(col, vec3(0.86, 0.89, 0.92) * (0.55 * sampleEnvLod(vec3(0.0, 1.0, 0.0), maxLod) + 0.25 * max(dot(N, normalize(vec3(0.3, 1.0, 0.2))), 0.0)), 0.8 * gStormStreak);
        if (gCapTrail > 0.0) {   // the caps' trail, matte and lit by the sky like the streaks
            col = mix(col, mix(mix(vec3(0.68, 0.72, 0.75), vec3(0.9, 0.92, 0.94), gCapTrailFresh), vec3(0.95, 0.96, 0.97), gCapTrailBright)
                           * (mix(0.65, 0.9, gCapTrailBright) * sampleEnvLod(vec3(0.0, 1.0, 0.0), maxLod) + 0.2 * max(dot(N, normalize(vec3(0.3, 1.0, 0.2))), 0.0)), gCapTrail);
            gCapTrail = 0.0; gCapTrailBright = 0.0;
        }
    }
    // spume. Past about 20 m/s the wind tears spray off the breaking crests and carries it downwind as a
    // translucent veil streaked along the wind (Beaufort 9-10), not as dots. Where fresh cap foam lies a few metres
    // upwind on the front of a crest, the water is veiled: thin, bright, fibrous along the wind. The wind speed rides in
    // the water material's otherwise unused thin-film thickness (iridescence is 0 on water; under 100 = m/s).
    const float spumeWind = pm.specularIntensity < 0.0 ? pm.dispersion : pm.iridescenceThicknessNm;
    // (Not under upstream's crest foam, kOceanCrestFoam: the veil's source is this field's caps, and with crest foam
    // on the whitecaps are crest foam's alone, natFoamScale 0 leaves the field only the vessels' foam.)
    if (kOceanCrestFoam == 0u && !below && pc.oceanFoamTileSize > 0.0 && spumeWind < 100.0) {
        const float spumeK = smoothstep(19.0, 26.0, spumeWind);
        const vec2  wvS = pm.sheenColor.rg * 2.0 - 1.0;
        if (spumeK > 0.0 && length(wvS) > 0.1) {
            const vec2  wdS = normalize(wvS);
#ifdef THREEPP_OCEAN_UNIFORM_CURRENT
            const vec2 spumeXZ = gFoamXZ - vec2(currentX, currentZ) * pc.timeSec;
#else
            const vec2 spumeXZ = gFoamXZ;
#endif
            float up = 0.0;
            // (storm look: the veil reaches further downwind, 6 taps over 15 m, denser, on a higher threshold)
            const bool stormV = (pc.oceanOptions & 2147483648u) != 0u;
            for (int k = 1; k <= (stormV ? 6 : 3); ++k)
                up = max(up, (1.0 - (stormV ? 0.12 : 0.22) * float(k)) * sampleFoamBicubic((spumeXZ - wdS * ((stormV ? 2.5 : 1.8) * float(k))) / pc.oceanFoamTileSize));
            const float hsS = max(0.5, (gSeaRelief - 0.75) / 0.5);
            const float crestS = smoothstep(-0.2 * hsS, 0.4 * hsS, P.y - (pm.sheenRoughness * 100.0 - 50.0));
            const vec2  pS = vec2(dot(gFoamXZ, wdS) - 9.0 * pc.timeSec, dot(gFoamXZ, vec2(-wdS.y, wdS.x)));
            const float fib = smoothstep(0.35, 0.75, fbm4(pS * vec2(0.06, 0.9)));
            const float veil = spumeK * (stormV ? smoothstep(0.2, 0.6, 2.5 * up) : smoothstep(0.1, 0.45, 2.5 * up)) * crestS * (0.35 + 0.65 * fib)
                             * (stormV ? smoothstep(0.9, 0.55, abs(V.y)) : 1.0);   // preset-only: a thin airborne layer, seen through from above (it drew grey smeared ovals in the nadir view)
            // THREEPP_OCEAN_RESIDUE_FILAMENTS drops the veil: on the residue-off path the field it samples upwind holds the
            // aged caps at full strength, and the veil drew them out as combed translucent slabs
            if (veil > 0.0 && (pc.oceanOptions2 & THREEPP_OCEAN_OPT2_RESIDUE_FILAMENTS) == 0u) col = mix(col, vec3(0.78, 0.81, 0.84) * sampleEnvLod(vec3(0.0, 1.0, 0.0), maxLod) * 0.6, (stormV ? 0.7 : 0.45) * veil);
#ifdef THREEPP_OCEAN_WATER_BODY_EXPLORATION
            debugWaterCover = 1.0 - (1.0-debugWaterCover)*(1.0-0.45*veil);
#endif
        }
    }
    // The breaking band: live, from the fold under this pixel, on the crest of a passing group. It can't drift off its wave, because it is the wave.
    // One whitecap model at a time: upstream's breaking crests (gCrestEps set) and crest foam (kOceanCrestFoam,
    // which draws its own folds) each replace this band, as they replace the field's caps (natFoamScale 0, the host).
    if (kOceanCrestFoam == 0u && !below && gCrestEps <= -1e29 && floor(pm.sheenColor.b * 256.0) > 74.5) {
        // the source of the fresh (stage A) caps again, as before: born where the wind sea's small
        // waves fold on the high part of a long crest's front, in groups that travel at the group velocity, and gone as
        // the crest passes (the field keeps their residue). The sea's Hs and the band's threshold ride in the water
        // material's unused thin-film IOR: 1 + Hs + 100 x round(100 x threshold).
        const float irB = max(pm.iridescenceIOR - 1.0, 0.0), thrK = floor(irB / 100.0);
        const float hsW = max(0.5, irB - 100.0 * thrK);
        const float dB = length(P - gPrimaryOrigin);
        // fade sub-pixel caps by flat-sea footprint, retaining aerial caps.
        // At a 57 m camera height this approximates the previous 400-600 m fade.
        const bool fadeByFootprint = (pc.oceanOptions2 & THREEPP_OCEAN_OPT2_BAND_FOOTPRINT_FADE) != 0u;
        const float footB = dB * 0.001 / max(abs(gPrimaryOrigin.y - P.y) / max(dB, 1e-3), 0.02);
        const float farB = fadeByFootprint ? 1.0 - smoothstep(2.8, 6.4, footB) : 1.0 - smoothstep(400.0, 600.0, dB);
        if (fadeByFootprint ? footB < 6.4 : dB < 600.0) {
            const vec2  wvB = pm.sheenColor.rg * 2.0 - 1.0;
            const vec2  wdB = length(wvB) > 0.1 ? normalize(wvB) : vec2(1.0, 0.0);
            const float cgB = 4.0 * sqrt(hsW);   // about half the peak phase speed
            const float grp = smoothstep(0.38, 0.6, fbm4(gFoamXZ * 0.012 - wdB * (0.012 * cgB) * pc.timeSec + 0.03 * vec2(-wdB.y, wdB.x) * pc.timeSec));
            const bool capTrail = (pc.oceanOptions2 & 524288u) != 0u;   // THREEPP_OCEAN_CAP_TRAIL
            const float upB = smoothstep(0.05 * hsW, 0.3 * hsW, P.y - (pm.sheenRoughness * 100.0 - 50.0));
            const float onCrest = upB * smoothstep(0.02, 0.15, dot(N.xz, wdB)); // upper breaking face
            const float thrB = thrK > 0.0 ? thrK / 100.0 : 0.62 - 0.03 * gSeaRelief;
            // The fold calibration describes the undisturbed spectrum. Hull
            // exclusion creates steep artificial crest gates beside the vessel;
            // the earlier, lower onset exposed them as instant opaque white shards.
            // Gate only this natural source in that local transition. Contact,
            // wash, residue, reflection and water-body terms are independent.
            const float hullKeep = (pc.oceanOptions & 8388608u) != 0u
                ? oceanNaturalCapHullKeep(geoms[instIdx].foamAddress, P.xz) : 1.0;
            // the band's onset is also pre-filtered over the pixel: the fold varies across its footprint by about
            // |grad fold| x footprint (central differences at half a footprint), and the onset ramp is widened by that spread
            // on both sides, the box-filtered smoothstep. A far cap's contour then fades over its pixels instead of ending at a
            // hard line; close up the spread is tiny and the ramp is unchanged.
            const float foldB = foamLiveFold(gFoamXZ);
            float spreadB = 0.0;
            if (fadeByFootprint && footB > 0.08) {
                const float hB = 0.5 * footB;
                spreadB = 0.5 * (abs(foamLiveFold(gFoamXZ + vec2(hB, 0.0)) - foamLiveFold(gFoamXZ - vec2(hB, 0.0)))
                               + abs(foamLiveFold(gFoamXZ + vec2(0.0, hB)) - foamLiveFold(gFoamXZ - vec2(0.0, hB))));
            }
            float band = smoothstep(thrB - spreadB, thrB + 0.22 + spreadB, foldB) * grp * onCrest * farB * hullKeep;
            float capDenseAge = 0.0;   // (THREEPP_OCEAN_CAP_TRAIL: how aged the dense patch behind a large front is here)
            // storm look only (bit 31): grouped spilling breakers. In a Beaufort 9-10 group the long crests break
            // over a stretch of tens of metres: white water over the upper half of the long crest (more on its front face), torn into streaks along the wind, born and
            // gone as the group passes (the field keeps the residue). Groups larger and rarer than the cap groups above (the chosen of three strengths).
            if ((pc.oceanOptions & 2147483648u) != 0u) {
                const vec2  wpB = vec2(-wdB.y, wdB.x);
                const float g2 = smoothstep(0.56, 0.72, fbm4(gFoamXZ * 0.0075 - wdB * (0.0075 * cgB) * pc.timeSec + 0.02 * wpB * pc.timeSec + 7.3));
                const float hRel = (P.y - (pm.sheenRoughness * 100.0 - 50.0)) / hsW;
                const float faceB = smoothstep(0.04, 0.22, dot(N.xz, wdB));
                const float tear = smoothstep(0.32, 0.62, fbm4(vec2(dot(gFoamXZ, wdB) * 0.18, dot(gFoamXZ, wpB) * 0.55)));
                const float spill = g2 * mix(0.6, 1.0, faceB) * smoothstep(-0.1, 0.3, hRel) * mix(0.3, 1.0, tear) * farB * hullKeep;
                if ((pc.oceanOptions2 & 8192u) == 0u) band = max(band, 0.95 * spill);   // (off under breakers in place)
            }
            // THREEPP_OCEAN_CAP_TRAIL: the white is the crest line only. The band above marks where a crest is breaking, a
            // region metres wide on the slope-budget sea (its fold is the waves of 2 m and up), and drawn whole it was a solid
            // patch draped over the crest. The line is the strip of it within 2 w of its downwind edge, the breaker's toe:
            // where the fold 2 w downwind is back under the threshold. w is 0.75 m, from half that where the crest barely
            // breaks to 1.5 times where it folds hard, so the line's weight follows the wave, and no noise shapes a small breaker's. A
            // line under a pixel goes to its share of the pixel. An in-place breaker (the field's, while it breaks) shows
            // as a line too, its lobe's downwind rim. Behind the line the rest of the breaking region is the trail.
            // (the line, its trail and the dense patch are the near look. With distance they hand back to the band as it is
            // without the switch, caps broken up by the fine waves' faces: the line used to widen there until it was the whole
            // breaking region, drawn solid, and from 150-200 m off the caps were smooth white bands, paint strokes)
            const float band0 = band;
            const float handB = capTrail ? max(smoothstep(0.6, 2.0, footB), capTrailHandOver(dB, abs(gPrimaryOrigin.y - P.y))) : 0.0;
            if (capTrail && handB < 1.0) {
                const float brkB = gCapBreaking * farB * hullKeep;
                // (the breaking region runs over the crest onto its back: held to the front face, a camera looking down the
                // wind saw neither line nor trail)
                const float bandR = smoothstep(thrB - spreadB, thrB + 0.22 + spreadB, foldB) * grp * upB * smoothstep(-0.3, -0.1, dot(N.xz, wdB)) * farB * hullKeep;
                band = max(band, bandR);
                if (max(bandR, brkB) > 0.01) {
                    // The event's size, how deep its breaking region is along the wind (the fold over the threshold 4, 9 and 16 m
                    // up and down the wind, 0..32 m): a small breaker keeps the thin line and the grey trail, a large one, from
                    // 8 m deep to full at 26 m, gets a front metres thick and a dense white patch behind it. And how far
                    // behind its front this point is (0..1 over 16 m: the breaking region downwind of it).
                    const float fU1 = foamLiveFold(gFoamXZ - 4.0 * wdB), fU2 = foamLiveFold(gFoamXZ - 9.0 * wdB), fU3 = foamLiveFold(gFoamXZ - 16.0 * wdB);
                    const float fD1 = foamLiveFold(gFoamXZ + 4.0 * wdB), fD2 = foamLiveFold(gFoamXZ + 9.0 * wdB), fD3 = foamLiveFold(gFoamXZ + 16.0 * wdB);
                    const float enB = smoothstep(8.0, 26.0, 4.0 * (smoothstep(thrB, thrB + 0.1, fU1) + smoothstep(thrB, thrB + 0.1, fD1))
                                                          + 5.0 * (smoothstep(thrB, thrB + 0.1, fU2) + smoothstep(thrB, thrB + 0.1, fD2))
                                                          + 7.0 * (smoothstep(thrB, thrB + 0.1, fU3) + smoothstep(thrB, thrB + 0.1, fD3)));
                    const float aheadB = (4.0 * smoothstep(thrB, thrB + 0.25, fD1) + 5.0 * smoothstep(thrB, thrB + 0.25, fD2) + 7.0 * smoothstep(thrB, thrB + 0.25, fD3)) / 16.0;
                    // (far off the line keeps a pixel and a half's width, up to 2.5 m: a breaking face stands toward a low camera and
                    // is not foreshortened as the flat sea's footprint assumes. From overhead the footprint is small and w is the wave's)
                    const vec2  ptP = capTrailPatches(gFoamXZ, wdB, pc.timeSec);
                    // (a large event's front is metres thick, and its back edge is torn, not a contour of the fold)
                    const float wL = max(mix(0.75 * mix(0.5, 1.5, smoothstep(0.0, 0.35, foldB - thrB)), 2.5 * mix(1.0, 0.45 + 0.9 * smoothstep(-1.2, 1.2, ptP.x), enB), enB),
                                         min(0.75 * footB, 2.5));
                    // (the fine waves' own shape, their front faces and folds, and how hard the crest breaks here: what breaks
                    // the dense patch up inside)
                    const float keepB = smoothstep(0.1, 0.6, (foamWaveShape(N, gFoamXZ, wdB) - 0.4) / 0.9 + 0.35 * smoothstep(0.0, 0.35, foldB - thrB));
                    const float foldD = foamLiveFold(gFoamXZ + 2.0 * wL * wdB);
                    const float cutB = smoothstep(thrB - spreadB, thrB + 0.1 + spreadB, foldD);
                    // (behind the line, the rest of the breaking region is the foam the line has just left: grey, broken, thinning
                    // with distance behind the line. The field keeps a trail only inside its breaking events, so most caps had
                    // none and drew as a bare line. The pattern is fixed to the water and the crest moves over it)
                    const float inB = bandR * cutB;
                    float denseB = 0.0;
                    if (inB > 0.01 && !below) {
                        const float backB = smoothstep(thrB - spreadB, thrB + 0.1 + spreadB, foamLiveFold(gFoamXZ + (2.0 * wL + mix(2.5, 5.0, enB)) * wdB));
                        const float fpB = dB * 0.001 / max(NdotV, 0.05);
                        // (the trail is foam, not a film: the field trail's patches (above), near white over most of the first
                        // 2.5 m behind the line, thinning to grey patches a fifth covered past 6 m, drawn as the same opaque
                        // strands. As a translucent veil it read from deck height as a glaze on the wave)
                        const float back2 = smoothstep(thrB - spreadB, thrB + 0.1 + spreadB, foamLiveFold(gFoamXZ + (2.0 * wL + mix(6.0, 12.0, enB)) * wdB));
                        const float distB = 0.5 * (backB + back2);
                        const float thrP = mix(-0.3, 0.9, distB);
                        // (behind a large event's front a dense white patch, the foam layer's own: solid for the first metres,
                        // then its holes and lace as the amount falls with distance behind the front, gone by about 16 m)
                        // (and broken by the waves under it, to a third on their backs: with the noise alone it was one white sheet)
                        denseB = enB * inB * (1.0 - smoothstep(0.05, 0.95, aheadB)) * mix(0.55, 1.0, smoothstep(-0.8, 0.6, ptP.x)) * mix(0.35, 1.0, keepB);
                        float trB = min(0.92, 1.5 * inB * smoothstep(thrP, thrP + mix(0.6, 1.0, distB), ptP.x) * mix(smoothstep(-0.7, 0.5, ptP.y), 1.0, 0.6 * (1.0 - distB))
                                                * mix(0.6, 0.25, distB) * 0.8) * (1.0 - smoothstep(0.1, 0.4, denseB));
                        // (and ages with its distance behind the line, as the field's trail does with its age)
                        if (trB > 0.005) trB = mix(1.0, 0.4, smoothstep(0.2, 0.85, distB)) * capTrailLace(gFoamXZ, normalize(V.xz + vec2(1e-5)), dB * 0.001, fpB, trB, mix(1.6, 5.0, distB), distB);
                        const float nearB = 1.0 - smoothstep(0.0, 0.75, distB);
                        trB *= 1.0 - handB;
                        col = mix(col, mix(vec3(0.62, 0.66, 0.69), vec3(0.92, 0.94, 0.96), nearB)
                                       * (0.7 * sampleEnvLod(vec3(0.0, 1.0, 0.0), maxLod) + 0.2 * max(dot(N, normalize(vec3(0.3, 1.0, 0.2))), 0.0)), trB);
                    }
                    band *= 1.0 - cutB;
                    band = max(band, gCapBreakLine * farB * hullKeep);
                    band *= min(1.0, 2.0 * wL / max(max(footB, 2.0 * wL), 1e-3));
                    // (the dense patch ages with its distance behind the front, from 4 m to 12 m: see below)
                    capDenseAge = (1.0 - handB) * smoothstep(0.05, 0.6, aheadB) * smoothstep(0.0, 0.2, denseB - band);
                    band = max(band, denseB);
                } else band = 0.0;
                band = mix(band, band0, handB);
            }
            if (band > 0.01) {
#ifdef THREEPP_OCEAN_INSPECT
                gInspectLiveBand=true;
#endif
            const float capGain = 1.4;
            const vec3 colWaterB = col;
            col = foamLayer(col, P, N, V, pm.sheenRoughness, F, reflectColor, min(1.0, capGain * band), gFoamXZ, dB * 0.001 / max(NdotV, 0.05), below, doShadows, maxLod, seed);   // (1.4, was 0.95: the caps solid and bright)
            // (THREEPP_OCEAN_CAP_TRAIL: foam a few seconds old is a thin layer of bubbles, and the water shows through it: the back of
            // a large front's patch greys out, to 30 % opaque. White to its last metre, the patch was a cut-out on the sea)
            if (capDenseAge > 0.0) col = mix(col, colWaterB, 0.7 * capDenseAge);
#ifdef THREEPP_OCEAN_INSPECT
                gInspectLiveBand=false;
#endif
            }
        }
    }
    // The wake field's film: wash, near caps, contact.
    if (kOceanCrestFoam == 0u && (pc.oceanOptions & 64u) == 0u && gWake.r > 4e-3) {
        const vec2 wvN = pm.sheenColor.rg * 2.0 - 1.0;
        // (where the water is full of air it is a wash, a sheet over the waves: following their faces broke it into clumps)
        // (caps belong on the crests of the long waves too: thinner down in a trough)
        const float hsF = max(0.5, (gSeaRelief - 0.75) / 0.5);
        const float crestF = mix(0.45, 1.0, smoothstep(-0.35 * hsF, 0.25 * hsF, P.y - (pm.sheenRoughness * 100.0 - 50.0)));
        const float wshape = mix(crestF * foamWaveShape(N, gFoamXZ, length(wvN) > 0.1 ? normalize(wvN) : vec2(1.0, 0.0)), 1.0, smoothstep(0.04, 0.15, gWake.g));
        col = foamLayer(col, P, N, V, pm.sheenRoughness, F, reflectColor, gWake.r * wshape * smoothstep(0.02, 0.15, gWake.r), gFoamXZ,
                        length(P - gPrimaryOrigin) * 0.001 / max(NdotV, 0.05), below, doShadows, maxLod, seed);
    }

#ifdef THREEPP_OCEAN_INSPECT
    if (pc.inspectMode != 0u) {
        float denom=max(dot(col,vec3(0.2126,0.7152,0.0722)),1e-8);
        gInspectReflShare=clamp(dot(gInspectReflection,vec3(0.2126,0.7152,0.0722))/denom,0.0,1.0);
        gInspectTransShare=clamp(dot(gInspectTransmission,vec3(0.2126,0.7152,0.0722))/denom,0.0,1.0);
        if(pc.inspectSolo==1u) col=gInspectReflection;
    }
#endif
#ifdef THREEPP_OCEAN_WATER_BODY_EXPLORATION
    if (waterDebug == 16u) return vec3(debugWaterCover);
#endif
    if (waterDebug == 8u) return 0.125 * col;
    return col;
#endif// THREEPP_OCEAN_UPSTREAM_WATER
}

// In-glass transit for the refraction path. Marches from the entry point along
// the refracted ray to find the EXIT INTERFACE (the next transmissive surface —
// normally the glass's own back face), while handling content embedded inside
// the volume on the way (TransmissionOrderTest middle column):
//   • blend surfaces composite src-over into (over, overT),
//   • cutout surfaces alpha-test (holes pass, solid texels BLOCK),
//   • opaque surfaces BLOCK — the caller stops the refraction there and lets
//     the behind-ray shade the blocker instead of mistaking it for an exit
//     face (which bent the ray with the blocker's normal and never shaded it).
// miss=true if the ray escapes the scene.
// `origin` is the entry point ON the glass surface, not pushed inside it. Real
// glassware walls are 1-2 mm, and the old start (pushed SHADOW_EPS = 1 mm in,
// then a 1 mm tMin) stepped over any wall thinner than ~2 mm along the ray: the
// "exit" became the candle or the far wall and Beer-Lambert ran over the whole
// interior. glTF GlassHurricaneCandleHolder (1.2 mm walls, attenuationDistance
// 1 mm) went deep blue above the bulge, near-black under the rim.
// gGlassExitHeadOn: the exit interface found was met on its FRONT face, so it is
// not this glass's way out but the surface of something nested inside it (a
// liquid in a vessel); the caller hands the ray on before it.
bool gGlassExitHeadOn = false;
void traceGlassInterior(vec3 origin, vec3 dir, float maxLod, bool doShadows, inout uint seed,
                        out vec3 hitP, out vec3 hitN, out float dist, out bool miss, out bool blocked,
                        inout vec3 over, inout float overT) {
    vec3  o         = origin;
    float travelled = 0.0;
    // A few float ULPs at the SHADOW_EPS world scale: enough not to re-hit the
    // triangle `o` sits on, far below any wall thickness.
    const float tMin = SHADOW_EPS * 0.05;
    miss = false; blocked = false;
    hitP = origin; hitN = vec3(0.0, 1.0, 0.0); dist = 0.0;
    for (int s = 0; s < 4; ++s) {
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, topAS, gl_RayFlagsOpaqueEXT, kRayMaskAll, o, tMin, dir, 1e30);
        while (rayQueryProceedEXT(rq)) {}
        if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
            miss = true; hitP = o; dist = travelled;
            return;
        }
        const int          hitId  = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
        const MaterialDesc hm     = entryMat(hitId);
        const int    primId = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
        const vec2   bary   = rayQueryGetIntersectionBarycentricsEXT(rq, true);
        const float  tHit   = rayQueryGetIntersectionTEXT(rq, true);
        const mat4x3 w2o    = rayQueryGetIntersectionWorldToObjectEXT(rq, true);
        vec2 uv;
        fetchHit(hitId, primId, bary, w2o, hitN, uv);
        hitP = o + dir * tHit;
        dist = travelled + tHit;

        if (hm.transmission > 0.0) {
            // The entry point is reconstructed from depth and can sit a hair in
            // FRONT of its triangle, so the first segment may meet the entry face
            // itself. Inside the glass the exit face is met from behind (its
            // outward normal along the ray); a face met head-on right at the start
            // is the entry face. Step onto it and go on.
            if (s == 0 && tHit < SHADOW_EPS && dot(hitN, dir) < 0.0) {
                o = hitP;
                travelled += tHit;
                continue;
            }
            gGlassExitHeadOn = rayQueryGetIntersectionFrontFaceEXT(rq, true);
            return;// exit interface (glass back face / nested glass)
        }

        if (hm.alphaCutoff > 0.0) {// cutout: holes pass, solid texels block
            if (hitTexAlpha(hm.albedoTexIndex, hm.uvTransform, uv) < hm.alphaCutoff) {
                o = o + dir * (tHit + 1e-3);
                travelled += tHit + 1e-3;
                continue;
            }
            blocked = true;
            return;
        }
        if (hm.alphaCutoff < 0.0) {// blend content inside the glass: composite
            const float a = hitTexAlpha(hm.albedoTexIndex, hm.uvTransform, uv);
            if (a > 0.01) {
                vec3 n = hitN;
                if (dot(n, -dir) < 0.0) n = -n;
                const vec3 alb     = hitTex(hm.albedoTexIndex, hm.uvTransform, uv, hm.albedo);
                const vec3 diffInd = sampleEnvDiffuse(n) + lights.ambient + hemiAmbient(n);
                over  += overT * a * shadeDiffuseDirect(hitP, n, -dir, alb,
                                                        clamp(hm.roughness, 0.04, 1.0), clamp(hm.metalness, 0.0, 1.0),
                                                        hm.emissive * hm.emissiveIntensity,
                                                        doShadows, diffInd,
                                                        hm.sheenColor, hm.sheenRoughness,
                                                        dielectricF0(hm.ior, hm.specularIntensity, hm.specularColor),
                                                        hm.iridescence, hm.iridescenceIOR, hm.iridescenceThicknessNm, seed,
                                                        /*addEmissive=*/true, /*cheapHit=*/true);
                overT *= (1.0 - a);
                if (overT < 0.02) {// effectively solid — nothing left to transmit
                    blocked = true;
                    return;
                }
            }
            o = o + dir * (tHit + 1e-3);
            travelled += tHit + 1e-3;
            continue;
        }
        blocked = true;// embedded opaque content
        return;
    }
    blocked = true;// layer budget exhausted — treat the last hit as the blocker
}

// Refractive glass (CLOSED transmissive mesh — sphere / bottle / goblet, NOT the
// thin-shell water handled above). Deterministic 2-interface refraction so it's
// noise-free for smooth glass:
//   Fresnel-mix( transmit , reflect )
// • reflect  = ray-traced sky + scene (mirror; the surface is near-smooth),
// • transmit = enter the front face → trace through to the back face → refract
//   out → continue to the SCENE BEHIND, all distortion-correct, then tint by the
//   glass colour + Beer-Lambert over the in-glass path length (ior>1.01 → tint =
//   albedo) but in 2 bounces
//   rather than full path continuation, so deep multi-bounce caustics inside
//   concave glass are approximated (fine for spheres; goblet stems lose a little).
// Tunable middleground "finish" for glass: a gentle frost FLOOR so glass is
// never razor-sharp ("too perfect"), and the sample count that CONVERGES it so
// it's a clean soft — not the milky 1-sample grain. Raise kGlassFrost for
// softer/dreamier glass, drop to 0 for a perfect lens; raise GSAMP if the soft
// shows residual grain on moving glass (costs proportionally).
const int   kGlassSamples = 1;// ONE sharp Fresnel reflect+refract sample → NO discrete-microfacet ghost
                              // copies. The frost (roughness blur) is applied afterwards by the reflection
                              // filter (deferred_refl_filter.comp), exactly like opaque reflections.


// matRough is the pixel's G-buffer roughness (factor x roughness map), not
// pm.roughness: glTF leaves roughnessFactor at 1 when a map carries it, and the
// factor alone frosted every such glass to the kGlassMaxRough cap, which the
// reflection filter then blurred (GlassHurricaneCandleHolder: a clear pane with
// frosted stripes read as one soft, milky shell).
// shadeGlass in three parts around its two traces, both of which main() makes
// at call sites it shares with other surfaces (each call site of traceRadiance
// is a whole inlined copy of it, and the copies are what the driver's compile
// time is made of): its reflection at the opaque surfaces' reflection call, and
// what is behind it at the additive and flat-alpha blends' call.
// shadeGlassReflect draws the facet and asks for the reflection ray (a
// TraceReq); shadeGlassBegin takes the reflected radiance and does everything
// up to the point where the glass knows whether it wants the trace behind it,
// and from where, in which direction and along the view or not (a second
// TraceReq); shadeGlassEnd takes what that trace returned and finishes.
// GlassCtx carries what each part needs of the one before. kGlassSamples is 1,
// and the split takes that one sample (the loop it ran in is gone).
struct GlassCtx {
    vec3  Ns;           // the facet the sample was drawn on
    vec3  R;            // the reflection ray, from reflOrig
    vec3  reflOrig;
    float missLod;
    float ior;
    vec3  reflectColor;
    vec3  F;
    vec3  transmitColor;// set here only on a total internal reflection at the front face
    float inDist;       // solid glass: the distance inside it
    vec3  overC;        // solid glass: blend content composited inside the volume
    float overT;
    float gr;
    float r0;
};
// A trace of what is behind a transmissive surface (deferred_shade.comp makes it):
// from orig along dir, the env miss at missLod, with gTraceAlongView set around it
// when `along` (the ray is the view ray, not bent). gSeenThroughHits is set for
// every such trace.
struct TraceReq {
    bool  want;
    vec3  orig;
    vec3  dir;
    float missLod;
    bool  along;
};
void shadeGlassReflect(vec3 P, vec3 N, vec3 V, MaterialDesc pm, float matRough, float maxLod,
                       inout uint seed, out GlassCtx gc, out TraceReq reflReq) {
    const float ior     = max(pm.ior, 1.0);
    // Explicit square — same negative-base pow() rationale as shadeWater above.
    const float f0n     = (1.0 - ior) / (1.0 + ior);
    const float r0      = f0n * f0n;
    // Microfacet roughness: the material's, floored to kGlassFrost and capped at
    // kGlassMaxRough (see the const). The slight softening is the
    // "less perfect" the user wanted AND it lets the multi-bounce surface
    // REFLECTIONS read through the refraction (a perfectly clear lens hides
    // them — "no recursive reflections"). The reflection itself is the
    // multi-bounce traceRadiance, so it's genuinely recursive (reflected
    // metal/glossy reflects on too).
    const float gr      = glassRough(matRough, ior);
    // Env MISSES (sky) read a slightly blurred mip so the HDR sun's single hot
    // texel doesn't speckle on curved glass (the env is plain equirect mips,
    // not a smooth PMREM). The old 0.30 floor smeared the whole sky into the
    // glass — with the lobe now capped near-mirror, a gentle floor keeps the
    // sky crisp and the sun reads as a real glint. Scene hits stay sharp.
    // Under RR the microfacet below is drawn per pixel and per frame, so the
    // blur is in the directions and the miss reads the floor's mip.
    const float missLod = max(kRR ? 0.0 : gr, 0.10) * maxLod;


    const int s = 0;// the one sample (kGlassSamples is 1)
    // Under DLSS Ray Reconstruction one random facet per pixel and frame (the
    // seed moves with the frame there): RR integrates the frosted lobe. Else
    // the one deterministic sample and the reflection filter's blur.
    // The facet is drawn at the material's own roughness (the ior ramp, no
    // kGlassFrost floor: that floor is for the filter and the glass marker),
    // and only for glass rougher than 0.05, the path tracer's threshold for a
    // thin sheet's scatter. Below it the mirror normal, as without RR: the
    // twin's Volvo windows (0.03) went soft at a distance and smeared under
    // RR, which reprojects what the glass shows as a reflection would be.
    const float lobeRough = matRough * clamp(ior * 2.0 - 2.0, 0.0, 1.0);
    vec3 Ns = (kRR && lobeRough > 0.05) ? ggxHalfVector(N, lobeRough, seed)
                                        : ggxHalfVectorFib(N, gr, s, kGlassSamples);
    if (dot(Ns, V) <= 0.0) Ns = N;// reject back-facing microfacets
    const float NdotV = max(dot(Ns, V), 1e-4);
    const vec3  F     = glassFresnel(r0, NdotV, pm.iridescence, pm.iridescenceIOR, pm.iridescenceThicknessNm);

    // Reflection (sky + scene) — multi-bounce / recursive.
    const vec3 R = reflect(-V, Ns);
    const vec3 reflOrig = P + N * SHADOW_EPS;
    gc.Ns       = Ns;
    gc.R        = R;
    gc.reflOrig = reflOrig;
    gc.missLod  = missLod;
    gc.ior      = ior;
    gc.F        = F;
    gc.gr       = gr;
    gc.r0       = r0;
    // Traced by main() with cheapHits false (glass shows hits SHARP:
    // deterministic shading), probeHitFill true and this surface's
    // envMapIntensity, from the seed.
    reflReq = TraceReq(true, reflOrig, R, missLod, false);
}

// reflected: what the reflection ray shadeGlassReflect asked for returned.
void shadeGlassBegin(vec3 P, vec3 N, vec3 V, MaterialDesc pm, vec3 albedo, bool doShadows, float maxLod,
                     inout uint seed, vec3 reflected, inout GlassCtx gc, out TraceReq req) {
    const vec3  Ns       = gc.Ns;
    const vec3  R        = gc.R;
    const vec3  reflOrig = gc.reflOrig;
    const float missLod  = gc.missLod;
    const float ior      = gc.ior;
    const vec3  F        = gc.F;
    const float gr       = gc.gr;
    // NOT const: the splat-volume leg below composites into it. Same value,
    // same expression — the write is behind a uniform flag that is 0 on
    // every splat-free scene.
    vec3 reflectColor = reflected;
    if (kRR) {
        gRRSpecF     = F;
        gRRSpecRough = gr;
        gRRReflHitT  = gTraceHitT;
    }
    // Splats in the mirror, the glass leg — the shadeWater block's twin, on
    // this path's own origin/direction and bounded by its own traced hit
    // distance. See that block for the argument; flags bit 12 is the same
    // single uniform gate, primary view only.
    if (flagOn(4096u)) {
        vec3 svEmis;
        const float svT = svLeg(reflOrig, R, gTraceHitT > 0.0 ? gTraceHitT : 1e30, svEmis);
        reflectColor = reflectColor * svT + svEmis;
    }

    vec3 transmitColor = vec3(0.0);
    // The ray to what is behind the glass, set by whichever of the two
    // branches below wants one (a thin sheet always; solid glass unless it
    // reflects totally at the front face) and traced once after them: each
    // call site of traceRadiance is a whole inlined copy of it, and the
    // copies are what the driver's compile time is made of.
    bool  behindWanted = false;
    vec3  behindOrig   = vec3(0.0);
    vec3  behindDir    = vec3(0.0);
    bool  behindAlong  = false;// thin sheet: the view ray, not bent
    float inDist       = 0.0;  // solid glass: the distance inside it
    vec3  overC        = vec3(0.0);
    float overT        = 1.0;
    if (pm.thinWalled != 0) {
        // THIN-SHELL glass (glTF default, e.g. a watch crystal): a thin parallel
        // shell refracts on ENTRY then back on EXIT → NET ~zero bend (a tiny
        // parallel offset). So pass the view ray ~STRAIGHT through and read the
        // surface directly behind — undistorted, full detail. A single refract()
        // (the old code) bends by the full first-interface angle → it shifts/
        // warps the face beneath and lets rays escape to the sky = a blue
        // "washed, detail-destroyed" look over e.g. a watch dial.
        // A ROUGH sheet scatters what it lets through (thinSheetTransmit): with
        // the facet sampled under RR the view through frosted glass is frosted.
        // Traced below, by the call this branch shares with the solid one:
        // straight through when smooth, what the pane shows lit as seen
        // beside it (gSeenThroughHits) and by the lamps of its own cluster
        // cell (gTraceAlongView).
        behindWanted = true;
        behindOrig   = P - N * SHADOW_EPS;
        behindDir    = thinSheetTransmit(V, N, Ns);
        behindAlong  = true;
    } else {
        // SOLID glass (closed mesh): full 2-interface refraction → the
        // inverted, magnified background a real glass ball shows. Beer-Lambert
        // over the ACTUAL in-glass distance → deeper tint through the centre.
        vec3 dIn = refract(-V, Ns, 1.0 / ior);
        if (dot(dIn, dIn) < 1e-6) {
            transmitColor = reflectColor;// TIR at the front face
        } else {
            dIn = normalize(dIn);
            vec3 exitP, exitN; bool exitMiss, blockedInside;
            gGlassExitHeadOn = false;
            traceGlassInterior(P, dIn, maxLod, doShadows, seed,
                               exitP, exitN, inDist, exitMiss, blockedInside, overC, overT);
            vec3 dOut;
            if (exitMiss) {
                dOut = dIn; inDist = 0.0;// open/odd geometry — straight through
            } else if (blockedInside || gGlassExitHeadOn) {
                // Opaque content inside the volume: stop the refraction at
                // it. Back the origin off so the behind-ray re-hits the
                // blocker and traceRadiance shades it (with its own
                // cutout/blend handling) — no exit bend through a surface
                // the ray never actually leaves.
                //
                // The same for a surface met HEAD-ON inside the glass: it
                // is not this glass's way out but the surface of something
                // inside it, a liquid in a vessel. It was bent as glass to
                // air with this glass's ior, where the ray crosses into the
                // liquid: at a slant that is a total internal reflection,
                // and a glass teapot's tea showed the reflected glass instead
                // of the tea. traceRadiance crosses it by its own rules.
                dOut  = dIn;
                exitP = exitP - dIn * (4.0 * SHADOW_EPS);
            } else {
                vec3 nb = exitN;
                if (dot(nb, dIn) < 0.0) nb = -nb;           // orient outward
                vec3 o = refract(dIn, -nb, ior);            // glass→air
                dOut = (dot(o, o) < 1e-6) ? reflect(dIn, nb)// TIR → internal reflect
                                          : normalize(o);
            }
            // Lit where it stands, as behind a thin pane (gSeenThroughHits);
            // not along the view (the ray is bent), so no cluster-cell lamps.
            behindWanted = true;
            behindOrig   = exitP + dOut * SHADOW_EPS;
            behindDir    = dOut;
        }
    }
    gc.reflectColor  = reflectColor;
    gc.transmitColor = transmitColor;
    gc.inDist        = inDist;
    gc.overC         = overC;
    gc.overT         = overT;
    req.want    = behindWanted;
    req.orig    = behindOrig;
    req.dir     = behindDir;
    req.missLod = missLod;
    req.along   = behindAlong;
}

// traced: main() made the trace shadeGlassBegin asked for (req.want), and
// `behind` is what it returned.
vec3 shadeGlassEnd(vec3 P, vec3 N, vec3 V, MaterialDesc pm, vec3 albedo, bool doShadows,
                   GlassCtx gc, bool traced, vec3 behind) {
    const float gr            = gc.gr;
    const float r0            = gc.r0;
    const vec3  F             = gc.F;
    const vec3  reflectColor  = gc.reflectColor;
    const float inDist        = gc.inDist;
    const vec3  overC         = gc.overC;
    const float overT         = gc.overT;
    vec3        transmitColor = gc.transmitColor;
    vec3 sum = vec3(0.0);
    if (traced) {
        vec3 tint = albedo;
        if (pm.thinWalled != 0) {
            if (pm.attenuationDistance > 0.0 && pm.thickness > 0.0)
                tint *= pow(max(pm.attenuationColor, vec3(1e-6)), vec3(pm.thickness / pm.attenuationDistance));
            transmitColor = behind * tint;
        } else {
            if (pm.attenuationDistance > 0.0)
                tint *= pow(max(pm.attenuationColor, vec3(1e-6)), vec3(inDist / pm.attenuationDistance));
            // Blend content composited inside the volume layers over the
            // (remaining) behind term; the glass tint applies to both —
            // an embedded symbol is seen through the front surface, so it
            // should physically pick up the glass colour.
            transmitColor = (overC + overT * behind) * tint;
        }
    }
    // Firefly-clamp the glass env contribution — the HDR sun reflected/
    // refracted through curved glass otherwise scatters into bright STATIC
    // speckle. 6.0 (was 2.5): with the lobe capped near-mirror the scatter
    // is coherent, and the old cap flattened every highlight — glass needs
    // its bright glints or it reads dull.
    vec3 contrib = mix(transmitColor, reflectColor, F);
    const float cl = max(max(contrib.r, contrib.g), contrib.b);
    if (cl > 6.0) contrib *= 6.0 / cl;
    sum += contrib;

    // Direct-light glint. The recombine scales the opaque base by
    // (1 − transmission), which erases its analytic specular entirely for
    // transmission=1 glass — so glass under a sun or a lamp had NO surface
    // glint at all, a big part of the "dull" look. Spec-only Cook-Torrance at
    // the glass lobe + Fresnel from the ior (no diffuse — the transmit path
    // carries that energy). Deterministic, so it's added sharp on top of the
    // (possibly denoise-blurred) reflect/refract channel, like a real
    // highlight rides on frosted glass.
    vec3 glint = vec3(0.0);
    {
        const float NdotV = max(dot(N, V), 1e-4);
        const float kG    = (gr + 1.0) * (gr + 1.0) / 8.0;
        for (uint i = 0u; i < lights.dirCount; ++i) {
            vec3        L     = normalize(lights.dirLights[i].direction);
            const vec3  leg   = murkSunLeg(P, L);// submerged glass: refracted, attenuated
            const float NdotL = max(dot(N, L), 0.0);
            if (NdotL <= 0.0) continue;
            const vec3 vis = doShadows ? primarySunVis(i, P, N, P + N * SHADOW_EPS, L, false) * leg : 1.0 * leg;
            if (max(max(vis.r, vis.g), vis.b) <= 0.0) continue;
            const vec3  H = normalize(V + L);
            const float D = distGGX(max(dot(N, H), 0.0), gr);
            const float G = geomSmithG1(NdotV, kG) * geomSmithG1(NdotL, kG);
            const vec3  F = glassFresnel(r0, max(dot(V, H), 0.0), pm.iridescence, pm.iridescenceIOR, pm.iridescenceThicknessNm);
            // spec·NdotL = D·G·F / (4·NdotV·NdotL) · NdotL
            glint += lights.dirLights[i].color * (D * G * F / max(4.0 * NdotV, 1e-4)) * vis;
        }
        // The point and spot lamps' glints, the same lobe. Glass had the sun's
        // alone: a street lamp or a car's head lamps in a shop window or a
        // windscreen at night left no highlight at all, where the path tracer
        // (dielectricGlints) and a photo show a bright one. The lamps are this
        // pixel's cluster cell's (its own depth, as the water's glints read
        // them); a parallel projection has no cells along a ray and goes
        // without. A lamp's shadow ray is cast only for a glint worth one.
        if (pc.clusterLightCount > 0u && cam.camAux.x < 0.5) {
            const uint cellBase = clusterCellBase(ivec2(gl_GlobalInvocationID.xy),
                                                  max(dot(P - gPrimaryOrigin, cam.camAux.yzw), 0.0));
            const uint cCnt     = min(clusterGrid[cellBase], kClusterMaxPerCell);
            for (uint ci = 0u; ci < cCnt; ++ci) {
                const uint li = min(clusterGrid[cellBase + 1u + ci], pc.clusterLightCount - 1u);
                vec3 toL = clusterLights[li].position - P;
                const float dist = length(toL);
                if (dist < 1e-4) continue;
                toL /= dist;
                const float NdotL = dot(N, toL);
                if (NdotL <= 0.0) continue;
                float atten;
                if (clusterLights[li].type > 0.5) {
                    const float spotAtten = smoothstep(clusterLights[li].cosAngleOuter, clusterLights[li].cosAngleInner,
                                                       dot(-toL, clusterLights[li].direction));
                    if (spotAtten <= 0.0) continue;
                    atten = spotAtten / max(pow(dist, clusterLights[li].decay), 0.01);
                } else {
                    atten = 1.0 / max(distFalloff(dist, clusterLights[li].decay), 0.01);
                }
                const float range = clusterLights[li].range;
                if (range > 0.0) {
                    const float tt  = dist / range;
                    const float wnd = max(1.0 - tt * tt * tt * tt, 0.0);
                    atten *= wnd * wnd;
                }
                const vec3  H = normalize(V + toL);
                const float D = distGGX(max(dot(N, H), 0.0), gr);
                const float G = geomSmithG1(NdotV, kG) * geomSmithG1(NdotL, kG);
                const vec3  F = glassFresnel(r0, max(dot(V, H), 0.0), pm.iridescence, pm.iridescenceIOR, pm.iridescenceThicknessNm);
                vec3 c = clusterLights[li].color * (atten * D * G / max(4.0 * NdotV, 1e-4)) * F;
                if (max(max(c.r, c.g), c.b) < 1e-4) continue;
                if (doShadows) {
                    const float smV = kSM ? smLampVis(clusterLights[li].shadowSlot, clusterLights[li].type > 0.5, P, N,
                                                      clusterLights[li].position, vec2(gl_GlobalInvocationID.xy))
                                          : -1.0;
                    c *= smV >= 0.0 ? smV : shadowVis(P + N * SHADOW_EPS, toL, dist - 1e-2);
                }
                const float cl   = max(max(c.r, c.g), c.b);
                const float gMax = pc.fireflyClamp * 4.0;
                if (gMax > 0.0 && cl > gMax) c *= gMax / cl;
                glint += c;
            }
        }
    }
    return sum / float(kGlassSamples) + glint;
}
