// Split from deferred_shade.comp: thin-shell water shading (RasterFirst),
// in-glass interior transit marching, and refractive glass shading.

// Term isolation for water-artefact triage (-DSCOUT_WATER=N):
//   1 = reflection only, 2 = transmission only, 3 = Fresnel F as grey,
//   4 = foam mask as grey, 5 = |N - up| amplified, 6 = the reflection
//   hit's distance class, 8 = how the reflection ray ended (see there).
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
// The Henyey-Greenstein phase the fog/particle/cloud volumetrics all use — the
// water column gets the SAME phase function, not a second one (hg_phase.glsl,
// included by deferred_shade_60).
float hgPhase(float mu, float g);

// ── Water-body IN-SCATTER calibration ───────────────────────────────────────
// kSky ties the in-scatter's skylight source to the SAME scale the analytic
// deep body already uses (`skyLum * 0.35` below), so the two terms are one
// calibration, not two. A fully scattering column (single-scattering albedo
// -> 1, leg >> 1/sigma_t) then in-scatters exactly the skylight the old body
// showed at tint == 1 — it reproduces that brightness rather than adding a
// second, independently scaled copy of it.
const float kWaterInScatterSky = 0.35;
// Forward-scattering water: suspended matter throws light along the ray far
// more than back at it. 0.75 is the fog/cloud family's mid-forward value.
const float kWaterInScatterG = 0.75;
// The HG lobe peaks at 28x isotropic for g = 0.75 (mu -> 1), i.e. looking down
// the sun's own refracted direction. Single scattering has no multiple-bounce
// budget to spend that on, so the spike is capped — the same reasoning as the
// glint loop's fireflyClamp, at the one place it can bite.
const float kWaterInScatterPhaseMax = 8.0;

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

vec3 shadeWater(vec3 P, vec3 N, vec3 V, MaterialDesc pm, int instIdx,
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
        F = r0 + (1.0 - r0) * pow(1.0 - NdotV, 5.0);
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
            //     takes no phase factor. Scaled by kWaterInScatterSky against the
            //     SAME skyLum the deep body uses (see the constant's note).
            //   • sun: a delta light, so it gets the HG phase between the refracted
            //     view leg and the refracted sun leg. Shadowed exactly as the
            //     surface is — a cliff shadow on the water must darken the glow
            //     under it, or the column lights up inside the shadow.
            vec3 Ein = vec3(skyLum * kWaterInScatterSky) + lights.ambient;
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
                                        kWaterInScatterPhaseMax);// 1 == isotropic
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
        if ((kOceanClimate & 2u) != 0u) {
            // The wave climate: the texture keeps the foam this water would
            // have with cascades 0 and 1 at 1, 2/3 and 1/3 of the sea
            // (foam_crest.comp, OCEAN_CLIMATE_LEVELS); take it at the map's
            // level here, cascade 1's (the waves the fold foam counts are
            // mostly its), linearly between levels and to none at 0.
            const vec4  lv = textureGrad(oceanCrestFoam, q * invTile, gA * invTile, gX * invTile);
            const float L  = clamp(oceanClimate(oceanClimateMap, q).y, 0.0, 1.0) * 3.0;
            foamF = L >= 2.0 ? mix(lv.b, lv.r, L - 2.0) : L >= 1.0 ? mix(lv.a, lv.b, L - 1.0) : lv.a * L;
        } else {
            foamF = textureGrad(oceanCrestFoam, q * invTile, gA * invTile, gX * invTile).r;
        }
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
            const float layer  = mix(0.35, 0.75, clamp(foamF, 0.0, 1.0));
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
// shadeGlass in two halves around the trace of what is behind the glass, which
// main() makes: the call it shares with the additive and flat-alpha blends (one
// call site of traceRadiance for all three; each call site is a whole inlined
// copy of it, and the copies are what the driver's compile time is made of).
// shadeGlassBegin does everything up to the point where the glass knows whether
// it wants that trace, and from where, in which direction and along the view or
// not (the TraceReq); shadeGlassEnd takes what the trace returned and finishes.
// GlassCtx carries what the second half needs of the first. kGlassSamples is 1,
// and the split takes that one sample (the loop it ran in is gone).
struct GlassCtx {
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
void shadeGlassBegin(vec3 P, vec3 N, vec3 V, MaterialDesc pm, vec3 albedo, float matRough,
                     bool doShadows, float maxLod, inout uint seed, out GlassCtx gc, out TraceReq req) {
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
    // NOT const: the splat-volume leg below composites into it. Same value,
    // same expression — the write is behind a uniform flag that is 0 on
    // every splat-free scene.
    vec3 reflectColor = traceRadiance(reflOrig, R, doShadows, maxLod, missLod, seed,
                                      /*cheapHits=*/false,// glass shows hits SHARP — deterministic shading
                                      /*probeHitFill=*/true,
                                      /*envInt=*/pm.envMapIntensity);
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
    gc.F             = F;
    gc.transmitColor = transmitColor;
    gc.inDist        = inDist;
    gc.overC         = overC;
    gc.overT         = overT;
    gc.gr            = gr;
    gc.r0            = r0;
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
