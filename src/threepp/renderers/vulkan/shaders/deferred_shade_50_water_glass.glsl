// Split from deferred_shade.comp: thin-shell water shading (RasterFirst),
// in-glass interior transit marching, and refractive glass shading.

// Term isolation for water-artefact triage (-DSCOUT_WATER=N):
//   1 = reflection only, 2 = transmission only, 3 = Fresnel F as grey,
//   4 = foam mask as grey, 5 = |N - up| amplified. 0 = normal shading.
#ifndef SCOUT_WATER
#define SCOUT_WATER 0
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
vec3 shadeWater(vec3 P, vec3 N, vec3 V, MaterialDesc pm, int instIdx,
                bool doShadows, float maxLod, uint frame, inout uint seed,
                float slopeVarSq, vec3 Nmacro) {
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
    gTraceSkipWater = true;
    const vec3 reflOrig = P + Nmacro * SHADOW_EPS;// this water's own reflection: pass through crests
    vec3 reflectColor = traceRadiance(reflOrig, R, doShadows, maxLod,
                                      reflLod, seed, /*cheapHits=*/true,// water: blur+temporal absorb
                                      /*probeHitFill=*/true,
                                      /*envInt=*/pm.envMapIntensity);
    gTraceSkipWater = false;
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
    if ((pc.flags & 2048u) != 0u && pd.counts.y != 0u) {
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
    if ((pc.flags & 4096u) != 0u) {
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
    } else if (murkLive() && fog.murkDensity >= kMurkIsWater) {
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
        vec3 dRef = refract(-V, N, 1.0 / ior);
        // From the air side a transmitted ray always exists; the guard only
        // covers a degenerate normal.
        if (dot(dRef, dRef) < 1e-6) dRef = -N;
        dRef = normalize(dRef);
        transmitColor = applyMurkSky(dRef);
        // Beyond ~6 optical depths the hit term is < ~2% — invisible. The same
        // budget the material path uses, on the medium that is actually there.
        const float maxVis   = min(6.0 / fog.murkDensity, 1.0e4);
        const vec3  uwOrigin = P - N * SHADOW_EPS;
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, topAS, gl_RayFlagsOpaqueEXT, kRayMaskAll,
                              uwOrigin, 1e-3, dRef, maxVis);
        while (rayQueryProceedEXT(rq)) {}
        if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
            const int   hitId = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
            const float dBot  = rayQueryGetIntersectionTEXT(rq, true);
            if (geoms[hitId].foamAddress == 0ul) {// not a wave-crest self-hit
                // doShadows=false for the reason the material path gives:
                // water is opaque-masked for occlusion queries, so a shadow ray
                // from the hit would read the surface overhead as a full
                // occluder. murkSunCaustic on the hit's direct light is what
                // the from-below primary gets too.
                const vec3 bottom = traceRadiance(uwOrigin, dRef, /*doShadows=*/false,
                                                  maxLod, 0.12 * maxLod, seed,
                                                  /*cheapHits=*/false,
                                                  /*probeHitFill=*/false,
                                                  /*envInt=*/1.0);// transmitted, not IBL
                transmitColor = applyMurk(bottom, uwOrigin, uwOrigin + dRef * dBot);
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
    const float skyLum    = dot(sampleEnvLod(vec3(0.0, 1.0, 0.0), maxLod), vec3(0.2126, 0.7152, 0.0722));
    const float depthFade = mix(1.0, 0.4, clamp(V.y, 0.0, 1.0));// darker looking down into the column
    const vec3  deepBody  = tint * (skyLum * 0.35 + lights.ambient) * depthFade;
    transmitColor = deepBody;

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
        const float maxVis = 6.0 * pm.attenuationDistance;
        vec3 dRef = refract(-V, N, 1.0 / ior);
        if (dot(dRef, dRef) > 1e-6) {
            dRef = normalize(dRef);
            const vec3 uwOrigin = P - N * SHADOW_EPS;
            rayQueryEXT rq;
            rayQueryInitializeEXT(rq, topAS, gl_RayFlagsOpaqueEXT, kRayMaskAll,
                                  uwOrigin, 1e-3, dRef, maxVis);
            while (rayQueryProceedEXT(rq)) {}
            if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
                const int   hitId = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
                const float dBot  = rayQueryGetIntersectionTEXT(rq, true);
                if (geoms[hitId].foamAddress == 0ul) {// not a wave-crest self-hit
                    const vec3 bottom = traceRadiance(uwOrigin, dRef, /*doShadows=*/false,
                                                      maxLod, 0.12 * maxLod, seed,
                                                      /*cheapHits=*/false,// the floor IS the subject — keep it sharp
                                                      /*probeHitFill=*/false,// probes can't resolve the underwater cavity
                                                      /*envInt=*/1.0);// transmitted, not IBL
                    // View path + approximate sun path (vertical depth) through
                    // the column, Beer-Lambert per channel.
                    const float sunPath = dBot * (1.0 + clamp(-dRef.y, 0.0, 1.0));
                    const vec3  Tbot    = pow(max(pm.attenuationColor, vec3(1e-6)),
                                              vec3(sunPath / pm.attenuationDistance));
                    // The body veil builds up over the water column: SCALAR
                    // exponential saturation with a half-attenuation-length
                    // constant. Deep water (d ≥ ~2 lengths) keeps ≥96% of the
                    // calibrated deepBody — visually the old look — while a
                    // metre-deep pond sheds most of the veil so its floor
                    // reads instead of drowning in deep-ocean teal. Scalar on
                    // purpose: a per-channel weight would hue-shift the body.
                    const float bodyW = 1.0 - exp(-dBot / (0.5 * pm.attenuationDistance));
                    transmitColor = deepBody * bodyW + bottom * Tbot;
                }
            }
        }
    }
    }

#if SCOUT_WATER == 6
    // Reflection hit classification: red = SELF-HIT on water (< 60 m),
    // green = other geometry, blue = escaped to sky.
    if (gTraceHitT < 0.0)      return vec3(0.0, 0.0, 1.0);
    if (gTraceHitT < 60.0)     return vec3(1.0, 0.0, 0.0);
    return vec3(0.0, 1.0, 0.0);
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
            const float vis = doShadows ? shadowVis(sunOrig, L, 1e30) : 1.0;
            if (vis <= 0.0) continue;
            const vec3  H   = normalize(V + L);
            const float ndh = max(dot(N, H), 0.0);
            const float vdh = max(dot(V, H), 0.0);
            const float Gs  = geomSmithG1(NdotV, kG) * geomSmithG1(ndl, kG);
            const float Fb  = r0 + (1.0 - r0) * pow(1.0 - vdh, 5.0);
            float spec = distGGX(ndh, specRough) * Gs * Fb / max(4.0 * NdotV * ndl, 1e-4);
            if (pm.clearcoat > 0.0) {
                const float Fc = 0.04 + 0.96 * pow(1.0 - vdh, 5.0);
                spec += pm.clearcoat * distGGX(ndh, ccRough) * Gs * Fc / max(4.0 * NdotV * ndl, 1e-4);
            }
            vec3 c = spec * ndl * lights.dirLights[i].color * (vis * cloudShadowSample(P));
            // Directional lights are deltas → GGX spikes; clamp so a glint blooms
            // bright but never fireflies. (×4 over the global clamp → bright suns.)
            const float cl = max(max(c.r, c.g), c.b);
            const float gMax = pc.fireflyClamp * 4.0;
            if (cl > gMax) c *= gMax / cl;
            col += c;
        }
        // SPOT lights — same GGX glints with cone + range + inverse-square
        // attenuation, so a sweeping beam (lighthouse) lights the waves it
        // crosses instead of existing only as the in-scattered air volume.
        for (uint i = 0u; i < lights.spotCount; ++i) {
            vec3 toL = lights.spotLights[i].position - P;
            const float dist = length(toL);
            if (dist < 1e-4) continue;
            toL /= dist;
            const float ndl = dot(N, toL);
            if (ndl <= 0.0) continue;
            const float spotCos   = dot(-toL, lights.spotLights[i].direction);
            const float spotAtten = smoothstep(lights.spotLights[i].cosAngleOuter,
                                               lights.spotLights[i].cosAngleInner, spotCos);
            if (spotAtten <= 0.0) continue;
            float atten = spotAtten / max(pow(dist, lights.spotLights[i].decay), 0.01);
            const float range = lights.spotLights[i].range;
            if (range > 0.0) {
                const float tt = dist / range;
                const float t4 = tt * tt * tt * tt;
                const float wnd = max(1.0 - t4, 0.0);
                atten *= wnd * wnd;
            }
            if (atten <= 1e-6) continue;
            const float vis = doShadows ? shadowVis(sunOrig, toL, dist - 1e-2) : 1.0;
            if (vis <= 0.0) continue;
            const vec3  H   = normalize(V + toL);
            const float ndh = max(dot(N, H), 0.0);
            const float vdh = max(dot(V, H), 0.0);
            const float Gs  = geomSmithG1(NdotV, kG) * geomSmithG1(ndl, kG);
            const float Fb  = r0 + (1.0 - r0) * pow(1.0 - vdh, 5.0);
            float spec = distGGX(ndh, specRough) * Gs * Fb / max(4.0 * NdotV * ndl, 1e-4);
            if (pm.clearcoat > 0.0) {
                const float Fc = 0.04 + 0.96 * pow(1.0 - vdh, 5.0);
                spec += pm.clearcoat * distGGX(ndh, ccRough) * Gs * Fc / max(4.0 * NdotV * ndl, 1e-4);
            }
            // Beam POOL — a broad faint sheen where the beam strikes the
            // surface. Real water returns part of a grazing beam diffusely
            // (micro-facets beyond the GGX tail, spray, top-layer scatter);
            // at roughness 0.04 the pure specular lobe is so tight the lit
            // pool would otherwise be invisible unless perfectly aligned.
            spec += 0.035 / PI;
            vec3 c = spec * ndl * lights.spotLights[i].color * atten * vis;
            const float cl = max(max(c.r, c.g), c.b);
            const float gMax = pc.fireflyClamp * 4.0;
            if (cl > gMax) c *= gMax / cl;
            col += c;
        }
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
                const vec3  foamDiff = sampleEnvLod(N, maxLod) + lights.ambient + hemiAmbient(N);
                const vec3  foamLit  = shadeDiffuseDirect(P, N, V, foamCol, foamRough, 0.0,
                                                          vec3(0.0), doShadows, foamDiff,
                                                          vec3(0.0), 0.0,
                                                          1.0, vec3(1.0),
                                                          0.0, 1.3, 0.0, seed,
                                                          /*addEmissive=*/true,// foam: no iridescence
                                                          /*cheapHit=*/false);
                col = mix(col, foamLit, foamMask);
            }
        }
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
void traceGlassInterior(vec3 origin, vec3 dir, float maxLod, bool doShadows, inout uint seed,
                        out vec3 hitP, out vec3 hitN, out float dist, out bool miss, out bool blocked,
                        inout vec3 over, inout float overT) {
    vec3  o         = origin;
    float travelled = 0.0;
    miss = false; blocked = false;
    hitP = origin; hitN = vec3(0.0, 1.0, 0.0); dist = 0.0;
    for (int s = 0; s < 4; ++s) {
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, topAS, gl_RayFlagsOpaqueEXT, kRayMaskAll, o, 1e-3, dir, 1e30);
        while (rayQueryProceedEXT(rq)) {}
        if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
            miss = true; hitP = o; dist = travelled;
            return;
        }
        const int          hitId  = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
        const MaterialDesc hm     = mats[hitId];
        const int    primId = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
        const vec2   bary   = rayQueryGetIntersectionBarycentricsEXT(rq, true);
        const float  tHit   = rayQueryGetIntersectionTEXT(rq, true);
        const mat4x3 w2o    = rayQueryGetIntersectionWorldToObjectEXT(rq, true);
        vec2 uv;
        fetchHit(hitId, primId, bary, w2o, hitN, uv);
        hitP = o + dir * tHit;
        dist = travelled + tHit;

        if (hm.transmission > 0.0) return;// exit interface (glass back face / nested glass)

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
                const vec3 diffInd = sampleEnvLod(n, maxLod) + lights.ambient + hemiAmbient(n);
                over  += overT * a * shadeDiffuseDirect(hitP, n, -dir, alb,
                                                        clamp(hm.roughness, 0.04, 1.0), clamp(hm.metalness, 0.0, 1.0),
                                                        hm.emissive * hm.emissiveIntensity,
                                                        doShadows, diffInd,
                                                        hm.sheenColor, hm.sheenRoughness,
                                                        hm.specularIntensity, hm.specularColor,
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
const float kGlassFrost = 0.04;// near-sharp glass: a wide frost lobe scatters the HDR sun into static speckle
const float kGlassMaxRough = 0.14;// Clamps the primary refraction lobe to
                              // alpha 0.02 (≈ roughness 0.14) because a wide refraction lobe is
                              // per-pixel variance no denoiser can fix. Deferred used the raw material
                              // roughness here, so asset glass with a mid-grey roughness map (e.g. the
                              // OpenChessSet pawn domes) blurred into a milky ball — structureless,
                              // duller than intended. The trade: frosted glass
                              // renders polished at primary, but it stays GLASS.
const int   kGlassSamples = 1;// ONE sharp Fresnel reflect+refract sample → NO discrete-microfacet ghost
                              // copies. The frost (roughness blur) is applied afterwards by the reflection
                              // filter (deferred_refl_filter.comp), exactly like opaque reflections.

// Effective glass shading/blur roughness — used by shadeGlass AND the demod
// recombine below; both must agree or the denoise blur mismatches the lobe.
float glassRough(float matRoughness) {
    return clamp(matRoughness, kGlassFrost, kGlassMaxRough);
}

vec3 shadeGlass(vec3 P, vec3 N, vec3 V, MaterialDesc pm, vec3 albedo,
                bool doShadows, float maxLod, inout uint seed) {
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
    const float gr      = glassRough(pm.roughness);
    // Env MISSES (sky) read a slightly blurred mip so the HDR sun's single hot
    // texel doesn't speckle on curved glass (the env is plain equirect mips,
    // not a smooth PMREM). The old 0.30 floor smeared the whole sky into the
    // glass — with the lobe now capped near-mirror, a gentle floor keeps the
    // sky crisp and the sun reads as a real glint. Scene hits stay sharp.
    const float missLod = max(gr, 0.10) * maxLod;

    // Sky visibility at the glass surface, consumed by the transmit legs'
    // probeHitFill=false env hit fill (see gEnvFillVis). Without it the
    // enclosed-interior scene showed a sky-lit world through the glass while
    // probe GI darkened everything around it. The reflect leg passes
    // probeHitFill=true and is unaffected.
    gEnvFillVis = probeEnvFillVis(P, N, maxLod);

    vec3 sum = vec3(0.0);
    for (int s = 0; s < kGlassSamples; ++s) {
        vec3 Ns = ggxHalfVectorFib(N, gr, s, kGlassSamples);// distinct per-iteration Fibonacci sample
        if (dot(Ns, V) <= 0.0) Ns = N;// reject back-facing microfacets
        const float NdotV = max(dot(Ns, V), 1e-4);
        const float F     = r0 + (1.0 - r0) * pow(1.0 - NdotV, 5.0);

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
        // Splats in the mirror, the glass leg — the shadeWater block's twin, on
        // this path's own origin/direction and bounded by its own traced hit
        // distance. See that block for the argument; flags bit 12 is the same
        // single uniform gate, primary view only.
        if ((pc.flags & 4096u) != 0u) {
            vec3 svEmis;
            const float svT = svLeg(reflOrig, R, gTraceHitT > 0.0 ? gTraceHitT : 1e30, svEmis);
            reflectColor = reflectColor * svT + svEmis;
        }

        vec3 transmitColor;
        if (pm.thinWalled != 0) {
            // THIN-SHELL glass (glTF default, e.g. a watch crystal): a thin parallel
            // shell refracts on ENTRY then back on EXIT → NET ~zero bend (a tiny
            // parallel offset). So pass the view ray ~STRAIGHT through and read the
            // surface directly behind — undistorted, full detail. A single refract()
            // (the old code) bends by the full first-interface angle → it shifts/
            // warps the face beneath and lets rays escape to the sky = a blue
            // "washed, detail-destroyed" look over e.g. a watch dial.
            const vec3 dT = -V;// straight-through (net-zero thin-shell bend)
            {
                const vec3 behind = traceRadiance(P - N * SHADOW_EPS, dT, doShadows, maxLod, missLod, seed,
                                                  /*cheapHits=*/false,// refracted content is sharp
                                                  /*probeHitFill=*/false,// content under glass: probes can't resolve the cavity
                                                  /*envInt=*/1.0);// transmitted, not IBL
                vec3 tint = albedo;
                if (pm.attenuationDistance > 0.0 && pm.thickness > 0.0)
                    tint *= pow(max(pm.attenuationColor, vec3(1e-6)), vec3(pm.thickness / pm.attenuationDistance));
                transmitColor = behind * tint;
            }
        } else {
            // SOLID glass (closed mesh): full 2-interface refraction → the
            // inverted, magnified background a real glass ball shows. Beer-Lambert
            // over the ACTUAL in-glass distance → deeper tint through the centre.
            vec3 dIn = refract(-V, Ns, 1.0 / ior);
            if (dot(dIn, dIn) < 1e-6) {
                transmitColor = reflectColor;// TIR at the front face
            } else {
                dIn = normalize(dIn);
                vec3 exitP, exitN; float inDist; bool exitMiss, blockedInside;
                vec3  overC = vec3(0.0);
                float overT = 1.0;
                traceGlassInterior(P - N * SHADOW_EPS, dIn, maxLod, doShadows, seed,
                                   exitP, exitN, inDist, exitMiss, blockedInside, overC, overT);
                vec3 dOut;
                if (exitMiss) {
                    dOut = dIn; inDist = 0.0;// open/odd geometry — straight through
                } else if (blockedInside) {
                    // Opaque content inside the volume: stop the refraction at
                    // it. Back the origin off so the behind-ray re-hits the
                    // blocker and traceRadiance shades it (with its own
                    // cutout/blend handling) — no exit bend through a surface
                    // the ray never actually leaves.
                    dOut  = dIn;
                    exitP = exitP - dIn * (4.0 * SHADOW_EPS);
                } else {
                    vec3 nb = exitN;
                    if (dot(nb, dIn) < 0.0) nb = -nb;           // orient outward
                    vec3 o = refract(dIn, -nb, ior);            // glass→air
                    dOut = (dot(o, o) < 1e-6) ? reflect(dIn, nb)// TIR → internal reflect
                                              : normalize(o);
                }
                const vec3 behind = traceRadiance(exitP + dOut * SHADOW_EPS, dOut, doShadows, maxLod, missLod, seed,
                                                  /*cheapHits=*/false,// refracted content is sharp
                                                  /*probeHitFill=*/false,// content under glass: probes can't resolve the cavity
                                                  /*envInt=*/1.0);// transmitted, not IBL
                vec3 tint = albedo;
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
    }

    // Direct-light glint. The recombine scales the opaque base by
    // (1 − transmission), which erases its analytic specular entirely for
    // transmission=1 glass — so glass under a sun/dir light had NO surface
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
            const float leg   = murkSunLeg(P, L);// submerged glass: refracted, attenuated
            const float NdotL = max(dot(N, L), 0.0);
            if (NdotL <= 0.0) continue;
            const float vis = doShadows ? shadowVis(P + N * SHADOW_EPS, L, 1e30) * leg : 1.0 * leg;
            if (vis <= 0.0) continue;
            const vec3  H = normalize(V + L);
            const float D = distGGX(max(dot(N, H), 0.0), gr);
            const float G = geomSmithG1(NdotV, kG) * geomSmithG1(NdotL, kG);
            const float F = r0 + (1.0 - r0) * pow(1.0 - max(dot(V, H), 0.0), 5.0);
            // spec·NdotL = D·G·F / (4·NdotV·NdotL) · NdotL
            glint += lights.dirLights[i].color * (D * G * F / max(4.0 * NdotV, 1e-4)) * vis;
        }
    }
    return sum / float(kGlassSamples) + glint;
}
