// Split from deferred_shade.comp: volumetric spot-light beam scattering,
// directional-light volumetric scattering (god rays), Henyey-Greenstein
// phase, scene fog (Beer-Lambert), sky aerial perspective, procedural star
// field, and the fogged sky/background colour helper.

// ── Volumetric SPOT-light beams (single scattering, ray-marched) ─────────────
// In-scattered radiance along [ro, ro+rd·tMax] from every spot light — the
// lighthouse-beam / searchlight look. The march window is the ray∩sphere
// (light, range) segment: any in-cone point lies within `range` of the light,
// and a ray-sphere clip is unconditionally robust where exact cone-interval
// logic (two nappes, axis slab) is fiddly; the per-step cone smoothstep then
// carves the actual beam shape. ONE jittered march per pixel per frame (PCG,
// frame-keyed) → TAA averages the stratification to smooth. No shadow rays
// through the medium — open-air beams are rarely occluded and this keeps the
// pass pure ALU. The base surface is NOT extinction-attenuated (σ is haze-thin;
// attenuating would wash the whole scene for a localized effect) — the term is
// purely additive in-scatter. Skipped entirely when pc.volDensity == 0.
float hgPhase(float mu, float g) {
    const float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * mu, 1e-4), 1.5));
}

// ── Scene fog (deferred) ─────────────────────────────────────────────────────
// Beer-Lambert extinction toward the fog colour over the PRIMARY distance —
// an approximation of true volumetric scattering (a full path-traced solution
// would also fog reflected/refracted sub-paths; here only the camera→surface leg is
// fogged). Sky pixels stay unfogged (three.js background semantics). The
// fogged path is clipped to y < waterSurfaceY, so the underwater medium ends
// exactly at the wave surface.
float fogPathLength(vec3 a, vec3 b) {
    const float full = distance(a, b);
    if (fog.waterSurfaceY > 1e29) return full;
    const float ya = a.y - fog.waterSurfaceY;
    const float yb = b.y - fog.waterSurfaceY;
    if (ya >= 0.0 && yb >= 0.0) return 0.0;       // segment entirely above the medium
    if (ya < 0.0 && yb < 0.0) return full;        // entirely inside
    const float t = ya / (ya - yb);               // surface crossing
    return (ya < 0.0) ? full * t : full * (1.0 - t);
}
// ── Heterogeneous height-fog surface extinction (setHeightFog) ───────────────
// Closed-form exponential-height-fog optical depth along [a,b] (ignores the
// noise modulation — a smooth mean is exactly what the FAR remainder wants).
float heightFogOpticalDepth(vec3 a, vec3 b) {
    if (clouds.hfDensity <= 0.0) return 0.0;
    const float H   = max(clouds.hfFalloff, 1e-3);
    const float ya  = max(a.y - clouds.hfBaseY, 0.0);
    const float yb  = max(b.y - clouds.hfBaseY, 0.0);
    // Clamp the leg so a sentinel / near-infinite end point can NEVER overflow.
    // compositeClouds fogs the cloud in-scatter over camP→(camP+dir·meanDist); on
    // a clear-SKY pixel meanDist falls back to sceneDist = 1e30, and distance()
    // SQUARES the components: (1e30)² = 1e60 ≫ fp32 max (3.4e38) → Inf. That Inf
    // then poisons the product below — Inf·f = Inf, or Inf·0 = NaN when f underflows
    // for a grazing/long leg (camera high above a shallow layer, ya/H ≳ 87 ⇒ ea→0)
    // — and exp(-NaN) = NaN blacks out the whole sky. 1e7 m dwarfs any real scene
    // leg; beyond it e^{-od} is already 0, so the clamp is invisible when legit.
    const float len = min(distance(a, b), 1.0e7);
    // ∫ σ0 e^{-max(y,base)/H} ds along the segment (y linear in s):
    //   σ0·len·(e^{-ya/H} − e^{-yb/H})/((yb−ya)/H).
    // ea, eb both ≤ 1 (arguments ≤ 0) so they NEVER overflow. The DIFFERENCE form
    // (ea−eb)/x is exact everywhere except x→0, where it subtracts two near-equal
    // fp32 values — catastrophic cancellation when H is HUGE (the near-uniform
    // default scene.fog profile; banded thick uniform fog + mis-weighted the
    // GI/reflection recombine). There the Taylor series of (1−e^{−x})/x is exact.
    // KEEP IN SYNC with deferred_filter_common.glsl and particle.frag.
    const float ea = exp(-ya / H);
    const float eb = exp(-yb / H);
    const float x  = (yb - ya) / H;
    const float f  = (abs(x) < 1e-3) ? (ea * (1.0 - 0.5 * x + x * x * (1.0 / 6.0)))
                                     : ((ea - eb) / x);
    // Saturate the optical depth: exp(-80) ≈ 1.8e-35 ≈ 0, so anything thicker is
    // fully extinct anyway. With len already finite the product is finite, so this
    // guarantees a FINITE, non-NaN result at every caller's exp(-od) (the general
    // fog-hardening rule: no exp(-opticalDepth) is ever fed an Inf/NaN).
    return min(clouds.hfDensity * len * f, 80.0);
}
// Surface fog in heterogeneous mode: CLOSED-FORM height-fog extinction over the
// WHOLE camera→surface leg + the ambient/skylight in-scatter fade toward the haze.
//
// Extinction uses the analytic optical-depth integral (heightFogOpticalDepth), NOT
// the froxel LUT's .a transmittance. The froxel medium σ is height-fog ONLY (the
// same exponential profile this closed form integrates — mediumExtinction in
// cloud_density.glsl excludes clouds), so for a SMOOTH medium the two are
// identical; the closed form is used because it is exact at every distance and
// avoids the LUT's per-slice quantisation, which a THICK, near-uniform medium
// (scene.fog → the huge-falloff default profile) surfaced as a bright horizon
// band in an enclosed room. For the uniform default this is EXACTLY applySceneFog's
// Beer-Lambert, so scene.fog fades identically whether the froxels are on or not.
// The froxel LUT still carries the POINT-light glow in-scatter (the directional
// sun is a scale-independent per-pixel march — see volumetricDirScatter) — added
// SEPARATELY by the caller via froxelInscatter — so the near-field glow is
// unaffected. (Extinction drops the height fog's noise modulation, a smooth mean,
// which is exactly what an extinction wants; the sun march keeps that same mean.)
//
// Ambient in-scatter = medium ALBEDO × an ambient-light estimate (env mean + scene
// ambient), the SAME closed form A·fogLight·(1−T) applySceneFog uses, keyed on the
// whole-path transmittance T → exact over the WHOLE path with NO 512 m seam and
// disjoint from the froxel's direct-light term (no double count). fogLight + albedo
// mirror applySceneFog AND the froxel injector's medium convention, so distant
// terrain and the fogged sky (applySkyFog) converge to ONE haze. fuv retained for
// signature compatibility (the LUT extinction it selected is now analytic).
vec3 applyHeteroSurfaceFog(vec3 col, vec2 fuv, float viewDist, vec3 ro, vec3 hit) {
    const float T = exp(-heightFogOpticalDepth(ro, hit));
    const vec3 fogLight  = lights.ambient
                         + sampleEnvLod(vec3(0.0, 1.0, 0.0), float(max(pc.envMipCount, 1u) - 1u));
    const vec3 medAlbedo = (fog.enabled > 0.5) ? fog.color : vec3(1.0);
    return col * T + medAlbedo * fogLight * (vec3(1.0) - T);
}
// ── ParticleField dust over an arbitrary camera leg (plan §3.3, phase 2) ─────
//
// A PER-PIXEL march of the WORLD-ANCHORED r16f density volumes — not the
// froxel LUT. The first ship of this feature read the dust's transmittance
// out of the LUT's .a channel and both of that grid's resolutions showed:
// 128×72 across the screen pixelated a compact plume into ~15-px cells, and
// the view-anchored depth slices (~12.6% of distance) re-quantised the cloud
// every time the camera moved — VIEW-DEPENDENT dust, on the renderer whose
// entire sensor argument is that every consumer sees one world state. The
// volume itself is world-anchored and the march is along the world-space ray,
// so this form is view-independent by construction and resolves to the
// volume's own voxels, not the froxel grid's.
//
// Extinction is the marched optical depth (Beer-Lambert). In-scatter is the
// house closed form (ambient + env mean, unphased — applyHeteroSurfaceFog's
// convention) PLUS the directional sun with the medium's HG phase — the term
// volumetricDirScatter cannot supply, because its march is the height-fog
// σ profile and it early-outs entirely when scene fog is off. The sun is
// shadowed ONCE, at the σ-weighted centroid of the traversed dust (a full
// per-step shadow ray is shaft-resolution the plume scale does not need),
// dimmed by the cloud-shadow map at the same point — and SELF-shadowed by the
// medium itself (pdLightTransmittance at the same centroid): the scene shadow
// map holds geometry only, density volumes never render into it, so without
// the medium's own light-leg transmittance a deep soot column was fully
// sunlit through its entire depth and read as white ash at any albedo.
//
// ── POINT LIGHTS ARE MARCHED HERE, NOT READ OUT OF THE FROXEL LUT ────────────
// They used to arrive through froxelInscatter: the injector multiplied its
// clustered point-light in-scatter by the dust's own σ, on the argument that
// "a glow is low-frequency in exactly the way a silhouette is not". That
// argument fails when the light sits INSIDE a thin plume, which is the entire
// campfire geometry. The froxel grid is 128×72×64 and VIEW-anchored (10 px per
// cell laterally at 720p, depth slices ~12.6% of distance); its σ is sampled at
// ONE per-frame-jittered point per cell and converged by a temporal EMA whose
// history cap collapses to 2 frames as soon as the camera moves. Inside a plume
// the sub-cell σ variance is enormous, so during camera motion that estimator
// does not converge: the product σ·glow boils in cell-sized blocks GLUED TO THE
// SCREEN while the world slides underneath. Measured on the campfire (frozen
// effect, 22°/s orbit): the frame-to-frame difference over the smoke column
// carried 1.24× more variance when blocked on the froxel lattice than when
// blocked half a cell out of phase — structure locked to the grid — and
// dropping this term took that to 1.05× (i.e. gone).
//
// It is the SAME mechanism the header above evicted from the extinction path,
// arriving a second time through the in-scatter path. So the point-light glow
// is computed per step of THIS march instead, from the 8-cap LightsUbo set,
// with the same per-step terms the injector used (1 m-clamped inverse square,
// quartic range window, HG phase, light-leg extinction) — world-anchored by
// construction, at the volume's own resolution.
//
// The two media now partition cleanly, each sourcing its in-scatter in the
// representation that can actually resolve it: HEIGHT FOG (smooth, analytic,
// low-frequency) keeps the froxel grid; PARTICLE DUST (high-frequency, world-
// anchored) is marched. froxel_inject.comp drops σ_particles from its in-scatter
// SOURCE term to match, so nothing is counted twice — it keeps dust in the
// light-leg extinction, where a plume dimming a lamp behind it is correct and
// genuinely low-frequency.
//
// SPOT lights are skipped, as they are in the injector: a tight cone is the
// high-frequency case that already owns a dedicated per-pixel march
// (volumetricSpotScatter). That march integrates the height-fog/beam medium and
// not the dust, so a spot shining into a plume gets no dust glow — a v1 gap,
// recorded rather than papered over.
//
// ∫σ·e^(-τ) dt over the dust segment is accumulated as `wsum`; with dust the
// only extinction on that segment it equals 1 - T exactly, so the in-scatter
// needs no separate normalisation.
//
// Applied to EVERY leg the shade terminates: surfaces, water, and the sky
// background. Gated on flags bit 11 = "a ParticleField density volume is live
// this frame", so a scene without dust pays four compares and an early return.
//
// ── EMISSION (fire) — plans/particle-atmosphere.md F-A ───────────────────────
// A field with DensityRepr::emissiveIntensity > 0 adds an emitted-radiance term
// to the same march, per step and per emissive volume:
//
//     T   = mix(bottomK, topK, pow(heightFraction, falloff))
//     E  += intensity * blackbodyRGB(T) * sigma_i * e^(-tau) * dt
//
// Emission scales with THAT VOLUME's own sigma because that is the physics —
// soot radiates in proportion to how much soot there is — and it is also what
// makes the flame's silhouette be the particle distribution rather than a
// billboard card: there is no flame texture anywhere in this renderer. The
// e^(-tau) factor is self-occlusion, so a deep flame's far side is dimmed by
// its own near side exactly as the in-scatter is. `heightFraction` is tt.y, the
// in-box normalised coordinate the loop already computes for the fetch.
//
// The whole emissive path sits behind `pd.counts.y != 0` — a UNIFORM branch on
// "any bound volume is emissive" — so a dust-only scene executes the identical
// arithmetic it did before this term existed. That is the F0 checkpoint, not a
// nicety.
//
// ── THE MARCH GRID: DEPTH-INDEPENDENT, ~1 m STEPS, SPATIALLY DITHERED ────────
// Three rules, each answering a defect a capture showed:
//
//   1. The step grid is a function of the DENSITY-BOX UNION alone — tMax does
//      not shape it. The first ship clipped the marched interval to scene
//      depth, which handed rays on either side of a depth edge DIFFERENT step
//      grids over the SAME plume: a sky ray (tMax 1e30) marched the full box
//      span while the ground ray beside it marched a truncated one, and the
//      two quantisations of one σ field disagree by a visible constant — a
//      hard seam tracing the horizon through every plume that straddled it.
//      The surface clip is per STEP instead (the slab straddling tMax
//      contributes its front fraction, later slabs nothing): that is the
//      physical clip, and it is continuous in tMax where reparameterising the
//      whole march was not.
//   2. The step count is the fixed 24 (32 emissive) of the first ship, RAISED
//      — never lowered — toward a ~1 m world-space step for long traversals,
//      capped at 64. A fixed count over a box-sized interval is a step length
//      that scales with the box: a yard-scale plume marched in ~4 m slabs and
//      showed them as shells, while a campfire keeps its sub-metre steps
//      through the base count, unchanged.
//   3. A per-PIXEL hash offset on the step phase — for EVERY march now, not
//      only emissive ones, since rule 2's cap still leaves metre-scale slabs
//      on a big plume. This is spatial dithering: a function of the pixel
//      coordinate ALONE, with no frame term anywhere, so a static scene still
//      renders the same bytes every frame and every run — unlike the pcg
//      jitter volumetricSpotScatter/volumetricDirScatter use, which are keyed
//      on pc.frame and are therefore off limits here (the phase-2 determinism
//      contract). TAA converges the dither exactly as it converges those
//      marches' jitter; without TAA it reads as fine stationary noise instead
//      of hard shells, which is the better failure.
vec3 applyParticleFog(vec3 col, vec3 ro, vec3 rd, float tMax) {
    if ((pc.flags & 2048u) == 0u) return col;
#ifdef PD_LINEAR
    const uint n = min(pd.counts.x, uint(kMaxDensityFields));
    if (n == 0u) return col;

    // Union of the ray's box overlaps — one interval, so disjoint plumes cost
    // steps in the gap, but K is fixed and two plumes usually share a yard.
    // The BOXES alone bound it — not tMax — so the step grid below is the same
    // for every ray of a pixel-neighbourhood whatever each one hit (rule 1).
    float t0 = 1e30, t1 = 0.0;
    const vec3 inv = 1.0 / rd;// ±inf on axis-parallel components is fine below
    for (uint i = 0u; i < n; ++i) {
        const vec3  bmin  = pd.boxMin[i].xyz;
        const vec3  bsize = 1.0 / pd.boxInvSize[i].xyz;
        const vec3  ta = (bmin - ro) * inv;
        const vec3  tb = (bmin + bsize - ro) * inv;
        const vec3  lo = min(ta, tb), hi = max(ta, tb);
        const float e = max(max(lo.x, lo.y), max(lo.z, 0.0));
        const float x = min(min(hi.x, hi.y), hi.z);
        if (x > e) { t0 = min(t0, e); t1 = max(t1, x); }
    }
    if (t1 <= t0 || tMax <= t0) return col;// no overlap / all of it behind the hit

    // Uniform branches, both of them: every pixel in the dispatch takes the
    // same side, so neither costs divergence.
    const bool emissive = pd.counts.y != 0u;
    const bool multi    = n > 1u;

    // Rule 2: base count, raised toward a 1 m world step on long traversals.
    const float span  = t1 - t0;
    const int   STEPS = clamp(int(ceil(span)), emissive ? 32 : 24, 64);
    const float dt    = span / float(STEPS);
    // Step phase: the per-pixel hash dither of rule 3, for every march.
    // pcgHash of the pixel coordinate only — NO pc.frame term, deliberately.
    uint seed = pcgHash(uint(gPixelCoord.x) * 7919u + pcgHash(uint(gPixelCoord.y) * 104729u));
    const float phase = rnd(seed);
    float tau = 0.0, wsum = 0.0;
    vec3  cen = vec3(0.0);
    vec3  emis = vec3(0.0);// sum of intensity * blackbody * sigma_i * e^-tau * dt
    vec3  albW = vec3(0.0);// sigma-weighted albedo, already carrying the w weight
    float gW   = 0.0;      // sigma-weighted HG g, same weighting
    vec3  pnt  = vec3(0.0);// point-light in-scatter, ALREADY albedo-multiplied
    for (int s = 0; s < STEPS; ++s) {
        const float slab0 = t0 + float(s) * dt;
        // Rule 1's surface clip: the slab straddling tMax contributes its
        // front fraction, everything behind the hit contributes nothing. The
        // grid stations themselves never move with tMax.
        const float ds = min(dt, tMax - slab0);
        if (ds <= 0.0) break;
        const float t = slab0 + phase * ds;
        const vec3  x = ro + rd * t;
        float sig = 0.0;
        vec3  albStep = vec3(0.0);
        float gStep   = 0.0;
        vec3  emStep  = vec3(0.0);
        for (uint i = 0u; i < n; ++i) {
            const vec3 tt = (x - pd.boxMin[i].xyz) * pd.boxInvSize[i].xyz;
            // Skip, don't clamp: CLAMP_TO_EDGE would smear each face's voxels
            // over everything outside the box.
            if (any(lessThan(tt, vec3(0.0))) || any(greaterThan(tt, vec3(1.0)))) continue;
            const float si = texture(particleDensityLinTex[i], tt).r;
            sig += si;
            if (multi) {
                albStep += pd.albedoAniso[i].rgb * si;
                gStep   += pd.albedoAniso[i].a * si;
            }
            // Blackbody ramp over the height fraction inside THIS volume's
            // box. The expression itself lives in particle_density.glsl so
            // that the reflected-leg march (pdEmissiveLeg) cannot drift away
            // from it — ONE emission model, called twice.
            if (emissive && pd.emission[i].x > 0.0) emStep += pdEmissionAt(i, tt.y, si);
        }
        if (sig <= 0.0) continue;
        const float tr = exp(-tau);
        const float w  = sig * tr * ds;// KEEP the grouping: this is the pre-emission expression
        tau  += sig * ds;
        wsum += w;
        cen  += w * x;
        if (multi) {
            albW += albStep * (tr * ds);
            gW   += gStep * (tr * ds);
        }
        if (emissive) emis += emStep * (tr * ds);
        // ── Point-light glow, per step ──────────────────────────────────────
        // Unlike the sun, a point light's geometry changes ALONG the ray
        // (direction turns, inverse square falls off), so it cannot be factored
        // out to the one centroid evaluation the sun gets — it has to be inside
        // the loop. Uniform branch: pointCount is the same for every pixel.
        if (lights.pointCount > 0u) {
            // The LOCAL mixture params here, not the ray-averaged medAlbedo /
            // medG the sun and ambient use. The phase has to be evaluated per
            // step regardless (toL turns), so the albedo rides along on the
            // same divide, and a warm flame overlapping a grey smoke column
            // then tints its own share of the glow instead of the ray's mean.
            const vec3  aL = multi ? albStep / sig : pd.albedoAniso[0].rgb;
            const float gL = multi ? gStep / sig : pd.albedoAniso[0].a;
            vec3 pl = vec3(0.0);
            for (uint i = 0u; i < lights.pointCount; ++i) {
                vec3        toL = lights.pointLights[i].position - x;
                const float d   = length(toL);
                if (d < 1e-3) continue;
                toL /= d;
                // The froxel injector's per-step terms verbatim, so moving the
                // light here changes WHERE the glow is resolved and not what it
                // is: 1 m-clamped inverse square (no firefly at the source),
                // the same quartic range window the surface path applies, HG
                // phase, and the light-leg transmittance. That last one uses
                // the LOCAL dust sigma over the whole leg — the injector's own
                // homogeneous-at-x approximation, and a good one here because
                // a light inside a plume has a short leg.
                float atten = 1.0 / max(d * d, 1.0);
                const float range = (lights.pointLights[i].range > 0.0)
                                          ? lights.pointLights[i].range : 200.0;
                const float rt  = d / range;
                const float r4  = rt * rt * rt * rt;
                const float wnd = max(1.0 - r4, 0.0);
                atten *= wnd * wnd;
                pl += lights.pointLights[i].color
                    * (atten * hgPhase(dot(toL, rd), gL) * exp(-sig * d));
            }
            // w is sigma * e^-tau * dt — the same weight the ambient/sun term
            // integrates, so the point glow is scattered by this step's dust
            // and self-occluded by the dust in front of it, consistently.
            pnt += pl * aL * w;
        }
        if (tau > 8.0) break;// e^-8: nothing behind this survives
    }
    if (wsum <= 0.0) return col;
    cen /= wsum;

    // The medium's own single-scatter albedo and phase (DensityRepr::albedo /
    // anisotropy), not the FogUbo's — a dust field can be present with no scene
    // fog at all. With ONE volume bound they are that field's values verbatim,
    // which is the pre-F0 expression bit for bit; with several they are the
    // sigma-weighted mixture along this ray, so a warm fire and a grey smoke
    // column each tint their own share of the in-scatter.
    vec3  medAlbedo = pd.albedoAniso[0].rgb;
    float medG      = pd.albedoAniso[0].a;
    if (multi) {
        medAlbedo = albW / wsum;
        medG      = gW / wsum;
    }

    vec3 sun = vec3(0.0);
    for (uint i = 0u; i < lights.dirCount; ++i) {
        const vec3 L = normalize(lights.dirLights[i].direction);
        // Three occluders on the sun's leg, disjoint by construction:
        // shadowVis is GEOMETRY (the RT scene — the medium is not in it),
        // cloudShadowSample is the cloud layer overhead, and
        // pdLightTransmittance is the medium ITSELF — Beer-Lambert through
        // the density mirrors from the centroid toward the sun, the term
        // that makes a soot column's shaded side actually dark.
        sun += lights.dirLights[i].color
             * (hgPhase(dot(L, rd), medG)
                * shadowVis(cen, L, 1e30) * cloudShadowSample(cen)
                * pdLightTransmittance(cen, L));
    }
    const vec3 amb = lights.ambient
                   + sampleEnvLod(vec3(0.0, 1.0, 0.0), float(max(pc.envMipCount, 1u) - 1u));
    // ── EVERYTHING THIS MEDIUM ADDS IS STILL BEHIND THE AIR ─────────────────
    // The invariant: radiance a medium adds to the frame must be attenuated by
    // every medium between it and the camera. The caller has already fogged
    // `col` (applyHeteroSurfaceFog / applySceneFog, then applyMurk) and the
    // extinction of that background by this dust is the `col * exp(-tau)` term
    // — but the dust's OWN radiance, the three terms after it, was added raw.
    // A plume 70 m away therefore composited as if it were at the lens: a small
    // crisp wisp sitting in front of the murk, visibly less attenuated than the
    // trees at its own depth. That is the third appearance of the
    // media-don't-compose class (F4 defect 2 was the same symptom on the legacy
    // sprite path against the snow haze), and this is its deferred-path twin.
    //
    // NO DOUBLE COUNT — the F1 rule holds, extinction and in-scatter keep
    // separate owners. This multiplies the dust's added radiance by the air's
    // TRANSMITTANCE only; it does not re-add one photon of air in-scatter,
    // which the caller already composited. The expression mirrors the surface
    // path's own choice of medium (hetero height fog when the froxels are
    // active, the vestigial homogeneous fog otherwise) so a plume and the tree
    // behind it fade through the same profile, and with no air medium bound
    // every factor is exactly 1 and the return is textually the pre-fix one.
    //
    // AT THE CENTROID, not per step. `cen` is the sigma-weighted centroid the
    // march already computed and already uses for the sun's shadow ray and the
    // cloud-shadow sample, so this reuses that approximation rather than
    // inventing a second one: the air's optical depth varies by well under a
    // percent across a ten-metre plume seventy metres out, while a per-step
    // form would put two exps and a distance() inside the march loop.
    //
    // AND THE OTHER HALF, which a capture caught and arithmetic would not have:
    // attenuating only the added term turns a plume in THICK fog into a DARK
    // HOLE. `col` arrives carrying the air's in-scatter over the WHOLE leg, and
    // `col · e^-tau` extinguishes all of it — including the share scattered
    // BETWEEN the camera and the plume, which is in FRONT of the plume and which
    // the plume therefore cannot occlude. In thin fog that share is nothing; at
    // --mist 0.010 it is most of the pixel, and the previously-unattenuated
    // added term had been quietly cancelling the error. Fixing one without the
    // other swapped a plume that would not fade for a black smudge.
    //
    // The split is exact and costs one lerp. With I the air's in-scatter
    // radiance and T its transmittance to the plume, the caller handed us
    //     col = colSurf·T_hit + I·(1 − T_hit),
    // and the composition we want is
    //     colSurf·T_hit·e^-tau + I·(T_cen − T_hit)·e^-tau + I·(1 − T_cen) + add·T_cen,
    // which reduces, substituting col, to
    //     col·e^-tau + I·(1 − T_cen)·(1 − e^-tau) + add·T_cen.
    // So: put back the near-side haze the dust wrongly ate, in proportion to how
    // much it ate. I and T are built from the SAME expressions the surface path
    // used (hetero height fog when the froxels are active, else the vestigial
    // homogeneous fog, then murk), so plume and background fade through one
    // profile; `amb` above is already applyHeteroSurfaceFog's `fogLight` term
    // for term, so it is reused rather than sampled a second time.
    vec3 trAir = vec3(1.0);
    vec3 airIn = vec3(0.0);// I·(1 − T), summed per medium
    if (clouds.heteroActive > 0.5) {
        const vec3 T = vec3(exp(-heightFogOpticalDepth(ro, cen)));
        airIn += ((fog.enabled > 0.5) ? fog.color : vec3(1.0)) * amb * (vec3(1.0) - T);
        trAir *= T;
    } else if (fog.enabled > 0.5) {
        const vec3 T = exp(-fog.sigmaT * fogPathLength(ro, cen));
        airIn += fog.color * amb * (vec3(1.0) - T);
        trAir *= T;
    }
    if (fog.murkDensity > 0.0) {
        const vec3 T = exp(-vec3(fog.murkDensity) * fogPathLength(ro, cen));
        airIn += fog.murkColor * amb * (vec3(1.0) - T);
        trAir *= T;
    }
    // With NO air medium bound both are identity (trAir 1, airIn 0) and the
    // return below is the pre-fix expression textually, not merely equivalently.
    //
    // Approximation left standing, and named: with air fog AND murk both live
    // the two in-scatter shares are summed rather than nested, so the air's
    // share misses the murk's transmittance factor. Murk is clipped to the
    // below-waterline leg, so this is exact for every camera above water — the
    // fjord's case, and the reported one.
    const float trDust = exp(-tau);

    // Emission is ADDED, not scattered: it is radiance the medium produced, so
    // it neither multiplies by the albedo (that is the fraction of INCIDENT
    // light re-emitted) nor attenuates the background any further than tau
    // already did. `pnt` is already albedo-weighted and already carries its own
    // per-step w, so it is added rather than folded into the wsum product.
    return col * trDust
         + airIn * (1.0 - trDust)
         + (medAlbedo * (amb + sun) * wsum + emis + pnt) * trAir;
#else
    return col;
#endif
}

vec3 applySceneFog(vec3 col, vec3 ro, vec3 hit) {
    if (fog.enabled < 0.5) return col;
    const float d = fogPathLength(ro, hit);
    if (d <= 0.0) return col;
    const vec3 tr = exp(-fog.sigmaT * d);
    // In-scatter = fog ALBEDO × an ambient-light estimate (env mean + scene
    // ambient). The fog colour is a single-scattering albedo in this engine
    // (a full volumetric NEE solution would light the medium directly)
    // — mixing toward it directly,
    // three.js-style, makes fog GLOW in dark scenes (a black Cornell room
    // washed out to white). Spot-light in-scatter is added separately by the
    // fog-driven volumetricSpotScatter march (the god rays).
    const vec3 fogLight = lights.ambient
                        + sampleEnvLod(vec3(0.0, 1.0, 0.0), float(max(pc.envMipCount, 1u) - 1u));
    return col * tr + fog.color * fogLight * (vec3(1.0) - tr);
}
// ── Underwater murk (setUnderwaterMurk) ──────────────────────────────────────
// A SEPARATE homogeneous medium clipped to the BELOW-waterSurfaceY portion of the
// camera→surface leg — the water body's own absorption/tint, DECOUPLED from the
// air fog (scene.fog) in Phase 2 so a scene can hold clear air above the waterline
// and murk below (the fjord). Beer-Lambert extinction toward the murk colour with
// the same ambient in-scatter applySceneFog uses. Applied AFTER the air-fog
// surface dispatch: air fog and murk own DISJOINT leg portions in the common
// camera-above-water case (fogPathLength returns 0 above the surface), so there is
// no double count; a fully-submerged leg would get both, an accepted approximation.
// murkDensity == 0 → exact no-op (byte-identical when murk is off).
vec3 applyMurk(vec3 col, vec3 ro, vec3 hit) {
    if (fog.murkDensity <= 0.0) return col;
    const float d = fogPathLength(ro, hit);// clipped to y < waterSurfaceY
    if (d <= 0.0) return col;
    const vec3  tr = exp(-vec3(fog.murkDensity) * d);
    const vec3  fogLight = lights.ambient
                         + sampleEnvLod(vec3(0.0, 1.0, 0.0), float(max(pc.envMipCount, 1u) - 1u));
    return col * tr + fog.murkColor * fogLight * (vec3(1.0) - tr);
}
// ── Sky aerial perspective (deferred) ────────────────────────────────────────
// The HDR background is infinitely far, so applySceneFog never touches it — a
// foggy scene then shows distant geometry fading to fog colour against a crisp,
// full-bright sky: the fog "wall" never closes and the HDR shines through. Blend
// the sky toward the SAME inscatter colour distant geometry fades to, weighted to
// the HORIZON (where the view ray crosses the most medium) and fading out toward
// the zenith — aerial perspective / a horizon haze band. Density drives overall
// strength so thin fog barely tints the sky and thick fog closes it off. The
// volumetric sun glow is added AFTER this, so it still blooms through the haze.
// Closed-form height-fog optical depth of an INFINITE view ray from the camera
// toward `dir` — the sky's aerial-perspective haze band (the surface path's
// finite heightFogOpticalDepth taken to t→∞). World-Y profile, matching the
// surface path so a distant valley wall and the sky behind it converge to the
// SAME haze at the horizon. The ray geometry itself horizon-weights it (a short
// column at the zenith → clear for thin mist; a long grazing column near the
// horizon → closed). Near-horizontal / downward rays never leave the layer, so
// the elevation is floored (else OD→∞).
float heightFogSkyOpticalDepth(vec3 dir) {
    if (clouds.hfDensity <= 0.0) return 0.0;
    const float H    = max(clouds.hfFalloff, 1e-3);
    // Height of the ray's START above the fog base. Under a parallel projection
    // that is this pixel's own near-plane point, not the shared eye — a top-down
    // ortho view would otherwise integrate every column from the same altitude.
    const float hCam = gPrimaryOrigin.y - clouds.hfBaseY;
    const float m    = max(dir.y, 0.02);                     // elevation; floor the horizon
    // ∫₀^∞ σ0·e^(−max(hCam+m·t,0)/H) dt. Above base: a pure decaying column;
    // below base: a constant-σ0 slab (−hCam/m long) precedes the decaying part.
    return (hCam >= 0.0) ? clouds.hfDensity * H / m * exp(-hCam / H)
                         : clouds.hfDensity / m * (H - hCam);
}
vec3 applySkyFog(vec3 sky, vec3 dir) {
    // Phase 2: ONE air medium. Its density is hfDensity when the froxels are
    // active (scene.fog OR setHeightFog both feed it now), else the homogeneous
    // scene-fog σ (a vestigial path — the air medium is always hetero when fog is
    // present). Composing BOTH bands would double the sky haze, so pick one.
    const bool  hetero = clouds.hfDensity > 0.0;
    const float sigma  = hetero ? clouds.hfDensity
                       : ((fog.enabled > 0.5) ? max(dot(fog.sigmaT, vec3(1.0 / 3.0)), 0.0) : 0.0);
    if (sigma <= 0.0) return sky;
    const vec3  up   = (dot(fog.worldUp, fog.worldUp) > 1e-6) ? normalize(fog.worldUp) : vec3(0.0, 1.0, 0.0);
    const float elev = clamp(dot(dir, up), 0.0, 1.0);        // 0 horizon .. 1 zenith
    // Sky transmittance. A LAYERED height fog (finite falloff) uses the closed-
    // form infinite-ray optical depth of its exponential profile — the sky clears
    // above the layer and closes through it (true aerial perspective; preserves
    // the Phase 1 --mist horizon closure). A NEAR-UNIFORM medium (the default
    // scene.fog profile, huge falloff) would give a DIVERGENT OD (a uniform slab
    // to infinity → the horizon column is infinite), so it falls back to the
    // bounded horizon-band heuristic the homogeneous fog used — subtle sky tint
    // while the distant SURFACE haze (applyHeteroSurfaceFog, physical over the
    // finite leg) does the horizon closure, exactly as the homogeneous march did.
    float T;
    if (hetero && clouds.hfFalloff < 4000.0) {
        T = exp(-min(heightFogSkyOpticalDepth(dir), 80.0));// saturate: no Inf into exp
    } else {
        const float band     = exp(-elev * 2.5);            // haze concentrated near the horizon
        const float strength = 1.0 - exp(-sigma * 45.0);    // density → how much sky is obscured
        T = 1.0 - band * strength;
    }
    const vec3 fogLight = lights.ambient + sampleEnvLod(up, float(max(pc.envMipCount, 1u) - 1u));
    const vec3 hazeCol  = (fog.enabled > 0.5) ? fog.color : vec3(1.0);
    return mix(sky, hazeCol * fogLight, 1.0 - T);
}
// ── Procedural star field (sky pixels) ───────────────────────────────────────
// Hash-based stars evaluated in DIRECTION space — resolution-independent and
// pixel-crisp at any FOV/zoom. Baked env-texture stars are sub-texel features,
// so bilinear magnification (+ the TAA upscale) rendered every one of them as a
// soft ~14 px blob no matter the env resolution. Cells on the (azimuth,
// elevation) grid; ~2% of cells host a star at a hashed sub-cell position with
// a power-law brightness; a tight gaussian gives ~1 px points that twinkle
// subtly under the TAA jitter (which is what real stars do). Stable across
// frames (no frame term). Applied to SKY pixels only — points this small are
// invisible in rough reflections anyway.
vec3 proceduralStars(vec3 dir) {
    if (pc.starIntensity <= 0.0 || dir.y <= 0.0) return vec3(0.0);
    const float SCALE = 220.0;
    const vec2  sph = vec2(atan(dir.z, dir.x), asin(clamp(dir.y, -1.0, 1.0)));
    const vec2  g   = vec2(sph.x, sph.y) * SCALE;
    vec3 sum = vec3(0.0);
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {// 3×3 so stars near cell edges don't clip
        const vec2  cell = floor(g) + vec2(float(dx), float(dy));
        const float h    = hash21(cell * 0.193 + 11.71);
        if (h > 0.02) continue;// ~2% of cells → naked-eye star density
        const vec2  sp = cell + vec2(hash21(cell + 31.7), hash21(cell + 57.3));
        const vec2  d  = g - sp;
        const float r2 = dot(d, d);
        const float bright = 0.4 + 2.4 * pow(hash21(cell + 91.1), 8.0);// many dim, few bright
        sum += vec3(1.0, 1.0, 1.06) * (bright * exp(-r2 * 24.0));
    }
    // Horizon fade — stars dissolve into the atmospheric band, no hard pop.
    return sum * (pc.starIntensity * smoothstep(0.0, 0.05, dir.y));
}

// (Only the clustered POINT-light glow lives in froxel_inject/integrate.comp —
// the shade samples its integrated LUT through froxelInscatter above (soft
// omnidirectional glow the old spot-only march never gave them). Both the SPOT
// beams below AND the directional SUN (volumetricDirScatter) stay per-pixel:
// each is a high-frequency shaft the froxel grid cannot resolve — a tight bright
// cone (the lighthouse turns to mush), a thin tree-gap god ray (a cell can't
// resolve a shaft narrower than itself). Per-pixel marching is scale-independent.)

vec3 volumetricSpotScatter(vec3 ro, vec3 rd, float tMax, ivec2 px) {
    // Scene fog drives the march when active (σ, HG g, and the medium albedo
    // tint all come from the fog UBO → spotlight god rays appear automatically
    // in any fogged deferred scene); pc.volDensity remains the explicit knob
    // for clear-air beams (the lighthouse).
    const bool  fogDriven = fog.enabled > 0.5;
    const float sigma = fogDriven ? max(dot(fog.sigmaT, vec3(1.0 / 3.0)), pc.volDensity)
                                  : pc.volDensity;
    const float hgG   = fogDriven ? fog.anisotropy : pc.volAniso;
    const vec3  medAlbedo = fogDriven ? fog.color : vec3(1.0);
    if (sigma <= 0.0 || lights.spotCount == 0u) return vec3(0.0);
    const int   STEPS = 40;
    uint seed = pcgHash(uint(px.x) * 7919u + pcgHash(uint(px.y) * 104729u + pc.frame * 6271u));
    const float jitter = rnd(seed);
    vec3 sum = vec3(0.0);
    for (uint i = 0u; i < lights.spotCount; ++i) {
        const SpotLight sl = lights.spotLights[i];
        const float range = (sl.range > 0.0) ? sl.range : 200.0;
        const vec3  oc   = ro - sl.position;
        const float b    = dot(oc, rd);
        const float c    = dot(oc, oc) - range * range;
        const float disc = b * b - c;
        if (disc <= 0.0) continue;
        const float sq = sqrt(disc);
        const float t0 = max(-b - sq, 0.0);
        const float t1 = min(-b + sq, tMax);
        if (t1 <= t0) continue;
        const float dt = (t1 - t0) / float(STEPS);
        vec3 acc = vec3(0.0);
        for (int s = 0; s < STEPS; ++s) {
            const float t = t0 + (float(s) + jitter) * dt;
            const vec3  x = ro + rd * t;
            vec3  toL = sl.position - x;
            const float dist = length(toL);
            if (dist < 1e-3) continue;
            toL /= dist;
            const float spotCos = dot(-toL, sl.direction);
            const float cone    = smoothstep(sl.cosAngleOuter, sl.cosAngleInner, spotCos);
            if (cone <= 0.0) continue;
            // Inverse-square with a 1 m clamp (no firefly core at the lamp) +
            // the same range window the surface path applies.
            float atten = 1.0 / max(dist * dist, 1.0);
            const float tr  = dist / range;
            const float t4  = tr * tr * tr * tr;
            const float wnd = max(1.0 - t4, 0.0);
            atten *= wnd * wnd;
            // HG phase: incoming (light→x) vs outgoing (x→camera = -rd).
            const float mu    = dot(-toL, -rd);
            const float phase = hgPhase(mu, hgG);
            // Transmittance camera→x and light→x (uniform thin haze).
            const float trans = exp(-sigma * (t + dist));
            acc += sl.color * (cone * atten * phase * trans);
        }
        // σ_s = σ_t · albedo: the medium's tint scales what gets IN-scattered.
        sum += acc * medAlbedo * (sigma * dt);
    }
    return sum;
}

// ── Directional-light volumetric single scattering (god rays / aerial glow) ───
// Ray-marches the WHOLE camera→surface (or →sky-cap) segment [0, tEnd]; at each
// step an RT shadow ray toward each sun carves real light shafts through
// trees/terrain, and the Henyey-Greenstein phase brightens the haze TOWARD the
// sun (the "volume" a flat extinction haze lacks). This per-pixel march is the
// SOLE owner of the directional-sun in-scatter over the ENTIRE ray — near field
// included. Sun shafts are HIGH-frequency (a thin occluder — a 0.3–2 m tree gap —
// against thin fog integrated over the leg); the froxel grid's depth cells span
// ~12.6% of the distance (~1.3 m at 10 m), so a cell CANNOT resolve a shaft
// narrower than itself: close-range gap god rays averaged into a featureless glow
// with nothing for the phase to shape (ridge-scale fjord shafts spanned many
// cells, which masked this). A per-pixel march is scale-INDEPENDENT — it resolves
// the gap wherever it lands — so the froxel sun loop is DELETED (froxel_inject.comp
// keeps only the clustered POINT-light glow + the medium-extinction LUT) and this
// owns [0, tEnd] outright. The per-pixel jittered step offset relies on TAA to
// average into smooth shafts.
vec3 volumetricDirScatter(vec3 ro, vec3 rd, float tMax, ivec2 px) {
    // WHOLE-RAY sun shafts (Phase 3 unified fog). No 512 m split any more — there
    // is no sun seam by construction. Same height-fog σ PROFILE and HG phase as
    // before; the camera→x transmittance is now the closed-form MEAN height-fog
    // optical depth over the WHOLE leg [0, x] (the far-tail steps always used this
    // closed form — now the near leg does too, so the whole leg is one analytic
    // integral, EXACTLY the extinction applyHeteroSurfaceFog applies to the
    // surface). The noise modulation is intentionally excluded (an extinction
    // wants the smooth mean — the established far-tail precedent). Every step
    // traces an RT shadow ray, so shafts are shadowed end to end; cloud shadow
    // overhead dims each step uniformly. Sky pixels (large tMax) march to the
    // optical-depth horizon the same way.
    if (clouds.heteroActive < 0.5 || clouds.hfDensity <= 0.0 || lights.dirCount == 0u) return vec3(0.0);
    const float hgG       = fog.anisotropy;// medium HG g — set unconditionally by updateFogUbo (matches froxel_inject's hetero g)
    const vec3  medAlbedo = (fog.enabled > 0.5) ? fog.color : vec3(1.0);
    const float H         = max(clouds.hfFalloff, 1e-3);
    // March the whole ray from the camera. The first sample sits at jitter·dt > 0
    // (never exactly at ro), so no epsilon is needed; heightFogOpticalDepth(ro,ro)
    // is 0 (T=1) anyway.
    const float tStart = 0.0;
    // Cap the march at the optical-depth horizon (~6/σ ⇒ e^-6 ≈ 0.25% left) so a
    // sky pixel's huge tMax doesn't spend all 16 steps on fully-attenuated
    // distance; the per-step `trCam` break below is the exact cutoff, this only
    // sizes dt. σ at the ray origin (camera height, small floor) is the near-end
    // density that dominates the leg.
    const float sigmaRef = max(clouds.hfDensity * exp(-max(ro.y - clouds.hfBaseY, 0.0) / H), 1e-6);
    const float tEnd = min(tMax, tStart + 6.0 / sigmaRef);
    if (tEnd <= tStart) return vec3(0.0);

    const int   STEPS  = 16;
    uint        seed   = pcgHash(uint(px.x) * 7919u + pcgHash(uint(px.y) * 104729u + pc.frame * 6271u));
    const float jitter = rnd(seed);
    const float dt     = (tEnd - tStart) / float(STEPS);

    vec3 sum = vec3(0.0);
    for (int s = 0; s < STEPS; ++s) {
        const float t = tStart + (float(s) + jitter) * dt;
        const vec3  x = ro + rd * t;
        // Local height-fog σ (smooth analytic mean — the noise modulation is a
        // near-field froxel concern; the march wants the mean). Same profile as
        // heightFogOpticalDepth / mediumExtinction.
        const float sigmaX = clouds.hfDensity * exp(-max(x.y - clouds.hfBaseY, 0.0) / H);
        if (sigmaX <= 1e-7) continue;
        const float trCam = exp(-heightFogOpticalDepth(ro, x));// closed-form camera → x over the WHOLE leg
        if (trCam < 0.003) break;                              // remaining contribution negligible
        vec3 stepSum = vec3(0.0);
        for (uint i = 0u; i < lights.dirCount; ++i) {
            const vec3  L   = normalize(lights.dirLights[i].direction);
            const float vis = shadowVis(x, L, 1e30);// tree/terrain shaft occlusion
            if (vis <= 0.0) continue;
            const float phase = hgPhase(dot(L, rd), hgG);// forward glow toward the sun
            stepSum += lights.dirLights[i].color * (phase * vis);
        }
        // σ_s = σ_t · albedo (albedo applied once below); cloud shadow overhead
        // dims the shafts exactly like the (now-deleted) froxel sun term did.
        sum += stepSum * (trCam * sigmaX * cloudShadowSample(x));
    }
    return sum * medAlbedo * dt;
}

// Fogged sky/background colour along this pixel's view ray — the term dispatch A
// blends at MSAA complex pixels for the SKY-minority coverage (and the full sky
// path uses the same expression). Volumetric in-scatter is NOT included: that is
// a per-pixel view-ray integral added once, at full weight, by the caller.
vec3 skyBackground(vec2 ndc, vec3 ro) {
    const vec3 dirWS = camRayDir(ndc);
    // Same dust term the full sky path applies (deferred_shade.comp's skyRad):
    // this is the MSAA sky-minority blend, and a minority that skipped the dust
    // would put an undusted rim along every silhouette inside the cloud.
    return applyParticleFog(
            applySkyFog(sampleEnvLod(dirWS, 0.0) + proceduralStars(dirWS), dirWS),
            ro, dirWS, 1e30);
}
