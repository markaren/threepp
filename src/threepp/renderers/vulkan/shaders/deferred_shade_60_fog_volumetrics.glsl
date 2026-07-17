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
// The froxel LUT still carries the DIRECT-light (sun shaft + point glow) in-scatter
// — added SEPARATELY by the caller via froxelInscatter — so the near-field shafts
// are unaffected. (Extinction drops the height fog's noise modulation, a smooth
// mean, which is exactly what an extinction wants; the shafts keep it.)
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
    const float hCam = cam.viewInverse[3].y - clouds.hfBaseY;// camera height above the base
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

// (The per-pixel DIRECTIONAL march moved into froxel_inject/integrate.comp —
// the shade samples its integrated LUT through froxelInscatter above: same
// terms, ~2 orders of magnitude fewer shadow rays — and clustered POINT
// lights gained a froxel fog glow the old spot-only march never gave them.
// SPOT beams stay per-pixel below: a tight bright cone is a high-frequency
// feature the froxel grid cannot resolve — the lighthouse turned to mush.)

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
// Ray-marches the camera→surface (or →sky-cap) segment; at each step an RT shadow
// ray toward each sun carves real light shafts through trees/terrain, and the
// Henyey-Greenstein phase brightens the haze TOWARD the sun (the "volume" a flat
// extinction haze lacks). Sun shafts are HIGH-frequency (thin occluders, thin
// fog integrated over kilometres) — the froxel grid blurred them to nothing,
// so they stay per-pixel. Gated on fog.enabled AND the volumetric-fog flag
// (bit 4) so the per-step shadow-ray cost is opt-in; the per-pixel jittered
// step offset relies on TAA to average into smooth shafts.
vec3 volumetricDirScatter(vec3 ro, vec3 rd, float tMax, ivec2 px) {
    // FAR-TAIL sun shafts (Phase 2 unified fog). The froxels own the near field
    // [0, 512 m] — their SHADOWED sun in-scatter (1 RT ray/froxel) — and this
    // per-pixel march owns the remainder [512 m, tMax]: the kilometre-scale god
    // rays the froxel grid blurs to nothing. Same height-fog σ PROFILE, same HG
    // phase, and the camera→x transmittance is the froxel NEAR transmittance ×
    // the closed-form FAR extinction, so the two halves are SEAM-CONSISTENT at
    // 512 m by construction (no brightness step, no double count — the froxel term
    // stops exactly where this one starts). Every step traces an RT shadow ray, so
    // the far shafts are shadowed too (matching the froxel sun at the boundary —
    // this is what Phase 1 deferred: an UNshadowed far term would band at the
    // seam). Runs whenever the medium is present; the old setVolumetricFog opt-in
    // (flags bit 4) is gone. Sky pixels (huge tMax) march the tail the same way.
    if (clouds.heteroActive < 0.5 || clouds.hfDensity <= 0.0 || lights.dirCount == 0u) return vec3(0.0);
    const float tStart = kFroxelZMax;             // the froxel far plane (512 m)
    if (tMax <= tStart) return vec3(0.0);         // surface inside the froxel volume → froxels own it
    const float hgG       = fog.anisotropy;// medium HG g — set unconditionally by updateFogUbo (matches froxel_inject's hetero g)
    const vec3  medAlbedo = (fog.enabled > 0.5) ? fog.color : vec3(1.0);
    const float H         = max(clouds.hfFalloff, 1e-3);
    // camera→seam transmittance (the froxel LUT carried the near [0, 512 m] σ
    // front-to-back); the [512 m, x] remainder rides the closed-form integral.
    const vec2  uv    = (vec2(px) + 0.5) / vec2(float(pc.width), float(pc.height));
    const vec3  seam  = ro + rd * tStart;
    const float Tnear = froxelTransmittance(uv, tStart);
    // Cap the march where the tail transmittance is essentially extinct so a sky
    // pixel's huge tMax doesn't spend steps on fully-attenuated distance. σ at the
    // seam is a good scale for the roughly-constant far density.
    const float sigmaSeam = max(clouds.hfDensity * exp(-max(seam.y - clouds.hfBaseY, 0.0) / H), 1e-6);
    const float tEnd = min(tMax, tStart + 6.0 / sigmaSeam);
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
        // near-field froxel concern; the far tail wants the mean). Same profile as
        // heightFogOpticalDepth / mediumExtinction.
        const float sigmaX = clouds.hfDensity * exp(-max(x.y - clouds.hfBaseY, 0.0) / H);
        if (sigmaX <= 1e-7) continue;
        const float Tfar  = exp(-heightFogOpticalDepth(seam, x));// [512 m, x] closed form
        const float trCam = Tnear * Tfar;                        // camera → x
        if (trCam < 0.003) break;                                // remaining contribution negligible
        vec3 stepSum = vec3(0.0);
        for (uint i = 0u; i < lights.dirCount; ++i) {
            const vec3  L   = normalize(lights.dirLights[i].direction);
            const float vis = shadowVis(x, L, 1e30);// tree/terrain shaft occlusion
            if (vis <= 0.0) continue;
            const float phase = hgPhase(dot(L, rd), hgG);// forward glow toward the sun
            stepSum += lights.dirLights[i].color * (phase * vis);
        }
        // σ_s = σ_t · albedo (albedo applied once below); cloud shadow overhead
        // dims the far shafts exactly like the froxel sun term.
        sum += stepSum * (trCam * sigmaX * cloudShadowSample(x));
    }
    return sum * medAlbedo * dt;
}

// Fogged sky/background colour along this pixel's view ray — the term dispatch A
// blends at MSAA complex pixels for the SKY-minority coverage (and the full sky
// path uses the same expression). Volumetric in-scatter is NOT included: that is
// a per-pixel view-ray integral added once, at full weight, by the caller.
vec3 skyBackground(vec2 ndc) {
    const vec4 tVS   = cam.projInverse * vec4(ndc, 1.0, 1.0);
    const vec3 dirVS = normalize(tVS.xyz / tVS.w);
    const vec3 dirWS = normalize((cam.viewInverse * vec4(dirVS, 0.0)).xyz);
    return applySkyFog(sampleEnvLod(dirWS, 0.0) + proceduralStars(dirWS), dirWS);
}
