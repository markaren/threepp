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
// the raster-side approximation of the PT's true volumetric scattering (which
// also fogs reflected/refracted sub-paths; here only the camera→surface leg is
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
vec3 applySceneFog(vec3 col, vec3 ro, vec3 hit) {
    if (fog.enabled < 0.5) return col;
    const float d = fogPathLength(ro, hit);
    if (d <= 0.0) return col;
    const vec3 tr = exp(-fog.sigmaT * d);
    // In-scatter = fog ALBEDO × an ambient-light estimate (env mean + scene
    // ambient). The fog colour is a single-scattering albedo in this engine
    // (the PT lights the medium with real NEE) — mixing toward it directly,
    // three.js-style, makes fog GLOW in dark scenes (a black Cornell room
    // washed out to white). Spot-light in-scatter is added separately by the
    // fog-driven volumetricSpotScatter march (the god rays).
    const vec3 fogLight = lights.ambient
                        + sampleEnvLod(vec3(0.0, 1.0, 0.0), float(max(pc.envMipCount, 1u) - 1u));
    return col * tr + fog.color * fogLight * (vec3(1.0) - tr);
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
vec3 applySkyFog(vec3 sky, vec3 dir) {
    if (fog.enabled < 0.5) return sky;
    const float sigma = max(dot(fog.sigmaT, vec3(1.0 / 3.0)), 0.0);
    if (sigma <= 0.0) return sky;
    const vec3  up       = (dot(fog.worldUp, fog.worldUp) > 1e-6) ? normalize(fog.worldUp) : vec3(0.0, 1.0, 0.0);
    const float elev     = clamp(dot(dir, up), 0.0, 1.0);                 // 0 horizon .. 1 zenith
    const float band     = exp(-elev * 2.5);                              // haze concentrated near the horizon
    const float strength = 1.0 - exp(-sigma * 45.0);                      // density → how much sky is obscured
    const vec3  fogLight = lights.ambient + sampleEnvLod(up, float(max(pc.envMipCount, 1u) - 1u));
    return mix(sky, fog.color * fogLight, band * strength);
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
    if (fog.enabled < 0.5 || (pc.flags & 16u) == 0u || lights.dirCount == 0u) return vec3(0.0);
    const float sigma = max(dot(fog.sigmaT, vec3(1.0 / 3.0)), 0.0);
    if (sigma <= 0.0) return vec3(0.0);
    const float hgG       = fog.anisotropy;
    const vec3  medAlbedo = fog.color;
    // Cap the march where transmittance is essentially extinct (σ·t ≈ 4) so a sky
    // pixel's huge tMax doesn't spend steps on fully-attenuated distance.
    const float march = min(tMax, 4.0 / sigma);
    if (march <= 0.0) return vec3(0.0);

    const int   STEPS  = 16;
    uint        seed   = pcgHash(uint(px.x) * 7919u + pcgHash(uint(px.y) * 104729u + pc.frame * 6271u));
    const float jitter = rnd(seed);
    const float dt     = march / float(STEPS);

    vec3 sum = vec3(0.0);
    for (int s = 0; s < STEPS; ++s) {
        const float t     = (float(s) + jitter) * dt;
        const float trCam = exp(-sigma * t);          // camera→x transmittance
        if (trCam < 0.003) break;                     // remaining contribution negligible
        const vec3 x = ro + rd * t;
        for (uint i = 0u; i < lights.dirCount; ++i) {
            const vec3  L     = normalize(lights.dirLights[i].direction);
            const float vis   = shadowVis(x, L, 1e30); // tree/terrain shaft occlusion
            if (vis <= 0.0) continue;
            const float phase = hgPhase(dot(L, rd), hgG);// forward glow toward the sun
            sum += lights.dirLights[i].color * (phase * vis * trCam);
        }
    }
    // σ_s = σ_t · albedo: the medium tint scales the in-scattered radiance.
    return sum * medAlbedo * (sigma * dt);
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
