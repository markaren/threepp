// Split from deferred_shade.comp: froxel in-scatter sample, reflection
// reproject/SVGF temporal, BRDF helpers, sheen,
// thin-film iridescence, shadow-ray helpers, per-light Cook-Torrance eval,
// RNG + value-noise/fBm, foam bicubic reconstruction, and emissive NEE
// (diffuse + specular).

// Integrated volumetric in-scatter for the camera→(uv, viewDist) leg — ONE
// trilinear LUT sample replaces the per-pixel volumetric marches (see the
// binding-53 note). flags bit 8 = the froxel passes ran this frame.
vec3 froxelInscatter(vec2 fuv, float viewDist) {
    if ((pc.flags & 256u) == 0u) return vec3(0.0);
    const float t = log(max(viewDist, kFroxelZMin) / kFroxelZMin)
                  / log(kFroxelZMax / kFroxelZMin);
    // Texel k holds the cumulative integral to slice k's FAR edge
    // (froxel_integrate stores AFTER folding slice k), so the unshifted
    // normalized read landed half a slice DEEP — a deterministic
    // 2048^(0.5/64) ≈ 6.1% over-integration of every volumetric leg. Shift
    // back half a texel. 64.0 = kFroxelZ — KEEP IN SYNC with froxel_integrate.
    const float w = max(clamp(t, 0.0, 1.0) * 64.0 - 0.5, 0.0) / 64.0;
    return texture(froxelLutTex, vec3(fuv, w)).rgb;
}
// Cloud shadow transmittance at a world point (bilinear from the top-down
// cloud shadow map, binding 65). 1.0 (full sun) when clouds/shadows are off or
// the point is outside the 8 km camera-centred square. KEEP the extent + centre
// in sync with cloud_shadow.comp.
float cloudShadowSample(vec3 worldPos) {
    if (clouds.shadowActive < 0.5) return 1.0;
    const float halfExt = 4000.0;// == kCloudShadowHalfExt
    const vec2  cenXZ   = cam.viewInverse[3].xz;
    const vec2  suv     = (worldPos.xz - cenXZ) / (2.0 * halfExt) + 0.5;
    if (any(lessThan(suv, vec2(0.0))) || any(greaterThan(suv, vec2(1.0)))) return 1.0;
    return texture(cloudShadowTex, suv).r;
}
// Integrated PARTICLE transmittance (LUT .a) for the same leg — exp(-∫σ_dust)
// over camera→(uv, viewDist), folded front-to-back by froxel_integrate from
// the ParticleField density volumes (plan §3.3). 1.0 when the froxels didn't
// run, and 1.0 everywhere when no field has a density representation, so the
// surface path this feeds is an exact no-op on every scene without dust.
//
// The channel used to carry the medium's TOTAL transmittance and was read by
// nobody: heterogeneous surface extinction is the closed-form height-fog
// integral (applyHeteroSurfaceFog says why), and dust — neither smooth nor
// analytic — is the term that genuinely wants the LUT's per-slice walk.
float froxelParticleTransmittance(vec2 fuv, float viewDist) {
    if ((pc.flags & 256u) == 0u) return 1.0;
    const float t = log(max(viewDist, kFroxelZMin) / kFroxelZMin)
                  / log(kFroxelZMax / kFroxelZMin);
    // Same half-texel shift as froxelInscatter above (texel = slice FAR edge).
    const float w = max(clamp(t, 0.0, 1.0) * 64.0 - 0.5, 0.0) / 64.0;
    return texture(froxelLutTex, vec3(fuv, w)).a;
}

// ReBLUR-style TEMPORAL accumulation for the reflection (+ glass). Adaptive blend
// α = 1/historyLength → fast convergence early, then a long stable mean. Tracks the
// luminance 2nd moment E[L²] so the channel-1 denoise can form temporal variance
// (var = E[L²] − E[L]²) and size its blur by it. Writes the AUX buffer (binding 31:
// .x=history, .y=E[L²]). The accumulated RADIANCE is the return value (→ reflectImage).
//   RESET (history→1) when: off-screen reproject, surface mismatch (disocclusion), OR
//   camera MOTION — the reflection is VIEW-DEPENDENT, so a surface-motion reproject can't
//   follow the reflected content (it would ghost); on reset the spatial denoise blurs the
//   fresh 1-sample wide (history=1 ⇒ wide), and on a held-still view history climbs and
//   the blur narrows back to the converged (sharp) lobe. This is why it's pretty AND
//   stable where the fixed-α version was only one or the other.
// Shared reflection-channel REPROJECTION + history policy. Two callers — the
// temporal accumulation below AND the checkerboard skip decision made BEFORE the
// reflection trace — one logic, so the "may I hold history?" answer can't drift
// from the "how do I blend history?" answer.
// ROUGHNESS-AWARE motion policy: the old blanket reset (any camera motion →
// history=1) starved exactly the surfaces that need history most: a ROUGH lobe
// integrates a wide solid angle, so its reflection is nearly VIEW-INDEPENDENT
// and survives a surface reproject — like diffuse GI, which reprojects under
// motion with a short cap. Only near-MIRROR content actually shifts with the
// camera (a surface reproject can't follow it → ghost), so only that resets.
// In between, the cap blends: motion-scaled (24→6, the GI channel's pattern)
// for rough, down to 1 (reset) as roughness → mirror. Thresholds use a
// view-dependence factor (1 - smoothstep(0.05, 0.30, roughness)).
bool reflReproject(vec2 uv, vec3 N, float rough, float viewDist, out vec2 pUv, out float histCap) {
    const vec2 mv   = texture(gbufMotionTex, paneToPhys(gbufMotionTex, uv)).rg;
    const vec2 pNdc = vec2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0)) + mv;
    pUv = vec2(pNdc.x * 0.5 + 0.5, 0.5 - pNdc.y * 0.5);
    bool valid = all(greaterThanEqual(pUv, vec2(0.0))) && all(lessThanEqual(pUv, vec2(1.0)));
    if (valid) {
        const vec3 pn = texture(gbufNormalPrevTex, paneToPhys(gbufNormalPrevTex, pUv)).xyz * 2.0 - 1.0;
        // DEPTH-discontinuity disocclusion (mirrors the GI reproject below, ~line
        // 2630 — the reason GI doesn't smear at silhouettes but the reflection did).
        // The reflection channel did a NORMAL-only match (dot>0.9); at a car-body
        // SILHOUETTE under camera ROTATION the bilinear prev-normal tap stays inside
        // that cone, so stale reflection from ACROSS the edge bled into the temporal
        // history → "edges smear when turning". motion .b carries THIS surface's prev
        // NDC depth; gbufDepthPrevTex at pUv is the prev depth BUFFER there — a
        // mismatch means the reproject crossed the silhouette onto a DIFFERENT surface
        // ⇒ reject ⇒ the edge resets to a fresh (noisy, un-smeared) reflection. Normal
        // loosened to 0.7 to match the GI (depth is now the primary disocclusion gate).
        const float surfD = texture(gbufMotionTex, paneToPhys(gbufMotionTex, uv)).b;     // this surface's prev NDC depth
        const float bufD  = texture(gbufDepthPrevTex, paneToPhys(gbufDepthPrevTex, pUv)).x; // prev depth buffer at the reproject
        const vec4  svh   = cam.projInverse * vec4(pNdc, surfD, 1.0);
        const vec4  bvh   = cam.projInverse * vec4(pNdc, bufD,  1.0);
        const float svz   = svh.z / svh.w;
        valid = dot(pn, pn) > 1e-6 && dot(N, normalize(pn)) > 0.7 &&
                abs(svz - bvh.z / bvh.w) <= 0.1 * max(abs(svz), 1e-3);// same-surface depth gate
        // MOVING-MESH TRAILING-EDGE GUARD (mirrors the GI reproject's): ground a
        // moving glossy object just vacated reprojects INTO the object's bright
        // reflection pixels — the normal (car roof + road both face up) and
        // relative-depth (10% of view-Z > car height at driving distances) gates
        // both pass — so the object's reflection enters the ground's history and
        // trails it as a wake while the silhouette re-feeds it every frame. The
        // ID compare is exact where those gates are blind. Compares the STABLE
        // per-object id (.y) + the prev texel's own moved-sticky bit (.z,
        // kInstFlagMoving) — identity-stable across the entry-list renumber a
        // terrain tile split/merge causes, unlike prev ids .x, which indexes the
        // PREV draw list and made geoms[pid-1] read the wrong (or an out-of-
        // range) entry for the whole streaming burst. Static scenes (tile
        // seams) never false-reset: no moved bit, guard inert.
        // All 4 texels of the bilinear history footprint are checked.
        if (valid) {
            const vec2  sz  = vec2(float(pc.width), float(pc.height));
            const uint  sid = texelFetch(gbufIdsTex, ivec2(uv * sz), 0).y;
            const ivec2 mxP = ivec2(int(pc.width) - 1, int(pc.height) - 1);
            const ivec2 fb  = ivec2(floor(pUv * sz - 0.5));
            // Object motion OR camera translation — see the trailing-edge
            // guard in deferred_shade.comp: parallax past a static occluder
            // has no moved bit, and with identity-stable ids the mismatch is
            // a true surface change whenever the camera moved.
            const bool camMovedR = pc.camDelta > 0.001;
            for (int dy = 0; dy <= 1 && valid; ++dy)
                for (int dx = 0; dx <= 1 && valid; ++dx) {
                    const uvec4 pids = texelFetch(gbufIdsPrevTex, clamp(fb + ivec2(dx, dy), ivec2(0), mxP), 0);
                    // Prev-sky texels carry stale last-surface history — reject
                    // under camera motion (see the trailing-edge guard note in
                    // deferred_shade.comp).
                    const bool skyStale = pids.x == 0u && camMovedR;
                    const bool objChange = pids.x > 0u && pids.y != sid && (ifMoving(pids.z) || camMovedR);
                    if (skyStale || objChange) valid = false;
                }
        }
    }
    // Static-view cap: 512 (was 32). The reflection ray is GGX-sampled with a
    // per-frame seed, so a held-still view keeps integrating fresh stochastic
    // samples; a 32-frame mean leaves sigma/sqrt(32) residual boil that reads
    // as "steel never settles" on high-contrast (sun/env) reflections. Motion
    // caps below are unchanged — this only affects a stationary camera.
    histCap = 512.0;
    // Motion = screen-space surface motion OR camera WORLD motion. The latter
    // catches the CHASE-CAM case: a surface tracked by the camera (car sunroof,
    // followed boat) is screen-stationary — mv ≈ 0 — while every meter of
    // camera travel slides its view-dependent reflection content. Translation
    // is converted to an equivalent NDC shift via the parallax of content at
    // ~the surface's own distance (the best single-fetch proxy for reflected-
    // content distance); rotation maps through the FOV scale (~1.7 NDC/rad at
    // 60°). Pure camera rotation also moves the surface on screen, so max()
    // (not sum) keeps it from double-counting.
    const float camNdc = pc.camRot * 1.7 + 1.7 * pc.camDelta / max(viewDist, 0.25);
    const float mvLen = max(length(mv), camNdc);
    if (mvLen > 0.004) {// above jitter
        const float viewDepT  = 1.0 - smoothstep(0.05, 0.30, rough);// 1 = mirror-like
        const float motionCap = mix(24.0, 6.0, smoothstep(0.004, 0.03, mvLen));
        histCap = mix(motionCap, 1.0, viewDepT);
        if (histCap < 1.5) valid = false;// near-mirror: hard reset (view-dependent)
    }
    return valid;
}
// hitT: first-bounce reflected-content distance for THIS frame's sample —
// > 0 = geometry hit at that distance, 0 = env miss (radiance already
// lobe-filtered via missLod), < 0 = unknown (glass) → the denoiser falls
// back to the conservative full-width gloss blur. EMA'd into aux .w so the
// denoiser's footprint doesn't flicker with per-frame lobe samples.
vec4 reflSVGFTemporal(vec4 cur, ivec2 px, vec2 uv, vec3 N, float viewDist, bool hold, float hitT, bool hitMoved) {
    const float curLum = dot(cur.rgb, vec3(0.2126, 0.7152, 0.0722));
    const float rough  = abs(cur.a);// .a < 0 = glass marker (frost gr) — same policy
    const float hitEnc = hitT < 0.0 ? -1.0 : min(hitT, 1e4);// rgba16f-safe
    vec2 pUv; float histCap;
    bool valid = reflReproject(uv, N, rough, viewDist, pUv, histCap);
    // MOVING REFLECTED CONTENT — the reflected hit is a moving mesh (its content
    // slides frame-to-frame while the reflecting surface reproject tracks the
    // surface, not the content). Near-MIRROR: HARD-reset history so the moving
    // reflection is re-estimated fresh each frame (the roughness-scaled spatial
    // blur is the gloss); no accumulation = no ghost comb. ROUGH: the lobe
    // integrates a wide solid angle, so a frame's worth of content motion barely
    // changes it — the same argument the camera-motion policy above makes — and
    // a 1-sample restart every frame of a 1-spp GGX lobe is permanent grain
    // (a deforming chrome shell on a roughness-0.4 steel bed: speckled bed for
    // as long as the shell moved). Short cap instead (6, the fast-motion cap)
    // and let the content-change antilag below catch genuine jumps.
    if (hitMoved) {
        const float viewDepT = 1.0 - smoothstep(0.05, 0.30, rough);// 1 = mirror-like
        histCap = mix(min(histCap, 6.0), 1.0, viewDepT);
        if (histCap < 1.5) valid = false;
    }
    const vec4 prevR = valid ? texture(reflectPrevTex, paneToPhys(reflectPrevTex, pUv)) : vec4(0.0);
    if (valid && any(greaterThan(abs(prevR.rgb), vec3(1e6)))) valid = false;// garbage guard
    if (hold && valid) {
        // Checkerboard SKIP frame: no new sample — carry the history through
        // UNCHANGED (no histLen increment, no blend, trend carried), the GI
        // channel's giSkip pattern. Freezing histLen keeps the blend alpha tied
        // to the number of REAL samples, so the running mean stays unbiased at
        // half rate.
        const vec4 pa = texture(reflAuxPrevTex, paneToPhys(reflAuxPrevTex, pUv));
        imageStore(reflAuxWrite, px, vec4(clamp(pa.x, 1.0, histCap), pa.y, pa.z, pa.w));
        return vec4(prevR.rgb, cur.a);
    }
    float histLen, moment, trend, hitW;
    vec3  accum;
    if (valid) {
        const vec4  pa = texture(reflAuxPrevTex, paneToPhys(reflAuxPrevTex, pUv));
        // CONTENT-CHANGE ANTILAG — fixes "glass/mirror reflections never update
        // under a static camera". The static-512 cap freezes the running mean:
        // CONTENT motion (an object sliding behind glass, or moving in a mirror
        // — no camera motion, no surface motion vectors, no disocclusion) was
        // blended at alpha=1/512 → an ~8-second smear that reads as frozen.
        // Distinguish content change from sampling noise by the SIGN TREND of
        // the deviation (the TAA streak lesson, adapted): jitter/GGX-lobe BOIL
        // ALTERNATES around the converged mean (EMA → 0), real content change
        // pushes the SAME direction every frame (EMA → ±1). On a sustained
        // trend, cut history multiplicatively → the channel re-blends at
        // content rate; once the content settles, the trend decays and history
        // regrows to the full static cap (settling unharmed).
        const float prevLum = dot(prevR.rgb, vec3(0.2126, 0.7152, 0.0722));
        const float sdev = clamp((curLum - prevLum) / (0.05 + 0.25 * max(curLum, prevLum)), -1.0, 1.0);
        trend = mix(pa.z, sdev, 0.34);// ~3-frame EMA, carried in aux .z
        // Smooth trend→cap mapping (a hard threshold + multiplicative cut made a
        // relaxation oscillator out of pixels hovering at the boundary: slash →
        // pop → regrow → slash). Alternating boil EMAs to ~±0.2, DC-biased edge
        // boil reaches ~0.6 (mild cap, settling barely touched); real content
        // change pegs |trend| ≈ 1 → cap ~3 → re-blends at content rate.
        const float trendCap = mix(histCap, 3.0, smoothstep(0.45, 0.9, abs(trend)));
        histLen = max(min(pa.x + 1.0, trendCap), 1.0);
        const float a = 1.0 / histLen;
        accum   = mix(prevR.rgb, cur.rgb, a);
        // ANTI-GHOST temporal clamp (near-mirror only). A car sweeping the reflection
        // of a WORLD-STATIC road leaves stale bright history the reproject can't clear:
        // the departed pixel has ~zero screen motion, so it keeps the 512-frame static
        // cap, and the current ray no longer hits the moving car (hitMoved can't reset
        // it). Its moderate brightness (~2× road) stays under the trend antilag's
        // magnitude threshold, so it lingers seconds as a comb trail (the "render the
        // car from above" ghost). Clamp the accumulated luminance toward THIS frame's
        // reflection: on a near-mirror surface the deterministic ray is STABLE
        // frame-to-frame, so a trailing pixel that now reflects sky pulls its stale
        // bright history down to the current (dark) sample in ~1 frame, while a pixel
        // genuinely reflecting the car keeps its (bright) current sample as the ceiling
        // and a settled surface has accum≈cur so the clamp never binds. Restricted to
        // low roughness because a wide GGX lobe's 1-spp sample is too noisy to clamp
        // against (a dark-miss frame would wrongly dim a converged reflection) — and
        // rough asphalt reflects nothing, so the trail is a glossy-surface artifact
        // to begin with.
        if (rough < 0.22) {
            const float accLum  = dot(accum, vec3(0.2126, 0.7152, 0.0722));
            const float lumCeil = curLum * 1.4 + 0.02;
            if (accLum > lumCeil) accum *= lumCeil / accLum;
        }
        moment  = mix(pa.y, curLum * curLum, a);
        // Hit-distance EMA (floor 0.25: content distance changes matter within a
        // few frames — a slow 1/512 blend would size the gloss blur from stale
        // geometry). Unknown current sample (< 0) carries the history through.
        hitW    = hitEnc < 0.0 ? pa.w
                                : (pa.w < 0.0 ? hitEnc : mix(pa.w, hitEnc, max(a, 0.25)));
    } else {
        histLen = 1.0; accum = cur.rgb; moment = curLum * curLum; trend = 0.0;
        hitW    = hitEnc;
    }
    imageStore(reflAuxWrite, px, vec4(histLen, moment, trend, hitW));
    return vec4(accum, cur.a);
}

// ── BRDF helpers (verbatim from shade_common.glsl) ──────────────────────────
float distGGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}
float geomSmithG1(float NdotX, float k) { return NdotX / (NdotX * (1.0 - k) + k); }
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
vec2 envBRDFApprox(float NdotV, float r) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    const vec4 v  = r * c0 + c1;
    const float a004 = min(v.x * v.x, exp2(-9.28 * NdotV)) * v.x + v.y;
    return vec2(-1.04, 1.04) * a004 + v.zw;
}
vec3 sampleEnvLod(vec3 dir, float lod) {
    const float u = 0.5 + atan(dir.z, dir.x) / TWO_PI;
    const float v = 0.5 + asin(clamp(dir.y, -1.0, 1.0)) / PI;
    return textureLod(envTex, vec2(u, v), lod).rgb;
}

// HemisphereLight surface term — the zero-mean directional remainder of
// mix(ground, sky, 0.5*dot(N,up)+0.5). The CPU traversal (updateLightsUbo)
// folds each hemi's angular mean into lights.ambient, so every isotropic
// ambient consumer already carries the hemi's energy; add THIS wherever
// lights.ambient feeds a surface diffuse term with a known normal, gated
// exactly like the ambient it rides with. ambient + hemiAmbient(N) ≥ 0
// per hemi (it reconstructs the mix), and all-zero rows (no hemi in the
// scene) make it an exact no-op.
vec3 hemiAmbient(vec3 N) {
    return vec3(dot(lights.hemiDeltaR, N),
                dot(lights.hemiDeltaG, N),
                dot(lights.hemiDeltaB, N));
}

// ── Sheen (KHR_materials_sheen). The deferred base BRDF omitted
// this, so satin/velvet/fabric read flat.
float D_Charlie(float NdotH, float roughness) {
    const float invAlpha = 1.0 / max(roughness * roughness, 1e-4);
    const float sin2h    = max(1.0 - NdotH * NdotH, 1e-7);
    return (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / (2.0 * PI);
}
float V_Neubelt(float NdotV, float NdotL) {
    return clamp(1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV)), 0.0, 1.0);
}
float IBLSheenBRDF(float dotNV, float roughness) {
    const float r2 = roughness * roughness;
    const float a = roughness < 0.25 ? -339.2 * r2 + 161.4 * roughness - 25.9
                                     :   -8.48 * r2 +  14.3 * roughness -  9.95;
    const float b = roughness < 0.25 ?   44.0 * r2 -  23.7 * roughness +  3.26
                                     :    1.97 * r2 -   3.27 * roughness +  0.72;
    const float DG = exp(a * dotNV + b) + (roughness < 0.25 ? 0.0 : 0.1 * (roughness - 0.25));
    return clamp(DG / PI, 0.0, 1.0);
}

// ── Thin-film iridescence (KHR_materials_iridescence, Belcour & Barla 2017).
// The deferred base BRDF omitted
// this, so soap-film / oil-slick / nacre F0 read as plain dielectric. Modulates
// dielectric F0 with wavelength-dependent thin-film interference; the lobe shape
// (GGX) is unchanged, only the Fresnel base shifts per channel.
vec3 iridFresnel0ToIor(vec3 F0) {
    vec3 sqrtF0 = sqrt(F0);
    return (vec3(1.0) + sqrtF0) / (vec3(1.0) - sqrtF0);
}
vec3 iridIorToFresnel0_v(vec3 transmittedIor, float incidentIor) {
    return pow((transmittedIor - vec3(incidentIor)) / (transmittedIor + vec3(incidentIor)), vec3(2.0));
}
float iridIorToFresnel0_s(float transmittedIor, float incidentIor) {
    return pow((transmittedIor - incidentIor) / (transmittedIor + incidentIor), 2.0);
}
vec3 iridSensitivity(float OPD, vec3 shift) {
    float phase = 2.0 * PI * OPD * 1.0e-9;
    vec3 val = vec3(5.4856e-13, 4.4201e-13, 5.2481e-13);
    vec3 pos = vec3(1.6810e+06, 1.7953e+06, 2.2084e+06);
    vec3 vr  = vec3(4.3278e+09, 9.3046e+09, 6.6121e+09);
    vec3 xyz = val * sqrt(2.0 * PI * vr) * cos(pos * phase + shift) * exp(-(phase*phase) * vr);
    xyz.x   += 9.7470e-14 * sqrt(2.0 * PI * 4.5282e+09) * cos(2.2399e+06 * phase + shift.x) * exp(-4.5282e+09 * (phase*phase));
    xyz     /= 1.0685e-7;
    // XYZ → linear sRGB (D65)
    mat3 XYZ_TO_REC709 = mat3(
         3.2404542, -0.9692660,  0.0556434,
        -1.5371385,  1.8760108, -0.2040259,
        -0.4985314,  0.0415560,  1.0572252);
    return XYZ_TO_REC709 * xyz;
}
vec3 evalIridescence(float outsideIOR, float eta2, float cosTheta1,
                     float thinFilmThickness, vec3 baseF0) {
    // Force iridescenceIOR -> outsideIOR when thinFilmThickness == 0 → reduces to base F0.
    float iridescenceIor = mix(outsideIOR, eta2, smoothstep(0.0, 0.03, thinFilmThickness));
    // Snell to the angle inside the film.
    float sinTheta2Sq = (outsideIOR / iridescenceIor) * (outsideIOR / iridescenceIor) *
                       (1.0 - cosTheta1 * cosTheta1);
    float cosTheta2Sq = 1.0 - sinTheta2Sq;
    if (cosTheta2Sq < 0.0) return vec3(1.0);// total internal reflection at the film top
    float cosTheta2 = sqrt(cosTheta2Sq);
    // First interface (above the film).
    float R0   = iridIorToFresnel0_s(iridescenceIor, outsideIOR);
    float R12  = R0 + (1.0 - R0) * pow(1.0 - cosTheta1, 5.0);
    float T121 = 1.0 - R12;
    float phi12 = 0.0;
    if (iridescenceIor < outsideIOR) phi12 = PI;
    float phi21 = PI - phi12;
    // Second interface (substrate). Recover the substrate's IOR from baseF0.
    vec3 baseIOR = iridFresnel0ToIor(clamp(baseF0, vec3(0.0), vec3(0.9999)));
    vec3 R1      = iridIorToFresnel0_v(baseIOR, iridescenceIor);
    vec3 R23     = R1 + (vec3(1.0) - R1) * pow(1.0 - cosTheta2, 5.0);
    vec3 phi23   = vec3(0.0);
    if (baseIOR.x < iridescenceIor) phi23.x = PI;
    if (baseIOR.y < iridescenceIor) phi23.y = PI;
    if (baseIOR.z < iridescenceIor) phi23.z = PI;
    // Optical path difference and phase.
    float OPD     = 2.0 * iridescenceIor * thinFilmThickness * cosTheta2;
    vec3  phi     = vec3(phi21) + phi23;
    // Compound reflectance — exact formulas from Belcour 2017 §3.
    vec3  R123    = clamp(R12 * R23, vec3(1e-5), vec3(0.9999));
    vec3  r123    = sqrt(R123);
    vec3  Rs      = (T121 * T121) * R23 / (vec3(1.0) - R123);
    // First-order Fourier (m=0) — base reflectance.
    vec3 C0 = R12 + Rs;
    vec3 I  = C0;
    // Higher-order Fourier terms: spectral cosine integrals (m=1,2).
    vec3 Cm = Rs - T121;
    for (int m = 1; m <= 2; ++m) {
        Cm     *= r123;
        vec3 Sm = 2.0 * iridSensitivity(float(m) * OPD, float(m) * phi);
        I      += Cm * Sm;
    }
    return max(I, vec3(0.0));
}

// Interpolated UV at a hit (lean — UV only, for the alpha-tested shadow test).
// Mirrors fetchHit's index + barycentric interpolation.
vec2 fetchUvAt(int instIdx, int primId, vec2 bary) {
    const GeometryDesc g = geoms[instIdx];
    if (g.uvAddress == 0ul) return vec2(0.0);
    const uvec3 idx = gfetchTri(g, primId);
    // Packed-aware (unorm16x2 on packed static geometry — see packedAttrs).
    const vec2 u0 = gfetchUv(g, idx.x);
    const vec2 u1 = gfetchUv(g, idx.y);
    const vec2 u2 = gfetchUv(g, idx.z);
    return (1.0 - bary.x - bary.y) * u0 + bary.x * u1 + bary.y * u2;
}

// Does a shadow-ray candidate hit actually OCCLUDE? Policy mirrors the RT
// shadow_anyhit.rahit so the deferred and ray-traced backends agree:
//   transmission > 0 (glass / alpha-blend transparency / additive) → never blocks
//   cutoff < 0 (blend)  → never blocks (e.g. textured decals — light passes through)
//   cutoff > 0 (cutout) → occludes iff alpha ≥ cutoff (foliage, grids)
//   opaque              → occludes
bool shadowOccludes(int instIdx, int primId, vec2 bary) {
    const MaterialDesc m = mats[instIdx];
    // STRONG emitters never occlude — an analytic light that sits INSIDE its
    // own glowing housing (a lighthouse lamp room) must shine through it. Same
    // "a light never shadows itself" rule emissiveNEE/restirOccluded apply to
    // emissive-triangle lights; threshold 1.0 keeps faintly-emissive props
    // (glowing screens etc.) casting normal shadows.
    const vec3 em = m.emissive * m.emissiveIntensity;
    if (max(max(em.r, em.g), em.b) >= 1.0) return false;
    // Transmissive surfaces — glass, alpha-blend transparency (host encodes
    // opacity<1 as transmission=1-opacity, ior=1) and additive blends
    // (transmission>1) — do NOT cast a solid shadow. Mirrors shadow_anyhit.rahit,
    // which ignores these candidate hits so direct light passes through. Without
    // this the deferred path paints opaque shadows behind transparent materials.
    if (m.transmission > 0.0) return false;
    // Alpha-blend (host sets alphaCutoff < 0: transparent=true + texture, opacity=1).
    // Like shadow_anyhit.rahit's isBlend branch these do NOT block light, so blend
    // decals (bullet-scorch splats etc.) stop painting a hard shadow halo that the
    // GL backend never produces — decals default to castShadow=false there.
    if (m.alphaCutoff < 0.0) return false;
    if (m.albedoTexIndex < 0 || m.alphaCutoff == 0.0) return true;// opaque
    // Alpha-cutout (alphaCutoff > 0): occlude only where the texel is solid.
    const vec2  uv = fetchUvAt(instIdx, primId, bary);
    const int   ti = clamp(m.albedoTexIndex, 0, int(kMaxMaterialTextures) - 1);
    const float a  = textureLod(albedoMaps[nonuniformEXT(ti)], (m.uvTransform * vec3(uv, 1.0)).xy, 0.0).a;
    return a >= m.alphaCutoff;
}

// Inline hard-shadow test (TerminateOnFirstHit). 1.0 lit / 0.0 occluded.
// NOT forced-opaque: geometry is non-opaque in the BLAS, so alpha-tested casters
// (foliage, grids, cutout decals) yield CANDIDATES we alpha-test here — otherwise
// they'd cast a solid silhouette. Opaque candidates confirm on the first hit (one
// buffer read), so the cost on solid geometry stays negligible.
// kRayMaskOpaque culls blend/transmissive instances in HW — shadowOccludes
// would ignore those candidates anyway (transmission>0 / alphaCutoff<0 never
// block), so this is a pure traversal saving; water stays opaque-mask and is
// still ignored by the material test.
float shadowVis(vec3 origin, vec3 dir, float tMax) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topAS, gl_RayFlagsTerminateOnFirstHitEXT,
                          kRayMaskOpaque, origin, 0.0, dir, tMax);
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT &&
            shadowOccludes(rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false),
                           rayQueryGetIntersectionPrimitiveIndexEXT(rq, false),
                           rayQueryGetIntersectionBarycentricsEXT(rq, false)))
            rayQueryConfirmIntersectionEXT(rq);
    }
    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionNoneEXT)
        return 1.0;
    // MOVING OCCLUDER flag for the denoised shadow channel: this shadow lands on
    // the receiver from a mesh that is currently moving (GeometryDesc.flags bit 0),
    // so the shadow SWEEPS the receiver and its history must stay short — the
    // trend/σ-step antilags key off history statistics that the sweep itself
    // degrades (the à-trous feedback rewrites the mean but not E[R²], inflating σ
    // over the swept band), while this is a direct, statistics-free signal.
    if ((geoms[rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true)].flags & 1u) != 0u)
        gShadowMovingOccluder = true;
    return 0.0;
}

// Cook-Torrance specular + Lambert diffuse for one analytic light direction L.
vec3 evalLight(vec3 N, vec3 V, vec3 L, float NdotV, vec3 F0, vec3 albedo,
               float roughness, float metalness, float k,
               vec3 sheenColor, float sheenRoughness) {
    const float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return vec3(0.0);
    const vec3  H     = normalize(V + L);
    const float NdotH = max(dot(N, H), 0.0);
    const float VdotH = max(dot(V, H), 0.0);
    const vec3  F     = fresnelSchlick(VdotH, F0);
    const float D     = distGGX(NdotH, roughness);
    const float G     = geomSmithG1(NdotV, k) * geomSmithG1(NdotL, k);
    const vec3  spec  = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);
    const vec3  kd    = (vec3(1.0) - F) * (1.0 - metalness);
    const vec3  diff  = kd * albedo / PI;
    // KHR_materials_sheen Charlie lobe (matches shade_common's per-light sheen).
    vec3 sheen = vec3(0.0);
    if (dot(sheenColor, sheenColor) > 0.0)
        sheen = sheenColor * D_Charlie(NdotH, sheenRoughness) * V_Neubelt(NdotV, NdotL);
    return (diff + spec + sheen) * NdotL;
}

// ── RNG + RT ambient occlusion / env GI ─────────────────────────────────────
uint pcgHash(uint v) {
    v = v * 747796405u + 2891336453u;
    const uint s = ((v >> ((v >> 28) + 4u)) ^ v) * 277803737u;
    return (s >> 22) ^ s;
}
float rnd(inout uint seed) {
    seed = seed * 747796405u + 2891336453u;
    uint r = ((seed >> ((seed >> 28) + 4u)) ^ seed) * 277803737u;
    r = (r >> 22) ^ r;
    return float(r) * (1.0 / 4294967296.0);
}

// (sunShadowVis lives after blueNoiseDef — it dithers off the blue-noise tile.)

// ── Value-noise / fBm (verbatim from shade_common.glsl) — foam shading only ──
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float vnoise21(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm4(vec2 p) {
    float a = 0.0;
    float w = 0.5;
    for (int i = 0; i < 4; ++i) {
        a += w * vnoise21(p);
        p *= 2.03;
        w *= 0.5;
    }
    return a;
}

// Bicubic B-spline reconstruction of the world foam accumulator via 4
// bilinear taps (Sigg & Hadwiger, GPU Gems 2 ch. 20). Plain bilinear
// renders every isolated foam texel as a soft axis-aligned SQUARE (the
// bilinear tent's support is a square), which reads as a grid artifact up
// close; the cubic kernel reconstructs round, C1-smooth blobs.
float sampleFoamBicubic(vec2 uv) {
    const vec2 res = vec2(textureSize(oceanFoamWorld, 0));
    const vec2 st  = uv * res - 0.5;
    const vec2 i   = floor(st);
    const vec2 f   = st - i;
    const vec2 f2  = f * f;
    const vec2 f3  = f2 * f;
    const vec2 w0  = (-f3 + 3.0 * f2 - 3.0 * f + 1.0) * (1.0 / 6.0);
    const vec2 w1  = (3.0 * f3 - 6.0 * f2 + 4.0)       * (1.0 / 6.0);
    const vec2 w2  = (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) * (1.0 / 6.0);
    const vec2 w3  = f3 * (1.0 / 6.0);
    const vec2 g0  = w0 + w1;
    const vec2 g1  = w2 + w3;
    // Two sample coords per axis, each placed so ONE bilinear fetch
    // integrates a weighted texel pair; REPEAT sampler handles the wrap.
    const vec2 c0  = (i - 0.5 + w1 / g0) / res;
    const vec2 c1  = (i + 1.5 + w3 / g1) / res;
    const float t00 = textureLod(oceanFoamWorld, vec2(c0.x, c0.y), 0.0).r;
    const float t10 = textureLod(oceanFoamWorld, vec2(c1.x, c0.y), 0.0).r;
    const float t01 = textureLod(oceanFoamWorld, vec2(c0.x, c1.y), 0.0).r;
    const float t11 = textureLod(oceanFoamWorld, vec2(c1.x, c1.y), 0.0).r;
    return g0.y * (g0.x * t00 + g1.x * t10) + g1.y * (g0.x * t01 + g1.x * t11);
}

// Emissive-mesh NEE: directly sample the emissive-triangle area lights (power
// CDF), shadow-test, add the contribution. Clean (samples the emitter, not a
// random hemisphere), so an emissive sphere actually illuminates the room.
//
// DETERMINISTIC + COHERENT across pixels (like gatherEnv) — NO per-pixel
// rnd(seed). The old random version gave per-pixel speckle that ONLY temporal
// accumulation could hide, so on the deterministic deferred path (jitter off,
// FXAA) any scene with emissives was "crazy noisy". Coherent area-light shading
// is correct (diffuse contribution is a smooth function of position — no "copies"
// like a mirror reflection would have). Sample count now controls emitter
// COVERAGE (bias), not noise: stratified power-CDF picks span the emitters ∝
// power; a low-discrepancy (R2) barycentric covers each picked triangle's area.
// Settles instantly, no denoiser. (`seed` kept for ABI; unused.)
// The pick itself — and the per-light COVERAGE mode that resolves several small
// emitters (the sailboat's nav lights) — now lives in emissive_lights.glsl.
vec3 emissiveNEE(int EM_SAMPLES, vec3 P, vec3 N, vec3 V, float NdotV, vec3 F0, vec3 albedo,
                 float roughness, float metalness, float k, inout uint seed, bool doShadows) {
    if (pc.emissiveCount == 0u || pc.emissiveTotalPower <= 0.0) return vec3(0.0);
    // EM_SAMPLES: 16 at primaries, small at reflection-bounce hits. The PICK (global
    // strata, or per-light COVERAGE with point proxies for small lights) lives in
    // emissive_lights.glsl; this loop owns the BRDF and the emitter-skipping shadow ray.
    const vec3   shadowOrig = P + N * SHADOW_EPS;
    const EmPlan plan = emPlanBuild(P, EM_SAMPLES);
    vec3 sum = vec3(0.0);
    for (int s = 0; s < plan.count; ++s) {
        EmSample es;
        if (!emPlanSample(plan, s, P, /*twoSided=*/false, es)) continue;
        const vec3 L = es.L;
        if (dot(N, L) <= 0.01) continue;
        if (doShadows) {
            // Shadow ray to the SAME emitter point used for lighting. Non-opaque + SKIP
            // emitters: a light never shadows itself, so the emitter's own far side can't
            // paint a self-occlusion "fence" across lit surfaces; real occluders right up
            // to the light still cast soft shadows (no tMax margin). A point proxy stops
            // at the light's bounding radius (es.backoff).
            const vec3  toLs  = es.lp - shadowOrig;
            const float dists = length(toLs);
            const float tMax  = dists - es.backoff - 1e-2;
            if (tMax > 1e-3) {
                rayQueryEXT rq;
                // kRayMaskOpaque: blend/transmissive surfaces (decals, glass) must
                // not block emissive light — this loop only filters emitters.
                rayQueryInitializeEXT(rq, topAS, gl_RayFlagsTerminateOnFirstHitEXT,
                                      kRayMaskOpaque, shadowOrig, 1e-3, toLs / max(dists, 1e-6), tMax);
                while (rayQueryProceedEXT(rq)) {
                    if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
                        const MaterialDesc hm = mats[rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false)];
                        const vec3 hem = hm.emissive * hm.emissiveIntensity;
                        if (max(max(hem.r, hem.g), hem.b) < 0.05)// not an emitter → real occluder
                            rayQueryConfirmIntersectionEXT(rq);
                    }
                }
                if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT)
                    continue;
            }
        }
        // DIFFUSE-ONLY emitter NEE. Light sampling (sampling the emitter) is the
        // correct, low-variance estimator for the broad DIFFUSE term — but the WRONG
        // one for the peaked SPECULAR lobe: at low roughness a single sample lands
        // near the GGX peak and the firefly clamp leaves a bright "probe speckle"
        // highlight (a proper BRDF-sampling + MIS estimator would never show this).
        // Emitter specular is owned by the reflection ray at near-mirror roughness and by
        // emissiveSpecNEE on rough lobes (the specNEET split in main()).
        const vec3  H  = normalize(V + L);
        const vec3  Fr = fresnelSchlick(max(dot(V, H), 0.0), F0);
        const vec3  kd = (vec3(1.0) - Fr) * (1.0 - metalness);
        vec3 c = (kd * albedo * (1.0 / PI)) * max(dot(N, L), 0.0) * es.Le * es.gw;
        // Firefly clamp: cap a single sample's luminance so a stray spike (grazing
        // emitter / near hit) can't dominate — the dominant noise on animated
        // geometry where TAA can't accumulate it away.
        const float lum = max(max(c.r, c.g), c.b);
        if (lum > pc.fireflyClamp) c *= pc.fireflyClamp / lum;
        sum += c * es.w;
    }
    return sum;
}

// Coherent emitter SPECULAR NEE — the GGX-lobe analogue of emissiveNEE (same
// deterministic stratified power-CDF picks + R2 barycentrics → settles
// instantly, zero per-frame noise, same accepted emitter-coverage bias). For a
// ROUGH reflective surface the stochastic reflection ray hits a small bright
// emitter with probability p ≪ 1 → a firefly-clamped binomial spike train that
// neither the temporal EMA nor the roughness blur can settle (= the constant
// boiling around reflected lamps). For a BROAD lobe, light-sampling is the
// low-variance estimator (the MIS logic): evaluate D·G·F toward the coherent
// emitter set and scale the emitter's self-emission OUT of the stochastic ray
// (gReflEmitterScale) so nothing is double-counted. Near-mirror stays with the
// BRDF ray — a tight lobe either hits the emitter every frame (no variance) or
// never, and NEE of a peaked lobe is the probe-speckle regime that made
// emissiveNEE diffuse-only in the first place.
vec3 emissiveSpecNEE(vec3 P, vec3 N, vec3 V, float NdotV, vec3 F0,
                     float roughness, float k, bool doShadows) {
    if (pc.emissiveCount == 0u || pc.emissiveTotalPower <= 0.0) return vec3(0.0);
    const int    EM_SAMPLES = 8;// was 16 — halves the per-glossy-pixel shadow rays;
                                 // still deterministic (zero temporal noise), slightly
                                 // coarser emitter coverage on wide-lobe metals.
    const vec3   shadowOrig = P + N * SHADOW_EPS;
    const EmPlan plan = emPlanBuild(P, EM_SAMPLES);// same pick as emissiveNEE (emissive_lights.glsl)
    vec3 sum = vec3(0.0);
    for (int s = 0; s < plan.count; ++s) {
        EmSample es;
        if (!emPlanSample(plan, s, P, /*twoSided=*/false, es)) continue;
        const vec3  L     = es.L;
        const float NdotL = dot(N, L);
        if (NdotL <= 0.01) continue;
        // GGX spec toward the emitter sample — BRDF eval BEFORE the shadow ray:
        // most samples sit outside the lobe (D≈0), so their ray is skipped and
        // the per-pixel cost concentrates on the few in-lobe samples.
        const vec3  H     = normalize(V + L);
        const float NdotH = max(dot(N, H), 0.0);
        const vec3  F     = fresnelSchlick(max(dot(V, H), 0.0), F0);
        const float D     = distGGX(NdotH, roughness);
        const float G     = geomSmithG1(NdotV, k) * geomSmithG1(NdotL, k);
        vec3 c = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4) * NdotL * es.Le * es.gw;
        const float lum = max(max(c.r, c.g), c.b);
        if (lum <= 1e-4) continue;// outside the lobe → skip the shadow ray
        if (lum > pc.fireflyClamp) c *= pc.fireflyClamp / lum;
        if (doShadows) {
            // Emitter-skipping shadow ray (a light never shadows itself) — same
            // query as emissiveNEE's (kRayMaskOpaque: decals/glass don't block).
            const vec3  toLs  = es.lp - shadowOrig;
            const float dists = length(toLs);
            const float tMax  = dists - es.backoff - 1e-2;
            if (tMax > 1e-3) {
                rayQueryEXT rq;
                rayQueryInitializeEXT(rq, topAS, gl_RayFlagsTerminateOnFirstHitEXT,
                                      kRayMaskOpaque, shadowOrig, 1e-3, toLs / max(dists, 1e-6), tMax);
                while (rayQueryProceedEXT(rq)) {
                    if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
                        const MaterialDesc hm = mats[rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false)];
                        const vec3 hem = hm.emissive * hm.emissiveIntensity;
                        if (max(max(hem.r, hem.g), hem.b) < 0.05)// not an emitter → real occluder
                            rayQueryConfirmIntersectionEXT(rq);
                    }
                }
                if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT)
                    continue;
            }
        }
        sum += c * es.w;
    }
    return sum;
}
