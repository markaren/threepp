// Split from deferred_shade.comp: thin-shell water shading (RasterFirst),
// in-glass interior transit marching, and refractive glass shading.

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
                float slopeVarSq) {
    const float NdotV = max(dot(N, V), 1e-4);
    const float ior   = max(pm.ior, 1.0);
    const float r0    = pow((1.0 - ior) / (1.0 + ior), 2.0);
    const float F     = r0 + (1.0 - r0) * pow(1.0 - NdotV, 5.0);

    // Effective specular roughness = base water roughness ⊕ the sub-pixel chop
    // variance banked out of N (LEAN spec-AA). Distant water fattens its lobe →
    // the sun glint reads as a soft coherent sheen; near water (slopeVarSq→0)
    // keeps its crisp near-mirror sparkle.
    const float effRough = sqrt(pm.roughness * pm.roughness + slopeVarSq);

    // Reflected sky + scene. The reflection ray points up/out (away from the
    // water), so it escapes to the sky or hits scene geometry cleanly. Env
    // MISSES read a MILDLY blurred mip — point features in the env (stars, a
    // small bright moon) otherwise reflect off the chop-perturbed normals as
    // per-pixel white speckle; smooth day skies are visually unchanged.
    const vec3 R = reflect(-V, N);
    // Env-miss reflection blur widens with the banked chop variance: distant
    // sub-pixel wave facets reflect a point-feature sky (sun disc / bright cloud
    // edge) as per-pixel speckle unless the reflection is filtered by the same
    // footprint the glint is. Floor at 0.12 (the near-water sheen).
    const float reflLod = max(0.12, effRough) * maxLod;
    const vec3 reflectColor = traceRadiance(P + N * SHADOW_EPS, R, doShadows, maxLod,
                                            reflLod, seed, /*cheapHits=*/true,// water: blur+temporal absorb
                                            /*probeHitFill=*/true);

    // Transmission: ANALYTIC deep-water body — deliberately NOT a refraction
    // ray. A downward ray self-intersects this same choppy surface (adjacent
    // wave crests), which traceRadiance then shades as opaque water → a chaotic
    // black/white speckle that fills the screen when viewed from ABOVE (F≈0.02
    // there, so the transmission term dominates). The seabed is near-black and
    // metres down anyway, so true see-through buys almost nothing. Instead: the
    // Beer-Lambert absorption tint lit by ambient skylight → a smooth deep
    // blue-green. (The path tracer can refract correctly because it CONTINUES
    // the path with medium tracking across the bounce; a single inline ray can't.)
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
    const vec3  transmitColor = tint * (skyLum * 0.35 + lights.ambient) * depthFade;

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
            const vec3  camPosF = (cam.viewInverse * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
            const float distF   = length(P - camPosF);
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
            const float foamMask = max(sheet, web * 0.8);
            if (foamMask > 0.0) {
                const float micro = textureLod(foamDetailTex, P.xz * 0.25 - drift * 0.1, lodMicro).r;
                vec3 foamCol = mix(vec3(0.62, 0.68, 0.72), vec3(0.97, 0.99, 1.00), micro);
                foamCol = mix(foamCol, vec3(0.97, 0.99, 1.00), 0.35 * foamCoverage);// fresh caps brighter
                const float foamRough= mix(1.0, 0.45, micro);
                const vec3  foamDiff = sampleEnvLod(N, maxLod) + lights.ambient;
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
        rayQueryInitializeEXT(rq, topAS, gl_RayFlagsOpaqueEXT, 0xFFu, o, 1e-3, dir, 1e30);
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
                const vec3 diffInd = sampleEnvLod(n, maxLod) + lights.ambient;
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
    const float r0      = pow((1.0 - ior) / (1.0 + ior), 2.0);
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

    vec3 sum = vec3(0.0);
    for (int s = 0; s < kGlassSamples; ++s) {
        vec3 Ns = ggxHalfVectorFib(N, gr, s, kGlassSamples);// distinct per-iteration Fibonacci sample
        if (dot(Ns, V) <= 0.0) Ns = N;// reject back-facing microfacets
        const float NdotV = max(dot(Ns, V), 1e-4);
        const float F     = r0 + (1.0 - r0) * pow(1.0 - NdotV, 5.0);

        // Reflection (sky + scene) — multi-bounce / recursive.
        const vec3 R = reflect(-V, Ns);
        const vec3 reflectColor = traceRadiance(P + N * SHADOW_EPS, R, doShadows, maxLod, missLod, seed,
                                                /*cheapHits=*/false,// glass shows hits SHARP — deterministic shading
                                                /*probeHitFill=*/true);

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
                                                  /*probeHitFill=*/false);// content under glass: probes can't resolve the cavity
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
                                                  /*probeHitFill=*/false);// content under glass: probes can't resolve the cavity
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
            const vec3  L     = normalize(lights.dirLights[i].direction);
            const float NdotL = max(dot(N, L), 0.0);
            if (NdotL <= 0.0) continue;
            const float vis = doShadows ? shadowVis(P + N * SHADOW_EPS, L, 1e30) : 1.0;
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
