// main() prologue, part 2 of 4 (inside main(), after the sky and unlit
// early-outs): world position from depth, the self-intersection epsilon, N,
// the material, V, the per-pixel RNG stream and the ocean's fine-cascade
// tilt of N with its wake field (the gWake globals).
    // Reconstruct world position from depth.
    const float depth    = (msSampleIdx >= 0) ? texelFetch(gbufDepthMS, px, msSampleIdx).x : texelFetch(gbufDepthTex, px, 0).x;
    const vec4  viewH    = cam.projInverse * vec4(ndc, depth, 1.0);
    const vec3  worldPos = (cam.viewInverse * vec4(viewH.xyz / viewH.w, 1.0)).xyz;

    // World-scale-aware self-intersection offset for this pixel's shadow /
    // reflection rays. Floored at 1e-3 (scenes within ~100 units unchanged),
    // grows ~linearly past that so the sun's tmin=0.0 shadow ray clears the
    // surface even when float precision at large |worldPos| swallows 1mm.
    // It grows with the distance from the eye as well: worldPos is rebuilt from
    // the depth buffer through the projection, and that error grows with the
    // distance, not with |worldPos|. A roof a kilometre off near the origin
    // rebuilt a millimetre or more inside itself, its sun ray hit the roof, and
    // which pixels did so changed with the sub-pixel jitter: far surfaces
    // flickered in and out of their own shadow (1 cm at 1 km).
    SHADOW_EPS = max(1e-3, max(max(max(abs(worldPos.x), abs(worldPos.y)), abs(worldPos.z)),
                                length(viewH.xyz / viewH.w)) * 1e-5);

    vec3        N         = normalize(nTap.xyz * 2.0 - 1.0);// mutable: FFT chop below
    // The RASTERISED normal, captured before the ocean branch below flips and tilts
    // N. The specular-AA gather (sunk past the water branch, see there) differences
    // its four G-buffer neighbours against THIS one: a thin-walled displaced mesh
    // that misses the water branch still falls through to the opaque path with N
    // already flipped and chop-tilted, and differencing raster neighbours against
    // that mutated normal reads as enormous variance — a matte band across the mesh.
    const vec3  Nraster   = N;

    const MaterialDesc pm = entryMat(id - 1u);

    // Crest foam: an ocean's UVs hold each vertex's drawn horizontal
    // displacement (water_displace.comp).
    if (geoms[id - 1u].foamAddress != 0ul) {
        const vec4 crestTap = (msSampleIdx >= 0) ? texelFetch(gbufUvMS, px, msSampleIdx) : texelFetch(gbufUvTex, px, 0);
        if (kOceanCrestFoam != 0u) gCrestDisp = crestTap.rg;
    }

    const vec3 V          = normalize(camPos - worldPos);
    const bool shadowsOn  = flagOn(1u);
    // Per-pixel RNG stream (frame-jittered) for emissive NEE; advanced through
    // the primary + reflection shades so the draws decorrelate. TAA averages.
    // Temporally stable (no `frame`) — same reasoning as gatherEnv: reflections /
    // emissive-NEE settle to a fixed result instead of flickering. Different
    // constant so this stream decorrelates from the AO/GI stream above.
    uint seed = pcgHash(uint(px.x) ^ pcgHash(uint(px.y) ^ 0x85ebca6bu));

    // ── FFT fine-cascade normal: recover the sub-mesh chop the rasterised wave
    // geometry can't resolve, via an oceanFineHeight central-difference gradient.
    // Gated on thin-shell + an active ocean (dummy tex + tile size 0 otherwise),
    // so it's a no-op for every non-ocean surface. Must run before N is used.
    // Removed sub-pixel FFT-chop slope variance, banked into the water roughness
    // below (declared out here so it survives the block's scope).
    float fineKernelSq = 0.0;
    // MACRO surface normal — the gbuffer normal BEFORE the fine-cascade tilt
    // (and before any stochastic jitter). shadeWater clamps its reflection ray
    // against this plane; see the self-hit note there. Orientation-agnostic,
    // so a rotated (Z-up) ocean works without a world-up assumption.
    vec3 Nmacro = N;
    // foamAddress gate: the same ocean marker the water branch in main() uses.
    // oceanFineTileSize is scene-GLOBAL, so without it every glass surface in an
    // ocean scene had its normal flipped and tilted by wave slope sampled at its
    // own worldPos.xz (the lantern glazing wobbled at ocean-wave frequency). A
    // DisplacedMesh that is not water (an opaque displaced surface) has no
    // transmission and keeps its own normal.
    if (pm.transmission > 0.0 && pc.oceanFineTileSize > 0.0 && geoms[id - 1u].foamAddress != 0ul) {
        // The FFT DisplacedMesh's gbuffer normal comes out INVERTED (pointing down
        // into the water) → NdotV≈0 → Fresnel pinned at ~1 → the ocean read as a
        // flat, dull sky reflection with NO depth gradient. Water is
        // always viewed from above, so face the camera. Done BEFORE the fine-cascade
        // perturbation so the chop tilt is added in the correct (up) frame.
        // The side is the WATER's, not the tilted normal's: seen at a grazing
        // angle (the open sea toward the horizon, a camera a few metres up), a
        // vertex normal tilted away from the eye by more than the view's
        // elevation faces away while its water faces the eye, and the facing
        // test alone left it inverted: its reflection went down into the water
        // (the sea bed: black) and it took no sun. Level water (along the
        // water's own up, fog.waterUp: a Z-up world turns it) is turned to the
        // eye's side of the point; a normal that faces away from the eye even
        // so is turned too.
        const float nUp = dot(N, fog.waterUp);
        if (dot(N, V) < 0.0 || (abs(nUp) > 0.5 && nUp * dot(V, fog.waterUp) < 0.0)) N = -N;
        Nmacro = N;// post-flip, pre-tilt: the surface's own plane
        // The water's own plane (fog.waterX, fog.waterUp: world x, y and z
        // for a Y-up water): the fine cascade, the wake and their footprints
        // are laid on it and the tilt goes along it, so a rotated ocean (a
        // Z-up world) reads its waves where they are (along world xz there
        // they were read across the water, and its ripples were lost).
        const vec3 wAxisX = fog.waterX;
        const vec3 wAxisZ = cross(fog.waterX, fog.waterUp);
        const vec2 wPos   = vec2(dot(worldPos, wAxisX), dot(worldPos, wAxisZ));
        const vec2 wV     = vec2(dot(V, wAxisX), dot(V, wAxisZ));
      if (kOceanChop != 0u) {
        // ── CHOP NORMALS (DisplacedMesh::Params::chopNormals, water::ChopMoments)
        // Compiled in only for a frame that has them on (spec constant 7).
        // oceanFineHeight is the fine cascade's moment chain here:
        // (h, mean slope x, mean slope z, mean squared slope) of the DISPLACED
        // surface, from exact derivatives (no stencil) with the chop's
        // pinching folded in (ocean_chop_moments.comp). One filtered fetch
        // over this pixel's footprint on the water replaces the stencil, the
        // footprint strip and the fades below: the mean slope tilts N and
        // what the average removed, E[s^2] - |E[s]|^2, is exactly the slope
        // spread the pixel cannot show, so it goes to the roughness (LEAN
        // mapping, Olano and Baker 2010). The footprint is a strip: one pixel
        // across the line of sight, pixWorld / cos(view) along it, so the
        // fetch is anisotropic with those two axes as its gradients.
        const float invTile  = 1.0 / pc.oceanFineTileSize;
        const vec2  uvW      = wPos * invTile;
        const float pixWorld = (cam.camAux.x > 0.5 ? 1.0 : length(worldPos - camPos))
                             * 2.0 * cam.projInverse[1][1]
                             / max(float(pc.height), 1.0);            // world m / pixel (FOV+res correct)
        const float pixAlong = pixWorld / max(dot(Nmacro, V), 0.02);
        const float vlM      = length(wV);
        const vec2  alongM   = vlM > 1e-4 ? wV / vlM : vec2(1.0, 0.0);
        const vec4  mom      = textureGrad(oceanFineHeight, uvW,
                                           alongM * (pixAlong * invTile),
                                           vec2(-alongM.y, alongM.x) * (pixWorld * invTile));
        float slopeVar = max(mom.a - dot(mom.gb, mom.gb), 0.0);
        vec2 gKept = mom.gb;
        {
            // ── DETILE ──────────────────────────────────────────────────────
            // The fine cascade repeats every tile, a few metres: from a mast
            // or a drone the same ripples come round again and again, a
            // scale pattern. A second placement of the same cascade, turned
            // 20 degrees, a little larger and shifted, is an independent
            // sample of the same sea (its texels are far from the first's),
            // and two independent fields summed over sqrt(2) have the
            // variance and the spectrum of one, and no common period. Its
            // slope comes back through the transpose of its rotation; the
            // mean slopes sum over sqrt(2), and the spread under the pixel is
            // the mean of the two spreads: the two samples are independent,
            // so their covariance under a footprint is nothing. (Taken as the
            // mean squares averaged less the summed mean's square, it lost the
            // cross term g1.g2 of the two means: on the near water, where each
            // read is one point of its sea, a spread of -g1.g2, as large as
            // the drawn waves' own slope at half the pixels and nothing at the
            // rest: the reflection's lobe there scattered what the drawn
            // ripples should have mirrored, the coloured streaks of a red
            // house in the ripples smoothed away.)
            const vec2  rD  = vec2(0.9393727, 0.3428978);// cos, sin of 0.35 rad
            const float sD  = 0.77;
            const vec2  uvD = vec2(rD.x * uvW.x - rD.y * uvW.y, rD.y * uvW.x + rD.x * uvW.y) * sD + vec2(0.37, 0.61);
            const vec2  dA  = alongM * (pixAlong * invTile * sD);
            const vec2  dX  = vec2(-alongM.y, alongM.x) * (pixWorld * invTile * sD);
            const vec4  momD = textureGrad(oceanFineHeight, uvD,
                                           vec2(rD.x * dA.x - rD.y * dA.y, rD.y * dA.x + rD.x * dA.y),
                                           vec2(rD.x * dX.x - rD.y * dX.y, rD.y * dX.x + rD.x * dX.y));
            const vec2  gD   = vec2(rD.x * momD.g + rD.y * momD.b, -rD.y * momD.g + rD.x * momD.b);
            gKept    = (mom.gb + gD) * 0.70710678;
            slopeVar = 0.5 * (max(mom.a - dot(mom.gb, mom.gb), 0.0) + max(momD.a - dot(momD.gb, momD.gb), 0.0));
        }
        // The part of that spread no footprint resolves, the same in every
        // direction: one texel's own (the fold fade's remainder and the
        // banked short waves, ocean_chop_moments.comp). The rest is the
        // cascade's drawn waves that this footprint averages, and that part
        // has a direction (THE FOOTPRINT'S TWO AXES, below).
        float slopeVarSub;
        {
            const ivec2 tsz = textureSize(oceanFineHeight, 0);
            const vec4  m0  = texelFetch(oceanFineHeight, ivec2(fract(uvW) * vec2(tsz)) % tsz, 0);
            slopeVarSub = min(max(m0.a - dot(m0.gb, m0.gb), 0.0), slopeVar);
        }
        if ((kOceanClimate & 1u) != 0u) {
            // ── WAVE CLIMATE (DisplacedMesh::WaveClimate, spec constant 12) ─
            // The site's map scales this cascade's amplitude by its .z at the
            // pixel's rest point (as water_displace.comp scales its geometry):
            // the slope by s, the slope spread the pixel cannot show by s^2.
            // Not the banked short waves no cascade holds (centimetres long):
            // they come to equilibrium with the local wind within metres of
            // fetch (Cox and Munk's slope statistics follow the wind, not the
            // fetch), so the glitter of sheltered water stays a wind's. The
            // map changes over tens of metres, so its own gradient is left out.
            const float s  = oceanClimate(oceanClimateMap, wPos - gCrestDisp).z;
            const float bk = oceanClimateBanked(oceanClimateMap);
            gKept       *= s;
            slopeVar     = s * s * max(slopeVar - bk, 0.0) + min(slopeVar, bk);
            slopeVarSub  = s * s * max(slopeVarSub - bk, 0.0) + min(slopeVarSub, bk);
        }
        // ── THE DEPTH (ocean_shore_depth.glsl) ───────────────────────────────
        // Over shallow water the cascades are scaled to the depth, as the
        // geometry's are (water_displace.comp). The gain takes this cascade's
        // height; its slope lies mostly in its shortest waves, which the depth
        // leaves until it is about their own height, so the resolved slope
        // takes the gain's square root and the spread the pixel averages of
        // it the gain. The spread no footprint resolves (one texel's own, the
        // banked short waves) is the local wind's, in equilibrium within
        // metres of fetch, and stays.
        if (geoms[id - 1u].foamAddress > 1ul) {
            const uvec2 sa = WakeTable(geoms[id - 1u].foamAddress).shoreDepth;
            if (sa != uvec2(0u)) {
                // (and the long cascades' refraction over the bed: the
                // factors in the gains, cascade 1's departure for the crest
                // foam)
                vec2  bendP, bendA, bendW;
                float bendH;
                oceanShoreBendAt(packUint2x32(sa), wPos - gCrestDisp, bendP, bendA, bendW, bendH);
                gShoreGain    = oceanShoreGainWith(packUint2x32(sa), wPos - gCrestDisp, bendA);
                gShoreBend1   = bendW * bendP.y;
                const float s = sqrt(gShoreGain.z);
                gKept   *= s;
                slopeVar = slopeVarSub + s * s * max(slopeVar - slopeVarSub, 0.0);
            }
        }
        // ── WAKE FIELD ────────────────────────────────────────────────────────
        // Where something has churned the water the short wind waves are gone
        // (turbulence under the surface breaks them up: the smooth lane a boat
        // leaves across a ruffled sea), and while the churn lasts the surface
        // has a quick, fine ruffle of its own, the boil. So the lane scales the
        // fine cascade's slope down and the turbulence adds its own slope.
        float wakeCalm    = 1.0;
        float wakeBoilVar = 0.0;// slope variance of the boil that went under a pixel
        if (geoms[id - 1u].foamAddress > 1ul) {
            gWakeAlbedo = WakeTable(geoms[id - 1u].foamAddress).albedo;
            gWakeLace   = WakeTable(geoms[id - 1u].foamAddress).lace;
            // Read at the water's REST position (gCrestDisp is zero without
            // crest foam): what the field holds lies ON the water, foam, air
            // and churn, so it rides the waves' orbital motion as the sea's
            // own foam does and the two never slide over each other.
            gWake = oceanWakeSample(geoms[id - 1u].foamAddress, wPos - gCrestDisp,
                                    clamp(sqrt(pixWorld * pixAlong), pixWorld, 8.0 * pixWorld));
            wakeCalm = 1.0 - 0.9 * clamp(gWake.a, 0.0, 1.0);
            gKept *= wakeCalm;
            if (gWake.b > 2e-3) {
                // Slope of the boil, by the turbulent speed (m/s), capped. Two
                // scales (about 0.3 m and 0.1 m). What a pixel can no longer
                // hold is not dropped: its variance goes to the roughness, so
                // from far off churned water is the matt strip it is and not
                // a mirror with nothing on it.
                const float amp = min(1.1 * gWake.b, 0.24);
                const float k1  = 1.0 - smoothstep(0.10, 0.35, pixAlong);
                const float k2  = 1.0 - smoothstep(0.035, 0.12, pixAlong);
                const vec2  b1  = oceanWakeNoised(vec3(wPos * 3.1, pc.timeSec * 1.7)).yz;
                const vec2  b2  = oceanWakeNoised(vec3(wPos * 8.3 + 11.0, pc.timeSec * 2.9)).yz;
                gKept += amp * (0.6 * k1 * b1 + 0.4 * k2 * b2);
                wakeBoilVar = amp * amp * (0.36 * (1.0 - k1 * k1) + 0.16 * (1.0 - k2 * k2));
            }
            // The waves the producers made (the fan a hull throws, a paddle's
            // rings). Their slope goes on the normal as the wind waves' does,
            // read over the pixel's footprint on the water, a strip along the
            // line of sight; what the strip is too coarse to hold goes to the
            // roughness with the boil's. The lane does not touch them.
            {
                const float vl    = length(wV);
                const vec2  along = vl > 1e-4 ? wV / vl : vec2(1.0, 0.0);
                const vec3  rp    = oceanWakeRipples(geoms[id - 1u].foamAddress, wPos,
                                                     vec2(-along.y, along.x) * pixWorld,
                                                     along * min(pixAlong, 64.0 * pixWorld));
                gKept       += rp.xy;
                wakeBoilVar += rp.z;
            }
        }
        N = normalize(N - gKept.x * wAxisX - gKept.y * wAxisZ);
        // The slope variance the pixel cannot hold widens the lobe. For a
        // microfacet surface the total slope variance (both axes) is alpha^2
        // (Walter et al. 2007, "Microfacet Models for Refraction through Rough
        // Surfaces", section 5.1: sigma^2 per axis = alpha^2/2), so the
        // material's alpha = r^2 and the variance add as squares:
        // alpha^2 = r^4 + var. shadeWater takes effRough = sqrt(r^2 + fks)
        // and distGGX squares it to alpha = r^2 + fks, so the banked term is
        // fks = sqrt(r^4 + var) - r^2. The lane takes the short waves away,
        // banked or not; the boil's and the producers' ripples' unresolved
        // variance (the same both-axes measure) goes in beside them.
        const float r2   = pm.roughness * pm.roughness;
        // ── THE FOOTPRINT'S TWO AXES ────────────────────────────────────────
        // The pixel's footprint on the water is a strip, pixAlong along the
        // line of sight and pixWorld across it, so the drawn waves it
        // averages are not averaged alike in every direction, and the slope
        // they leave to the reflection's lobe has a direction too. A wave of
        // wavenumber k at an angle phi to the line of sight has its slope
        // along k (cos phi of it along the line of sight, sin phi across);
        // the footprint's box keeps sinc(k cos phi La / 2) sinc(k sin phi Lc
        // / 2) of it, and what it does not keep is banked on both axes in
        // proportion cos^2 phi : sin^2 phi (the slope covariance of the
        // averaged waves; LEAN mapping's second moments, Olano and Baker
        // 2010). The moment chain holds only the trace of that covariance (a
        // fourth channel, the mean squared slope), so its split is modelled:
        // the cascade's slope variance spread evenly over ln k from its tile
        // to its Nyquist (a saturated range, Phillips 1958: the slope of a
        // k^-4 spectrum is the same each octave) and evenly over directions
        // (the short waves' directional spread is wide, Banner 1990), the
        // box's sinc^2 taken as exp(-(kL)^2 / 12) (the same curvature at the
        // origin), Lc held to La / 16 (the sampler's anisotropy). It is
        // nearly even where the strip is long against every drawn wave (a
        // wave much shorter than La is averaged out at all but the
        // directions square to the line of sight, whose slope is across) and
        // leans along the line of sight where La reaches the drawn waves and
        // Lc does not. The boil's spread and the material's go to both axes
        // alike; the texel's own (slopeVarSub) is taken below.
        // A FOOTPRINT UNDER A TEXEL. The fetch filters over a texel at least
        // (bilinear), and the bilinear blend of the texels' mean squares less
        // the square of their blended mean is the spread between the texels,
        // which the drawn surface does not have inside one: it is smooth
        // there (its shortest wave is six texels, the cascade's roll-off). So
        // on the near water, where the pixel is a fraction of a texel, the
        // read spread was a texel's whatever the pixel (on a 2 cm texel a
        // pixel a few metres off a USV's camera banked the slope of the
        // cascade's top octave), and the lobe drawn from it blurred the
        // ripples the normal already drew. The same model gives the share of the band the
        // footprint banks at its own size and at the size the fetch read
        // (each axis held to a texel), and the drawn waves' spread is scaled
        // by their ratio: 1 once the strip is a texel each way.
        {
            const float tex = pc.oceanFineTileSize / float(textureSize(oceanFineHeight, 0).x);
            const float La  = pixAlong;
            const float Lc  = max(pixWorld, pixAlong * (1.0 / 16.0));
            const float LaR = max(La, tex);
            const float LcR = max(Lc, tex);
            const float lk0 = log(TWO_PI * invTile);
            const float lk1 = log(PI / tex);
            float fa = 0.0, fc = 0.0, fR = 0.0;
            for (int i = 0; i < 6; ++i) {
                const float k2 = exp(2.0 * mix(lk0, lk1, (float(i) + 0.5) / 6.0)) * (1.0 / 12.0);
                for (int j = 0; j < 4; ++j) {
                    const float ph = (float(j) + 0.5) * (0.25 * 0.5 * PI);
                    const float c2 = cos(ph) * cos(ph);
                    const float b  = 1.0 - exp(-k2 * (c2 * La * La + (1.0 - c2) * Lc * Lc));
                    fa += c2 * b;
                    fc += (1.0 - c2) * b;
                    fR += 1.0 - exp(-k2 * (c2 * LaR * LaR + (1.0 - c2) * LcR * LcR));
                }
            }
            const float shareBand = (fa + fc) > 1e-6 ? fa / (fa + fc) : 0.5;
            const float vBand     = (slopeVar - slopeVarSub) * clamp((fa + fc) / max(fR, 1e-6), 0.0, 1.0);
            // The texel's own spread is the short waves past the grid
            // (centimetres: the short-wave option's capillary peak, about
            // k_m = 370 rad/m, 1.7 cm). Past the texel is not past the
            // pixel: a pixel a few metres off the camera is a centimetre
            // across and draws a definite facet of them, not their spread.
            // The same box model over them (a log-normal band about k_m, ln
            // deviation 0.5, held above the grid's Nyquist; four Gauss-Hermite
            // nodes) gives what of it lies under the pixel, and on which axis.
            float sa = 0.0, sc = 0.0;
            {
                const float xg[4] = float[4](-2.3344142, -0.7419637, 0.7419637, 2.3344142);
                const float wg[4] = float[4](0.0458759, 0.4541241, 0.4541241, 0.0458759);
                for (int i = 0; i < 4; ++i) {
                    const float kS = max(370.0 * exp(0.5 * xg[i]), PI / tex);
                    const float k2 = kS * kS * (1.0 / 12.0);
                    for (int j = 0; j < 4; ++j) {
                        const float ph = (float(j) + 0.5) * (0.25 * 0.5 * PI);
                        const float c2 = cos(ph) * cos(ph);
                        const float b  = 1.0 - exp(-k2 * (c2 * La * La + (1.0 - c2) * Lc * Lc));
                        sa += 0.25 * wg[i] * c2 * b;
                        sc += 0.25 * wg[i] * (1.0 - c2) * b;
                    }
                }
            }
            const float vSub = slopeVarSub * (sa + sc);// 1 when all of it is under the pixel, half each axis
            slopeVar = vSub + vBand;
            const float vTotS  = slopeVar * wakeCalm * wakeCalm + wakeBoilVar;
            const float vAlong = (sa * slopeVarSub + shareBand * vBand) * wakeCalm * wakeCalm + 0.5 * wakeBoilVar;
            fineKernelSq    = sqrt(r2 * r2 + vTotS) - r2;
            gWaterLobeAlong = vTotS > 1e-10 ? vAlong / vTotS : 0.5;
            // ── THE LOBE'S SHARE ────────────────────────────────────────────
            // How much of that spread the reflection's lobe draws (the glint
            // and the facet Fresnel take all of it). The spread is under the
            // pixel, but under a pixel that shows the drawn waves it is the
            // centimetre waves riding on each drawn facet (the short-wave
            // option's capillaries, under any pixel past a few metres) and
            // the top of the cascade the strip just reaches, and drawn as a
            // lobe it smoothed each drawn facet's crisp mirror of the shore
            // and the sky into an even sheen at every wind and every
            // distance: on the near water of a USV's camera at 10 m/s the
            // high-pass detail fell to 40 %. A still with the sea's clock
            // held for 300 frames of Ray Reconstruction's history, and the
            // same lobe integrated by 64 rays a pixel with no noise for RR to
            // remove, both kept that loss: it is the lobe's width, not the
            // denoiser. So the lobe draws the spread as far as the pixel has
            // averaged the drawn waves away: the share the strip's long axis
            // banks of the drawn cascade's slope (fa over the same samples'
            // weight, the sum of cos^2 phi over them: 12; the box model
            // above), squared. Squared because what makes the drawn facets'
            // reflections read is the bulk of the drawn slope, not its top
            // octave: 10 m from a camera at a USV's height the strip banks a
            // fifth of the cascade (its 15 to 30 cm waves), and a fifth of
            // the capillaries' spread drawn there still smoothed the
            // reflections of the metre waves under it. So: nothing where a
            // pixel spans a few centimetres of water (each drawn facet's
            // mirror, as before the lobe), a twentieth at 10 m, a quarter at
            // 20 m, and all of it at range, where the pixel's strip runs over
            // the drawn waves and the sea is the even sheen that darkens
            // under the land. It grows with the footprint, so it has no edge
            // on the water.
            const float bankedAlong = clamp(fa * (1.0 / 12.0), 0.0, 1.0);
            gWaterLobeShare = bankedAlong * bankedAlong;
        }
        // The footprint floor for the coarse mesh's own sub-pixel normals
        // (cascades 0+1 in the vertex normal), as on the other path, held to
        // the slope those normals carry here (oceanRasterSlopeVar).
        const float fksDrawn  = fineKernelSq;
        const float footRough = 0.45 * smoothstep(0.25, 4.0, pixWorld);
        if (footRough > 0.0)// (near water floors at nothing: no neighbours to read)
            fineKernelSq = max(fineKernelSq, min(footRough * footRough, sqrt(oceanRasterSlopeVar(px, Nraster))));
        fineKernelSq = min(fineKernelSq, 0.25);
        // What the floor adds is the raster normals' spread, read from the
        // pixel's neighbours on screen: both axes alike.
        if (fineKernelSq > fksDrawn) {
            const float vD = (fksDrawn + r2) * (fksDrawn + r2) - r2 * r2;
            const float vF = (fineKernelSq + r2) * (fineKernelSq + r2) - r2 * r2;
            gWaterLobeAlong = (gWaterLobeAlong * vD + 0.5 * (vF - vD)) / max(vF, 1e-10);
            // The raster normals' spread is waves the pixel cannot show: the
            // lobe draws all of it.
            gWaterLobeShare = (gWaterLobeShare * vD + (vF - vD)) / max(vF, 1e-10);
        }
      } else {
        const float invTile = 1.0 / pc.oceanFineTileSize;
        const float eps     = pc.oceanFineTileSize / 32.0;
        const float depsUV  = eps * invTile;
        const vec2  uvW     = wPos * invTile;
        const float hL = textureLod(oceanFineHeight, uvW + vec2(-depsUV, 0.0), 0.0).r * invTile;
        const float hR = textureLod(oceanFineHeight, uvW + vec2( depsUV, 0.0), 0.0).r * invTile;
        const float hD = textureLod(oceanFineHeight, uvW + vec2(0.0, -depsUV), 0.0).r * invTile;
        const float hU = textureLod(oceanFineHeight, uvW + vec2(0.0,  depsUV), 0.0).r * invTile;
        const vec2  grad = vec2(hR - hL, hU - hD) / (2.0 * eps);
        // ── Distance filtering (LEAN-style specular AA for water) ──────────────
        // oceanFineHeight is a single-mip texture (no prefilter), sampled here at
        // a FIXED world-space eps regardless of distance. At the altitudes this
        // water is viewed from (100 m – 2 km up) the fine cascade's metre-and-
        // below wave detail sits far below the pixel footprint, so point-sampling
        // its slope and shading a razor-sharp GGX lobe against it decorrelates the
        // sun glint into a pure-noise speckle field. Fix: FADE the chop out of N
        // as the pixel footprint outgrows the ~2·eps slope baseline, and BANK the
        // removed slope variance into the water roughness (fineKernelSq) so the
        // glint broadens into a coherent glitter sheen instead of aliasing. Near
        // water (small footprint) keeps its full crisp sparkle.
        // Under a PARALLEL projection the footprint doesn't grow with distance —
        // projInverse[1][1] is already half the frustum height in world units,
        // so the leg drops out and every pixel covers the same patch of water.
        const float pixWorld = (cam.camAux.x > 0.5 ? 1.0 : length(worldPos - camPos))
                             * 2.0 * cam.projInverse[1][1]
                             / max(float(pc.height), 1.0);            // world m / pixel (FOV+res correct)
        // The central-difference `tilt` measures wave slope at the ~2·eps (≈0.44 m)
        // scale; once the pixel footprint outgrows that, the slope it samples is
        // sub-pixel and must move from GEOMETRY (a razor-glinted N) into ROUGHNESS
        // (a broadened glitter lobe), or the sun highlight aliases to speckle. The
        // old fade completed at 12·eps (≈2.6 m/px) — but the demo views this ocean
        // from 100 m–2 km up, where much of the water sits at 0.3–1.5 m/px: the
        // chop stayed in N, razor-glinted, = the reported per-pixel noise. Complete
        // the hand-off by ~5·eps so mid-distance water converts to sheen.
        const float subPix   = smoothstep(0.5 * eps, 1.5 * eps, pixWorld);// 0 near → 1 sub-pixel
        // THE FOOTPRINT IS NOT ROUND. pixWorld is the pixel's width ACROSS the line
        // of sight; ALONG it the pixel covers pixWorld / cos(view, surface normal)
        // of water, which from a camera 2 m over the sea is 50x as much 100 m out.
        // Fading on pixWorld alone kept the point-sampled chop in N until the
        // across-footprint outgrew it (300 m for a 12 m fine tile at 0.6 render
        // scale) while every pixel past ~30 m already spanned whole waves
        // lengthwise. And the TAA jitter moves the sample by half that length
        // every frame: the slope was a different wave's each frame, so the mirror
        // ray landed on shore, sky or water body in turn -- speckle in the far
        // water seen along the surface, and a blink in what it reflects, even with
        // the sea frozen (none from above, where cos ~ 1).
        // So once the footprint is longer than the stencil the slope is the
        // footprint's MEAN, both components (the across slope varies along the
        // strip as much as the along slope does):
        //   along   exactly, from the height at the strip's two ends
        //   across  the stencil's across-difference at four places along it
        // The mean is taken over TWO strip lengths (this pixel's and half of each
        // neighbour's): a box one strip long still passes waves between one and two
        // strips long, which one sample a strip cannot carry -- thin bright lines a
        // pixel row high, shimmering under the jitter.
        // Past a quarter of the tile the mean is eased out: it is small there, a
        // hard mirror ray makes much of little at grazing (the far shoreline's
        // reflection breathed), and at a whole tile it is exactly zero anyway. So
        // the far water mirrors its macro plane, steadily, and the lost slope is
        // banked as roughness below. Straight down the strip is pixWorld: no change.
        const float pixAlong = pixWorld / max(dot(Nmacro, V), 0.02);
        vec2 gMean = grad;
        if (pixAlong > eps) {
            const float vl     = length(wV);
            const vec2  along  = vl > 1e-4 ? wV / vl : vec2(1.0, 0.0);
            const vec2  across = vec2(-along.y, along.x);
            const float len    = min(2.0 * pixAlong, pc.oceanFineTileSize);
            const vec2  aUV    = along * (len * invTile);
            const vec2  xUV    = across * depsUV;
            const float hF = textureLod(oceanFineHeight, uvW + 0.5 * aUV, 0.0).r;
            const float hB = textureLod(oceanFineHeight, uvW - 0.5 * aUV, 0.0).r;
            float dAcross = 0.0;
            for (int k = 0; k < 4; ++k) {
                const vec2 c = uvW + aUV * ((float(k) + 0.5) * 0.25 - 0.5);
                dAcross += textureLod(oceanFineHeight, c + xUV, 0.0).r - textureLod(oceanFineHeight, c - xUV, 0.0).r;
            }
            const vec2 mean = along * ((hF - hB) * invTile / len) + across * (dAcross * invTile / (8.0 * eps));
            const float keep = 1.0 - smoothstep(0.25 * pc.oceanFineTileSize, 0.5 * pc.oceanFineTileSize, pixAlong);
            // eased in over one stencil so the two estimates meet without a seam
            gMean = mix(grad, mean * keep, smoothstep(eps, 2.0 * eps, pixAlong));
        }
        vec2 gKept = (1.0 - subPix) * gMean;
        // ── WAKE FIELD ────────────────────────────────────────────────────────
        // Where something has churned the water the short wind waves are gone
        // (turbulence under the surface breaks them up: the smooth lane a boat
        // leaves across a ruffled sea), and while the churn lasts the surface
        // has a quick, fine ruffle of its own, the boil. So the lane scales the
        // fine cascade's slope down and the turbulence adds its own slope.
        float wakeCalm    = 1.0;
        float wakeBoilVar = 0.0;// slope variance of the boil that went under a pixel
        if (geoms[id - 1u].foamAddress > 1ul) {
            gWakeAlbedo = WakeTable(geoms[id - 1u].foamAddress).albedo;
            gWakeLace   = WakeTable(geoms[id - 1u].foamAddress).lace;
            // Read at the water's REST position (gCrestDisp is zero without
            // crest foam): what the field holds lies ON the water, foam, air
            // and churn, so it rides the waves' orbital motion as the sea's
            // own foam does and the two never slide over each other.
            gWake = oceanWakeSample(geoms[id - 1u].foamAddress, wPos - gCrestDisp,
                                    clamp(sqrt(pixWorld * pixAlong), pixWorld, 8.0 * pixWorld));
            wakeCalm = 1.0 - 0.9 * clamp(gWake.a, 0.0, 1.0);
            gKept *= wakeCalm;
            if (gWake.b > 2e-3) {
                // Slope of the boil, by the turbulent speed (m/s), capped. Two
                // scales (about 0.3 m and 0.1 m). What a pixel can no longer
                // hold is not dropped: its variance goes to the roughness, so
                // from far off churned water is the matt strip it is and not
                // a mirror with nothing on it.
                const float amp = min(1.1 * gWake.b, 0.24);
                const float k1  = 1.0 - smoothstep(0.10, 0.35, pixAlong);
                const float k2  = 1.0 - smoothstep(0.035, 0.12, pixAlong);
                const vec2  b1  = oceanWakeNoised(vec3(wPos * 3.1, pc.timeSec * 1.7)).yz;
                const vec2  b2  = oceanWakeNoised(vec3(wPos * 8.3 + 11.0, pc.timeSec * 2.9)).yz;
                gKept += amp * (0.6 * k1 * b1 + 0.4 * k2 * b2);
                wakeBoilVar = amp * amp * (0.36 * (1.0 - k1 * k1) + 0.16 * (1.0 - k2 * k2));
            }
            // The waves the producers made (the fan a hull throws, a paddle's
            // rings). Their slope goes on the normal as the wind waves' does,
            // read over the pixel's footprint on the water, a strip along the
            // line of sight; what the strip is too coarse to hold goes to the
            // roughness with the boil's. The lane does not touch them.
            {
                const float vl    = length(wV);
                const vec2  along = vl > 1e-4 ? wV / vl : vec2(1.0, 0.0);
                const vec3  rp    = oceanWakeRipples(geoms[id - 1u].foamAddress, wPos,
                                                     vec2(-along.y, along.x) * pixWorld,
                                                     along * min(pixAlong, 64.0 * pixWorld));
                gKept       += rp.xy;
                wakeBoilVar += rp.z;
            }
        }
        // Full physical slope: N ∝ up − ∇h. Until 2026-08 this carried a 0.5
        // (undocumented, from the first deferred commit) that rendered the
        // sub-mesh chop at HALF its steepness — the 1–9 m wind waves read as
        // a glassy sheen and a 10 m hull looked like a bath toy on the swell.
        // The speckle that factor presumably guarded against is handled by
        // the footprint fade + roughness banking below.
        N = normalize(N - gKept.x * wAxisX - gKept.y * wAxisZ);
        // σ_slope² ≈ 2·roughness²; move the faded-out tilt's variance into roughness
        // so shadeWater's glint + reflection lobe broaden as the chop leaves N: all of
        // the slope where even pixWorld is sub-pixel, and what the strip's mean took.
        fineKernelSq = 2.0 * (dot(grad, grad) * subPix
                              + (1.0 - subPix) * max(dot(grad, grad) - dot(gMean, gMean), 0.0));
        // the lane took those waves away, banked or not; the boil's own go in
        fineKernelSq = fineKernelSq * wakeCalm * wakeCalm + 2.0 * wakeBoilVar;
        // FOOTPRINT FLOOR. The fine-cascade slope alone is tiny for a calm sea
        // (Phillips amplitude ∝ windSpeed⁴ → a 4 m/s coastal sea has a millimetric
        // fine cascade), so banking only IT leaves effRough ≈ the base 0.03 and the
        // razor sun lobe still glints the RASTERISED mesh normal — the coarse FFT
        // vertex grid (here ~25 m verts over a 9.6 km tile) displaced by cascades
        // 0/1, itself sub-pixel at 100 m–2 km and aliasing to the reported speckle.
        // The fine-cascade fade can't touch that geometry normal, so floor the
        // banked variance on the pixel FOOTPRINT: distant water always carries a
        // broad-enough lobe to integrate the unresolved wave-normal distribution
        // into a coherent glitter sheen. Near water (footprint ≤ ~0.25 m/px) floors
        // at ~0 and keeps its crisp near-mirror sparkle. Independent of wave energy,
        // so it holds for a glassy calm sea and a choppy one alike.
        // Held to the slope the rasterised normals carry (oceanRasterSlopeVar):
        // the floor stands for their spread, so where they hold little (a
        // calm sea; the open sea's far patches, whose normals keep only the
        // long waves) a footprint that reaches the horizon no longer makes the
        // far water rougher than its waves: a rough lobe at a grazing view is
        // a low facet Fresnel, and the sea past a few hundred footprints went
        // a band darker than the water before it.
        const float footRough = 0.45 * smoothstep(0.25, 4.0, pixWorld);
        if (footRough > 0.0)// (near water floors at nothing: no neighbours to read)
            fineKernelSq = max(fineKernelSq, min(footRough * footRough, sqrt(oceanRasterSlopeVar(px, Nraster))));
        fineKernelSq = min(fineKernelSq, 0.25);
      }
    }
