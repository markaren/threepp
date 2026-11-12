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
    SHADOW_EPS = max(1e-3, max(max(abs(worldPos.x), abs(worldPos.y)), abs(worldPos.z)) * 1e-5);

    vec3        N         = normalize(nTap.xyz * 2.0 - 1.0);// mutable: FFT chop below
    // The RASTERISED normal, captured before the ocean branch below flips and tilts
    // N. The specular-AA gather (sunk past the water branch, see there) differences
    // its four G-buffer neighbours against THIS one: a thin-walled displaced mesh
    // that misses the water branch still falls through to the opaque path with N
    // already flipped and chop-tilted, and differencing raster neighbours against
    // that mutated normal reads as enormous variance — a matte band across the mesh.
    const vec3  Nraster   = N;

    const MaterialDesc pm = entryMat(id - 1u);

    // Breaking crests: an ocean whose UVs carry G = -1 has its vertices' crest
    // steepness in R (water_displace.comp); a plane's own UVs are in [0, 1].
    if (geoms[id - 1u].foamAddress != 0ul) {
        const vec4 crestTap = (msSampleIdx >= 0) ? texelFetch(gbufUvMS, px, msSampleIdx) : texelFetch(gbufUvTex, px, 0);
        // Crest foam: the same channel holds the vertex's drawn horizontal
        // displacement instead, and no tag (its z can be anything).
        if (kOceanCrestFoam != 0u) gCrestDisp = crestTap.rg;
        else if (crestTap.g < -0.5) gCrestEps = crestTap.r;
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
    float waterBankedMss = 0.0;
    // MACRO surface normal — the gbuffer normal BEFORE the fine-cascade tilt
    // (and before any stochastic jitter). shadeWater clamps its reflection ray
    // against this plane; see the self-hit note there. Orientation-agnostic,
    // so a rotated (Z-up) ocean works without a world-up assumption.
    vec3 Nmacro = N;
    // foamAddress gate — same ocean marker the water branch below uses, for the
    // same reason its comment gives: glTF transmission glass is almost always
    // thin-walled too (no KHR_volume → thinWalled), and oceanFineTileSize is
    // scene-GLOBAL, so without it every thin-walled glass surface in an ocean
    // scene had its normal flipped and tilted by wave slope sampled at its own
    // worldPos.xz (the lantern glazing wobbled at ocean-wave frequency).
    if (pm.transmission > 0.0 && pc.oceanFineTileSize > 0.0 && geoms[id - 1u].foamAddress != 0ul) {
        // Signed half floats retain the tiny slopes lost by biased [0,1]
        // encoding. Only the underwater path consumes these extra components;
        // above-water defaults retain their existing bit-exact normal path.
        // Water shades only dispatch A's dominant sample; dispatch B skips it.
        // (THREEPP_OCEAN_WATER_GBUFFER_NORMAL, kWaterGbufN: off, the G-buffer holds upstream's packing and
        // the normal stays the decoded G-buffer normal)
        if (kOceanExt && kWaterGbufN && camUnderwater()) {
            const vec2 xz = vec2(texelFetch(gbufMotionTex, px, 0).w,
                                 texelFetch(gbufUvTex, px, 0).w);
            N = normalize(vec3(xz.x, (N.y < 0.0 ? -1.0 : 1.0) *
                               sqrt(max(0.0, 1.0 - dot(xz, xz))), xz.y));
        }
        // The G-buffer preserves the water mesh's outward normal. Orient the
        // interface toward its incident medium before adding the fine slope.
        // Preserve the physical interface side. A view-facing flip can leave
        // a downward normal on visible grazing crests, producing black polygon
        // sheets when the fine-wave tilt and horizon clamp use that wrong side.
        if (kOceanExt) {
            if (camUnderwater()) N = -N;
            // A rough sea's long waves are what shows: their slope gains with the sea state (carried in the fraction of sheenColor.b), the short ripples give way
            gSeaRelief = 1.0 + 3.0 * fract(pm.sheenColor.b * 256.0);
            N = normalize(N * vec3(gSeaRelief, 1.0, gSeaRelief));
        } else {
            // upstream: the FFT DisplacedMesh's gbuffer normal comes out INVERTED (pointing down
            // into the water) -> NdotV~0 -> Fresnel pinned at ~1. Water is always viewed from
            // above, so face the camera, before the fine-cascade perturbation.
            if (dot(N, V) < 0.0) N = -N;
        }
        Nmacro = N;// post-flip, pre-tilt: the surface's own plane
        gFoamXZ = worldPos.xz;
        if (kOceanExt && pc.oceanFoamTileSize > 0.0) {   // as foam_world.comp's sampleDispXZ; 1.0: the application's Ocean::Options::choppiness
            const float inv0 = 1.0 / pc.oceanFoamTileSize;
            for (int it = 0; it < 2; ++it)
                gFoamXZ = worldPos.xz - 1.0 * ((it == 0 ? textureLod(oceanDisplace0, gFoamXZ * inv0, 0.0).rg : disp0Smooth(gFoamXZ * inv0)) * inv0 + textureLod(oceanDisplace2, gFoamXZ / pc.oceanFineTileSize, 0.0).rg / pc.oceanFineTileSize);
        }
      if (kOceanChop != 0u) {
        // ── CHOP NORMALS (DisplacedMesh::Params::chopNormals, water::ChopMoments)
        // Compiled in only for a frame that has them on (spec constant 7).
        // The one fine-cascade normal while it is on: the extended ocean's stencil path (the else branch: tile wander,
        // THREEPP_OCEAN_FINE_NORMAL_AGREE, the physical tangents, the band and rain reads, the skirt's banked slope) does not
        // run, so the fine slope tilts N once. The CDLOD mesh's own normal leaves cascade 2 out for the same reason.
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
        const vec2  uvW      = worldPos.xz * invTile;
        const float pixWorld = (cam.camAux.x > 0.5 ? 1.0 : length(worldPos - camPos))
                             * 2.0 * cam.projInverse[1][1]
                             / max(float(pc.height), 1.0);            // world m / pixel (FOV+res correct)
        const float pixAlong = pixWorld / max(dot(Nmacro, V), 0.02);
        const float vlM      = length(V.xz);
        const vec2  alongM   = vlM > 1e-4 ? V.xz / vlM : vec2(1.0, 0.0);
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
            // moments add: the mean slopes sum, the mean squares average.
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
            slopeVar = max(0.5 * (mom.a + momD.a) - dot(gKept, gKept), 0.0);
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
            const float s  = oceanClimate(oceanClimateMap, worldPos.xz - gCrestDisp).z;
            const float bk = oceanClimateBanked(oceanClimateMap);
            gKept    *= s;
            slopeVar  = s * s * max(slopeVar - bk, 0.0) + min(slopeVar, bk);
        }
        // ── WAKE FIELD ────────────────────────────────────────────────────────
        // Where something has churned the water the short wind waves are gone
        // (turbulence under the surface breaks them up: the smooth lane a boat
        // leaves across a ruffled sea), and while the churn lasts the surface
        // has a quick, fine ruffle of its own, the boil. So the lane scales the
        // fine cascade's slope down and the turbulence adds its own slope.
        const vec2 naturalKept = gKept;
        float wakeCalm    = 1.0;
        float wakeBoilVar = 0.0;// slope variance of the boil that went under a pixel
        if (geoms[id - 1u].foamAddress > 1ul) {
            gWakeAlbedo = WakeTable(geoms[id - 1u].foamAddress).albedo;
            gWake = oceanWakeSample(geoms[id - 1u].foamAddress, kOceanExt ? gFoamXZ : worldPos.xz,
                                    clamp(sqrt(pixWorld * pixAlong), pixWorld, 8.0 * pixWorld));
          if (kOceanExt) {
            // Foam and aeration read a little off to the side, by slow noise: a source's rounded outline does not survive it
#ifdef THREEPP_OCEAN_UNIFORM_CURRENT
            const vec2 fwXZ = gFoamXZ - vec2(currentX,currentZ)*pc.timeSec + 0.06 * pc.timeSec * vec2(0.6, -0.8);
#else
            const vec2 fwXZ = gFoamXZ + 0.06 * pc.timeSec * vec2(0.6, -0.8);
#endif
            // (a cap keeps to the wave that made it; the warp is the wash's)
            const float fwK = smoothstep(0.04, 0.15, gWake.g);
            const vec2 fw = fwK * 3.0 * vec2(fbm4(fwXZ * 0.5) - 0.47, fbm4(fwXZ * 0.5 + vec2(37.0, 11.0)) - 0.47)
                          + fwK * 1.4 * vec2(fbm4(fwXZ * 1.6 + vec2(5.0, 71.0)) - 0.47, fbm4(fwXZ * 1.6 + vec2(53.0, 29.0)) - 0.47);
            gWake.rg = oceanWakeSample(geoms[id - 1u].foamAddress, gFoamXZ + fw,
                                       clamp(sqrt(pixWorld * pixAlong), pixWorld, 8.0 * pixWorld)).rg;
            // the wash's foam drifts off downwind and thins: the film is the larger of here and a little
            // upwind (2.5 and 5 m, weaker), so a source's patch trails a soft tail along the wind instead of ending at a
            // hard outline (the round, opaque blobs round the TP base and her stern)
            if ((pc.oceanOptions & 1024u) == 0u && (gWake.r > 2e-3 || gWake.g > 2e-3)) {
                const vec2 wvS = pm.sheenColor.rg * 2.0 - 1.0;
                if (length(wvS) > 0.1) {
                    const vec2 wdS = normalize(wvS);
                    const float fp = clamp(sqrt(pixWorld * pixAlong), pixWorld, 8.0 * pixWorld);
                    const float r1 = oceanWakeSample(geoms[id - 1u].foamAddress, gFoamXZ + fw - 2.5 * wdS, fp).r;
                    const float r2 = oceanWakeSample(geoms[id - 1u].foamAddress, gFoamXZ + fw - 5.0 * wdS, fp).r;
                    gWake.r = max(gWake.r, max(0.7 * r1, 0.45 * r2));
                }
            }
          }
            // (0.9, upstream's, left a glassy sheet where a wash had passed)
            wakeCalm = kOceanExt ? 1.0 - 0.55 * clamp(gWake.a, 0.0, 1.0) : 1.0 - 0.9 * clamp(gWake.a, 0.0, 1.0);
            gKept *= wakeCalm;
            if (gWake.b > 2e-3) {
                // Slope of the boil, by the turbulent speed (m/s), capped. Two
                // scales (about 0.3 m and 0.1 m). What a pixel can no longer
                // hold is not dropped: its variance goes to the roughness, so
                // from far off churned water is the matt strip it is and not
                // a mirror with nothing on it.
                const float amp = kOceanExt ? min(0.3 * gWake.b, 0.08) : min(1.1 * gWake.b, 0.24);
                const float k1  = 1.0 - smoothstep(0.10, 0.35, pixAlong);
                const float k2  = 1.0 - smoothstep(0.035, 0.12, pixAlong);
                const vec2  b1  = oceanWakeNoised(vec3(worldPos.xz * 3.1, pc.timeSec * 1.7)).yz;
                const vec2  b2  = oceanWakeNoised(vec3(worldPos.xz * 8.3 + 11.0, pc.timeSec * 2.9)).yz;
                gKept += amp * (0.6 * k1 * b1 + 0.4 * k2 * b2);
                wakeBoilVar = amp * amp * (0.36 * (1.0 - k1 * k1) + 0.16 * (1.0 - k2 * k2));
            }
            // The waves the producers made (the fan a hull throws, a paddle's
            // rings). Their slope goes on the normal as the wind waves' does,
            // read over the pixel's footprint on the water, a strip along the
            // line of sight; what the strip is too coarse to hold goes to the
            // roughness with the boil's. The lane does not touch them.
            {
                const float vl    = length(V.xz);
                const vec2  along = vl > 1e-4 ? V.xz / vl : vec2(1.0, 0.0);
                const vec3  rp    = oceanWakeRipples(geoms[id - 1u].foamAddress, worldPos.xz,
                                                     vec2(-along.y, along.x) * pixWorld,
                                                     along * min(pixAlong, 64.0 * pixWorld));
                gKept       += rp.xy;
                wakeBoilVar += rp.z;
            }
        }
        if (kOceanExt) {
            const float waterSide = camUnderwater() ? -1.0 : 1.0;
            N = normalize(N + waterSide * vec3(-gKept.x, 0.0, -gKept.y));
        } else {
            N = normalize(N + vec3(-gKept.x, 0.0, -gKept.y));
        }
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
        const float vTot = slopeVar * wakeCalm * wakeCalm + wakeBoilVar;
        waterBankedMss = vTot;
        fineKernelSq = sqrt(r2 * r2 + vTot) - r2;
        // The footprint floor for the coarse mesh's own sub-pixel normals
        // (cascades 0+1 in the vertex normal), as on the other path.
        const float footRough = 0.45 * smoothstep(0.25, 4.0, pixWorld);
        fineKernelSq = min(max(fineKernelSq, footRough * footRough), 0.25);
      } else {
        const float invTile = 1.0 / pc.oceanFineTileSize;
        const float eps     = pc.oceanFineTileSize / 32.0;
        gFoamXZ = worldPos.xz;
        if (kOceanExt && pc.oceanFoamTileSize > 0.0) {   // as foam_world.comp's sampleDispXZ; 1.0: the application's Ocean::Options::choppiness
            const float inv0 = 1.0 / pc.oceanFoamTileSize;
            for (int it = 0; it < 2; ++it)
                gFoamXZ = worldPos.xz - 1.0 * ((it == 0 ? textureLod(oceanDisplace0, gFoamXZ * inv0, 0.0).rg : disp0Smooth(gFoamXZ * inv0)) * inv0 + textureLod(oceanDisplace2, gFoamXZ * invTile, 0.0).rg * invTile);
        }
        const float depsUV  = eps * invTile;
        vec2 normalRest = worldPos.xz;
        if (kOceanExt && pc.oceanFoamTileSize > 0.0) {
            const float inv0 = 1.0 / pc.oceanFoamTileSize;
            // (bilinear to converge, the B-spline for the last two: the rest position C1 across the swell's texels)
            for (int it=0; it<6; ++it) normalRest = worldPos.xz - (it < 4 ? textureLod(oceanDisplace0, normalRest*inv0,0).rg : disp0Smooth(normalRest*inv0))*inv0 - textureLod(oceanDisplace2, normalRest*invTile,0).rg*invTile;
        }
        // (sines, not value noise: value noise's slope is zero along its lattice lines, which showed as straight seams)
        vec2 tileWander = vec2(0.0);
        if (kOceanExt) for (int iw = 0; iw < 4; ++iw) {
            const float aw = 2.399963 * float(iw) + 0.7, kw = 0.021 * pow(1.83, float(iw));
            const vec2  dw = vec2(cos(aw), sin(aw));
            tileWander += vec2(sin(kw * dot(dw, normalRest) + 0.004 * pc.timeSec + 1.3 * float(iw)), cos(kw * dot(vec2(-dw.y, dw.x), normalRest) - 0.003 * pc.timeSec + 2.1 * float(iw))) * (3.2 / pow(1.6, float(iw)));
        }
        // THREEPP_OCEAN_FINE_NORMAL_AGREE (oceanOptions2 bit 30, on by default): where the mesh carries the fine cascade as geometry (its weight is 1 to
        // a 1 m footprint, 64 m from the camera or the contact square, and gone at 4 m), the fine normal is read
        // where the mesh read the height. The wander (there to hide the fine tile's repeat) puts it up to some metres
        // off: a wave's far side shaded on another wave's near side, which no crest hides: hard-edged black shapes
        // from a low camera. Within 64 m the fine normals now repeat with the tile, as the mesh's fine waves did.
        if (kOceanExt && (pc.oceanOptions2 & 1073741824u) != 0u) {
            const vec2 offW = abs(worldPos.xz - vec2(pc.oceanSkirtX, pc.oceanSkirtZ));
            tileWander *= smoothstep(64.0, 256.0, min(max(offW.x, offW.y), max(0.0, max(abs(worldPos.x), abs(worldPos.z)) - 128.0)));
        }
        const vec2  uvW     = kOceanExt ? (normalRest + tileWander) * invTile : worldPos.xz * invTile;
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
            const float vl     = length(V.xz);
            const vec2  along  = vl > 1e-4 ? V.xz / vl : vec2(1.0, 0.0);
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
            gMean = mix(grad, mean * keep, smoothstep(eps, (kOceanExt ? 7.0 : 2.0) * eps, pixAlong));// ocean loop: over a long range: at 2 eps this was a ring of brightness round the camera
        }
        vec2 gKept = (1.0 - subPix) * gMean;
        // THREEPP_OCEAN_RAIN: rain on the sea. It damps the short waves (Tsimplis & Thorpe 1989; Le Mehaute & Khangaonkar
        // 1990: the falling drops' momentum and the turbulence they stir take energy from waves of centimetres to decimetres,
        // markedly from ~10 mm/h), and every drop leaves a ring of capillary ripples that runs out at ~0.3 m/s and dies in
        // a fraction of a second. The rings are drawn only where a pixel resolves them (a few cm); past that their slope
        // goes into the roughness (below).
        float rainRingFade = 0.0;
        if (kOceanExt && pc.rainRate > 0.0) {
            gKept *= 1.0 / (1.0 + pc.rainRate / 12.0);
            rainRingFade = 1.0 - smoothstep(0.015, 0.06, pixWorld);
            if (rainRingFade > 0.0) gKept += rainRingFade * rainRings(worldPos.xz, pc.timeSec, pc.rainRate);
        }
        // ── SHORT WAVES ───────────────────────────────────────────────────────
        // The stencil above spans 2·eps = tile/16 (1.3 m on a 21 m tile), so it holds
        // nothing shorter than about 2.6 m: the near sea read as smooth cloth under
        // blurred wisps of reflected sky, though the height field carries waves down
        // to its 4 cm texels. Two detail bands put them back, each the difference
        // between a narrower stencil and the next wider one, so they add up to the
        // narrowest slope. A band stays on the normal while the pixel's long footprint
        // is shorter than its stencil and goes to the roughness after that, as the
        // long-wave slope does below.
        float bandVarSq = 0.0;
        if (kOceanExt) {
            const float texelM = pc.oceanFineTileSize / float(textureSize(oceanFineHeight, 0).x);
            vec2 gWide = grad;
            float eB = 0.16;
            for (int b = 0; b < 2; ++b, eB *= 0.33) {
                const float e = max(eB, 1.5 * texelM);
                if (e >= eps) continue;
                const float du = e * invTile;
                const vec2 g = vec2(textureLod(oceanFineHeight, uvW + vec2(du, 0.0), 0.0).r - textureLod(oceanFineHeight, uvW - vec2(du, 0.0), 0.0).r,
                                    textureLod(oceanFineHeight, uvW + vec2(0.0, du), 0.0).r - textureLod(oceanFineHeight, uvW - vec2(0.0, du), 0.0).r)
                               * invTile / (2.0 * e);
                const vec2 d = g - gWide;
                gWide = g;
                // Over a long range: a band's fade is a ring round the camera, and in the reflection
                // a step in brightness (tilted facets see higher, darker sky than the plane does).
                const float keep = 1.0 - smoothstep(0.7 * e, 7.0 * e, pixAlong);
                gKept += keep * (1.0 - subPix) * d;
                bandVarSq += (1.0 - keep * keep) * min(dot(d, d), 0.02);
            }
        }
        // The tile's ripples repeat every 21 m and its short waves can line up as fish scales close to: a
        // second, turned read of the same tile at another scale breaks them. Fades as the short bands do.
        if (kOceanExt) {
            // Accepted tile-hiding hack: this rotated, scaled second read is
            // NOT an independent physical wave spectrum. Retained for Stage B review.
            const float ca = 0.8253, sa = 0.5646, sc = 1.618 * invTile, du = 0.11 * sc;
            // (and it drifts over the first, so short waves cross: a sea's waves don't all share one speed)
            const vec2  p2 = vec2(ca * worldPos.x - sa * worldPos.z, sa * worldPos.x + ca * worldPos.z) * sc + vec2(0.37, 0.71) + vec2(-0.9, 0.6) * invTile * pc.timeSec;
            const vec2  g2 = vec2(textureLod(oceanFineHeight, p2 + vec2(du, 0.0), 0.0).r - textureLod(oceanFineHeight, p2 - vec2(du, 0.0), 0.0).r,
                                  textureLod(oceanFineHeight, p2 + vec2(0.0, du), 0.0).r - textureLod(oceanFineHeight, p2 - vec2(0.0, du), 0.0).r)
                            * invTile / (2.0 * 0.11);
            const float keep2 = 1.0 - smoothstep(0.11, 0.6, pixAlong);
            gKept += 0.35 * keep2 * (1.0 - subPix) * vec2(ca * g2.x + sa * g2.y, -sa * g2.x + ca * g2.y);
        }
        if (kOceanExt) {
            // Accepted tile-hiding hack, like the read above: not an independent spectrum. Longer repeat (34 m), other angle, wandering offset of a few metres.
            const float ca3 = 0.3090, sa3 = 0.9511, sc3 = 0.618 * invTile, du3 = 0.16 * sc3;
            const float tW = pc.timeSec;
            const vec2  walk = vec2(sin(0.13 * tW) + 0.5 * sin(0.31 * tW + 1.0) + 0.25 * sin(0.71 * tW + 4.0), sin(0.17 * tW + 2.0) + 0.5 * sin(0.37 * tW) + 0.25 * sin(0.83 * tW + 1.5));
            const vec2  warp = 0.12 * vec2(sin(0.031 * dot(worldPos.xz, vec2(0.8, 0.6)) + 0.021 * tW) + sin(0.057 * dot(worldPos.xz, vec2(-0.3, 0.95)) - 0.013 * tW), sin(0.029 * dot(worldPos.xz, vec2(0.5, -0.87)) - 0.017 * tW) + sin(0.061 * dot(worldPos.xz, vec2(0.99, 0.1)) + 0.023 * tW));
            const vec2  p3 = vec2(ca3 * worldPos.x - sa3 * worldPos.z, sa3 * worldPos.x + ca3 * worldPos.z) * sc3 + vec2(0.13, 0.59) + (3.0 * walk + 14.0 * warp) * invTile;
            const vec2  g3 = vec2(textureLod(oceanFineHeight, p3 + vec2(du3, 0.0), 0.0).r - textureLod(oceanFineHeight, p3 - vec2(du3, 0.0), 0.0).r,
                                  textureLod(oceanFineHeight, p3 + vec2(0.0, du3), 0.0).r - textureLod(oceanFineHeight, p3 - vec2(0.0, du3), 0.0).r)
                            * invTile / (2.0 * 0.16);
            const float keep3 = 1.0 - smoothstep(0.3, 1.5, pixAlong);
            const float amp3 = 0.7 * keep3 * (1.0 - subPix) * clamp(0.5 + 2.0 * warp.x, 0.2, 1.0);
            gKept = (gKept + amp3 * vec2(ca3 * g3.x + sa3 * g3.y, -sa3 * g3.x + ca3 * g3.y)) / sqrt(1.0 + amp3 * amp3);
        }
        // (what the ripples give up below goes to the roughness, not away: LEAN-style, so the water stays matt, not glass)
        const vec2 gK0 = gKept;
        // Wave groups: the local chop comes and goes, travelling at about a group's speed
        // (no gain or group modulation on the slope: the waves are what the spectrum gives)
        if (kOceanExt) bandVarSq += 0.5 * min(max(dot(gK0, gK0) - dot(gKept, gKept), 0.0), 0.04);
        // ── WAKE FIELD ────────────────────────────────────────────────────────
        // Where something has churned the water the short wind waves are gone
        // (turbulence under the surface breaks them up: the smooth lane a boat
        // leaves across a ruffled sea), and while the churn lasts the surface
        // has a quick, fine ruffle of its own, the boil. So the lane scales the
        // fine cascade's slope down and the turbulence adds its own slope.
        const vec2 naturalKept = gKept;
        float wakeCalm    = 1.0;
        float wakeBoilVar = 0.0;// slope variance of the boil that went under a pixel
        if (geoms[id - 1u].foamAddress > 1ul) {
            gWakeAlbedo = WakeTable(geoms[id - 1u].foamAddress).albedo;
            gWake = oceanWakeSample(geoms[id - 1u].foamAddress, kOceanExt ? gFoamXZ : worldPos.xz,
                                    clamp(sqrt(pixWorld * pixAlong), pixWorld, 8.0 * pixWorld));
          if (kOceanExt) {
            // Foam and aeration read a little off to the side, by slow noise: a source's rounded outline does not survive it
#ifdef THREEPP_OCEAN_UNIFORM_CURRENT
            const vec2 fwXZ = gFoamXZ - vec2(currentX,currentZ)*pc.timeSec + 0.06 * pc.timeSec * vec2(0.6, -0.8);
#else
            const vec2 fwXZ = gFoamXZ + 0.06 * pc.timeSec * vec2(0.6, -0.8);
#endif
            // (a cap keeps to the wave that made it; the warp is the wash's)
            const float fwK = smoothstep(0.04, 0.15, gWake.g);
            const vec2 fw = fwK * 3.0 * vec2(fbm4(fwXZ * 0.5) - 0.47, fbm4(fwXZ * 0.5 + vec2(37.0, 11.0)) - 0.47)
                          + fwK * 1.4 * vec2(fbm4(fwXZ * 1.6 + vec2(5.0, 71.0)) - 0.47, fbm4(fwXZ * 1.6 + vec2(53.0, 29.0)) - 0.47);
            gWake.rg = oceanWakeSample(geoms[id - 1u].foamAddress, gFoamXZ + fw,
                                       clamp(sqrt(pixWorld * pixAlong), pixWorld, 8.0 * pixWorld)).rg;
            // the wash's foam drifts off downwind and thins: the film is the larger of here and a little
            // upwind (2.5 and 5 m, weaker), so a source's patch trails a soft tail along the wind instead of ending at a
            // hard outline (the round, opaque blobs round the TP base and her stern)
            if ((pc.oceanOptions & 1024u) == 0u && (gWake.r > 2e-3 || gWake.g > 2e-3)) {
                const vec2 wvS = pm.sheenColor.rg * 2.0 - 1.0;
                if (length(wvS) > 0.1) {
                    const vec2 wdS = normalize(wvS);
                    const float fp = clamp(sqrt(pixWorld * pixAlong), pixWorld, 8.0 * pixWorld);
                    const float r1 = oceanWakeSample(geoms[id - 1u].foamAddress, gFoamXZ + fw - 2.5 * wdS, fp).r;
                    const float r2 = oceanWakeSample(geoms[id - 1u].foamAddress, gFoamXZ + fw - 5.0 * wdS, fp).r;
                    gWake.r = max(gWake.r, max(0.7 * r1, 0.45 * r2));
                }
            }
          }
            // (0.9, upstream's, left a glassy sheet where a wash had passed)
            wakeCalm = kOceanExt ? 1.0 - 0.55 * clamp(gWake.a, 0.0, 1.0) : 1.0 - 0.9 * clamp(gWake.a, 0.0, 1.0);
            gKept *= wakeCalm;
            if (gWake.b > 2e-3) {
                // Slope of the boil, by the turbulent speed (m/s), capped. Two
                // scales (about 0.3 m and 0.1 m). What a pixel can no longer
                // hold is not dropped: its variance goes to the roughness, so
                // from far off churned water is the matt strip it is and not
                // a mirror with nothing on it.
                const float amp = kOceanExt ? min(0.3 * gWake.b, 0.08) : min(1.1 * gWake.b, 0.24);
                const float k1  = 1.0 - smoothstep(0.10, 0.35, pixAlong);
                const float k2  = 1.0 - smoothstep(0.035, 0.12, pixAlong);
                const vec2  b1  = oceanWakeNoised(vec3(worldPos.xz * 3.1, pc.timeSec * 1.7)).yz;
                const vec2  b2  = oceanWakeNoised(vec3(worldPos.xz * 8.3 + 11.0, pc.timeSec * 2.9)).yz;
                gKept += amp * (0.6 * k1 * b1 + 0.4 * k2 * b2);
                wakeBoilVar = amp * amp * (0.36 * (1.0 - k1 * k1) + 0.16 * (1.0 - k2 * k2));
            }
            // The waves the producers made (the fan a hull throws, a paddle's
            // rings). Their slope goes on the normal as the wind waves' does,
            // read over the pixel's footprint on the water, a strip along the
            // line of sight; what the strip is too coarse to hold goes to the
            // roughness with the boil's. The lane does not touch them.
            {
                const float vl    = length(V.xz);
                const vec2  along = vl > 1e-4 ? V.xz / vl : vec2(1.0, 0.0);
                const vec3  rp    = oceanWakeRipples(geoms[id - 1u].foamAddress, worldPos.xz,
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
      if (!kOceanExt) {
        N = normalize(N + vec3(-gKept.x, 0.0, -gKept.y));// upstream
      } else {
        // Physics: cross the full displaced tangents, retaining shear as well as squeeze.
        // The raster normal already includes coarse horizontal displacement.
        const float eMap = max(eps, pc.oceanFineTileSize / float(textureSize(oceanDisplace2, 0).x));
        const vec2 qMap = normalRest;
        const vec2 fx = invTile / (2.0 * eMap) * (textureLod(oceanDisplace2, (qMap + vec2(eMap,0)) * invTile, 0).rg - textureLod(oceanDisplace2, (qMap - vec2(eMap,0)) * invTile, 0).rg);
        const vec2 fz = invTile / (2.0 * eMap) * (textureLod(oceanDisplace2, (qMap + vec2(0,eMap)) * invTile, 0).rg - textureLod(oceanDisplace2, (qMap - vec2(0,eMap)) * invTile, 0).rg);
        const float keepMap = (1.0 - subPix) * (1.0 - smoothstep(eps, 7.0 * eps, pixAlong));
        mat2 coarseMap = mat2(1.0);
        if (pc.oceanFoamTileSize > 0.0) {
            const float inv0 = 1.0 / pc.oceanFoamTileSize;
            const float e0 = pc.oceanFoamTileSize / float(textureSize(oceanDisplace0,0).x);
            // (differenced from the B-spline: a smooth Jacobian, not a piecewise-linear one with creases on the texel grid)
            const vec2 cx = inv0/(2.0*e0)*(disp0Smooth((qMap+vec2(e0,0))*inv0)-disp0Smooth((qMap-vec2(e0,0))*inv0));
            const vec2 cz = inv0/(2.0*e0)*(disp0Smooth((qMap+vec2(0,e0))*inv0)-disp0Smooth((qMap-vec2(0,e0))*inv0));
            coarseMap += mat2(cx,cz);
        }
        const mat2 fullMap = coarseMap + keepMap * mat2(fx, fz);
        // The ratio is orientation-invariant. Clamping a downward normal's
        // negative Y to +epsilon turned tiny underside slopes into near-vertical
        // facets, opening false Snell windows with quantized patch boundaries.
        const float slopeNy = camUnderwater() ? min(N.y, -0.0001) : max(N.y, 0.0001);
        const vec2 slopeMap = transpose(coarseMap) * (-N.xz / slopeNy) + naturalKept * wakeCalm;
        vec3 areaN = cross(vec3(fullMap[1].x, slopeMap.y, fullMap[1].y), vec3(fullMap[0].x, slopeMap.x, fullMap[0].y));
        vec3 waveN = dot(areaN, areaN) > 1e-12 ? normalize(areaN) : N;
        // THREEPP_OCEAN_NORMAL_NO_FLIP (oceanOptions2 bit 22, on by default; =0 rolls back): over the water a normal that
        // leans away from the eye is not negated. Negating it gave a normal pointing DOWN into the water (opposite
        // to the surface's own) on the far side of every short wave a low camera looks along: a patch that the
        // reflection optics could only fill flat from the macro normal, with a thin black line where the two
        // normals cancelled. It now stays what it is, and the optics take the facet as seen edge-on
        // (waterVisibleNormal).
        if (dot(waveN, V) < 0.0 && (camUnderwater() || (pc.oceanOptions2 & 4194304u) == 0u)) waveN = -waveN;
        vec3 detailN = vec3(-(gKept - naturalKept * wakeCalm).x, 0.0, -(gKept - naturalKept * wakeCalm).y);
        if (camUnderwater() && dot(waveN, areaN) < 0.0) detailN = -detailN;
        N = normalize(waveN + detailN);
      }
        // σ_slope² ≈ 2·roughness²; move the faded-out tilt's variance into roughness
        // so shadeWater's glint + reflection lobe broaden as the chop leaves N: all of
        // the slope where even pixWorld is sub-pixel, and what the strip's mean took.
        fineKernelSq = kOceanExt ? 2.0 * (dot(grad, grad) * subPix
                                          + (1.0 - subPix) * max(dot(grad, grad) - dot(gMean, gMean), 0.0)
                                          + bandVarSq)
                                 : 2.0 * (dot(grad, grad) * subPix
                                          + (1.0 - subPix) * max(dot(grad, grad) - dot(gMean, gMean), 0.0));
        // the lane took those waves away, banked or not; the boil's own go in
        fineKernelSq = fineKernelSq * wakeCalm * wakeCalm + 2.0 * wakeBoilVar;
        // The ocean's far-field skirt lays the long waves flat as its rings widen
        // (water_displace.comp); bank the slope they took (times the relief their
        // normals are drawn with) so the glitter and the sky's reflection carry on
        // across the fade. 0 on the inner grid.
        // the rain's rings where they are under a pixel: their mean square slope (a few 1e-3 at moderate rain)
        if (kOceanExt && pc.rainRate > 0.0) fineKernelSq += 2.0 * 0.0012 * sqrt(pc.rainRate) * (1.0 - rainRingFade);
        if (kOceanExt && pc.oceanSkirtD1 > pc.oceanSkirtD0) {
            const vec2  off = abs(worldPos.xz - vec2(pc.oceanSkirtX, pc.oceanSkirtZ));
            const float amp = 1.0 - smoothstep(pc.oceanSkirtD0, pc.oceanSkirtD1, max(off.x, off.y));
            #ifdef THREEPP_CDLOD
            // Same continuous footprint as displacement; E[s^2] comes from
            // averaging the unfiltered slope moments, never squaring a mean.
            float d=min(max(off.x,off.y),max(0.0,max(abs(worldPos.x),abs(worldPos.z))-128.0));
            float footprint=d<=2048.0?max(0.5,d/64.0):(d<10000.0?32.0*pow(d/2048.0,log(2.0)/log(10000.0/2048.0)):d/156.25);
            float altitude=camPos.y-(pm.sheenRoughness*100.0-50.0);
            if(altitude>80.0){
                float cameraD=max(off.x,off.y),lo=0.5,prev=0.0;
                float cameraFoot=footprint;
                float a=smoothstep(80.0,180.0,altitude);
                for(float s=0.5;s<=1024.0;s*=2.0){
                    float r=(s>=64.0?156.25:64.0)*s;
                    r=s<2.0?r*(1.0-a)+(s>=1.0?32.0*a:0.0):r+a*min(128.0*s,768.0);
                    if(cameraD<=r){cameraFoot=prev>0.0?lo*pow(max(cameraD,prev)/prev,log(s/lo)/log(r/prev)):lo*pow(s/lo,clamp(cameraD/max(1e-6,r),0.0,1.0));break;}lo=s;prev=r;
                }
                float contactD=max(0.0,max(abs(worldPos.x),abs(worldPos.z))-128.0);
                float contactFoot=contactD<=2048.0?max(0.5,contactD/64.0):(contactD<10000.0?32.0*pow(contactD/2048.0,log(2.0)/log(10000.0/2048.0)):contactD/156.25);
                footprint=min(cameraFoot,contactFoot);
            }
            float level=max(0.0,log2(footprint*float(textureSize(oceanDisplace0,0).x)/pc.oceanFoamTileSize));
            if(level>0.0) {
            float l=floor(level);vec2 uv=normalRest/pc.oceanFoamTileSize;
            // The inherited fine-wave sampler clamps LOD to zero. Explicit
            // texelFetch preserves that sampler for every existing consumer
            // while the new moment channel reads the actual pyramid levels.
            float moments[2];
            for(int j=0;j<2;++j){
                int mip=min(int(l)+j,textureQueryLevels(oceanDisplace0)-1);
                ivec2 n=textureSize(oceanDisplace0,mip);
                vec2 st=uv*vec2(n)-0.5,f=fract(st);ivec2 q=ivec2(floor(st));
                q=(q%n+n)%n;
                float a=texelFetch(oceanDisplace0,q,mip).b;
                float b=texelFetch(oceanDisplace0,(q+ivec2(1,0))%n,mip).b;
                float c=texelFetch(oceanDisplace0,(q+ivec2(0,1))%n,mip).b;
                float e=texelFetch(oceanDisplace0,(q+ivec2(1,1))%n,mip).b;
                moments[j]=mix(mix(a,b,f.x),mix(c,e,f.x),f.y);
            }
            float mss=mix(moments[0],moments[1],fract(level));
            float retained=dot(Nmacro.xz,Nmacro.xz)/max(Nmacro.y*Nmacro.y,0.01)/(gSeaRelief*gSeaRelief);
            float bank=max(0.0,mss-retained);
            // At mip zero the original reconstruction is still fully resolved.
            fineKernelSq+=2.0*gSeaRelief*gSeaRelief*bank*smoothstep(0.0,1.0,level);
            }
#else
            fineKernelSq += 2.0 * pc.oceanSkirtMss * gSeaRelief * gSeaRelief * (1.0 - amp * amp);
#endif
        }
        // The same removed slope energy must reach the sky/Fresnel integral,
        // not only the glint lobe. Capture it before the artistic footprint
        // roughness floor: fineKernelSq above is twice the physical MSS.
        waterBankedMss = 0.5 * fineKernelSq;
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
        const float footRough = 0.45 * smoothstep(0.25, 4.0, pixWorld);
        fineKernelSq = min(max(fineKernelSq, footRough * footRough), 0.25);
      }
    }
