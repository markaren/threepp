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
    // foamAddress gate — same ocean marker the water branch below uses, for the
    // same reason its comment gives: glTF transmission glass is almost always
    // thin-walled too (no KHR_volume → thinWalled), and oceanFineTileSize is
    // scene-GLOBAL, so without it every thin-walled glass surface in an ocean
    // scene had its normal flipped and tilted by wave slope sampled at its own
    // worldPos.xz (the lantern glazing wobbled at ocean-wave frequency).
    if (pm.thinWalled != 0 && pc.oceanFineTileSize > 0.0 && geoms[id - 1u].foamAddress != 0ul) {
        // The FFT DisplacedMesh's gbuffer normal comes out INVERTED (pointing down
        // into the water) → NdotV≈0 → Fresnel pinned at ~1 → the ocean read as a
        // flat, dull sky reflection with NO depth gradient. Water is
        // always viewed from above, so face the camera. Done BEFORE the fine-cascade
        // perturbation so the chop tilt is added in the correct (up) frame.
        if (dot(N, V) < 0.0) N = -N;
        Nmacro = N;// post-flip, pre-tilt: the surface's own plane
        const float invTile = 1.0 / pc.oceanFineTileSize;
        const float eps     = pc.oceanFineTileSize / 32.0;
        const float depsUV  = eps * invTile;
        const vec2  uvW     = worldPos.xz * invTile;
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
            gWake = oceanWakeSample(geoms[id - 1u].foamAddress, worldPos.xz,
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
        N = normalize(N + vec3(-gKept.x, 0.0, -gKept.y));
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
        const float footRough = 0.45 * smoothstep(0.25, 4.0, pixWorld);
        fineKernelSq = min(max(fineKernelSq, footRough * footRough), 0.25);
    }
