
@compute @workgroup_size(8, 8)
fn rt_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res   = rt.iRes.xy;
    let resXu = u32(res.x);
    // Direct pixel assignment: no compaction has happened yet, every pixel needs
    // shading.  2-D tiled dispatch keeps adjacent pixels in the same workgroup
    // → coherent material/texture access.
    let pixel = vec2<i32>(gid.xy);
    if (pixel.x >= i32(res.x) || pixel.y >= i32(res.y)) { return; }
    let pixelIdx = u32(pixel.y) * resXu + u32(pixel.x);
    {
        let fc         = u32(rt.frameCount.x);
        let foveatedOn = rt.params.z > 0.5;

    // Don't foveate sky/env-map pixels — they're cheap to trace (BVH miss)
    // and skipping creates visible zone boundaries in uniform backgrounds.
    // Use previous frame's gBuf depth: env/sky pixels have depth <= 0.
    let prevDepth = textureLoad(gBufRead, pixel, 0).w;
    let prevWasEnv = prevDepth <= 0.0;
    // Peek at this frame's primary hit result — needed BEFORE the checker/
    // foveated-skip decision so we can disable those skips on pixels that
    // transitioned from object (prev) to sky (current).  Without this check,
    // the skip path copies the leader's OBJECT radiance into a pixel whose
    // primary ray missed → visible "eraser mark" trail along moving silhouettes.
    let primaryMissThisFrame = primaryHitBuf[pixelIdx].triIdx < 0;
    // isEnvPixel is the "skip-foveated-and-checker" predicate: treat both
    // persistent-sky pixels and object→sky transitions as env-like so neither
    // skip fires on them.
    let isEnvPixel = prevWasEnv || primaryMissThisFrame;

    // --- Material classification from previous frame's G-buffer ---
    // Used for material-aware bounce cap (camera motion) and foveated scheduling.
    // matClass: 0 = specular (glass/metal/mirror), 1 = glossy, 2 = rough diffuse, 3 = sky
    var matClass = 3u;  // default: sky/unknown → conservative (full bounces, no aggressive skip)
    if (!isEnvPixel) {
        let prevMatIdx = i32(textureLoad(hitMeshRead, pixel, 0).g);
        if (prevMatIdx >= 0) {
            let m0 = textureLoad(matData, vec2<i32>(prevMatIdx, 0), 0);   // .w = shininess (alpha = roughness^2)
            let m1 = textureLoad(matData, vec2<i32>(prevMatIdx, 1), 0);   // .y = metalness
            let m2 = textureLoad(matData, vec2<i32>(prevMatIdx, 2), 0);   // .w = transmission
            let shininess    = m0.w;
            let metalness    = m1.y;
            let transmission = m2.w;
            let isGlass  = transmission > 0.05;
            let isMetal  = metalness > 0.5;
            let isMirror = shininess < 0.05;
            let isGlossy = shininess < 0.25;
            if (isGlass || isMetal || isMirror) { matClass = 0u; }
            else if (isGlossy)                  { matClass = 1u; }
            else                                { matClass = 2u; }
        }
    }

    // Material-aware first-frame bounce cap.  On the very first sample (fc==0,
    // i.e. after forceReset) reduce bounces per material class to warm up faster.
    // Glass/metal/mirror need 4 (refraction chains), diffuse only needs 2.
    // Gated on clampsOn — setFireflyClamp(0) disables this for validation mode.
    var maxBounces = i32(rt.params.x);
    if (fc == 0u && rt.emissiveInfo.z < 1e20) {
        if      (matClass <= 0u) { maxBounces = min(maxBounces, 3); }  // specular
        else if (matClass == 1u) { maxBounces = min(maxBounces, 2); }  // glossy
        else if (matClass == 2u) { maxBounces = min(maxBounces, 1); }  // rough diffuse
        else                     { maxBounces = min(maxBounces, 3); }  // sky/unknown
    }

    // --- Foveated rendering: progressive spatial coarsening ---
    // Instead of temporal frame-skipping (which leaves stale pixels → ghosting
    // during camera motion), we reduce spatial resolution outside the center cone.
    // Pixels in the same NxN block share the trace from the block's top-left
    // corner (the "leader"). Every pixel gets a fresh value each frame — no
    // ghosting — but the periphery traces fewer unique rays.
    //
    // Zone layout (normalized distance from center):
    //   dist <= 0.30 : 1×1 (full resolution)
    //   dist <= 0.55 : 2×2 blocks
    //   dist >  0.55 : 4×4 blocks
    // During static camera: fall back to every-pixel tracing (fc > 0 handles
    // convergence naturally, no skip needed).
    let center = res * 0.5;
    let dxy = (vec2<f32>(f32(pixel.x), f32(pixel.y)) - center) / center;
    let dist = length(dxy);
    let camMovingFov = (u32(rt.params.w) & 2u) != 0u;
    var fovBlockSize = 1u;
    if (foveatedOn && camMovingFov && !isEnvPixel) {
        if (dist > 0.55) {
            fovBlockSize = 4u;
            maxBounces = min(maxBounces, 1);  // periphery: direct light only
        } else if (dist > 0.30) {
            fovBlockSize = 2u;
            maxBounces = min(maxBounces, 2);  // middle: one indirect bounce
        }
    }
    // Snap to block leader (top-left pixel of the block)
    let fovLeader = vec2<i32>(
        i32((u32(pixel.x) / fovBlockSize) * fovBlockSize),
        i32((u32(pixel.y) / fovBlockSize) * fovBlockSize));
    let isFovLeader = all(pixel == fovLeader);
    let foveatedSkip = fovBlockSize > 1u && !isFovLeader;

    // Checkerboard skip: during camera motion skip half the pixels each frame,
    // alternating which half via globalFrameCounter so both patterns are covered across
    // two consecutive frames.
    //
    // Fires during rotation, pan, and orbit.  Disabled for forward/backward dolly
    // motion: dolly changes depth of every pixel proportionally, and the screen-
    // space parallax scales inversely with depth, so near geometry moves many
    // pixels per frame — a single alternation can't reconstruct that cleanly and
    // produces visible ghosting on close surfaces.  Pan and orbit have motion
    // roughly perpendicular to the view direction, so screen-space parallax is
    // small and uniform across depths, safe for checkerboarding.
    //
    // Staleness during allowed motion is handled by the follower-copy branch
    // capping pixelFC to 4 at stamp time: the next fresh sample at that pixel
    // gets ~20% EMA weight and the stale value decays in ~5 frames.
    // Foveated coarsening still excludes checker to avoid compounding sparsity.
    let camMovedEarly = (u32(rt.params.w) & 2u) != 0u;  // params.w bit 1 = camMoved
    let dTrans = rt.camOri.xyz - rt.prevCamOri.xyz;
    let tLen = length(dTrans);
    // "Dolly" = translation aligned with view axis.  cos(45°)≈0.7.  Pan is
    // perpendicular (ratio ≈ 0), orbit is tangential (ratio ≈ 0), dolly is
    // parallel (ratio ≈ 1).
    let dollyRatio = select(0.0, abs(dot(dTrans, rt.camFwd.xyz)) / tLen, tLen > 1e-5);
    let isDolly = dollyRatio > 0.7;
    let checkerSkip = !foveatedOn && camMovedEarly && !isDolly && !isEnvPixel &&
        ((u32(pixel.x) + u32(pixel.y) + u32(rt.params.y)) & 1u) == 0u;

    // Foveated spatial coarsening: follower pixels copy from the block leader's
    // previous-frame result.  The leader itself always traces fresh this frame.
    // This gives spatially coherent blocks with at most 1-frame latency — no
    // multi-frame ghosting.
    //
    // Checkerboard skip: pass through previous accumulation unchanged.
    if (foveatedSkip || checkerSkip) {
        // Foveated followers read from the leader; checkerboard reads from self.
        let src = select(pixel, fovLeader, foveatedSkip);
        // Cap history on leader-copied samples.  The follower never truly sampled
        // its own world position — its color is spatially stamped from the leader.
        // Storing the leader's pixelFC (possibly 256) would give the next fresh
        // sample only ~0.4% weight at the moment motion ends, burning in whatever
        // noisy value the last foveated leader produced.  Cap to 4 so the first
        // post-motion sample gets ~20% weight and any outlier decays in ~20 frames.
        let leaderDiff  = textureLoad(diffAccumRead, src, 0);
        let leaderSpec  = textureLoad(specAccumRead, src, 0);
        let histCap = 4.0;
        textureStore(diffAccumWrite, pixel, vec4<f32>(leaderDiff.xyz,  min(leaderDiff.w,  histCap)));
        textureStore(specAccumWrite, pixel, vec4<f32>(leaderSpec.xyz,  min(leaderSpec.w,  histCap)));
        textureStore(hitMeshWrite, pixel, textureLoad(hitMeshRead, src, 0));
        textureStore(gBufWrite,    pixel, textureLoad(gBufRead, src, 0));
        textureStore(albedoWrite,  pixel, vec4<f32>(vec3<f32>(0.0), 0.0));
        textureStore(momentsWrite, pixel, textureLoad(momentsRead, src, 0));
        if (rt.restirParams.x > 0.5) {
            textureStore(reservoirWrite,  pixel, textureLoad(reservoirRead,  src, 0));
            textureStore(reservoirWWrite, pixel, textureLoad(reservoirWRead, src, 0));
        }
        // Mark this pixel as skipAccum so rt_bounces_main doesn't re-accumulate
        // on top of the leader-copied values.  Only the flag bits matter; other
        // PathStateEntry fields are left zero (harmless because skipAccum
        // short-circuits the bounce + accumulation path).
        {
            var entry: PathStateEntry;
            let flagBits = 32u;  // bit 5 = skipAccum
            entry.w0 = vec4<f32>(0.0);
            entry.w1 = vec4<f32>(0.0);
            entry.w2 = vec4<f32>(0.0, 0.0, 0.0, f32(flagBits));
            entry.w3 = vec4<f32>(0.0);
            entry.w4 = vec4<f32>(0.0);
            entry.w5 = vec4<f32>(0.0);
            entry.w6 = vec4<f32>(0.0);
            entry.w7 = vec4<f32>(0.0);
            entry.w8 = vec4<f32>(0.0);
            entry.w9 = vec4<f32>(0.0);
            entry.w10 = vec4<f32>(0.0);
            entry.w11 = vec4<f32>(0.0);
            pathStateBuf[pixelIdx] = entry;
        }
        if (rt.spp.x > 0.5) {
            textureStore(giResWrite,   pixel, textureLoad(giResRead,   src, 0));
            textureStore(giResWWrite,  pixel, textureLoad(giResWRead,  src, 0));
            textureStore(giResLoWrite, pixel, textureLoad(giResLoRead, src, 0));
        }
        return;
    }

    // AOV mode: only need primary-hit geometry data — skip all secondary bounces.
    // Mode 6 also caps to 1 bounce (geometry is all the heatmap needs).
    var varianceReducedBounces = maxBounces;
    let aovMode = i32(rt.emissiveInfo.w);
    if (aovMode > 0) { varianceReducedBounces = 1; }

    // pixelHistory and camMovedNow are consumed by the AOV mode 6 eligibility
    // check below.
    let pixelHistory = textureLoad(diffAccumRead, pixel, 0).w;
    let camMovedNow  = (u32(rt.params.w) & 2u) != 0u;

    // Spatio-temporal blue noise (Heitz & Belcour 2019).
    // Camera jitter and the integrator both consume the same per-pixel BN state
    // (defined in csCommonWGSL): R2/golden-ratio QMC across frames+dims, with a
    // per-pixel Cranley-Patterson rotation that has blue-noise spectral falloff.
    // Sub-pixel jitter centred on pixel centre (0.5, 0.5), range ±0.375.
    //
    // Jitter is always on: blue-noise decorrelation keeps sub-pixel offsets
    // spatio-temporally varied, so neighbouring pixels' stored gbuf values
    // differ by bounded (~1% depth, <5° normal on curved surfaces) amounts
    // that the à-trous edge-stops (depth scale 4, albedo-similarity stop)
    // tolerate. The RT accumulation's bilinear reprojection absorbs the
    // worldPos sub-pixel mismatch, accumulating correctly toward the true
    // super-sampled mean.
    bnInit(u32(pixel.x), u32(pixel.y), fc);
    let camBn = bnNext2d();
    let jx = (camBn.x - 0.5) * 0.75;
    let jy = (camBn.y - 0.5) * 0.75;
    // PCG seed kept for the many integrator dims that don't yet use BN
    // (RR termination, lobe selection, ReSTIR M-sampling, etc.).
    var seed = pcg(pcg(u32(pixel.x) + u32(pixel.y) * 65537u) + fc * 12979u);

    let apBn = bnNext2d();
    let ray = makeRay(vec2<f32>(f32(pixel.x) + 0.5 + jx, f32(pixel.y) + 0.5 + jy), res, apBn);

    // Read the primary hit from the buffer populated by rt_primary_main.
    let packed = primaryHitBuf[pixelIdx];

    // Sky fast-path (kernel-split optimisation): skip pathTrace entirely for
    // primary-miss pixels.  A full pathTrace call for a miss costs ~function
    // call + loop setup + default-Hit construction + miss branch + output-param
    // writes.  Handling it inline saves register allocation for the function-
    // scope Hit struct (~20 vec/scalar fields) and lets the thread pull the
    // next queued pixel sooner.
    //
    // This also handles AOV modes for sky (aovColor is black/zero for all
    // modes since there's no geometry).  The full AOV write path below is
    // skipped in favour of a minimal sentinel write set.
    if (packed.triIdx < 0) {
        let bgColor = sampleBackground(ray.dir);
        let lum3    = vec3<f32>(0.2126, 0.7152, 0.0722);
        // NaN guard
        let bgClean = select(vec3<f32>(0.0), bgColor, bgColor.x == bgColor.x);
        let bgLum   = dot(bgClean, lum3);

        // Progressive accumulation for sky.  Differs from the general accum
        // path because the sky fast-path doesn't reproject — it blends current
        // bgColor into prev-pixel's stale radiance.  Several things conspire
        // against naive progressive blend here:
        //
        //   1. Disocclusion (prev=object, now=sky): old contains object color,
        //      blending at alpha=1/(pxFC+1) leaves a slow-fading object ghost.
        //      Fix: reset pxFC when prev gBuf depth was > 0.
        //
        //   2. Camera rotation: sky fast-path doesn't reproject, so old is the
        //      env radiance along the PREV camera's ray through this pixel, not
        //      this pixel's current ray direction.  Blending two different env
        //      directions smears the sky.  Fix: reset pxFC when camera moved.
        //
        //   3. Historical contamination: if the accum buffer already has ghost
        //      baked in (from an earlier bug or cold-start warmup), fading at
        //      alpha=1/1025 takes thousands of frames.  Fix: cap sky pxFC at 32
        //      so any stale value decays within ~1s at 30fps.  Sky is
        //      deterministic modulo sub-pixel jitter; 32 samples is plenty for
        //      env-map jitter anti-aliasing.
        let prev   = textureLoad(diffAccumRead, pixel, 0);
        let old    = prev.xyz;
        var pxFC   = prev.w;
        let forceRst = (u32(rt.params.w) & 1u) != 0u;
        let camMovedFlag = (u32(rt.params.w) & 2u) != 0u;
        if (forceRst) { pxFC = 0.0; }
        // (1) object → sky transition.
        // Reset ONLY when the leaving object was moving (real disocclusion).
        // For static-scene silhouette jitter (prev object static, current ray
        // missed by sub-pixel displacement), do NOT reset — both outcomes are
        // valid sub-samples of a pixel whose true color is a mix. Letting FC
        // grow lets alpha shrink, and the running average converges to the
        // true sub-pixel coverage. The historical-contamination concern from
        // the old unconditional reset is superseded by forceReset (manual
        // clear) and camMovedFlag / mesh-moved gating.
        let prevMeshSky = u32(textureLoad(hitMeshRead, pixel, 0).r);
        let prevMovedSky = prevMeshSky < 128u && isMeshMoved(i32(prevMeshSky));
        if (!prevWasEnv && prevMovedSky) { pxFC = 0.0; }
        if (camMovedFlag) { pxFC = 0.0; }  // (2) no reprojection → stale env direction
        let alpha = 1.0 / (pxFC + 1.0);
        let blended = old * (1.0 - alpha) + bgClean * alpha;
        let newFC   = pxFC + 1.0;
        // Sky is all diffuse (no specular component).
        textureStore(diffAccumWrite, pixel, vec4<f32>(blended, newFC));
        textureStore(specAccumWrite, pixel, vec4<f32>(vec3<f32>(0.0), newFC));
        textureStore(momentsWrite,   pixel, vec4<f32>(bgLum, bgLum * bgLum, 0.0, 0.0));

        // Sentinel metadata — matches what pathTrace would write for a miss.
        textureStore(hitMeshWrite, pixel, vec4<f32>(128.0, -1.0, 0.0, 0.0));
        textureStore(albedoWrite,  pixel, vec4<f32>(1.0, 1.0, 1.0, 1.0));
        textureStore(gBufWrite,    pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));

        // Clear reservoirs on miss (mirrors pathTrace's i==0 miss handling).
        if (rt.restirParams.x > 0.5) {
            textureStore(reservoirWrite,  pixel, vec4<f32>(0.0));
            textureStore(reservoirWWrite, pixel, vec4<f32>(0.0));
        }
        if (rt.spp.x > 0.5) {
            textureStore(giResWrite,   pixel, vec4<f32>(0.0));
            textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
            textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
        }
        // Sky fast-path wrote accum itself — mark skipAccum for rt_bounces_main.
        {
            var entry: PathStateEntry;
            let flagBits = 32u;  // bit 5 = skipAccum
            entry.w0 = vec4<f32>(0.0);
            entry.w1 = vec4<f32>(0.0);
            entry.w2 = vec4<f32>(0.0, 0.0, 0.0, f32(flagBits));
            entry.w3 = vec4<f32>(0.0);
            entry.w4 = vec4<f32>(0.0);
            entry.w5 = vec4<f32>(0.0);
            entry.w6 = vec4<f32>(0.0);
            entry.w7 = vec4<f32>(0.0);
            entry.w8 = vec4<f32>(0.0);
            entry.w9 = vec4<f32>(0.0);
            entry.w10 = vec4<f32>(0.0);
            entry.w11 = vec4<f32>(0.0);
            pathStateBuf[pixelIdx] = entry;
        }
        return;
    }

    let rh = RawHit(packed.t, packed.triIdx, packed.u, packed.v);
    var primaryHit = loadHitMaterial(rh, ray);

    var primaryMeshIdx: u32;
    var primaryNormal:  vec3<f32>;
    var primaryDepth:   f32;
    var primaryAlbedo:  vec3<f32>;
    var primaryRough:   f32;
    var primaryMatIdx:  i32;
    var primaryTriIdx:  i32;
    var touchedMoved:   bool = false;

    // Run primary shading (bounce 0 NEE/ReSTIR DI + first BRDF sample).
    let shadeResult = primaryShade(primaryHit, ray, &seed, pixel, varianceReducedBounces,
                                   &primaryMeshIdx, &primaryNormal, &primaryDepth,
                                   &primaryAlbedo, &primaryRough, &primaryMatIdx, &primaryTriIdx);

    // Serialize PrimaryShadeResult -> PathStateEntry for rt_bounces_main.  Packed
    // per the layout declared at binding 37.  Seed captured AFTER primaryShade so
    // rt_bounces_main resumes from the same PCG state primaryShade left behind.
    // touchedMoved is initialized here (primary never sets it) and updated
    // in-place in rt_bounces_main via bit 2 of the flags word.
    {
        let flagBits = (select(0u, 1u, shadeResult.firstBounceSpec))
                     | (select(0u, 2u, shadeResult.afterTransmission))
                     | (select(0u, 4u, touchedMoved))
                     | (select(0u, 8u, shadeResult.pathAlive))
                     | (select(0u, 16u, shadeResult.giResStored));
        var entry: PathStateEntry;
        // seed | 0x00800000u forces the lowest exponent bit set so the stored f32
        // is never subnormal — drivers that flush-denormals-to-zero in storage ops
        // would otherwise zero it out for ~0.2% of seeds.  Recovered on read with
        // the same OR (idempotent: the bit is always 1 in both store and load).
        entry.w0  = vec4<f32>(shadeResult.rayOrigin, bitcast<f32>(seed | 0x00800000u));
        // Small positive ints bitcast to f32 produce subnormals (e.g., matIdx=2 →
        // 2.8e-45).  Some WebGPU drivers flush subnormals to zero in storage
        // buffer ops, causing matIdx/meshIdx/effectiveBounces/flagBits to read
        // back as 0.  Use plain f32 cast for small ints; reserve bitcast for
        // seeds (full 32-bit u32 with subnormal guard applied above).
        entry.w1  = vec4<f32>(shadeResult.rayDir,    f32(shadeResult.effectiveBounces));
        entry.w2  = vec4<f32>(shadeResult.throughput, f32(flagBits));
        entry.w3  = vec4<f32>(shadeResult.diffRad,    shadeResult.prevMetalness);
        entry.w4  = vec4<f32>(shadeResult.specRad,    shadeResult.prevAlpha);
        entry.w5  = vec4<f32>(shadeResult.prevNormal, shadeResult.primaryDepth);
        entry.w6  = vec4<f32>(shadeResult.prevWo,     0.0);
        entry.w7  = vec4<f32>(shadeResult.b0Point,    shadeResult.b0Alpha);
        entry.w8  = vec4<f32>(shadeResult.b0Normal,   shadeResult.b0Metal);
        entry.w9  = vec4<f32>(shadeResult.b0Wo,       f32(shadeResult.b0MeshIdx));
        entry.w10 = vec4<f32>(shadeResult.b0Albedo,   f32(shadeResult.primaryMatIdx));
        entry.w11 = vec4<f32>(shadeResult.b0F0,       0.0);
        pathStateBuf[pixelIdx] = entry;
    }

    // runBounces + radiance accumulation live in rt_bounces_main.  rt_main's
    // responsibility ends with serialize + primary-metadata texture writes.
    // AOV / sky / foveated paths short-circuit earlier because they need
    // primary-derived data (primaryDepth/etc) that are rt_main locals.

    // AOV visualization mode — write noise-free primary-hit data and return early.
    // Writes to diffAccumWrite (display shader reads diffTex + specTex).
    // Modes 10+ are POST-bounce diagnostic AOVs handled in rt_accum_main — let
    // those flow through the normal pipeline.
    if (aovMode > 0 && aovMode < 10) {
        var aovColor = vec3<f32>(0.0);
        if (aovMode == 1)      { aovColor = vec3<f32>(primaryDepth / (primaryDepth + 1.0)); }
        else if (aovMode == 2) { aovColor = primaryNormal * 0.5 + 0.5; }
        else if (aovMode == 3) { aovColor = primaryAlbedo; }
        else if (aovMode == 4) { aovColor = vec3<f32>(f32(primaryMeshIdx % 32u) / 32.0,
                                                       f32((primaryMeshIdx * 7u) % 32u) / 32.0,
                                                       f32((primaryMeshIdx * 13u) % 32u) / 32.0); }
        else if (aovMode == 5) { aovColor = vec3<f32>(primaryRough); }
        else if (aovMode == 6) {
            // Use actual per-pixel roughness (includes roughnessMap texture, not just base value).
            // threshold: roughness² > 0.05  ↔  roughness > 0.224  (matches isMirror in pathTrace)
            // Sky pixels: primaryDepth == 0 → always blue (no geometry, no saving)
            let isEligible = primaryRough > 0.224 && primaryDepth > 0.0 && !camMovedNow;
            aovColor = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), isEligible);
        }
        else if (aovMode == 7) {
            // Focus-plane visualization: green at the focal plane, red in front
            // (closer than focus), blue behind. Makes DOF tuning a visual task —
            // slide focusDistance until the subject glows green.
            let focusDist = rt.lens.y;
            let rel = (primaryDepth - focusDist) / max(focusDist, 0.01);
            let r   = clamp(-rel, 0.0, 1.0);
            let b   = clamp( rel, 0.0, 1.0);
            let g   = 1.0 - min(abs(rel), 1.0);
            aovColor = select(vec3<f32>(r, g, b), vec3<f32>(0.0), primaryDepth <= 0.0);
        }
        // Preserve pixelHistory in .w so adaptive bounce reduction keeps accumulating
        // across AOV mode 6 frames and doesn't reset to 1 each frame.
        textureStore(diffAccumWrite, pixel, vec4<f32>(aovColor, pixelHistory + 1.0));
        textureStore(specAccumWrite, pixel, vec4<f32>(0.0, 0.0, 0.0, pixelHistory));
        textureStore(hitMeshWrite, pixel, vec4<f32>(f32(primaryMeshIdx), f32(primaryMatIdx), 0.0, 0.0));
        textureStore(gBufWrite, pixel, vec4<f32>(primaryNormal, primaryDepth));
        textureStore(albedoWrite, pixel, vec4<f32>(primaryAlbedo, primaryRough));
        // AOV already wrote accum — OR skipAccum into the flags word written by
        // the earlier serialize.  Direct overwrite of pathStateBuf[pixelIdx].w2.w
        // preserves other bits we don't care about (path terminates at AOV).
        {
            let prevFlags = u32(pathStateBuf[pixelIdx].w2.w);
            pathStateBuf[pixelIdx].w2.w = f32(prevFlags | 32u);
        }
        return;
    }

    // accumulation + hitMesh write live in rt_bounces_main (hitMesh depends on
    // touchedMoved).  rt_main still writes primary-only metadata here.
    textureStore(albedoWrite,  pixel, vec4<f32>(primaryAlbedo, primaryRough));

    textureStore(gBufWrite, pixel, vec4<f32>(primaryNormal, primaryDepth));
    }  // end of per-pixel block
}

// ---------------------------------------------------------------------------
// Primary-hit kernel (kernel split, step 1 of 2).
// Traces the camera ray through the BVH and writes a compact RawHit record
// (triIdx, t, u, v) to primaryHitBuf.  No material lookup, no shading, no
// bounces — those happen in rt_main, which reconstructs the full Hit via
// loadHitMaterial.  Splitting primary traversal off the megakernel reduces
// register pressure in both halves: this kernel has only BVH state (~50
// registers), and rt_main no longer needs to carry primary-traversal
// locals alongside its shading/bounce state.
//
// Seeding note: this kernel consumes one bnNext2d() for camera jitter and may
// consume PCG randomness inside stochastic alpha tests in testTriangle().
// rt_main re-seeds from the same pixel+frame values (same initial seed) and
// re-derives the same ray — the primary is NOT re-traced, so correlation is
// inert.  See bnInit/seed setup in rt_main for the matching convention.
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn rt_primary_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res   = rt.iRes.xy;
    let resXu = u32(res.x);
    // Direct pixel assignment: no compaction has happened yet, every pixel needs
    // tracing.  2-D tiled dispatch (ceil(w/8), ceil(h/8)) gives each workgroup an
    // 8×8 tile → BVH traversal shares cache across adjacent rays.
    let pixel = vec2<i32>(gid.xy);
    if (pixel.x >= i32(res.x) || pixel.y >= i32(res.y)) { return; }
    let pixelIdx = u32(pixel.y) * resXu + u32(pixel.x);
    {
        let fc = u32(rt.frameCount.x);
        // Jitter derivation must match rt_main exactly so both kernels trace the
        // same camera ray for this pixel.  sceneHitRaw's stochastic alpha uses
        // its own PCG seeded from (triIdx, frame, t) — it doesn't consume any
        // thread-local seed, so no seed variable is needed in this kernel.
        bnInit(u32(pixel.x), u32(pixel.y), fc);
        let camBn = bnNext2d();
        let jx = (camBn.x - 0.5) * 0.75;
        let jy = (camBn.y - 0.5) * 0.75;

        let apBn = bnNext2d();
        let ray = makeRay(vec2<f32>(f32(pixel.x) + 0.5 + jx, f32(pixel.y) + 0.5 + jy), res, apBn);
        let rh = sceneHitRaw(ray, 1e30);

        var packed: PrimaryHitPacked;
        packed.triIdx = rh.triIdx;
        packed.t      = rh.t;
        packed.u      = rh.u;
        packed.v      = rh.v;
        primaryHitBuf[pixelIdx] = packed;

        // Vestigial: keeps binding 37 (pathStateBuf) in primaryPipeline's
        // bind-group layout so C++ setStorageBuffer(37, ...) stays valid for the
        // shared bind-group helpers.  rt_main overwrites this entry immediately
        // in the next pass, so the value doesn't matter.
        pathStateBuf[pixelIdx].w0 = vec4<f32>(0.0);
    }
}

// ---------------------------------------------------------------------------
// rt_compact_main (Stage F1).
// Reads pathStateBuf flags, writes pixel indices of non-skipAccum paths to
// aliveQueue via atomic append.  rt_bounces_main then iterates aliveQueue
// instead of all pixels — warps are packed with live work, no wasted lanes
// on skipAccum pixels (foveated/checker/sky fast-path/AOV already wrote their
// own accum in rt_main and don't need any further work).
//
// Persistent-thread pattern: same as rt_main.  Uses compactCounter to pull
// work (atomic input-steal) and aliveCount to append (atomic output).
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn rt_compact_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = rt.iRes.xy;
    let totalPixels = u32(res.x) * u32(res.y);
    loop {
        let pixelIdx = atomicAdd(&compactCounter, 1u);
        if (pixelIdx >= totalPixels) { break; }
        let entry = pathStateBuf[pixelIdx];
        let flagBits = u32(entry.w2.w);
        let skipAccum = (flagBits & 32u) != 0u;
        if (!skipAccum) {
            let slot = atomicAdd(&aliveCount, 1u);
            aliveQueue[slot] = pixelIdx;
            // F2c: count for bucket-sort before bounce1.  primaryMatIdx is
            // stored at entry.w10.w (i32 bitcast).  Hash via low 8 bits →
            // 256 buckets.  Dead-at-primary pixels get counted too and end
            // up grouped together in the sorted queue — bounce1 early-exits
            // on them so coherent grouping is a net win.
            let matIdx = i32(entry.w10.w);
            let bucket = u32(matIdx) & 0xFFu;
            atomicAdd(&matBucketCount[bucket], 1u);
        }
    }
}

// ---------------------------------------------------------------------------
// rt_sort_prefix_main (F2c: exclusive prefix sum over material buckets).
// Dispatch 1 workgroup of 1 thread.  256 serial adds — trivial cost.
// Side effect: zeroes matBucketCount so the scatter pass can reuse it as
// a per-bucket fill counter via atomicAdd.
// ---------------------------------------------------------------------------
@compute @workgroup_size(1)
fn rt_sort_prefix_main() {
    var sum: u32 = 0u;
    for (var i = 0u; i < 256u; i = i + 1u) {
        let cnt = atomicLoad(&matBucketCount[i]);
        matBucketOffset[i] = sum;
        sum = sum + cnt;
        atomicStore(&matBucketCount[i], 0u);
    }
}

// ---------------------------------------------------------------------------
// rt_sort_scatter_main (F2c: bucket-sort aliveQueue by primaryMatIdx).
// Reads each aliveQueue entry's matIdx from pathStateBuf, scatters the pixel
// index into sortedAliveQueue at offset[bucket] + atomicAdd(count[bucket]).
// After this pass, bounce1 reads sortedAliveQueue so warp lanes execute
// materially-coherent BRDF code.
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn rt_sort_scatter_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let aliveTotal = atomicLoad(&aliveCount);
    loop {
        let slot = atomicAdd(&sortCounter, 1u);
        if (slot >= aliveTotal) { break; }
        let pixelIdx = aliveQueue[slot];
        let matIdx = i32(pathStateBuf[pixelIdx].w10.w);
        let bucket = u32(matIdx) & 0xFFu;
        let outSlot = matBucketOffset[bucket] + atomicAdd(&matBucketCount[bucket], 1u);
        sortedAliveQueue[outSlot] = pixelIdx;
    }
}

// ---------------------------------------------------------------------------
// rt_bounces_main (F2b: reads alive1Queue — bounce1-survivors only).
// Consumes PathStateEntry entries that rt_bounce1_main wrote back with
// pathAlive=true, runs the bounce loop starting at i=2 (runBounces), and
// writes the updated diffRad / specRad / touchedMoved back to pathStateBuf.
// Accumulation has MOVED to rt_accum_main which runs over aliveQueue (all
// non-skipAccum pixels — including dead-at-primary and dead-at-bounce1).
//
// The work reduction is the whole point of F2b: on Bistro only ~12% of
// paths survive bounce 1, so this kernel's dispatch is 8× smaller than
// rt_bounce1_main's.
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn rt_bounces_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = rt.iRes.xy;
    let aliveTotal = atomicLoad(&alive1Count);
    let resXu = u32(res.x);
    loop {
        // Pull from the compacted bounce1-survivor queue.
        let slot = atomicAdd(&bounceCounter, 1u);
        if (slot >= aliveTotal) { break; }
        let pixelIdx = alive1Queue[slot];
        let pixel = vec2<i32>(i32(pixelIdx % resXu), i32(pixelIdx / resXu));
        // Restore per-pixel BN state (same reason as rt_bounce1_main: each kernel
        // dispatch starts with bnPx=bnPy=bnFc=0, so without this call, runBounces'
        // sampleEmissiveTriCdf/cosineHemisphere would use identical samples for
        // every pixel every frame).
        // bnDim=16: skip primary's ~7 dims + bounce1's ~8 dims so bounce2's samples
        // don't alias earlier bounces' sample dimensions.
        bnInit(u32(pixel.x), u32(pixel.y), u32(rt.frameCount.x));
        bnDim = 16u;

        let entry = pathStateBuf[pixelIdx];
        let flagBits = u32(entry.w2.w);

        // Deserialize PrimaryShadeResult from PathStateEntry.  All paths in
        // alive1Queue have pathAlive=true (bounce1 kernel only appends survivors),
        // so no pathAlive guard needed here.
        var state: PrimaryShadeResult;
        state.rayOrigin        = entry.w0.xyz;
        var seed               = bitcast<u32>(entry.w0.w) | 0x00800000u;
        state.rayDir           = entry.w1.xyz;
        state.effectiveBounces = i32(entry.w1.w);
        state.throughput       = entry.w2.xyz;
        state.firstBounceSpec  = (flagBits & 1u)  != 0u;
        state.afterTransmission = (flagBits & 2u) != 0u;
        var touchedMoved       = (flagBits & 4u)  != 0u;
        state.pathAlive        = true;
        state.giResStored      = (flagBits & 16u) != 0u;
        state.diffRad          = entry.w3.xyz;
        state.prevMetalness    = entry.w3.w;
        state.specRad          = entry.w4.xyz;
        state.prevAlpha        = entry.w4.w;
        state.prevNormal       = entry.w5.xyz;
        state.primaryDepth     = entry.w5.w;
        state.prevWo           = entry.w6.xyz;
        state.b0Point          = entry.w7.xyz;
        state.b0Alpha          = entry.w7.w;
        state.b0Normal         = entry.w8.xyz;
        state.b0Metal          = entry.w8.w;
        state.b0Wo             = entry.w9.xyz;
        state.b0MeshIdx        = i32(entry.w9.w);
        state.b0Albedo         = entry.w10.xyz;
        state.primaryMatIdx    = i32(entry.w10.w);
        state.b0F0             = entry.w11.xyz;

        // Run bounces 2..N.  runBounces' for-loop starts at i=2 post-F2a.
        let primaryMeshIdx_u   = u32(state.b0MeshIdx);
        let bouncesResult = runBounces(state, &seed, pixel, primaryMeshIdx_u, &touchedMoved);
        let finalDiff = state.diffRad + bouncesResult.diff;
        let finalSpec = state.specRad + bouncesResult.spec;

        // Write back to pathStateBuf.  rt_accum_main reads diffRad/specRad +
        // flagBits.touchedMoved from here.  We only modify the fields that
        // can change; everything else (b0*, primaryDepth, primaryMatIdx) is
        // already up to date from primaryShade / bounce1.
        var flagBitsOut = flagBits;
        if (touchedMoved) { flagBitsOut |= 4u; }

        var newEntry = entry;
        newEntry.w2 = vec4<f32>(entry.w2.xyz, f32(flagBitsOut));
        newEntry.w3 = vec4<f32>(finalDiff, entry.w3.w);
        newEntry.w4 = vec4<f32>(finalSpec, entry.w4.w);
        pathStateBuf[pixelIdx] = newEntry;
    }
}

