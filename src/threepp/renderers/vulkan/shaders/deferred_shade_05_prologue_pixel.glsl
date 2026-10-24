// main() prologue, part 1 of 4 (included INSIDE main(), not at file scope):
// the pixel, the MSAA tap selection and dispatch B gate, the jitter-aware
// primary ray and its globals, the instance id and the G-buffer's normal and
// albedo taps. Every shade kernel opens with the same text, so the values it
// derives agree to the bit between kernels; the parts are split where the
// shade itself early-outs (sky, unlit, ocean), and each kernel puts its own
// early-out between them.
    const ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (px.x >= int(pc.width) || px.y >= int(pc.height)) return;
    const vec2 size = vec2(float(pc.width), float(pc.height));
    const vec2 uv   = (vec2(px) + 0.5) / size;

    // ── MSAA sample count (setGbufferMsaa; 0=1x/1=2x/2=4x, pc.flags bits 5-6) ──
    const uint msaaCode = (effFlags() >> 5u) & 0x3u;
    const int  K = (msaaCode == 2u) ? 4 : (msaaCode == 1u ? 2 : 1);

    // ── Dispatch B (shadeMode==1, MSAA edge shading) gate + tap selection ──
    // ids.w's bit 15 (packed by gbuf_resolve.comp) marks pixels where at
    // least one raster sample fell outside the dominant surface cluster — an
    // edge/silhouette pixel. Non-complex pixels are already fully correct
    // from dispatch A (the dominant sample IS the only sample), so bail
    // immediately: this keeps dispatch B's cost proportional to edge pixel
    // count, not frame resolution.
    // FALLBACK VARIANT (documented in the MSAA plan): rather than looping
    // every non-dominant cluster, this shades AT MOST the single largest
    // non-dominant cluster and folds the rest of the minority weight into
    // it — 2-cluster pixels (~99% of real edges) are exact; 3+-cluster
    // pixels get one representative sample standing in for the whole
    // minority, a documented approximation.
    const uint idsW0 = texelFetch(gbufIdsTex, px, 0).w;
    const bool isComplex = (idsW0 & 0x8000u) != 0u;
    if (pc.shadeMode == 1u && !isComplex) return;

    // Coverage accounting at complex pixels — the three weights sum to
    // exactly 1.0: dispatch A stores domWeight·lit PLUS skyEdgeW·sky (the
    // sky needs no per-sample shading, it is a background colour dispatch A
    // can evaluate itself — an earlier version skipped sky minorities
    // entirely, which forfeited edge AA on exactly the most visible edges,
    // every silhouette against the sky), and dispatch B adds geoMinW·lit
    // for the non-sky minority. When dispatch B is NOT active (flags bit 7
    // clear) the geometry-minority weight folds back into the dominant
    // surface so no energy is lost.
    const uint domMask0 = idsW0 & 0xFu;
    float domWeight = 1.0;
    float skyEdgeW  = 0.0;
    if (pc.shadeMode == 0u && isComplex && K > 1) {
        const uint domIdx0 = (idsW0 >> 4u) & 0x3u;
        int skyCount = 0, geoMinCount = 0;
        int repIdx = -1, repCount = 0;// dispatch B's minority representative (mirror its selection exactly)
        for (int i = 0; i < K; ++i) {
            if (i == int(domIdx0) || (domMask0 & (1u << uint(i))) != 0u) continue;
            if (texelFetch(gbufIdsMS, px, i).x == 0u) { ++skyCount; continue; }
            ++geoMinCount;
            int count = 1;
            for (int j = i + 1; j < K; ++j) {
                if (j == int(domIdx0) || (domMask0 & (1u << uint(j))) != 0u) continue;
                if (texelFetch(gbufIdsMS, px, j).x == texelFetch(gbufIdsMS, px, i).x) ++count;
            }
            if (count > repCount) { repCount = count; repIdx = i; }
        }
        skyEdgeW = float(skyCount) / float(K);
        // Reserve the geometry-minority weight ONLY if dispatch B will actually
        // pay it back. B bails without accumulating on UNLIT and TRANSMISSIVE
        // minority representatives (ocean water, glass, alpha and additive
        // blends: their shade paths re-trace what is behind them and own
        // dispatch-A-only temporal state) — reserving weight B never pays is a
        // dark rim on exactly those edges, so mirror B's rep selection +
        // early-outs and fold the weight back into the dominant surface in
        // those cases. (Glass used to be reserved for and then paid with its
        // OPAQUE shade, the transmission skipped: a thin stem in front of a
        // pane at MSAA 2/4 turned into a white dotted line.)
        float geoMinW = 0.0;
        if (flagOn(128u) && repIdx >= 0) {
            const uint repId    = texelFetch(gbufIdsMS, px, repIdx).x;
            const bool repUnlit = texelFetch(gbufNormalMS, px, repIdx).w < 0.0;
            const bool repTransmissive = mats[repId - 1u].transmission > 0.0;
            if (!repUnlit && !repTransmissive) geoMinW = float(geoMinCount) / float(K);
        }
        domWeight = 1.0 - skyEdgeW - geoMinW;
    }

    int msSampleIdx = -1;// >=0 selects the MS per-sample taps below
    float msWeight = 1.0;// dispatch-B contribution weight (geo-minority population / K)
    if (pc.shadeMode == 1u) {
        const uint domMask = idsW0 & 0xFu;
        const uint domIdx  = (idsW0 >> 4u) & 0x3u;
        int bestCluster = -1, bestCount = 0, geoMinCount = 0;
        for (int i = 0; i < K; ++i) {
            if (i == int(domIdx) || (domMask & (1u << uint(i))) != 0u) continue;// skip dominant cluster members
            if (texelFetch(gbufIdsMS, px, i).x == 0u) continue;// skip sky — dispatch A blends it as a background colour
            ++geoMinCount;
            int count = 1;
            for (int j = i + 1; j < K; ++j) {
                if (j == int(domIdx) || (domMask & (1u << uint(j))) != 0u) continue;
                if (texelFetch(gbufIdsMS, px, j).x == texelFetch(gbufIdsMS, px, i).x) ++count;
            }
            if (count > bestCount) { bestCount = count; bestCluster = i; }
        }
        if (bestCluster < 0) return;// pure sky/geometry edge — dispatch A handled the whole pixel
        msSampleIdx = bestCluster;
        // Weight = the ENTIRE non-sky minority population (any 3rd cluster
        // folds onto this representative — the documented fallback
        // approximation). Must mirror dispatch A's reserved geoMinW exactly
        // or the pixel gains/loses energy.
        msWeight = float(geoMinCount) / float(K);
    }

    // Jitter-aware NDC. The G-buffer is rasterized with a sub-pixel jittered
    // projection (TSR), so the depth at this pixel belongs to geometry whose
    // UNJITTERED NDC is offset by the jitter. Add it back before reconstructing
    // worldPos with the unjittered projInverse — without it, GRAZING surfaces
    // reconstruct OFF the surface (the depth gradient is steep, so a sub-pixel NDC
    // error becomes a large position error), the gatherEnv ray origin lands behind
    // the surface → every ray occluded → the surface renders BLACK. cam.jitter.xy
    // equals the gbuffer's Halton jitter (same haltonFrame_, same frame).
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0)) + cam.jitter.xy;
    // MSAA: the depth this pixel reconstructs from belongs to ONE raster
    // sample — the dominant one (dispatch A) or the minority representative
    // (dispatch B) — which sits at that sample's standard location inside
    // the pixel, not at the centre. Reconstructing it along the centre ray
    // lands OFF the surface by depth-slope × sample offset (up to 0.375 px),
    // which on a curved glass ball away from its centre exceeds the 1 mm
    // refraction epsilon: the refraction ray started OUTSIDE the glass,
    // re-hit the front face as its "exit" and escaped to the environment —
    // a glass sphere under MSAA 2x/4x showed sky/background everywhere but
    // a jagged central cap, worse the fewer pixels it covered (distance,
    // small windows). Cast the primary ray through the sample's own location
    // instead (standard Vulkan 2x/4x sample positions). Opaque surfaces at
    // grazing angles got the same off-surface origin for their shadow /
    // reflection rays; they simply tolerated it better.
    if (K > 1) {
        const int sIdx = (msSampleIdx >= 0) ? msSampleIdx : int((idsW0 >> 4u) & 0x3u);
        vec2 off;// pixel units from the pixel centre, y down
        if (K == 2) off = (sIdx == 0) ? vec2(0.25, 0.25) : vec2(-0.25, -0.25);
        else        off = (sIdx == 0) ? vec2(-0.125, -0.375) : (sIdx == 1) ? vec2(0.375, -0.125)
                        : (sIdx == 2) ? vec2(-0.375, 0.125) : vec2(0.125, 0.375);
        ndc += vec2(off.x * 2.0 / float(pc.width), -off.y * 2.0 / float(pc.height));
    }
    // Where THIS pixel is looked at from. The eye under a perspective camera —
    // unchanged — and this pixel's own near-plane point under a parallel one,
    // which is what makes every camPos-relative term below (the view vector V,
    // the fog and murk legs, the volumetric march origins, the foam LOD
    // distance) come out right without each one growing its own ortho branch:
    // the shading point lies on this ray, so camPos → worldPos IS the view leg.
    const vec3 camPos = camRayOrigin(ndc);
    gPrimaryOrigin    = camPos;
    gPrimaryDir       = camRayDir(ndc);
    gPixelCoord       = px;
    gNearPointY       = camNearPoint(ndc).y;// this pixel's port point — MEDIUM tests only
    const float maxLod = float(max(pc.envMipCount, 1u) - 1u);

    const uint id = (msSampleIdx >= 0) ? texelFetch(gbufIdsMS, px, msSampleIdx).x
                                        : texelFetch(gbufIdsTex, px, 0).x;// 0 == sky

    const vec4  nTap   = (msSampleIdx >= 0) ? texelFetch(gbufNormalMS, px, msSampleIdx) : texelFetch(gbufNormalTex, px, 0);
    const vec4  am     = (msSampleIdx >= 0) ? texelFetch(gbufAlbedoMS, px, msSampleIdx) : texelFetch(gbufAlbedoTex, px, 0);
    const vec3  albedo = am.rgb;
