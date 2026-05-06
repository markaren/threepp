

// ---------------------------------------------------------------------------
// rt_accum_main (F2b: accumulation kernel).
// Reads pixel indices from aliveQueue (non-skipAccum pixels — ALL path
// outcomes: dead-at-primary, dead-at-bounce1, and bounce1-survivors).  Pulls
// final accumulated radiance + touchedMoved + primary hit info from
// pathStateBuf and runs the full accumulation / temporal-reprojection
// pipeline that previously lived inline at the end of rt_bounces_main.
//
// Register budget in this kernel is tiny (no BVH traversal, no BRDF
// evaluation), so splitting it out is pure win for rt_bounces_main's
// occupancy.
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn rt_accum_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = rt.iRes.xy;
    let aliveTotal = atomicLoad(&aliveCount);
    let resXu = u32(res.x);
    loop {
        let slot = atomicAdd(&accumCounter, 1u);
        if (slot >= aliveTotal) { break; }
        let pixelIdx = aliveQueue[slot];
        let pixel = vec2<i32>(i32(pixelIdx % resXu), i32(pixelIdx / resXu));

        let entry = pathStateBuf[pixelIdx];
        let flagBits = u32(entry.w2.w);

        // Deserialize only what the accumulation block needs.
        let diffRadFinal   = entry.w3.xyz;
        let specRadFinal   = entry.w4.xyz;
        let touchedMoved   = (flagBits & 4u) != 0u;
        let b0Point        = entry.w7.xyz;
        let b0Alpha        = entry.w7.w;
        let b0MeshIdx      = i32(entry.w9.w);
        let primaryMatIdx  = i32(entry.w10.w);
        let primaryDepth   = entry.w5.w;
        let primaryMeshIdx = u32(b0MeshIdx);
        let primaryRough   = sqrt(b0Alpha);

        // --- Fog application at primary-ray boundary ---
        // L_pixel = T(t)·L_surface + single_scatter(t)
        // Single-scattering samples a point in the volume and NEEs to lights so
        // fog glows near real sources rather than carrying a flat ambient tint.
        // Shadow rays pick up their own transmittance via traceShadowRay.
        // Miss pixels use a bounded 1e4 "sky distance" so inscatter still fades.
        var ptDiff = diffRadFinal;
        var ptSpec = specRadFinal;
        if (fogEnabled()) {
            let fogDist = select(primaryDepth, 1e4, primaryDepth <= 0.0);
            let T = fogTransmittance(fogDist);
            // Reconstruct primary ray direction (cheap; matches makeRay()).
            let aspect = res.x / res.y;
            let ndcFog = vec2<f32>(
                (f32(pixel.x) + 0.5) / res.x * 2.0 - 1.0,
                 1.0 - (f32(pixel.y) + 0.5) / res.y * 2.0);
            let rayDir = normalize(rt.camFwd.xyz
                                 + rt.camRgt.xyz * (ndcFog.x * rt.tanHalfFov.x * aspect)
                                 + rt.camUp.xyz  * (ndcFog.y * rt.tanHalfFov.x));
            // sampleEmissiveTriCdf uses BN state — initialize it here.
            bnInit(u32(pixel.x), u32(pixel.y), u32(rt.frameCount.x));
            var fogSeed = pcg(pixelIdx ^ u32(rt.frameCount.x) * 2654435761u);
            let volume = volumeInscatter(rt.camOri.xyz, rayDir, fogDist, &fogSeed);
            // Inscatter lands on the diffuse channel (phase is broad / isotropic).
            ptDiff = ptDiff * T + volume;
            ptSpec = ptSpec * T;
        }
        var sample = ptDiff + ptSpec;

        // ---------------------------------------------------------------
        // DIAG AOV modes 10+: dump specific pathStateBuf fields to accum
        // for per-pixel visual inspection.  Per-mesh banding in these
        // overlays tells us which field has mesh-dependent divergence.
        //   10: diffRadFinal (entry.w3.xyz)     — total diffuse radiance
        //   11: specRadFinal (entry.w4.xyz)     — total specular radiance
        //   12: touchedMoved bit                 — red = bit set
        //   13: flagBits split into RGB          — bits 0-2 / 3-5 / 6-8
        //   14: b0Point magnitude (mod 1)        — diag for primary hit pos
        //   15: primaryDepth normalized
        //   16: primaryMeshIdx as color
        //   17: primaryMatIdx as color
        //   18: b0Alpha (roughness²)
        {
            let aovMode = i32(rt.emissiveInfo.w);
            if (aovMode >= 10) {
                var dbg = vec3<f32>(0.0);
                if (aovMode == 10) {
                    dbg = diffRadFinal;
                } else if (aovMode == 11) {
                    dbg = specRadFinal;
                } else if (aovMode == 12) {
                    dbg = select(vec3<f32>(0.1, 0.1, 0.1), vec3<f32>(1.0, 0.0, 0.0), touchedMoved);
                } else if (aovMode == 13) {
                    dbg = vec3<f32>(
                        f32(flagBits & 7u) / 7.0,
                        f32((flagBits >> 3u) & 7u) / 7.0,
                        f32((flagBits >> 6u) & 7u) / 7.0);
                } else if (aovMode == 14) {
                    dbg = fract(abs(b0Point));
                } else if (aovMode == 15) {
                    dbg = vec3<f32>(primaryDepth / (primaryDepth + 10.0));
                } else if (aovMode == 16) {
                    dbg = vec3<f32>(
                        f32(primaryMeshIdx % 7u) / 7.0,
                        f32((primaryMeshIdx * 5u) % 11u) / 11.0,
                        f32((primaryMeshIdx * 11u) % 13u) / 13.0);
                } else if (aovMode == 17) {
                    dbg = vec3<f32>(
                        f32(u32(primaryMatIdx) % 7u) / 7.0,
                        f32((u32(primaryMatIdx) * 5u) % 11u) / 11.0,
                        f32((u32(primaryMatIdx) * 11u) % 13u) / 13.0);
                } else if (aovMode == 18) {
                    dbg = vec3<f32>(sqrt(b0Alpha));
                }
                textureStore(diffAccumWrite, pixel, vec4<f32>(dbg, 1.0));
                textureStore(specAccumWrite, pixel, vec4<f32>(0.0));
                textureStore(hitMeshWrite,   pixel, vec4<f32>(f32(primaryMeshIdx), f32(primaryMatIdx), 0.0, 0.0));
                continue;
            }
        }

        // === Accumulation + reprojection (moved verbatim from rt_bounces_main) ===
        let prevDiffRaw4 = textureLoad(diffAccumRead, pixel, 0);
        let prevSpecRaw  = textureLoad(specAccumRead, pixel, 0).xyz;
        let prevDiffRaw  = prevDiffRaw4.xyz;
        var pixelFC      = prevDiffRaw4.w;
        var oldDiff = cleanFinite3(prevDiffRaw);
        var oldSpec = cleanFinite3(prevSpecRaw);
        let prevMom = textureLoad(momentsRead, pixel, 0);
        var prevMomM1 = prevMom.x;
        var prevMomM2 = prevMom.y;

        let forceReset   = (u32(rt.params.w) & 1u) != 0u;
        let camMovedFlag = (u32(rt.params.w) & 2u) != 0u;
        if (forceReset) { pixelFC = 0.0; }

        let prevMeshU = u32(textureLoad(hitMeshRead, pixel, 0).r);
        let primaryMoved = primaryMeshIdx < 128u && isMeshMoved(i32(primaryMeshIdx));

        if (primaryMeshIdx != prevMeshU && prevMeshU < 128u && isMeshMoved(i32(prevMeshU))) {
            pixelFC = 0.0;
        }
        if (touchedMoved) {
            let emissiveMoved = rt.restirParams.z > 0.5;
            pixelFC = min(pixelFC, select(8.0, 2.0, emissiveMoved));
        }

        if (primaryMoved || camMovedFlag) {
            var reprojOk = false;
            if (pixelFC > 0.0 && primaryDepth > 0.0) {
                let worldPos = b0Point;
                var prevWorldPos = worldPos;
                if (primaryMoved) {
                    prevWorldPos = (rtMotionMats[primaryMeshIdx] * vec4<f32>(worldPos, 1.0)).xyz;
                }
                let toPoint = prevWorldPos - rt.prevCamOri.xyz;
                let prevZ   = dot(toPoint, rt.prevCamFwd.xyz);
                if (prevZ > 0.001) {
                    let aspect   = res.x / res.y;
                    let prevNdcX = dot(toPoint, rt.prevCamRgt.xyz) / (prevZ * rt.tanHalfFov.x * aspect);
                    let prevNdcY = dot(toPoint, rt.prevCamUp.xyz)  / (prevZ * rt.tanHalfFov.x);
                    let prevU = (prevNdcX + 1.0) * 0.5 * res.x;
                    let prevV = (1.0 - prevNdcY) * 0.5 * res.y;
                    let pxBase = vec2<i32>(i32(floor(prevU)), i32(floor(prevV)));
                    if (pxBase.x >= 0 && pxBase.x + 1 < i32(res.x) &&
                        pxBase.y >= 0 && pxBase.y + 1 < i32(res.y)) {
                        let fx = prevU - f32(pxBase.x);
                        let fy = prevV - f32(pxBase.y);
                        let w00 = (1.0 - fx) * (1.0 - fy);
                        let w10 = fx         * (1.0 - fy);
                        let w01 = (1.0 - fx) *         fy;
                        let w11 = fx         *         fy;
                        let p00 = pxBase;
                        let p10 = pxBase + vec2<i32>(1, 0);
                        let p01 = pxBase + vec2<i32>(0, 1);
                        let p11 = pxBase + vec2<i32>(1, 1);
                        let m00 = u32(textureLoad(hitMeshRead, p00, 0).r);
                        let m10 = u32(textureLoad(hitMeshRead, p10, 0).r);
                        let m01 = u32(textureLoad(hitMeshRead, p01, 0).r);
                        let m11 = u32(textureLoad(hitMeshRead, p11, 0).r);
                        let v00 = select(0.0, w00, m00 == primaryMeshIdx);
                        let v10 = select(0.0, w10, m10 == primaryMeshIdx);
                        let v01 = select(0.0, w01, m01 == primaryMeshIdx);
                        let v11 = select(0.0, w11, m11 == primaryMeshIdx);
                        let wSum = v00 + v10 + v01 + v11;
                        if (wSum > 1e-4) {
                            // Sanitize each tap before blending — a single NaN/Inf
                            // tap (from a prior pathological accum write) otherwise
                            // poisons the bilinear sum, then propagates outward as
                            // each subsequent camera step reprojects more pixels
                            // onto the contaminated location.
                            let d00 = textureLoad(diffAccumRead, p00, 0);
                            let d10 = textureLoad(diffAccumRead, p10, 0);
                            let d01 = textureLoad(diffAccumRead, p01, 0);
                            let d11 = textureLoad(diffAccumRead, p11, 0);
                            let d00x = cleanFinite3(d00.xyz);
                            let d10x = cleanFinite3(d10.xyz);
                            let d01x = cleanFinite3(d01.xyz);
                            let d11x = cleanFinite3(d11.xyz);
                            let inv = 1.0 / wSum;
                            oldDiff = cleanFinite3((d00x * v00 + d10x * v10
                                     + d01x * v01 + d11x * v11) * inv);
                            // FC is stored in diffAccum.w (spec has same value).
                            let prevFC = (d00.w * v00 + d10.w * v10
                                        + d01.w * v01 + d11.w * v11) * inv;
                            let s00 = cleanFinite3(textureLoad(specAccumRead, p00, 0).xyz);
                            let s10 = cleanFinite3(textureLoad(specAccumRead, p10, 0).xyz);
                            let s01 = cleanFinite3(textureLoad(specAccumRead, p01, 0).xyz);
                            let s11 = cleanFinite3(textureLoad(specAccumRead, p11, 0).xyz);
                            oldSpec = cleanFinite3((s00 * v00 + s10 * v10 + s01 * v01 + s11 * v11) * inv);
                            let mom00 = textureLoad(momentsRead, p00, 0).xy;
                            let mom10 = textureLoad(momentsRead, p10, 0).xy;
                            let mom01 = textureLoad(momentsRead, p01, 0).xy;
                            let mom11 = textureLoad(momentsRead, p11, 0).xy;
                            prevMomM1 = (mom00.x*v00 + mom10.x*v10 + mom01.x*v01 + mom11.x*v11) * inv;
                            prevMomM2 = (mom00.y*v00 + mom10.y*v10 + mom01.y*v01 + mom11.y*v11) * inv;
                            let staticCap = mix(8.0, 256.0, smoothstep(0.15, 0.7, primaryRough));
                            let movingCap = mix(8.0, 64.0,  smoothstep(0.15, 0.7, primaryRough));
                            if (primaryMoved) {
                                pixelFC = min(prevFC * 0.5, movingCap);
                            } else {
                                pixelFC = min(prevFC * 0.5, staticCap);
                            }
                            reprojOk = true;
                        }
                    }
                }
            }
            if (!reprojOk) {
                pixelFC = 0.0;
                oldDiff = vec3<f32>(0.0);
                oldSpec = vec3<f32>(0.0);
            }
        }

        // NaN+Inf guard. Firefly clamps are per-channel below (on diff/spec).
        let sampleClean = cleanFinite3(sample);
        let lum3 = vec3<f32>(0.2126, 0.7152, 0.0722);
        let clampsOn = rt.emissiveInfo.z < 1e20;
        let hardCap = 12.0;
        let alpha = 1.0 / (pixelFC + 1.0);

        // Moments — driven by the per-channel-clamped sample luminance.
        let sampleLum = dot(sampleClean, lum3);
        let momAlpha = max(alpha, 0.1);
        var m1 = prevMomM1;
        var m2 = prevMomM2;
        if (pixelFC < 1.0) { m1 = sampleLum; m2 = sampleLum * sampleLum; }
        else {
            m1 = m1 * (1.0 - momAlpha) + sampleLum * momAlpha;
            m2 = m2 * (1.0 - momAlpha) + sampleLum * sampleLum * momAlpha;
        }
        textureStore(momentsWrite, pixel, vec4<f32>(m1, m2, 0.0, 0.0));

        // Split accum (diffuse/specular)
        {
            let diffSample = cleanFinite3(ptDiff);
            let specSample = cleanFinite3(ptSpec);
            var dClamped = diffSample;
            let dLum = dot(diffSample, lum3);
            if (clampsOn && dLum > hardCap) { dClamped = diffSample * (hardCap / dLum); }
            // Mirror-like spec bypasses the atrous filter (specRoughBlend → 0 at
            // roughness < 0.05), so this cap is the only firefly gate for them.
            let specHardCap = mix(2.0, 5.0, primaryRough);
            var sClamped = specSample;
            let sLum = dot(specSample, lum3);
            if (clampsOn && sLum > specHardCap) { sClamped = specSample * (specHardCap / sLum); }
            textureStore(diffAccumWrite, pixel, vec4<f32>(oldDiff * (1.0 - alpha) + dClamped * alpha, pixelFC + 1.0));
            textureStore(specAccumWrite, pixel, vec4<f32>(oldSpec * (1.0 - alpha) + sClamped * alpha, pixelFC + 1.0));
        }

        // hitMesh includes touchedMoved — written here since it depends on runBounces output.
        textureStore(hitMeshWrite, pixel, vec4<f32>(f32(primaryMeshIdx), f32(primaryMatIdx), select(0.0, 1.0, touchedMoved), 0.0));
    }
}
