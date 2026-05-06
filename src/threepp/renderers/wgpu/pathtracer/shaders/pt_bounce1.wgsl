

// ---------------------------------------------------------------------------
// rt_bounce1_main (F2a: per-bounce split).
// Processes ONE bounce iteration (i=1, including the ReSTIR GI reservoir block)
// for each pixel in aliveQueue, then writes updated PathStateEntry back to
// pathStateBuf.  rt_bounces_main's for-loop starts at i=2, so no double-processing.
//
// Architectural win: register-pressure relief.  This kernel doesn't need to
// carry i=2+ state in registers.  rt_bounces_main doesn't need to carry this
// kernel's ReSTIR GI block state in its register budget.  Higher occupancy
// in both kernels → more concurrent warps hiding BVH + shadow-ray latency.
//
// Dead-at-primary paths (pathAlive=false at entry) are skipped by this kernel
// via early `continue` of the work-stealing loop.  rt_bounces_main handles
// their accumulation via its own pathAlive guard.
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn rt_bounce1_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = rt.iRes.xy;
    let aliveTotal = atomicLoad(&aliveCount);
    let resXu = u32(res.x);
    loop {
        let slot = atomicAdd(&bounce1Counter, 1u);
        if (slot >= aliveTotal) { break; }
        // F2c: read from material-sorted queue so adjacent warp lanes hit
        // coherent BRDF code.  Same pixels as aliveQueue, just permuted.
        let pixelIdx = sortedAliveQueue[slot];
        let pixel = vec2<i32>(i32(pixelIdx % resXu), i32(pixelIdx / resXu));

        let entry = pathStateBuf[pixelIdx];
        let flagBitsIn = u32(entry.w2.w);
        let pathAliveIn = (flagBitsIn & 8u) != 0u;
        if (!pathAliveIn) { continue; }
        // Skip only paths with effectiveBounces < 1 (i.e., no indirect at all).
        // For effectiveBounces == 1 we MUST still run this kernel: primaryShade
        // already generated the BSDF-sampled direction and applied its MIS weight
        // to env NEE assuming a complementary BSDF path contribution.  Skipping
        // this kernel at mb=1 drops that complement and silently biases the
        // env-furnace result by the MIS fraction (~50% darkening on rough
        // diffuse). The kernel breaks after the env miss anyway (no further
        // bounces spawned), so letting it run costs one traced ray per pixel
        // and restores the unbiased estimator.
        let effectiveBounces1 = i32(entry.w1.w);
        if (effectiveBounces1 < 1) { continue; }
        // Restore per-pixel BN state so NEE and BRDF samples have the correct
        // spatial blue-noise offset and temporal variation.  Without this, all
        // threads start with bnPx=bnPy=bnFc=0 → identical samples for all
        // pixels → biased NEE (always triangle 0) and correlated BRDF directions.
        // bnDim=8: skip the ~7 dimensions consumed by primary (jitter + NEE + BRDF
        // sample) so bounce1's samples don't alias primary's sample dimensions.
        bnInit(u32(pixel.x), u32(pixel.y), u32(rt.frameCount.x));
        bnDim = 8u;

        // Deserialize state from PathStateEntry.  Matches runBounces' local-var
        // initialization from PrimaryShadeResult, but pulled straight from the
        // persistent buffer.
        var ray: Ray;
        ray.origin = entry.w0.xyz;
        // seed | 0x00800000u forces the lowest exponent bit set → never subnormal,
        // never flushed to zero by drivers that flush-denormals-to-zero in storage ops.
        var seed = bitcast<u32>(entry.w0.w) | 0x00800000u;
        ray.dir = entry.w1.xyz;
        var throughput = entry.w2.xyz;
        let firstBounceSpec = (flagBitsIn & 1u) != 0u;
        var afterTransmission = (flagBitsIn & 2u) != 0u;
        var touchedMoved = (flagBitsIn & 4u) != 0u;
        var giResStored = (flagBitsIn & 16u) != 0u;
        let diffRadAcc = entry.w3.xyz;
        let prevMetalnessIn = entry.w3.w;
        let specRadAcc = entry.w4.xyz;
        let prevAlphaIn = entry.w4.w;
        var prevNormal = entry.w5.xyz;
        var prevWo = entry.w6.xyz;
        var prevAlpha = prevAlphaIn;
        var prevMetalness = prevMetalnessIn;
        let b0Point = entry.w7.xyz;
        let b0Alpha = entry.w7.w;
        let b0Normal = entry.w8.xyz;
        let b0Metal = entry.w8.w;
        let b0Wo = entry.w9.xyz;
        let b0MeshIdx = i32(entry.w9.w);
        let b0Albedo = entry.w10.xyz;
        let b0F0 = entry.w11.xyz;

        // Per-bounce radiance (ADDED to entry's accumulated radiance at write-back).
        var diffRad = vec3<f32>(0.0);
        var specRad = vec3<f32>(0.0);

        let primaryMeshIdx = u32(b0MeshIdx);
        let i = 1;
        var pathStillAlive = true;

        // Do-once loop to enable early-exit via `break` for path-death / continue
        // translations.  Body is a direct transform of runBounces' for-loop iteration.
        loop {
            var h = sceneHit(ray);
            if (h.t >= 1e29) {
                // Bounced miss: env IBL with MIS complement to BRDF env-importance NEE.
                var envMisW = 1.0;
                if (HAS_ENV_CDF && rt.envColor.w > 1.5 && prevAlpha > 0.01) {
                    let pdf_env  = envImportancePdf(ray.dir);
                    let pdf_brdf = brdfPdf(prevWo, normalize(ray.dir), prevNormal, prevAlpha, prevMetalness);
                    envMisW = pdf_brdf / max(pdf_brdf + pdf_env, 1e-8);
                }
                addSplit(&diffRad, &specRad,
                    throughput * sampleEnv(ray.dir) * envMisW,
                    rt.emissiveInfo.z, i, firstBounceSpec);
                if (i == 1 && rt.spp.x > 0.5 && !giResStored) {
                    textureStore(giResWrite,   pixel, vec4<f32>(0.0));
                    textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
                    textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
                    giResStored = true;
                }
                pathStillAlive = false; break;
            }
            // Secondary bounce landed on a DIFFERENT moved mesh than primary — flag for
            // temporal-reprojection anti-lag so the denoiser won't trust history here.
            if (isMeshMoved(h.meshIdx) && u32(h.meshIdx) != primaryMeshIdx) {
                touchedMoved = true;
            }

            let emTriCount = i32(rt.emissiveInfo.x);
            let totalPower = rt.emissiveInfo.y;

            // Emissive MIS at bounce >= 1.
            if (length(h.emissive) > 0.0) {
                if (emTriCount == 0 || afterTransmission) {
                    // No NEE available, or after transmission: full weight to this path.
                    addSplit(&diffRad, &specRad, throughput * h.emissive,
                             rt.emissiveInfo.z, i, firstBounceSpec);
                } else if (i == 1 && rt.restirParams.x > 0.5 && !firstBounceSpec) {
                    // Diffuse bounce-0 used ReSTIR DI which already produced unbiased direct;
                    // adding BRDF-sampled emissive here would double-count.  Specular bounce-0
                    // still needs this contribution (handled via firstBounceSpec == true else).
                } else {
                    let cosLight = abs(dot(h.geoNormal, -ray.dir));
                    if (cosLight > 1e-6) {
                        let emLum = 0.2126 * h.emissive.r + 0.7152 * h.emissive.g + 0.0722 * h.emissive.b;
                        let pdf_light = (emLum * h.t * h.t) / (totalPower * cosLight);
                        let pdf_brdf  = brdfPdf(prevWo, normalize(ray.dir), prevNormal, prevAlpha, prevMetalness);
                        let w = pdf_brdf / max(pdf_brdf + pdf_light, 1e-8);
                        if (w == w && w < 1e10) {
                            addSplit(&diffRad, &specRad, throughput * h.emissive * w,
                                     rt.emissiveInfo.z, i, firstBounceSpec);
                        }
                    }
                }
            }

            var albedo = h.albedo;
            if (h.texSlot >= 0.0) { albedo *= srgbToLinear(sampleAtlas(h.uv, h.texSlot)); }

            // Unlit: flat colour, no bouncing.  No i==0 reservoir-zero branch here.
            if (h.shininess < 0.0) {
                diffRad += throughput * albedo;
                if (i == 1 && rt.spp.x > 0.5 && !giResStored) {
                    textureStore(giResWrite,   pixel, vec4<f32>(0.0));
                    textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
                    textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
                    giResStored = true;
                }
                pathStillAlive = false; break;
            }

            let wo = normalize(-ray.dir);
            let F0_h = computeF0(albedo, h.metalness, h.specularColor, h.specularIntensity);

            // ReSTIR GI gating: fires only at the first secondary hit (i == 1).
            let giAlphaMin = select(0.01, 0.1, b0Metal > 0.5);
            let useReSTIRGI = i == 1 && rt.spp.x > 0.5 && !afterTransmission
                              && h.transmission < 0.05 && b0Alpha > giAlphaMin;
            var giLo = vec3<f32>(0.0);

            // ==== Classic NEE ==== (useReSTIR is i==0-only, so always false here.)
            let lcount = i32(rt.lightCount.x);

            // Analytical lights — stochastic pick when lcount >= 2
            let useStochLights = lcount >= 2;
            var pickedLi = 0;
            var analScale = 1.0;
            var analLiMax = lcount;
            if (useStochLights) {
                pickedLi = min(i32(rand(&seed) * f32(lcount)), lcount - 1);
                analScale = f32(lcount);
                analLiMax = 1;
            }
            for (var li = 0; li < 4; li++) {
                if (li >= analLiMax) { break; }
                let actualLi = select(li, pickedLi, useStochLights);
                let le = evalAnalyticalLight(actualLi, h.point);
                let NdotL = dot(h.normal, le.dir);
                if (NdotL <= 0.0) { continue; }
                let shadowAtten = traceShadowRay(h.point, h.normal, le.dir, le.dist - 1e-3, 4);
                if (shadowAtten.x + shadowAtten.y + shadowAtten.z > 0.001) {
                    let cap = rt.emissiveInfo.z;
                    let lobeSum = evalBrdfFull(wo, le.dir, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                               h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                    let neeContrib = shadowAtten * lobeSum * NdotL * le.color * analScale;
                    if (useReSTIRGI && !giResStored) {
                        giLo += neeContrib;
                    } else {
                        addSplit(&diffRad, &specRad, throughput * neeContrib, cap, i, firstBounceSpec);
                    }
                }
            }

            // Emissive-triangle vs env NEE — stochastic 50/50 pick (see
            // runBounces for MC reasoning). Saves one shadow-ray BVH
            // traversal per pixel on emissive+env scenes.
            let hasEm  = emTriCount > 0;
            let hasEnv = HAS_ENV_CDF && rt.envColor.w > 1.5 && h.shininess > 0.01 && i < 4;
            let bothNee = hasEm && hasEnv;
            let doEm    = hasEm && (!bothNee || rand(&seed) < 0.5);
            let doEnv   = hasEnv && !doEm;
            let neeScale = select(1.0, 2.0, bothNee);

            if (doEm) {
                let totalPower2 = rt.emissiveInfo.y;
                let es = sampleEmissiveTriCdf(&seed, totalPower2, emTriCount);
                let toLight = es.point - h.point;
                let dist    = length(toLight);
                let ln      = toLight / dist;
                let NdotL   = dot(h.normal, ln);
                let cosLight = abs(dot(es.normal, -ln));
                if (NdotL > 0.0 && cosLight > 1e-6) {
                    let emAtten = traceShadowRay(h.point, h.normal, ln, dist - 1e-2, 4);
                    if (emAtten.x + emAtten.y + emAtten.z > 0.001) {
                        let eMatIdx = i32(textureLoad(triData, triCoord(es.triIdx, 0), 0).w);
                        let emColor = textureLoad(matData, vec2<i32>(eMatIdx, 2), 0).xyz;
                        let pdf = (es.power * dist * dist) / (totalPower2 * es.area * cosLight);
                        let pdf_brdf_nee = brdfPdf(wo, ln, h.normal, h.shininess, h.metalness);
                        let w_light = pdf / max(pdf + pdf_brdf_nee, 1e-8);
                        let cap = rt.emissiveInfo.z;
                        let lobeSum3 = evalBrdfFull(wo, ln, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                                     h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                        let emNeeContrib = emAtten * lobeSum3 * NdotL * emColor * w_light / pdf * neeScale;
                        if (useReSTIRGI && !giResStored) {
                            giLo += emNeeContrib;
                        } else {
                            addSplit(&diffRad, &specRad, throughput * emNeeContrib, cap, i, firstBounceSpec);
                        }
                    }
                }
            }

            if (doEnv) {
                let envSample = sampleEnvImportance(&seed);
                let envDir    = envSample.xyz;
                let envPdf    = envSample.w;
                let envNdotL  = dot(h.normal, envDir);
                if (envNdotL > 0.0 && envPdf > 1e-8) {
                    let envAtten = traceShadowRay(h.point, h.normal, envDir, 1e30, 4);
                    if (envAtten.x + envAtten.y + envAtten.z > 0.001) {
                        let envCol = sampleEnv(envDir);
                        let pdf_brdf_env = brdfPdf(wo, envDir, h.normal, h.shininess, h.metalness);
                        let w_env = envPdf / max(envPdf + pdf_brdf_env, 1e-8);
                        let cap = rt.emissiveInfo.z;
                        let lobeSum4 = evalBrdfFull(wo, envDir, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                                     h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                        let envNeeContrib = envAtten * lobeSum4 * envNdotL * envCol * w_env / envPdf * neeScale;
                        if (useReSTIRGI && !giResStored) {
                            giLo += envNeeContrib;
                        } else {
                            addSplit(&diffRad, &specRad, throughput * envNeeContrib, cap, i, firstBounceSpec);
                        }
                    }
                }
            }

            // ==== ReSTIR GI reservoir (runs only at i==1 when useReSTIRGI) ====

            let giSecDist2 = dot(h.point - b0Point, h.point - b0Point);
            if (useReSTIRGI && !giResStored && giSecDist2 > 0.04) {
                let giP_hat = giTargetPdf(b0Point, b0Normal, b0Wo, b0Albedo, b0Metal, b0Alpha, b0F0,
                                           h.point, h.normal, giLo);
                var giW_sum = giP_hat;
                var giM = 1.0;
                var giSecPos  = h.point;
                var giSecNorm = h.normal;
                var giLoRes   = giLo;
                var giPhat    = giP_hat;

                // Temporal reuse of previous-frame GI reservoir.
                if (rt.frameCount.x > 0.0) {
                    let giMeshMoved = b0MeshIdx >= 0 && b0MeshIdx < 128 && isMeshMoved(b0MeshIdx);
                    var giReprojPt = b0Point;
                    var giExpectedN = b0Normal;
                    if (giMeshMoved) {
                        let giMotMat = rtMotionMats[b0MeshIdx];
                        giReprojPt = (giMotMat * vec4<f32>(b0Point, 1.0)).xyz;
                        giExpectedN = normalize((giMotMat * vec4<f32>(b0Normal, 0.0)).xyz);
                    }
                    let giRelP = giReprojPt - vec3<f32>(rt.prevCamOri.x, rt.prevCamOri.y, rt.prevCamOri.z);
                    let giPrevFwd = vec3<f32>(rt.prevCamFwd.x, rt.prevCamFwd.y, rt.prevCamFwd.z);
                    let giPrevRgt = vec3<f32>(rt.prevCamRgt.x, rt.prevCamRgt.y, rt.prevCamRgt.z);
                    let giPrevUp  = vec3<f32>(rt.prevCamUp.x, rt.prevCamUp.y, rt.prevCamUp.z);
                    let giDz = dot(giRelP, giPrevFwd);
                    if (giDz > 0.0) {
                        let giDx = dot(giRelP, giPrevRgt);
                        let giDy = dot(giRelP, giPrevUp);
                        let giThf = rt.tanHalfFov.x;
                        let giAspect = rt.iRes.x / rt.iRes.y;
                        let giPrevU = (giDx / (giDz * giThf * giAspect) * 0.5 + 0.5) * rt.iRes.x;
                        let giPrevV = (0.5 - giDy / (giDz * giThf) * 0.5) * rt.iRes.y;
                        let giPrevPx = vec2<i32>(i32(floor(giPrevU)), i32(floor(giPrevV)));

                        if (giPrevPx.x >= 0 && giPrevPx.y >= 0 &&
                            giPrevPx.x < i32(rt.iRes.x) && giPrevPx.y < i32(rt.iRes.y)) {

                            let giPrevSGB = textureLoad(gBufRead, giPrevPx, 0);
                            let giPrevN = giPrevSGB.xyz;
                            let giPrevD = giPrevSGB.w;
                            let giCurD  = length(giRelP);
                            let giPrevMesh = i32(textureLoad(hitMeshRead, giPrevPx, 0).r);
                            let giValid = dot(giExpectedN, giPrevN) > 0.95 &&
                                          abs(giCurD - giPrevD) / max(giCurD, 1e-6) < 0.05 &&
                                          giPrevMesh == b0MeshIdx;

                            if (giValid) {
                                let prevGiSample = textureLoad(giResRead, giPrevPx, 0);
                                let prevGiWeight = textureLoad(giResWRead, giPrevPx, 0);
                                let prevGiLo     = textureLoad(giResLoRead, giPrevPx, 0).xyz;

                                let prevSecPos  = prevGiSample.xyz;
                                let prevSecNorm = unpackOctNormal(prevGiSample.w);
                                let prevW_sum = prevGiWeight.x;
                                let prevM     = min(prevGiWeight.y, 20.0);
                                let prevW     = prevGiWeight.z;
                                let prevPhat  = prevGiWeight.w;


                                let prevSecDist2 = dot(prevSecPos - b0Point, prevSecPos - b0Point);
                                let giPhatPrev = select(0.0,
                                    giTargetPdf(b0Point, b0Normal, b0Wo, b0Albedo, b0Metal, b0Alpha, b0F0,
                                                prevSecPos, prevSecNorm, prevGiLo),
                                    prevSecDist2 > 0.04);
                                if (giPhatPrev > 0.0 && prevW > 0.0) {
                                    let J = reconnJacobian(prevSecPos, prevSecNorm, giReprojPt, b0Point);
                                    let giWPrev = giPhatPrev * prevM * prevW * J;
                                    giW_sum += giWPrev;
                                    giM += prevM;
                                    if (rand(&seed) < giWPrev / max(giW_sum, 1e-20)) {
                                        giSecPos  = prevSecPos;
                                        giSecNorm = prevSecNorm;
                                        giLoRes   = prevGiLo;
                                        giPhat    = giPhatPrev;
                                    }
                                }
                            }
                        }
                    }
                }

                // Snapshot pre-spatial reservoir (stored for next-frame temporal reuse).
                let giPreSpW_sum = giW_sum;
                let giPreSpM     = giM;
                let giPreSpPhat  = giPhat;
                let giPreSpSecPos  = giSecPos;
                let giPreSpSecNorm = giSecNorm;
                let giPreSpLoRes   = giLoRes;

                // Spatial reuse from 4 random neighbours (20px disk).
                {
                    for (var spI = 0u; spI < 4u; spI++) {
                        let spAngle = rand(&seed) * 2.0 * PI;
                        let spR     = sqrt(rand(&seed)) * 20.0;
                        let spOff   = vec2<i32>(i32(spR * cos(spAngle)), i32(spR * sin(spAngle)));
                        if (all(spOff == vec2<i32>(0))) { continue; }
                        let spPx = clamp(pixel + spOff,
                                         vec2<i32>(0),
                                         vec2<i32>(i32(rt.iRes.x) - 1, i32(rt.iRes.y) - 1));

                        let spSGB = textureLoad(gBufRead, spPx, 0);
                        let spMesh = i32(textureLoad(hitMeshRead, spPx, 0).r);
                        let b0Depth = length(b0Point - vec3<f32>(rt.camOri.x, rt.camOri.y, rt.camOri.z));
                        if (dot(b0Normal, spSGB.xyz) < 0.95 ||
                            abs(b0Depth - spSGB.w) / max(b0Depth, 1e-3) > 0.05 ||
                            spMesh != b0MeshIdx) { continue; }

                        let spNdc = vec2<f32>(
                            (f32(spPx.x) + 0.5) / rt.iRes.x * 2.0 - 1.0,
                            1.0 - (f32(spPx.y) + 0.5) / rt.iRes.y * 2.0);
                        let spAspect = rt.iRes.x / rt.iRes.y;
                        let spDir = normalize(rt.prevCamFwd.xyz
                            + rt.prevCamRgt.xyz * (spNdc.x * rt.tanHalfFov.x * spAspect)
                            + rt.prevCamUp.xyz  * (spNdc.y * rt.tanHalfFov.x));
                        let spPrimary = rt.prevCamOri.xyz + spDir * spSGB.w;

                        let spGiSample = textureLoad(giResRead, spPx, 0);
                        let spGiWeight = textureLoad(giResWRead, spPx, 0);
                        let spGiLo     = textureLoad(giResLoRead, spPx, 0).xyz;

                        let spSecPos  = spGiSample.xyz;
                        let spSecNorm = unpackOctNormal(spGiSample.w);
                        let spM = min(spGiWeight.y, 4.0);
                        let spW = spGiWeight.z;
                        if (spW <= 0.0 || spM <= 0.0) { continue; }

                        let spSecDist2 = dot(spSecPos - b0Point, spSecPos - b0Point);
                        if (spSecDist2 < 0.04) { continue; }

                        let spGiPhat = giTargetPdf(b0Point, b0Normal, b0Wo, b0Albedo, b0Metal, b0Alpha, b0F0,
                                                    spSecPos, spSecNorm, spGiLo);
                        if (spGiPhat > 0.0) {
                            let J = reconnJacobian(spSecPos, spSecNorm, spPrimary, b0Point);
                            let spGiW = spGiPhat * spM * spW * J;
                            giW_sum += spGiW;
                            giM += spM;
                            if (rand(&seed) < spGiW / max(giW_sum, 1e-20)) {
                                giSecPos  = spSecPos;
                                giSecNorm = spSecNorm;
                                giLoRes   = spGiLo;
                                giPhat    = spGiPhat;
                            }
                        }
                    }
                }
                // Finalize + visibility + shade + store pre-spatial reservoir.
                var giW = select(0.0, giW_sum / max(giM * giPhat, 1e-20), giPhat > 0.0);
                giW = min(giW, 3.0);

                let giWi = normalize(giSecPos - b0Point);
                let giNdotL = max(dot(b0Normal, giWi), 0.0);
                var giVisAtten = vec3<f32>(0.0);
                if (giNdotL > 0.0 && giW > 0.0) {
                    let giDist = length(giSecPos - b0Point);
                    giVisAtten = traceShadowRay(b0Point, b0Normal, giWi, giDist - 1e-3, 4);
                }

                if (giVisAtten.x + giVisAtten.y + giVisAtten.z > 0.001 && giNdotL > 0.0) {
                    let giDelta = giSecPos - b0Point;
                    let giDist2 = dot(giDelta, giDelta);
                    let giCosTheta2 = max(dot(giSecNorm, -giWi), 0.0);
                    let giG = giCosTheta2 / max(giDist2, 0.01);
                    let giBrdf = evalBrdfFullSplit(b0Wo, giWi, b0Normal, b0Albedo, b0Metal, b0Alpha, b0F0,
                                                    vec3<f32>(0.0), 0.0, 0.0, 0.0);
                    let giShade = giVisAtten * giW * giNdotL * giLoRes * giG;
                    let giCap = rt.emissiveInfo.z;
                    var giContribD = giShade * giBrdf.diff;
                    var giContribS = giShade * giBrdf.spec;
                    let giLum = luminance(giContribD + giContribS);
                    let giHardCap = 2.0;
                    if (giLum > giHardCap) {
                        let giScale = giHardCap / giLum;
                        giContribD *= giScale;
                        giContribS *= giScale;
                    }
                    addSplit(&diffRad, &specRad, giContribD, giCap, 1, false);
                    addSplit(&diffRad, &specRad, giContribS, giCap, 1, true);
                }

                let giVisible = giVisAtten.x + giVisAtten.y + giVisAtten.z > 0.001;
                let giStoreW_f = select(0.0, select(0.0, giPreSpW_sum / max(giPreSpM * giPreSpPhat, 1e-20), giPreSpPhat > 0.0), giVisible);
                let giStoreM_f = select(0.0, giPreSpM, giVisible);
                textureStore(giResWrite,   pixel, vec4<f32>(giPreSpSecPos, packOctNormal(giPreSpSecNorm)));
                textureStore(giResWWrite,  pixel, vec4<f32>(giPreSpW_sum, giStoreM_f, min(giStoreW_f, 3.0), giPreSpPhat));
                textureStore(giResLoWrite, pixel, vec4<f32>(giPreSpLoRes, 0.0));
                giResStored = true;
            }
            // GI fallback — add captured Lo back if reservoir was skipped (close hit).
            if (useReSTIRGI && !giResStored && luminance(giLo) > 0.0) {
                addSplit(&diffRad, &specRad, throughput * giLo, rt.emissiveInfo.z, 1, firstBounceSpec);
            }

            // Russian roulette (i > 0 branch — always true in runBounces).
            {
                let p_base = max(max(throughput.r, throughput.g), throughput.b);
                let rough  = sqrt(h.shininess);
                let weight = mix(0.25, 1.0, 1.0 - rough);
                let p = clamp(p_base * weight, 0.02, 1.0);
                if (rand(&seed) > p) { pathStillAlive = false; break; }
                throughput /= p;
            }

            // Clearcoat lobe.
            if (h.clearcoat > 0.0 && h.transmission < 0.01) {
                let ccF0 = 0.04;
                let ccCos = max(0.0, dot(wo, h.normal));
                let ccFresnel = ccF0 + (1.0 - ccF0) * pow(1.0 - ccCos, 5.0);
                let ccWeight = h.clearcoat * ccFresnel;
                let ccProb = max(ccWeight, 0.15 * h.clearcoat);
                if (rand(&seed) < ccProb) {
                    let wi_cc = sampleVNDF(wo, h.normal, h.clearcoatAlpha, &seed);
                    let cos_cc = dot(h.normal, wi_cc);
                    if (cos_cc <= 0.0) { pathStillAlive = false; break; }  // outer death
                    let G1_cc = ggxG1(cos_cc, h.clearcoatAlpha);
                    throughput *= vec3<f32>(ccWeight * G1_cc / ccProb);
                    prevWo        = wo;
                    prevNormal    = h.normal;
                    prevAlpha     = h.clearcoatAlpha;
                    prevMetalness = 0.0;
                    afterTransmission = false;
                    ray.origin = h.point + h.normal * 1e-3;
                    ray.dir    = wi_cc;
                    break;  // F2a: clearcoat continue → exit do-once
                }
                throughput *= (1.0 - ccWeight) / (1.0 - ccProb);
            }

            // Sheen.
            let sheenLumPT = dot(h.sheenColor, vec3<f32>(0.2126, 0.7152, 0.0722));
            if (sheenLumPT > 0.001 && h.transmission < 0.01) {
                let sheenAlpha = max(h.sheenRoughness * h.sheenRoughness, 1e-4);
                let NdotV_sh = max(0.001, dot(wo, h.normal));
                let sheenFresnel = sheenLumPT * pow(1.0 - NdotV_sh, 3.0);
                throughput *= (1.0 - sheenFresnel);
            }

            // Transmission.
            if (h.transmission > 0.0 && rand(&seed) < h.transmission) {
                let entering = h.frontFace > 0.5;
                let tMat4 = textureLoad(matData, vec2<i32>(h.matIdx, 4), 0);
                let tAttColor = tMat4.xyz;
                let tAttDist  = tMat4.w;
                var channelMask = vec3<f32>(1.0);
                var ior_eff = h.ior;
                if (h.dispersion > 0.0) {
                    let lambda = array<f32, 3>(0.6563, 0.55, 0.4861);
                    let ref_inv_sq = 1.0 / (0.5893 * 0.5893);
                    let ch = u32(rand(&seed) * 3.0) % 3u;
                    let inv_sq = 1.0 / (lambda[ch] * lambda[ch]);
                    let B = (h.ior - 1.0) * h.dispersion / 38.2;
                    ior_eff = h.ior + B * (inv_sq - ref_inv_sq);
                    channelMask = vec3<f32>(0.0);
                    channelMask[ch] = 3.0;
                }
                let eta = select(ior_eff, 1.0 / ior_eff, entering);
                var tNorm = h.normal;
                var usedMicrofacet = false;
                let wo_t = normalize(-ray.dir);
                if (h.shininess > 1e-3) {
                    let wi_micro = sampleVNDF(wo_t, h.normal, h.shininess, &seed);
                    let hm = normalize(wo_t + wi_micro);
                    if (dot(hm, h.normal) > 0.0) { tNorm = hm; usedMicrofacet = true; }
                }
                let cosI = abs(dot(normalize(ray.dir), tNorm));
                let r0   = pow((1.0 - ior_eff) / (1.0 + ior_eff), 2.0);
                let sin2I = max(0.0, 1.0 - cosI * cosI);
                let cosSchlick = select(cosI,
                    sqrt(max(0.0, 1.0 - ior_eff * ior_eff * sin2I)),
                    !entering);
                let fresnel = r0 + (1.0 - r0) * pow(1.0 - cosSchlick, 5.0);
                var wi_t: vec3<f32>;
                var didRefract = false;
                if (rand(&seed) < fresnel) {
                    wi_t = reflect(ray.dir, tNorm);
                    ray.origin = h.point + h.normal * 1e-3;
                } else {
                    let refracted = refract(normalize(ray.dir), tNorm, eta);
                    if (length(refracted) < 0.001) {
                        wi_t = reflect(ray.dir, tNorm);
                        ray.origin = h.point + h.normal * 1e-3;
                    } else {
                        wi_t = refracted;
                        ray.origin = h.point - h.normal * 1e-3;
                        didRefract = true;
                    }
                }
                var microWeight = 1.0;
                if (usedMicrofacet) {
                    let cosOut = abs(dot(wi_t, h.normal));
                    microWeight = ggxG1(cosOut, h.shininess);
                }
                if (didRefract) {
                    let glassTint = albedo * microWeight;
                    var volAtten = vec3<f32>(1.0);
                    if (tAttDist > 0.0 && !entering) {
                        let absorbCoeff = -log(max(tAttColor, vec3<f32>(1e-6))) / tAttDist;
                        let pathLen = select(h.t, h.thickness, h.t < 1e-2 && h.thickness > 0.0);
                        volAtten = exp(-absorbCoeff * pathLen);
                    }
                    throughput *= glassTint * volAtten / (eta * eta) * channelMask;
                } else {
                    throughput *= vec3<f32>(microWeight) * channelMask;
                }
                afterTransmission = true;
                ray.dir = wi_t;
                break;  // F2a: continue → exit do-once (path alive)
            }

            // BRDF sample — firstBounceSpec is i==0-only in the original, so no update here.
            let F0_b = F0_h;
            var wi_b: vec3<f32>;
            let p_spec = mix(0.5, 0.98, h.metalness);
            let isSpecBounce = rand(&seed) < p_spec;
            if (isSpecBounce) {
                wi_b = sampleVNDF(wo, h.normal, h.shininess, &seed);
                let cos_b = dot(h.normal, wi_b);
                if (cos_b <= 0.0) { pathStillAlive = false; break; }
                let hb  = normalize(wo + wi_b);
                let Fb  = schlick(max(0.0, dot(wo, hb)), F0_b);
                let G1L = ggxG1(cos_b, h.shininess);
                let msC = msCompensation(F0_b, max(1e-4, dot(h.normal, wo)), h.shininess);
                throughput *= Fb * G1L * msC / p_spec;
            } else {
                wi_b = cosineHemisphere(h.normal, &seed);
                let cos_b = dot(h.normal, wi_b);
                if (cos_b <= 0.0) { pathStillAlive = false; break; }
                // K-C diffuse ms-comp throughput boost — matches evalBrdf kcDiff.
                let NdotV_kc = max(1e-4, dot(h.normal, wo));
                let E_kc     = sampleGgxELut(NdotV_kc, h.shininess);
                let F_avg_kc = (20.0 * F0_b + vec3<f32>(1.0)) / 21.0;
                let kcBoost  = vec3<f32>(1.0) + F_avg_kc * max(0.0, 1.0 - E_kc);
                throughput *= albedo * (1.0 - h.metalness) * kcBoost / (1.0 - p_spec);
            }
            prevWo        = wo;
            prevNormal    = h.normal;
            prevAlpha     = h.shininess;
            prevMetalness = h.metalness;
            afterTransmission = false;
            ray.origin = h.point + h.normal * 1e-3;
            ray.dir    = wi_b;

            break;  // F2a: fall-through end-of-iteration (path alive, BRDF sampled)
        }  // end do-once

        // Safety net: path died without storing GI → write empty reservoir so
        // denoiser's spatial reuse doesn't pick up stale data next frame.
        if (!pathStillAlive && rt.spp.x > 0.5 && !giResStored) {
            textureStore(giResWrite,   pixel, vec4<f32>(0.0));
            textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
            textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
            giResStored = true;
        }

        // Write back to pathStateBuf: update ray / throughput / prev* / flags /
        // accumulated radiance.  b0* + primaryDepth + primaryMatIdx + effectiveBounces
        // are unchanged (bounce1 doesn't touch them; bounces_main starts at i=2 and
        // doesn't use b0* anyway).
        var flagBitsOut = flagBitsIn & (1u | 32u);  // preserve firstBounceSpec (bit 0) + skipAccum (bit 5, always 0 here)
        if (afterTransmission) { flagBitsOut |= 2u;  }
        if (touchedMoved)      { flagBitsOut |= 4u;  }
        if (pathStillAlive)    { flagBitsOut |= 8u;  }
        if (giResStored)       { flagBitsOut |= 16u; }

        var newEntry = entry;
        newEntry.w0 = vec4<f32>(ray.origin, bitcast<f32>(seed | 0x00800000u));


        newEntry.w1 = vec4<f32>(ray.dir, entry.w1.w);
        newEntry.w2 = vec4<f32>(throughput, f32(flagBitsOut));
        newEntry.w3 = vec4<f32>(diffRadAcc + diffRad, prevMetalness);
        newEntry.w4 = vec4<f32>(specRadAcc + specRad, prevAlpha);
        newEntry.w5 = vec4<f32>(prevNormal, entry.w5.w);
        newEntry.w6 = vec4<f32>(prevWo,     entry.w6.w);
        pathStateBuf[pixelIdx] = newEntry;

        // F2b: append to alive1Queue if path survives bounce 1.  rt_bounces_main
        // will read this compacted list, dispatching on ~12% of non-skipAccum
        // pixels instead of 100%.
        if (pathStillAlive) {
            let outSlot = atomicAdd(&alive1Count, 1u);
            alive1Queue[outSlot] = pixelIdx;
        }
    }
}
