
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
                                if (rand(seed) < giWPrev / max(giW_sum, 1e-20)) {
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
                    let spAngle = rand(seed) * 2.0 * PI;
                    let spR     = sqrt(rand(seed)) * 20.0;
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
                        if (rand(seed) < spGiW / max(giW_sum, 1e-20)) {
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
            if (rt.emissiveInfo.z < 1e20) { giW = min(giW, 3.0); }

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
            if (rand(seed) > p) { break; }
            throughput /= p;
        }

        // Clearcoat lobe.
        if (h.clearcoat > 0.0 && h.transmission < 0.01) {
            let ccF0 = 0.04;
            let ccCos = max(0.0, dot(wo, h.normal));
            let ccFresnel = ccF0 + (1.0 - ccF0) * pow(1.0 - ccCos, 5.0);
            let ccWeight = h.clearcoat * ccFresnel;
            let ccProb = max(ccWeight, 0.15 * h.clearcoat);
            if (rand(seed) < ccProb) {
                let wi_cc = sampleVNDF(wo, h.normal, h.clearcoatAlpha, seed);
                let cos_cc = dot(h.normal, wi_cc);
                if (cos_cc <= 0.0) { break; }
                let G1_cc = ggxG1(cos_cc, h.clearcoatAlpha);
                throughput *= vec3<f32>(ccWeight * G1_cc / ccProb);
                prevWo        = wo;
                prevNormal    = h.normal;
                prevAlpha     = h.clearcoatAlpha;
                prevMetalness = 0.0;
                afterTransmission = false;
                ray.origin = h.point + h.normal * 1e-3;
                ray.dir    = wi_cc;
                continue;
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
        if (h.transmission > 0.0 && rand(seed) < h.transmission) {
            let entering = h.frontFace > 0.5;
            let tMat4 = textureLoad(matData, vec2<i32>(h.matIdx, 4), 0);
            let tAttColor = tMat4.xyz;
            let tAttDist  = tMat4.w;
            var channelMask = vec3<f32>(1.0);
            var ior_eff = h.ior;
            if (h.dispersion > 0.0) {
                let lambda = array<f32, 3>(0.6563, 0.55, 0.4861);
                let ref_inv_sq = 1.0 / (0.5893 * 0.5893);
                let ch = u32(rand(seed) * 3.0) % 3u;
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
                let wi_micro = sampleVNDF(wo_t, h.normal, h.shininess, seed);
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
            if (rand(seed) < fresnel) {
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
            continue;
        }

        // BRDF sample — firstBounceSpec is i==0-only in the original, so no update here.
        let F0_b = F0_h;
        var wi_b: vec3<f32>;
        // p_spec must match brdfPdf() exactly — MIS balance heuristic requires the
        // pdf used in weighting to equal the pdf used in sampling, or contributions
        // miscalibrate.
        let p_spec = mix(0.5, 0.98, h.metalness);
        let isSpecBounce = rand(seed) < p_spec;
        if (isSpecBounce) {
            wi_b = sampleVNDF(wo, h.normal, h.shininess, seed);
            let cos_b = dot(h.normal, wi_b);
            if (cos_b <= 0.0) { break; }
            let hb  = normalize(wo + wi_b);
            let Fb  = schlick(max(0.0, dot(wo, hb)), F0_b);
            let G1L = ggxG1(cos_b, h.shininess);
            let msC = msCompensation(F0_b, max(1e-4, dot(h.normal, wo)), h.shininess);
            throughput *= Fb * G1L * msC / p_spec;
        } else {
            wi_b = cosineHemisphere(h.normal, seed);
            let cos_b = dot(h.normal, wi_b);
            if (cos_b <= 0.0) { break; }
            // Pure Lambertian diffuse: no Fresnel coupling, matches evalBrdf.
            throughput *= albedo * (1.0 - h.metalness) / (1.0 - p_spec);
        }
        prevWo        = wo;
        prevNormal    = h.normal;
        prevAlpha     = h.shininess;
        prevMetalness = h.metalness;
        afterTransmission = false;
        ray.origin = h.point + h.normal * 1e-3;
        ray.dir    = wi_b;
    }

    // Safety net: write empty GI reservoir if none of the paths above stored one.
    if (rt.spp.x > 0.5 && !giResStored) {
        textureStore(giResWrite,   pixel, vec4<f32>(0.0));
        textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
        textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
    }

    return SplitRadiance(diffRad, specRad);
}
