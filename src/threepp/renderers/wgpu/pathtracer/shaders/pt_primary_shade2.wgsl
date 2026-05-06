
        // ======= Classic NEE (ReSTIR disabled or skipped) =======
        // Analytical lights — stochastic pick when lcount >= 2
        let useStochLights = lcount >= 2;
        var pickedLi = 0;
        var analScale = 1.0;
        var analLiMax = lcount;
        if (useStochLights) {
            pickedLi = min(i32(rand(seed) * f32(lcount)), lcount - 1);
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
                let bs = evalBrdfFullSplit(wo, le.dir, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                          h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                let shade = throughput * shadowAtten * NdotL * le.color * analScale;
                addSplit(&diffRad, &specRad, shade * bs.diff, cap, 0, false);
                addSplit(&diffRad, &specRad, shade * bs.spec, cap, 1, true);
            }
        }

        // Emissive surface NEE
        if (emTriCount > 0) {
            let totalPower2 = rt.emissiveInfo.y;
            let es = sampleEmissiveTriCdf(seed, totalPower2, emTriCount);
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
                    let bs = evalBrdfFullSplit(wo, ln, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                               h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                    let shade = throughput * emAtten * NdotL * emColor * w_light / pdf;
                    addSplit(&diffRad, &specRad, shade * bs.diff, cap, 0, false);
                    addSplit(&diffRad, &specRad, shade * bs.spec, cap, 1, true);
                }
            }
        }

    } // end ReSTIR vs classic NEE

    // Env NEE (always runs at bounce 0 regardless of ReSTIR).
    if (HAS_ENV_CDF && rt.envColor.w > 1.5 && h.shininess > 0.01) {
        let envSample = sampleEnvImportance(seed);
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
                let bs = evalBrdfFullSplit(wo, envDir, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                           h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                let shade = throughput * envAtten * envNdotL * envCol * w_env / envPdf;
                addSplit(&diffRad, &specRad, shade * bs.diff, cap, 0, false);
                addSplit(&diffRad, &specRad, shade * bs.spec, cap, 1, true);
            }
        }
    }

    // Zero out reservoir when ReSTIR was globally enabled but skipped (glass/etc).
    if (rt.restirParams.x > 0.5 && !useReSTIR) {
        textureStore(reservoirWrite,  pixel, vec4<f32>(0.0));
        textureStore(reservoirWWrite, pixel, vec4<f32>(0.0));
    }

    // NOTE: ReSTIR GI block (i==1) and Russian roulette (i>0) are NOT present at bounce 0.

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
            if (cos_cc <= 0.0) {
                // Original: `break` — safety net writes empty GI.
                if (rt.spp.x > 0.5 && !result.giResStored) {
                    textureStore(giResWrite,   pixel, vec4<f32>(0.0));
                    textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
                    textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
                    result.giResStored = true;
                }
                result.diffRad = diffRad;
                result.specRad = specRad;
                result.pathAlive = false;
                return result;
            }
            let G1_cc = ggxG1(cos_cc, h.clearcoatAlpha);
            throughput *= vec3<f32>(ccWeight * G1_cc / ccProb);
            result.prevWo        = wo;
            result.prevNormal    = h.normal;
            result.prevAlpha     = h.clearcoatAlpha;
            result.prevMetalness = 0.0;
            result.afterTransmission = false;
            ray.origin = h.point + h.normal * 1e-3;
            ray.dir    = wi_cc;
            // Original `continue` — hand off to runBounces with outgoing ray set.
            result.rayOrigin  = ray.origin;
            result.rayDir     = ray.dir;
            result.throughput = throughput;
            result.diffRad    = diffRad;
            result.specRad    = specRad;
            result.pathAlive  = true;
            return result;
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
