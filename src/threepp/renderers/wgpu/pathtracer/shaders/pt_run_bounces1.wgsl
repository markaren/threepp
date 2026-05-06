
fn runBounces(state:           PrimaryShadeResult,
              seed:             ptr<function, u32>,
              pixel:            vec2<i32>,
              primaryMeshIdx:   u32,
              touchedMoved:     ptr<function, bool>) -> SplitRadiance {
    // Reconstruct loop locals from primaryShade state.
    var ray: Ray;
    ray.origin = state.rayOrigin;
    ray.dir    = state.rayDir;
    var throughput    = state.throughput;
    // runBounces returns the BOUNCE contribution only; caller sums with state.diffRad/specRad.
    var diffRad       = vec3<f32>(0.0);
    var specRad       = vec3<f32>(0.0);
    var firstBounceSpec  = state.firstBounceSpec;
    var prevNormal       = state.prevNormal;
    var prevAlpha        = state.prevAlpha;
    var prevMetalness    = state.prevMetalness;
    var prevWo           = state.prevWo;
    var afterTransmission = state.afterTransmission;
    let effectiveBounces = state.effectiveBounces;
    let b0Point    = state.b0Point;
    let b0Normal   = state.b0Normal;
    let b0Wo       = state.b0Wo;
    let b0Albedo   = state.b0Albedo;
    let b0F0       = state.b0F0;
    let b0Metal    = state.b0Metal;
    let b0Alpha    = state.b0Alpha;
    let b0MeshIdx  = state.b0MeshIdx;
    var giResStored = state.giResStored;

    // F2a: bounce1 is handled by rt_bounce1_main in a separate pass.  This
    // loop now starts at i=2, processing only bounces 2..N.  For maxBounces=3
    // that's one iteration.  The `if (i == 1)` branches inside are dead when
    // entered from this path; kept for the pathTrace() wrapper which is still
    // compiled but not called from rt_main (vestigial, see comment on wrapper).
    for (var i = 2; i < effectiveBounces; i++) {
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
            break;
        }
        // Secondary bounce landed on a DIFFERENT moved mesh than primary — flag for
        // temporal-reprojection anti-lag so the denoiser won't trust history here.
        if (isMeshMoved(h.meshIdx) && u32(h.meshIdx) != primaryMeshIdx) {
            *touchedMoved = true;
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
            break;
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

        // Analytical lights — stochastic pick when lcount >= 2 (saves
        // lcount-1 shadow rays, +20% on 3-light scenes). Deterministic
        // loop at lcount <= 1 so lcount=0 pays zero rand() cost.
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

        // Emissive-triangle vs env NEE — stochastic 50/50 pick when both
        // are available. Cuts one shadow-ray BVH traversal per pixel on
        // emissive+env scenes (Bistro-class). Unbiased: contribution scaled
        // by 2 preserves expectation, variance absorbed by temporal denoiser.
        let hasEm  = emTriCount > 0;
        let hasEnv = HAS_ENV_CDF && rt.envColor.w > 1.5 && h.shininess > 0.01 && i < 4;
        let both   = hasEm && hasEnv;
        let doEm   = hasEm && (!both || rand(seed) < 0.5);
        let doEnv  = hasEnv && !doEm;
        let neeScale = select(1.0, 2.0, both);

        if (doEm) {
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
