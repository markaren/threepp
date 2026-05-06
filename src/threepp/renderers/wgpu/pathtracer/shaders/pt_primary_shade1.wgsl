
fn primaryShade(primaryHit:     Hit,
                ray_in:         Ray,
                seed:           ptr<function, u32>,
                pixel:          vec2<i32>,
                maxBounces:     i32,
                primaryMeshIdx: ptr<function, u32>,
                primaryNormal:  ptr<function, vec3<f32>>,
                primaryDepth:   ptr<function, f32>,
                primaryAlbedo:  ptr<function, vec3<f32>>,
                primaryRough:   ptr<function, f32>,
                primaryMatIdx:  ptr<function, i32>,
                primaryTriIdx:  ptr<function, i32>) -> PrimaryShadeResult {
    // primary* out-param defaults (match original pathTrace for sky/miss case).
    *primaryMeshIdx = 128u;
    *primaryNormal  = vec3<f32>(0.0);
    *primaryDepth   = 0.0;
    *primaryAlbedo  = vec3<f32>(1.0);
    *primaryRough   = 1.0;
    *primaryMatIdx  = -1;
    *primaryTriIdx  = -1;

    var result: PrimaryShadeResult;
    result.rayOrigin        = ray_in.origin;
    result.rayDir           = ray_in.dir;
    result.throughput       = vec3<f32>(1.0);
    result.diffRad          = vec3<f32>(0.0);
    result.specRad          = vec3<f32>(0.0);
    result.prevNormal       = vec3<f32>(0.0, 1.0, 0.0);
    result.prevWo           = vec3<f32>(0.0);
    result.prevAlpha        = 0.0;
    result.prevMetalness    = 0.0;
    result.afterTransmission = false;
    result.firstBounceSpec  = false;
    result.effectiveBounces = maxBounces;
    result.b0Point   = vec3<f32>(0.0);
    result.b0Normal  = vec3<f32>(0.0, 1.0, 0.0);
    result.b0Wo      = vec3<f32>(0.0);
    result.b0Albedo  = vec3<f32>(1.0);
    result.b0F0      = vec3<f32>(0.04);
    result.b0Metal   = 0.0;
    result.b0Alpha   = 1.0;
    result.b0MeshIdx = -1;
    result.giResStored = false;
    result.pathAlive   = false;
    result.primaryDepth  = 0.0;   // sentinel for sky/miss (matches *primaryDepth default)
    result.primaryMatIdx = -1;

    var ray        = ray_in;
    var throughput = vec3<f32>(1.0);
    var diffRad    = vec3<f32>(0.0);
    var specRad    = vec3<f32>(0.0);

    let h = primaryHit;
    if (h.t >= 1e29) {
        // Primary ray miss: show background.
        diffRad += throughput * sampleBackground(ray.dir);
        if (rt.restirParams.x > 0.5) {
            textureStore(reservoirWrite,  pixel, vec4<f32>(0.0));
            textureStore(reservoirWWrite, pixel, vec4<f32>(0.0));
        }
        if (rt.spp.x > 0.5) {
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

    // Primary hit: fill out-params + adaptive bounce cap.
    *primaryMeshIdx = u32(h.meshIdx);
    *primaryNormal  = h.normal;
    *primaryDepth   = h.t;
    *primaryRough   = sqrt(h.shininess);
    *primaryMatIdx  = h.matIdx;
    *primaryTriIdx  = h.triIdx;
    // Mirror primary metadata into result so rt_bounces_main reads without out-params.
    result.primaryDepth  = h.t;
    result.primaryMatIdx = h.matIdx;
    // Capture b0MeshIdx/b0Alpha/b0Point/b0Normal EARLY so unlit early-exit paths
    // still propagate valid primary-hit metadata to the accumulation stage.  The
    // other b0* fields (b0Wo/b0Albedo/b0F0/b0Metal) depend on wo/albedo computed
    // below and are captured there.  Unlit paths don't need them — runBounces
    // isn't called, and the accumulation stage only reads b0Point/b0Normal/b0Alpha.
    result.b0MeshIdx = h.meshIdx;
    result.b0Alpha   = h.shininess;
    result.b0Point   = h.point;
    result.b0Normal  = h.normal;

    var effectiveBounces = maxBounces;
    // Adaptive cap: diffuse / glossy surfaces stop gaining visible detail after
    // a few indirect bounces, so we cap them short for a large perf win on
    // production scenes (~5 FPS on Sponza). This is BIASED — it truncates the
    // Neumann-series energy sum and causes ρ≈1 cavities to under-read. Gate on
    // `rt.emissiveInfo.z < 1e20` ("clampsOn"): when the user sets
    // setFireflyClamp(0) as the validation-mode signal, skip the cap so the
    // white-furnace test can accumulate the full series.
    if (rt.emissiveInfo.z < 1e20) {
        let isGlass  = h.transmission > 0.05;
        let isMetal  = h.metalness > 0.5;
        let isMirror = h.shininess < 0.05;
        let isGlossy = h.shininess < 0.25;
        if (!isGlass && !isMetal && !isMirror) {
            if (isGlossy) {
                effectiveBounces = min(maxBounces, 4);
            } else {
                effectiveBounces = min(maxBounces, 3);
            }
        }
    }
    result.effectiveBounces = effectiveBounces;

    let emTriCount = i32(rt.emissiveInfo.x);
    let totalPower = rt.emissiveInfo.y;

    // Emissive at bounce 0 — full weight, no clamp (primary ray is physically correct).
    if (length(h.emissive) > 0.0) {
        diffRad += throughput * h.emissive;
    }

    var albedo = h.albedo;
    if (h.texSlot >= 0.0) { albedo *= srgbToLinear(sampleAtlas(h.uv, h.texSlot)); }
    *primaryAlbedo = albedo;

    // Unlit: flat colour, no bouncing.
    if (h.shininess < 0.0) {
        diffRad += throughput * albedo;
        if (rt.restirParams.x > 0.5) {
            textureStore(reservoirWrite,  pixel, vec4<f32>(0.0));
            textureStore(reservoirWWrite, pixel, vec4<f32>(0.0));
        }
        if (rt.spp.x > 0.5) {
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

    let wo   = normalize(-ray.dir);
    let F0_h = computeF0(albedo, h.metalness, h.specularColor, h.specularIntensity);

    // Capture bounce-0 surface fields that require wo/F0 (remaining b0* captures;
    // b0MeshIdx/b0Alpha/b0Point/b0Normal were set earlier for unlit compatibility).
    result.b0Wo      = wo;
    result.b0Albedo  = albedo;
    result.b0F0      = F0_h;
    result.b0Metal   = h.metalness;

    let lcount    = i32(rt.lightCount.x);
    // ReSTIR DI gating at bounce 0: skip for transmissive and near-mirror surfaces.
    let useReSTIR = rt.restirParams.x > 0.5 && h.transmission < 0.05;

    if (useReSTIR) {
        // ======= ReSTIR DI: Initial candidate generation =======
        var reservoir = emptyReservoir();

        // 1. Analytical lights
        {
            let p_source_a = 1.0 / max(f32(lcount), 1.0);
            for (var li = 0; li < 4; li++) {
                if (li >= lcount) { break; }
                let le = evalAnalyticalLight(li, h.point);
                let lightP = rt.lightPos[li].xyz;
                let p_hat_a = restirTargetPdf(h.point, h.normal, wo, albedo, h.metalness, h.shininess, F0_h,
                                              lightP, f32(li), le.color);
                reservoir.M += 1.0;
                if (p_hat_a > 0.0) {
                    let w = p_hat_a / p_source_a;
                    reservoir.W_sum += w;
                    if (rand(seed) < w / max(reservoir.W_sum, 1e-20)) {
                        reservoir.lightPos  = lightP;
                        reservoir.lightType = f32(li);
                        reservoir.p_hat     = p_hat_a;
                    }
                }
            }
        }

        // 2. Emissive triangles — CDF samples
        if (emTriCount > 0 && totalPower > 0.0) {
            for (var ei = 0; ei < 4; ei++) {
                let es = sampleEmissiveTriCdf(seed, totalPower, emTriCount);
                let toLight = es.point - h.point;
                let dist = length(toLight);
                let ln_e = toLight / dist;
                let cosLight = abs(dot(es.normal, -ln_e));
                let NdotL_e = dot(h.normal, ln_e);
                reservoir.M += 1.0;
                if (NdotL_e > 0.0 && cosLight > 1e-6) {
                    let eMatIdx = i32(textureLoad(triData, triCoord(es.triIdx, 0), 0).w);
                    let emColor = textureLoad(matData, vec2<i32>(eMatIdx, 2), 0).xyz;
                    let p_hat_e = restirTargetPdf(h.point, h.normal, wo, albedo, h.metalness, h.shininess, F0_h,
                                                  es.point, 1000.0 + f32(es.triIdx), emColor);
                    if (p_hat_e > 0.0) {
                        let p_source_e = (es.power / totalPower) * (dist * dist) / (es.area * cosLight);
                        let w = p_hat_e / p_source_e;
                        reservoir.W_sum += w;
                        if (rand(seed) < w / max(reservoir.W_sum, 1e-20)) {
                            reservoir.lightPos  = es.point;
                            reservoir.lightType = 1000.0 + f32(es.triIdx);
                            reservoir.p_hat     = p_hat_e;
                        }
                    }
                }
            }
        }

        // 3. Env — NOT sampled via reservoir (handled unconditionally by env NEE below).
        finalizeReservoir(&reservoir);

        // === TEMPORAL REUSE ===
        if (rt.frameCount.x > 0.0) {
            let primaryMoved = h.meshIdx >= 0 && h.meshIdx < 128 && isMeshMoved(h.meshIdx);
            var reprojPoint = h.point;
            var expectedPrevN = h.normal;
            if (primaryMoved) {
                let M = rtMotionMats[h.meshIdx];
                reprojPoint = (M * vec4<f32>(h.point, 1.0)).xyz;
                expectedPrevN = normalize((M * vec4<f32>(h.normal, 0.0)).xyz);
            }
            let relP = reprojPoint - vec3<f32>(rt.prevCamOri.x, rt.prevCamOri.y, rt.prevCamOri.z);
            let prevFwd = vec3<f32>(rt.prevCamFwd.x, rt.prevCamFwd.y, rt.prevCamFwd.z);
            let prevRgt = vec3<f32>(rt.prevCamRgt.x, rt.prevCamRgt.y, rt.prevCamRgt.z);
            let prevUp  = vec3<f32>(rt.prevCamUp.x, rt.prevCamUp.y, rt.prevCamUp.z);
            let dz = dot(relP, prevFwd);
            if (dz > 0.0) {
                let dx = dot(relP, prevRgt);
                let dy = dot(relP, prevUp);
                let thf = rt.tanHalfFov.x;
                let aspect = rt.iRes.x / rt.iRes.y;
                let ndcX = dx / (dz * thf * aspect);
                let ndcY = dy / (dz * thf);
                let prevU = (ndcX * 0.5 + 0.5) * rt.iRes.x;
                let prevV = (0.5 - ndcY * 0.5) * rt.iRes.y;
                let prevPx = vec2<i32>(i32(floor(prevU)), i32(floor(prevV)));

                if (prevPx.x >= 0 && prevPx.y >= 0 &&
                    prevPx.x < i32(rt.iRes.x) && prevPx.y < i32(rt.iRes.y)) {

                    let prevSGB = textureLoad(gBufRead, prevPx, 0);
                    let prevN = prevSGB.xyz;
                    let prevD = prevSGB.w;
                    let curD = length(relP);
                    let valid = dot(expectedPrevN, prevN) > 0.9 &&
                                abs(curD - prevD) / max(curD, 1e-6) < 0.1;

                    if (valid) {
                        let prevSample = textureLoad(reservoirRead, prevPx, 0);
                        let prevWeight = textureLoad(reservoirWRead, prevPx, 0);

                        var rPrev: Reservoir;
                        rPrev.lightPos  = prevSample.xyz;
                        rPrev.lightType = prevSample.w;
                        rPrev.W_sum = prevWeight.x;
                        rPrev.M     = min(prevWeight.y, rt.restirParams.y);
                        rPrev.W     = prevWeight.z;
                        rPrev.p_hat = prevWeight.w;

                        let prevLe = evalLightRadiance(rPrev.lightPos, rPrev.lightType, h.point);
                        let p_hat_prev = restirTargetPdf(h.point, h.normal, wo, albedo,
                                                         h.metalness, h.shininess, F0_h,
                                                         rPrev.lightPos, rPrev.lightType, prevLe);
                        if (p_hat_prev > 0.0 && rPrev.W > 0.0) {
                            let w_prev = p_hat_prev * rPrev.M * rPrev.W;
                            reservoir.W_sum += w_prev;
                            reservoir.M += rPrev.M;
                            if (rand(seed) < w_prev / max(reservoir.W_sum, 1e-20)) {
                                reservoir.lightPos  = rPrev.lightPos;
                                reservoir.lightType = rPrev.lightType;
                                reservoir.p_hat     = p_hat_prev;
                            }
                            finalizeReservoir(&reservoir);
                        }
                    }
                }
            }
        }

        // Snapshot reservoir before spatial reuse.
        let preSpReservoir = reservoir;

        // === SPATIAL REUSE — random neighbours from previous frame ===
        // Skip entirely once the pixel is well-converged: past FC≈48 the denoiser
        // atrous filter has already faded to zero (see feedback_denoiser_fc_fade),
        // and neighbour reservoirs carry strictly more variance than this pixel's
        // own. pixelFC resets to 0 on camera motion / disocclusion in rt_accum_main,
        // so this check self-gates on motion.
        let pixelFC_prev = textureLoad(diffAccumRead, pixel, 0).w;
        {
            let camMoving = (u32(rt.params.w) & 2u) != 0u;
            let spMax = select(5u, 2u, camMoving);
            let mTarget = 20.0;
            let skipSpatial = pixelFC_prev > 48.0;
            for (var spI = 0u; spI < spMax; spI++) {
                if (skipSpatial) { break; }
                if (reservoir.M >= mTarget) { break; }
                let spAngle = rand(seed) * 2.0 * PI;
                let spR     = sqrt(rand(seed)) * 20.0;
                let spOff   = vec2<i32>(i32(spR * cos(spAngle)), i32(spR * sin(spAngle)));
                if (all(spOff == vec2<i32>(0))) { continue; }
                let spPx = clamp(pixel + spOff,
                                 vec2<i32>(0),
                                 vec2<i32>(i32(rt.iRes.x) - 1, i32(rt.iRes.y) - 1));

                let spSGB = textureLoad(gBufRead, spPx, 0);
                if (dot(h.normal, spSGB.xyz) < 0.906 ||
                    abs(h.t - spSGB.w) / max(h.t, 1e-3) > 0.1) { continue; }

                let spSmp = textureLoad(reservoirRead,  spPx, 0);
                let spWt  = textureLoad(reservoirWRead, spPx, 0);
                var rSp: Reservoir;
                rSp.lightPos  = spSmp.xyz;
                rSp.lightType = spSmp.w;
                rSp.W_sum = spWt.x;
                rSp.M     = min(spWt.y, 4.0);
                rSp.W     = spWt.z;
                rSp.p_hat = spWt.w;
                if (rSp.W <= 0.0 || rSp.M <= 0.0) { continue; }

                let spLe = evalLightRadiance(rSp.lightPos, rSp.lightType, h.point);
                let p_hat_sp = restirTargetPdf(h.point, h.normal, wo, albedo,
                                                h.metalness, h.shininess, F0_h,
                                                rSp.lightPos, rSp.lightType, spLe);
                if (p_hat_sp > 0.0) {
                    let w_sp = p_hat_sp * rSp.M * rSp.W;
                    reservoir.W_sum += w_sp;
                    reservoir.M     += rSp.M;
                    if (rand(seed) < w_sp / max(reservoir.W_sum, 1e-20)) {
                        reservoir.lightPos  = rSp.lightPos;
                        reservoir.lightType = rSp.lightType;
                        reservoir.p_hat     = p_hat_sp;
                    }
                }
            }
            finalizeReservoir(&reservoir);
        }
        if (rt.emissiveInfo.z < 1e20) { reservoir.W = min(reservoir.W, 5.0); }

        // === VISIBILITY TEST ===
        var reservoirShadowAtten = vec3<f32>(0.0);
        if (reservoir.p_hat > 0.0 && reservoir.W > 0.0) {
            let rd = reservoirLightDir(reservoir.lightPos, reservoir.lightType, h.point);
            reservoirShadowAtten = traceShadowRay(h.point, h.normal, rd.dir, rd.maxDist, 4);
        }

        // === SHADE FROM RESERVOIR (split diffuse/specular) ===
        if (reservoirShadowAtten.x + reservoirShadowAtten.y + reservoirShadowAtten.z > 0.001) {
            let rd = reservoirLightDir(reservoir.lightPos, reservoir.lightType, h.point);
            let rLe = evalLightRadiance(reservoir.lightPos, reservoir.lightType, h.point);
            let rNdotL = max(dot(h.normal, rd.dir), 0.0);
            let rSplit = evalBrdfFullSplit(wo, rd.dir, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                          h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
            let cap = rt.emissiveInfo.z;
            let shade = throughput * reservoirShadowAtten * reservoir.W * rNdotL * rLe;
            addSplit(&diffRad, &specRad, shade * rSplit.diff, cap, 0, false);
            addSplit(&diffRad, &specRad, shade * rSplit.spec, cap, 1, true);
        }

        // === STORE RESERVOIR (pre-spatial; RTXDI-style — no post-spatial visibility gate) ===
        let rW = select(0.0, preSpReservoir.W, preSpReservoir.W == preSpReservoir.W);
        textureStore(reservoirWrite, pixel,
            vec4<f32>(preSpReservoir.lightPos, preSpReservoir.lightType));
        textureStore(reservoirWWrite, pixel,
            vec4<f32>(preSpReservoir.W_sum, preSpReservoir.M, rW, preSpReservoir.p_hat));

    } else {
