
    // Transmission lobe.
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
        ray.dir = wi_t;
        // Original: set afterTransmission=true then `continue`.  Hand off to runBounces.
        result.afterTransmission = true;
        result.rayOrigin  = ray.origin;
        result.rayDir     = ray.dir;
        result.throughput = throughput;
        result.diffRad    = diffRad;
        result.specRad    = specRad;
        // NOTE: prevNormal/prevAlpha/prevMetalness/prevWo left at struct defaults
        // here — matches original, which also doesn't update prev* on the transmission
        // `continue` branch at i==0.  firstBounceSpec also stays false; see pathTrace
        // comment "Transmission at bounce 0 also counts as specular for routing" —
        // that code path is unreachable in the original due to the `continue` above
        // the BRDF-sample `if (i == 0 && afterTransmission)` line, so we preserve
        // that behaviour verbatim.
        result.pathAlive = true;
        return result;
    }

    // BRDF sample (non-transmission path).
    let F0_b = F0_h;
    var wi_b: vec3<f32>;
    let p_spec = mix(0.5, 0.98, h.metalness);
    let isSpecBounce = rand(seed) < p_spec;
    result.firstBounceSpec = isSpecBounce;
    // Note: `if (i == 0 && afterTransmission)` from original is unreachable here (see
    // transmission return above), so we don't replicate that assignment.
    if (isSpecBounce) {
        wi_b = sampleVNDF(wo, h.normal, h.shininess, seed);
        let cos_b = dot(h.normal, wi_b);
        if (cos_b <= 0.0) {
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
        let hb  = normalize(wo + wi_b);
        let Fb  = schlick(max(0.0, dot(wo, hb)), F0_b);
        let G1L = ggxG1(cos_b, h.shininess);
        let msC = msCompensation(F0_b, max(1e-4, dot(h.normal, wo)), h.shininess);
        throughput *= Fb * G1L * msC / p_spec;
    } else {
        wi_b = cosineHemisphere(h.normal, seed);
        let cos_b = dot(h.normal, wi_b);
        if (cos_b <= 0.0) {
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
        // K-C diffuse ms-comp throughput boost — matches the additive kcDiff in evalBrdf.
        let NdotV_kc = max(1e-4, dot(h.normal, wo));
        let E_kc     = sampleGgxELut(NdotV_kc, h.shininess);
        let F_avg_kc = (20.0 * F0_b + vec3<f32>(1.0)) / 21.0;
        let kcBoost  = vec3<f32>(1.0) + F_avg_kc * max(0.0, 1.0 - E_kc);
        throughput *= albedo * (1.0 - h.metalness) * kcBoost / (1.0 - p_spec);
    }
    result.prevWo        = wo;
    result.prevNormal    = h.normal;
    result.prevAlpha     = h.shininess;
    result.prevMetalness = h.metalness;
    result.afterTransmission = false;
    ray.origin = h.point + h.normal * 1e-3;
    ray.dir    = wi_b;

    result.rayOrigin  = ray.origin;
    result.rayDir     = ray.dir;
    result.throughput = throughput;
    result.diffRad    = diffRad;
    result.specRad    = specRad;
    result.pathAlive  = true;
    return result;
}
