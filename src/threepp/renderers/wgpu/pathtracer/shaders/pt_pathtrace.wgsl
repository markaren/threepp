

@group(0) @binding(12) var envCdfTex:     texture_2d<f32>;  // conditional CDF (per-row), R32Float
@group(0) @binding(13) var envMargTex:    texture_2d<f32>;  // marginal CDF (1-column), R32Float

const HAS_ENV_CDF: bool = /*ENV_CDF_FLAG*/false;

// r2Seq, ign, ign_t and the bnNext1d/2d helpers are defined in csCommonWGSL.

fn cosineHemisphere(n: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    let bn  = bnNext2d();
    let u1  = bn.x;
    let u2  = bn.y;
    let r   = sqrt(u1);
    let phi = 6.28318530718 * u2;
    let lx  = r * cos(phi);
    let ly  = r * sin(phi);
    let lz  = sqrt(max(0.0, 1.0 - u1));
    let nt  = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.99);
    let rgt = normalize(cross(nt, n));
    let up  = cross(n, rgt);
    return normalize(lx * rgt + ly * up + lz * n);
}

// Heitz 2018 VNDF (Visible Normal Distribution Function) sampling.
// Samples half-vectors proportional to the visible microfacet area,
// eliminating wasted below-horizon samples from plain D(h) sampling.
fn sampleVNDF(wo: vec3<f32>, n: vec3<f32>, alpha: f32,
              seed: ptr<function, u32>) -> vec3<f32> {
    // Build local frame around n
    let nt  = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.99);
    let t1  = normalize(cross(nt, n));
    let t2  = cross(n, t1);
    // Transform wo to local frame (t1, t2, n)
    let woLocal = vec3<f32>(dot(wo, t1), dot(wo, t2), dot(wo, n));
    // Stretch to isotropic configuration (isotropic alpha)
    let woStr = normalize(vec3<f32>(alpha * woLocal.x, alpha * woLocal.y, woLocal.z));
    // Build orthonormal basis around stretched wo
    let lensq = woStr.x * woStr.x + woStr.y * woStr.y;
    let T1 = select(vec3<f32>(1.0, 0.0, 0.0),
                    vec3<f32>(-woStr.y, woStr.x, 0.0) / sqrt(lensq),
                    lensq > 1e-7);
    let T2 = cross(woStr, T1);
    // Sample projected disk — blue-noise stratified.
    let bn  = bnNext2d();
    let u1  = bn.x;
    let u2  = bn.y;
    let r   = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let t1s = r * cos(phi);
    let s   = 0.5 * (1.0 + woStr.z);
    let t2s = mix(sqrt(max(0.0, 1.0 - t1s * t1s)), r * sin(phi), s);
    // Compute half-vector in stretched space, then unstretch
    let nhLocal = t1s * T1 + t2s * T2
                + sqrt(max(0.0, 1.0 - t1s * t1s - t2s * t2s)) * woStr;
    let hLocal = normalize(vec3<f32>(alpha * nhLocal.x, alpha * nhLocal.y, max(1e-6, nhLocal.z)));
    // Transform back to world space
    let hm = hLocal.x * t1 + hLocal.y * t2 + hLocal.z * n;
    return reflect(-wo, hm);
}

// PDF of the VNDF sampling strategy, expressed in reflected-direction (wi) space.
fn vndfPdf(wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>, alpha: f32) -> f32 {
    let hm    = normalize(wo + wi);
    let NdotH = max(0.0, dot(n, hm));
    let NdotV = max(1e-6, dot(n, wo));
    let D     = ggxD(NdotH, alpha);
    let G1v   = ggxG1(NdotV, alpha);
    // VNDF PDF_h = D * G1 * VdotH / NdotV; Jacobian to wi: / (4 * VdotH)
    // VdotH cancels → PDF_wi = D * G1 / (4 * NdotV)
    return D * G1v / (4.0 * NdotV);
}

// Combined BRDF PDF (mixed specular + diffuse lobes, matching the path tracer's sampling strategy).
fn brdfPdf(wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>, alpha: f32, metalness: f32) -> f32 {
    let NdotL = dot(n, wi);
    if (NdotL <= 0.0) { return 0.0; }
    let p_spec  = mix(0.5, 0.98, metalness);
    let specPdf = vndfPdf(wo, wi, n, alpha);
    let diffPdf = NdotL / PI;
    return p_spec * specPdf + (1.0 - p_spec) * diffPdf;
}

// Route a clamped contribution to diffuse or specular radiance buffer.
// Bounce 0 NEE: always diffuse (specular split handled at ReSTIR shade site).
// Bounce > 0:   follows firstBounceSpec flag.
fn addSplit(diff: ptr<function, vec3<f32>>,
            spec: ptr<function, vec3<f32>>,
            contrib: vec3<f32>, cap: f32,
            bounce: i32, firstSpec: bool) {
    var c = contrib;
    if (cap > 0.0) {
        let lum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
        if (lum > cap) { c *= cap / lum; }
    }
    if (bounce == 0 || !firstSpec) {
        *diff += c;
    } else {
        // Indirect specular: aggressive per-bounce clamp.
        // Bounce 1 = direct specular NEE (from bounce 0 split) → use full cap.
        // Bounce 2+ = indirect specular → clamp to 2.0 / bounce.
        // This is biased but eliminates the firefly speckle that spatial
        // filtering alone cannot remove at interactive sample counts.
        if (bounce >= 2) {
            let indirectCap = 2.0 / f32(bounce);
            let sLum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
            if (sLum > indirectCap) { c *= indirectCap / sLum; }
        }
        *spec += c;
    }
}

// ---------------------------------------------------------------------------
// ReSTIR DI — Reservoir data structure and helpers
// ---------------------------------------------------------------------------
struct Reservoir {
    lightPos:  vec3<f32>,   // world-space position (area/point) or direction (env/dir)
    lightType: f32,         // 0..999 = analytical light index, 1000+ = emissive tri (1000+triIdx), -1 = env
    W_sum:     f32,         // running weight sum
    M:         f32,         // candidate count
    W:         f32,         // final weight = W_sum / (M * p_hat)
    p_hat:     f32,         // target PDF of selected sample
}

fn emptyReservoir() -> Reservoir {
    return Reservoir(vec3<f32>(0.0), -1.0, 0.0, 0.0, 0.0, 0.0);
}

fn updateReservoir(r: ptr<function, Reservoir>,
                   pos: vec3<f32>, ltype: f32, w: f32,
                   p_hat_new: f32, seed: ptr<function, u32>) {
    (*r).W_sum += w;
    (*r).M += 1.0;
    if (rand(seed) < w / max((*r).W_sum, 1e-20)) {
        (*r).lightPos  = pos;
        (*r).lightType = ltype;
        (*r).p_hat     = p_hat_new;
    }
}

fn finalizeReservoir(r: ptr<function, Reservoir>) {
    (*r).W = (*r).W_sum / max((*r).M * (*r).p_hat, 1e-20);
}


// Evaluate unshadowed target function for a reservoir sample.
// Returns NdotL * luminance(Le) — geometry-weighted light intensity without BRDF.
//
// Using a BRDF-based target PDF introduced texture-pattern bias: the roughness clamp
// (safeAlpha) needed to tame GGX D spikes on smooth surfaces created a systematic
// mismatch between the reservoir weight and the actual shade-time BRDF evaluation.
// The mismatch was spatially correlated with the roughness texture, making the texture
// pattern visible in the lighting.
//
// A pure luminance target is unbiased, roughness-independent, and produces good
// reservoir quality for diffuse-dominant scenes. Specular variance is slightly higher
// but handled by the temporal accumulation.
fn restirTargetPdf(point: vec3<f32>, normal: vec3<f32>, wo: vec3<f32>,
                   albedo: vec3<f32>, metalness: f32, alpha: f32, F0: vec3<f32>,
                   lightPos: vec3<f32>, lightType: f32,
                   lightLe: vec3<f32>) -> f32 {
    // Determine direction to light
    let typeCode = i32(lightType);
    var ln: vec3<f32>;
    if (typeCode < 0) {
        // Environment: lightPos is direction
        ln = normalize(lightPos);
    } else if (typeCode < 1000) {
        // Analytical light
        let ltype = i32(rt.lightType[typeCode].x);
        if (ltype == 1) {
            ln = normalize(lightPos); // directional
        } else {
            ln = normalize(lightPos - point);
        }
    } else {
        // Emissive triangle
        ln = normalize(lightPos - point);
    }

    let NdotL = dot(normal, ln);
    if (NdotL <= 0.0) { return 0.0; }

    return NdotL * luminance(lightLe);
}

// Binary search a 1D CDF stored in a texture row.
// Returns the index where cdf[index] >= xi.
fn cdfSearch(tex: texture_2d<f32>, row: i32, size: i32, xi: f32) -> i32 {
    var lo = 0;
    var hi = size - 1;
    for (var iter = 0; iter < 16; iter++) {
        if (lo >= hi) { break; }
        let mid = (lo + hi) >> 1;
        if (textureLoad(tex, vec2<i32>(mid, row), 0).x < xi) { lo = mid + 1; } else { hi = mid; }
    }
    return lo;
}

// Direction → equirectangular UV (matches sampleEnv mapping)
fn dirToUV(d: vec3<f32>) -> vec2<f32> {
    let nd = normalize(d);
    let phi   = atan2(nd.z, nd.x);
    let theta = asin(clamp(nd.y, -1.0, 1.0));
    return vec2<f32>(0.5 + phi / (2.0 * PI), 0.5 + theta / PI);
}

// UV → direction (inverse of dirToUV)
fn uvToDir(uv: vec2<f32>) -> vec3<f32> {
    let phi   = (uv.x - 0.5) * 2.0 * PI;
    let theta = (uv.y - 0.5) * PI;
    let ct = cos(theta);
    return vec3<f32>(ct * cos(phi), sin(theta), ct * sin(phi));
}

// Importance-sample the environment map using precomputed 2D CDF.
// Returns: xyz = sampled direction, w = PDF (in solid angle measure).
fn sampleEnvImportance(seed: ptr<function, u32>) -> vec4<f32> {
    let envW = i32(rt.envIntensity.y);
    let envH = i32(rt.envIntensity.z);

    // 1) Sample marginal CDF to pick a row (v)
    let xi_v = rand(seed);
    let row  = cdfSearch(envMargTex, 0, envH, xi_v);

    // 2) Sample conditional CDF at that row to pick a column (u)
    let xi_u = rand(seed);
    let col  = cdfSearch(envCdfTex, row, envW, xi_u);

    // 3) Convert to UV with sub-pixel JITTER (not center snap). The pdf below
    // treats the density as uniform-within-pixel (pdf_uv * W * H). Snapping to
    // centers turns this into a Riemann midpoint rule and biases smooth BRDF
    // integrands — the white-furnace env test saw a ~7% deficit at 8×4 env.
    let ju = rand(seed);
    let jv = rand(seed);
    let u = (f32(col) + ju) / f32(envW);
    let v = (f32(row) + jv) / f32(envH);
    let dir = uvToDir(vec2<f32>(u, v));

    // 4) Compute PDF = luminance(pixel) / totalLuminance, converted to solid angle.
    // totalSum is Σ lum*cos(lat) (see buildEnvCdf), so p(pixel) = lum*cos/totalSum
    // and pdf_uv = lum*cos*W*H/totalSum. The equirect Jacobian 2π²*cos(lat) in the
    // uv→ω conversion cancels the cos(lat) in pdf_uv exactly, so pdf_ω is just
    // lum*W*H / (2π²*totalSum). Writing the cos factor on both sides and then
    // cancelling was a bug — the stray 1/cos made polar samples badly weighted.
    let envCol = textureLoad(envTex, vec2<i32>(col, row), 0).xyz;
    let lum = 0.2126 * envCol.r + 0.7152 * envCol.g + 0.0722 * envCol.b + 1e-10;
    let totalSum = rt.envIntensity.w;
    let pdf = lum * f32(envW * envH) / (2.0 * PI * PI * max(totalSum, 1e-10));

    return vec4<f32>(dir, pdf);
}

// PDF for a given direction under env importance sampling (for MIS).
// Matches sampleEnvImportance — cos(lat) in pdf_uv numerator cancels the
// equirect Jacobian's cos(lat), so pdf_ω = lum*W*H / (2π²*totalSum).
fn envImportancePdf(d: vec3<f32>) -> f32 {
    let envW = i32(rt.envIntensity.y);
    let envH = i32(rt.envIntensity.z);
    let envCol = sampleEnv(d);
    let lum = 0.2126 * envCol.r + 0.7152 * envCol.g + 0.0722 * envCol.b + 1e-10;
    let totalSum = rt.envIntensity.w;
    return lum * f32(envW * envH) / (2.0 * PI * PI * max(totalSum, 1e-10));
}

// Henyey-Greenstein phase. g∈(-1,1): 0 isotropic, >0 forward (god rays), <0 back.
fn phaseHG(cosTh: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cosTh;
    return (1.0 - g2) / (4.0 * PI * max(denom * sqrt(max(denom, 1e-6)), 1e-6));
}

// Single-scattering fog inscatter: samples a scatter point along the primary
// ray (truncated free-flight) and does NEE to emissive triangles, analytical
// lights, and the env map so the volume lights up near real sources instead of
// showing a flat ambient tint. Henyey-Greenstein phase (g=fogColor.w) gives
// forward-scattering "god rays"; shadow rays pick up fog attenuation via
// traceShadowRay so the light-up softens with distance automatically.
fn volumeInscatter(rayOrigin: vec3<f32>, rayDir: vec3<f32>, maxT: f32,
                   seed: ptr<function, u32>) -> vec3<f32> {
    let sigmaT = rt.fog.x;
    if (sigmaT <= 0.0) { return vec3<f32>(0.0); }
    let tMax = min(maxT, 1e5);
    let T_max = exp(-sigmaT * tMax);
    let p1 = 1.0 - T_max;
    if (p1 < 1e-4) { return vec3<f32>(0.0); }

    // Estimator factor: σ_s·T(t)/p(t) = (σ_s/σ_t)·(1 − T_max).
    // rt.fogColor.xyz acts as the single-scattering albedo σ_s/σ_t.
    let scatterFactor = rt.fogColor.xyz * p1;
    let g = clamp(rt.fogColor.w, -0.95, 0.95);

    // N stratified depth samples per pixel. Beams come from spatial occlusion
    // variation in the volume — stratifying t exposes lit/shadowed segments of
    // the camera ray in a single frame, so "god rays" resolve 4× faster than
    // with one independent sample per frame. Each stratum casts its own shadow
    // rays since occlusion varies with scatter position.
    const N: u32 = 4u;
    var inscatter = vec3<f32>(0.0);
    let envMode    = i32(rt.envColor.w);
    let lcount     = i32(rt.lightCount.x);
    let emTriCount = i32(rt.emissiveInfo.x);
    let totalPower = rt.emissiveInfo.y;

    for (var i = 0u; i < N; i++) {
        // Truncated free-flight, stratified: ξ ∈ [i/N, (i+1)/N).
        // p(t) = σ_t·e^(-σ_t·t)/p1 over [0, tMax]; inverse CDF gives t.
        let xi = (f32(i) + rand(seed)) / f32(N);
        let tScatter = -log(max(1.0 - xi * p1, 1e-30)) / sigmaT;
        let x = rayOrigin + rayDir * tScatter;

        // --- NEE to emissive triangles ---
        if (emTriCount > 0 && totalPower > 0.0) {
            let es = sampleEmissiveTriCdf(seed, totalPower, emTriCount);
            let toL  = es.point - x;
            let dist = max(length(toL), 1e-4);
            let dir  = toL / dist;
            let cosLight = max(0.0, dot(-dir, es.normal));
            if (cosLight > 1e-4) {
                let atten   = traceShadowRay(x, vec3<f32>(0.0), dir, dist - 1e-2, 4);
                let pLight  = es.power / max(totalPower, 1e-6);
                let pdfArea = pLight / max(es.area, 1e-6);
                let pdfOmega = pdfArea * dist * dist / cosLight;
                let matIdx   = i32(textureLoad(triData, triCoord(es.triIdx, 0), 0).w);
                let emission = textureLoad(matData, vec2<i32>(matIdx, 2), 0).xyz;
                let phE = phaseHG(dot(rayDir, dir), g);
                if (pdfOmega > 1e-10) {
                    inscatter += phE * emission * atten / pdfOmega;
                }
            }
        }

        // --- NEE to analytical lights (point/dir/spot) ---
        for (var li = 0; li < lcount; li++) {
            let le = evalAnalyticalLight(li, x);
            let shadowDist = select(le.dist - 1e-2, 1e30, le.dist >= 1e29);
            let atten = traceShadowRay(x, vec3<f32>(0.0), le.dir, shadowDist, 4);
            let phA = phaseHG(dot(rayDir, le.dir), g);
            inscatter += phA * le.color * atten;
        }

        // --- NEE to environment (sky) ---
        if (envMode == 2 && HAS_ENV_CDF) {
            let es = sampleEnvImportance(seed);
            let dirE = es.xyz;
            let pdfE = es.w;
            if (pdfE > 1e-10) {
                let atten = traceShadowRay(x, vec3<f32>(0.0), dirE, 1e30, 4);
                let Le = sampleEnv(dirE);
                let phV = phaseHG(dot(rayDir, dirE), g);
                inscatter += phV * Le * atten / pdfE;
            }
        } else if (envMode == 1) {
            let u1 = rand(seed);
            let u2 = rand(seed);
            let z  = 1.0 - 2.0 * u1;
            let r  = sqrt(max(0.0, 1.0 - z * z));
            let ph = 6.2831853 * u2;
            let dirE = vec3<f32>(r * cos(ph), r * sin(ph), z);
            let pdfE = 1.0 / (4.0 * PI);
            let atten = traceShadowRay(x, vec3<f32>(0.0), dirE, 1e30, 4);
            let Le = rt.envColor.xyz;
            let phV = phaseHG(dot(rayDir, dirE), g);
            inscatter += phV * Le * atten / pdfE;
        }
    }

    return inscatter * scatterFactor / f32(N);
}

// Per-contribution firefly clamp for injection-time MIS spikes.
fn addClamped(rad: ptr<function, vec3<f32>>, contrib: vec3<f32>, cap: f32) {
    if (!(contrib.x == contrib.x) || !(contrib.y == contrib.y) || !(contrib.z == contrib.z)) {
        return;
    }
    let lum = dot(contrib, vec3<f32>(0.2126, 0.7152, 0.0722));
    if (lum > cap) { *rad += contrib * (cap / lum); }
    else           { *rad += contrib; }
}

// Finite check on all 3 channels — catches both NaN (x != x) and Inf
// (abs(x) > 1e30). Existing accumulator guards used `select(0, v, v.x == v.x)`
// which only catches NaN on the x channel; an Inf in any channel or a NaN in
// y/z slipped through, then a single bilinear-reproj sample at camera motion
// spread the bad value across the scene as ever more pixels reprojected onto
// the contaminated tap.
fn finite3(v: vec3<f32>) -> bool {
    return v.x == v.x && v.y == v.y && v.z == v.z &&
           abs(v.x) < 1e30 && abs(v.y) < 1e30 && abs(v.z) < 1e30;
}
fn cleanFinite3(v: vec3<f32>) -> vec3<f32> {
    return select(vec3<f32>(0.0), v, finite3(v));
}

fn isMeshMoved(idx: i32) -> bool {
    if (idx < 0 || idx >= 128) { return false; }
    let ui  = u32(idx);
    let bit = ui & 31u;
    let wi  = ui >> 5u;  // 0..3 — selects x/y/z/w of movedMeshBits
    return ((rt.movedMeshBits[wi] >> bit) & 1u) != 0u;
}

// -------- Octahedral normal encoding (for GI reservoir packing) --------
fn octEncode(n: vec3<f32>) -> vec2<f32> {
    let t = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) {
        return (1.0 - abs(t.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), t.xy >= vec2<f32>(0.0));
    }
    return t;
}
fn octDecode(e: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        let xy = (1.0 - abs(n.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0));
        n = vec3<f32>(xy, n.z);
    }
    return normalize(n);
}
fn packOctNormal(n: vec3<f32>) -> f32 {
    let e = octEncode(n);
    return bitcast<f32>(pack2x16snorm(e));
}
fn unpackOctNormal(p: f32) -> vec3<f32> {
    return octDecode(unpack2x16snorm(bitcast<u32>(p)));
}

// -------- GI target PDF: importance of a secondary hit as a virtual light --------
fn giTargetPdf(point: vec3<f32>, normal: vec3<f32>, wo: vec3<f32>,
               albedo: vec3<f32>, metalness: f32, alpha: f32, F0: vec3<f32>,
               secHitPos: vec3<f32>, secHitNorm: vec3<f32>, Lo: vec3<f32>) -> f32 {
    // Importance-sampling p_hat — only proportionality to true contribution
    // matters, not physical accuracy. The final GI shade (~line 2119) uses the
    // real evalBrdfFullSplit; W's `W_sum / (M * p_hat_chosen)` cancels the
    // proxy provided source + reuse use the same proxy consistently.
    // Proxy: lambertian reflectance (albedo for dielectrics, F0 for metals)
    // times NdotL times geometry term. Drops GGX D/G/F + LUT fetch — ~30 FLOPs
    // + 1 texture sample saved per call, 5 calls per primary pixel for GI.
    let wi = normalize(secHitPos - point);
    let NdotL = dot(normal, wi);
    if (NdotL <= 0.0) { return 0.0; }
    let delta = secHitPos - point;
    let dist2 = dot(delta, delta);
    let cosTheta2 = max(dot(secHitNorm, -wi), 0.0);
    let G = cosTheta2 / max(dist2, 0.01);
    let rhoProxy = albedo * (1.0 - metalness) + F0 * metalness;
    return luminance(rhoProxy * Lo) * NdotL * G;
}

// Jacobian of the reconnection shift: ratio of differential solid angles subtended
// by the secondary hit (secPos/secNorm) as seen from two different primary shading
// points (fromPrimary = source, toPrimary = current).
// J = (cosθ_to · d²_from) / (cosθ_from · d²_to)
// where θ = angle at secNorm vs. direction toward the respective primary.
// Equals 1 when from == to (static scene).  Clamped to [0, 4] to prevent
// fireflies from near-grazing secondary hits.
fn reconnJacobian(secPos: vec3<f32>, secNorm: vec3<f32>,
                  fromPrimary: vec3<f32>, toPrimary: vec3<f32>) -> f32 {
    let vFrom = fromPrimary - secPos;
    let vTo   = toPrimary   - secPos;
    let d2From = dot(vFrom, vFrom);
    let d2To   = dot(vTo,   vTo);
    let cosFrom = abs(dot(secNorm, vFrom) / max(sqrt(d2From), 1e-6));
    let cosTo   = abs(dot(secNorm, vTo)   / max(sqrt(d2To),   1e-6));
    return clamp((cosTo * d2From) / max(cosFrom * d2To, 1e-6), 0.0, 4.0);
}

struct SplitRadiance { diff: vec3<f32>, spec: vec3<f32> }

// PrimaryShadeResult: all state that flows from primaryShade (bounce-0) into
// runBounces (bounces 1..N).  Serialized to pathStateBuf at the kernel boundary
// between rt_main and rt_bounces_main — keep field set in sync with that layout.
struct PrimaryShadeResult {
    rayOrigin:         vec3<f32>,
    rayDir:            vec3<f32>,
    throughput:        vec3<f32>,
    diffRad:           vec3<f32>,  // radiance accumulated during primaryShade
    specRad:           vec3<f32>,
    prevNormal:        vec3<f32>,
    prevWo:            vec3<f32>,
    prevAlpha:         f32,
    prevMetalness:     f32,
    afterTransmission: bool,
    firstBounceSpec:   bool,
    effectiveBounces:  i32,
    // Bounce-0 surface snapshot — consumed by ReSTIR GI at bounce 1 inside runBounces.
    b0Point:   vec3<f32>,
    b0Normal:  vec3<f32>,
    b0Wo:      vec3<f32>,
    b0Albedo:  vec3<f32>,
    b0F0:      vec3<f32>,
    b0Metal:   f32,
    b0Alpha:   f32,
    b0MeshIdx: i32,
    giResStored: bool,
    // false = primaryShade terminated the path (miss/unlit/backface/RR) and
    // runBounces must be skipped.  Callers should return diffRad+specRad directly.
    pathAlive: bool,
    // Primary-hit metadata needed by rt_bounces_main's accumulation.
    // primaryDepth == 0.0 for sky/miss (denoiser sentinel).  primaryMatIdx is
    // propagated to hitMesh.g channel.
    primaryDepth:  f32,
    primaryMatIdx: i32,
}
