// Shared primary-hit shading for the hybrid raster + PT pipeline.
//
// Used by raygen.rgen when the hybrid push-constant bit is set: instead of
// casting a primary traceRayEXT, raygen reads the raster G-buffer's hit
// info and calls shadePrimaryHitOpaque() to produce the same payload outputs
// chit's main() would have written. Bounce loop continues unchanged.
//
// Stage 1A v1 scope:
//   ✓ albedo/roughness/metalness lookup with bindless textures
//   ✓ Cook-Torrance specular + Lambert diffuse
//   ✓ Kulla-Conty multi-scatter compensation (matches chit)
//   ✓ NEE for env (importance sampling via CDF when bound, uniform fallback)
//   ✓ NEE for analytic lights (dirs, points, spots, rects) with shadow rays
//   ✓ NEE for emissive triangles (CDF binary search)
//   ✓ 3-way stochastic BSDF sample (clearcoat / spec / diff) for next bounce
//   ✗ Iridescence / clearcoat / transmission / sheen (transmissive surfaces
//      fall back to chit's primary trace; non-zero clearcoat/iridescence/
//      sheen materials get slightly degraded primary, secondary bounces are
//      handled correctly by chit on subsequent traces)
//   ✗ Photon caustic gather (secondary bounces still gather via chit)
//   ✗ Foam / FFT-water primary handling (water hits the transmission fallback)
//
// Includer must have declared:
//   - PI / TWO_PI constants
//   - struct Payload payload (rayPayloadEXT or rayPayloadInEXT)
//   - layout(set=0, binding=0)  accelerationStructureEXT topAS
//   - layout(set=0, binding=4)  MatDescBuf  { MaterialDesc mats[]; }
//   - layout(set=0, binding=5)  LightsUbo  lights
//   - layout(set=0, binding=6)  sampler2D  envTex
//   - layout(set=0, binding=8)  sampler2D  albedoMaps[kMaxMaterialTextures]
//   - layout(set=0, binding=14) EmissiveTriBuf { EmTri emissiveTris[]; }
//   - layout(set=0, binding=18) sampler2D envCdfTex
//   - layout(set=0, binding=19) sampler2D envMargTex
//   - layout(push_constant) Pc with envCdfWidth/Height/TotalSum, fireflyClamp,
//     emissiveCount, emissiveTotalPower
//   - rayPayloadEXT float shadowVisibility @ location 1
//   - urand(inout uint), pcgNext(inout uint) PRNG helpers
//   - sampleEquirectFog(vec3) or equivalent envTex sampler
//   - DirLight / PointLight / SpotLight / RectLight / EmTri struct definitions
//   - kMaxMaterialTextures from vulkan_shared.h

#ifndef THREEPP_VULKAN_SHADE_PRIMARY_GLSL
#define THREEPP_VULKAN_SHADE_PRIMARY_GLSL

// ── Hit context: caller fills before calling shadePrimaryHitOpaque ────────
struct HitContext {
    vec3 worldPos;          // hit point in world space
    vec3 worldNormalGeom;   // geometric/interpolated normal (normalized)
    vec2 uv;                // material UV (zeros if mesh has no UV)
    uint instanceId;        // gl_InstanceCustomIndexEXT for the hit mesh
    vec3 rayOrigin;         // hit ray origin (camera or prev hit)
    vec3 rayDir;            // hit ray direction (toward surface, normalized)
    float hitT;             // distance from rayOrigin to worldPos
};

// ── BSDF math helpers (mirrored from closest_hit.rchit) ───────────────────
float spDistGGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float spGeomSmithG1(float NdotX, float k) {
    return NdotX / (NdotX * (1.0 - k) + k);
}

vec3 spFresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float spGgxEApprox(float NdotV, float alpha) {
    const float a2 = alpha * alpha;
    return 1.0 - 0.5 / (1.0 + 7.6 * a2 * max(NdotV, 1e-3));
}

vec3 spKcDiff(vec3 albedo, float metalness, vec3 F0, float NdotV, float alpha) {
    const float E_v   = spGgxEApprox(NdotV, alpha);
    const vec3  F_avg = (20.0 * F0 + vec3(1.0)) / 21.0;
    return albedo * (1.0 - metalness) * F_avg * max(0.0, 1.0 - E_v) / PI;
}

vec3 spKcBoost(vec3 F0, float NdotV, float alpha) {
    const float E_kc  = spGgxEApprox(NdotV, alpha);
    const vec3  F_avg = (20.0 * F0 + vec3(1.0)) / 21.0;
    return vec3(1.0) + F_avg * max(0.0, 1.0 - E_kc);
}

float spVndfPdf(vec3 wo, vec3 wi, vec3 n, float roughness) {
    const vec3 hm = normalize(wo + wi);
    const float NdotH = max(0.0, dot(n, hm));
    const float NdotV = max(1e-6, dot(n, wo));
    const float D     = spDistGGX(NdotH, roughness);
    const float k     = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    const float G1v   = spGeomSmithG1(NdotV, k);
    return D * G1v / (4.0 * NdotV);
}

float spBrdfPdf(vec3 wo, vec3 wi, vec3 n, float roughness, float metalness) {
    const float NdotL = dot(n, wi);
    if (NdotL <= 0.0) return 0.0;
    const float pSpec   = mix(0.5, 0.98, metalness);
    const float specPdf = spVndfPdf(wo, wi, n, roughness);
    const float diffPdf = NdotL * (1.0 / PI);
    return pSpec * specPdf + (1.0 - pSpec) * diffPdf;
}

vec3 spCosineHemisphere(vec2 u) {
    const float r   = sqrt(u.x);
    const float phi = TWO_PI * u.y;
    return vec3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u.x)));
}

mat3 spMakeTBN(vec3 N) {
    const vec3 up = abs(N.z) < 0.99 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    const vec3 T  = normalize(cross(up, N));
    const vec3 B  = cross(N, T);
    return mat3(T, B, N);
}

// Spherical-cap VNDF (Dupuy 2023). Always above-horizon — no rejection retries.
vec3 spSampleVNDF_H(vec3 wo, vec3 n, float alpha, vec2 u) {
    const vec3 nt = abs(n.y) < 0.99 ? vec3(0.0, 1.0, 0.0)
                                    : vec3(1.0, 0.0, 0.0);
    const vec3 t1 = normalize(cross(nt, n));
    const vec3 t2 = cross(n, t1);
    const mat3 W2L = mat3(t1, t2, n);
    const mat3 L2W = transpose(W2L);

    const vec3 woLocal = vec3(dot(wo, t1), dot(wo, t2), dot(wo, n));
    const vec3 woHemi  = normalize(vec3(woLocal.xy * alpha, woLocal.z));
    const float phi = TWO_PI * u.x;
    const float a   = clamp(woHemi.z, -1.0, 1.0);
    const float z   = (1.0 - u.y) * (1.0 + a) - a;
    const float sinT = sqrt(max(0.0, 1.0 - z*z));
    const vec3 c     = vec3(sinT * cos(phi), sinT * sin(phi), z);
    const vec3 nhHemi = c + woHemi;
    const vec3 hLocal = normalize(vec3(nhHemi.xy * alpha, max(0.0, nhHemi.z)));
    return normalize(L2W * hLocal);
}

vec3 spSampleVNDF(vec3 wo, vec3 n, float alpha, vec2 u) {
    const vec3 h  = spSampleVNDF_H(wo, n, alpha, u);
    return reflect(-wo, h);
}

// ── Env importance sampling (mirror of chit) ──────────────────────────────
vec2 spDirToEquirectUV(vec3 dir) {
    return vec2(0.5 + atan(dir.z, dir.x) / TWO_PI,
                0.5 + asin(clamp(dir.y, -1.0, 1.0)) / PI);
}

vec3 spUvToEquirectDir(vec2 uv) {
    const float phi   = (uv.x - 0.5) * TWO_PI;
    const float theta = (uv.y - 0.5) * PI;
    const float ct    = cos(theta);
    return vec3(ct * cos(phi), sin(theta), ct * sin(phi));
}

int spEnvCdfSearch(sampler2D tex, int row, int size, float xi) {
    int lo = 0;
    int hi = size - 1;
    for (int it = 0; it < 16; ++it) {
        if (lo >= hi) break;
        const int mid = (lo + hi) >> 1;
        const float v = texelFetch(tex, ivec2(mid, row), 0).x;
        if (v < xi) lo = mid + 1; else hi = mid;
    }
    return lo;
}

vec4 spSampleEnvImportance(inout uint seed) {
    if (pc.envCdfTotalSum <= 0.0) return vec4(0.0);
    const int W = int(pc.envCdfWidth);
    const int H = int(pc.envCdfHeight);
    if (W <= 0 || H <= 0) return vec4(0.0);

    const int row = spEnvCdfSearch(envMargTex, 0, H, urand(seed));
    const int col = spEnvCdfSearch(envCdfTex, row, W, urand(seed));
    const vec2 uv = vec2((float(col) + urand(seed)) / float(W),
                        (float(row) + urand(seed)) / float(H));
    const vec3 dir = normalize(spUvToEquirectDir(uv));

    const vec3 envCol = texelFetch(envTex, ivec2(col, row), 0).rgb;
    const float lum   = dot(envCol, vec3(0.2126, 0.7152, 0.0722)) + 1e-10;
    const float pdfOmega = lum * float(W * H) / (2.0 * PI * PI * max(pc.envCdfTotalSum, 1e-10));
    return vec4(dir, pdfOmega);
}

float spEnvImportancePdf(vec3 dir) {
    if (pc.envCdfTotalSum <= 0.0) return 0.0;
    const int W = int(pc.envCdfWidth);
    const int H = int(pc.envCdfHeight);
    if (W <= 0 || H <= 0) return 0.0;
    const vec2 uv = spDirToEquirectUV(normalize(dir));
    const int col = clamp(int(uv.x * float(W)), 0, W - 1);
    const int row = clamp(int(uv.y * float(H)), 0, H - 1);
    const vec3 envCol = texelFetch(envTex, ivec2(col, row), 0).rgb;
    const float lum   = dot(envCol, vec3(0.2126, 0.7152, 0.0722)) + 1e-10;
    return lum * float(W * H) / (2.0 * PI * PI * max(pc.envCdfTotalSum, 1e-10));
}

// ── Shadow ray: returns 1.0 if light is unoccluded, 0.0 if occluded.
// Reuses raygen's existing shadowVisibility payload at location 1.
// maxDist = ray length (use 1e20 for directional / env "infinite" lights).
float spShadowRay(vec3 origin, vec3 dir, float maxDist) {
    shadowVisibility = 0.0;
    traceRayEXT(topAS,
                gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT |
                        gl_RayFlagsSkipClosestHitShaderEXT,
                0xFFu,
                /*sbtRecordOffset*/ 0u,
                /*sbtRecordStride*/ 0u,
                /*missIndex*/ 1u,// shadow miss
                origin, 0.001,
                dir, maxDist, /*payload location*/ 1);
    return shadowVisibility;
}

// ── Sample a direction inside a uniform sphere octant, used for env NEE
// fallback when no CDF is bound. Cosine-weighted on the upper hemisphere
// gives less variance than uniform-sphere when the env is approximately
// uniform; matches WGPU pattern.
vec3 spSampleHemisphereCos(vec3 N, vec2 u, out float pdf) {
    const vec3 d = spCosineHemisphere(u);
    const mat3 tbn = spMakeTBN(N);
    pdf = max(d.z, 0.0) / PI;
    return tbn * d;
}

// ── Main entry point: shade a primary hit on an opaque surface ────────────
// Caller fills HitContext from raster G-buffer reads, then invokes this.
// Writes payload outputs (radiance/brdfWeight/nextOrigin/nextDir/etc.) so
// raygen's bounce loop can continue from bounce 1 as if a chit had run.
//
// Returns false if the material is non-opaque (transmission > threshold) or
// otherwise unsupported by this v1 path; caller should fall back to a real
// primary traceRayEXT in that case.
bool shadePrimaryHitOpaque(HitContext ctx, inout uint seed) {
    const MaterialDesc m = mats[ctx.instanceId];

    // Transmissive materials (glass / water) — defer to existing chit path.
    if (m.transmission > 0.05 || m.thinWalled != 0) return false;
    // Unlit MeshBasicMaterial — defer (chit emits the unlit early-out).
    if (m.roughness < 0.0) return false;
    // Alpha-test cutout materials — the raster gbuffer pass doesn't
    // discard on alpha yet, so its IDs are unreliable at supposed-to-be-
    // transparent pixels. Defer to chit which has the alpha any-hit path.
    if (m.alphaCutoff > 0.0) return false;

    // ── Material lookup with bindless textures ─────────────────────────────
    const vec2 uvAlbedo     = (m.uvTransform           * vec3(ctx.uv, 1.0)).xy;
    const vec2 uvRoughMetal = (m.uvTransformRoughMetal * vec3(ctx.uv, 1.0)).xy;
    const vec2 uvEmissive   = (m.uvTransformEmissive   * vec3(ctx.uv, 1.0)).xy;

    vec3 albedo = m.albedo;
    if (m.albedoTexIndex >= 0) {
        const int i = clamp(m.albedoTexIndex, 0, int(kMaxMaterialTextures) - 1);
        albedo *= texture(albedoMaps[i], uvAlbedo).rgb;
    }
    float roughness = m.roughness;
    if (m.roughnessTexIndex >= 0) {
        const int i = clamp(m.roughnessTexIndex, 0, int(kMaxMaterialTextures) - 1);
        roughness *= texture(albedoMaps[i], uvRoughMetal).g;
    }
    float metalness = m.metalness;
    if (m.metalnessTexIndex >= 0) {
        const int i = clamp(m.metalnessTexIndex, 0, int(kMaxMaterialTextures) - 1);
        metalness *= texture(albedoMaps[i], uvRoughMetal).b;
    }
    roughness = clamp(roughness, 0.04, 1.0);
    metalness = clamp(metalness, 0.0,  1.0);

    // Emissive contribution (self-emission at the surface).
    vec3 emissive = m.emissive * m.emissiveIntensity;
    if (m.emissiveTexIndex >= 0) {
        const int i = clamp(m.emissiveTexIndex, 0, int(kMaxMaterialTextures) - 1);
        emissive *= texture(albedoMaps[i], uvEmissive).rgb;
    }

    const vec3 N = ctx.worldNormalGeom;
    const vec3 V = normalize(-ctx.rayDir);
    const float NdotV = max(dot(N, V), 1e-3);

    const vec3 F0    = mix(vec3(0.04) * m.specularIntensity * m.specularColor, albedo, metalness);
    const float k    = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    const float alpha = roughness * roughness;

    // Shadow ray origin offset along N to avoid self-intersection.
    const vec3 shadowOrigin = ctx.worldPos + N * 1e-3;

    // ── Direct lighting: ambient + analytic NEE + emissive NEE + env NEE ──
    vec3 direct = emissive + lights.ambient * albedo * (1.0 - metalness);

    const float fc = pc.fireflyClamp;

    // Helper: evaluate Cook-Torrance BSDF for a given light direction.
    // Returns (diff + spec) · NdotL · light.color (caller applies).
    // -- inlined per loop below --

    // --- Directional lights ---
    for (uint i = 0u; i < lights.dirCount; ++i) {
        const vec3  L     = -normalize(lights.dirLights[i].direction);
        const float NdotL = dot(N, L);
        if (NdotL <= 0.0) continue;
        const float vis = spShadowRay(shadowOrigin, L, 1e20);
        if (vis <= 0.0) continue;

        const vec3 H     = normalize(V + L);
        const float NdotH = max(dot(N, H), 0.0);
        const float VdotH = max(dot(V, H), 0.0);
        const float D    = spDistGGX(NdotH, roughness);
        const float G    = spGeomSmithG1(NdotV, k) * spGeomSmithG1(NdotL, k);
        const vec3  F    = spFresnelSchlick(VdotH, F0);
        const vec3  spec = D * G * F / max(4.0 * NdotV * NdotL, 1e-6);
        const vec3  diff = albedo * (1.0 - metalness) * (vec3(1.0) - F) / PI;
        vec3 contrib     = (diff + spec) * NdotL * lights.dirLights[i].color;
        // Add Kulla-Conty diffuse multi-scatter compensation (matches chit).
        contrib += spKcDiff(albedo, metalness, F0, NdotV, alpha) * NdotL * lights.dirLights[i].color;
        const float lum = dot(contrib, vec3(0.2126, 0.7152, 0.0722));
        if (lum > fc) contrib *= fc / lum;
        direct += contrib;
    }

    // --- Point lights ---
    for (uint i = 0u; i < lights.pointCount; ++i) {
        const PointLight pl = lights.pointLights[i];
        const vec3 toL  = pl.position - ctx.worldPos;
        const float dist = length(toL);
        if (dist < 1e-4) continue;
        const vec3 L = toL / dist;
        const float NdotL = dot(N, L);
        if (NdotL <= 0.0) continue;
        const float vis = spShadowRay(shadowOrigin, L, dist - 1e-3);
        if (vis <= 0.0) continue;

        // Inverse-square attenuation with optional smooth range cutoff (matches three.js).
        float atten = 1.0 / max(dist * dist, 1e-4);
        if (pl.range > 0.0) atten *= clamp(1.0 - pow(dist / pl.range, 4.0), 0.0, 1.0);
        if (pl.decay > 0.0) atten *= pow(max(dist, 1e-4), -pl.decay + 2.0);

        const vec3 H     = normalize(V + L);
        const float NdotH = max(dot(N, H), 0.0);
        const float VdotH = max(dot(V, H), 0.0);
        const float D    = spDistGGX(NdotH, roughness);
        const float G    = spGeomSmithG1(NdotV, k) * spGeomSmithG1(NdotL, k);
        const vec3  F    = spFresnelSchlick(VdotH, F0);
        const vec3  spec = D * G * F / max(4.0 * NdotV * NdotL, 1e-6);
        const vec3  diff = albedo * (1.0 - metalness) * (vec3(1.0) - F) / PI;
        vec3 contrib = (diff + spec) * NdotL * pl.color * atten;
        contrib += spKcDiff(albedo, metalness, F0, NdotV, alpha) * NdotL * pl.color * atten;
        const float lum = dot(contrib, vec3(0.2126, 0.7152, 0.0722));
        if (lum > fc) contrib *= fc / lum;
        direct += contrib;
    }

    // --- Spot lights ---
    for (uint i = 0u; i < lights.spotCount; ++i) {
        const SpotLight sl = lights.spotLights[i];
        const vec3 toL = sl.position - ctx.worldPos;
        const float dist = length(toL);
        if (dist < 1e-4) continue;
        const vec3 L = toL / dist;
        const float NdotL = dot(N, L);
        if (NdotL <= 0.0) continue;
        // Cone falloff
        const float cosToLight = dot(-L, sl.direction);
        if (cosToLight < sl.cosAngleOuter) continue;
        const float coneAtten = smoothstep(sl.cosAngleOuter, sl.cosAngleInner, cosToLight);

        const float vis = spShadowRay(shadowOrigin, L, dist - 1e-3);
        if (vis <= 0.0) continue;

        float atten = 1.0 / max(dist * dist, 1e-4);
        if (sl.range > 0.0) atten *= clamp(1.0 - pow(dist / sl.range, 4.0), 0.0, 1.0);
        if (sl.decay > 0.0) atten *= pow(max(dist, 1e-4), -sl.decay + 2.0);
        atten *= coneAtten;

        const vec3 H     = normalize(V + L);
        const float NdotH = max(dot(N, H), 0.0);
        const float VdotH = max(dot(V, H), 0.0);
        const float D    = spDistGGX(NdotH, roughness);
        const float G    = spGeomSmithG1(NdotV, k) * spGeomSmithG1(NdotL, k);
        const vec3  F    = spFresnelSchlick(VdotH, F0);
        const vec3  spec = D * G * F / max(4.0 * NdotV * NdotL, 1e-6);
        const vec3  diff = albedo * (1.0 - metalness) * (vec3(1.0) - F) / PI;
        vec3 contrib = (diff + spec) * NdotL * sl.color * atten;
        contrib += spKcDiff(albedo, metalness, F0, NdotV, alpha) * NdotL * sl.color * atten;
        const float lum = dot(contrib, vec3(0.2126, 0.7152, 0.0722));
        if (lum > fc) contrib *= fc / lum;
        direct += contrib;
    }

    // --- Rect lights (sample a single point at the rect center for v1) ---
    for (uint i = 0u; i < lights.rectCount; ++i) {
        const RectLight rl = lights.rectLights[i];
        const vec2 ru = vec2(urand(seed), urand(seed)) * 2.0 - 1.0;
        const vec3 lp = rl.position + ru.x * rl.halfU + ru.y * rl.halfV;
        const vec3 toL = lp - ctx.worldPos;
        const float dist = length(toL);
        if (dist < 1e-4) continue;
        const vec3 L = toL / dist;
        const float NdotL = dot(N, L);
        if (NdotL <= 0.0) continue;
        const float lightCos = max(dot(rl.normal, -L), 0.0);
        if (lightCos <= 0.0) continue;// behind the emitter
        const float vis = spShadowRay(shadowOrigin, L, dist - 1e-3);
        if (vis <= 0.0) continue;

        const float area = 4.0 * length(rl.halfU) * length(rl.halfV);
        const float atten = (lightCos * area) / max(dist * dist, 1e-4);
        const vec3 H     = normalize(V + L);
        const float NdotH = max(dot(N, H), 0.0);
        const float VdotH = max(dot(V, H), 0.0);
        const float D    = spDistGGX(NdotH, roughness);
        const float G    = spGeomSmithG1(NdotV, k) * spGeomSmithG1(NdotL, k);
        const vec3  F    = spFresnelSchlick(VdotH, F0);
        const vec3  spec = D * G * F / max(4.0 * NdotV * NdotL, 1e-6);
        const vec3  diff = albedo * (1.0 - metalness) * (vec3(1.0) - F) / PI;
        vec3 contrib = (diff + spec) * NdotL * rl.color * atten;
        contrib += spKcDiff(albedo, metalness, F0, NdotV, alpha) * NdotL * rl.color * atten;
        const float lum = dot(contrib, vec3(0.2126, 0.7152, 0.0722));
        if (lum > fc) contrib *= fc / lum;
        direct += contrib;
    }

    // --- Env NEE (importance sample if CDF is bound, else uniform hemisphere) ---
    {
        vec4 envSample = spSampleEnvImportance(seed);
        vec3 Lenv;
        float pdfEnv;
        bool envValid = (envSample.w > 0.0);
        if (envValid) {
            Lenv = envSample.xyz;
            pdfEnv = envSample.w;
        } else {
            // Uniform-sphere fallback
            float u1 = urand(seed), u2 = urand(seed);
            float cosTh = 1.0 - 2.0 * u1;
            float sinTh = sqrt(max(0.0, 1.0 - cosTh*cosTh));
            float phi = TWO_PI * u2;
            Lenv = vec3(sinTh * cos(phi), cosTh, sinTh * sin(phi));
            pdfEnv = 1.0 / (4.0 * PI);
            envValid = true;
        }
        const float NdotL = dot(N, Lenv);
        if (NdotL > 0.0 && envValid) {
            const float vis = spShadowRay(shadowOrigin, Lenv, 1e20);
            if (vis > 0.0) {
                const vec3 envCol = texture(envTex, spDirToEquirectUV(Lenv)).rgb;
                const vec3 H     = normalize(V + Lenv);
                const float NdotH = max(dot(N, H), 0.0);
                const float VdotH = max(dot(V, H), 0.0);
                const float D    = spDistGGX(NdotH, roughness);
                const float G    = spGeomSmithG1(NdotV, k) * spGeomSmithG1(NdotL, k);
                const vec3  F    = spFresnelSchlick(VdotH, F0);
                const vec3  spec = D * G * F / max(4.0 * NdotV * NdotL, 1e-6);
                const vec3  diff = albedo * (1.0 - metalness) * (vec3(1.0) - F) / PI;
                // MIS weight: balance heuristic between env importance and BSDF.
                const float bsdfPdf = spBrdfPdf(V, Lenv, N, roughness, metalness);
                const float mis = pdfEnv / max(pdfEnv + bsdfPdf, 1e-6);
                vec3 contrib = (diff + spec) * NdotL * envCol * mis / max(pdfEnv, 1e-6);
                contrib += spKcDiff(albedo, metalness, F0, NdotV, alpha) * NdotL * envCol * mis / max(pdfEnv, 1e-6);
                const float lum = dot(contrib, vec3(0.2126, 0.7152, 0.0722));
                if (lum > fc) contrib *= fc / lum;
                direct += contrib;
            }
        }
    }

    // --- Emissive triangle NEE (CDF binary search, single sample) ---
    if (pc.emissiveCount > 0u && pc.emissiveTotalPower > 0.0) {
        const float xi = urand(seed) * pc.emissiveTotalPower;
        // Binary search for CDF row whose cumPower exceeds xi.
        int lo = 0, hi = int(pc.emissiveCount) - 1;
        for (int it = 0; it < 24; ++it) {
            if (lo >= hi) break;
            const int mid = (lo + hi) >> 1;
            const float cm = emissiveTris[mid].v1.w;
            if (cm < xi) lo = mid + 1; else hi = mid;
        }
        const EmTri et = emissiveTris[lo];
        // Sample a point on the triangle (uniform area).
        float u1 = urand(seed), u2 = urand(seed);
        if (u1 + u2 > 1.0) { u1 = 1.0 - u1; u2 = 1.0 - u2; }
        const vec3 lp = et.v0.xyz + u1 * (et.v1.xyz - et.v0.xyz) + u2 * (et.v2.xyz - et.v0.xyz);
        const vec3 toL = lp - ctx.worldPos;
        const float dist = length(toL);
        if (dist > 1e-4) {
            const vec3 L = toL / dist;
            const float NdotL = dot(N, L);
            if (NdotL > 0.0) {
                const vec3 triE = normalize(cross(et.v1.xyz - et.v0.xyz, et.v2.xyz - et.v0.xyz));
                const float lightCos = max(dot(triE, -L), 0.0);
                if (lightCos > 0.0) {
                    const float vis = spShadowRay(shadowOrigin, L, dist - 1e-3);
                    if (vis > 0.0) {
                        const float power = et.v2.w;
                        const float pickPdf = power / max(pc.emissiveTotalPower, 1e-10);
                        const float areaPdf = 1.0 / max(et.v0.w, 1e-10);
                        const float pdfOmega = pickPdf * areaPdf * dist * dist / max(lightCos, 1e-6);

                        const vec3 H     = normalize(V + L);
                        const float NdotH = max(dot(N, H), 0.0);
                        const float VdotH = max(dot(V, H), 0.0);
                        const float D    = spDistGGX(NdotH, roughness);
                        const float G    = spGeomSmithG1(NdotV, k) * spGeomSmithG1(NdotL, k);
                        const vec3  F    = spFresnelSchlick(VdotH, F0);
                        const vec3  spec = D * G * F / max(4.0 * NdotV * NdotL, 1e-6);
                        const vec3  diff = albedo * (1.0 - metalness) * (vec3(1.0) - F) / PI;
                        vec3 contrib = (diff + spec) * NdotL * et.emission.rgb / max(pdfOmega, 1e-6);
                        contrib += spKcDiff(albedo, metalness, F0, NdotV, alpha) * NdotL * et.emission.rgb / max(pdfOmega, 1e-6);
                        const float lum = dot(contrib, vec3(0.2126, 0.7152, 0.0722));
                        if (lum > fc) contrib *= fc / lum;
                        direct += contrib;
                    }
                }
            }
        }
    }

    // ── Bounce sampling: 2-way stochastic split (spec / diff). Clearcoat/
    // transmission lobes are out of scope for this opaque path. ────────────
    const float pSpec = mix(0.5, 0.98, metalness);
    vec3 nextDir;
    vec3 brdfWeight;
    float bsdfPdfOut;
    const float xi = urand(seed);
    if (xi < pSpec) {
        // Specular VNDF sample
        nextDir = spSampleVNDF(V, N, alpha, vec2(urand(seed), urand(seed)));
        const float NdotL = max(dot(N, nextDir), 0.0);
        if (NdotL <= 0.0) {
            // Below horizon — terminate path.
            payload.radiance      = direct;
            payload.brdfWeight    = vec3(0.0);
            payload.nextOrigin    = shadowOrigin;
            payload.nextDir       = N;
            payload.flags         = 1u;// terminate
            payload.hitWorldPos   = ctx.worldPos;
            payload.hitInstanceId = ctx.instanceId + 1u;
            payload.hitRoughness  = roughness;
            payload.hitMetalness  = metalness;
            payload.hitTransmission = 0.0;
            payload.bsdfPdf       = 0.0;
            payload.currentIor    = 1.0;
            return true;
        }
        const vec3 H = normalize(V + nextDir);
        const float NdotH = max(dot(N, H), 0.0);
        const float VdotH = max(dot(V, H), 0.0);
        const float D = spDistGGX(NdotH, roughness);
        const float G = spGeomSmithG1(NdotV, k) * spGeomSmithG1(NdotL, k);
        const vec3 F = spFresnelSchlick(VdotH, F0);
        const float specPdf = spVndfPdf(V, nextDir, N, roughness);
        const vec3 specBrdf = D * G * F / max(4.0 * NdotV * NdotL, 1e-6);
        // weight = brdf * NdotL / (pSpec * specPdf), spread across the mixture.
        brdfWeight = specBrdf * NdotL / max(pSpec * specPdf, 1e-6);
        bsdfPdfOut = pSpec * specPdf;
    } else {
        // Diffuse cosine-weighted sample
        const mat3 tbn = spMakeTBN(N);
        const vec3 dLocal = spCosineHemisphere(vec2(urand(seed), urand(seed)));
        nextDir = tbn * dLocal;
        const float NdotL = max(dot(N, nextDir), 0.0);
        const float diffPdf = NdotL / PI;
        // Kulla-Conty boost so the additive ms-comp stays unbiased.
        brdfWeight = albedo * (1.0 - metalness) * spKcBoost(F0, NdotV, alpha) / max(1.0 - pSpec, 1e-6);
        bsdfPdfOut = (1.0 - pSpec) * diffPdf;
    }

    payload.radiance        = direct;
    payload.brdfWeight      = brdfWeight;
    payload.nextOrigin      = shadowOrigin;
    payload.nextDir         = nextDir;
    payload.flags           = 0u;
    payload.hitWorldPos     = ctx.worldPos;
    payload.hitInstanceId   = ctx.instanceId + 1u;
    payload.hitRoughness    = roughness;
    payload.hitMetalness    = metalness;
    payload.hitTransmission = 0.0;
    payload.bsdfPdf         = bsdfPdfOut;
    payload.currentIor      = 1.0;
    return true;
}

#endif// THREEPP_VULKAN_SHADE_PRIMARY_GLSL
