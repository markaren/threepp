// Shared primary-hit shading for the hybrid raster + PT pipeline.
//
// Used by raygen.rgen when the hybrid push-constant bit is set: instead of
// casting a primary traceRayEXT, raygen reads the raster G-buffer's hit
// info and shades the primary surface here. Bounce loop continues unchanged.
//
// Stage 1A v2: split into three entry points so spp gets cheaper —
//   primaryShadeSetup()  : material + BSDF inputs (cheap, runs once / pixel)
//   primaryShadeDirect() : NEE for env + analytic + emissive (runs once / pixel,
//                          deterministic w.r.t. its seed). Saves spp×NEE shadow
//                          rays vs. the v1 fused entry point.
//   primaryShadeBounce() : per-sample 2-way BSDF split → payload.nextDir.
//
// Stage 1A v1 scope (kept for reference; same as v2):
//   ✓ albedo/roughness/metalness lookup with bindless textures
//   ✓ Cook-Torrance specular + Lambert diffuse
//   ✓ Kulla-Conty multi-scatter compensation
//   ✓ NEE for env (importance sampled when CDF bound, uniform fallback)
//   ✓ NEE for analytic lights (dirs, points, spots, rects) with shadow rays
//   ✓ NEE for emissive triangles (CDF binary search)
//   ✓ 2-way stochastic BSDF sample (spec / diff) for next bounce
//   ✗ Iridescence / clearcoat / transmission / sheen — transmissive surfaces
//     fall back to chit's primary trace; cc/iridescence/sheen materials get
//     slightly degraded primary, secondary bounces correct.
//   ✗ Photon caustic gather (secondary bounces still gather via chit)
//
// Includer must have declared (see raygen.rgen for the pattern):
//   PI / TWO_PI; struct Payload payload; topAS; mats[]; lights; envTex;
//   albedoMaps[]; emissiveTris[]; envCdfTex/envMargTex; pc; shadowVisibility;
//   urand/pcgNext; DirLight/PointLight/SpotLight/RectLight/EmTri.

#ifndef THREEPP_VULKAN_SHADE_PRIMARY_GLSL
#define THREEPP_VULKAN_SHADE_PRIMARY_GLSL

// ── Hit context: caller fills before calling primaryShadeSetup ────────────
struct HitContext {
    vec3 worldPos;          // hit point in world space
    vec3 worldNormalGeom;   // geometric/interpolated normal (normalized)
    vec2 uv;                // material UV (zeros if mesh has no UV)
    uint instanceId;        // gl_InstanceCustomIndexEXT for the hit mesh
    vec3 rayOrigin;         // hit ray origin (camera or prev hit)
    vec3 rayDir;            // hit ray direction (toward surface, normalized)
    float hitT;             // distance from rayOrigin to worldPos
    uint flags;             // bit 0 = is_water, bit 1 = transmissive, bit 2 = thinWalled
};

// ── Cached primary surface state — produced by primaryShadeSetup, read by
// primaryShadeDirect and primaryShadeBounce. Lets raygen amortize the
// material lookup + BSDF input compute across a per-pixel direct call and
// spp per-sample bounce calls.
struct PrimaryShadeState {
    vec3  albedo;
    float roughness;
    float metalness;
    vec3  emissive;
    vec3  F0;
    vec3  N;
    vec3  V;
    float NdotV;
    float k;
    float alpha;
    vec3  worldPos;
    vec3  shadowOrigin;
    uint  instanceId;
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

// ── primaryShadeSetup ────────────────────────────────────────────────────
// Material lookup + BSDF inputs. Returns false if the material falls outside
// this opaque-PBR path (transmissive, unlit, alpha-test). Caller falls back
// to chit primary trace in that case.
bool primaryShadeSetup(HitContext ctx, out PrimaryShadeState s) {
    s.instanceId = ctx.instanceId;
    s.worldPos   = ctx.worldPos;

    const MaterialDesc m = mats[ctx.instanceId];

    // Water (DisplacedMesh, is_water flag): defer to chit unconditionally.
    // The water material may not declare transmission > 0 explicitly, but
    // chit applies the FFT cascade normal map, foam coverage, and the
    // water-specific reflect+refract split — none of which this opaque
    // path replicates.
    if ((ctx.flags & 1u) != 0u) return false;
    // Transmissive (glass): defer to chit.
    if (m.transmission > 0.05 || m.thinWalled != 0) return false;
    // Unlit MeshBasicMaterial (sentinel: roughness < 0).
    if (m.roughness < 0.0) return false;
    // Alpha-test cutout: defer to chit (raster doesn't discard yet).
    if (m.alphaCutoff > 0.0) return false;

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

    vec3 emissive = m.emissive * m.emissiveIntensity;
    if (m.emissiveTexIndex >= 0) {
        const int i = clamp(m.emissiveTexIndex, 0, int(kMaxMaterialTextures) - 1);
        emissive *= texture(albedoMaps[i], uvEmissive).rgb;
    }

    s.albedo    = albedo;
    s.roughness = roughness;
    s.metalness = metalness;
    s.emissive  = emissive;
    s.N         = ctx.worldNormalGeom;
    s.V         = normalize(-ctx.rayDir);
    s.NdotV     = max(dot(s.N, s.V), 1e-3);
    s.F0        = mix(vec3(0.04) * m.specularIntensity * m.specularColor, albedo, metalness);
    s.k         = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    s.alpha     = roughness * roughness;
    // Bias along N so shadow rays don't self-intersect the source surface.
    s.shadowOrigin = ctx.worldPos + s.N * 1e-3;
    return true;
}

// Cook-Torrance + KC ms-comp for a single light direction, with firefly cap.
// Helper used inside primaryShadeDirect — not part of the public API.
vec3 spEvalLight(PrimaryShadeState s, vec3 L, float NdotL, vec3 lightCol, float fc) {
    const vec3 H     = normalize(s.V + L);
    const float NdotH = max(dot(s.N, H), 0.0);
    const float VdotH = max(dot(s.V, H), 0.0);
    const float D    = spDistGGX(NdotH, s.roughness);
    const float G    = spGeomSmithG1(s.NdotV, s.k) * spGeomSmithG1(NdotL, s.k);
    const vec3  F    = spFresnelSchlick(VdotH, s.F0);
    const vec3  spec = D * G * F / max(4.0 * s.NdotV * NdotL, 1e-6);
    const vec3  diff = s.albedo * (1.0 - s.metalness) * (vec3(1.0) - F) / PI;
    vec3 contrib     = (diff + spec) * NdotL * lightCol;
    contrib += spKcDiff(s.albedo, s.metalness, s.F0, s.NdotV, s.alpha) * NdotL * lightCol;
    const float lum = dot(contrib, vec3(0.2126, 0.7152, 0.0722));
    if (lum > fc) contrib *= fc / lum;
    return contrib;
}

// ── primaryShadeDirect ───────────────────────────────────────────────────
// All NEE shadow rays at the primary surface — deterministic given (state,
// seed). Run ONCE per pixel per frame (independent of spp) — that's the
// optimization: spp samples share this contribution, save (spp-1)×NEE work.
vec3 primaryShadeDirect(PrimaryShadeState s, inout uint seed) {
    const float fc = pc.fireflyClamp;
    vec3 direct = s.emissive + lights.ambient * s.albedo * (1.0 - s.metalness);

    // Directional lights
    for (uint i = 0u; i < lights.dirCount; ++i) {
        const vec3  L     = -normalize(lights.dirLights[i].direction);
        const float NdotL = dot(s.N, L);
        if (NdotL <= 0.0) continue;
        if (spShadowRay(s.shadowOrigin, L, 1e20) <= 0.0) continue;
        direct += spEvalLight(s, L, NdotL, lights.dirLights[i].color, fc);
    }

    // Point lights
    for (uint i = 0u; i < lights.pointCount; ++i) {
        const PointLight pl = lights.pointLights[i];
        const vec3 toL  = pl.position - s.worldPos;
        const float dist = length(toL);
        if (dist < 1e-4) continue;
        const vec3 L = toL / dist;
        const float NdotL = dot(s.N, L);
        if (NdotL <= 0.0) continue;
        if (spShadowRay(s.shadowOrigin, L, dist - 1e-3) <= 0.0) continue;
        float atten = 1.0 / max(dist * dist, 1e-4);
        if (pl.range > 0.0) atten *= clamp(1.0 - pow(dist / pl.range, 4.0), 0.0, 1.0);
        if (pl.decay > 0.0) atten *= pow(max(dist, 1e-4), -pl.decay + 2.0);
        direct += spEvalLight(s, L, NdotL, pl.color * atten, fc);
    }

    // Spot lights
    for (uint i = 0u; i < lights.spotCount; ++i) {
        const SpotLight sl = lights.spotLights[i];
        const vec3 toL = sl.position - s.worldPos;
        const float dist = length(toL);
        if (dist < 1e-4) continue;
        const vec3 L = toL / dist;
        const float NdotL = dot(s.N, L);
        if (NdotL <= 0.0) continue;
        const float cosToLight = dot(-L, sl.direction);
        if (cosToLight < sl.cosAngleOuter) continue;
        if (spShadowRay(s.shadowOrigin, L, dist - 1e-3) <= 0.0) continue;
        float atten = 1.0 / max(dist * dist, 1e-4);
        if (sl.range > 0.0) atten *= clamp(1.0 - pow(dist / sl.range, 4.0), 0.0, 1.0);
        if (sl.decay > 0.0) atten *= pow(max(dist, 1e-4), -sl.decay + 2.0);
        atten *= smoothstep(sl.cosAngleOuter, sl.cosAngleInner, cosToLight);
        direct += spEvalLight(s, L, NdotL, sl.color * atten, fc);
    }

    // Rect lights — random point sample for v1
    for (uint i = 0u; i < lights.rectCount; ++i) {
        const RectLight rl = lights.rectLights[i];
        const vec2 ru = vec2(urand(seed), urand(seed)) * 2.0 - 1.0;
        const vec3 lp = rl.position + ru.x * rl.halfU + ru.y * rl.halfV;
        const vec3 toL = lp - s.worldPos;
        const float dist = length(toL);
        if (dist < 1e-4) continue;
        const vec3 L = toL / dist;
        const float NdotL = dot(s.N, L);
        if (NdotL <= 0.0) continue;
        const float lightCos = max(dot(rl.normal, -L), 0.0);
        if (lightCos <= 0.0) continue;
        if (spShadowRay(s.shadowOrigin, L, dist - 1e-3) <= 0.0) continue;
        const float area  = 4.0 * length(rl.halfU) * length(rl.halfV);
        const float atten = (lightCos * area) / max(dist * dist, 1e-4);
        direct += spEvalLight(s, L, NdotL, rl.color * atten, fc);
    }

    // Env NEE — importance sample if CDF bound, uniform sphere fallback
    {
        vec4 envSample = spSampleEnvImportance(seed);
        vec3 Lenv;
        float pdfEnv;
        if (envSample.w > 0.0) {
            Lenv = envSample.xyz;
            pdfEnv = envSample.w;
        } else {
            float u1 = urand(seed), u2 = urand(seed);
            float cosTh = 1.0 - 2.0 * u1;
            float sinTh = sqrt(max(0.0, 1.0 - cosTh*cosTh));
            float phi = TWO_PI * u2;
            Lenv = vec3(sinTh * cos(phi), cosTh, sinTh * sin(phi));
            pdfEnv = 1.0 / (4.0 * PI);
        }
        const float NdotL = dot(s.N, Lenv);
        if (NdotL > 0.0 && spShadowRay(s.shadowOrigin, Lenv, 1e20) > 0.0) {
            const vec3 envCol = texture(envTex, spDirToEquirectUV(Lenv)).rgb;
            const float bsdfPdf = spBrdfPdf(s.V, Lenv, s.N, s.roughness, s.metalness);
            const float mis = pdfEnv / max(pdfEnv + bsdfPdf, 1e-6);
            const vec3 col = envCol * mis / max(pdfEnv, 1e-6);
            direct += spEvalLight(s, Lenv, NdotL, col, fc);
        }
    }

    // Emissive triangle NEE — single sample via CDF binary search
    if (pc.emissiveCount > 0u && pc.emissiveTotalPower > 0.0) {
        const float xi = urand(seed) * pc.emissiveTotalPower;
        int lo = 0, hi = int(pc.emissiveCount) - 1;
        for (int it = 0; it < 24; ++it) {
            if (lo >= hi) break;
            const int mid = (lo + hi) >> 1;
            const float cm = emissiveTris[mid].v1.w;
            if (cm < xi) lo = mid + 1; else hi = mid;
        }
        const EmTri et = emissiveTris[lo];
        float u1 = urand(seed), u2 = urand(seed);
        if (u1 + u2 > 1.0) { u1 = 1.0 - u1; u2 = 1.0 - u2; }
        const vec3 lp = et.v0.xyz + u1 * (et.v1.xyz - et.v0.xyz) + u2 * (et.v2.xyz - et.v0.xyz);
        const vec3 toL = lp - s.worldPos;
        const float dist = length(toL);
        if (dist > 1e-4) {
            const vec3 L = toL / dist;
            const float NdotL = dot(s.N, L);
            if (NdotL > 0.0) {
                const vec3 triE = normalize(cross(et.v1.xyz - et.v0.xyz, et.v2.xyz - et.v0.xyz));
                const float lightCos = max(dot(triE, -L), 0.0);
                if (lightCos > 0.0 && spShadowRay(s.shadowOrigin, L, dist - 1e-3) > 0.0) {
                    const float power     = et.v2.w;
                    const float pickPdf   = power / max(pc.emissiveTotalPower, 1e-10);
                    const float areaPdf   = 1.0 / max(et.v0.w, 1e-10);
                    const float pdfOmega  = pickPdf * areaPdf * dist * dist / max(lightCos, 1e-6);
                    const vec3 col = et.emission.rgb / max(pdfOmega, 1e-6);
                    direct += spEvalLight(s, L, NdotL, col, fc);
                }
            }
        }
    }
    return direct;
}

// ── primaryShadeBounce ───────────────────────────────────────────────────
// Per-sample: pick the BSDF-sampled bounce direction, fill payload's bounce
// fields. Doesn't touch payload.radiance — caller seeds that with the
// per-pixel direct contribution (computed once via primaryShadeDirect).
void primaryShadeBounce(PrimaryShadeState s, inout uint seed, inout Payload pay) {
    pay.hitWorldPos     = s.worldPos;
    pay.hitInstanceId   = s.instanceId + 1u;
    pay.hitRoughness    = s.roughness;
    pay.hitMetalness    = s.metalness;
    pay.hitTransmission = 0.0;
    pay.currentIor      = 1.0;
    pay.flags           = 0u;
    pay.nextOrigin      = s.shadowOrigin;

    const float pSpec = mix(0.5, 0.98, s.metalness);
    const float xi    = urand(seed);
    if (xi < pSpec) {
        const vec3 nextDir = spSampleVNDF(s.V, s.N, s.alpha, vec2(urand(seed), urand(seed)));
        const float NdotL  = max(dot(s.N, nextDir), 0.0);
        if (NdotL <= 0.0) {
            pay.brdfWeight = vec3(0.0);
            pay.nextDir    = s.N;
            pay.flags      = 1u;// terminate
            pay.bsdfPdf    = 0.0;
            return;
        }
        const vec3 H = normalize(s.V + nextDir);
        const float NdotH = max(dot(s.N, H), 0.0);
        const float VdotH = max(dot(s.V, H), 0.0);
        const float D = spDistGGX(NdotH, s.roughness);
        const float G = spGeomSmithG1(s.NdotV, s.k) * spGeomSmithG1(NdotL, s.k);
        const vec3 F  = spFresnelSchlick(VdotH, s.F0);
        const float specPdf = spVndfPdf(s.V, nextDir, s.N, s.roughness);
        const vec3 specBrdf = D * G * F / max(4.0 * s.NdotV * NdotL, 1e-6);
        pay.brdfWeight = specBrdf * NdotL / max(pSpec * specPdf, 1e-6);
        pay.nextDir    = nextDir;
        pay.bsdfPdf    = pSpec * specPdf;
    } else {
        const mat3 tbn = spMakeTBN(s.N);
        const vec3 dLocal = spCosineHemisphere(vec2(urand(seed), urand(seed)));
        const vec3 nextDir = tbn * dLocal;
        const float NdotL = max(dot(s.N, nextDir), 0.0);
        const float diffPdf = NdotL / PI;
        pay.brdfWeight = s.albedo * (1.0 - s.metalness) * spKcBoost(s.F0, s.NdotV, s.alpha) / max(1.0 - pSpec, 1e-6);
        pay.nextDir    = nextDir;
        pay.bsdfPdf    = (1.0 - pSpec) * diffPdf;
    }
}

// ── Combined entry point (kept for callers that don't split spp) ─────────
// Fully equivalent to primaryShadeSetup → primaryShadeDirect → primaryShadeBounce
// in one call. raygen now uses the split form to amortize NEE across spp.
bool shadePrimaryHitOpaque(HitContext ctx, inout uint seed) {
    PrimaryShadeState s;
    if (!primaryShadeSetup(ctx, s)) return false;
    const vec3 direct = primaryShadeDirect(s, seed);
    primaryShadeBounce(s, seed, payload);
    payload.radiance = direct;
    return true;
}

#endif// THREEPP_VULKAN_SHADE_PRIMARY_GLSL
