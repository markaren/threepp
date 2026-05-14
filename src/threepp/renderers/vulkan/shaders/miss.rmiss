#version 460
#extension GL_EXT_ray_tracing : require

// Phase 7: primary miss samples the scene environment (equirect HDR) so
// rays that escape geometry pick up the sky / IBL background. If no env
// texture is bound, a 1×1 black dummy is bound by the host so the sample
// returns zero — same as the Phase 2 fallback.
//
// Phase 9: payload is now a struct shared with raygen + closest_hit. We
// write the env radiance and set bit 0 of `flags` so raygen terminates the
// path (no further bounce ray is launched).

struct Payload {
    vec3 radianceDiff;
    vec3 radianceSpec;
    vec3 brdfWeight;
    vec3 nextOrigin;
    vec3 nextDir;
    uint flags;
    uint seed;
    vec3 hitWorldPos;  // unused by miss (kept for layout match with raygen/closest_hit)
    uint hitInstanceId;// must be 0 on miss so raygen sees sky/background as no-reproject
    vec3 prevWorldPos; // unused by miss (kept for layout match)
    float hitRoughness;// unused by miss (sky cold-starts on cam motion; cap doesn't apply)
    uint inFlags;      // unused by miss (kept for layout match)
    float hitMetalness;// unused by miss (kept for layout match)
    float hitTransmission;// unused by miss (kept for layout match)
    float bsdfPdf;     // pdf of the BSDF-sampled bounce direction (set by chit; used here for MIS)
    float currentIor;  // unused by miss (kept for layout match)
    float hitSpecFrac; // unused by miss (kept for layout match — sky has no surface)
    vec4 primaryAlbedo;// unused by miss (kept for layout match — sky has no surface)
};

layout(set = 0, binding = 6) uniform sampler2D envTex;
// Env luminance CDF — bound 1×1 dummy with envCdfTotalSum=0 when env is
// solid color or default. Used to compute pdf_env(rayDir) for the MIS
// complement to chit's importance-sampled env NEE.
layout(set = 0, binding = 18) uniform sampler2D envCdfTex;
layout(set = 0, binding = 19) uniform sampler2D envMargTex;

// Push-constant block must match raygen / closest_hit layout. Miss only reads
// envCdfWidth/Height/TotalSum.
layout(push_constant) uniform Pc {
    uint sampleIndex;
    uint envMipCount;
    uint _pad1;
    uint _pad2;
    uint motionFlags;
    uint emissiveCount;
    float emissiveTotalPower;
    uint _padSpp;
    uint envCdfWidth;
    uint envCdfHeight;
    float envCdfTotalSum;
    float fireflyClamp;     // unused in miss; declared for layout parity with primary RT
    float oceanFineTileSize;// unused in miss; declared for layout parity
} pc;

layout(location = 0) rayPayloadInEXT Payload payload;

const float PI = 3.14159265358979;
const float TWO_PI = 6.28318530717958;

// GL / three.js equirect convention (matches WgpuPathTracerShaders_Rt.cpp):
// u = 0.5 + atan2(z, x) / (2π); v = 0.5 + asin(y) / π. Using acos here gave
// a vertically-mirrored sky.
vec3 sampleEquirect(vec3 dir) {
    const float u = 0.5 + atan(dir.z, dir.x) / TWO_PI;
    const float v = 0.5 + asin(clamp(dir.y, -1.0, 1.0)) / PI;
    return texture(envTex, vec2(u, v)).rgb;
}

vec2 dirToEquirectUV(vec3 dir) {
    return vec2(0.5 + atan(dir.z, dir.x) / TWO_PI,
                0.5 + asin(clamp(dir.y, -1.0, 1.0)) / PI);
}

// Mirror of closest_hit.rchit's envImportancePdf — same formula, but reading
// the env CDF push-constant data from miss's own Pc block. Returns 0 when
// CDF disabled (totalSum<=0); chit then falls back to constant 0.5 MIS.
float envImportancePdf(vec3 dir) {
    if (pc.envCdfTotalSum <= 0.0) return 0.0;
    const int W = int(pc.envCdfWidth);
    const int H = int(pc.envCdfHeight);
    if (W <= 0 || H <= 0) return 0.0;
    const vec2 uv = dirToEquirectUV(normalize(dir));
    const int col = clamp(int(uv.x * float(W)), 0, W - 1);
    const int row = clamp(int(uv.y * float(H)), 0, H - 1);
    const vec3 envCol = texelFetch(envTex, ivec2(col, row), 0).rgb;
    const float lum   = dot(envCol, vec3(0.2126, 0.7152, 0.0722)) + 1e-10;
    return lum * float(W * H) / (2.0 * PI * PI * max(pc.envCdfTotalSum, 1e-10));
}

void main() {
    const vec3 dir  = normalize(gl_WorldRayDirectionEXT);
    const vec3 envR = sampleEquirect(dir);
    // Primary miss (bit 1 unset) returns the env at full strength so the
    // background is visible. Non-primary miss (bit 1 set by raygen) is the
    // BSDF half of an MIS pair against chit's env NEE.
    //   • envCdfTotalSum > 0 → chit used env importance sampling; MIS weight
    //     here is pdf_brdf / (pdf_brdf + pdf_env). pdf_brdf is what chit
    //     wrote into payload.bsdfPdf when sampling the bounce.
    //   • envCdfTotalSum <= 0 → chit used BSDF-sampled env NEE; pdfs match
    //     so the balance-heuristic weight collapses to 0.5.
    // Phase 1: route all env miss radiance to diff channel for now.
    // Phase 1b will route bounce-escape env to the spec channel when
    // payload.flags bit indicates primary lobe was spec.
    vec3 envContrib;
    if ((payload.flags & 2u) != 0u) {
        if (pc.envCdfTotalSum > 0.0) {
            const float pdfEnv  = envImportancePdf(dir);
            const float pdfBrdf = payload.bsdfPdf;
            const float wBrdf   = pdfBrdf / max(pdfBrdf + pdfEnv, 1e-8);
            envContrib = wBrdf * envR;
        } else {
            envContrib = 0.5 * envR;
        }
    } else {
        envContrib = envR;
    }
    payload.radianceDiff = envContrib;
    payload.radianceSpec = vec3(0.0);
    payload.flags |= 1u;// terminate path — no scatter beyond the env
}
