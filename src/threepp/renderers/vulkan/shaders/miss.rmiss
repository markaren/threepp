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
    vec3 radiance;
    vec3 brdfWeight;
    vec3 nextOrigin;
    vec3 nextDir;
    uint flags;
    uint seed;
    vec3 hitWorldPos;  // unused by miss (kept for layout match with raygen/closest_hit)
    uint hitInstanceId;// must be 0 on miss so raygen sees sky/background as no-reproject
    float hitRoughness;// unused by miss (sky cold-starts on cam motion; cap doesn't apply)
};

layout(set = 0, binding = 6) uniform sampler2D envTex;

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

void main() {
    // Primary miss (bit 1 unset) returns the env at full strength so the
    // background is visible. Non-primary miss (bit 1 set by raygen) is the
    // BSDF half of an MIS pair: closest_hit's env NEE shot a shadow ray with
    // BSDF-importance sampling at the previous shade point, so we'd double-
    // count if we returned the full env here. With the same sampling
    // distribution on both sides, the balance-heuristic weight is a constant
    // 0.5 (see the closest_hit Env NEE + MIS comment). Apply it here.
    const vec3 envR = sampleEquirect(normalize(gl_WorldRayDirectionEXT));
    if ((payload.flags & 2u) != 0u) {
        payload.radiance = 0.5 * envR;
    } else {
        payload.radiance = envR;
    }
    payload.flags |= 1u;// terminate path — no scatter beyond the env
}
