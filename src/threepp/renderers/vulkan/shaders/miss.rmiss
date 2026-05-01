#version 460
#extension GL_EXT_ray_tracing : require

// Phase 7: primary miss samples the scene environment (equirect HDR) so
// rays that escape geometry pick up the sky / IBL background. If no env
// texture is bound, a 1×1 black dummy is bound by the host so the sample
// returns zero — same as the Phase 2 fallback.
layout(set = 0, binding = 6) uniform sampler2D envTex;

layout(location = 0) rayPayloadInEXT vec3 payload;

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
    payload = sampleEquirect(normalize(gl_WorldRayDirectionEXT));
}
