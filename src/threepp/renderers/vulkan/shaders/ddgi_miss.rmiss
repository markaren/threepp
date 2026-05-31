#version 460
#extension GL_EXT_ray_tracing : require

// DDGI probe-ray miss — the ray escaped the scene, so the probe "sees" the
// environment in that direction. Writes the equirect env radiance and a
// negative distance (the miss sentinel the blend reads for visibility later).
// GL/three.js equirect convention, matching miss.rmiss / shade_common.

layout(set = 0, binding = 3) uniform sampler2D envTex;
layout(location = 0) rayPayloadInEXT vec4 prd;

const float PI     = 3.14159265358979;
const float TWO_PI = 6.28318530717958;

void main() {
    const vec3  d = normalize(gl_WorldRayDirectionEXT);
    const float u = 0.5 + atan(d.z, d.x) / TWO_PI;
    const float v = 0.5 + asin(clamp(d.y, -1.0, 1.0)) / PI;
    prd = vec4(texture(envTex, vec2(u, v)).rgb, -1.0);
}
