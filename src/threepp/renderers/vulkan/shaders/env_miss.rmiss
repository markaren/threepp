#version 460
#extension GL_EXT_ray_tracing : require

// Phase 7: env probe miss handler. Closest-hit traces a mirror reflection
// ray with miss index = 2 to fetch the environment color for the spec IBL
// term. Same equirect sample as the primary miss but writes the dedicated
// `envProbe` payload at location 2 so the closest-hit's primary radiance
// (location 0) is not stomped.
layout(set = 0, binding = 6) uniform sampler2D envTex;

layout(location = 2) rayPayloadInEXT vec3 envProbe;

const float PI = 3.14159265358979;
const float TWO_PI = 6.28318530717958;

// GL / three.js equirect convention; see miss.rmiss for the full note.
vec3 sampleEquirect(vec3 dir) {
    const float u = 0.5 + atan(dir.z, dir.x) / TWO_PI;
    const float v = 0.5 + asin(clamp(dir.y, -1.0, 1.0)) / PI;
    return texture(envTex, vec2(u, v)).rgb;
}

void main() {
    envProbe = sampleEquirect(normalize(gl_WorldRayDirectionEXT));
}
