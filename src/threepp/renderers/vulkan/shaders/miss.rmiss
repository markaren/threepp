#version 460
#extension GL_EXT_ray_tracing : require

// Phase 2: empty-TLAS fallback. Black means "ray hit nothing".
layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    payload = vec3(0.0, 0.0, 0.0);
}
