#version 460
#extension GL_EXT_ray_tracing : require

// Phase 6b: shadow miss handler. Reached only when a shadow ray clears the
// scene without an opaque occluder, so the surface is lit by this light.
// Closest_hit traces shadow rays with TERMINATE_ON_FIRST_HIT |
// SKIP_CLOSEST_HIT, default-initializing visibility to 0.0 (occluded);
// this shader flips it to 1.0 (visible).
layout(location = 1) rayPayloadInEXT float visibility;

void main() {
    visibility = 1.0;
}
