#version 460
#extension GL_EXT_ray_tracing : require

// Shadow miss handler. shadowVisibility is now initialised to 1.0 by the
// caller and multiplied down by shadow_anyhit.rahit for glass/cutout surfaces.
// Miss means no opaque blocker was hit so the accumulated transmittance is
// already correct — nothing to do here.
layout(location = 1) rayPayloadInEXT float visibility;

void main() {
    // no-op: caller reads the accumulated shadowVisibility as-is.
}
