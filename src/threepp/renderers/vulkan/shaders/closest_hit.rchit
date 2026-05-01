#version 460
#extension GL_EXT_ray_tracing : require

// Phase 3: report barycentric coords as RGB so each triangle vertex paints a
// primary colour (v0=red, v1=green, v2=blue) and the interior interpolates.
// `attribs` is the built-in (u, v) hit attribute for triangle hit groups;
// w = 1 - u - v is the third barycentric.
hitAttributeEXT vec2 attribs;
layout(location = 0) rayPayloadInEXT vec3 payload;

void main() {
    const float w = 1.0 - attribs.x - attribs.y;
    payload = vec3(w, attribs.x, attribs.y);
}
