#version 460

// Pair to overlay_point.vert. Modulates the material color tint
// (pc.color.rgb) by the per-vertex color, with full alpha — the
// push constant's .w slot encodes point size for this pipeline, not
// opacity, so blend alpha is hard-coded to 1.0.
//
// gl_PointCoord is in [0,1]² across the sprite; we use it to discard
// fragments outside a unit-radius disk, giving round LIDAR-style dots
// instead of square sprites. Costs one mul + one discard per fragment;
// the GPU optimises out the discarded fragments early.

layout(location = 0) in vec3 vColor;

layout(push_constant) uniform Pc {
    mat4 mvp;
    vec4 color;
} pc;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    outColor = vec4(pc.color.rgb * vColor, 1.0);
}
