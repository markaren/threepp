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

// Same linear->sRGB OETF as overlay.frag: the swapchain is UNORM and holds
// display-referred data, so writing the linear product raw would render the
// cloud darker than the identical PointsMaterial on the GL backend.
vec3 linearToSRGB(vec3 x) {
    const vec3 cutoff = vec3(lessThan(x, vec3(0.0031308)));
    const vec3 lower  = 12.92 * x;
    const vec3 higher = 1.055 * pow(max(x, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(higher, lower, cutoff);
}

void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    outColor = vec4(linearToSRGB(pc.color.rgb * vColor), 1.0);
}
