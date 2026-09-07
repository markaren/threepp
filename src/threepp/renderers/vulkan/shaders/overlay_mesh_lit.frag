#version 460

// Pair to overlay_mesh_lit.vert — Lambert shading for the lit split-screen
// pane. Deliberately a PREVIEW, not the deferred pipeline: one directional
// light plus a flat ambient, no shadows, no GI, no fog. What it must get
// right is being recognizably the same scene under the same sun — a face
// toward the light reads brighter than a face away from it, which is exactly
// the claim the flat overlay fill could not make.
//
// Output conventions match overlay.frag: the swapchain is UNORM (no hardware
// sRGB write-out) and holds display-referred values, so tone-map (the ACES
// fit, matching the editor's ToneMapping::ACESFilmic default) then apply the
// linear→sRGB OETF here. Without the tone map a 2.5-intensity sun clips the
// lit side to pure white and the preview reads overexposed next to the
// deferred main view.

layout(push_constant) uniform Pc {
    mat4 mvp;
    vec4 nrm0;
    vec4 nrm1;
    vec4 nrm2;
    vec4 color;
} pc;

// Constant across the pane — extracted from the scene once per record().
layout(set = 0, binding = 0) uniform PaneLightUbo {
    vec4 sunDir;  // xyz = world direction TOWARD the light, w = intensity
    vec4 sunColor;// rgb (w unused)
    vec4 ambient; // rgb = Σ AmbientLight color·intensity (w unused)
} lit;

layout(location = 0) in vec3 vNormal;

layout(location = 0) out vec4 outColor;

vec3 linearToSRGB(vec3 x) {
    const vec3 cutoff = vec3(lessThan(x, vec3(0.0031308)));
    const vec3 lower  = 12.92 * x;
    const vec3 higher = 1.055 * pow(max(x, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(higher, lower, cutoff);
}

// Narkowicz ACES fit — close enough to the main pipeline's filmic curve for a
// preview pane, and a single fma chain rather than the full RRT/ODT.
vec3 acesFilm(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    vec3 N = normalize(vNormal);
    // CULL_NONE pipeline: a back face (the inside of a box, the underside of a
    // double-sided ground plane) shades with the flipped normal, matching how
    // the raster G-buffer orients normals viewer-ward.
    if (!gl_FrontFacing) N = -N;
    const float ndl = max(dot(N, normalize(lit.sunDir.xyz)), 0.0);
    vec3 c = pc.color.rgb * (lit.ambient.rgb + lit.sunColor.rgb * lit.sunDir.w * ndl);
    c = acesFilm(c);
    outColor = vec4(linearToSRGB(c), pc.color.a);
}
