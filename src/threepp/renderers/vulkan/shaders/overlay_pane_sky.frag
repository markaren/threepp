#version 460

// Lit split-screen pane sky. Samples the SAME equirect the deferred primary
// miss shows (Impl::envImage), along this pixel's view ray through the pane
// camera — so a preview pane's background is the scene's sky, not a flat
// clear. Only drawn when the scene's environment is a real texture: a solid
// background colour stays a verbatim loadOp clear, because the deferred
// path's solid-bg bypass restores that colour display-referred and the two
// views must agree on what "empty" looks like.
//
// Same output conventions as overlay_mesh_lit.frag: exposure scale, ACES
// fit, then the linear→sRGB OETF (UNORM swapchain, display-referred values).
// The deferred sky goes through the full post chain (pre-exposure,
// auto-exposure, composite tone map), so under a driven exposure the pane
// sky can sit a stop off the main view — a preview, not a calibration.

layout(push_constant) uniform Pc {
    mat4 invVP; // inverse(proj · view), GL clip conventions
    vec4 rect;  // pane rect in framebuffer pixels: x, y, w, h (top-left origin)
    vec4 params;// .x = exposure (toneMappingExposure), .yzw unused
} pc;

layout(set = 0, binding = 0) uniform sampler2D envTex;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outMask;// unused (single color attachment) — layout parity

const float PI     = 3.14159265358979;
const float TWO_PI = 6.28318530717959;

vec3 linearToSRGB(vec3 x) {
    const vec3 cutoff = vec3(lessThan(x, vec3(0.0031308)));
    const vec3 lower  = 12.92 * x;
    const vec3 higher = 1.055 * pow(max(x, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(higher, lower, cutoff);
}

vec3 acesFilm(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    // Pane-relative GL-style NDC (y up). gl_FragCoord is top-left origin,
    // which is also the rect's convention, so only y flips.
    vec2 ndc;
    ndc.x = ((gl_FragCoord.x - pc.rect.x) / pc.rect.z) * 2.0 - 1.0;
    ndc.y = -(((gl_FragCoord.y - pc.rect.y) / pc.rect.w) * 2.0 - 1.0);

    // View ray: unproject the near and far plane points and subtract. Works
    // for perspective and parallel pane cameras alike (under ortho the two
    // points differ by the constant view direction).
    const vec4 p0 = pc.invVP * vec4(ndc, -1.0, 1.0);
    const vec4 p1 = pc.invVP * vec4(ndc, 1.0, 1.0);
    const vec3 dir = normalize(p1.xyz / p1.w - p0.xyz / p0.w);

    // Equirect lookup — MUST match sampleEnvLod in the deferred shade
    // (Y-up convention) or the pane's sky rotates against the main view's.
    const float u = 0.5 + atan(dir.z, dir.x) / TWO_PI;
    const float v = 0.5 + asin(clamp(dir.y, -1.0, 1.0)) / PI;

    vec3 c = textureLod(envTex, vec2(u, v), 0.0).rgb * pc.params.x;
    c = acesFilm(c);
    outColor = vec4(linearToSRGB(c), 1.0);
    outMask  = vec4(1.0);
}
