#version 460

// Particle billboard fragment shader. Mirrors the GL ParticleSystem fragment
// shader (modulate per-particle vertex color × particle texture) and the HUD
// overlay_sprite.frag color convention.
//
// The vertex color (HSL→RGB) and the texture are LINEAR (an sRGB-tagged texture
// is hardware-decoded to linear on sample). The swapchain is B8G8R8A8_UNORM with
// no hardware sRGB write-out and the path-traced image is sRGB-encoded by
// denoise.comp, so we apply the same linear→sRGB OETF here. Without it particles
// composite darker/more-saturated than the rest of the frame. Blending (alpha vs
// additive) is set by the pipeline variant, not here. Untextured particle systems
// bind a 1×1 white default so the sampler is always valid.

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) in vec4 vColor;
layout(location = 1) in vec2 vUv;

layout(location = 0) out vec4 outColor;

vec3 linearToSRGB(vec3 x) {
    const vec3 cutoff = vec3(lessThan(x, vec3(0.0031308)));
    const vec3 lower  = 12.92 * x;
    const vec3 higher = 1.055 * pow(max(x, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(higher, lower, cutoff);
}

void main() {
    vec4 t = texture(tex, vUv);
    outColor = vec4(linearToSRGB(vColor.rgb * t.rgb), vColor.a * t.a);
}
