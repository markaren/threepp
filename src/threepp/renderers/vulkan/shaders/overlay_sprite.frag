#version 460

// HUD sprite fragment shader. Pair to overlay_sprite.vert. Samples the
// per-sprite atlas texture (e.g. TextSprite's rasterized glyph bitmap)
// and modulates by the push-constant tint+opacity. Alpha-blending is set
// by the pipeline's color-blend state (standard non-premultiplied alpha).
//
// The atlas + tint are LINEAR (TextSprite bakes color.r*255 linear bytes;
// non-sRGB atlases are sampled raw). The swapchain is B8G8R8A8_UNORM with
// no hardware sRGB write-out, and the path-traced image is encoded by
// denoise.comp, so we apply the same linear→sRGB OETF here. Without it,
// non-white sprites/text render darker than the rest of the frame.

layout(set = 0, binding = 0) uniform sampler2D spriteMap;

layout(location = 0) in vec2 vUv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Pc {
    mat4 projection;
    vec4 mvPos;
    vec2 scale;
    vec2 center;
    vec4 color;
    float rotation;
    float _pad0;
    float _pad1;
    float _pad2;
} pc;

vec3 linearToSRGB(vec3 x) {
    const vec3 cutoff = vec3(lessThan(x, vec3(0.0031308)));
    const vec3 lower  = 12.92 * x;
    const vec3 higher = 1.055 * pow(max(x, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(higher, lower, cutoff);
}

void main() {
    vec4 t = texture(spriteMap, vUv);
    outColor = vec4(linearToSRGB(pc.color.rgb * t.rgb), pc.color.a * t.a);
}
