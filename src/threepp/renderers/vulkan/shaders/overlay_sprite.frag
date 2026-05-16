#version 460

// HUD sprite fragment shader. Pair to overlay_sprite.vert. Samples the
// per-sprite atlas texture (e.g. TextSprite's rasterized glyph bitmap)
// and modulates by the push-constant tint+opacity. Alpha-blending is set
// by the pipeline's color-blend state (standard non-premultiplied alpha).

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

void main() {
    vec4 t = texture(spriteMap, vUv);
    outColor = vec4(pc.color.rgb * t.rgb, pc.color.a * t.a);
}
