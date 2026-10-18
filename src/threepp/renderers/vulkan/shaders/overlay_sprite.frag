#version 460
#extension GL_GOOGLE_include_directive : enable

// HUD sprite fragment shader. Pair to overlay_sprite.vert. Samples the
// per-sprite atlas texture (e.g. TextSprite's rasterized glyph bitmap)
// and modulates by the push-constant tint+opacity. Alpha-blending is set
// by the pipeline's color-blend state (standard non-premultiplied alpha).
//
// The atlas + tint are LINEAR (TextSprite bakes color.r*255 linear bytes;
// non-sRGB atlases are sampled raw). The swapchain is B8G8R8A8_UNORM with
// no hardware sRGB write-out, and the rendered image is encoded by
// post_composite.comp, so we apply the same linear→sRGB OETF here. Without it,
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

#include "overlay_display.glsl"

void main() {
    vec4 t = texture(spriteMap, vUv);
    outColor = vec4(odLinearToSRGB(pc.color.rgb * t.rgb), pc.color.a * t.a);
}
