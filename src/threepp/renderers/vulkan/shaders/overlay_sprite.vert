#version 460

// HUD sprite vertex shader. Mirrors GL's sprite_vert.glsl — the local
// quad is the standard [-0.5, 0.5] x [-0.5, 0.5] in XY (z=0), with per-
// vertex UVs at the corners. CPU pre-computes the sprite-center position
// in view space (mvPos) and extracts the world-scale of the modelMatrix
// so the shader only needs to apply the center/rotation/scale math that's
// per-vertex, then re-projects the screen-aligned quad through the ortho
// projection. Used by the ortho HUD overlay pass — TextSprite + Sprite —
// drawn after the PT image is composited onto the swapchain.

layout(location = 0) in vec3 inPos;   // local quad position, -0.5..0.5
layout(location = 1) in vec2 inUv;    // 0..1 corner UVs

layout(location = 0) out vec2 vUv;

layout(push_constant) uniform Pc {
    mat4 projection;   // 64 — ortho projection matrix (HUD camera)
    vec4 mvPos;        // 16 — sprite center in view space (xyz used)
    vec2 scale;        // 8  — world-space scale extracted from modelMatrix
    vec2 center;       // 8  — pivot offset (0..1), matches Sprite::center
    vec4 color;        // 16 — tint rgb + opacity (.a)
    float rotation;    // 4  — radians, SpriteMaterial.rotation
    float _pad0;       // 4
    float _pad1;       // 4
    float _pad2;       // 4
} pc;

void main() {
    vec2 aligned = ((inPos.xy - pc.center) + vec2(0.5)) * pc.scale;
    float s = sin(pc.rotation);
    float c = cos(pc.rotation);
    vec2 rotated = vec2(c * aligned.x - s * aligned.y,
                        s * aligned.x + c * aligned.y);
    vec4 view = vec4(pc.mvPos.xyz, 1.0);
    view.xy += rotated;
    vec4 clip = pc.projection * view;
    // threepp's projection follows GL convention: Y up in NDC, and z mapped
    // to [-1, 1]. Vulkan NDC has Y down and z clipped to [0, 1] — without
    // the z remap the HUD ortho near plane (view-space z = -near) produces
    // clip.z = -clip.w, which fails Vulkan's 0 ≤ z ≤ w clip and the whole
    // quad is culled (HUD invisible despite a successful draw call). Map
    // GL z-range [-w, w] → Vk [0, w], then flip Y to match overlay.vert.
    clip.z = (clip.z + clip.w) * 0.5;
    clip.y = -clip.y;
    gl_Position = clip;
    // No UV flip needed: threepp's Font::rasterize stores image rows
    // bottom-up (row 0 = bottom of glyph) and Sprite's GL-style UVs put
    // v=0 at the sprite's bottom — both conventions sample row 0 for the
    // sprite's bottom corner regardless of GL-vs-Vulkan framebuffer y
    // direction (texture coord space is identical between APIs). A
    // 1.0 - inUv.y flip would invert the text vertically.
    vUv = inUv;
}
