#version 460

// World-space Sprite billboard vertex shader. Vulkan port of the GL/WGPU sprite
// path for sprites with screenSpace == false (3D-positioned camera-facing quads —
// e.g. the TPS shooter's impact "particles"). The screen-space HUD sprites go
// through overlay_sprite.vert (ortho); this is the perspective, depth-tested
// counterpart drawn in the post-TAA overlay block.
//
// Same push-constant layout as overlay_sprite (SpritePC) so the fragment shader
// (overlay_sprite.frag) is shared. The ONLY difference from overlay_sprite.vert:
// the projection here is the perspective reverse-Z matrix, which already maps z
// to Vulkan's [0,w] — so we DROP the GL→Vulkan z-remap (that remap is only
// correct for the GL-convention ortho HUD projection).

layout(location = 0) in vec3 inPos;   // local quad position, -0.5..0.5
layout(location = 1) in vec2 inUv;    // 0..1 corner UVs

layout(location = 0) out vec2 vUv;

layout(push_constant) uniform Pc {
    mat4 projection;   // 64 — perspective reverse-Z projection (unjittered)
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
    // Camera-facing: offset the view-space center in view XY, then project. The
    // offset is in view (≈ world) units, so the sprite keeps a constant world
    // size and shrinks with distance under perspective (sizeAttenuation).
    vec4 view = vec4(pc.mvPos.xyz, 1.0);
    view.xy += rotated;
    vec4 clip = pc.projection * view;
    // threepp projection follows the GL NDC convention (Y up); Vulkan NDC is Y
    // down, so flip Y at the clip boundary (same as overlay.vert / particle.vert).
    // The reverse-Z projection already maps z to [0,w], so — unlike the ortho
    // overlay_sprite path — NO z-remap is applied.
    clip.y = -clip.y;
    gl_Position = clip;
    vUv = inUv;
}
