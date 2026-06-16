#version 460

// World-space particle billboard vertex shader. Vulkan port of the GL/WGPU
// ParticleSystem shader (src/threepp/objects/ParticleSystem.cpp): the renderer
// has no generic ShaderMaterial path, so the particle Mesh is drawn here by a
// dedicated billboard pass in the post-TAA overlay block rather than the
// PT/G-buffer path.
//
// The geometry stores 4 coincident verts per particle (all at the particle
// CENTER); the quad is expanded HERE from the per-vertex data:
//   inPos    = particle center (model space)
//   inNormal = {size, angle, opacity}
//   inUv     = quad corner offset, -0.5..0.5
//   inColor  = particle RGB
//
// CPU pushes modelView (= viewUnjittered · meshWorld) and the reverse-Z
// projection separately because the distance-attenuated billboard scale needs
// view-space depth and proj[1][1] individually (it can't use a combined MVP).

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;
layout(location = 3) in vec3 inColor;

layout(location = 0) out vec4 vColor;
layout(location = 1) out vec2 vUv;

layout(push_constant) uniform Pc {
    mat4 modelView;   // 64 — viewUnjittered · meshWorld
    mat4 proj;        // 64 — reverse-Z Vulkan projection (unjittered)
} pc;

void main() {
    float pSize    = inNormal.x;
    float pAngle   = inNormal.y;
    float pOpacity = inNormal.z;

    vColor = vec4(inColor, pOpacity);

    // Rotate the corner offset by the particle angle (screen-aligned).
    float c = cos(pAngle);
    float s = sin(pAngle);
    vec2 rotated = vec2(c * inUv.x - s * inUv.y,
                        s * inUv.x + c * inUv.y);

    vec4 mvPosition = pc.modelView * vec4(inPos, 1.0);
    vec4 clipPos    = pc.proj * mvPosition;

    // proj[1][1] is the Y scale (≈ 1/tan(fov/2)); reverse-Z only rewrites the
    // depth row, so this matches the GL billboard scale term exactly. Multiply
    // by clipPos.w to offset in clip space (perspective-correct billboard size).
    float scale = pSize * pc.proj[1][1] / length(mvPosition.xyz);
    clipPos.xy += rotated * scale * clipPos.w;

    // threepp projection follows the GL NDC convention (Y up); Vulkan NDC is Y
    // down, so flip Y at the clip boundary — same correction overlay.vert
    // applies. The reverse-Z projection already maps z to [0,1], so no z-remap
    // is needed (unlike the ortho overlay_sprite path).
    clipPos.y = -clipPos.y;
    gl_Position = clipPos;

    // Texture UV derived from the corner offset (matches GL: uv + 0.5).
    vUv = inUv + vec2(0.5);
}
