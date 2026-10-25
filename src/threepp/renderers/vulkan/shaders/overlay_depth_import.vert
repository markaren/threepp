#version 460

// Fullscreen triangle for the overlay depth IMPORT (overlay_depth_import.frag):
// the big-triangle trick of overlay_fullscreen.vert, plus the texture
// coordinate the fragment stage samples the G-buffer depth with. NDC (-1,-1)
// is the top-left of the viewport under Vulkan's y-down raster, and row 0 of
// the G-buffer is its top row too, so uv = ndc * 0.5 + 0.5 with no flip.

layout(location = 0) out vec2 vUv;

void main() {
    const vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
    vUv         = pos;
}
