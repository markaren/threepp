#version 460

// Fullscreen triangle for the overlay edge-AA pass (overlay_aa.frag).
// No vertex inputs — positions derived from gl_VertexIndex.

void main() {
    const vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
