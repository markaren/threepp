#version 460

// Lit split-screen pane sky — fullscreen-triangle vertex stage. All the work
// happens in the fragment shader (per-pixel view ray → equirect sample); this
// just covers the pane's scissor rect. z = 0 with depth test off, so the sky
// never contends with the pane's meshes.

void main() {
    const vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
