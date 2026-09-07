#version 460

// Fullscreen-triangle vertex stage for the MSAA depth resolve (see
// gbuf_resolve_depth.frag). No vertex buffer — 3 vertices, positions
// derived from gl_VertexIndex (the standard "big triangle" trick covering
// the full viewport with no seam).

void main() {
    const vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
