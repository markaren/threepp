#version 460

// Trivial pair to overlay_depth.vert. Vulkan requires a fragment stage on
// graphics pipelines unless rasterizerDiscardEnable is true; the depth
// attachment is updated automatically from gl_Position.z, so we just need
// an empty main. No color outputs declared — the prepass renders to a
// depth-only attachment.

void main() {}
