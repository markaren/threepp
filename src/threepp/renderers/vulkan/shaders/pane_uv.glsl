// ── Pane-UV → physical-UV rescale (split-screen) ────────────────────────────
// Under a scissored pane render (setScissorTest + a sub-rect viewport) the
// deferred pipeline renders PANE-SIZED content at the ORIGIN of the same
// full-render-extent images used fullscreen. Every screen-space normalized
// coordinate in this shader family ([0,1] over pc.width/height) must therefore
// be rescaled by pane/physical before it touches a sampler, or the tap lands
// at a stretched position — the visible symptom was shadow/GI history ghosts
// offset from their casters in lidar_slam's split view. Fullscreen the ratio
// is exactly 1.0 (w/w), so this is a no-op there. Applies ONLY to samplers
// backed by render-extent images (gbuf*, *Prev, history); env/noise/froxel/
// cloud-shadow textures live in their own spaces and take their uv raw.
//
// Needs the ShadePush block declared as `pc` before the include (the deferred
// shade family, rtao.comp and cloud_march.comp bind it under that name).
vec2 paneToPhys(sampler2D t, vec2 paneUv) {
    return paneUv * vec2(float(pc.width), float(pc.height)) / vec2(textureSize(t, 0));
}

// The same for the HALF-res images (rtao, rtaoAux, cloud, fog half). Their
// passes run one thread per half-PANE pixel, so the valid content spans the
// pane's half extent at the image origin, while textureSize reports the half
// of the PHYSICAL extent. Sizing a half-res lookup by textureSize under a pane
// stretched it 2x across a half-width pane: lidar_slam's left pane read its AO
// from twice its own x, and past the pane's centre from texels nothing wrote
// (ao 0), which drew black screen-aligned bands and rectangles across the floor.
// The ratio is formed first so it is exactly 1.0 fullscreen.
ivec2 paneHalfSize() {
    return ivec2(int(pc.width + 1u) / 2, int(pc.height + 1u) / 2);
}
vec2 paneHalfToPhys(sampler2D t, vec2 paneUv) {
    return paneUv * (vec2(paneHalfSize()) / vec2(textureSize(t, 0)));
}
