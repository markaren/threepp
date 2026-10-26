// The blur velocity both motion-blur passes read: the tile max
// (motion_tilemax.comp) finds each tile's dominant one and the reconstruction
// (motion_blur.comp) gathers along it, so the two must clamp to the same tile
// radius. The includer declares gbufMotionTex and a push block `pc` carrying
// width, height, shutter and skyReproj.

#ifndef MOTION_BLUR_VELOCITY_GLSL
#define MOTION_BLUR_VELOCITY_GLSL

// Tile size = max blur radius in px.
const float kTilePx = 32.0;

// Blur velocity (curr − prev) at `uv`, in OUTPUT pixels, shutter-scaled and
// clamped to the tile radius. Motion is stored as prevNDC − currNDC (GL NDC,
// Y-up); the pixel-space conversion folds in the negation and the Y flip.
vec2 blurVelocityPx(vec2 uv) {
    vec2 motionGL = texture(gbufMotionTex, uv).rg;
    const vec2 currNDC = vec2(uv.x * 2.0 - 1.0, -(uv.y * 2.0 - 1.0));
    if (all(equal(motionGL, vec2(0.0)))) {
        const vec4 pclip = pc.skyReproj * vec4(currNDC, 0.0, 1.0);// reversed-Z far
        if (abs(pclip.w) > 1e-6) motionGL = pclip.xy / pclip.w - currNDC;
    }
    vec2 vPx = vec2(-motionGL.x, motionGL.y) * 0.5
             * vec2(float(pc.width), float(pc.height)) * pc.shutter;
    const float len = length(vPx);
    if (len > kTilePx) vPx *= kTilePx / len;
    return vPx;
}

#endif// MOTION_BLUR_VELOCITY_GLSL
