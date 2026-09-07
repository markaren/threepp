
#include "threepp/postprocessing/MaskPass.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/renderers/GLRenderer.hpp"

using namespace threepp;


MaskPass::MaskPass(Object3D& scene, Camera& camera)
    : scene_(&scene), camera_(&camera) {

    clear = true;
    needsSwap = false;
}

void MaskPass::render(GLRenderer& renderer, RenderTarget* writeBuffer, RenderTarget* readBuffer, float, bool) {

    auto& state = renderer.state();

    // The mask scene contributes a silhouette, not pixels: colour and depth
    // are masked off and locked so the material state the renderer sets per
    // draw cannot re-enable them.
    state.colorBuffer.setMask(false);
    state.depthBuffer.setMask(false);

    state.colorBuffer.setLocked(true);
    state.depthBuffer.setLocked(true);

    const int writeValue = inverse ? 0 : 1;
    const int clearValue = inverse ? 1 : 0;

    state.stencilBuffer.setTest(true);
    state.stencilBuffer.setOp(StencilOp::Replace, StencilOp::Replace, StencilOp::Replace);
    state.stencilBuffer.setFunc(StencilFunc::Always, writeValue, 0xffffffff);
    state.stencilBuffer.setClear(clearValue);
    state.stencilBuffer.setLocked(true);

    // Both buffers get the mask: the chain ping-pongs between them, and a pass
    // downstream may read either.
    renderer.setRenderTarget(readBuffer);
    if (clear) renderer.clear();
    renderer.render(*scene_, *camera_);

    renderer.setRenderTarget(writeBuffer);
    if (clear) renderer.clear();
    renderer.render(*scene_, *camera_);

    state.colorBuffer.setLocked(false);
    state.depthBuffer.setLocked(false);

    // Hand the stencil over to the passes that follow: draw only where the
    // mask was written.
    state.stencilBuffer.setLocked(false);
    state.stencilBuffer.setFunc(StencilFunc::Equal, 1, 0xffffffff);
    state.stencilBuffer.setOp(StencilOp::Keep, StencilOp::Keep, StencilOp::Keep);
    state.stencilBuffer.setLocked(true);
}

ClearMaskPass::ClearMaskPass() {

    needsSwap = false;
}

void ClearMaskPass::render(GLRenderer& renderer, RenderTarget*, RenderTarget*, float, bool) {

    auto& state = renderer.state();

    state.stencilBuffer.setLocked(false);
    state.stencilBuffer.setTest(false);
}
