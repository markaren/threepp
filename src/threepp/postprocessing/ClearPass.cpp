
#include "threepp/postprocessing/ClearPass.hpp"

#include "threepp/renderers/GLRenderer.hpp"

using namespace threepp;


ClearPass::ClearPass() {

    needsSwap = false;
}

ClearPass::ClearPass(const Color& clearColor, float clearAlpha)
    : clearColor(clearColor), clearAlpha(clearAlpha) {

    needsSwap = false;
}

void ClearPass::render(GLRenderer& renderer, RenderTarget*, RenderTarget* readBuffer, float, bool) {

    Color oldClearColor;
    float oldClearAlpha = 1.f;

    if (clearColor) {

        renderer.getClearColor(oldClearColor);
        oldClearAlpha = renderer.getClearAlpha();
        renderer.setClearColor(*clearColor, clearAlpha);
    }

    renderer.setRenderTarget(renderToScreen ? nullptr : readBuffer);
    renderer.clear();

    if (clearColor) renderer.setClearColor(oldClearColor, oldClearAlpha);
}
