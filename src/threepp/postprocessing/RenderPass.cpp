
#include "threepp/postprocessing/RenderPass.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/scenes/Scene.hpp"

using namespace threepp;


RenderPass::RenderPass(Object3D& scene, Camera& camera)
    : scene_(&scene), camera_(&camera) {

    clear = true;
    needsSwap = false;
}

void RenderPass::render(GLRenderer& renderer, RenderTarget* writeBuffer, RenderTarget* readBuffer, float, bool) {

    // The pass decides when to clear; renderer.render() must not also do it
    // behind our back on the way in.
    const bool oldAutoClear = renderer.autoClear;
    renderer.autoClear = false;

    std::shared_ptr<Material> oldOverrideMaterial;
    auto* asScene = scene_->as<Scene>();

    if (overrideMaterial && asScene) {

        oldOverrideMaterial = asScene->overrideMaterial;
        asScene->overrideMaterial = overrideMaterial;
    }

    Color oldClearColor;
    float oldClearAlpha = 1.f;

    if (clearColor) {

        renderer.getClearColor(oldClearColor);
        oldClearAlpha = renderer.getClearAlpha();
        renderer.setClearColor(*clearColor, clearAlpha);
    }

    renderer.setRenderTarget(renderToScreen ? nullptr : readBuffer);

    // After the bind, not before it as in the three.js original — there the
    // clear lands on whatever target the previous pass left bound, which is
    // never the one this pass is about to draw into.
    if (clearDepth) renderer.clearDepth();

    if (clear) renderer.clear(renderer.autoClearColor, renderer.autoClearDepth, renderer.autoClearStencil);

    renderer.render(*scene_, *camera_);

    if (clearColor) renderer.setClearColor(oldClearColor, oldClearAlpha);

    if (overrideMaterial && asScene) asScene->overrideMaterial = oldOverrideMaterial;

    renderer.autoClear = oldAutoClear;
}
