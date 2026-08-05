
#include "threepp/postprocessing/SavePass.hpp"

#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/postprocessing/shaders/CopyShader.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"

#include <utility>

using namespace threepp;

namespace {

    std::shared_ptr<ShaderMaterial> copyMaterial() {

        const auto shader = shaders::copyShader();

        auto material = ShaderMaterial::create();
        material->uniforms = shader.uniforms;
        material->vertexShader = shader.vertexShader;
        material->fragmentShader = shader.fragmentShader;
        material->depthTest = false;
        material->depthWrite = false;

        return material;
    }

    std::shared_ptr<RenderTarget> createTarget(unsigned int width, unsigned int height) {

        RenderTarget::Options parameters;
        parameters.minFilter = Filter::Linear;
        parameters.magFilter = Filter::Linear;
        parameters.format = Format::RGBA;
        parameters.generateMipmaps = false;
        parameters.stencilBuffer = false;

        return RenderTarget::create(width, height, parameters);
    }

}// namespace


SavePass::SavePass(std::shared_ptr<RenderTarget> renderTarget)
    : ownsTarget_(renderTarget == nullptr),
      renderTarget_(std::move(renderTarget)),
      material_(copyMaterial()),
      fsQuad_(std::make_unique<FullScreenQuad>(material_)) {

    needsSwap = false;
}

RenderTarget* SavePass::renderTarget() const {

    return renderTarget_.get();
}

void SavePass::setSize(unsigned int width, unsigned int height) {

    if (!ownsTarget_) return;

    if (!renderTarget_) {
        renderTarget_ = createTarget(width, height);
        renderTarget_->texture->name = "SavePass.rt";
    } else {
        renderTarget_->setSize(width, height);
    }
}

void SavePass::render(GLRenderer& renderer, RenderTarget*, RenderTarget* readBuffer, float, bool) {

    if (!renderTarget_) return;

    material_->uniforms.at("tDiffuse").setValue(readBuffer ? readBuffer->texture.get() : static_cast<Texture*>(nullptr));

    renderer.setRenderTarget(renderTarget_.get());

    if (clear) renderer.clear();

    fsQuad_->render(renderer);
}

SavePass::~SavePass() = default;
