
#include "threepp/postprocessing/ShaderPass.hpp"

#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"

#include <utility>

using namespace threepp;

namespace {

    std::shared_ptr<ShaderMaterial> materialFrom(const Shader& shader) {

        auto material = ShaderMaterial::create();
        material->uniforms = shader.uniforms;
        material->vertexShader = shader.vertexShader;
        material->fragmentShader = shader.fragmentShader;

        // A pass covers the frame unconditionally. Leaving the depth test on
        // would make the result depend on whatever depth the target happens to
        // be carrying from an earlier pass — and it can only ever reject.
        material->depthTest = false;
        material->depthWrite = false;

        return material;
    }

}// namespace


ShaderPass::ShaderPass(const Shader& shader, std::string textureID)
    : textureID_(std::move(textureID)),
      material_(materialFrom(shader)),
      fsQuad_(std::make_unique<FullScreenQuad>(material_)) {}

ShaderPass::ShaderPass(std::shared_ptr<ShaderMaterial> material, std::string textureID)
    : textureID_(std::move(textureID)),
      material_(std::move(material)),
      fsQuad_(std::make_unique<FullScreenQuad>(material_)) {}

UniformMap& ShaderPass::uniforms() {

    return material_->uniforms;
}

std::shared_ptr<ShaderMaterial> ShaderPass::material() const {

    return material_;
}

void ShaderPass::render(GLRenderer& renderer, RenderTarget* writeBuffer, RenderTarget* readBuffer, float, bool maskActive) {

    if (auto it = material_->uniforms.find(textureID_); it != material_->uniforms.end()) {

        it->second.setValue(readBuffer ? readBuffer->texture.get() : static_cast<Texture*>(nullptr));
    }

    // Under an active mask the target carries the stencil the mask wrote, and
    // the pixels the mask protects. The implicit clear inside render() would
    // take out both — leaving a pass that draws nowhere over a black frame.
    // (three.js never hit this because its masking example only ever brackets
    // TexturePass, which turns autoClear off for its own reasons.)
    const bool oldAutoClear = renderer.autoClear;
    if (maskActive) renderer.autoClear = false;

    if (renderToScreen) {

        renderer.setRenderTarget(nullptr);

    } else {

        renderer.setRenderTarget(writeBuffer);

        if (clear) renderer.clear(renderer.autoClearColor, renderer.autoClearDepth, renderer.autoClearStencil);
    }

    fsQuad_->render(renderer);

    renderer.autoClear = oldAutoClear;
}

ShaderPass::~ShaderPass() = default;
