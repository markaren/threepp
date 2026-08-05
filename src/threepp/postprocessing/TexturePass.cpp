
#include "threepp/postprocessing/TexturePass.hpp"

#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/postprocessing/shaders/CopyShader.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/textures/Texture.hpp"

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

}// namespace


TexturePass::TexturePass(std::shared_ptr<Texture> map, float opacity)
    : opacity(opacity),
      map_(std::move(map)),
      material_(copyMaterial()),
      fsQuad_(std::make_unique<FullScreenQuad>(material_)) {

    needsSwap = false;
}

void TexturePass::setTexture(std::shared_ptr<Texture> map) {

    map_ = std::move(map);
}

void TexturePass::render(GLRenderer& renderer, RenderTarget*, RenderTarget* readBuffer, float, bool) {

    // Blending onto the existing image only works if nothing clears it first.
    const bool oldAutoClear = renderer.autoClear;
    renderer.autoClear = false;

    material_->uniforms.at("opacity").setValue(opacity);
    material_->uniforms.at("tDiffuse").setValue(map_.get());
    material_->transparent = opacity < 1.f;

    renderer.setRenderTarget(renderToScreen ? nullptr : readBuffer);

    if (clear) renderer.clear();

    fsQuad_->render(renderer);

    renderer.autoClear = oldAutoClear;
}

TexturePass::~TexturePass() = default;
