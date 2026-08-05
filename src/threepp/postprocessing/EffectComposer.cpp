
#include "threepp/postprocessing/EffectComposer.hpp"

#include "threepp/postprocessing/MaskPass.hpp"
#include "threepp/postprocessing/Pass.hpp"
#include "threepp/postprocessing/ShaderPass.hpp"
#include "threepp/postprocessing/shaders/CopyShader.hpp"

#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"

#include <algorithm>
#include <utility>

using namespace threepp;


struct EffectComposer::Impl {

    GLRenderer& renderer;

    Options options;

    unsigned int width;
    unsigned int height;
    float pixelRatio;

    std::unique_ptr<RenderTarget> renderTarget1;
    std::unique_ptr<RenderTarget> renderTarget2;

    RenderTarget* readBuffer;
    RenderTarget* writeBuffer;

    std::vector<std::shared_ptr<Pass>> passes;

    // Used mid-chain to carry a masked region across a buffer swap.
    std::shared_ptr<ShaderPass> copyPass;

    // Owns the draw to the screen, and with it the output colour transform.
    std::shared_ptr<ShaderPass> outputPass;

    Impl(GLRenderer& renderer, const Options& options)
        : renderer(renderer),
          options(options),
          width(static_cast<unsigned int>(renderer.size().width())),
          height(static_cast<unsigned int>(renderer.size().height())),
          pixelRatio(renderer.getTargetPixelRatio()),
          copyPass(std::make_shared<ShaderPass>(shaders::copyShader())),
          outputPass(std::make_shared<ShaderPass>(shaders::copyShader())) {

        allocateTargets();
    }

    void allocateTargets() {

        RenderTarget::Options parameters;
        parameters.minFilter = Filter::Linear;
        parameters.magFilter = Filter::Linear;
        parameters.format = Format::RGBA;
        parameters.generateMipmaps = false;
        parameters.depthBuffer = options.depthBuffer;
        parameters.stencilBuffer = options.stencilBuffer;
        parameters.samples = options.samples;
        if (options.type) parameters.type = *options.type;

        const auto w = effectiveWidth();
        const auto h = effectiveHeight();

        renderTarget1 = RenderTarget::create(w, h, parameters);
        renderTarget1->texture->name = "EffectComposer.rt1";

        renderTarget2 = RenderTarget::create(w, h, parameters);
        renderTarget2->texture->name = "EffectComposer.rt2";

        writeBuffer = renderTarget1.get();
        readBuffer = renderTarget2.get();
    }

    [[nodiscard]] unsigned int effectiveWidth() const {

        return std::max(1u, static_cast<unsigned int>(static_cast<float>(width) * pixelRatio));
    }

    [[nodiscard]] unsigned int effectiveHeight() const {

        return std::max(1u, static_cast<unsigned int>(static_cast<float>(height) * pixelRatio));
    }

    void swapBuffers() {

        std::swap(readBuffer, writeBuffer);
    }

    [[nodiscard]] bool isLastEnabledPass(size_t index) const {

        for (size_t i = index + 1; i < passes.size(); i++) {
            if (passes[i]->enabled) return false;
        }

        return true;
    }

    void setSize(unsigned int width, unsigned int height) {

        this->width = width;
        this->height = height;

        const auto w = effectiveWidth();
        const auto h = effectiveHeight();

        renderTarget1->setSize(w, h);
        renderTarget2->setSize(w, h);

        for (auto& pass : passes) {
            pass->setSize(w, h);
        }
    }

    void render(float deltaTime, bool renderToScreen) {

        // The composer takes over the render target for the duration of the
        // chain; whatever the caller had bound is put back at the end.
        auto* currentRenderTarget = renderer.getRenderTarget();

        bool maskActive = false;

        for (size_t i = 0; i < passes.size(); i++) {

            auto& pass = passes[i];

            if (!pass->enabled) continue;

            pass->render(renderer, writeBuffer, readBuffer, deltaTime, maskActive);

            if (pass->needsSwap) {

                if (maskActive) {

                    // Carry the pixels the mask excludes across the swap, so
                    // the next pass reads an image and not a hole.
                    auto& stencil = renderer.state().stencilBuffer;
                    stencil.setFunc(StencilFunc::NotEqual, 1, 0xffffffff);
                    copyPass->render(renderer, writeBuffer, readBuffer, deltaTime, maskActive);
                    stencil.setFunc(StencilFunc::Equal, 1, 0xffffffff);
                }

                swapBuffers();
            }

            if (dynamic_cast<MaskPass*>(pass.get())) {
                maskActive = true;
            } else if (dynamic_cast<ClearMaskPass*>(pass.get())) {
                maskActive = false;
            }
        }

        if (renderToScreen) {

            outputPass->renderToScreen = true;
            outputPass->render(renderer, nullptr, readBuffer, deltaTime, maskActive);
        }

        renderer.setRenderTarget(currentRenderTarget);
    }
};


EffectComposer::EffectComposer(GLRenderer& renderer)
    : pimpl_(std::make_unique<Impl>(renderer, Options{})) {}

EffectComposer::EffectComposer(GLRenderer& renderer, const Options& options)
    : pimpl_(std::make_unique<Impl>(renderer, options)) {}

void EffectComposer::addPass(const std::shared_ptr<Pass>& pass) {

    pimpl_->passes.emplace_back(pass);
    pass->setSize(pimpl_->effectiveWidth(), pimpl_->effectiveHeight());
}

void EffectComposer::insertPass(const std::shared_ptr<Pass>& pass, size_t index) {

    auto& passes = pimpl_->passes;
    passes.insert(passes.begin() + static_cast<std::ptrdiff_t>(std::min(index, passes.size())), pass);
    pass->setSize(pimpl_->effectiveWidth(), pimpl_->effectiveHeight());
}

void EffectComposer::removePass(const Pass* pass) {

    auto& passes = pimpl_->passes;
    passes.erase(std::remove_if(passes.begin(), passes.end(),
                                [pass](const std::shared_ptr<Pass>& p) { return p.get() == pass; }),
                 passes.end());
}

const std::vector<std::shared_ptr<Pass>>& EffectComposer::passes() const {

    return pimpl_->passes;
}

bool EffectComposer::isLastEnabledPass(size_t index) const {

    return pimpl_->isLastEnabledPass(index);
}

void EffectComposer::render(float deltaTime) {

    pimpl_->render(deltaTime, renderToScreen);
}

void EffectComposer::setSize(unsigned int width, unsigned int height) {

    pimpl_->setSize(width, height);
}

void EffectComposer::setPixelRatio(float pixelRatio) {

    pimpl_->pixelRatio = pixelRatio;
    pimpl_->setSize(pimpl_->width, pimpl_->height);
}

float EffectComposer::getPixelRatio() const {

    return pimpl_->pixelRatio;
}

RenderTarget& EffectComposer::readBuffer() const {

    return *pimpl_->readBuffer;
}

RenderTarget& EffectComposer::writeBuffer() const {

    return *pimpl_->writeBuffer;
}

void EffectComposer::reset(const Options& options) {

    pimpl_->options = options;
    pimpl_->allocateTargets();
}

std::vector<unsigned char> EffectComposer::readRGBPixels() const {

    auto& renderer = pimpl_->renderer;

    auto* currentRenderTarget = renderer.getRenderTarget();
    renderer.setRenderTarget(pimpl_->readBuffer);
    auto pixels = renderer.readRGBPixels();
    renderer.setRenderTarget(currentRenderTarget);

    return pixels;
}

EffectComposer::~EffectComposer() = default;
