
#include "threepp/postprocessing/SSAARenderPass.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/postprocessing/shaders/CopyShader.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

using namespace threepp;

namespace {

    // The standard multisample patterns (1, 2, 4, 8, 16 and 32 samples) on a
    // grid of sixteenths of a pixel, in [-8, 8).
    // https://msdn.microsoft.com/en-us/library/windows/desktop/ff476218%28v=vs.85%29.aspx
    const std::array<std::vector<std::array<int, 2>>, 6> jitterVectors{{
            {{0, 0}},
            {{4, 4}, {-4, -4}},
            {{-2, -6}, {6, -2}, {-6, 2}, {2, 6}},
            {{1, -3}, {-1, 3}, {5, 1}, {-3, -5},
             {-5, 5}, {-7, -1}, {3, 7}, {7, -7}},
            {{1, 1}, {-1, -3}, {-3, 2}, {4, -1},
             {-5, -2}, {2, 5}, {5, 3}, {3, -5},
             {-2, 6}, {0, -7}, {-4, -6}, {-6, 4},
             {-8, 0}, {7, -4}, {6, 7}, {-7, -8}},
            {{-4, -7}, {-7, -5}, {-3, -5}, {-5, -4},
             {-1, -4}, {-2, -2}, {-6, -1}, {-4, 0},
             {-7, 1}, {-1, 2}, {-6, 3}, {-3, 3},
             {-7, 6}, {-3, 6}, {-5, 7}, {-1, 7},
             {5, -7}, {1, -6}, {6, -5}, {4, -4},
             {2, -3}, {7, -2}, {1, -1}, {4, -1},
             {2, 1}, {6, 2}, {0, 4}, {4, 4},
             {2, 5}, {7, 5}, {5, 6}, {3, 7}},
    }};

    std::shared_ptr<ShaderMaterial> accumulateMaterial() {

        const auto shader = shaders::copyShader();

        auto material = ShaderMaterial::create();
        material->uniforms = shader.uniforms;
        material->vertexShader = shader.vertexShader;
        material->fragmentShader = shader.fragmentShader;
        // each sample adds its weighted share to what the others left
        material->transparent = true;
        material->depthTest = false;
        material->depthWrite = false;
        material->premultipliedAlpha = true;
        material->blending = Blending::Additive;

        return material;
    }

    void setViewOffset(Camera& camera, const CameraView& v) {

        if (auto* perspective = camera.as<PerspectiveCamera>()) {
            perspective->setViewOffset(v.fullWidth, v.fullHeight, v.offsetX, v.offsetY, v.width, v.height);
        } else if (auto* orthographic = camera.as<OrthographicCamera>()) {
            orthographic->setViewOffset(v.fullWidth, v.fullHeight, v.offsetX, v.offsetY, v.width, v.height);
        }
    }

    void clearViewOffset(Camera& camera) {

        if (auto* perspective = camera.as<PerspectiveCamera>()) {
            perspective->clearViewOffset();
        } else if (auto* orthographic = camera.as<OrthographicCamera>()) {
            orthographic->clearViewOffset();
        }
    }

}// namespace


SSAARenderPass::SSAARenderPass(Object3D& scene, Camera& camera, const Color& clearColor, float clearAlpha)
    : clearColor(clearColor),
      clearAlpha(clearAlpha),
      scene_(&scene),
      camera_(&camera),
      copyMaterial_(accumulateMaterial()),
      fsQuad_(std::make_unique<FullScreenQuad>(copyMaterial_)) {}

void SSAARenderPass::setSize(unsigned int width, unsigned int height) {

    if (sampleRenderTarget_) sampleRenderTarget_->setSize(width, height);
}

void SSAARenderPass::render(GLRenderer& renderer, RenderTarget* writeBuffer, RenderTarget* readBuffer, float, bool) {

    if (!sampleRenderTarget_) {

        RenderTarget::Options options;
        options.type = Type::HalfFloat;
        options.stencilBuffer = stencilBuffer;
        sampleRenderTarget_ = RenderTarget::create(readBuffer->width, readBuffer->height, options);
        sampleRenderTarget_->texture->name = "SSAARenderPass.sample";
    }

    const auto& jitterOffsets = jitterVectors[std::clamp(sampleLevel, 0, 5)];
    const auto count = static_cast<float>(jitterOffsets.size());

    const bool oldAutoClear = renderer.autoClear;
    renderer.autoClear = false;

    Color oldClearColor;
    renderer.getClearColor(oldClearColor);
    const float oldClearAlpha = renderer.getClearAlpha();

    const float baseSampleWeight = 1.f / count;
    const float roundingRange = 1.f / 32;
    copyMaterial_->uniforms.at("tDiffuse").setValue(sampleRenderTarget_->texture.get());

    // the whole picture, unless the camera already draws a window of a larger one
    const std::optional<CameraView> originalView = camera_->view;
    CameraView view{true,
                    static_cast<float>(readBuffer->width), static_cast<float>(readBuffer->height),
                    0.f, 0.f,
                    static_cast<float>(readBuffer->width), static_cast<float>(readBuffer->height)};
    if (originalView && originalView->enabled) view = *originalView;

    // the scene several times, each shifted a little from the last, and accumulated
    for (size_t i = 0; i < jitterOffsets.size(); ++i) {

        const auto& jitter = jitterOffsets[i];
        CameraView jittered = view;
        jittered.offsetX += static_cast<float>(jitter[0]) * 0.0625f;// sixteenths of a pixel
        jittered.offsetY += static_cast<float>(jitter[1]) * 0.0625f;
        setViewOffset(*camera_, jittered);

        float sampleWeight = baseSampleWeight;
        if (unbiased) {
            // Equal weights accumulate their rounding errors; weights spread
            // evenly over a small range around 1/N cancel them instead.
            const float uniformCenteredDistribution = -0.5f + (static_cast<float>(i) + 0.5f) / count;
            sampleWeight += roundingRange * uniformCenteredDistribution;
        }
        copyMaterial_->uniforms.at("opacity").setValue(sampleWeight);

        renderer.setClearColor(clearColor, clearAlpha);
        renderer.setRenderTarget(sampleRenderTarget_.get());
        renderer.clear();
        renderer.render(*scene_, *camera_);

        renderer.setRenderTarget(renderToScreen ? nullptr : writeBuffer);

        if (i == 0) {
            renderer.setClearColor(Color(0x000000), 0.f);
            renderer.clear();
        }

        fsQuad_->render(renderer);
    }

    if (originalView && originalView->enabled) {
        setViewOffset(*camera_, *originalView);
    } else {
        clearViewOffset(*camera_);
    }

    renderer.autoClear = oldAutoClear;
    renderer.setClearColor(oldClearColor, oldClearAlpha);
}

SSAARenderPass::~SSAARenderPass() = default;
