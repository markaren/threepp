
#include "threepp/postprocessing/BokehPass.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/materials/MeshDepthMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/postprocessing/shaders/BokehShader.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/scenes/Scene.hpp"

using namespace threepp;


BokehPass::BokehPass(Scene& scene, Camera& camera, float focus, float aperture, float maxblur)
    : focus(focus), aperture(aperture), maxblur(maxblur), aspect(1.f),
      scene_(&scene), camera_(&camera) {

    // Depth is packed into RGBA rather than sampled from a depth texture, so
    // the prepass target is an ordinary colour target — and has to be point
    // sampled, since interpolating between two packed depths gives a depth
    // that belongs to neither.
    RenderTarget::Options options;
    options.minFilter = Filter::Nearest;
    options.magFilter = Filter::Nearest;
    options.generateMipmaps = false;

    renderTargetDepth_ = RenderTarget::create(1, 1, options);
    renderTargetDepth_->texture->name = "BokehPass.depth";

    depthMaterial_ = MeshDepthMaterial::create();
    depthMaterial_->depthPacking = DepthPacking::RGBA;
    depthMaterial_->blending = Blending::None;

    const auto shader = shaders::bokehShader();

    bokehMaterial_ = ShaderMaterial::create();
    bokehMaterial_->uniforms = shader.uniforms;
    bokehMaterial_->vertexShader = shader.vertexShader;
    bokehMaterial_->fragmentShader = shader.fragmentShader;
    bokehMaterial_->defines["DEPTH_PACKING"] = "1";
    bokehMaterial_->defines["PERSPECTIVE_CAMERA"] = dynamic_cast<OrthographicCamera*>(&camera) ? "0" : "1";
    bokehMaterial_->depthTest = false;
    bokehMaterial_->depthWrite = false;

    bokehMaterial_->uniforms.at("tDepth").setValue(renderTargetDepth_->texture.get());

    fsQuad_ = std::make_unique<FullScreenQuad>(bokehMaterial_);
}

void BokehPass::setSize(unsigned int width, unsigned int height) {

    renderTargetDepth_->setSize(width, height);

    if (aspectFollowsSize_ && height > 0) {
        aspect = static_cast<float>(width) / static_cast<float>(height);
    }
}

void BokehPass::render(GLRenderer& renderer, RenderTarget* writeBuffer, RenderTarget* readBuffer, float, bool) {

    Color oldClearColor;
    renderer.getClearColor(oldClearColor);
    const float oldClearAlpha = renderer.getClearAlpha();
    const bool oldAutoClear = renderer.autoClear;

    renderer.autoClear = false;

    // Depth prepass. Cleared to white so anything the scene does not cover
    // reads as maximum distance rather than as sitting on the near plane.
    auto oldOverrideMaterial = scene_->overrideMaterial;
    scene_->overrideMaterial = depthMaterial_;

    renderer.setClearColor(Color(0xffffff), 1.f);
    renderer.setRenderTarget(renderTargetDepth_.get());
    renderer.clear();
    renderer.render(*scene_, *camera_);

    scene_->overrideMaterial = oldOverrideMaterial;

    // Composite.
    bokehMaterial_->uniforms.at("tColor").setValue(readBuffer ? readBuffer->texture.get() : static_cast<Texture*>(nullptr));
    bokehMaterial_->uniforms.at("focus").setValue(focus);
    bokehMaterial_->uniforms.at("aperture").setValue(aperture);
    bokehMaterial_->uniforms.at("maxblur").setValue(maxblur);
    bokehMaterial_->uniforms.at("aspect").setValue(aspect);
    bokehMaterial_->uniforms.at("nearClip").setValue(camera_->nearPlane);
    bokehMaterial_->uniforms.at("farClip").setValue(camera_->farPlane);

    fsQuad_->setMaterial(bokehMaterial_);

    if (renderToScreen) {

        renderer.setRenderTarget(nullptr);

    } else {

        renderer.setRenderTarget(writeBuffer);
        renderer.clear();
    }

    fsQuad_->render(renderer);

    renderer.setClearColor(oldClearColor, oldClearAlpha);
    renderer.autoClear = oldAutoClear;
}

BokehPass::~BokehPass() = default;
