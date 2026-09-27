
#include "threepp/postprocessing/OutlinePass.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/materials/MeshDepthMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/postprocessing/shaders/CopyShader.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/gl/GLShadowMap.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

using namespace threepp;

namespace {

    constexpr int MAX_EDGE_THICKNESS = 4;
    constexpr int MAX_EDGE_GLOW = 4;

    const Vector2 blurDirectionX{1.f, 0.f};
    const Vector2 blurDirectionY{0.f, 1.f};

    unsigned int roundDiv(unsigned int v, float d) {

        return std::max(1u, static_cast<unsigned int>(std::round(static_cast<float>(v) / d)));
    }

    std::unique_ptr<RenderTarget> makeTarget(unsigned int w, unsigned int h, const std::string& name,
                                             std::optional<Type> type, bool depthBuffer) {

        RenderTarget::Options options;
        options.minFilter = Filter::Linear;
        options.magFilter = Filter::Linear;
        options.format = Format::RGBA;
        options.type = type;
        options.depthBuffer = depthBuffer;
        options.stencilBuffer = false;
        options.generateMipmaps = false;

        auto target = RenderTarget::create(w, h, options);
        target->texture->name = name;
        target->texture->generateMipmaps = false;

        return target;
    }

    const char* fullScreenVertex = R"(
            varying vec2 vUv;

            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
            })";

    std::shared_ptr<ShaderMaterial> prepareMaskMaterial(bool perspective) {

        auto material = ShaderMaterial::create();

        material->uniforms = UniformMap{
                {"depthTexture", Uniform()},
                {"cameraNearFar", Uniform(Vector2(0.5f, 0.5f))},
                {"textureMatrix", Uniform(Matrix4())}};

        // three.js string-replaces DEPTH_TO_VIEW_Z; a define does the same.
        material->defines["DEPTH_TO_VIEW_Z"] = perspective ? "perspectiveDepthToViewZ" : "orthographicDepthToViewZ";

        // The skinning and instancing chunks keep a posed or instanced
        // selection in the shape it is drawn in; the renderer turns them on
        // from the object being drawn, whatever the material.
        material->vertexShader = R"(
                #include <morphtarget_pars_vertex>
                #include <skinning_pars_vertex>

                varying vec4 projTexCoord;
                varying vec4 vPosition;
                uniform mat4 textureMatrix;

                void main() {

                    #include <skinbase_vertex>
                    #include <begin_vertex>
                    #include <morphtarget_vertex>
                    #include <skinning_vertex>
                    #include <project_vertex>

                    vPosition = mvPosition;

                    vec4 worldPosition = vec4( transformed, 1.0 );

                    #ifdef USE_INSTANCING

                        worldPosition = instanceMatrix * worldPosition;

                    #endif

                    worldPosition = modelMatrix * worldPosition;

                    projTexCoord = textureMatrix * worldPosition;

                })";

        material->fragmentShader = R"(
                #include <packing>
                varying vec4 vPosition;
                varying vec4 projTexCoord;
                uniform sampler2D depthTexture;
                uniform vec2 cameraNearFar;

                void main() {

                    float depth = unpackRGBAToDepth(texture2DProj( depthTexture, projTexCoord ));
                    float viewZ = - DEPTH_TO_VIEW_Z( depth, cameraNearFar.x, cameraNearFar.y );
                    float depthTest = (-vPosition.z > viewZ) ? 1.0 : 0.0;
                    gl_FragColor = vec4(0.0, depthTest, 1.0, 1.0);

                })";

        material->side = Side::Double;

        return material;
    }

    std::shared_ptr<ShaderMaterial> edgeDetectionMaterial() {

        auto material = ShaderMaterial::create();

        material->uniforms = UniformMap{
                {"maskTexture", Uniform()},
                {"texSize", Uniform(Vector2(0.5f, 0.5f))},
                {"visibleEdgeColor", Uniform(Color(1, 1, 1))},
                {"hiddenEdgeColor", Uniform(Color(1, 1, 1))}};

        material->vertexShader = fullScreenVertex;

        material->fragmentShader = R"(
                varying vec2 vUv;

                uniform sampler2D maskTexture;
                uniform vec2 texSize;
                uniform vec3 visibleEdgeColor;
                uniform vec3 hiddenEdgeColor;

                void main() {
                    vec2 invSize = 1.0 / texSize;
                    vec4 uvOffset = vec4(1.0, 0.0, 0.0, 1.0) * vec4(invSize, invSize);
                    vec4 c1 = texture2D( maskTexture, vUv + uvOffset.xy);
                    vec4 c2 = texture2D( maskTexture, vUv - uvOffset.xy);
                    vec4 c3 = texture2D( maskTexture, vUv + uvOffset.yw);
                    vec4 c4 = texture2D( maskTexture, vUv - uvOffset.yw);
                    float diff1 = (c1.r - c2.r)*0.5;
                    float diff2 = (c3.r - c4.r)*0.5;
                    float d = length( vec2(diff1, diff2) );
                    float a1 = min(c1.g, c2.g);
                    float a2 = min(c3.g, c4.g);
                    float visibilityFactor = min(a1, a2);
                    vec3 edgeColor = 1.0 - visibilityFactor > 0.001 ? visibleEdgeColor : hiddenEdgeColor;
                    gl_FragColor = vec4(edgeColor, 1.0) * vec4(d);
                })";

        material->depthTest = false;
        material->depthWrite = false;

        return material;
    }

    std::shared_ptr<ShaderMaterial> separableBlurMaterial(int maxRadius) {

        auto material = ShaderMaterial::create();

        material->defines["MAX_RADIUS"] = std::to_string(maxRadius);

        material->uniforms = UniformMap{
                {"colorTexture", Uniform()},
                {"texSize", Uniform(Vector2(0.5f, 0.5f))},
                {"direction", Uniform(Vector2(0.5f, 0.5f))},
                {"kernelRadius", Uniform(1.f)}};

        material->vertexShader = fullScreenVertex;

        material->fragmentShader = R"(
                #include <common>
                varying vec2 vUv;
                uniform sampler2D colorTexture;
                uniform vec2 texSize;
                uniform vec2 direction;
                uniform float kernelRadius;

                float gaussianPdf(in float x, in float sigma) {
                    return 0.39894 * exp( -0.5 * x * x/( sigma * sigma))/sigma;
                }

                void main() {
                    vec2 invSize = 1.0 / texSize;
                    float sigma = kernelRadius/2.0;
                    float weightSum = gaussianPdf(0.0, sigma);
                    vec4 diffuseSum = texture2D( colorTexture, vUv) * weightSum;
                    vec2 delta = direction * invSize * kernelRadius/float(MAX_RADIUS);
                    vec2 uvOffset = delta;
                    for( int i = 1; i <= MAX_RADIUS; i ++ ) {
                        float x = kernelRadius * float(i) / float(MAX_RADIUS);
                        float w = gaussianPdf(x, sigma);
                        vec4 sample1 = texture2D( colorTexture, vUv + uvOffset);
                        vec4 sample2 = texture2D( colorTexture, vUv - uvOffset);
                        diffuseSum += ((sample1 + sample2) * w);
                        weightSum += (2.0 * w);
                        uvOffset += delta;
                    }
                    gl_FragColor = diffuseSum/weightSum;
                })";

        material->depthTest = false;
        material->depthWrite = false;

        return material;
    }

    std::shared_ptr<ShaderMaterial> overlayMaterial() {

        auto material = ShaderMaterial::create();

        material->uniforms = UniformMap{
                {"maskTexture", Uniform()},
                {"edgeTexture1", Uniform()},
                {"edgeTexture2", Uniform()},
                {"patternTexture", Uniform()},
                {"edgeStrength", Uniform(1.f)},
                {"edgeGlow", Uniform(1.f)},
                {"usePatternTexture", Uniform(false)}};

        material->vertexShader = fullScreenVertex;

        material->fragmentShader = R"(
                varying vec2 vUv;

                uniform sampler2D maskTexture;
                uniform sampler2D edgeTexture1;
                uniform sampler2D edgeTexture2;
                uniform sampler2D patternTexture;
                uniform float edgeStrength;
                uniform float edgeGlow;
                uniform bool usePatternTexture;

                void main() {
                    vec4 edgeValue1 = texture2D(edgeTexture1, vUv);
                    vec4 edgeValue2 = texture2D(edgeTexture2, vUv);
                    vec4 maskColor = texture2D(maskTexture, vUv);
                    vec4 patternColor = texture2D(patternTexture, 6.0 * vUv);
                    float visibilityFactor = 1.0 - maskColor.g > 0.0 ? 1.0 : 0.5;
                    vec4 edgeValue = edgeValue1 + edgeValue2 * edgeGlow;
                    vec4 finalColor = edgeStrength * maskColor.r * edgeValue;
                    if(usePatternTexture)
                        finalColor += + visibilityFactor * (1.0 - maskColor.r) * (1.0 - patternColor.r);
                    gl_FragColor = finalColor;
                })";

        material->blending = Blending::Additive;
        material->depthTest = false;
        material->depthWrite = false;
        material->transparent = true;

        return material;
    }

    float nowMillis() {

        using namespace std::chrono;
        static const auto start = steady_clock::now();
        return duration<float, std::milli>(steady_clock::now() - start).count();
    }

}// namespace


OutlinePass::OutlinePass(const Vector2& resolution, Scene& scene, Camera& camera, std::vector<Object3D*> selectedObjects)
    : selectedObjects(std::move(selectedObjects)),
      resolution(resolution),
      renderScene_(&scene),
      renderCamera_(&camera),
      perspective_(dynamic_cast<OrthographicCamera*>(&camera) == nullptr) {

    needsSwap = false;

    const auto w = std::max(1u, static_cast<unsigned int>(resolution.x));
    const auto h = std::max(1u, static_cast<unsigned int>(resolution.y));
    const auto resx = roundDiv(w, downSampleRatio);
    const auto resy = roundDiv(h, downSampleRatio);

    // Byte mask with a depth buffer of its own, so the selection occludes
    // itself; half-float everything else, as in three.js.
    renderTargetMaskBuffer_ = makeTarget(w, h, "OutlinePass.mask", std::nullopt, true);
    renderTargetDepthBuffer_ = makeTarget(w, h, "OutlinePass.depth", Type::HalfFloat, true);
    renderTargetMaskDownSampleBuffer_ = makeTarget(resx, resy, "OutlinePass.depthDownSample", Type::HalfFloat, false);
    renderTargetBlurBuffer1_ = makeTarget(resx, resy, "OutlinePass.blur1", Type::HalfFloat, false);
    renderTargetBlurBuffer2_ = makeTarget(roundDiv(resx, 2), roundDiv(resy, 2), "OutlinePass.blur2", Type::HalfFloat, false);
    renderTargetEdgeBuffer1_ = makeTarget(resx, resy, "OutlinePass.edge1", Type::HalfFloat, false);
    renderTargetEdgeBuffer2_ = makeTarget(roundDiv(resx, 2), roundDiv(resy, 2), "OutlinePass.edge2", Type::HalfFloat, false);

    depthMaterial_ = MeshDepthMaterial::create();
    depthMaterial_->side = Side::Double;
    depthMaterial_->depthPacking = DepthPacking::RGBA;
    depthMaterial_->blending = Blending::None;

    prepareMaskMaterial_ = prepareMaskMaterial(perspective_);
    edgeDetectionMaterial_ = edgeDetectionMaterial();

    separableBlurMaterial1_ = separableBlurMaterial(MAX_EDGE_THICKNESS);
    separableBlurMaterial1_->uniforms.at("texSize").setValue(Vector2(static_cast<float>(resx), static_cast<float>(resy)));
    separableBlurMaterial1_->uniforms.at("kernelRadius").setValue(1.f);
    separableBlurMaterial2_ = separableBlurMaterial(MAX_EDGE_GLOW);
    separableBlurMaterial2_->uniforms.at("texSize").setValue(Vector2(static_cast<float>(roundDiv(resx, 2)), static_cast<float>(roundDiv(resy, 2))));
    separableBlurMaterial2_->uniforms.at("kernelRadius").setValue(static_cast<float>(MAX_EDGE_GLOW));

    overlayMaterial_ = overlayMaterial();

    const auto copy = shaders::copyShader();
    materialCopy_ = ShaderMaterial::create();
    materialCopy_->uniforms = copy.uniforms;
    materialCopy_->vertexShader = copy.vertexShader;
    materialCopy_->fragmentShader = copy.fragmentShader;
    materialCopy_->blending = Blending::None;
    materialCopy_->depthTest = false;
    materialCopy_->depthWrite = false;

    placeholderTexture_ = Texture::create(Image(std::vector<unsigned char>{255, 255, 255, 255}, 1, 1));
    placeholderTexture_->needsUpdate();

    fsQuad_ = std::make_unique<FullScreenQuad>(materialCopy_);
}

void OutlinePass::setSize(unsigned int width, unsigned int height) {

    renderTargetMaskBuffer_->setSize(width, height);
    renderTargetDepthBuffer_->setSize(width, height);

    auto resx = roundDiv(width, downSampleRatio);
    auto resy = roundDiv(height, downSampleRatio);
    renderTargetMaskDownSampleBuffer_->setSize(resx, resy);
    renderTargetBlurBuffer1_->setSize(resx, resy);
    renderTargetEdgeBuffer1_->setSize(resx, resy);
    separableBlurMaterial1_->uniforms.at("texSize").setValue(Vector2(static_cast<float>(resx), static_cast<float>(resy)));

    resx = roundDiv(resx, 2);
    resy = roundDiv(resy, 2);

    renderTargetBlurBuffer2_->setSize(resx, resy);
    renderTargetEdgeBuffer2_->setSize(resx, resy);

    separableBlurMaterial2_->uniforms.at("texSize").setValue(Vector2(static_cast<float>(resx), static_cast<float>(resy)));
}

void OutlinePass::render(GLRenderer& renderer, RenderTarget*, RenderTarget* readBuffer, float, bool maskActive) {

    if (!selectedObjects.empty()) {

        Color oldClearColor;
        renderer.getClearColor(oldClearColor);
        const float oldClearAlpha = renderer.getClearAlpha();
        const bool oldAutoClear = renderer.autoClear;

        renderer.autoClear = false;

        if (maskActive) renderer.state().stencilBuffer.setTest(false);

        renderer.setClearColor(Color(0xffffff), 1.f);

        // The two internal renders draw depth and a mask, neither of which
        // reads a shadow map; the RenderPass has already brought the shadow
        // maps up to date for this frame, and hiding half the scene would only
        // make the renderer draw them again without it. three.js re-renders
        // them here; the result is the same, the cost is not.
        auto& shadowMap = renderer.shadowMap();
        const bool oldShadowAutoUpdate = shadowMap.autoUpdate;
        shadowMap.autoUpdate = false;

        updateSelectionCache();

        // Make selected objects invisible
        changeVisibilityOfSelectedObjects(false);

        const auto currentBackground = renderScene_->background;
        const auto currentOverrideMaterial = renderScene_->overrideMaterial;
        renderScene_->background = Background();

        // 1. Draw non-selected objects in the depth buffer
        renderScene_->overrideMaterial = depthMaterial_;
        renderer.setRenderTarget(renderTargetDepthBuffer_.get());
        renderer.clear();
        renderer.render(*renderScene_, *renderCamera_);

        // Make selected objects visible
        changeVisibilityOfSelectedObjects(true);
        visibilityCache_.clear();

        // Update texture matrix for the depth compare
        updateTextureMatrix();

        // Make non-selected objects invisible and draw only the selected ones,
        // comparing against the depth buffer of the non-selected ones.
        changeVisibilityOfNonSelectedObjects(false);
        renderScene_->overrideMaterial = prepareMaskMaterial_;
        prepareMaskMaterial_->uniforms.at("cameraNearFar").setValue(Vector2(renderCamera_->nearPlane, renderCamera_->farPlane));
        prepareMaskMaterial_->uniforms.at("depthTexture").setValue(renderTargetDepthBuffer_->texture.get());
        prepareMaskMaterial_->uniforms.at("textureMatrix").setValue(textureMatrix_);
        renderer.setRenderTarget(renderTargetMaskBuffer_.get());
        renderer.clear();
        renderer.render(*renderScene_, *renderCamera_);
        changeVisibilityOfNonSelectedObjects(true);
        visibilityCache_.clear();
        selectionCache_.clear();

        renderScene_->background = currentBackground;
        renderScene_->overrideMaterial = currentOverrideMaterial;

        shadowMap.autoUpdate = oldShadowAutoUpdate;

        // 2. Downsample to half resolution
        fsQuad_->setMaterial(materialCopy_);
        materialCopy_->uniforms.at("tDiffuse").setValue(renderTargetMaskBuffer_->texture.get());
        renderer.setRenderTarget(renderTargetMaskDownSampleBuffer_.get());
        renderer.clear();
        fsQuad_->render(renderer);

        Color pulseColor1 = visibleEdgeColor;
        Color pulseColor2 = hiddenEdgeColor;

        if (pulsePeriod > 0) {

            const float scalar = (1.f + 0.25f) / 2.f + std::cos(nowMillis() * 0.01f / pulsePeriod) * (1.f - 0.25f) / 2.f;
            pulseColor1.multiplyScalar(scalar);
            pulseColor2.multiplyScalar(scalar);
        }

        // 3. Apply edge detection
        fsQuad_->setMaterial(edgeDetectionMaterial_);
        edgeDetectionMaterial_->uniforms.at("maskTexture").setValue(renderTargetMaskDownSampleBuffer_->texture.get());
        edgeDetectionMaterial_->uniforms.at("texSize").setValue(Vector2(static_cast<float>(renderTargetMaskDownSampleBuffer_->width),
                                                                        static_cast<float>(renderTargetMaskDownSampleBuffer_->height)));
        edgeDetectionMaterial_->uniforms.at("visibleEdgeColor").setValue(pulseColor1);
        edgeDetectionMaterial_->uniforms.at("hiddenEdgeColor").setValue(pulseColor2);
        renderer.setRenderTarget(renderTargetEdgeBuffer1_.get());
        renderer.clear();
        fsQuad_->render(renderer);

        // 4. Blur at half resolution
        fsQuad_->setMaterial(separableBlurMaterial1_);
        separableBlurMaterial1_->uniforms.at("colorTexture").setValue(renderTargetEdgeBuffer1_->texture.get());
        separableBlurMaterial1_->uniforms.at("direction").setValue(blurDirectionX);
        separableBlurMaterial1_->uniforms.at("kernelRadius").setValue(edgeThickness);
        renderer.setRenderTarget(renderTargetBlurBuffer1_.get());
        renderer.clear();
        fsQuad_->render(renderer);
        separableBlurMaterial1_->uniforms.at("colorTexture").setValue(renderTargetBlurBuffer1_->texture.get());
        separableBlurMaterial1_->uniforms.at("direction").setValue(blurDirectionY);
        renderer.setRenderTarget(renderTargetEdgeBuffer1_.get());
        renderer.clear();
        fsQuad_->render(renderer);

        // ... and at quarter resolution
        fsQuad_->setMaterial(separableBlurMaterial2_);
        separableBlurMaterial2_->uniforms.at("colorTexture").setValue(renderTargetEdgeBuffer1_->texture.get());
        separableBlurMaterial2_->uniforms.at("direction").setValue(blurDirectionX);
        renderer.setRenderTarget(renderTargetBlurBuffer2_.get());
        renderer.clear();
        fsQuad_->render(renderer);
        separableBlurMaterial2_->uniforms.at("colorTexture").setValue(renderTargetBlurBuffer2_->texture.get());
        separableBlurMaterial2_->uniforms.at("direction").setValue(blurDirectionY);
        renderer.setRenderTarget(renderTargetEdgeBuffer2_.get());
        renderer.clear();
        fsQuad_->render(renderer);

        // Blend it additively over the input
        fsQuad_->setMaterial(overlayMaterial_);
        overlayMaterial_->uniforms.at("maskTexture").setValue(renderTargetMaskBuffer_->texture.get());
        overlayMaterial_->uniforms.at("edgeTexture1").setValue(renderTargetEdgeBuffer1_->texture.get());
        overlayMaterial_->uniforms.at("edgeTexture2").setValue(renderTargetEdgeBuffer2_->texture.get());
        // A sampler must always have a texture behind it; without a pattern, a
        // white texel, which the shader turns into no fill at all.
        overlayMaterial_->uniforms.at("patternTexture").setValue(patternTexture ? patternTexture.get() : placeholderTexture_.get());
        overlayMaterial_->uniforms.at("edgeStrength").setValue(edgeStrength);
        overlayMaterial_->uniforms.at("edgeGlow").setValue(edgeGlow);
        overlayMaterial_->uniforms.at("usePatternTexture").setValue(usePatternTexture);

        if (maskActive) renderer.state().stencilBuffer.setTest(true);

        renderer.setRenderTarget(readBuffer);
        fsQuad_->render(renderer);

        renderer.setClearColor(oldClearColor, oldClearAlpha);
        renderer.autoClear = oldAutoClear;
    }

    if (renderToScreen && readBuffer) {

        fsQuad_->setMaterial(materialCopy_);
        materialCopy_->uniforms.at("tDiffuse").setValue(readBuffer->texture.get());
        renderer.setRenderTarget(nullptr);
        fsQuad_->render(renderer);
    }
}

void OutlinePass::updateSelectionCache() {

    selectionCache_.clear();

    for (auto* selected : selectedObjects) {

        if (!selected) continue;

        selected->traverse([this](Object3D& object) {
            if (object.is<Mesh>()) selectionCache_.insert(&object);
        });
    }
}

void OutlinePass::changeVisibilityOfSelectedObjects(bool visible) {

    for (auto* mesh : selectionCache_) {

        if (visible) {

            mesh->visible = visibilityCache_[mesh];

        } else {

            visibilityCache_[mesh] = mesh->visible;
            mesh->visible = false;
        }
    }
}

void OutlinePass::changeVisibilityOfNonSelectedObjects(bool visible) {

    renderScene_->traverse([&](Object3D& object) {

        if (object.is<Points>() || object.is<Line>()) {

            // The visibility of points and lines is always off during the
            // mask render, so they do not affect the outline.
            if (visible) {
                object.visible = visibilityCache_[&object];
            } else {
                visibilityCache_[&object] = object.visible;
                object.visible = false;
            }

        } else if (object.is<Mesh>() || object.is<Sprite>()) {

            // Only meshes and sprites are supported by OutlinePass.
            if (!selectionCache_.count(&object)) {

                if (visible) {
                    object.visible = visibilityCache_[&object];
                } else {
                    visibilityCache_[&object] = object.visible;
                    object.visible = false;
                }
            }
        }
    });
}

void OutlinePass::updateTextureMatrix() {

    textureMatrix_.set(0.5f, 0.f, 0.f, 0.5f,
                       0.f, 0.5f, 0.f, 0.5f,
                       0.f, 0.f, 0.5f, 0.5f,
                       0.f, 0.f, 0.f, 1.f);
    textureMatrix_.multiply(renderCamera_->projectionMatrix);
    textureMatrix_.multiply(renderCamera_->matrixWorldInverse);
}

OutlinePass::~OutlinePass() = default;
