
#include "threepp/postprocessing/UnrealBloomPass.hpp"

#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/postprocessing/shaders/CopyShader.hpp"
#include "threepp/postprocessing/shaders/LuminosityHighPassShader.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"

#include <algorithm>
#include <cmath>
#include <string>

using namespace threepp;

namespace {

    const Vector2 blurDirectionX{1.f, 0.f};
    const Vector2 blurDirectionY{0.f, 1.f};

    RenderTarget::Options bloomTargetOptions() {

        RenderTarget::Options options;
        options.minFilter = Filter::Linear;
        options.magFilter = Filter::Linear;
        options.format = Format::RGBA;
        options.generateMipmaps = false;
        options.depthBuffer = false;
        options.stencilBuffer = false;

        return options;
    }

    std::unique_ptr<RenderTarget> makeTarget(unsigned int width, unsigned int height, const std::string& name) {

        auto target = RenderTarget::create(width, height, bloomTargetOptions());
        target->texture->name = name;

        return target;
    }

    // Separable Gaussian, one axis per draw. The kernel radius is a define
    // rather than a uniform so the loop unrolls — it grows with the mip index,
    // which is what makes the largest mip the widest blur in screen space.
    std::shared_ptr<ShaderMaterial> blurMaterial(int kernelRadius) {

        auto material = ShaderMaterial::create();

        material->defines["KERNEL_RADIUS"] = std::to_string(kernelRadius);
        material->defines["SIGMA"] = std::to_string(kernelRadius);

        material->uniforms = UniformMap{
                {"colorTexture", Uniform()},
                {"texSize", Uniform(Vector2(0.5f, 0.5f))},
                {"direction", Uniform(Vector2(0.5f, 0.5f))}};

        material->vertexShader = R"(
                varying vec2 vUv;

                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                })";

        material->fragmentShader = R"(
                #include <common>

                varying vec2 vUv;
                uniform sampler2D colorTexture;
                uniform vec2 texSize;
                uniform vec2 direction;

                float gaussianPdf(in float x, in float sigma) {
                    return 0.39894 * exp( -0.5 * x * x/( sigma * sigma))/sigma;
                }

                void main() {
                    vec2 invSize = 1.0 / texSize;
                    float fSigma = float(SIGMA);
                    float weightSum = gaussianPdf(0.0, fSigma);
                    vec3 diffuseSum = texture2D( colorTexture, vUv).rgb * weightSum;
                    for( int i = 1; i < KERNEL_RADIUS; i ++ ) {
                        float x = float(i);
                        float w = gaussianPdf(x, fSigma);
                        vec2 uvOffset = direction * invSize * x;
                        vec3 sample1 = texture2D( colorTexture, vUv + uvOffset).rgb;
                        vec3 sample2 = texture2D( colorTexture, vUv - uvOffset).rgb;
                        diffuseSum += (sample1 + sample2) * w;
                        weightSum += 2.0 * w;
                    }
                    gl_FragColor = vec4(diffuseSum/weightSum, 1.0);
                })";

        material->depthTest = false;
        material->depthWrite = false;

        return material;
    }

    std::shared_ptr<ShaderMaterial> compositeMaterial() {

        auto material = ShaderMaterial::create();

        material->defines["NUM_MIPS"] = "5";

        material->uniforms = UniformMap{
                {"blurTexture1", Uniform()},
                {"blurTexture2", Uniform()},
                {"blurTexture3", Uniform()},
                {"blurTexture4", Uniform()},
                {"blurTexture5", Uniform()},
                {"bloomStrength", Uniform(1.f)},
                {"bloomRadius", Uniform(0.f)},
                {"bloomFactors", Uniform(std::vector<float>{1.f, 0.8f, 0.6f, 0.4f, 0.2f})},
                {"bloomTintColors", Uniform(std::vector<Vector3>(5, Vector3(1, 1, 1)))}};

        material->vertexShader = R"(
                varying vec2 vUv;

                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                })";

        material->fragmentShader = R"(
                varying vec2 vUv;
                uniform sampler2D blurTexture1;
                uniform sampler2D blurTexture2;
                uniform sampler2D blurTexture3;
                uniform sampler2D blurTexture4;
                uniform sampler2D blurTexture5;
                uniform float bloomStrength;
                uniform float bloomRadius;
                uniform float bloomFactors[NUM_MIPS];
                uniform vec3 bloomTintColors[NUM_MIPS];

                float lerpBloomFactor(const in float factor) {
                    float mirrorFactor = 1.2 - factor;
                    return mix(factor, mirrorFactor, bloomRadius);
                }

                void main() {
                    gl_FragColor = bloomStrength * ( lerpBloomFactor(bloomFactors[0]) * vec4(bloomTintColors[0], 1.0) * texture2D(blurTexture1, vUv) +
                        lerpBloomFactor(bloomFactors[1]) * vec4(bloomTintColors[1], 1.0) * texture2D(blurTexture2, vUv) +
                        lerpBloomFactor(bloomFactors[2]) * vec4(bloomTintColors[2], 1.0) * texture2D(blurTexture3, vUv) +
                        lerpBloomFactor(bloomFactors[3]) * vec4(bloomTintColors[3], 1.0) * texture2D(blurTexture4, vUv) +
                        lerpBloomFactor(bloomFactors[4]) * vec4(bloomTintColors[4], 1.0) * texture2D(blurTexture5, vUv) );
                })";

        material->depthTest = false;
        material->depthWrite = false;

        return material;
    }

}// namespace


UnrealBloomPass::UnrealBloomPass(const Vector2& resolution, float strength, float radius, float threshold)
    : strength(strength), radius(radius), threshold(threshold) {

    tintColors.fill(Vector3(1, 1, 1));

    needsSwap = false;

    auto resx = static_cast<unsigned int>(std::max(1.f, std::round(resolution.x / 2.f)));
    auto resy = static_cast<unsigned int>(std::max(1.f, std::round(resolution.y / 2.f)));

    renderTargetBright_ = makeTarget(resx, resy, "UnrealBloomPass.bright");

    for (int i = 0; i < nMips; i++) {

        renderTargetsHorizontal_.emplace_back(makeTarget(resx, resy, "UnrealBloomPass.h" + std::to_string(i)));
        renderTargetsVertical_.emplace_back(makeTarget(resx, resy, "UnrealBloomPass.v" + std::to_string(i)));

        resx = std::max(1u, static_cast<unsigned int>(std::round(static_cast<float>(resx) / 2.f)));
        resy = std::max(1u, static_cast<unsigned int>(std::round(static_cast<float>(resy) / 2.f)));
    }

    const auto highPass = shaders::luminosityHighPassShader();
    highPassMaterial_ = ShaderMaterial::create();
    highPassMaterial_->uniforms = highPass.uniforms;
    highPassMaterial_->vertexShader = highPass.vertexShader;
    highPassMaterial_->fragmentShader = highPass.fragmentShader;
    highPassMaterial_->uniforms.at("luminosityThreshold").setValue(threshold);
    highPassMaterial_->uniforms.at("smoothWidth").setValue(0.01f);
    highPassMaterial_->depthTest = false;
    highPassMaterial_->depthWrite = false;

    const std::array<int, nMips> kernelSizes{3, 5, 7, 9, 11};
    for (int i = 0; i < nMips; i++) {
        blurMaterials_.emplace_back(blurMaterial(kernelSizes[i]));
    }

    compositeMaterial_ = compositeMaterial();

    // The bloom is added to the image, not blended with it: light is additive,
    // and the pass has no opinion about what was already there.
    const auto copy = shaders::copyShader();
    blendMaterial_ = ShaderMaterial::create();
    blendMaterial_->uniforms = copy.uniforms;
    blendMaterial_->vertexShader = copy.vertexShader;
    blendMaterial_->fragmentShader = copy.fragmentShader;
    blendMaterial_->blending = Blending::Additive;
    blendMaterial_->transparent = true;
    blendMaterial_->depthTest = false;
    blendMaterial_->depthWrite = false;

    basicMaterial_ = MeshBasicMaterial::create();
    basicMaterial_->depthTest = false;
    basicMaterial_->depthWrite = false;

    fsQuad_ = std::make_unique<FullScreenQuad>(highPassMaterial_);

    // setSize does the rest; the composer calls it when the pass is added.
    setSize(static_cast<unsigned int>(resolution.x), static_cast<unsigned int>(resolution.y));
}

void UnrealBloomPass::setSize(unsigned int width, unsigned int height) {

    auto resx = std::max(1u, static_cast<unsigned int>(std::round(static_cast<float>(width) / 2.f)));
    auto resy = std::max(1u, static_cast<unsigned int>(std::round(static_cast<float>(height) / 2.f)));

    renderTargetBright_->setSize(resx, resy);

    for (int i = 0; i < nMips; i++) {

        renderTargetsHorizontal_[i]->setSize(resx, resy);
        renderTargetsVertical_[i]->setSize(resx, resy);

        blurMaterials_[i]->uniforms.at("texSize").setValue(Vector2(static_cast<float>(resx), static_cast<float>(resy)));

        resx = std::max(1u, static_cast<unsigned int>(std::round(static_cast<float>(resx) / 2.f)));
        resy = std::max(1u, static_cast<unsigned int>(std::round(static_cast<float>(resy) / 2.f)));
    }
}

void UnrealBloomPass::render(GLRenderer& renderer, RenderTarget*, RenderTarget* readBuffer, float, bool maskActive) {

    Color oldClearColor;
    renderer.getClearColor(oldClearColor);
    const float oldClearAlpha = renderer.getClearAlpha();
    const bool oldAutoClear = renderer.autoClear;

    renderer.autoClear = false;
    renderer.setClearColor(Color(0, 0, 0), 0);

    // The mip chain is the pass's own business; a mask applies to where the
    // result lands, not to the intermediate targets.
    if (maskActive) renderer.state().stencilBuffer.setTest(false);

    if (renderToScreen) {

        // Nothing has put the unbloomed image on screen yet, and the blend
        // below only adds to what is there.
        basicMaterial_->map = readBuffer ? readBuffer->texture : nullptr;
        fsQuad_->setMaterial(basicMaterial_);

        renderer.setRenderTarget(nullptr);
        renderer.clear();
        fsQuad_->render(renderer);
    }

    // 1. Extract the bright areas.

    highPassMaterial_->uniforms.at("tDiffuse").setValue(readBuffer ? readBuffer->texture.get() : static_cast<Texture*>(nullptr));
    highPassMaterial_->uniforms.at("luminosityThreshold").setValue(threshold);
    fsQuad_->setMaterial(highPassMaterial_);

    renderer.setRenderTarget(renderTargetBright_.get());
    renderer.clear();
    fsQuad_->render(renderer);

    // 2. Blur each mip in turn, feeding the previous one in — so mip n has
    //    been through n+1 blurs by the time it is composited.

    RenderTarget* input = renderTargetBright_.get();

    for (int i = 0; i < nMips; i++) {

        fsQuad_->setMaterial(blurMaterials_[i]);

        blurMaterials_[i]->uniforms.at("colorTexture").setValue(input->texture.get());
        blurMaterials_[i]->uniforms.at("direction").setValue(blurDirectionX);
        renderer.setRenderTarget(renderTargetsHorizontal_[i].get());
        renderer.clear();
        fsQuad_->render(renderer);

        blurMaterials_[i]->uniforms.at("colorTexture").setValue(renderTargetsHorizontal_[i]->texture.get());
        blurMaterials_[i]->uniforms.at("direction").setValue(blurDirectionY);
        renderer.setRenderTarget(renderTargetsVertical_[i].get());
        renderer.clear();
        fsQuad_->render(renderer);

        input = renderTargetsVertical_[i].get();
    }

    // 3. Composite the mips.

    fsQuad_->setMaterial(compositeMaterial_);
    compositeMaterial_->uniforms.at("bloomStrength").setValue(strength);
    compositeMaterial_->uniforms.at("bloomRadius").setValue(radius);
    compositeMaterial_->uniforms.at("bloomTintColors").setValue(std::vector<Vector3>(tintColors.begin(), tintColors.end()));
    for (int i = 0; i < nMips; i++) {
        compositeMaterial_->uniforms.at("blurTexture" + std::to_string(i + 1)).setValue(renderTargetsVertical_[i]->texture.get());
    }

    renderer.setRenderTarget(renderTargetsHorizontal_[0].get());
    renderer.clear();
    fsQuad_->render(renderer);

    // 4. Add it back over the input.

    fsQuad_->setMaterial(blendMaterial_);
    blendMaterial_->uniforms.at("tDiffuse").setValue(renderTargetsHorizontal_[0]->texture.get());

    if (maskActive) renderer.state().stencilBuffer.setTest(true);

    renderer.setRenderTarget(renderToScreen ? nullptr : readBuffer);
    fsQuad_->render(renderer);

    renderer.setClearColor(oldClearColor, oldClearAlpha);
    renderer.autoClear = oldAutoClear;
}

UnrealBloomPass::~UnrealBloomPass() = default;
