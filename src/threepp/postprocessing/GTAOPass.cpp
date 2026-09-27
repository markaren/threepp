
#include "threepp/postprocessing/GTAOPass.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/materials/MeshNormalMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/postprocessing/shaders/CopyShader.hpp"
#include "threepp/postprocessing/shaders/GTAOShader.hpp"
#include "threepp/postprocessing/shaders/PoissonDenoiseShader.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/gl/GLShadowMap.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/textures/DepthTexture.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

using namespace threepp;

namespace {

    // --- three.js r186 GTAOShader.js generateMagicSquareNoise ---------------

    std::vector<int> generateMagicSquare(int size) {

        const int noiseSize = size % 2 == 0 ? size + 1 : size;
        const int noiseSquareSize = noiseSize * noiseSize;
        std::vector<int> magicSquare(noiseSquareSize, 0);
        int i = noiseSize / 2;
        int j = noiseSize - 1;

        for (int num = 1; num <= noiseSquareSize;) {

            if (i == -1 && j == noiseSize) {

                j = noiseSize - 2;
                i = 0;

            } else {

                if (j == noiseSize) j = 0;
                if (i < 0) i = noiseSize - 1;
            }

            if (magicSquare[i * noiseSize + j] != 0) {

                j -= 2;
                i++;
                continue;
            }

            magicSquare[i * noiseSize + j] = num++;

            j++;
            i--;
        }

        return magicSquare;
    }

    std::shared_ptr<Texture> generateMagicSquareNoise(int size = 5) {

        const int noiseSize = size % 2 == 0 ? size + 1 : size;
        const auto magicSquare = generateMagicSquare(noiseSize);
        const auto noiseSquareSize = static_cast<int>(magicSquare.size());
        std::vector<unsigned char> data(static_cast<size_t>(noiseSquareSize) * 4);

        for (int inx = 0; inx < noiseSquareSize; ++inx) {

            const double angle = (2 * math::PI * magicSquare[inx]) / noiseSquareSize;
            const double x = std::cos(angle), y = std::sin(angle);
            data[inx * 4] = static_cast<unsigned char>((x * 0.5 + 0.5) * 255);
            data[inx * 4 + 1] = static_cast<unsigned char>((y * 0.5 + 0.5) * 255);
            data[inx * 4 + 2] = 127;
            data[inx * 4 + 3] = 255;
        }

        auto texture = DataTexture::create(ImageData{std::move(data)}, noiseSize, noiseSize);
        texture->wrapS = TextureWrapping::Repeat;
        texture->wrapT = TextureWrapping::Repeat;
        texture->needsUpdate();

        return texture;
    }

    // --- three.js r186 SimplexNoise.js, 2D only ------------------------------
    //
    // three.js fills the permutation table from Math.random, so its denoise
    // rotation differs every page load. A fixed-seed generator gives the same
    // table every run instead: frames are reproducible, the statistics the
    // same.
    class SimplexNoise2D {

    public:
        explicit SimplexNoise2D(std::uint32_t seed = 0x9e3779b9u) {

            std::array<int, 256> p{};
            std::uint32_t state = seed;
            for (int& v : p) {
                state = state * 1664525u + 1013904223u;
                v = static_cast<int>((state >> 8) & 255u);
            }
            for (int i = 0; i < 512; i++) perm_[i] = p[i & 255];
        }

        [[nodiscard]] double noise(double xin, double yin) const {

            static constexpr int grad3[12][3] = {{1, 1, 0}, {-1, 1, 0}, {1, -1, 0}, {-1, -1, 0}, {1, 0, 1}, {-1, 0, 1}, {1, 0, -1}, {-1, 0, -1}, {0, 1, 1}, {0, -1, 1}, {0, 1, -1}, {0, -1, -1}};

            double n0, n1, n2;
            const double F2 = 0.5 * (std::sqrt(3.0) - 1.0);
            const double s = (xin + yin) * F2;
            const int i = static_cast<int>(std::floor(xin + s));
            const int j = static_cast<int>(std::floor(yin + s));
            const double G2 = (3.0 - std::sqrt(3.0)) / 6.0;
            const double t = (i + j) * G2;
            const double x0 = xin - (i - t);
            const double y0 = yin - (j - t);

            const int i1 = x0 > y0 ? 1 : 0;
            const int j1 = x0 > y0 ? 0 : 1;

            const double x1 = x0 - i1 + G2;
            const double y1 = y0 - j1 + G2;
            const double x2 = x0 - 1.0 + 2.0 * G2;
            const double y2 = y0 - 1.0 + 2.0 * G2;
            const int ii = i & 255;
            const int jj = j & 255;
            const int gi0 = perm_[ii + perm_[jj]] % 12;
            const int gi1 = perm_[ii + i1 + perm_[jj + j1]] % 12;
            const int gi2 = perm_[ii + 1 + perm_[jj + 1]] % 12;

            double t0 = 0.5 - x0 * x0 - y0 * y0;
            if (t0 < 0) n0 = 0.0;
            else {
                t0 *= t0;
                n0 = t0 * t0 * (grad3[gi0][0] * x0 + grad3[gi0][1] * y0);
            }

            double t1 = 0.5 - x1 * x1 - y1 * y1;
            if (t1 < 0) n1 = 0.0;
            else {
                t1 *= t1;
                n1 = t1 * t1 * (grad3[gi1][0] * x1 + grad3[gi1][1] * y1);
            }

            double t2 = 0.5 - x2 * x2 - y2 * y2;
            if (t2 < 0) n2 = 0.0;
            else {
                t2 *= t2;
                n2 = t2 * t2 * (grad3[gi2][0] * x2 + grad3[gi2][1] * y2);
            }

            return 70.0 * (n0 + n1 + n2);
        }

    private:
        std::array<int, 512> perm_{};
    };

    // GTAOPass._generateNoise: four decorrelated simplex channels.
    std::shared_ptr<Texture> generatePdNoise(int size = 64) {

        const SimplexNoise2D simplex;
        std::vector<unsigned char> data(static_cast<size_t>(size) * size * 4);

        const auto toByte = [](double n) { return static_cast<unsigned char>((n * 0.5 + 0.5) * 255); };

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                const double x = i, y = j;
                const size_t k = (static_cast<size_t>(i) * size + j) * 4;
                data[k] = toByte(simplex.noise(x, y));
                data[k + 1] = toByte(simplex.noise(x + size, y));
                data[k + 2] = toByte(simplex.noise(x, y + size));
                data[k + 3] = toByte(simplex.noise(x + size, y + size));
            }
        }

        auto texture = DataTexture::create(ImageData{std::move(data)}, size, size);
        texture->wrapS = TextureWrapping::Repeat;
        texture->wrapT = TextureWrapping::Repeat;
        texture->needsUpdate();

        return texture;
    }

    std::shared_ptr<ShaderMaterial> materialFrom(const Shader& shader) {

        auto material = ShaderMaterial::create();
        material->uniforms = shader.uniforms;
        material->vertexShader = shader.vertexShader;
        material->fragmentShader = shader.fragmentShader;

        return material;
    }

    std::unique_ptr<RenderTarget> aoTarget(unsigned int w, unsigned int h, const std::string& name) {

        RenderTarget::Options options;
        options.type = Type::HalfFloat;
        options.depthBuffer = false;
        options.stencilBuffer = false;
        options.generateMipmaps = false;

        auto target = RenderTarget::create(w, h, options);
        target->texture->name = name;

        return target;
    }

    template<class T>
    void setMatrix(ShaderMaterial& material, const std::string& name, const T& value) {

        material.uniforms.at(name).setValue(value);
    }

}// namespace


GTAOPass::GTAOPass(Scene& scene, Camera& camera, unsigned int width, unsigned int height,
                   const std::optional<GTAOParameters>& aoParameters,
                   const std::optional<PoissonDenoiseParameters>& pdParameters)
    : scene_(&scene), camera_(&camera), width_(std::max(1u, width)), height_(std::max(1u, height)) {

    clear = true;

    const bool perspective = dynamic_cast<OrthographicCamera*>(&camera) == nullptr;

    gtaoNoiseTexture_ = generateMagicSquareNoise();
    pdNoiseTexture_ = generatePdNoise();

    gtaoRenderTarget_ = aoTarget(width_, height_, "GTAOPass.ao");
    pdRenderTarget_ = aoTarget(width_, height_, "GTAOPass.denoise");

    gtaoMaterial_ = materialFrom(shaders::gtaoShader());
    gtaoMaterial_->defines = shaders::gtaoShaderDefines();
    gtaoMaterial_->defines["PERSPECTIVE_CAMERA"] = perspective ? "1" : "0";
    gtaoMaterial_->blending = Blending::None;
    gtaoMaterial_->depthTest = false;
    gtaoMaterial_->depthWrite = false;
    gtaoMaterial_->uniforms.at("tNoise").setValue(gtaoNoiseTexture_.get());
    gtaoMaterial_->uniforms.at("resolution").setValue(Vector2(static_cast<float>(width_), static_cast<float>(height_)));
    gtaoMaterial_->uniforms.at("cameraNear").setValue(camera.nearPlane);
    gtaoMaterial_->uniforms.at("cameraFar").setValue(camera.farPlane);

    normalMaterial_ = MeshNormalMaterial::create();
    normalMaterial_->blending = Blending::None;

    pdMaterial_ = materialFrom(shaders::poissonDenoiseShader());
    pdMaterial_->defines = shaders::poissonDenoiseShaderDefines();
    pdMaterial_->depthTest = false;
    pdMaterial_->depthWrite = false;
    pdMaterial_->uniforms.at("tDiffuse").setValue(gtaoRenderTarget_->texture.get());
    pdMaterial_->uniforms.at("tNoise").setValue(pdNoiseTexture_.get());
    pdMaterial_->uniforms.at("resolution").setValue(Vector2(static_cast<float>(width_), static_cast<float>(height_)));
    pdMaterial_->uniforms.at("lumaPhi").setValue(10.f);
    pdMaterial_->uniforms.at("depthPhi").setValue(2.f);
    pdMaterial_->uniforms.at("normalPhi").setValue(3.f);
    pdMaterial_->uniforms.at("radius").setValue(8.f);

    depthRenderMaterial_ = materialFrom(shaders::gtaoDepthShader());
    // three.js leaves this define at 1 for every camera; following the camera
    // is what makes the Depth view right for an orthographic one.
    depthRenderMaterial_->defines["PERSPECTIVE_CAMERA"] = perspective ? "1" : "0";
    depthRenderMaterial_->blending = Blending::None;
    depthRenderMaterial_->uniforms.at("cameraNear").setValue(camera.nearPlane);
    depthRenderMaterial_->uniforms.at("cameraFar").setValue(camera.farPlane);

    const auto copy = shaders::copyShader();
    copyMaterial_ = materialFrom(copy);
    copyMaterial_->transparent = true;
    copyMaterial_->depthTest = false;
    copyMaterial_->depthWrite = false;
    copyMaterial_->blending = Blending::None;

    // AO multiplies what is already there: result = src * dst.
    blendMaterial_ = materialFrom(shaders::gtaoBlendShader());
    blendMaterial_->transparent = true;
    blendMaterial_->depthTest = false;
    blendMaterial_->depthWrite = false;
    blendMaterial_->blending = Blending::Custom;
    blendMaterial_->blendSrc = BlendFactor::DstColor;
    blendMaterial_->blendDst = BlendFactor::Zero;
    blendMaterial_->blendEquation = BlendEquation::Add;
    blendMaterial_->blendSrcAlpha = BlendFactor::DstAlpha;
    blendMaterial_->blendDstAlpha = BlendFactor::Zero;
    blendMaterial_->blendEquationAlpha = BlendEquation::Add;

    fsQuad_ = std::make_unique<FullScreenQuad>(copyMaterial_);

    // setGBuffer() with no arguments: the pass renders its own normals and
    // depth. Depth-only 24-bit rather than three.js's depth-stencil - the
    // stencil half was never read.
    depthTexture_ = DepthTexture::create(Type::UnsignedInt);
    RenderTarget::Options gbufferOptions;
    gbufferOptions.minFilter = Filter::Nearest;
    gbufferOptions.magFilter = Filter::Nearest;
    gbufferOptions.type = Type::HalfFloat;
    gbufferOptions.generateMipmaps = false;
    gbufferOptions.stencilBuffer = false;
    gbufferOptions.depthTexture = depthTexture_;
    normalRenderTarget_ = RenderTarget::create(width_, height_, gbufferOptions);
    normalRenderTarget_->texture->name = "GTAOPass.normal";

    gtaoMaterial_->defines["NORMAL_VECTOR_TYPE"] = "1";
    gtaoMaterial_->defines["DEPTH_SWIZZLING"] = "x";
    gtaoMaterial_->uniforms.at("tNormal").setValue(normalRenderTarget_->texture.get());
    gtaoMaterial_->uniforms.at("tDepth").setValue(static_cast<Texture*>(depthTexture_.get()));

    pdMaterial_->defines["NORMAL_VECTOR_TYPE"] = "1";
    pdMaterial_->defines["DEPTH_SWIZZLING"] = "x";
    pdMaterial_->uniforms.at("tNormal").setValue(normalRenderTarget_->texture.get());
    pdMaterial_->uniforms.at("tDepth").setValue(static_cast<Texture*>(depthTexture_.get()));

    depthRenderMaterial_->uniforms.at("tDepth").setValue(static_cast<Texture*>(depthTexture_.get()));

    if (aoParameters) updateGtaoMaterial(*aoParameters);
    if (pdParameters) updatePdMaterial(*pdParameters);
}

void GTAOPass::setSize(unsigned int width, unsigned int height) {

    width_ = std::max(1u, width);
    height_ = std::max(1u, height);

    gtaoRenderTarget_->setSize(width_, height_);
    normalRenderTarget_->setSize(width_, height_);
    pdRenderTarget_->setSize(width_, height_);

    gtaoMaterial_->uniforms.at("resolution").setValue(Vector2(static_cast<float>(width_), static_cast<float>(height_)));
    setMatrix(*gtaoMaterial_, "cameraProjectionMatrix", camera_->projectionMatrix);
    setMatrix(*gtaoMaterial_, "cameraProjectionMatrixInverse", camera_->projectionMatrixInverse);

    pdMaterial_->uniforms.at("resolution").setValue(Vector2(static_cast<float>(width_), static_cast<float>(height_)));
    setMatrix(*pdMaterial_, "cameraProjectionMatrixInverse", camera_->projectionMatrixInverse);
}

Texture* GTAOPass::gtaoMap() const {

    return pdRenderTarget_->texture.get();
}

void GTAOPass::setSceneClipBox(const std::optional<Box3>& box) {

    if (box) {

        if (gtaoMaterial_->defines["SCENE_CLIP_BOX"] != "1") gtaoMaterial_->needsUpdate();
        gtaoMaterial_->defines["SCENE_CLIP_BOX"] = "1";
        gtaoMaterial_->uniforms.at("sceneBoxMin").setValue(box->min());
        gtaoMaterial_->uniforms.at("sceneBoxMax").setValue(box->max());

    } else {

        if (gtaoMaterial_->defines["SCENE_CLIP_BOX"] != "0") gtaoMaterial_->needsUpdate();
        gtaoMaterial_->defines["SCENE_CLIP_BOX"] = "0";
    }
}

void GTAOPass::updateGtaoMaterial(const GTAOParameters& parameters) {

    auto& u = gtaoMaterial_->uniforms;
    auto& defines = gtaoMaterial_->defines;

    if (parameters.radius) u.at("radius").setValue(*parameters.radius);
    if (parameters.distanceExponent) u.at("distanceExponent").setValue(*parameters.distanceExponent);
    if (parameters.thickness) u.at("thickness").setValue(*parameters.thickness);
    if (parameters.distanceFallOff) u.at("distanceFallOff").setValue(*parameters.distanceFallOff);
    if (parameters.scale) u.at("scale").setValue(*parameters.scale);

    if (parameters.samples && std::to_string(*parameters.samples) != defines["SAMPLES"]) {

        defines["SAMPLES"] = std::to_string(std::max(1, *parameters.samples));
        gtaoMaterial_->needsUpdate();
    }

    if (parameters.screenSpaceRadius) {

        const std::string value = *parameters.screenSpaceRadius ? "1" : "0";
        if (value != defines["SCREEN_SPACE_RADIUS"]) {
            defines["SCREEN_SPACE_RADIUS"] = value;
            gtaoMaterial_->needsUpdate();
        }
    }
}

void GTAOPass::updatePdMaterial(const PoissonDenoiseParameters& parameters) {

    bool updateShader = false;
    auto& u = pdMaterial_->uniforms;

    if (parameters.lumaPhi) u.at("lumaPhi").setValue(*parameters.lumaPhi);
    if (parameters.depthPhi) u.at("depthPhi").setValue(*parameters.depthPhi);
    if (parameters.normalPhi) u.at("normalPhi").setValue(*parameters.normalPhi);
    if (parameters.radius) u.at("radius").setValue(*parameters.radius);

    if (parameters.radiusExponent && *parameters.radiusExponent != pdRadiusExponent_) {
        pdRadiusExponent_ = *parameters.radiusExponent;
        updateShader = true;
    }

    if (parameters.rings && *parameters.rings != pdRings_) {
        pdRings_ = *parameters.rings;
        updateShader = true;
    }

    if (parameters.samples && *parameters.samples != pdSamples_) {
        pdSamples_ = std::max(2, *parameters.samples);
        updateShader = true;
    }

    if (updateShader) {

        pdMaterial_->defines["SAMPLES"] = std::to_string(pdSamples_);
        pdMaterial_->defines["SAMPLE_VECTORS"] = shaders::generatePdSamplePointInitializer(pdSamples_, pdRings_, pdRadiusExponent_);
        pdMaterial_->needsUpdate();
    }
}

void GTAOPass::render(GLRenderer& renderer, RenderTarget* writeBuffer, RenderTarget* readBuffer, float, bool) {

    // Off leaves the image in readBuffer, so there is nothing to swap to.
    // (three.js swaps anyway, onto whatever the write buffer last held.)
    needsSwap = output != Output::Off;
    if (output == Output::Off) return;

    // Render normals and depth (only meshes; points and lines do not
    // contribute to AO).
    overrideVisibility();
    renderOverride(renderer, normalRenderTarget_.get(), 0x7777ff, 1.f);
    restoreVisibility();

    // AO
    gtaoMaterial_->uniforms.at("cameraNear").setValue(camera_->nearPlane);
    gtaoMaterial_->uniforms.at("cameraFar").setValue(camera_->farPlane);
    setMatrix(*gtaoMaterial_, "cameraProjectionMatrix", camera_->projectionMatrix);
    setMatrix(*gtaoMaterial_, "cameraProjectionMatrixInverse", camera_->projectionMatrixInverse);
    setMatrix(*gtaoMaterial_, "cameraWorldMatrix", *camera_->matrixWorld);
    renderPass(renderer, gtaoMaterial_, gtaoRenderTarget_.get(), 0xffffff, 1.f);

    // Poisson denoise
    setMatrix(*pdMaterial_, "cameraProjectionMatrixInverse", camera_->projectionMatrixInverse);
    renderPass(renderer, pdMaterial_, pdRenderTarget_.get(), 0xffffff, 1.f);

    RenderTarget* target = renderToScreen ? nullptr : writeBuffer;
    const auto copyOf = [&](Texture* texture) {
        copyMaterial_->uniforms.at("tDiffuse").setValue(texture);
        copyMaterial_->blending = Blending::None;
        renderPass(renderer, copyMaterial_, target);
    };

    switch (output) {

        case Output::Diffuse:
            copyOf(readBuffer ? readBuffer->texture.get() : nullptr);
            break;

        case Output::AO:
            copyOf(gtaoRenderTarget_->texture.get());
            break;

        case Output::Denoise:
            copyOf(pdRenderTarget_->texture.get());
            break;

        case Output::Depth:
            depthRenderMaterial_->uniforms.at("cameraNear").setValue(camera_->nearPlane);
            depthRenderMaterial_->uniforms.at("cameraFar").setValue(camera_->farPlane);
            renderPass(renderer, depthRenderMaterial_, target);
            break;

        case Output::Normal:
            copyOf(normalRenderTarget_->texture.get());
            break;

        case Output::Default:
            copyOf(readBuffer ? readBuffer->texture.get() : nullptr);

            blendMaterial_->uniforms.at("intensity").setValue(blendIntensity);
            blendMaterial_->uniforms.at("tDiffuse").setValue(pdRenderTarget_->texture.get());
            renderPass(renderer, blendMaterial_, target);
            break;

        default:
            std::cerr << "THREE.GTAOPass: Unknown output type." << std::endl;
    }
}

void GTAOPass::renderPass(GLRenderer& renderer, const std::shared_ptr<ShaderMaterial>& material,
                          RenderTarget* target, std::optional<unsigned int> clearColor, float clearAlpha) {

    Color originalClearColor;
    renderer.getClearColor(originalClearColor);
    const float originalClearAlpha = renderer.getClearAlpha();
    const bool originalAutoClear = renderer.autoClear;

    renderer.setRenderTarget(target);

    renderer.autoClear = false;
    if (clearColor) {

        renderer.setClearColor(Color(*clearColor), clearAlpha);
        renderer.clear();
    }

    fsQuad_->setMaterial(material);
    fsQuad_->render(renderer);

    renderer.autoClear = originalAutoClear;
    renderer.setClearColor(originalClearColor, originalClearAlpha);
}

void GTAOPass::renderOverride(GLRenderer& renderer, RenderTarget* target, unsigned int clearColor, float clearAlpha) {

    Color originalClearColor;
    renderer.getClearColor(originalClearColor);
    const float originalClearAlpha = renderer.getClearAlpha();
    const bool originalAutoClear = renderer.autoClear;

    renderer.setRenderTarget(target);
    renderer.autoClear = false;

    renderer.setClearColor(Color(clearColor), clearAlpha);
    renderer.clear();

    // Normals do not read shadow maps, and the RenderPass has already drawn
    // them this frame. three.js redraws them here; the image is the same.
    auto& shadowMap = renderer.shadowMap();
    const bool oldShadowAutoUpdate = shadowMap.autoUpdate;
    shadowMap.autoUpdate = false;

    const auto oldOverrideMaterial = scene_->overrideMaterial;
    scene_->overrideMaterial = normalMaterial_;
    renderer.render(*scene_, *camera_);
    scene_->overrideMaterial = oldOverrideMaterial;

    shadowMap.autoUpdate = oldShadowAutoUpdate;

    renderer.autoClear = originalAutoClear;
    renderer.setClearColor(originalClearColor, originalClearAlpha);
}

void GTAOPass::overrideVisibility() {

    scene_->traverse([this](Object3D& object) {
        if ((object.is<Points>() || object.is<Line>()) && object.visible) {
            object.visible = false;
            visibilityCache_.push_back(&object);
        }
    });
}

void GTAOPass::restoreVisibility() {

    for (auto* object : visibilityCache_) object->visible = true;
    visibilityCache_.clear();
}

GTAOPass::~GTAOPass() = default;
