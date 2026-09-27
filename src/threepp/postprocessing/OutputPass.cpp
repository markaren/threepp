
#include "threepp/postprocessing/OutputPass.hpp"

#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"

#include <optional>

using namespace threepp;

namespace {

    // three.js r186 OutputShader. The curves come from the shared
    // <tonemapping_pars_fragment> chunk, selected by define, and the sRGB
    // encode from the encodings chunk every ShaderMaterial is compiled with;
    // nothing is duplicated here.
    //
    // toneMapped is off on the material, so the renderer neither compiles its
    // own toneMapping() in nor, since this pass may draw into a target, has
    // any say in whether the curve runs: the defines decide, as in three.js.
    std::shared_ptr<ShaderMaterial> makeMaterial() {

        auto material = ShaderMaterial::create();
        material->name = "OutputShader";
        material->uniforms = UniformMap{
                {"tDiffuse", Uniform()},
                {"toneMappingExposure", Uniform(1.f)}};
        material->vertexShader = R"(
            varying vec2 vUv;

            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
            })";
        material->fragmentShader = R"(
            uniform sampler2D tDiffuse;
            varying vec2 vUv;

            #include <tonemapping_pars_fragment>

            void main() {

                gl_FragColor = texture2D( tDiffuse, vUv );

                #ifdef LINEAR_TONE_MAPPING
                    gl_FragColor.rgb = LinearToneMapping( gl_FragColor.rgb );
                #elif defined( REINHARD_TONE_MAPPING )
                    gl_FragColor.rgb = ReinhardToneMapping( gl_FragColor.rgb );
                #elif defined( CINEON_TONE_MAPPING )
                    gl_FragColor.rgb = OptimizedCineonToneMapping( gl_FragColor.rgb );
                #elif defined( ACES_FILMIC_TONE_MAPPING )
                    gl_FragColor.rgb = ACESFilmicToneMapping( gl_FragColor.rgb );
                #elif defined( AGX_TONE_MAPPING )
                    gl_FragColor.rgb = AgXToneMapping( gl_FragColor.rgb );
                #elif defined( NEUTRAL_TONE_MAPPING )
                    gl_FragColor.rgb = NeutralToneMapping( gl_FragColor.rgb );
                #elif defined( CUSTOM_TONE_MAPPING )
                    gl_FragColor.rgb = CustomToneMapping( gl_FragColor.rgb );
                #endif

                #ifdef SRGB_TRANSFER
                    gl_FragColor = LinearTosRGB( gl_FragColor );
                #endif
            })";
        material->toneMapped = false;
        material->depthTest = false;
        material->depthWrite = false;

        return material;
    }

    const char* toneMappingDefine(ToneMapping toneMapping) {

        switch (toneMapping) {
            case ToneMapping::Linear: return "LINEAR_TONE_MAPPING";
            case ToneMapping::Reinhard: return "REINHARD_TONE_MAPPING";
            case ToneMapping::Cineon: return "CINEON_TONE_MAPPING";
            case ToneMapping::ACESFilmic: return "ACES_FILMIC_TONE_MAPPING";
            case ToneMapping::AgX: return "AGX_TONE_MAPPING";
            case ToneMapping::Neutral: return "NEUTRAL_TONE_MAPPING";
            case ToneMapping::Custom: return "CUSTOM_TONE_MAPPING";
            default: return nullptr;
        }
    }

}// namespace

struct OutputPass::Impl {

    std::shared_ptr<ShaderMaterial> material = makeMaterial();
    FullScreenQuad fsQuad{material};

    std::optional<ColorSpace> outputColorSpace;
    std::optional<ToneMapping> toneMapping;
};

OutputPass::OutputPass(): pimpl_(std::make_unique<Impl>()) {}

void OutputPass::render(GLRenderer& renderer, RenderTarget* writeBuffer, RenderTarget* readBuffer, float, bool maskActive) {

    auto& material = *pimpl_->material;

    material.uniforms.at("tDiffuse").setValue(readBuffer ? readBuffer->texture.get() : static_cast<Texture*>(nullptr));
    material.uniforms.at("toneMappingExposure").setValue(renderer.toneMappingExposure);

    // Rebuild the defines when the renderer's settings change, as three.js.
    if (pimpl_->outputColorSpace != renderer.outputColorSpace || pimpl_->toneMapping != renderer.toneMapping) {

        pimpl_->outputColorSpace = renderer.outputColorSpace;
        pimpl_->toneMapping = renderer.toneMapping;

        material.defines.clear();
        if (renderer.outputColorSpace == ColorSpace::sRGB) material.defines["SRGB_TRANSFER"] = "";
        if (const auto* define = toneMappingDefine(renderer.toneMapping)) material.defines[define] = "";

        material.needsUpdate();
    }

    // Under an active mask, keep the stencil and the protected pixels: the
    // same rule as ShaderPass.
    const bool oldAutoClear = renderer.autoClear;
    if (maskActive) renderer.autoClear = false;

    if (renderToScreen) {

        renderer.setRenderTarget(nullptr);

    } else {

        renderer.setRenderTarget(writeBuffer);
        if (clear) renderer.clear(renderer.autoClearColor, renderer.autoClearDepth, renderer.autoClearStencil);
    }

    pimpl_->fsQuad.render(renderer);

    renderer.autoClear = oldAutoClear;
}

OutputPass::~OutputPass() = default;
