
#include "threepp/postprocessing/OutputPass.hpp"

#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/renderers/common/WhiteBalance.hpp"

#include <cmath>
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
                {"toneMappingExposure", Uniform(1.f)},
                {"whiteBalance", Uniform(Matrix3())},
                {"gradeLift", Uniform(Vector3(0, 0, 0))},
                {"gradeGamma", Uniform(Vector3(1, 1, 1))},
                {"gradeGain", Uniform(Vector3(1, 1, 1))},
                {"gradeSaturation", Uniform(1.f)},
                {"gradeContrast", Uniform(1.f)}};
        material->vertexShader = R"(
            varying vec2 vUv;

            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
            })";
        material->fragmentShader = R"(
            uniform sampler2D tDiffuse;
            varying vec2 vUv;

            #ifdef USE_WHITE_BALANCE
                uniform mat3 whiteBalance;
            #endif

            #ifdef USE_COLOR_GRADE
                uniform vec3 gradeLift;
                uniform vec3 gradeGamma;
                uniform vec3 gradeGain;
                uniform float gradeSaturation;
                uniform float gradeContrast;
            #endif

            #include <tonemapping_pars_fragment>

            void main() {

                gl_FragColor = texture2D( tDiffuse, vUv );

                // White balance in linear scene-referred space, before the
                // curve (PostComposite: exposure, white balance, tone map).
                #ifdef USE_WHITE_BALANCE
                    gl_FragColor.rgb = whiteBalance * gl_FragColor.rgb;
                #endif

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

                // The grade on the encoded, display-referred result: what
                // PostComposite bakes into its LUT, evaluated directly.
                #ifdef USE_COLOR_GRADE
                    vec3 c = clamp( gl_FragColor.rgb, 0.0, 1.0 );
                    vec3 v = c * gradeGain + gradeLift * ( 1.0 - c );
                    v = pow( max( v, vec3( 0.0 ) ), 1.0 / max( gradeGamma, vec3( 1e-3 ) ) );
                    float luma = dot( v, vec3( 0.2126, 0.7152, 0.0722 ) );
                    v = luma + ( v - luma ) * gradeSaturation;
                    v = ( v - 0.5 ) * gradeContrast + 0.5;
                    gl_FragColor.rgb = clamp( v, 0.0, 1.0 );
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

    bool whiteBalanceActive = false;
    bool gradeActive = false;
    // The stages the material was last built with: one coming or going
    // rebuilds the program like a change of the renderer's settings.
    bool builtWhiteBalance = false;
    bool builtGrade = false;
};

OutputPass::OutputPass(): pimpl_(std::make_unique<Impl>()) {}

void OutputPass::setWhiteBalance(float temperatureK, float tint) {

    const bool neutral = std::abs(temperatureK - 6500.f) < 1.f && std::abs(tint) < 1e-4f;
    pimpl_->whiteBalanceActive = !neutral;
    pimpl_->material->uniforms.at("whiteBalance").value<Matrix3>().copy(
            neutral ? Matrix3() : whiteBalanceToD65(temperatureK, tint));
}

void OutputPass::setColorGrade(const ColorGrade& g) {

    auto near1 = [](float v) { return std::abs(v - 1.f) < 1e-4f; };
    auto near0 = [](float v) { return std::abs(v) < 1e-4f; };
    const bool identity =
            near0(g.lift.x) && near0(g.lift.y) && near0(g.lift.z) &&
            near1(g.gamma.x) && near1(g.gamma.y) && near1(g.gamma.z) &&
            near1(g.gain.x) && near1(g.gain.y) && near1(g.gain.z) &&
            near1(g.saturation) && near1(g.contrast);
    pimpl_->gradeActive = !identity;

    auto& uniforms = pimpl_->material->uniforms;
    uniforms.at("gradeLift").value<Vector3>().copy(g.lift);
    uniforms.at("gradeGamma").value<Vector3>().copy(g.gamma);
    uniforms.at("gradeGain").value<Vector3>().copy(g.gain);
    uniforms.at("gradeSaturation").value<float>() = g.saturation;
    uniforms.at("gradeContrast").value<float>() = g.contrast;
}

void OutputPass::render(GLRenderer& renderer, RenderTarget* writeBuffer, RenderTarget* readBuffer, float, bool maskActive) {

    auto& impl = *pimpl_;
    auto& material = *impl.material;

    material.uniforms.at("tDiffuse").setValue(readBuffer ? readBuffer->texture.get() : static_cast<Texture*>(nullptr));
    material.uniforms.at("toneMappingExposure").setValue(renderer.toneMappingExposure);

    // Rebuild the defines when the renderer's settings change, as three.js,
    // and when a white balance or a grade is switched on or off.
    if (impl.outputColorSpace != renderer.outputColorSpace || impl.toneMapping != renderer.toneMapping ||
        impl.builtWhiteBalance != impl.whiteBalanceActive || impl.builtGrade != impl.gradeActive) {

        impl.outputColorSpace = renderer.outputColorSpace;
        impl.toneMapping = renderer.toneMapping;
        impl.builtWhiteBalance = impl.whiteBalanceActive;
        impl.builtGrade = impl.gradeActive;

        material.defines.clear();
        if (renderer.outputColorSpace == ColorSpace::sRGB) material.defines["SRGB_TRANSFER"] = "";
        if (const auto* define = toneMappingDefine(renderer.toneMapping)) material.defines[define] = "";
        if (impl.whiteBalanceActive) material.defines["USE_WHITE_BALANCE"] = "";
        if (impl.gradeActive) material.defines["USE_COLOR_GRADE"] = "";

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
