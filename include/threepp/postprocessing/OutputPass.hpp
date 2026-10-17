
#ifndef THREEPP_POSTPROCESSING_OUTPUTPASS_HPP
#define THREEPP_POSTPROCESSING_OUTPUTPASS_HPP

#include "threepp/postprocessing/Pass.hpp"

#include "threepp/math/Vector3.hpp"

#include <memory>

namespace threepp {

    // Tone mapping and the output colour space, as the last step of an
    // EffectComposer chain (three.js OutputPass, r153+).
    //
    // Since three.js r154 the renderer tone-maps only what it draws to the
    // screen; everything rendered into a target, and so everything the passes
    // of a chain read and write, is scene-linear HDR. This pass applies the
    // renderer's toneMapping (with toneMappingExposure) and then encodes into
    // its outputColorSpace, both read from the renderer each frame.
    //
    // EffectComposer ends every chain with one of these implicitly, so most
    // code never names it. Add one explicitly when a pass needs display-ready
    // (tone-mapped, sRGB) input and must come after it; the composer then does
    // not apply its own, so the image is never tone-mapped twice.
    //
    // Two optional stages sit round the tone curve, in the order and with the
    // math of the Vulkan renderer's post composite (PostComposite.cpp), so one
    // look reads the same on both backends: a white balance applied in linear
    // scene-referred space before the curve (a Bradford adaptation from the
    // scene illuminant's Kelvin and tint to D65), and a display-referred colour
    // grade applied after the sRGB encode (lift, gamma and gain per channel,
    // a luma-preserving saturation, a pivot-0.5 contrast). Both default to the
    // identity, and an identity stage is left out of the shader altogether, so
    // a pass that sets neither compiles and renders exactly as before.
    class OutputPass: public Pass {

    public:
        struct ColorGrade {
            Vector3 lift{0.f, 0.f, 0.f}; // adds toward the shadows
            Vector3 gamma{1.f, 1.f, 1.f};// mid-tone power, per channel
            Vector3 gain{1.f, 1.f, 1.f}; // scales toward the highlights
            float saturation = 1.f;      // about the Rec. 709 luma
            float contrast = 1.f;        // about 0.5, display-referred
        };

        OutputPass();

        // Kelvin on the Planckian locus (6500 = neutral = off) and a green(-)
        // / magenta(+) tint, as VulkanRenderer::setWhiteBalance.
        void setWhiteBalance(float temperatureK, float tint = 0.f);

        // As VulkanRenderer::setColorGrade; the defaults are the identity.
        void setColorGrade(const ColorGrade& grade);

        void render(GLRenderer& renderer,
                    RenderTarget* writeBuffer,
                    RenderTarget* readBuffer,
                    float deltaTime,
                    bool maskActive) override;

        ~OutputPass() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_POSTPROCESSING_OUTPUTPASS_HPP
