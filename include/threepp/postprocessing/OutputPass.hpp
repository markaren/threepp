
#ifndef THREEPP_POSTPROCESSING_OUTPUTPASS_HPP
#define THREEPP_POSTPROCESSING_OUTPUTPASS_HPP

#include "threepp/postprocessing/Pass.hpp"

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
    class OutputPass: public Pass {

    public:
        OutputPass();

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
