// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/postprocessing/EffectComposer.js

#ifndef THREEPP_POSTPROCESSING_EFFECTCOMPOSER_HPP
#define THREEPP_POSTPROCESSING_EFFECTCOMPOSER_HPP

#include "threepp/constants.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace threepp {

    class GLRenderer;
    class Pass;
    class RenderTarget;

    // A chain of full-screen passes rendered through a pair of ping-pong
    // targets. OpenGL only — the Vulkan backend is swapchain-only and owns its
    // own post chain.
    //
    // Two things differ from three.js on purpose:
    //
    //  * The composer always finishes by drawing its result to the screen
    //    itself, rather than handing the screen to whichever pass happens to be
    //    last. Intermediate targets are linear, and threepp compiles the output
    //    transform per bound target, so a pass shader written for an offscreen
    //    target would silently skip the sRGB encode if it drew to the screen —
    //    the classic washed-out composer image. Owning the final draw puts the
    //    encode in exactly one place. Set `renderToScreen` false to keep the
    //    result offscreen and read it from readBuffer() instead.
    //
    //  * MSAA is a composer option (`Options::samples`) rather than something
    //    you lose by adding a composer at all: the internal targets can be
    //    multisampled and resolve on unbind.
    class EffectComposer {

    public:
        struct Options {

            // MSAA samples for the internal targets. 0 disables it. Clamped to
            // the driver's maximum. Costs bandwidth on every pass, not just the
            // one drawing geometry — the two targets alternate roles.
            unsigned int samples{0};

            bool depthBuffer{true};

            // On by default because MaskPass needs somewhere to put its mask.
            bool stencilBuffer{true};

            // Pixel type of the internal targets.
            //
            // HalfFloat by default, and not only so a chain stops clipping at
            // 1.0 between passes: the intermediates hold *linear* light, and
            // 8 bits of linear is not enough to survive the output encode. The
            // encode expands the bottom of the range hard — the first three
            // byte steps, 0/1/2, come out as 0, 13 and 22 — so a dark gradient
            // written to a byte target arrives on screen as visible contour
            // bands however smooth it was. Anything dim is mostly bottom of
            // the range, which is why a byte chain bands where a direct render
            // of the same scene does not.
            //
            // Set UnsignedByte to get the cheaper targets back; the banding
            // comes with them.
            std::optional<Type> type;

            Options() = default;
        };

        // Whether render() draws the finished image to the screen. When false,
        // the result is left in readBuffer().
        bool renderToScreen = true;

        // Two overloads rather than a defaulted argument: `= {}` on a nested
        // type needs that type's default member initializers complete while
        // the enclosing class still isn't, which GCC rejects.
        explicit EffectComposer(GLRenderer& renderer);

        EffectComposer(GLRenderer& renderer, const Options& options);

        EffectComposer(const EffectComposer&) = delete;
        EffectComposer& operator=(const EffectComposer&) = delete;

        void addPass(const std::shared_ptr<Pass>& pass);

        void insertPass(const std::shared_ptr<Pass>& pass, size_t index);

        void removePass(const Pass* pass);

        [[nodiscard]] const std::vector<std::shared_ptr<Pass>>& passes() const;

        // Is `index` the last pass that will actually run? Passes use this to
        // decide whether anything downstream will read their output.
        [[nodiscard]] bool isLastEnabledPass(size_t index) const;

        // deltaTime is forwarded to passes that animate; leave it at 0 for
        // passes that don't care.
        void render(float deltaTime = 0.f);

        // Resize the internal targets. Sizes are in framebuffer pixels; the
        // composer applies the renderer's pixel ratio itself.
        void setSize(unsigned int width, unsigned int height);

        void setPixelRatio(float pixelRatio);

        [[nodiscard]] float getPixelRatio() const;

        // The target holding the chain's current result.
        [[nodiscard]] RenderTarget& readBuffer() const;

        [[nodiscard]] RenderTarget& writeBuffer() const;

        // Drop and rebuild both targets — after changing sample count, say.
        void reset(const Options& options);

        // The finished image, as RGB bytes, bottom row first (same convention
        // as GLRenderer::readRGBPixels).
        [[nodiscard]] std::vector<unsigned char> readRGBPixels() const;

        ~EffectComposer();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_POSTPROCESSING_EFFECTCOMPOSER_HPP
