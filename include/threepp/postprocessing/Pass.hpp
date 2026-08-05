// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/postprocessing/Pass.js

#ifndef THREEPP_POSTPROCESSING_PASS_HPP
#define THREEPP_POSTPROCESSING_PASS_HPP

#include <memory>

namespace threepp {

    class GLRenderer;
    class Material;
    class Mesh;
    class OrthographicCamera;
    class RenderTarget;

    // One step of an EffectComposer chain.
    //
    // The composer owns two equally-sized render targets and hands every pass
    // both of them: `readBuffer` holds what the chain has produced so far, and
    // `writeBuffer` is where a pass that transforms the image should put its
    // result. A pass that leaves its result in `writeBuffer` must have
    // `needsSwap` true (the default) so the composer flips the two afterwards;
    // one that writes back into `readBuffer` — or draws nothing at all — sets
    // it false.
    class Pass {

    public:
        // A disabled pass is skipped entirely, including its buffer swap.
        bool enabled = true;

        // Whether the composer swaps read/write buffers after this pass.
        bool needsSwap = true;

        // Whether the pass clears its target before drawing.
        bool clear = false;

        // Set by the composer, not by user code: the composer always ends its
        // chain with an internal output pass (see EffectComposer), so for
        // ordinary passes this stays false.
        bool renderToScreen = false;

        Pass() = default;
        Pass(const Pass&) = delete;
        Pass& operator=(const Pass&) = delete;

        // Called when the composer is resized. Passes holding their own
        // targets or resolution-dependent uniforms override this.
        virtual void setSize(unsigned int width, unsigned int height) {}

        virtual void render(GLRenderer& renderer,
                            RenderTarget* writeBuffer,
                            RenderTarget* readBuffer,
                            float deltaTime,
                            bool maskActive) = 0;

        virtual ~Pass() = default;
    };

    // Draws a single triangle covering the whole viewport with the given
    // material — the workhorse of every image-space pass.
    //
    // A triangle rather than a quad: a quad's diagonal makes the GPU shade the
    // pixels along it twice (2x2 quads straddling the seam), which a single
    // oversized triangle avoids. Positions run to 3 in clip space and the UVs
    // to 2, so the [-1,1] / [0,1] region lands exactly where it would with a
    // quad.
    class FullScreenQuad {

    public:
        explicit FullScreenQuad(const std::shared_ptr<Material>& material);

        FullScreenQuad(const FullScreenQuad&) = delete;
        FullScreenQuad& operator=(const FullScreenQuad&) = delete;

        void setMaterial(const std::shared_ptr<Material>& material);

        [[nodiscard]] std::shared_ptr<Material> material() const;

        void render(GLRenderer& renderer);

        ~FullScreenQuad();

    private:
        std::shared_ptr<Mesh> mesh_;
        std::shared_ptr<OrthographicCamera> camera_;
    };

}// namespace threepp

#endif//THREEPP_POSTPROCESSING_PASS_HPP
