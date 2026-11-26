// https://github.com/mrdoob/three.js/blob/dev/examples/jsm/postprocessing/SSAARenderPass.js

#ifndef THREEPP_POSTPROCESSING_SSAARENDERPASS_HPP
#define THREEPP_POSTPROCESSING_SSAARENDERPASS_HPP

#include "threepp/math/Color.hpp"
#include "threepp/postprocessing/Pass.hpp"

#include <memory>

namespace threepp {

    class Camera;
    class Object3D;
    class ShaderMaterial;

    // Supersampling anti-aliasing: draws the scene several times, each time
    // with the camera shifted by a fraction of a pixel, and averages the
    // results. It takes the place of a RenderPass at the head of a chain.
    //
    // Unlike MSAA, which only supersamples the edges of triangles, every
    // sample here is fully shaded, so thin geometry at a distance, fine
    // texture and specular highlights all resolve instead of crawling into
    // moire. The price is one scene render per sample: 2^sampleLevel of them.
    //
    // Jitter offsets are the standard multisample patterns, on a 1/16 pixel
    // grid. Works with a PerspectiveCamera or an OrthographicCamera; any view
    // offset the camera already has is kept and jittered around.
    class SSAARenderPass: public Pass {

    public:
        // 2^sampleLevel samples: 0 is one (no anti-aliasing), 4 is sixteen,
        // 5 (the most) thirty-two.
        int sampleLevel = 4;

        // Spreads the sample weights a little around 1/N so that the rounding
        // errors of the accumulation cancel instead of piling up.
        bool unbiased = true;

        // Whether the per-sample target has a stencil buffer.
        bool stencilBuffer = false;

        // What each sample is cleared to before the scene is drawn into it.
        Color clearColor;
        float clearAlpha;

        SSAARenderPass(Object3D& scene, Camera& camera, const Color& clearColor = Color(0x000000), float clearAlpha = 0.f);

        void setSize(unsigned int width, unsigned int height) override;

        void render(GLRenderer& renderer,
                    RenderTarget* writeBuffer,
                    RenderTarget* readBuffer,
                    float deltaTime,
                    bool maskActive) override;

        ~SSAARenderPass() override;

    private:
        Object3D* scene_;
        Camera* camera_;

        std::unique_ptr<RenderTarget> sampleRenderTarget_;
        std::shared_ptr<ShaderMaterial> copyMaterial_;
        std::unique_ptr<FullScreenQuad> fsQuad_;
    };

}// namespace threepp

#endif//THREEPP_POSTPROCESSING_SSAARENDERPASS_HPP
