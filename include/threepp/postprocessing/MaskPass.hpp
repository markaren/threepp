// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/postprocessing/MaskPass.js

#ifndef THREEPP_POSTPROCESSING_MASKPASS_HPP
#define THREEPP_POSTPROCESSING_MASKPASS_HPP

#include "threepp/postprocessing/Pass.hpp"

namespace threepp {

    class Camera;
    class Object3D;

    // Confines every following pass to where `scene` draws, by stencilling that
    // silhouette into both buffers and leaving the stencil test armed.
    //
    // The mask stays in force until a ClearMaskPass ends it — so a masked
    // effect is always three entries: MaskPass, the effect, ClearMaskPass.
    class MaskPass: public Pass {

    public:
        // Mask everything *outside* the scene's silhouette instead of inside.
        bool inverse = false;

        MaskPass(Object3D& scene, Camera& camera);

        void render(GLRenderer& renderer,
                    RenderTarget* writeBuffer,
                    RenderTarget* readBuffer,
                    float deltaTime,
                    bool maskActive) override;

    private:
        Object3D* scene_;
        Camera* camera_;
    };

    // Releases the stencil test a MaskPass armed.
    class ClearMaskPass: public Pass {

    public:
        ClearMaskPass();

        void render(GLRenderer& renderer,
                    RenderTarget* writeBuffer,
                    RenderTarget* readBuffer,
                    float deltaTime,
                    bool maskActive) override;
    };

}// namespace threepp

#endif//THREEPP_POSTPROCESSING_MASKPASS_HPP
