// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/postprocessing/RenderPass.js

#ifndef THREEPP_POSTPROCESSING_RENDERPASS_HPP
#define THREEPP_POSTPROCESSING_RENDERPASS_HPP

#include "threepp/math/Color.hpp"
#include "threepp/postprocessing/Pass.hpp"

#include <optional>

namespace threepp {

    class Camera;
    class Material;
    class Object3D;

    // Draws the scene itself — the head of nearly every chain.
    //
    // Unlike the image-space passes this one writes into the *read* buffer and
    // does not swap: it is producing the image the rest of the chain consumes,
    // not transforming one.
    class RenderPass: public Pass {

    public:
        // Replaces every material in the scene for this pass — normals, depth,
        // IDs, whatever the effect downstream needs.
        std::shared_ptr<Material> overrideMaterial;

        // When set, the target is cleared to this instead of the renderer's
        // clear colour.
        std::optional<Color> clearColor;
        float clearAlpha = 0.f;

        // Clear only depth before drawing — for layering a second scene over a
        // first without erasing it.
        bool clearDepth = false;

        RenderPass(Object3D& scene, Camera& camera);

        void render(GLRenderer& renderer,
                    RenderTarget* writeBuffer,
                    RenderTarget* readBuffer,
                    float deltaTime,
                    bool maskActive) override;

    private:
        Object3D* scene_;
        Camera* camera_;
    };

}// namespace threepp

#endif//THREEPP_POSTPROCESSING_RENDERPASS_HPP
