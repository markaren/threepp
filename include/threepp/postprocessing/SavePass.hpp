// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/postprocessing/SavePass.js

#ifndef THREEPP_POSTPROCESSING_SAVEPASS_HPP
#define THREEPP_POSTPROCESSING_SAVEPASS_HPP

#include "threepp/postprocessing/Pass.hpp"

namespace threepp {

    class RenderTarget;
    class ShaderMaterial;

    // Copies the current image into a target of its own and leaves the chain
    // untouched — the way a later pass gets hold of an earlier frame or an
    // intermediate stage.
    class SavePass: public Pass {

    public:
        // Saves into `renderTarget`, or into one the pass allocates at the
        // composer's size when none is given.
        explicit SavePass(std::shared_ptr<RenderTarget> renderTarget = nullptr);

        [[nodiscard]] RenderTarget* renderTarget() const;

        void setSize(unsigned int width, unsigned int height) override;

        void render(GLRenderer& renderer,
                    RenderTarget* writeBuffer,
                    RenderTarget* readBuffer,
                    float deltaTime,
                    bool maskActive) override;

        ~SavePass() override;

    private:
        bool ownsTarget_;
        std::shared_ptr<RenderTarget> renderTarget_;
        std::shared_ptr<ShaderMaterial> material_;
        std::unique_ptr<FullScreenQuad> fsQuad_;
    };

}// namespace threepp

#endif//THREEPP_POSTPROCESSING_SAVEPASS_HPP
