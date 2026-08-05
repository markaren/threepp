// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/postprocessing/TexturePass.js

#ifndef THREEPP_POSTPROCESSING_TEXTUREPASS_HPP
#define THREEPP_POSTPROCESSING_TEXTUREPASS_HPP

#include "threepp/postprocessing/Pass.hpp"

namespace threepp {

    class ShaderMaterial;
    class Texture;

    // Draws a texture over the current image at a given opacity — a backdrop,
    // a stored frame, an overlay. Below opacity 1 it blends instead of
    // replacing.
    class TexturePass: public Pass {

    public:
        float opacity;

        explicit TexturePass(std::shared_ptr<Texture> map, float opacity = 1.f);

        void setTexture(std::shared_ptr<Texture> map);

        void render(GLRenderer& renderer,
                    RenderTarget* writeBuffer,
                    RenderTarget* readBuffer,
                    float deltaTime,
                    bool maskActive) override;

        ~TexturePass() override;

    private:
        std::shared_ptr<Texture> map_;
        std::shared_ptr<ShaderMaterial> material_;
        std::unique_ptr<FullScreenQuad> fsQuad_;
    };

}// namespace threepp

#endif//THREEPP_POSTPROCESSING_TEXTUREPASS_HPP
