// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/postprocessing/ClearPass.js

#ifndef THREEPP_POSTPROCESSING_CLEARPASS_HPP
#define THREEPP_POSTPROCESSING_CLEARPASS_HPP

#include "threepp/math/Color.hpp"
#include "threepp/postprocessing/Pass.hpp"

#include <optional>

namespace threepp {

    // Clears the read buffer. Useful ahead of passes that composite onto what
    // is already there rather than replacing it.
    class ClearPass: public Pass {

    public:
        std::optional<Color> clearColor;
        float clearAlpha = 0.f;

        ClearPass();

        explicit ClearPass(const Color& clearColor, float clearAlpha = 0.f);

        void render(GLRenderer& renderer,
                    RenderTarget* writeBuffer,
                    RenderTarget* readBuffer,
                    float deltaTime,
                    bool maskActive) override;
    };

}// namespace threepp

#endif//THREEPP_POSTPROCESSING_CLEARPASS_HPP
