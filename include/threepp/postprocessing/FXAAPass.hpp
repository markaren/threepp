// https://github.com/mrdoob/three.js/blob/r186/examples/jsm/postprocessing/FXAAPass.js

#ifndef THREEPP_FXAAPASS_HPP
#define THREEPP_FXAAPASS_HPP

#include "threepp/postprocessing/ShaderPass.hpp"
#include "threepp/postprocessing/shaders/FXAAShader.hpp"

namespace threepp {

    // FXAA as a composer pass. Add it after an OutputPass:
    //
    //   composer.addPass(std::make_shared<OutputPass>());
    //   composer.addPass(std::make_shared<FXAAPass>());
    //
    // The composer's MSAA averages samples in linear HDR, before the tone map,
    // so an edge between an emitter several times brighter than white and a
    // dark background resolves to a value that still tone-maps to nearly full
    // brightness: bright edges keep their stairs. FXAA runs on the tone-mapped
    // image, where those edges are ordinary contrast.
    class FXAAPass: public ShaderPass {

    public:
        FXAAPass(): ShaderPass(shaders::fxaaShader()) {}

        void setSize(unsigned int width, unsigned int height) override {

            uniforms().at("resolution").setValue(Vector2(1.f / static_cast<float>(width), 1.f / static_cast<float>(height)));
        }
    };

}// namespace threepp

#endif//THREEPP_FXAAPASS_HPP
