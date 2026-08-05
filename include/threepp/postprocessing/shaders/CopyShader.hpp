// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/shaders/CopyShader.js

#ifndef THREEPP_POSTPROCESSING_COPYSHADER_HPP
#define THREEPP_POSTPROCESSING_COPYSHADER_HPP

#include "threepp/core/Shader.hpp"

namespace threepp::shaders {

    // Full-screen copy with a constant opacity — the identity pass every
    // composer chain leans on.
    //
    // The fragment shader ends in <encodings_fragment>, which is what makes it
    // safe as the chain's last step: the renderer compiles linearToOutputTexel
    // against whatever is currently bound, so the same shader is a straight
    // copy into a linear intermediate target and the sRGB encode when it draws
    // to the screen.
    inline Shader copyShader() {

        return Shader{
                UniformMap{
                        {"tDiffuse", Uniform()},
                        {"opacity", Uniform(1.f)}},

                R"(
                varying vec2 vUv;

                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                })",

                R"(
                uniform float opacity;
                uniform sampler2D tDiffuse;
                varying vec2 vUv;

                void main() {
                    gl_FragColor = opacity * texture2D( tDiffuse, vUv );
                    #include <encodings_fragment>
                })"};
    }

}// namespace threepp::shaders

#endif//THREEPP_POSTPROCESSING_COPYSHADER_HPP
