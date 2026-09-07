// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/shaders/LuminosityHighPassShader.js

#ifndef THREEPP_POSTPROCESSING_LUMINOSITYHIGHPASSSHADER_HPP
#define THREEPP_POSTPROCESSING_LUMINOSITYHIGHPASSSHADER_HPP

#include "threepp/core/Shader.hpp"

namespace threepp::shaders {

    // Keeps what is brighter than `luminosityThreshold` and replaces the rest
    // with `defaultColor`, fading across `smoothWidth`. The first step of a
    // bloom chain: what survives here is what glows.
    inline Shader luminosityHighPassShader() {

        return Shader{
                UniformMap{
                        {"tDiffuse", Uniform()},
                        {"luminosityThreshold", Uniform(1.f)},
                        {"smoothWidth", Uniform(1.f)},
                        {"defaultColor", Uniform(Color(0x000000))},
                        {"defaultOpacity", Uniform(0.f)}},

                R"(
                varying vec2 vUv;

                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                })",

                R"(
                uniform sampler2D tDiffuse;
                uniform vec3 defaultColor;
                uniform float defaultOpacity;
                uniform float luminosityThreshold;
                uniform float smoothWidth;

                varying vec2 vUv;

                void main() {
                    vec4 texel = texture2D( tDiffuse, vUv );

                    vec3 luma = vec3( 0.299, 0.587, 0.114 );

                    float v = dot( texel.xyz, luma );

                    vec4 outputColor = vec4( defaultColor.rgb, defaultOpacity );

                    float alpha = smoothstep( luminosityThreshold, luminosityThreshold + smoothWidth, v );

                    gl_FragColor = mix( outputColor, texel, alpha );
                })"};
    }

}// namespace threepp::shaders

#endif//THREEPP_POSTPROCESSING_LUMINOSITYHIGHPASSSHADER_HPP
