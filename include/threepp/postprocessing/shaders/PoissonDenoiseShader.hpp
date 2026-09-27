// https://github.com/mrdoob/three.js/blob/r186/examples/jsm/shaders/PoissonDenoiseShader.js

#ifndef THREEPP_POSTPROCESSING_POISSONDENOISESHADER_HPP
#define THREEPP_POSTPROCESSING_POISSONDENOISESHADER_HPP

#include "threepp/core/Shader.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector2.hpp"

#include <cmath>
#include <sstream>
#include <string>
#include <unordered_map>

namespace threepp::shaders {

    // The sample pattern as a GLSL array initializer: `samples` points on a
    // spiral that turns `rings` times, each with its radius fraction
    // (i / (samples - 1))^radiusExponent in z. Baked into the shader as the
    // SAMPLE_VECTORS define, so changing any of the three recompiles it.
    inline std::string generatePdSamplePointInitializer(int samples, float rings, float radiusExponent) {

        std::ostringstream glsl;
        glsl.precision(9);
        glsl << "vec3[SAMPLES](";

        for (int i = 0; i < samples; i++) {

            const double angle = 2 * math::PI * rings * i / samples;
            const double radius = std::pow(static_cast<double>(i) / (samples - 1), radiusExponent);
            glsl << "vec3(" << std::fixed << std::cos(angle) << ", " << std::sin(angle) << ", " << radius << ")"
                 << (i < samples - 1 ? "," : ")");
        }

        return glsl.str();
    }

    inline std::unordered_map<std::string, std::string> poissonDenoiseShaderDefines() {

        return {
                {"SAMPLES", "16"},
                {"SAMPLE_VECTORS", generatePdSamplePointInitializer(16, 2, 1)},
                {"NORMAL_VECTOR_TYPE", "1"},
                {"DEPTH_VALUE_SOURCE", "0"}};
    }

    // Edge-aware blur for noisy screen-space buffers (GTAO's): a rotated
    // Poisson disk of taps, each weighted by how alike its luminance, plane
    // distance and normal are to the centre's, so the noise averages away
    // without bleeding across silhouettes or creases.
    inline Shader poissonDenoiseShader() {

        return Shader{
                UniformMap{
                        {"tDiffuse", Uniform()},
                        {"tNormal", Uniform()},
                        {"tDepth", Uniform()},
                        {"tNoise", Uniform()},
                        {"resolution", Uniform(Vector2())},
                        {"cameraProjectionMatrixInverse", Uniform(Matrix4())},
                        {"lumaPhi", Uniform(5.f)},
                        {"depthPhi", Uniform(5.f)},
                        {"normalPhi", Uniform(5.f)},
                        {"radius", Uniform(4.f)},
                        {"index", Uniform(0)}},

                R"(
                varying vec2 vUv;

                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                })",

                R"(
                varying vec2 vUv;

                uniform sampler2D tDiffuse;
                uniform sampler2D tNormal;
                uniform sampler2D tDepth;
                uniform sampler2D tNoise;
                uniform vec2 resolution;
                uniform mat4 cameraProjectionMatrixInverse;
                uniform float lumaPhi;
                uniform float depthPhi;
                uniform float normalPhi;
                uniform float radius;
                uniform int index;

                #include <common>
                #include <packing>

                #ifndef SAMPLE_LUMINANCE
                #define SAMPLE_LUMINANCE dot(vec3(0.2125, 0.7154, 0.0721), a)
                #endif

                #ifndef FRAGMENT_OUTPUT
                #define FRAGMENT_OUTPUT vec4(denoised, 1.)
                #endif

                float getLuminance(const in vec3 a) {
                    return SAMPLE_LUMINANCE;
                }

                const vec3 poissonDisk[SAMPLES] = SAMPLE_VECTORS;

                vec3 getViewPosition( const in vec2 screenPosition, const in float depth ) {
                    vec4 clipSpacePosition = vec4( vec3( screenPosition, depth ) * 2.0 - 1.0, 1.0 );
                    vec4 viewSpacePosition = cameraProjectionMatrixInverse * clipSpacePosition;
                    return viewSpacePosition.xyz / viewSpacePosition.w;
                }

                float getDepth(const vec2 uv) {
                #if DEPTH_VALUE_SOURCE == 1
                    return textureLod(tDepth, uv.xy, 0.0).a;
                #else
                    return textureLod(tDepth, uv.xy, 0.0).r;
                #endif
                }

                float fetchDepth(const ivec2 uv) {
                    #if DEPTH_VALUE_SOURCE == 1
                        return texelFetch(tDepth, uv.xy, 0).a;
                    #else
                        return texelFetch(tDepth, uv.xy, 0).r;
                    #endif
                }

                vec3 computeNormalFromDepth(const vec2 uv) {
                    vec2 size = vec2(textureSize(tDepth, 0));
                    ivec2 p = ivec2(uv * size);
                    float c0 = fetchDepth(p);
                    float l2 = fetchDepth(p - ivec2(2, 0));
                    float l1 = fetchDepth(p - ivec2(1, 0));
                    float r1 = fetchDepth(p + ivec2(1, 0));
                    float r2 = fetchDepth(p + ivec2(2, 0));
                    float b2 = fetchDepth(p - ivec2(0, 2));
                    float b1 = fetchDepth(p - ivec2(0, 1));
                    float t1 = fetchDepth(p + ivec2(0, 1));
                    float t2 = fetchDepth(p + ivec2(0, 2));
                    float dl = abs((2.0 * l1 - l2) - c0);
                    float dr = abs((2.0 * r1 - r2) - c0);
                    float db = abs((2.0 * b1 - b2) - c0);
                    float dt = abs((2.0 * t1 - t2) - c0);
                    vec3 ce = getViewPosition(uv, c0).xyz;
                    vec3 dpdx = (dl < dr) ?  ce - getViewPosition((uv - vec2(1.0 / size.x, 0.0)), l1).xyz
                                            : -ce + getViewPosition((uv + vec2(1.0 / size.x, 0.0)), r1).xyz;
                    vec3 dpdy = (db < dt) ?  ce - getViewPosition((uv - vec2(0.0, 1.0 / size.y)), b1).xyz
                                            : -ce + getViewPosition((uv + vec2(0.0, 1.0 / size.y)), t1).xyz;
                    return normalize(cross(dpdx, dpdy));
                }

                vec3 getViewNormal(const vec2 uv) {
                #if NORMAL_VECTOR_TYPE == 2
                    return normalize(textureLod(tNormal, uv, 0.).rgb);
                #elif NORMAL_VECTOR_TYPE == 1
                    return unpackRGBToNormal(textureLod(tNormal, uv, 0.).rgb);
                #else
                    return computeNormalFromDepth(uv);
                #endif
                }

                void denoiseSample(in vec3 center, in vec3 viewNormal, in vec3 viewPos, in vec2 sampleUv, inout vec3 denoised, inout float totalWeight) {
                    vec4 sampleTexel = textureLod(tDiffuse, sampleUv, 0.0);
                    float sampleDepth = getDepth(sampleUv);
                    vec3 sampleNormal = getViewNormal(sampleUv);
                    vec3 neighborColor = sampleTexel.rgb;
                    vec3 viewPosSample = getViewPosition(sampleUv, sampleDepth);

                    float normalDiff = dot(viewNormal, sampleNormal);
                    float normalSimilarity = pow(max(normalDiff, 0.), normalPhi);
                    float lumaDiff = abs(getLuminance(neighborColor) - getLuminance(center));
                    float lumaSimilarity = max(1.0 - lumaDiff / lumaPhi, 0.0);
                    float depthDiff = abs(dot(viewPos - viewPosSample, viewNormal));
                    float depthSimilarity = max(1. - depthDiff / depthPhi, 0.);
                    float w = lumaSimilarity * depthSimilarity * normalSimilarity;

                    denoised += w * neighborColor;
                    totalWeight += w;
                }

                void main() {
                    float depth = getDepth(vUv.xy);
                    vec3 viewNormal = getViewNormal(vUv);
                    if (depth == 1. || dot(viewNormal, viewNormal) == 0.) {
                        discard;
                        return;
                    }
                    vec4 texel = textureLod(tDiffuse, vUv, 0.0);
                    vec3 center = texel.rgb;
                    vec3 viewPos = getViewPosition(vUv, depth);

                    vec2 noiseResolution = vec2(textureSize(tNoise, 0));
                    vec2 noiseUv = vUv * resolution / noiseResolution;
                    vec4 noiseTexel = textureLod(tNoise, noiseUv, 0.0);
                    vec2 noiseVec = vec2(sin(noiseTexel[index % 4] * 2. * PI), cos(noiseTexel[index % 4] * 2. * PI));
                    mat2 rotationMatrix = mat2(noiseVec.x, -noiseVec.y, noiseVec.x, noiseVec.y);

                    float totalWeight = 1.0;
                    vec3 denoised = texel.rgb;
                    for (int i = 0; i < SAMPLES; i++) {
                        vec3 sampleDir = poissonDisk[i];
                        vec2 offset = rotationMatrix * (sampleDir.xy * (1. + sampleDir.z * (radius - 1.)) / resolution);
                        vec2 sampleUv = vUv + offset;
                        denoiseSample(center, viewNormal, viewPos, sampleUv, denoised, totalWeight);
                    }

                    if (totalWeight > 0.) {
                        denoised /= totalWeight;
                    }
                    gl_FragColor = FRAGMENT_OUTPUT;
                })"};
    }

}// namespace threepp::shaders

#endif//THREEPP_POSTPROCESSING_POISSONDENOISESHADER_HPP
