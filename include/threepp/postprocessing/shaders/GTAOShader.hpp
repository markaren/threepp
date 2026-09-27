// https://github.com/mrdoob/three.js/blob/r186/examples/jsm/shaders/GTAOShader.js

#ifndef THREEPP_POSTPROCESSING_GTAOSHADER_HPP
#define THREEPP_POSTPROCESSING_GTAOSHADER_HPP

#include "threepp/core/Shader.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"

#include <string>
#include <unordered_map>

namespace threepp::shaders {

    // Ground-truth ambient occlusion (Jimenez et al., "Practical Realtime
    // Strategies for Accurate Indirect Occlusion"): for a few screen-space
    // slice directions per pixel, march the depth buffer both ways to find the
    // horizon angles, and integrate the cosine-weighted visible arc between
    // them against the surface normal.
    //
    // Output is the AO factor, 1 = unoccluded, in all three channels.
    //
    // Defines (set by GTAOPass): PERSPECTIVE_CAMERA, SAMPLES,
    // NORMAL_VECTOR_TYPE (1 = packed RGB view normals, 2 = raw, 0 = from
    // depth), DEPTH_SWIZZLING, SCREEN_SPACE_RADIUS, SCREEN_SPACE_RADIUS_SCALE,
    // SCENE_CLIP_BOX.
    inline std::unordered_map<std::string, std::string> gtaoShaderDefines() {

        return {
                {"PERSPECTIVE_CAMERA", "1"},
                {"SAMPLES", "16"},
                {"NORMAL_VECTOR_TYPE", "1"},
                {"DEPTH_SWIZZLING", "x"},
                {"SCREEN_SPACE_RADIUS", "0"},
                {"SCREEN_SPACE_RADIUS_SCALE", "100.0"},
                {"SCENE_CLIP_BOX", "0"}};
    }

    inline Shader gtaoShader() {

        return Shader{
                UniformMap{
                        {"tNormal", Uniform()},
                        {"tDepth", Uniform()},
                        {"tNoise", Uniform()},
                        {"resolution", Uniform(Vector2())},
                        {"cameraNear", Uniform(0.1f)},
                        {"cameraFar", Uniform(1000.f)},
                        {"cameraProjectionMatrix", Uniform(Matrix4())},
                        {"cameraProjectionMatrixInverse", Uniform(Matrix4())},
                        {"cameraWorldMatrix", Uniform(Matrix4())},
                        {"radius", Uniform(0.25f)},
                        {"distanceExponent", Uniform(1.f)},
                        {"thickness", Uniform(1.f)},
                        {"distanceFallOff", Uniform(1.f)},
                        {"scale", Uniform(1.f)},
                        {"sceneBoxMin", Uniform(Vector3(-1, -1, -1))},
                        {"sceneBoxMax", Uniform(Vector3(1, 1, 1))}},

                R"(
                varying vec2 vUv;

                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                })",

                R"(
                varying vec2 vUv;
                uniform highp sampler2D tNormal;
                uniform highp sampler2D tDepth;
                uniform sampler2D tNoise;
                uniform vec2 resolution;
                uniform float cameraNear;
                uniform float cameraFar;
                uniform mat4 cameraProjectionMatrix;
                uniform mat4 cameraProjectionMatrixInverse;
                uniform mat4 cameraWorldMatrix;
                uniform float radius;
                uniform float distanceExponent;
                uniform float thickness;
                uniform float distanceFallOff;
                uniform float scale;
                #if SCENE_CLIP_BOX == 1
                    uniform vec3 sceneBoxMin;
                    uniform vec3 sceneBoxMax;
                #endif

                #include <common>
                #include <packing>

                #ifndef FRAGMENT_OUTPUT
                #define FRAGMENT_OUTPUT vec4(vec3(ao), 1.)
                #endif

                vec3 getViewPosition( const in vec2 screenPosition, const in float depth ) {
                    vec4 clipSpacePosition = vec4( vec3( screenPosition, depth ) * 2.0 - 1.0, 1.0 );
                    vec4 viewSpacePosition = cameraProjectionMatrixInverse * clipSpacePosition;
                    return viewSpacePosition.xyz / viewSpacePosition.w;
                }

                float getDepth(const vec2 uv) {
                    return textureLod(tDepth, uv.xy, 0.0).DEPTH_SWIZZLING;
                }

                float fetchDepth(const ivec2 uv) {
                    return texelFetch(tDepth, uv.xy, 0).DEPTH_SWIZZLING;
                }

                float getViewZ(const in float depth) {
                    #if PERSPECTIVE_CAMERA == 1
                        return perspectiveDepthToViewZ(depth, cameraNear, cameraFar);
                    #else
                        return orthographicDepthToViewZ(depth, cameraNear, cameraFar);
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
                    vec3 dpdx = (dl < dr) ? ce - getViewPosition((uv - vec2(1.0 / size.x, 0.0)), l1).xyz : -ce + getViewPosition((uv + vec2(1.0 / size.x, 0.0)), r1).xyz;
                    vec3 dpdy = (db < dt) ? ce - getViewPosition((uv - vec2(0.0, 1.0 / size.y)), b1).xyz : -ce + getViewPosition((uv + vec2(0.0, 1.0 / size.y)), t1).xyz;
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

                vec3 getSceneUvAndDepth(vec3 sampleViewPos) {
                    vec4 sampleClipPos = cameraProjectionMatrix * vec4(sampleViewPos, 1.);
                    vec2 sampleUv = sampleClipPos.xy / sampleClipPos.w * 0.5 + 0.5;
                    float sampleSceneDepth = getDepth(sampleUv);
                    return vec3(sampleUv, sampleSceneDepth);
                }

                void main() {
                    float depth = getDepth(vUv.xy);

                    if (depth >= 1.0) {
                        discard;
                        return;
                    }

                    vec3 viewPos = getViewPosition(vUv, depth);
                    vec3 viewNormal = getViewNormal(vUv);

                    float radiusToUse = radius;
                    float distanceFalloffToUse = thickness;
                    #if SCREEN_SPACE_RADIUS == 1
                        float radiusScale = getViewPosition(vec2(0.5 + float(SCREEN_SPACE_RADIUS_SCALE) / resolution.x, 0.0), depth).x;
                        radiusToUse *= radiusScale;
                        distanceFalloffToUse *= radiusScale;
                    #endif

                    #if SCENE_CLIP_BOX == 1
                        vec3 worldPos = (cameraWorldMatrix * vec4(viewPos, 1.0)).xyz;
                        float boxDistance = length(max(vec3(0.0), max(sceneBoxMin - worldPos, worldPos - sceneBoxMax)));
                        if (boxDistance > radiusToUse) {
                            discard;
                            return;
                        }
                    #endif

                    vec2 noiseResolution = vec2(textureSize(tNoise, 0));
                    vec2 noiseUv = vUv * resolution / noiseResolution;
                    vec4 noiseTexel = textureLod(tNoise, noiseUv, 0.0);
                    vec3 randomVec = noiseTexel.xyz * 2.0 - 1.0;
                    vec3 tangent = normalize(vec3(randomVec.xy, 0.));
                    vec3 bitangent = vec3(-tangent.y, tangent.x, 0.);
                    mat3 kernelMatrix = mat3(tangent, bitangent, vec3(0., 0., 1.));

                    const int DIRECTIONS = SAMPLES < 30 ? 3 : 5;
                    const int STEPS = (SAMPLES + DIRECTIONS - 1) / DIRECTIONS;
                    float ao = 0.0;
                    for (int i = 0; i < DIRECTIONS; ++i) {

                        float angle = float(i) / float(DIRECTIONS) * PI;
                        vec4 sampleDir = vec4(cos(angle), sin(angle), 0., 0.5 + 0.5 * noiseTexel.w);
                        sampleDir.xyz = normalize(kernelMatrix * sampleDir.xyz);

                        // three.js uses -viewPos for both cameras; an orthographic
                        // camera looks straight down -z everywhere, and taking the
                        // ray through the origin instead darkens off-centre faces
                        // seen at a grazing angle.
                        #if PERSPECTIVE_CAMERA == 1
                            vec3 viewDir = normalize(-viewPos.xyz);
                        #else
                            vec3 viewDir = vec3(0., 0., 1.);
                        #endif
                        vec3 sliceBitangent = normalize(cross(sampleDir.xyz, viewDir));
                        vec3 sliceTangent = cross(sliceBitangent, viewDir);
                        vec3 normalInSlice = normalize(viewNormal - sliceBitangent * dot(viewNormal, sliceBitangent));

                        vec3 tangentToNormalInSlice = cross(normalInSlice, sliceBitangent);
                        vec2 cosHorizons = vec2(dot(viewDir, tangentToNormalInSlice), dot(viewDir, -tangentToNormalInSlice));

                        for (int j = 0; j < STEPS; ++j) {
                            vec3 sampleViewOffset = sampleDir.xyz * radiusToUse * sampleDir.w * pow(float(j + 1) / float(STEPS), distanceExponent);

                            vec3 sampleSceneUvDepth = getSceneUvAndDepth(viewPos + sampleViewOffset);
                            vec3 sampleSceneViewPos = getViewPosition(sampleSceneUvDepth.xy, sampleSceneUvDepth.z);
                            vec3 viewDelta = sampleSceneViewPos - viewPos;
                            if (abs(viewDelta.z) < thickness) {
                                float sampleCosHorizon = dot(viewDir, normalize(viewDelta));
                                cosHorizons.x += max(0., (sampleCosHorizon - cosHorizons.x) * mix(1., 2. / float(j + 2), distanceFallOff));
                            }

                            sampleSceneUvDepth = getSceneUvAndDepth(viewPos - sampleViewOffset);
                            sampleSceneViewPos = getViewPosition(sampleSceneUvDepth.xy, sampleSceneUvDepth.z);
                            viewDelta = sampleSceneViewPos - viewPos;
                            if (abs(viewDelta.z) < thickness) {
                                float sampleCosHorizon = dot(viewDir, normalize(viewDelta));
                                cosHorizons.y += max(0., (sampleCosHorizon - cosHorizons.y) * mix(1., 2. / float(j + 2), distanceFallOff));
                            }
                        }

                        vec2 sinHorizons = sqrt(1. - cosHorizons * cosHorizons);
                        float nx = dot(normalInSlice, sliceTangent);
                        float ny = dot(normalInSlice, viewDir);
                        float nxb = 1. / 2. * (acos(cosHorizons.y) - acos(cosHorizons.x) + sinHorizons.x * cosHorizons.x - sinHorizons.y * cosHorizons.y);
                        float nyb = 1. / 2. * (2. - cosHorizons.x * cosHorizons.x - cosHorizons.y * cosHorizons.y);
                        float occlusion = nx * nxb + ny * nyb;
                        ao += occlusion;
                    }

                    ao = clamp(ao / float(DIRECTIONS), 0., 1.);
                #if SCENE_CLIP_BOX == 1
                    ao = mix(ao, 1., smoothstep(0., radiusToUse, boxDistance));
                #endif
                    ao = pow(ao, scale);

                    gl_FragColor = FRAGMENT_OUTPUT;
                })"};
    }

    // Visualises the depth buffer: 1 at the near plane, 0 at the far one,
    // linear in view distance.
    inline Shader gtaoDepthShader() {

        return Shader{
                UniformMap{
                        {"tDepth", Uniform()},
                        {"cameraNear", Uniform(0.1f)},
                        {"cameraFar", Uniform(1000.f)}},

                R"(
                varying vec2 vUv;

                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                })",

                R"(
                uniform sampler2D tDepth;
                uniform float cameraNear;
                uniform float cameraFar;
                varying vec2 vUv;

                #include <packing>

                float getLinearDepth( const in vec2 screenPosition ) {
                    #if PERSPECTIVE_CAMERA == 1
                        float fragCoordZ = texture2D( tDepth, screenPosition ).x;
                        float viewZ = perspectiveDepthToViewZ( fragCoordZ, cameraNear, cameraFar );
                        return viewZToOrthographicDepth( viewZ, cameraNear, cameraFar );
                    #else
                        return texture2D( tDepth, screenPosition ).x;
                    #endif
                }

                void main() {
                    float depth = getLinearDepth( vUv );
                    gl_FragColor = vec4( vec3( 1.0 - depth ), 1.0 );

                })"};
    }

    // Multiplies the image by the AO, faded towards 1 by (1 - intensity). The
    // multiply itself is the material's blend state, not the shader.
    inline Shader gtaoBlendShader() {

        return Shader{
                UniformMap{
                        {"tDiffuse", Uniform()},
                        {"intensity", Uniform(1.f)}},

                R"(
                varying vec2 vUv;

                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                })",

                R"(
                uniform float intensity;
                uniform sampler2D tDiffuse;
                varying vec2 vUv;

                void main() {
                    vec4 texel = texture2D( tDiffuse, vUv );
                    gl_FragColor = vec4(mix(vec3(1.), texel.rgb, intensity), texel.a);
                })"};
    }

}// namespace threepp::shaders

#endif//THREEPP_POSTPROCESSING_GTAOSHADER_HPP
