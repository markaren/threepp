// https://github.com/mrdoob/three.js/blob/r129/src/renderers/shaders/ShaderLib.js

#ifndef THREEPP_SHADERLIB_HPP
#define THREEPP_SHADERLIB_HPP

#include "threepp/renderers/shaders/Shader.hpp"
#include "threepp/renderers/shaders/ShaderChunk.hpp"

#include "threepp/renderers/shaders/UniformsLib.hpp"
#include "threepp/renderers/shaders/UniformsUtil.hpp"

namespace threepp::shaders {

    class ShaderLib {

    public:
        Shader basic{
                mergeUniforms({// clang-format off
                            UniformsLib::instance().common,
                            UniformsLib::instance().specularmap,
                            UniformsLib::instance().envmap,
                            UniformsLib::instance().aomap,
                            UniformsLib::instance().lightmap,
                            UniformsLib::instance().fog
                        }),// clang-format on

                ShaderChunk::instance().meshbasic_vert(),
                ShaderChunk::instance().meshbasic_frag()};

        Shader lambert{
                mergeUniforms({// clang-format off
                                      UniformsLib::instance().common,
                                      UniformsLib::instance().specularmap,
                                      UniformsLib::instance().envmap,
                                      UniformsLib::instance().aomap,
                                      UniformsLib::instance().lightmap,
                                      UniformsLib::instance().emissivemap,
                                      UniformsLib::instance().fog,
                                      UniformsLib::instance().lights,
                                      UniformMap{
                                              {"emissive", Uniform(Color(0x000000))}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().meshlambert_vert(),
                ShaderChunk::instance().meshlambert_frag()};

        Shader phong{
                mergeUniforms({// clang-format off
                                      UniformsLib::instance().common,
                                      UniformsLib::instance().specularmap,
                                      UniformsLib::instance().envmap,
                                      UniformsLib::instance().aomap,
                                      UniformsLib::instance().lightmap,
                                      UniformsLib::instance().emissivemap,
                                      UniformsLib::instance().bumpmap,
                                      UniformsLib::instance().normalmap,
                                      UniformsLib::instance().displacementmap,
                                      UniformsLib::instance().fog,
                                      UniformsLib::instance().lights,
                                      UniformMap{
                                            {"emissive", Uniform(Color(0x000000))},
                                            {"specular", Uniform(Color(0x111111))},
                                            {"shininess", Uniform(30.f)}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().meshphong_vert(),
                ShaderChunk::instance().meshphong_frag()};

        Shader standard{
                mergeUniforms({// clang-format off
                                      UniformsLib::instance().common,
                                      UniformsLib::instance().envmap,
                                      UniformsLib::instance().aomap,
                                      UniformsLib::instance().lightmap,
                                      UniformsLib::instance().emissivemap,
                                      UniformsLib::instance().bumpmap,
                                      UniformsLib::instance().normalmap,
                                      UniformsLib::instance().displacementmap,
                                      UniformsLib::instance().roughnessmap,
                                      UniformsLib::instance().metalnessmap,
                                      UniformsLib::instance().fog,
                                      UniformsLib::instance().lights,
                                      UniformMap{
                                              {"emissive", Uniform(Color(0x000000))},
                                              {"roughness", Uniform(1.f)},
                                              {"metalness", Uniform(0.f)},
                                              {"envMapIntensity", Uniform(1)}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().meshphong_vert(),
                ShaderChunk::instance().meshphysical_frag()};

        Shader toon{
                mergeUniforms({// clang-format off
                                      UniformsLib::instance().common,
                                      UniformsLib::instance().aomap,
                                      UniformsLib::instance().lightmap,
                                      UniformsLib::instance().emissivemap,
                                      UniformsLib::instance().bumpmap,
                                      UniformsLib::instance().normalmap,
                                      UniformsLib::instance().displacementmap,
                                      UniformsLib::instance().gradientmap,
                                      UniformsLib::instance().fog,
                                      UniformsLib::instance().lights,
                                      UniformMap{
                                              {"emissive", Uniform(Color(0x000000))}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().meshtoon_vert(),
                ShaderChunk::instance().meshtoon_frag()};

        Shader matcap{
                mergeUniforms({// clang-format off
                                      UniformsLib::instance().common,
                                      UniformsLib::instance().bumpmap,
                                      UniformsLib::instance().normalmap,
                                      UniformsLib::instance().displacementmap,
                                      UniformsLib::instance().fog,
                                      UniformMap{
                                              {"matcap", Uniform()}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().meshmatcap_vert(),
                ShaderChunk::instance().meshmatcap_frag()};

        Shader points{
                mergeUniforms({// clang-format off
                                      UniformsLib::instance().points,
                                      UniformsLib::instance().fog
                              }),// clang-format on

                ShaderChunk::instance().points_vert(),
                ShaderChunk::instance().points_frag()};

        Shader dashed{
                mergeUniforms({// clang-format off
                                      UniformsLib::instance().common,
                                      UniformsLib::instance().fog,
                                      UniformMap{
                                              {"scale", Uniform(1.f)},
                                              {"dashSize", Uniform(1.f)},
                                              {"totalSize", Uniform(2.f)}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().linedashed_vert(),
                ShaderChunk::instance().linedashed_frag()};

        Shader depth{
                mergeUniforms({
                        // clang-format off
                                      UniformsLib::instance().common,
                                      UniformsLib::instance().displacementmap,
                              }),// clang-format on

                ShaderChunk::instance().depth_vert(),
                ShaderChunk::instance().depth_frag()};

        Shader normal{
                mergeUniforms({// clang-format off
                                      UniformsLib::instance().common,
                                      UniformsLib::instance().bumpmap,
                                      UniformsLib::instance().normalmap,
                                      UniformsLib::instance().displacementmap,
                                      UniformMap{
                                              {"opacity", Uniform(1.f)}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().normal_vert(),
                ShaderChunk::instance().normal_frag()};

        Shader sprite{
                mergeUniforms({
                        // clang-format off
                                      UniformsLib::instance().sprite,
                                      UniformsLib::instance().fog,
                              }),// clang-format on

                ShaderChunk::instance().sprite_vert(),
                ShaderChunk::instance().sprite_frag()};

        Shader background{
                mergeUniforms({// clang-format off
                                      UniformMap{
                                              {"uvTransform", Uniform(Matrix3())},
                                              {"t2D", Uniform()}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().background_vert(),
                ShaderChunk::instance().background_frag()};

        Shader cube{
                mergeUniforms({// clang-format off
                                      UniformsLib::instance().envmap,
                                      UniformMap{
                                              {"opacity", Uniform(1.f)}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().cube_vert(),
                ShaderChunk::instance().cube_frag()};

        Shader equirect{
                mergeUniforms({// clang-format off
                                      UniformMap{
                                              {"tEquirect", Uniform()}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().equirect_vert(),
                ShaderChunk::instance().equirect_frag()};

        Shader distanceRGBA{
                mergeUniforms({// clang-format off
                                      UniformsLib::instance().common,
                                      UniformsLib::instance().displacementmap,
                                      UniformMap{
                                              {"referencePosition", Uniform(Vector3())},
                                              {"nearDistance", Uniform(1.f)},
                                              {"farDistance", Uniform(1000.f)}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().distanceRGBA_vert(),
                ShaderChunk::instance().distanceRGBA_frag()};

        Shader shadow{
                mergeUniforms({// clang-format off
                                      UniformsLib::instance().lights,
                                      UniformsLib::instance().fog,
                                      UniformMap{
                                              {"color", Uniform(Color(0x00000))},
                                              {"opacity", Uniform(1.f)}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().shadow_vert(),
                ShaderChunk::instance().shadow_frag()};

        Shader physical{
                mergeUniforms({// clang-format off
                                      standard.uniforms,
                                      UniformMap{
                                              {"clearcoat", Uniform(0.f)},
                                              {"clearcoatMap", Uniform()},
                                              {"clearcoatRoughness", Uniform(0.f)},
                                              {"clearcoatRoughnessMap", Uniform()},
                                              {"clearcoatNormalScale", Uniform(Vector2(1,1))},
                                              {"clearcoatNormalMap", Uniform()},
                                              {"sheen", Uniform(Color(0x000000))},
                                              {"transmission", Uniform(0.f)},
                                              {"transmissionMap", Uniform()},
                                              {"transmissionSamplerSize", Uniform(Vector2(0,0))},
                                              {"transmissionSamplerMap", Uniform()},
                                              {"thickness", Uniform(0.f)},
                                              {"thicknessMap", Uniform()},
                                              {"attenuationDistance", Uniform(0.f)},
                                              {"attenuationColor", Uniform(Color(0x000000))}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().meshphysical_vert(),
                ShaderChunk::instance().meshphysical_frag()};


        [[nodiscard]] Shader get(const std::string &name) const {

            if (name == "basic") {
                return basic;
            } else if (name == "lambert") {
                return lambert;
            } else if (name == "phong") {
                return phong;
            } else if (name == "standard") {
                return standard;
            } else if (name == "toon") {
                return toon;
            } else if (name == "matcap") {
                return matcap;
            } else if (name == "points") {
                return points;
            } else if (name == "dashed") {
                return dashed;
            } else if (name == "depth") {
                return depth;
            } else if (name == "normal") {
                return normal;
            } else if (name == "sprite") {
                return sprite;
            } else if (name == "background") {
                return background;
            } else if (name == "cube") {
                return cube;
            } else if (name == "equirect") {
                return equirect;
            } else if (name == "distanceRGBA") {
                return distanceRGBA;
            } else if (name == "shadow") {
                return shadow;
            } else if (name == "physical") {
                return physical;
            } else {
                throw std::runtime_error("No shader with name: " + name);
            }
        }

        ShaderLib(const ShaderLib &) = delete;
        void operator=(const ShaderLib &) = delete;

        static ShaderLib &instance() {
            static ShaderLib instance;
            return instance;
        }

    private:
        ShaderLib() = default;
    };

}// namespace threepp::shaders

#endif//THREEPP_SHADERLIB_HPP
