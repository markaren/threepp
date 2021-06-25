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
                                      std::unordered_map<std::string, Uniform>{
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
                                      std::unordered_map<std::string, Uniform>{
                                            {"emissive", Uniform(Color(0x000000))},
                                            {"specular", Uniform(Color(0x111111))},
                                            {"shininess", Uniform(30)}
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
                                      std::unordered_map<std::string, Uniform>{
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
                                      std::unordered_map<std::string, Uniform>{
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
                                      std::unordered_map<std::string, Uniform>{
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
                                      std::unordered_map<std::string, Uniform>{
                                              {"scale", Uniform(1)},
                                              {"dashSize", Uniform(1)},
                                              {"totalSize", Uniform(2)}
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
                                      std::unordered_map<std::string, Uniform>{
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
                                      std::unordered_map<std::string, Uniform>{
                                              {"uvTransform", Uniform(Matrix3())},
                                              {"t2D", Uniform()}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().background_vert(),
                ShaderChunk::instance().background_frag()};

        Shader cube{
                mergeUniforms({// clang-format off
                                      UniformsLib::instance().envmap,
                                      std::unordered_map<std::string, Uniform>{
                                              {"opacity", Uniform(1.f)}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().cube_vert(),
                ShaderChunk::instance().cube_frag()};

        Shader equirect{
                mergeUniforms({// clang-format off
                                      std::unordered_map<std::string, Uniform>{
                                              {"tEquirect", Uniform()}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().equirect_vert(),
                ShaderChunk::instance().equirect_frag()};

        Shader distanceRGBA{
                mergeUniforms({// clang-format off
                                      UniformsLib::instance().common,
                                      UniformsLib::instance().displacementmap,
                                      std::unordered_map<std::string, Uniform>{
                                              {"referencePosition", Uniform(Vector3())},
                                              {"nearDistance", Uniform(1)},
                                              {"farDistance", Uniform(1000)}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().distanceRGBA_vert(),
                ShaderChunk::instance().distanceRGBA_frag()};

        Shader shadow{
                mergeUniforms({// clang-format off
                                      UniformsLib::instance().lights,
                                      UniformsLib::instance().fog,
                                      std::unordered_map<std::string, Uniform>{
                                              {"color", Uniform(Color(0x00000))},
                                              {"opacity", Uniform(1.f)}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().shadow_vert(),
                ShaderChunk::instance().shadow_frag()};

        Shader physical{
                mergeUniforms({// clang-format off
                                      standard.uniforms,
                                      std::unordered_map<std::string, Uniform>{
                                              {"clearcoat", Uniform(0)},
                                              {"clearcoatMap", Uniform()},
                                              {"clearcoatRoughness", Uniform(0)},
                                              {"clearcoatRoughnessMap", Uniform()},
                                              {"clearcoatNormalScale", Uniform(Vector2(1,1))},
                                              {"clearcoatNormalMap", Uniform()},
                                              {"sheen", Uniform(Color(0x000000))},
                                              {"transmission", Uniform(0)},
                                              {"transmissionMap", Uniform()},
                                              {"transmissionSamplerSize", Uniform(Vector2(0,0))},
                                              {"transmissionSamplerMap", Uniform()},
                                              {"thickness", Uniform(0)},
                                              {"thicknessMap", Uniform()},
                                              {"attenuationDistance", Uniform(0)},
                                              {"attenuationColor", Uniform(Color(0x000000))}
                                      }
                              }),// clang-format on

                ShaderChunk::instance().meshphysical_vert(),
                ShaderChunk::instance().meshphysical_frag()};


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
