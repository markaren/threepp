// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLMaterials.js

#ifndef THREEPP_GLMATERIAL_HPP
#define THREEPP_GLMATERIAL_HPP

#include "threepp/materials/materials.hpp"

#include "threepp/core/Uniform.hpp"
#include "threepp/utils/InstanceOf.hpp"

#include <unordered_map>

namespace threepp::gl {


    struct GLMaterials {

        void refreshMaterialUniforms(Material* material) {

            if (instanceof <MeshBasicMaterial>(material)) {



            }

        }

        void refreshUniformsCommon(std::unordered_map<std::string, Uniform> uniforms, Material* material) {

            if (uniforms.count("opacity")) {

                uniforms.at("opacity").setValue(material->opacity);

            }

            if (instanceof <MaterialWithColor>(material)) {

                if (uniforms.count("diffuse")) {
                    auto m = dynamic_cast<MaterialWithColor*>(material);
                    uniforms.at("diffuse").value<Color>().copy(m->getColor());
                }

            }

            if (instanceof <MaterialWithEmissive>(material)) {

                if (uniforms.count("emissive")) {
                    auto m = dynamic_cast<MaterialWithEmissive*>(material);
                    uniforms.at("emissive").value<Color>().copy(m->getEmissiveColor()).multiplyScalar(m->getEmissiveIntensity());
                }

            }

        }

    };


}

#endif//THREEPP_GLMATERIAL_HPP
