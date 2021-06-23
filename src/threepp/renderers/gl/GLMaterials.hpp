// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLMaterials.js

#ifndef THREEPP_GLMATERIAL_HPP
#define THREEPP_GLMATERIAL_HPP

#include "threepp/materials/materials.hpp"

#include "threepp/core/Uniform.hpp"
#include "threepp/utils/InstanceOf.hpp"

#include <unordered_map>

namespace threepp::gl {


    struct GLMaterials {

        void refreshMaterialUniforms(std::unordered_map<std::string, Uniform> &uniforms, Material *material, int pixelRatio, float height) {

            if (instanceof <MeshBasicMaterial>(material)) {

                refreshUniformsCommon(uniforms, material);
            } else if (instanceof <LineBasicMaterial>(material)) {

                refreshUniformsLine(uniforms, dynamic_cast<LineBasicMaterial *>(material));
            } else if (instanceof <PointsMaterial>(material)) {

                refreshUniformsPoints(uniforms, dynamic_cast<PointsMaterial *>(material), pixelRatio, height);
            }
        }

        void refreshUniformsCommon(std::unordered_map<std::string, Uniform> &uniforms, Material *material) {

            uniforms["opacity"].setValue(material->opacity);

            if (instanceof <MaterialWithColor>(material)) {

                auto m = dynamic_cast<MaterialWithColor *>(material);
                uniforms["diffuse"].value<Color>().copy(m->getColor());
            }

            if (instanceof <MaterialWithEmissive>(material)) {

                auto m = dynamic_cast<MaterialWithEmissive *>(material);
                uniforms["emissive"].value<Color>().copy(m->getEmissiveColor()).multiplyScalar(m->getEmissiveIntensity());
            }

            if (instanceof <MaterialWithMap>(material)) {

                auto m = dynamic_cast<MaterialWithMap *>(material);
                auto map = m->getMap();
                if (map) {
                    uniforms["map"].setValue(map);
                }
            }

            if (instanceof <MaterialWithAlphaMap>(material)) {

                auto m = dynamic_cast<MaterialWithAlphaMap *>(material);
                auto alphaMap = m->getAlphaMap();
                if (alphaMap) {
                    uniforms["alphaMap"].setValue(alphaMap);
                }
            }
        }

        void refreshUniformsLine(std::unordered_map<std::string, Uniform> &uniforms, LineBasicMaterial *material) {

            uniforms["diffuse"].value<Color>().copy(material->getColor());
            uniforms["opacity"].value<float>() = material->opacity;
        }

        void refreshUniformsPoints(std::unordered_map<std::string, Uniform> &uniforms, PointsMaterial *material, int pixelRatio, float height) {

            uniforms["diffuse"].value<Color>().copy(material->getColor());
            uniforms["opacity"].value<float>() = material->opacity;
            uniforms["size"].value<int>() = (int) material->getSize() * pixelRatio;
            uniforms["scale"].value<float>() = height * 0.5f;
        }
    };


}// namespace threepp::gl

#endif//THREEPP_GLMATERIAL_HPP
