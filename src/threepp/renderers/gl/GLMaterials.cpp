
#include "GLMaterials.hpp"

#include "threepp/utils/InstanceOf.hpp"

using namespace threepp;
using namespace threepp::gl;

namespace {

    void refreshUniformsCommon(std::shared_ptr<UniformMap> &uniforms, Material *material) {

        uniforms->operator[]("opacity").setValue(material->opacity);

        if (instanceof <MaterialWithColor>(material)) {

            auto m = dynamic_cast<MaterialWithColor *>(material);
            uniforms->operator[]("diffuse").value<Color>().copy(m->color);
        }

        if (instanceof <MaterialWithEmissive>(material)) {

            auto m = dynamic_cast<MaterialWithEmissive *>(material);
            uniforms->operator[]("emissive").value<Color>().copy(m->emissiveColor).multiplyScalar(m->emissiveIntensity);
        }

        if (instanceof <MaterialWithMap>(material)) {

            auto& map = dynamic_cast<MaterialWithMap *>(material)->map;
            if (map) {
                uniforms->operator[]("map").setValue(*map);
            }
        }

        if (instanceof <MaterialWithAlphaMap>(material)) {

            auto &alphaMap = dynamic_cast<MaterialWithAlphaMap *>(material)->alphaMap;
            if (alphaMap) {
                uniforms->operator[]("alphaMap").setValue(*alphaMap);
            }
        }
    }

    void refreshUniformsLine(std::shared_ptr<UniformMap> &uniforms, LineBasicMaterial *material) {

        uniforms->operator[]("diffuse").value<Color>().copy(material->color);
        uniforms->operator[]("opacity").value<float>() = material->opacity;
    }

    void refreshUniformsPoints(std::shared_ptr<UniformMap> &uniforms, PointsMaterial *material, int pixelRatio, float height) {

        uniforms->operator[]("diffuse").value<Color>().copy(material->color);
        uniforms->operator[]("opacity").value<float>() = material->opacity;
        uniforms->operator[]("size").value<float>() = material->size * pixelRatio;
        uniforms->operator[]("scale").value<float>() = height * 0.5f;
    }

}

void GLMaterials::refreshFogUniforms(std::shared_ptr<UniformMap> &uniforms, FogVariant &fog) {

    if (fog.index() == 0) {

        Fog &f = std::get<Fog>(fog);
        uniforms->operator[]("fogColor").value<Color>().copy(f.color);

        uniforms->operator[]("fogNear").value<float>() = f.near;
        uniforms->operator[]("fogFar").value<float>() = f.far;
    } else {

        FogExp2 &f = std::get<FogExp2>(fog);
        uniforms->operator[]("fogColor").value<Color>().copy(f.color);

        uniforms->operator[]("fogDensity").value<float>() = f.density;
    }
}

void GLMaterials::refreshMaterialUniforms(std::shared_ptr<UniformMap> &uniforms, Material *material, int pixelRatio, float height) {

    if (instanceof <MeshBasicMaterial>(material)) {

        refreshUniformsCommon(uniforms, material);
    } else if (instanceof <LineBasicMaterial>(material)) {

        refreshUniformsLine(uniforms, dynamic_cast<LineBasicMaterial *>(material));
    } else if (instanceof <PointsMaterial>(material)) {

        refreshUniformsPoints(uniforms, dynamic_cast<PointsMaterial *>(material), pixelRatio, height);
    }
}
