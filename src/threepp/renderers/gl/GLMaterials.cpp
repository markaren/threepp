
#include "GLMaterials.hpp"

#include "threepp/materials/materials.hpp"
#include "threepp/utils/InstanceOf.hpp"

using namespace threepp;
using namespace threepp::gl;

namespace {

    void refreshUniformsCommon(std::shared_ptr<UniformMap> &uniforms, Material *material, GLProperties &properties) {

        uniforms->operator[]("opacity").setValue(material->opacity);

        if (instanceof <MaterialWithColor>(material)) {

            auto m = dynamic_cast<MaterialWithColor *>(material);
            uniforms->operator[]("diffuse").value<Color>().copy(m->color);
        }

        if (instanceof <MaterialWithEmissive>(material)) {

            auto m = dynamic_cast<MaterialWithEmissive *>(material);
            uniforms->operator[]("emissive").value<Color>().copy(m->emissive).multiplyScalar(m->emissiveIntensity);
        }

        if (instanceof <MaterialWithMap>(material)) {

            auto &map = dynamic_cast<MaterialWithMap *>(material)->map;
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

        if (instanceof <MaterialWithSpecularMap>(material)) {

            auto &specularMap = dynamic_cast<MaterialWithSpecularMap *>(material)->specularMap;
            if (specularMap) {
                uniforms->operator[]("specularMap").setValue(*specularMap);
            }
        }

        if (instanceof <MaterialWithLightMap>(material)) {

            auto m = dynamic_cast<MaterialWithLightMap *>(material);
            if (m->lightMap) {
                uniforms->operator[]("lightMap").setValue(*m->lightMap);
            }
            uniforms->operator[]("lightMapIntensity").setValue(m->lightMapIntensity);
        }

        if (instanceof <MaterialWithAoMap>(material)) {

            auto m = dynamic_cast<MaterialWithAoMap *>(material);
            if (m->aoMap) {
                uniforms->operator[]("aoMap").setValue(*m->aoMap);
            }
            uniforms->operator[]("aoMapIntensity").setValue(m->aoMapIntensity);
        }
    }

    void refreshUniformsLambert(std::shared_ptr<UniformMap> &uniforms, MeshLambertMaterial *material) {

        auto &m = material->emissiveMap;
        if (m) {

            uniforms->operator[]("emissiveMap").setValue(*m);
        }
    }

    void refreshUniformsPhong(std::shared_ptr<UniformMap> &uniforms, MeshPhongMaterial *material) {

        uniforms->operator[]("specular").value<Color>().copy(material->specular);
        uniforms->operator[]("shininess").setValue(std::max(material->shininess, (float) 1E-4));// to prevent pow( 0.0, 0.0 )

        auto &emissiveMap = material->emissiveMap;
        if (emissiveMap) {

            uniforms->operator[]("emissiveMap").setValue(*emissiveMap);
        }

        auto &bumpMap = material->bumpMap;
        if (bumpMap) {

            uniforms->operator[]("bumpMap").setValue(*bumpMap);
            uniforms->operator[]("bumpScale").setValue(material->bumpScale);
            if (material->side == BackSide) {
                float bumpScale = uniforms->operator[]("bumpScale").value<float>();
                uniforms->operator[]("bumpScale").setValue(bumpScale *= -1);
            }
        }

        auto &normalMap = material->normalMap;
        if (normalMap) {

            uniforms->operator[]("normalMap").setValue(*normalMap);
            uniforms->operator[]("normalScale").value<Vector2>().copy(material->normalScale);
            if (material->side == BackSide) {
                uniforms->operator[]("normalScale").value<Vector2>().negate();
            }
        }

        auto &displacementMap = material->displacementMap;
        if (displacementMap) {

            uniforms->operator[]("displacementMap").setValue(*displacementMap);
            uniforms->operator[]("displacementScale").setValue(material->displacementScale);
            uniforms->operator[]("displacementBias").setValue(material->displacementBias);
        }
    }

    void refreshUniformsToon(std::shared_ptr<UniformMap> &uniforms, MeshToonMaterial *material) {

        auto &gradientMap = material->gradientMap;
        if (gradientMap) {

            uniforms->operator[]("gradientMap").setValue(*gradientMap);
        }

        auto &emissiveMap = material->emissiveMap;
        if (emissiveMap) {

            uniforms->operator[]("emissiveMap").setValue(*emissiveMap);
        }

        auto &bumpMap = material->bumpMap;
        if (bumpMap) {

            uniforms->operator[]("bumpMap").setValue(*bumpMap);
            uniforms->operator[]("bumpScale").setValue(material->bumpScale);
            if (material->side == BackSide) {
                float bumpScale = uniforms->operator[]("bumpScale").value<float>();
                uniforms->operator[]("bumpScale").setValue(bumpScale *= -1);
            }
        }

        auto &normalMap = material->normalMap;
        if (normalMap) {

            uniforms->operator[]("normalMap").setValue(*normalMap);
            uniforms->operator[]("normalScale").value<Vector2>().copy(material->normalScale);
            if (material->side == BackSide) {
                uniforms->operator[]("normalScale").value<Vector2>().negate();
            }
        }

        auto &displacementMap = material->displacementMap;
        if (displacementMap) {

            uniforms->operator[]("displacementMap").setValue(*displacementMap);
            uniforms->operator[]("displacementScale").setValue(material->displacementScale);
            uniforms->operator[]("displacementBias").setValue(material->displacementBias);
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

}// namespace

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

        refreshUniformsCommon(uniforms, material, properties);
    } else if (instanceof <MeshLambertMaterial>(material)) {

        auto m = dynamic_cast<MeshLambertMaterial *>(material);
        refreshUniformsCommon(uniforms, m, properties);
        refreshUniformsLambert(uniforms, m);

    } else if (instanceof <MeshToonMaterial>(material)) {

        auto m = dynamic_cast<MeshToonMaterial *>(material);
        refreshUniformsCommon(uniforms, m, properties);
        refreshUniformsToon(uniforms, m);

    }  else if (instanceof <MeshPhongMaterial>(material)) {

        auto m = dynamic_cast<MeshPhongMaterial *>(material);
        refreshUniformsCommon(uniforms, m, properties);
        refreshUniformsPhong(uniforms, m);

    } else if (instanceof <LineBasicMaterial>(material)) {

        refreshUniformsLine(uniforms, dynamic_cast<LineBasicMaterial *>(material));

    } else if (instanceof <PointsMaterial>(material)) {

        refreshUniformsPoints(uniforms, dynamic_cast<PointsMaterial *>(material), pixelRatio, height);

    } else if (instanceof <ShaderMaterial>(material)) {

        dynamic_cast<ShaderMaterial*>(material)->uniformsNeedUpdate = false;

    }
}
