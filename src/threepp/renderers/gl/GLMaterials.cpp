
#include "threepp/renderers/gl/GLMaterials.hpp"

#include "threepp/materials/materials.hpp"
#include "threepp/utils/InstanceOf.hpp"

using namespace threepp;
using namespace threepp::gl;

namespace {

    void refreshUniformsCommon(std::shared_ptr<UniformMap> &uniforms, Material *material, GLProperties &properties) {

        auto mapMaterial = dynamic_cast<MaterialWithMap *>(material);
        auto specularMaterial = dynamic_cast<MaterialWithSpecularMap *>(material);
        auto displacementMaterial = dynamic_cast<MaterialWithDisplacementMap *>(material);
        auto normalMaterial = dynamic_cast<MaterialWithNormalMap *>(material);
        auto bumpMaterial = dynamic_cast<MaterialWithBumpMap *>(material);
        auto roughnessMaterial = dynamic_cast<MaterialWithRoughness *>(material);
        auto metalnessMaterial = dynamic_cast<MaterialWithMetalness *>(material);
        auto alphaMaterial = dynamic_cast<MaterialWithAlphaMap *>(material);
        auto emissiveMaterial = dynamic_cast<MaterialWithEmissive *>(material);
        // TODO clearcoat

        auto aoMaterial = dynamic_cast<MaterialWithAoMap *>(material);
        auto lightMaterial = dynamic_cast<MaterialWithLightMap *>(material);

        uniforms->operator[]("opacity").setValue(material->opacity);

        if (instanceof <MaterialWithColor>(material)) {

            auto m = dynamic_cast<MaterialWithColor *>(material);
            uniforms->operator[]("diffuse").value<Color>().copy(m->color);
        }

        if (emissiveMaterial) {

            uniforms->operator[]("emissive").value<Color>().copy(emissiveMaterial->emissive).multiplyScalar(emissiveMaterial->emissiveIntensity);
        }

        if (mapMaterial && mapMaterial->map) {

            uniforms->operator[]("map").setValue(mapMaterial->map);
        }

        if (alphaMaterial && alphaMaterial->alphaMap) {

            uniforms->operator[]("alphaMap").setValue(alphaMaterial->alphaMap);
        }

        if (specularMaterial && specularMaterial->specularMap) {

            uniforms->operator[]("specularMap").setValue(specularMaterial->specularMap);
        }

        auto envMap = properties.materialProperties.get(material->uuid).envMap;
        if (envMap) {

            uniforms->operator[]("envMap").setValue(envMap);
            uniforms->operator[]("flipEnvMap").value<bool>() = false;//TODO

            auto reflectiveMaterial = dynamic_cast<MaterialWithReflectivity *>(material);
            if (reflectiveMaterial) {
                uniforms->operator[]("reflectivity").value<float>() = reflectiveMaterial->reflectivity;
                uniforms->operator[]("refractionRatio").value<float>() = reflectiveMaterial->refractionRatio;
            }

            const auto &maxMipMapLevel = properties.textureProperties.get(envMap->uuid).maxMipLevel;
            if (maxMipMapLevel) {
                uniforms->operator[]("maxMipLevel").value<int>() = *maxMipMapLevel;
            }
        }

        if (lightMaterial) {

            if (lightMaterial->lightMap) {
                uniforms->operator[]("lightMap").setValue(lightMaterial->lightMap);
            }
            uniforms->operator[]("lightMapIntensity").setValue(lightMaterial->lightMapIntensity);
        }

        if (aoMaterial) {

            if (aoMaterial->aoMap) {
                uniforms->operator[]("aoMap").setValue(aoMaterial->aoMap);
            }
            uniforms->operator[]("aoMapIntensity").setValue(aoMaterial->aoMapIntensity);
        }

        // uv repeat and offset setting priorities
        // 1. color map
        // 2. specular map
        // 3. displacementMap map
        // 4. normal map
        // 5. bump map
        // 6. roughnessMap map
        // 7. metalnessMap map
        // 8. alphaMap map
        // 9. emissiveMap map
        // 10. clearcoat map
        // 11. clearcoat normal map
        // 12. clearcoat roughnessMap map


        std::optional<Texture> uvScaleMap{};

        if (mapMaterial && mapMaterial->map) {
            uvScaleMap = *mapMaterial->map;
        } else if (specularMaterial && specularMaterial->specularMap) {
            uvScaleMap = *specularMaterial->specularMap;
        } else if (displacementMaterial && displacementMaterial->displacementMap) {
            uvScaleMap = *displacementMaterial->displacementMap;
        } else if (normalMaterial && normalMaterial->normalMap) {
            uvScaleMap = *normalMaterial->normalMap;
        } else if (bumpMaterial && bumpMaterial->bumpMap) {
            uvScaleMap = *bumpMaterial->bumpMap;
        } else if (roughnessMaterial && roughnessMaterial->roughnessMap) {
            uvScaleMap = *roughnessMaterial->roughnessMap;
        } else if (metalnessMaterial && metalnessMaterial->metallnessMap) {
            uvScaleMap = *metalnessMaterial->metallnessMap;
        } else if (alphaMaterial && alphaMaterial->alphaMap) {
            uvScaleMap = *alphaMaterial->alphaMap;
        } else if (emissiveMaterial && emissiveMaterial->emissiveMap) {
            uvScaleMap = *emissiveMaterial->emissiveMap;
        }
        // TODO clearcoat

        if (uvScaleMap) {

            if (uvScaleMap->matrixAutoUpdate) {

                uvScaleMap->updateMatrix();
            }

            uniforms->operator[]("uvTransform").value<Matrix3>().copy(uvScaleMap->matrix);
        }

        // uv repeat and offset setting priorities for uv2
        // 1. ao map
        // 2. light map

        std::optional<Texture> uv2ScaleMap{};

        if (aoMaterial && aoMaterial->aoMap) {

            uv2ScaleMap = *aoMaterial->aoMap;

        } else if (lightMaterial && lightMaterial->lightMap) {

            uv2ScaleMap = *lightMaterial->lightMap;
        }

        if (uv2ScaleMap) {

            if (uv2ScaleMap->matrixAutoUpdate) {

                uv2ScaleMap->updateMatrix();
            }

            uniforms->operator[]("uv2Transform").value<Matrix3>().copy(uv2ScaleMap->matrix);
        }
    }

    void refreshUniformsLambert(std::shared_ptr<UniformMap> &uniforms, MeshLambertMaterial *material) {

        auto &emissiveMap = material->emissiveMap;
        if (emissiveMap) {
            uniforms->operator[]("emissiveMap").setValue(emissiveMap);
        }
    }

    void refreshUniformsPhong(std::shared_ptr<UniformMap> &uniforms, MeshPhongMaterial *material) {

        uniforms->operator[]("specular").value<Color>().copy(material->specular);
        uniforms->operator[]("shininess").value<float>() = std::max(material->shininess, (float) 1E-4);// to prevent pow( 0.0, 0.0 )

        auto &emissiveMap = material->emissiveMap;
        if (emissiveMap) {

            uniforms->operator[]("emissiveMap").setValue(emissiveMap);
        }

        auto &bumpMap = material->bumpMap;
        if (bumpMap) {

            uniforms->operator[]("bumpMap").setValue(bumpMap);
            uniforms->operator[]("bumpScale").setValue(material->bumpScale);
            if (material->side == BackSide) {
                uniforms->operator[]("bumpScale").value<float>() *= -1;
            }
        }

        auto &normalMap = material->normalMap;
        if (normalMap) {

            uniforms->operator[]("normalMap").setValue(normalMap);
            uniforms->operator[]("normalScale").value<Vector2>().copy(material->normalScale);
            if (material->side == BackSide) {
                uniforms->operator[]("normalScale").value<Vector2>().negate();
            }
        }

        auto &displacementMap = material->displacementMap;
        if (displacementMap) {

            uniforms->operator[]("displacementMap").setValue(displacementMap);
            uniforms->operator[]("displacementScale").value<float>() = material->displacementScale;
            uniforms->operator[]("displacementBias").value<float>() = material->displacementBias;
        }
    }

    void refreshUniformsToon(std::shared_ptr<UniformMap> &uniforms, MeshToonMaterial *material) {

        auto &gradientMap = material->gradientMap;
        if (gradientMap) {

            uniforms->operator[]("gradientMap").setValue(gradientMap);
        }

        auto &emissiveMap = material->emissiveMap;
        if (emissiveMap) {

            uniforms->operator[]("emissiveMap").setValue(emissiveMap);
        }

        auto &bumpMap = material->bumpMap;
        if (bumpMap) {

            uniforms->operator[]("bumpMap").setValue(bumpMap);
            uniforms->operator[]("bumpScale").value<float>() = material->bumpScale;
            if (material->side == BackSide) {
                uniforms->operator[]("bumpScale").value<float>() *= -1;
            }
        }

        auto &normalMap = material->normalMap;
        if (normalMap) {

            uniforms->operator[]("normalMap").setValue(normalMap);
            uniforms->operator[]("normalScale").value<Vector2>().copy(material->normalScale);
            if (material->side == BackSide) {
                uniforms->operator[]("normalScale").value<Vector2>().negate();
            }
        }

        auto &displacementMap = material->displacementMap;
        if (displacementMap) {

            uniforms->operator[]("displacementMap").setValue(displacementMap);
            uniforms->operator[]("displacementScale").value<float>() = material->displacementScale;
            uniforms->operator[]("displacementBias").value<float>() = material->displacementBias;
        }
    }


    void refreshUniformsLine(std::shared_ptr<UniformMap> &uniforms, LineBasicMaterial *material) {

        uniforms->operator[]("diffuse").value<Color>().copy(material->color);
        uniforms->operator[]("opacity").value<float>() = material->opacity;
    }

    void refreshUniformsPoints(std::shared_ptr<UniformMap> &uniforms, PointsMaterial *material, int pixelRatio, float height) {

        uniforms->operator[]("diffuse").value<Color>().copy(material->color);
        uniforms->operator[]("opacity").value<float>() = material->opacity;
        uniforms->operator[]("size").value<float>() = material->size * (float) pixelRatio;
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

void GLMaterials::refreshMaterialUniforms(std::shared_ptr<UniformMap> &uniforms, Material *material, int pixelRatio, int height) {

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

    } else if (instanceof <MeshPhongMaterial>(material)) {

        auto m = dynamic_cast<MeshPhongMaterial *>(material);
        refreshUniformsCommon(uniforms, m, properties);
        refreshUniformsPhong(uniforms, m);

    } else if (instanceof <LineBasicMaterial>(material)) {

        refreshUniformsLine(uniforms, dynamic_cast<LineBasicMaterial *>(material));

    } else if (instanceof <PointsMaterial>(material)) {

        refreshUniformsPoints(uniforms, dynamic_cast<PointsMaterial *>(material), pixelRatio, (float) height);

    } else if (instanceof <ShaderMaterial>(material)) {

        dynamic_cast<ShaderMaterial *>(material)->uniformsNeedUpdate = false;
    }
}
