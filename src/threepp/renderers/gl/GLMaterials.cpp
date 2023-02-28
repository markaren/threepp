
#include "threepp/renderers/gl/GLMaterials.hpp"

#include "threepp/materials/materials.hpp"

using namespace threepp;
using namespace threepp::gl;

namespace {

    void refreshUniformsCommon(UniformMap& uniforms, Material* material, GLProperties& properties) {

        auto colorMaterial = dynamic_cast<MaterialWithColor*>(material);
        auto mapMaterial = dynamic_cast<MaterialWithMap*>(material);
        auto specularMaterial = dynamic_cast<MaterialWithSpecularMap*>(material);
        auto displacementMaterial = dynamic_cast<MaterialWithDisplacementMap*>(material);
        auto normalMaterial = dynamic_cast<MaterialWithNormalMap*>(material);
        auto bumpMaterial = dynamic_cast<MaterialWithBumpMap*>(material);
        auto roughnessMaterial = dynamic_cast<MaterialWithRoughness*>(material);
        auto metalnessMaterial = dynamic_cast<MaterialWithMetalness*>(material);
        auto alphaMaterial = dynamic_cast<MaterialWithAlphaMap*>(material);
        auto emissiveMaterial = dynamic_cast<MaterialWithEmissive*>(material);
        auto spriteMaterial = dynamic_cast<SpriteMaterial*>(material);
        // TODO clearcoat

        auto aoMaterial = dynamic_cast<MaterialWithAoMap*>(material);
        auto lightMaterial = dynamic_cast<MaterialWithLightMap*>(material);

        uniforms.at("opacity").setValue(material->opacity);

        if (colorMaterial) {

            uniforms.at("diffuse").value<Color>().copy(colorMaterial->color);
        }

        if (emissiveMaterial) {

            uniforms.at("emissive").value<Color>().copy(emissiveMaterial->emissive).multiplyScalar(emissiveMaterial->emissiveIntensity);
        }

        if (mapMaterial && mapMaterial->map) {

            uniforms.at("map").setValue(mapMaterial->map);
        }

        if (alphaMaterial && alphaMaterial->alphaMap) {

            uniforms.at("alphaMap").setValue(alphaMaterial->alphaMap);
        }

        if (specularMaterial && specularMaterial->specularMap) {

            uniforms.at("specularMap").setValue(specularMaterial->specularMap);
        }

        auto envMap = properties.materialProperties.get(material->uuid())->envMap;
        if (envMap) {

            uniforms.at("envMap").setValue(envMap);
            uniforms.at("flipEnvMap").value<bool>() = false;//TODO

            auto reflectiveMaterial = dynamic_cast<MaterialWithReflectivity*>(material);
            if (reflectiveMaterial) {
                uniforms.at("reflectivity").value<float>() = reflectiveMaterial->reflectivity;
                uniforms.at("refractionRatio").value<float>() = reflectiveMaterial->refractionRatio;
            }

            const auto& maxMipMapLevel = properties.textureProperties.get(envMap->uuid)->maxMipLevel;
            if (maxMipMapLevel) {
                uniforms.at("maxMipLevel").value<int>() = *maxMipMapLevel;
            }
        }

        if (lightMaterial && lightMaterial->lightMap) {
            uniforms.at("lightMap").setValue(lightMaterial->lightMap);
            uniforms.at("lightMapIntensity").setValue(lightMaterial->lightMapIntensity);
        }

        if (aoMaterial && aoMaterial->aoMap) {
            uniforms.at("aoMap").setValue(aoMaterial->aoMap);
            uniforms.at("aoMapIntensity").setValue(aoMaterial->aoMapIntensity);
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


        std::shared_ptr<Texture> uvScaleMap = nullptr;

        if (mapMaterial && mapMaterial->map) {
            uvScaleMap = mapMaterial->map;
        } else if (specularMaterial && specularMaterial->specularMap) {
            uvScaleMap = specularMaterial->specularMap;
        } else if (displacementMaterial && displacementMaterial->displacementMap) {
            uvScaleMap = displacementMaterial->displacementMap;
        } else if (normalMaterial && normalMaterial->normalMap) {
            uvScaleMap = normalMaterial->normalMap;
        } else if (bumpMaterial && bumpMaterial->bumpMap) {
            uvScaleMap = bumpMaterial->bumpMap;
        } else if (roughnessMaterial && roughnessMaterial->roughnessMap) {
            uvScaleMap = roughnessMaterial->roughnessMap;
        } else if (metalnessMaterial && metalnessMaterial->metalnessMap) {
            uvScaleMap = metalnessMaterial->metalnessMap;
        } else if (alphaMaterial && alphaMaterial->alphaMap) {
            uvScaleMap = alphaMaterial->alphaMap;
        } else if (emissiveMaterial && emissiveMaterial->emissiveMap) {
            uvScaleMap = emissiveMaterial->emissiveMap;
        }
        // TODO clearcoat

        if (uvScaleMap) {

            if (uvScaleMap->matrixAutoUpdate) {

                uvScaleMap->updateMatrix();
            }

            uniforms.at("uvTransform").value<Matrix3>().copy(uvScaleMap->matrix);
        }

        // uv repeat and offset setting priorities for uv2
        // 1. ao map
        // 2. light map

        std::shared_ptr<Texture> uv2ScaleMap = nullptr;

        if (aoMaterial && aoMaterial->aoMap) {

            uv2ScaleMap = aoMaterial->aoMap;

        } else if (lightMaterial && lightMaterial->lightMap) {

            uv2ScaleMap = lightMaterial->lightMap;
        }

        if (uv2ScaleMap) {

            if (uv2ScaleMap->matrixAutoUpdate) {

                uv2ScaleMap->updateMatrix();
            }

            uniforms.at("uv2Transform").value<Matrix3>().copy(uv2ScaleMap->matrix);
        }
    }

    void refreshUniformsLambert(UniformMap& uniforms, MeshLambertMaterial* material) {

        auto& emissiveMap = material->emissiveMap;
        if (emissiveMap) {
            uniforms.at("emissiveMap").setValue(emissiveMap);
        }
    }

    void refreshUniformsPhong(UniformMap& uniforms, MeshPhongMaterial* material) {

        uniforms.at("specular").value<Color>().copy(material->specular);
        uniforms.at("shininess").value<float>() = std::max(material->shininess, (float) 1E-4);// to prevent pow( 0.0, 0.0 )

        if (material->emissiveMap) {

            uniforms.at("emissiveMap").setValue(material->emissiveMap);
        }

        if (material->bumpMap) {

            uniforms.at("bumpMap").setValue(material->bumpMap);
            uniforms.at("bumpScale").setValue(material->bumpScale);
            if (material->side == BackSide) {
                uniforms.at("bumpScale").value<float>() *= -1;
            }
        }

        if (material->normalMap) {

            uniforms.at("normalMap").setValue(material->normalMap);
            uniforms.at("normalScale").value<Vector2>().copy(material->normalScale);
            if (material->side == BackSide) {
                uniforms.at("normalScale").value<Vector2>().negate();
            }
        }

        if (material->displacementMap) {

            uniforms.at("displacementMap").setValue(material->displacementMap);
            uniforms.at("displacementScale").value<float>() = material->displacementScale;
            uniforms.at("displacementBias").value<float>() = material->displacementBias;
        }
    }

    void refreshUniformsStandard(UniformMap& uniforms, MeshStandardMaterial* material) {

        uniforms.at("roughness").value<float>() = material->roughness;
        uniforms.at("metalness").value<float>() = material->metalness;

        if (material->roughnessMap) {

            uniforms.at("roughnessMap").setValue(material->roughnessMap);

        }

        if (material->metalnessMap) {

            uniforms.at("metalnessMap").setValue(material->metalnessMap);

        }

        if (material->emissiveMap) {

            uniforms.at("emissiveMap").setValue(material->emissiveMap);
        }

        if (material->bumpMap) {

            uniforms.at("bumpMap").setValue(material->bumpMap);
            uniforms.at("bumpScale").setValue(material->bumpScale);
            if (material->side == BackSide) {
                uniforms.at("bumpScale").value<float>() *= -1;
            }
        }

        if (material->normalMap) {

            uniforms.at("normalMap").setValue(material->normalMap);
            uniforms.at("normalScale").value<Vector2>().copy(material->normalScale);
            if (material->side == BackSide) {
                uniforms.at("normalScale").value<Vector2>().negate();
            }
        }

        if (material->displacementMap) {

            uniforms.at("displacementMap").setValue(material->displacementMap);
            uniforms.at("displacementScale").value<float>() = material->displacementScale;
            uniforms.at("displacementBias").value<float>() = material->displacementBias;
        }

        // TODO envMap
    }

    void refreshUniformsMatcap( UniformMap& uniforms, MeshMatcapMaterial* material ) {

        if ( material->matcap ) {

            uniforms.at("matcap").setValue(material->matcap);

        }

        if (material->bumpMap) {

            uniforms.at("bumpMap").setValue(material->bumpMap);
            uniforms.at("bumpScale").setValue(material->bumpScale);
            if (material->side == BackSide) {
                uniforms.at("bumpScale").value<float>() *= -1;
            }
        }

        if (material->normalMap) {

            uniforms.at("normalMap").setValue(material->normalMap);
            uniforms.at("normalScale").value<Vector2>().copy(material->normalScale);
            if (material->side == BackSide) {
                uniforms.at("normalScale").value<Vector2>().negate();
            }
        }

        if (material->displacementMap) {

            uniforms.at("displacementMap").setValue(material->displacementMap);
            uniforms.at("displacementScale").value<float>() = material->displacementScale;
            uniforms.at("displacementBias").value<float>() = material->displacementBias;
        }

    }

    void refreshUniformsDepth( UniformMap& uniforms, MeshDepthMaterial* material ) {

        if ( material->displacementMap ) {

            uniforms.at("displacementMap").setValue(material->displacementMap);
            uniforms.at("displacementScale").value<float>() = material->displacementScale;
            uniforms.at("displacementBias").value<float>() = material->displacementBias;

        }

    }

    void refreshUniformsDistance( UniformMap&uniforms, MeshDistanceMaterial* material ) {

        if ( material->displacementMap ) {

            uniforms.at("displacementMap").setValue(material->displacementMap);
            uniforms.at("displacementScale").value<float>() = material->displacementScale;
            uniforms.at("displacementBias").value<float>() = material->displacementBias;

        }

        uniforms.at("referencePosition").value<Vector3>().copy( material->referencePosition );
        uniforms.at("nearDistance").value<float>() = material->nearDistance;
        uniforms.at("farDistance").value<float>() = material->farDistance;

    }

    void refreshUniformsToon(UniformMap& uniforms, MeshToonMaterial* material) {

        auto& gradientMap = material->gradientMap;
        if (gradientMap) {

            uniforms.at("gradientMap").setValue(gradientMap);
        }

        auto& emissiveMap = material->emissiveMap;
        if (emissiveMap) {

            uniforms.at("emissiveMap").setValue(emissiveMap);
        }

        auto& bumpMap = material->bumpMap;
        if (bumpMap) {

            uniforms.at("bumpMap").setValue(bumpMap);
            uniforms.at("bumpScale").value<float>() = material->bumpScale;
            if (material->side == BackSide) {
                uniforms.at("bumpScale").value<float>() *= -1;
            }
        }

        auto& normalMap = material->normalMap;
        if (normalMap) {

            uniforms.at("normalMap").setValue(normalMap);
            uniforms.at("normalScale").value<Vector2>().copy(material->normalScale);
            if (material->side == BackSide) {
                uniforms.at("normalScale").value<Vector2>().negate();
            }
        }

        auto& displacementMap = material->displacementMap;
        if (displacementMap) {

            uniforms.at("displacementMap").setValue(displacementMap);
            uniforms.at("displacementScale").value<float>() = material->displacementScale;
            uniforms.at("displacementBias").value<float>() = material->displacementBias;
        }
    }


    void refreshUniformsLine(UniformMap& uniforms, LineBasicMaterial* material) {

        uniforms.at("diffuse").value<Color>().copy(material->color);
        uniforms.at("opacity").value<float>() = material->opacity;
    }

    void refreshUniformsPoints(UniformMap& uniforms, PointsMaterial* material, int pixelRatio, float height) {

        uniforms.at("diffuse").value<Color>().copy(material->color);
        uniforms.at("opacity").value<float>() = material->opacity;
        uniforms.at("size").value<float>() = material->size * (float) pixelRatio;
        uniforms.at("scale").value<float>() = height * 0.5f;
    }

    void refreshUniformsSprites(UniformMap& uniforms, SpriteMaterial* material) {
        uniforms.at("diffuse").value<Color>().copy(material->color);
        uniforms.at("opacity").value<float>() = material->opacity;
        uniforms.at("rotation").value<float>() = material->rotation;

        if (material->map) {

            uniforms.at("map").setValue(material->map);
        }

        if (material->alphaMap) {

            uniforms.at("alphaMap").setValue(material->alphaMap);
        }

        // uv repeat and offset setting priorities
        // 1. color map
        // 2. alpha map

        std::shared_ptr<Texture> uvScaleMap = nullptr;

        if (material->map) {

            uvScaleMap = material->map;

        } else if (material->alphaMap) {

            uvScaleMap = material->alphaMap;
        }

        if (uvScaleMap) {

            if (uvScaleMap->matrixAutoUpdate) {

                uvScaleMap->updateMatrix();
            }

            uniforms.at("uvTransform").value<Matrix3>().copy(uvScaleMap->matrix);
        }
    }

}// namespace

void GLMaterials::refreshFogUniforms(UniformMap& uniforms, FogVariant& fog) {

    if (fog.index() == 0) {

        auto& f = std::get<Fog>(fog);
        uniforms.at("fogColor").value<Color>().copy(f.color);

        uniforms.at("fogNear").value<float>() = f.near;
        uniforms.at("fogFar").value<float>() = f.far;
    } else {

        auto& f = std::get<FogExp2>(fog);
        uniforms.at("fogColor").value<Color>().copy(f.color);

        uniforms.at("fogDensity").value<float>() = f.density;
    }
}

void GLMaterials::refreshMaterialUniforms(UniformMap& uniforms, Material* material, int pixelRatio, int height) {

    if (material->is<MeshBasicMaterial>()) {

        refreshUniformsCommon(uniforms, material, properties);

    } else if (material->is<MeshLambertMaterial>()) {

        auto m = material->as<MeshLambertMaterial>().get();
        refreshUniformsCommon(uniforms, m, properties);
        refreshUniformsLambert(uniforms, m);

    } else if (material->is<MeshToonMaterial>()) {

        auto m = material->as<MeshToonMaterial>().get();
        refreshUniformsCommon(uniforms, m, properties);
        refreshUniformsToon(uniforms, m);

    } else if (material->is<MeshPhongMaterial>()) {

        auto m = material->as<MeshPhongMaterial>().get();
        refreshUniformsCommon(uniforms, m, properties);
        refreshUniformsPhong(uniforms, m);

    } else if (material->is<MeshStandardMaterial>()) {

        auto m = material->as<MeshStandardMaterial>().get();
        refreshUniformsCommon(uniforms, material, properties);
        refreshUniformsStandard(uniforms, m);

    } else if (material->is<MeshMatcapMaterial>()) {

        auto m = material->as<MeshMatcapMaterial>().get();
        refreshUniformsCommon(uniforms, m, properties);
        refreshUniformsMatcap(uniforms, m);

    } else if (material->is<MeshDepthMaterial>()) {

        auto m = material->as<MeshDepthMaterial>().get();
        refreshUniformsCommon(uniforms, m, properties);
        refreshUniformsDepth(uniforms, m);

    } else if (material->is<MeshDistanceMaterial>()) {

        auto m = material->as<MeshDistanceMaterial>().get();
        refreshUniformsCommon(uniforms, m, properties);
        refreshUniformsDistance(uniforms, m);

    } else if (material->is<LineBasicMaterial>()) {

        refreshUniformsLine(uniforms, material->as<LineBasicMaterial>().get());

    } else if (material->is<PointsMaterial>()) {

        refreshUniformsPoints(uniforms, material->as<PointsMaterial>().get(), pixelRatio, static_cast<float>(height));

    } else if (material->is<SpriteMaterial>()) {

        auto m = material->as<SpriteMaterial>();
        refreshUniformsSprites(uniforms, material->as<SpriteMaterial>().get());

    } else if (material->is<ShaderMaterial>()) {

        material->as<ShaderMaterial>()->uniformsNeedUpdate = false;
    }
}
