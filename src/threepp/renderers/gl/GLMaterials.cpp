
#include "threepp/renderers/gl/GLMaterials.hpp"

#include "threepp/renderers/gl/GLProperties.hpp"

#include "threepp/materials/MeshDepthMaterial.hpp"
#include "threepp/materials/MeshDistanceMaterial.hpp"
#include "threepp/materials/MeshMatcapMaterial.hpp"
#include "threepp/materials/MeshToonMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/materials.hpp"

using namespace threepp;
using namespace threepp::gl;

struct GLMaterials::Impl {

    GLProperties& properties;

    explicit Impl(GLProperties& properties): properties(properties) {}

    void refreshUniformsCommon(UniformMap& uniforms, Material* material) {

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

            uniforms.at("map").setValue(mapMaterial->map.get());
        }

        if (alphaMaterial && alphaMaterial->alphaMap) {

            uniforms.at("alphaMap").setValue(alphaMaterial->alphaMap.get());
        }

        if (specularMaterial && specularMaterial->specularMap) {

            uniforms.at("specularMap").setValue(specularMaterial->specularMap.get());
        }

        auto envMap = properties.materialProperties.get(material->uuid())->envMap;
        if (envMap) {

            uniforms.at("envMap").setValue(envMap.get());
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
            uniforms.at("lightMap").setValue(lightMaterial->lightMap.get());
            uniforms.at("lightMapIntensity").setValue(lightMaterial->lightMapIntensity);
        }

        if (aoMaterial && aoMaterial->aoMap) {
            uniforms.at("aoMap").setValue(aoMaterial->aoMap.get());
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
            uniforms.at("emissiveMap").setValue(emissiveMap.get());
        }
    }

    void refreshUniformsPhong(UniformMap& uniforms, MeshPhongMaterial* material) {

        uniforms.at("specular").value<Color>().copy(material->specular);
        uniforms.at("shininess").value<float>() = std::max(material->shininess, (float) 1E-4);// to prevent pow( 0.0, 0.0 )

        if (material->emissiveMap) {

            uniforms.at("emissiveMap").setValue(material->emissiveMap.get());
        }

        if (material->bumpMap) {

            uniforms.at("bumpMap").setValue(material->bumpMap.get());
            uniforms.at("bumpScale").setValue(material->bumpScale);
            if (material->side == Side::Back) {
                uniforms.at("bumpScale").value<float>() *= -1;
            }
        }

        if (material->normalMap) {

            uniforms.at("normalMap").setValue(material->normalMap.get());
            uniforms.at("normalScale").value<Vector2>().copy(material->normalScale);
            if (material->side == Side::Back) {
                uniforms.at("normalScale").value<Vector2>().negate();
            }
        }

        if (material->displacementMap) {

            uniforms.at("displacementMap").setValue(material->displacementMap.get());
            uniforms.at("displacementScale").value<float>() = material->displacementScale;
            uniforms.at("displacementBias").value<float>() = material->displacementBias;
        }
    }

    void refreshUniformsStandard(UniformMap& uniforms, MeshStandardMaterial* material) {

        uniforms.at("roughness").value<float>() = material->roughness;
        uniforms.at("metalness").value<float>() = material->metalness;

        if (material->roughnessMap) {

            uniforms.at("roughnessMap").setValue(material->roughnessMap.get());
        }

        if (material->metalnessMap) {

            uniforms.at("metalnessMap").setValue(material->metalnessMap.get());
        }

        if (material->emissiveMap) {

            uniforms.at("emissiveMap").setValue(material->emissiveMap.get());
        }

        if (material->bumpMap) {

            uniforms.at("bumpMap").setValue(material->bumpMap.get());
            uniforms.at("bumpScale").setValue(material->bumpScale);
            if (material->side == Side::Back) {
                uniforms.at("bumpScale").value<float>() *= -1;
            }
        }

        if (material->normalMap) {

            uniforms.at("normalMap").setValue(material->normalMap.get());
            uniforms.at("normalScale").value<Vector2>().copy(material->normalScale);
            if (material->side == Side::Back) {
                uniforms.at("normalScale").value<Vector2>().negate();
            }
        }

        if (material->displacementMap) {

            uniforms.at("displacementMap").setValue(material->displacementMap.get());
            uniforms.at("displacementScale").value<float>() = material->displacementScale;
            uniforms.at("displacementBias").value<float>() = material->displacementBias;
        }

        // TODO envMap
    }

    void refreshUniformsMatcap(UniformMap& uniforms, MeshMatcapMaterial* material) {

        if (material->matcap) {

            uniforms.at("matcap").setValue(material->matcap.get());
        }

        if (material->bumpMap) {

            uniforms.at("bumpMap").setValue(material->bumpMap.get());
            uniforms.at("bumpScale").setValue(material->bumpScale);
            if (material->side == Side::Back) {
                uniforms.at("bumpScale").value<float>() *= -1;
            }
        }

        if (material->normalMap) {

            uniforms.at("normalMap").setValue(material->normalMap.get());
            uniforms.at("normalScale").value<Vector2>().copy(material->normalScale);
            if (material->side == Side::Back) {
                uniforms.at("normalScale").value<Vector2>().negate();
            }
        }

        if (material->displacementMap) {

            uniforms.at("displacementMap").setValue(material->displacementMap.get());
            uniforms.at("displacementScale").value<float>() = material->displacementScale;
            uniforms.at("displacementBias").value<float>() = material->displacementBias;
        }
    }

    void refreshUniformsDepth(UniformMap& uniforms, MeshDepthMaterial* material) {

        if (material->displacementMap) {

            uniforms.at("displacementMap").setValue(material->displacementMap.get());
            uniforms.at("displacementScale").value<float>() = material->displacementScale;
            uniforms.at("displacementBias").value<float>() = material->displacementBias;
        }
    }

    void refreshUniformsDistance(UniformMap& uniforms, MeshDistanceMaterial* material) {

        if (material->displacementMap) {

            uniforms.at("displacementMap").setValue(material->displacementMap.get());
            uniforms.at("displacementScale").value<float>() = material->displacementScale;
            uniforms.at("displacementBias").value<float>() = material->displacementBias;
        }

        uniforms.at("referencePosition").value<Vector3>().copy(material->referencePosition);
        uniforms.at("nearDistance").value<float>() = material->nearDistance;
        uniforms.at("farDistance").value<float>() = material->farDistance;
    }

    void refreshUniformsToon(UniformMap& uniforms, MeshToonMaterial* material) {

        auto& gradientMap = material->gradientMap;
        if (gradientMap) {

            uniforms.at("gradientMap").setValue(gradientMap.get());
        }

        auto& emissiveMap = material->emissiveMap;
        if (emissiveMap) {

            uniforms.at("emissiveMap").setValue(emissiveMap.get());
        }

        auto& bumpMap = material->bumpMap;
        if (bumpMap) {

            uniforms.at("bumpMap").setValue(bumpMap.get());
            uniforms.at("bumpScale").value<float>() = material->bumpScale;
            if (material->side == Side::Back) {
                uniforms.at("bumpScale").value<float>() *= -1;
            }
        }

        auto& normalMap = material->normalMap;
        if (normalMap) {

            uniforms.at("normalMap").setValue(normalMap.get());
            uniforms.at("normalScale").value<Vector2>().copy(material->normalScale);
            if (material->side == Side::Back) {
                uniforms.at("normalScale").value<Vector2>().negate();
            }
        }

        auto& displacementMap = material->displacementMap;
        if (displacementMap) {

            uniforms.at("displacementMap").setValue(displacementMap.get());
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
        uniforms.at("size").value<float>() = material->size * static_cast<float>(pixelRatio);
        uniforms.at("scale").value<float>() = height * 0.5f;

        if (material->map) {

            uniforms.at("map").setValue(material->map.get());
        }

        if (material->alphaMap) {

            uniforms.at("alphaMap").setValue(material->alphaMap.get());
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

    void refreshUniformsSprites(UniformMap& uniforms, SpriteMaterial* material) {
        uniforms.at("diffuse").value<Color>().copy(material->color);
        uniforms.at("opacity").value<float>() = material->opacity;
        uniforms.at("rotation").value<float>() = material->rotation;

        if (material->map) {

            uniforms.at("map").setValue(material->map.get());
        }

        if (material->alphaMap) {

            uniforms.at("alphaMap").setValue(material->alphaMap.get());
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

    void refreshFogUniforms(UniformMap& uniforms, FogVariant& fog) {

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

    void refreshMaterialUniforms(UniformMap& uniforms, Material* material, int pixelRatio, int height) {

        const auto type = material->type();

        if (type == "MeshBasicMaterial") {

            refreshUniformsCommon(uniforms, material);

        } else if (type == "MeshLambertMaterial") {

            auto m = material->as<MeshLambertMaterial>().get();
            refreshUniformsCommon(uniforms, m);
            refreshUniformsLambert(uniforms, m);

        } else if (type == "MeshToonMaterial") {

            auto m = material->as<MeshToonMaterial>().get();
            refreshUniformsCommon(uniforms, m);
            refreshUniformsToon(uniforms, m);

        } else if (type == "MeshPhongMaterial") {

            auto m = material->as<MeshPhongMaterial>().get();
            refreshUniformsCommon(uniforms, m);
            refreshUniformsPhong(uniforms, m);

        } else if (type == "MeshStandardMaterial") {

            auto m = material->as<MeshStandardMaterial>().get();
            refreshUniformsCommon(uniforms, material);
            refreshUniformsStandard(uniforms, m);

        } else if (type == "MeshMatcapMaterial") {

            auto m = material->as<MeshMatcapMaterial>().get();
            refreshUniformsCommon(uniforms, m);
            refreshUniformsMatcap(uniforms, m);

        } else if (type == "MeshDepthMaterial") {

            auto m = material->as<MeshDepthMaterial>().get();
            refreshUniformsCommon(uniforms, m);
            refreshUniformsDepth(uniforms, m);

        } else if (type == "MeshDistanceMaterial") {

            auto m = material->as<MeshDistanceMaterial>();
            refreshUniformsCommon(uniforms, m.get());
            refreshUniformsDistance(uniforms, m.get());

        } else if (type == "LineBasicMaterial") {

            auto m = material->as<LineBasicMaterial>();
            refreshUniformsLine(uniforms, m.get());

        } else if (type == "PointsMaterial") {

            auto m = material->as<PointsMaterial>().get();
            refreshUniformsPoints(uniforms, m, pixelRatio, static_cast<float>(height));

        } else if (type == "ShadowMaterial") {

            auto m = material->as<ShadowMaterial>();
            uniforms.at("color").value<Color>().copy(m->color);
            uniforms.at("opacity").value<float>() = material->opacity;

        } else if (type == "SpriteMaterial") {

            auto m = material->as<SpriteMaterial>();
            refreshUniformsSprites(uniforms, m.get());


        } else if (type == "ShaderMaterial") {

            auto m = material->as<ShaderMaterial>();
            m->uniformsNeedUpdate = false;
        }
    }
};

void GLMaterials::refreshFogUniforms(UniformMap& uniforms, FogVariant& fog) {

    return pimpl_->refreshFogUniforms(uniforms, fog);
}

void GLMaterials::refreshMaterialUniforms(UniformMap& uniforms, Material* material, int pixelRatio, int height) {

    pimpl_->refreshMaterialUniforms(uniforms, material, pixelRatio, height);
}

GLMaterials::GLMaterials(GLProperties& properties)
    : pimpl_(std::make_unique<Impl>(properties)) {}

GLMaterials::~GLMaterials() = default;
