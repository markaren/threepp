
#include "threepp/materials/MeshStandardMaterial.hpp"

using namespace threepp;


MeshStandardMaterial::MeshStandardMaterial()
    : MaterialWithColor(0xffffff),
      MaterialWithWireframe(false, 1),
      MaterialWithRoughness(1),
      MaterialWithMetalness(0),
      MaterialWithLightMap(1),
      MaterialWithAoMap(1),
      MaterialWithEmissive(0x000000, 1),
      MaterialWithBumpMap(1),
      MaterialWithEnvMap(1.f),
      MaterialWithDisplacementMap(1, 0),
      MaterialWithReflectivityRatio(0.98),
      MaterialWithNormalMap(TangentSpaceNormalMap, {1, 1}),
      MaterialWithVertexTangents(false),
      MaterialWithFlatShading(false) {

    defines["STANDARD"] = "";
}


std::string MeshStandardMaterial::type() const {

    return "MeshStandardMaterial";
}

std::shared_ptr<MeshStandardMaterial> MeshStandardMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<MeshStandardMaterial>(new MeshStandardMaterial());
    m->setValues(values);

    return m;
}

std::shared_ptr<Material> MeshStandardMaterial::clone() const {

    auto m = create();
    copyInto(m.get());

    m->defines["STANDARD"] = "";

    m->color.copy(color);
    m->roughness = roughness;
    m->metalness = metalness;

    m->map = map;

    m->lightMap = lightMap;
    m->lightMapIntensity = lightMapIntensity;

    m->aoMap = aoMap;
    m->aoMapIntensity = aoMapIntensity;

    m->emissive.copy(emissive);
    m->emissiveMap = emissiveMap;
    m->emissiveIntensity = emissiveIntensity;

    m->bumpMap = bumpMap;
    m->bumpScale = bumpScale;

    m->normalMap = normalMap;
    m->normalMapType = normalMapType;
    m->normalScale.copy(normalScale);

    m->displacementMap = displacementMap;
    m->displacementScale = displacementScale;
    m->displacementBias = displacementBias;

    m->roughnessMap = roughnessMap;

    m->metalnessMap = metalnessMap;

    m->alphaMap = alphaMap;

    m->envMap = envMap;
    m->envMapIntensity = envMapIntensity;

    m->refractionRatio = refractionRatio;

    m->wireframe = wireframe;
    m->wireframeLinewidth = wireframeLinewidth;

    m->flatShading = flatShading;

    m->vertexTangents = vertexTangents;

    return m;
}

bool MeshStandardMaterial::setValue(const std::string& key, const MaterialValue& value) {

    if (key == "color") {

        if (std::holds_alternative<int>(value)) {
            color = std::get<int>(value);
        } else {
            color.copy(std::get<Color>(value));
        }

        return true;

    } else if (key == "emissive") {

        if (std::holds_alternative<int>(value)) {
            emissive = std::get<int>(value);
        } else {
            emissive.copy(std::get<Color>(value));
        }

        return true;

    } else if (key == "emissiveIntensity") {

        emissiveIntensity = std::get<float>(value);
        return true;

    } else if (key == "emissiveMap") {

        emissiveMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "bumpMap") {

        bumpMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "bumpScale") {

        bumpScale = std::get<float>(value);
        return true;

    } else if (key == "lightMap") {

        lightMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "lightMapIntensity") {

        lightMapIntensity = std::get<float>(value);
        return true;

    } else if (key == "aoMap") {

        aoMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "aoMapIntensity") {

        aoMapIntensity = std::get<float>(value);
        return true;

    } else if (key == "normalMap") {

        normalMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "normalMapType") {

        normalMapType = std::get<int>(value);
        return true;

    } else if (key == "displacementMap") {

        displacementMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "displacementScale") {

        displacementScale = std::get<float>(value);
        return true;

    } else if (key == "displacementBias") {

        displacementBias = std::get<float>(value);
        return true;

    } else if (key == "alphaMap") {

        alphaMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "roughnessMap") {

        roughnessMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "roughness") {

        roughness = std::get<float>(value);
        return true;

    } else if (key == "metalnessMap") {

        metalnessMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "metalness") {

        metalness = std::get<float>(value);
        return true;

    } else if (key == "envMap") {

        envMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "envMapIntensity") {

        envMapIntensity = std::get<float>(value);
        return true;

    } else if (key == "refractionRatio") {

        refractionRatio = std::get<float>(value);
        return true;

    } else if (key == "wireframe") {

        wireframe = std::get<bool>(value);
        return true;

    } else if (key == "wireframeLinewidth") {

        wireframeLinewidth = std::get<float>(value);
        return true;

    } else if (key == "flatShading") {

        flatShading = std::get<bool>(value);
        return true;

    } else if (key == "refractionRatio") {

        vertexTangents = std::get<bool>(value);
        return true;
    }

    return false;
}
