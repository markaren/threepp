
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
      MaterialWithNormalMap(NormalMapType::TangentSpace, {1, 1}),
      MaterialWithVertexTangents(false),
      MaterialWithFlatShading(false) {

    defines["STANDARD"] = "";
}


std::string MeshStandardMaterial::type() const {

    return "MeshStandardMaterial";
}

std::shared_ptr<Material> MeshStandardMaterial::createDefault() const {

    return std::shared_ptr<MeshStandardMaterial>(new MeshStandardMaterial());
}

std::shared_ptr<MeshStandardMaterial> MeshStandardMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<MeshStandardMaterial>(new MeshStandardMaterial());
    m->setValues(values);

    return m;
}

void MeshStandardMaterial::copyInto(Material& material) const {

    Material::copyInto(material);

    auto m = material.as<MeshStandardMaterial>();

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
}

bool MeshStandardMaterial::setValue(const std::string& key, const MaterialValue& value) {

    if (key == "color") {

        color.copy(extractColor(value));
        return true;

    } else if (key == "map") {

        map = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "emissive") {

        emissive.copy(extractColor(value));
        return true;

    } else if (key == "emissiveIntensity") {

        emissiveIntensity = extractFloat(value);
        return true;

    } else if (key == "emissiveMap") {

        emissiveMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "bumpMap") {

        bumpMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "bumpScale") {

        bumpScale = extractFloat(value);
        return true;

    } else if (key == "lightMap") {

        lightMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "lightMapIntensity") {

        lightMapIntensity = extractFloat(value);
        return true;

    } else if (key == "aoMap") {

        aoMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "aoMapIntensity") {

        aoMapIntensity = extractFloat(value);
        return true;

    } else if (key == "normalMap") {

        normalMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "normalMapType") {

        normalMapType = std::get<NormalMapType>(value);
        return true;

    } else if (key == "displacementMap") {

        displacementMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "displacementScale") {

        displacementScale = extractFloat(value);
        return true;

    } else if (key == "displacementBias") {

        displacementBias = extractFloat(value);
        return true;

    } else if (key == "alphaMap") {

        alphaMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "roughnessMap") {

        roughnessMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "roughness") {

        roughness = extractFloat(value);
        return true;

    } else if (key == "metalnessMap") {

        metalnessMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "metalness") {

        metalness = extractFloat(value);
        return true;

    } else if (key == "envMap") {

        envMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "envMapIntensity") {

        envMapIntensity = extractFloat(value);
        return true;

    } else if (key == "refractionRatio") {

        refractionRatio = extractFloat(value);
        return true;

    } else if (key == "wireframe") {

        wireframe = std::get<bool>(value);
        return true;

    } else if (key == "wireframeLinewidth") {

        wireframeLinewidth = extractFloat(value);
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
