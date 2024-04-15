
#include "threepp/materials/MeshPhongMaterial.hpp"

using namespace threepp;

MeshPhongMaterial::MeshPhongMaterial()
    : MaterialWithColor(0xffffff),
      MaterialWithCombine(CombineOperation::Multiply),
      MaterialWithFlatShading(false),
      MaterialWithSpecular(0x111111, 30),
      MaterialWithLightMap(1),
      MaterialWithAoMap(1),
      MaterialWithEmissive(0x000000, 1),
      MaterialWithBumpMap(1),
      MaterialWithNormalMap(NormalMapType::TangentSpace, {1, 1}),
      MaterialWithDisplacementMap(1, 0),
      MaterialWithReflectivity(1, 0.98f),
      MaterialWithWireframe(false, 1) {}


std::string MeshPhongMaterial::type() const {

    return "MeshPhongMaterial";
}

void MeshPhongMaterial::copyInto(threepp::Material& material) const {

    Material::copyInto(material);

    auto m = material.as<MeshPhongMaterial>();

    m->color.copy(color);
    m->specular.copy(specular);
    m->shininess = shininess;

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

    m->specularMap = specularMap;

    m->alphaMap = alphaMap;

    m->envMap = envMap;
    m->combine = combine;
    m->reflectivity = reflectivity;
    m->refractionRatio = refractionRatio;

    m->wireframe = wireframe;
    m->wireframeLinewidth = wireframeLinewidth;

    m->flatShading = flatShading;
}

std::shared_ptr<Material> MeshPhongMaterial::createDefault() const {

    return std::shared_ptr<MeshPhongMaterial>(new MeshPhongMaterial());
}

std::shared_ptr<MeshPhongMaterial> MeshPhongMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<MeshPhongMaterial>(new MeshPhongMaterial());
    m->setValues(values);

    return m;
}

bool MeshPhongMaterial::setValue(const std::string& key, const MaterialValue& value) {

    if (key == "color") {

        color.copy(extractColor(value));
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

    } else if (key == "wireframe") {

        wireframe = std::get<bool>(value);
        return true;

    } else if (key == "wireframeLinewidth") {

        wireframeLinewidth = extractFloat(value);
        return true;

    } else if (key == "flatShading") {

        flatShading = std::get<bool>(value);
        return true;

    } else if (key == "map") {

        map = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "aoMap") {

        aoMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "aoMapIntensity") {

        aoMapIntensity = extractFloat(value);
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

    } else if (key == "normalMap") {

        normalMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "normalMapType") {

        normalMapType = std::get<NormalMapType>(value);
        return true;

    } else if (key == "alphaMap") {

        alphaMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "specularMap") {

        specularMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "specular") {

        specular.copy(extractColor(value));
        return true;

    } else if (key == "displacementMap") {

        displacementMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "displacementBias") {

        displacementBias = extractFloat(value);
        return true;

    } else if (key == "displacementScale") {

        displacementScale = extractFloat(value);
        return true;

    } else if (key == "shininess") {

        shininess = extractFloat(value);
        return true;

    } else if (key == "envMap") {

        envMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "combine") {

        combine = std::get<CombineOperation>(value);
        return true;

    } else if (key == "reflectivity") {

        reflectivity = extractFloat(value);
        return true;

    } else if (key == "refractionRatio") {

        refractionRatio = extractFloat(value);
        return true;

    } else if (key == "normalScale") {

        normalScale = std::get<Vector2>(value);
        return true;
    }

    return false;
}
