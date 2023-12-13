
#include "threepp/materials/MeshLambertMaterial.hpp"

using namespace threepp;

MeshLambertMaterial::MeshLambertMaterial()
    : MaterialWithColor(0xffffff),
      MaterialWithWireframe(false, 1),
      MaterialWithReflectivity(1, 0.98f),
      MaterialWithLightMap(1),
      MaterialWithEmissive(0x000000, 1),
      MaterialWithAoMap(1),
      MaterialWithCombine(CombineOperation::Multiply) {}


std::string MeshLambertMaterial::type() const {

    return "MeshLambertMaterial";
}

std::shared_ptr<Material> MeshLambertMaterial::clone() const {

    auto m = create();
    copyInto(m.get());

    m->color.copy(color);

    m->map = map;

    m->lightMap = lightMap;
    m->lightMapIntensity = lightMapIntensity;

    m->aoMap = aoMap;
    m->aoMapIntensity = aoMapIntensity;

    m->emissive.copy(emissive);
    m->emissiveMap = emissiveMap;
    m->emissiveIntensity = emissiveIntensity;

    m->specularMap = specularMap;

    m->alphaMap = alphaMap;

    m->envMap = envMap;
    m->combine = combine;
    m->reflectivity = reflectivity;
    m->refractionRatio = refractionRatio;

    m->wireframe = wireframe;
    m->wireframeLinewidth = wireframeLinewidth;

    return m;
}

std::shared_ptr<MeshLambertMaterial> MeshLambertMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<MeshLambertMaterial>(new MeshLambertMaterial());
    m->setValues(values);

    return m;
}

bool MeshLambertMaterial::setValue(const std::string& key, const MaterialValue& value) {

    if (key == "color") {

        color.copy(extractColor(value));

        return true;

    } else if (key == "emissive") {

        emissive.copy(extractColor(value));

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

    } else if (key == "alphaMap") {

        alphaMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "specularMap") {

        specularMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "lightMap") {

        lightMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "lightMapIntensity") {

        lightMapIntensity = extractFloat(value);
        return true;

    } else if (key == "wireframe") {

        wireframe = std::get<bool>(value);
        return true;

    } else if (key == "wireframeLinewidth") {

        wireframeLinewidth = extractFloat(value);
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
    }

    return false;
}
