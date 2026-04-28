
#include "threepp/materials/MeshPhysicalMaterial.hpp"

using namespace threepp;


MeshPhysicalMaterial::MeshPhysicalMaterial()
    : MaterialWithReflectivity(0.5f, 0.98f) {

    defines["STANDARD"] = "";
    defines["PHYSICAL"] = "";

    clearcoat = 0.f;
    clearcoatRoughness = 0.f;

    transmission = 0.f;

    thickness = 0.01f;

    attenuationDistance = 0.f;
    attenuationColor = Color(1, 1, 1);
}


std::string MeshPhysicalMaterial::type() const {

    return "MeshPhysicalMaterial";
}

std::shared_ptr<Material> MeshPhysicalMaterial::createDefault() const {

    return std::shared_ptr<MeshPhysicalMaterial>(new MeshPhysicalMaterial());
}

std::shared_ptr<MeshPhysicalMaterial> MeshPhysicalMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<MeshPhysicalMaterial>(new MeshPhysicalMaterial());
    m->setValues(values);

    return m;
}

void MeshPhysicalMaterial::copyInto(Material& material) const {

    MeshStandardMaterial::copyInto(material);

    auto m = material.as<MeshPhysicalMaterial>();

    m->defines["STANDARD"] = "";
    m->defines["PHYSICAL"] = "";

    m->reflectivity = reflectivity;

    m->clearcoat = clearcoat;
    m->clearcoatMap = clearcoatMap;
    m->clearcoatRoughness = clearcoatRoughness;
    m->clearcoatRoughnessMap = clearcoatRoughnessMap;
    m->clearcoatNormalScale.copy(clearcoatNormalScale);
    m->clearcoatNormalMap = clearcoatNormalMap;

    m->sheen = sheen;

    m->transmission = transmission;
    m->transmissionMap = transmissionMap;
    m->dispersion = dispersion;

    m->thickness = thickness;
    m->thicknessMap = thicknessMap;

    m->attenuationDistance = attenuationDistance;
    m->attenuationColor.copy(attenuationColor);
}

bool MeshPhysicalMaterial::setValue(const std::string& key, const MaterialValue& value) {

    if (key == "reflectivity") {

        reflectivity = extractFloat(value);
        return true;

    } else if (key == "ior") {

        setIor(extractFloat(value));
        return true;

    } else if (key == "clearcoat") {

        clearcoat = extractFloat(value);
        return true;

    } else if (key == "clearcoatMap") {

        clearcoatMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "clearcoatRoughness") {

        clearcoatRoughness = extractFloat(value);
        return true;

    } else if (key == "clearcoatRoughnessMap") {

        clearcoatRoughnessMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "clearcoatNormalMap") {

        clearcoatNormalMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "dispersion") {

        dispersion = extractFloat(value);
        return true;

    } else if (key == "transmission") {

        transmission = extractFloat(value);
        return true;

    } else if (key == "transmissionMap") {

        transmissionMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "thickness") {

        thickness = extractFloat(value);
        return true;

    } else if (key == "thicknessMap") {

        thicknessMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "attenuationDistance") {

        attenuationDistance = extractFloat(value);
        return true;

    } else if (key == "attenuationColor") {

        attenuationColor.copy(extractColor(value));
        return true;
    }

    return MeshStandardMaterial::setValue(key, value);
}
