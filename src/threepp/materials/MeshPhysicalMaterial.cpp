
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

std::shared_ptr<MeshPhysicalMaterial> MeshPhysicalMaterial::create(const Params& p) {

    auto m = std::shared_ptr<MeshPhysicalMaterial>(new MeshPhysicalMaterial());

    p.applyBaseTo(*m);

    // Apply only the fields the caller set; everything else keeps the constructor default.
    // Params stores each value in a `field_` member; the material's field is `field`.
#define TPP_SET(field) \
    if (p.field##_) m->field = *p.field##_;
#define TPP_TEX(field) \
    if (p.field##_) m->field = p.field##_;

    // --- Standard (inherited) fields ---
    TPP_SET(color)
    TPP_SET(roughness)
    TPP_SET(metalness)
    TPP_TEX(map)
    TPP_TEX(roughnessMap)
    TPP_TEX(metalnessMap)
    TPP_SET(emissive)
    TPP_SET(emissiveIntensity)
    TPP_TEX(emissiveMap)
    TPP_TEX(normalMap)
    TPP_SET(normalMapType)
    TPP_SET(normalScale)
    TPP_TEX(bumpMap)
    TPP_SET(bumpScale)
    TPP_TEX(aoMap)
    TPP_SET(aoMapIntensity)
    TPP_TEX(displacementMap)
    TPP_SET(displacementScale)
    TPP_SET(displacementBias)
    TPP_TEX(alphaMap)
    TPP_TEX(lightMap)
    TPP_SET(lightMapIntensity)
    TPP_TEX(envMap)
    TPP_SET(envMapIntensity)
    // refractionRatio is inherited via two base paths (MeshStandardMaterial and
    // MaterialWithReflectivity -> MaterialWithRefractionRatio), so an unqualified m->refractionRatio
    // is ambiguous here. Qualify through the MeshStandardMaterial path to disambiguate.
    if (p.refractionRatio_) m->MeshStandardMaterial::refractionRatio = *p.refractionRatio_;
    TPP_SET(wireframe)
    TPP_SET(wireframeLinewidth)
    TPP_SET(flatShading)
    TPP_SET(vertexTangents)

    // --- Physical-specific fields ---
    TPP_SET(reflectivity)
    // ior is special: setIor() also recomputes reflectivity (preserve the map-based side effect).
    if (p.ior_) m->setIor(*p.ior_);
    TPP_SET(clearcoat)
    TPP_TEX(clearcoatMap)
    TPP_SET(clearcoatRoughness)
    TPP_TEX(clearcoatRoughnessMap)
    TPP_TEX(clearcoatNormalMap)
    TPP_SET(dispersion)
    TPP_SET(transmission)
    TPP_TEX(transmissionMap)
    TPP_SET(thickness)
    TPP_TEX(thicknessMap)
    TPP_SET(attenuationDistance)
    TPP_SET(attenuationColor)
    TPP_SET(iridescence)
    TPP_SET(iridescenceIOR)
    TPP_SET(iridescenceThicknessNm)

#undef TPP_SET
#undef TPP_TEX

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

    m->iridescence = iridescence;
    m->iridescenceIOR = iridescenceIOR;
    m->iridescenceThicknessNm = iridescenceThicknessNm;
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

    } else if (key == "iridescence") {

        iridescence = extractFloat(value);
        return true;

    } else if (key == "iridescenceIOR") {

        iridescenceIOR = extractFloat(value);
        return true;

    } else if (key == "iridescenceThicknessNm") {

        iridescenceThicknessNm = extractFloat(value);
        return true;
    }

    return MeshStandardMaterial::setValue(key, value);
}
