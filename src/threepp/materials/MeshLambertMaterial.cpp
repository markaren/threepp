
#include "threepp/materials/MeshLambertMaterial.hpp"

using namespace threepp;

MeshLambertMaterial::MeshLambertMaterial()
    : MaterialWithColor(0xffffff),
      MaterialWithLightMap(1),
      MaterialWithAoMap(1),
      MaterialWithEmissive(0x000000, 1),
      MaterialWithReflectivity(1, 0.98f),
      MaterialWithWireframe(false, 1),
      MaterialWithCombine(CombineOperation::Multiply) {}


std::string MeshLambertMaterial::type() const {

    return "MeshLambertMaterial";
}

void MeshLambertMaterial::copyInto(threepp::Material& material) const {

    Material::copyInto(material);

    auto m = material.as<MeshLambertMaterial>();

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
}

std::shared_ptr<Material> MeshLambertMaterial::createDefault() const {

    return std::shared_ptr<MeshLambertMaterial>(new MeshLambertMaterial());
}

std::shared_ptr<MeshLambertMaterial> MeshLambertMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<MeshLambertMaterial>(new MeshLambertMaterial());
    m->setValues(values);

    return m;
}

std::shared_ptr<MeshLambertMaterial> MeshLambertMaterial::create(const Params& p) {

    auto m = std::shared_ptr<MeshLambertMaterial>(new MeshLambertMaterial());

    p.applyBaseTo(*m);

    // Apply only the fields the caller set; everything else keeps the constructor default.
    // Params stores each value in a `field_` member; the material's field is `field`.
#define TPP_SET(field) \
    if (p.field##_) m->field = *p.field##_;
#define TPP_TEX(field) \
    if (p.field##_) m->field = p.field##_;

    TPP_SET(color)
    TPP_SET(emissive)
    TPP_TEX(map)
    TPP_TEX(aoMap)
    TPP_SET(aoMapIntensity)
    TPP_TEX(alphaMap)
    TPP_TEX(specularMap)
    TPP_TEX(lightMap)
    TPP_SET(lightMapIntensity)
    TPP_SET(wireframe)
    TPP_SET(wireframeLinewidth)
    TPP_TEX(envMap)
    TPP_SET(combine)
    TPP_SET(reflectivity)
    TPP_SET(refractionRatio)

#undef TPP_SET
#undef TPP_TEX

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
