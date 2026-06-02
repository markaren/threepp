
#include "threepp/materials/SpriteMaterial.hpp"

using namespace threepp;

SpriteMaterial::SpriteMaterial()
    : MaterialWithColor(0xffffff),
      MaterialWithSize(0, true) {
    transparent = true;
}

std::string SpriteMaterial::type() const {

    return "SpriteMaterial";
}

void SpriteMaterial::copyInto(Material& material) const {

    Material::copyInto(material);

    auto m = material.as<SpriteMaterial>();

    m->color.copy(color);

    m->map = map;

    m->alphaMap = alphaMap;

    m->rotation = rotation;

    m->sizeAttenuation = sizeAttenuation;
}


std::shared_ptr<Material> SpriteMaterial::createDefault() const {

    return std::shared_ptr<SpriteMaterial>(new SpriteMaterial());
}

std::shared_ptr<SpriteMaterial> SpriteMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<SpriteMaterial>(new SpriteMaterial());
    m->setValues(values);

    return m;
}

std::shared_ptr<SpriteMaterial> SpriteMaterial::create(const Params& p) {

    auto m = std::shared_ptr<SpriteMaterial>(new SpriteMaterial());

    p.applyBaseTo(*m);

    // Apply only the fields the caller set; everything else keeps the constructor default.
    // Params stores each value in a `field_` member; the material's field is `field`.
#define TPP_SET(field) \
    if (p.field##_) m->field = *p.field##_;
#define TPP_TEX(field) \
    if (p.field##_) m->field = p.field##_;

    TPP_SET(color)
    TPP_TEX(map)
    TPP_TEX(alphaMap)
    TPP_SET(rotation)
    TPP_SET(sizeAttenuation)

#undef TPP_SET
#undef TPP_TEX

    return m;
}

bool SpriteMaterial::setValue(const std::string& key, const MaterialValue& value) {

    if (key == "color") {

        color.copy(extractColor(value));
        return true;

    } else if (key == "map") {

        map = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "alphaMap") {

        alphaMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "rotation") {

        rotation = extractFloat(value);
        return true;

    } else if (key == "sizeAttenuation") {

        rotation = std::get<bool>(value);
        return true;
    }

    return false;
}
