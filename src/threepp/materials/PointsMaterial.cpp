
#include "threepp/materials/PointsMaterial.hpp"

using namespace threepp;

PointsMaterial::PointsMaterial()
    : MaterialWithColor(0xffffff),
      MaterialWithSize(1, true) {}


std::string PointsMaterial::type() const {

    return "PointsMaterial";
}

void PointsMaterial::copyInto(Material& material) const {

    Material::copyInto(material);

    auto m = material.as<PointsMaterial>();

    m->color.copy(color);

    m->map = map;

    m->alphaMap = alphaMap;

    m->size = size;
    m->sizeAttenuation = sizeAttenuation;
}

std::shared_ptr<Material> PointsMaterial::createDefault() const {

    return std::shared_ptr<PointsMaterial>(new PointsMaterial());
}

std::shared_ptr<PointsMaterial> PointsMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<PointsMaterial>(new PointsMaterial());
    m->setValues(values);

    return m;
}

bool PointsMaterial::setValue(const std::string& key, const MaterialValue& value) {

    if (key == "color") {

        color.copy(extractColor(value));

        return true;
    } else if (key == "size") {

        size = extractFloat(value);
        return true;

    } else if (key == "sizeAttenuation") {

        sizeAttenuation = std::get<bool>(value);
        return true;

    } else if (key == "map") {

        map = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "alphaMap") {

        alphaMap = std::get<std::shared_ptr<Texture>>(value);
        return true;
    }

    return false;
}
