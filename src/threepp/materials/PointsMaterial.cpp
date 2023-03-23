
#include "threepp/materials/PointsMaterial.hpp"

using namespace threepp;

PointsMaterial::PointsMaterial()
    : MaterialWithColor(0xffffff),
      MaterialWithSize(1, true) {}


std::string PointsMaterial::type() const {

    return "PointsMaterial";
}

std::shared_ptr<Material> PointsMaterial::clone() const {

    auto m = create();
    copyInto(m.get());

    m->color.copy(color);

    m->map = map;

    m->alphaMap = alphaMap;

    m->size = size;
    m->sizeAttenuation = sizeAttenuation;

    return m;
}

std::shared_ptr<PointsMaterial> PointsMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<PointsMaterial>(new PointsMaterial());
    m->setValues(values);

    return m;
}

bool PointsMaterial::setValue(const std::string& key, const MaterialValue& value) {

    if (key == "color") {

        if (std::holds_alternative<int>(value)) {
            color = std::get<int>(value);
        } else {
            color.copy(std::get<Color>(value));
        }

        return true;
    } else if (key == "size") {

        size = std::get<float>(value);
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
