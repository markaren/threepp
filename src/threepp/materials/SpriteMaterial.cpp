
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

std::shared_ptr<Material> SpriteMaterial::clone() const {

    auto m = create();
    copyInto(m.get());

    m->color.copy(color);

    m->map = map;

    m->alphaMap = alphaMap;

    m->rotation = rotation;

    m->sizeAttenuation = sizeAttenuation;

    return m;
}

std::shared_ptr<SpriteMaterial> SpriteMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<SpriteMaterial>(new SpriteMaterial());
    m->setValues(values);

    return m;
}

bool SpriteMaterial::setValue(const std::string& key, const MaterialValue& value) {

    if (key == "color") {

        if (std::holds_alternative<int>(value)) {
            color = std::get<int>(value);
        } else {
            color.copy(std::get<Color>(value));
        }

        return true;
    } else if (key == "map") {

        map = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "alphaMap") {

        alphaMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "rotation") {

        rotation = std::get<float>(value);
        return true;

    } else if (key == "sizeAttenuation") {

        rotation = std::get<bool>(value);
        return true;
    }

    return false;
}
