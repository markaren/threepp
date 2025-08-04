
#include "threepp/materials/LineBasicMaterial.hpp"

using namespace threepp;

LineBasicMaterial::LineBasicMaterial()
    : MaterialWithColor(0xffffff),
      MaterialWithLineWidth(1) {}


std::string LineBasicMaterial::type() const {

    return "LineBasicMaterial";
}

void LineBasicMaterial::copyInto(Material& material) const {

    Material::copyInto(material);

    auto m = material.as<LineBasicMaterial>();

    m->color.copy(color);

    m->linewidth = linewidth;
}

std::shared_ptr<Material> LineBasicMaterial::createDefault() const {

    return std::shared_ptr<LineBasicMaterial>(new LineBasicMaterial());
}

std::shared_ptr<LineBasicMaterial> LineBasicMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<LineBasicMaterial>(new LineBasicMaterial());
    m->setValues(values);

    return m;
}

bool LineBasicMaterial::setValue(const std::string& key, const MaterialValue& value) {

    if (key == "color") {

        color.copy(extractColor(value));
        return true;

    } else if (key == "linewidth") {

        linewidth = std::get<float>(value);
        return true;
    }

    return false;
}
