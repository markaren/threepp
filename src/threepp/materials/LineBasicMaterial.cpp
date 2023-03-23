
#include "threepp/materials/LineBasicMaterial.hpp"

using namespace threepp;

LineBasicMaterial::LineBasicMaterial()
    : MaterialWithColor(0xffffff),
      MaterialWithLineWidth(1) {}


std::string LineBasicMaterial::type() const {

    return "LineBasicMaterial";
}

std::shared_ptr<Material> LineBasicMaterial::clone() const {

    auto m = create();
    copyInto(m.get());

    m->color.copy(color);

    m->linewidth = linewidth;

    return m;
}

std::shared_ptr<LineBasicMaterial> LineBasicMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<LineBasicMaterial>(new LineBasicMaterial());
    m->setValues(values);

    return m;
}

bool LineBasicMaterial::setValue(const std::string& key, const MaterialValue& value) {

    if (key == "color") {

        if (std::holds_alternative<int>(value)) {
            color = std::get<int>(value);
        } else {
            color.copy(std::get<Color>(value));
        }

        return true;
    } else if (key == "linewidth") {

        linewidth = std::get<float>(value);
        return true;
    }

    return false;
}
