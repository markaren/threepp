
#include "threepp/materials/LineDashedMaterial.hpp"

using namespace threepp;

LineDashedMaterial::LineDashedMaterial() {}

std::string LineDashedMaterial::type() const {

    return "LineDashedMaterial";
}

void LineDashedMaterial::copyInto(Material& material) const {

    LineBasicMaterial::copyInto(material);

    auto m = material.as<LineDashedMaterial>();

    m->dashSize = dashSize;
    m->gapSize = gapSize;
    m->scale = scale;
}

std::shared_ptr<Material> LineDashedMaterial::createDefault() const {

    return std::shared_ptr<LineDashedMaterial>(new LineDashedMaterial());
}

std::shared_ptr<LineDashedMaterial> LineDashedMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<LineDashedMaterial>(new LineDashedMaterial());
    m->setValues(values);

    return m;
}

bool LineDashedMaterial::setValue(const std::string& key, const MaterialValue& value) {

    if (key == "dashSize") {

        dashSize = extractFloat(value);
        return true;

    } else if (key == "gapSize") {

        gapSize = extractFloat(value);
        return true;

    } else if (key == "scale") {

        scale = extractFloat(value);
        return true;
    }

    return LineBasicMaterial::setValue(key, value);
}
