
#include "threepp/materials/MeshNormalMaterial.hpp"

using namespace threepp;

MeshNormalMaterial::MeshNormalMaterial()
    : MaterialWithFlatShading(false),
      MaterialWithWireframe(false, 1),
      MaterialWithDisplacementMap(1, 0),
      MaterialWithNormalMap(NormalMapType::TangentSpace, {1, 1}),
      MaterialWithBumpMap(1) {

    this->fog = false;
}


std::string MeshNormalMaterial::type() const {

    return "MeshNormalMaterial";
}


std::shared_ptr<Material> MeshNormalMaterial::createDefault() const {

    return std::shared_ptr<MeshNormalMaterial>(new MeshNormalMaterial());
}

std::shared_ptr<MeshNormalMaterial> MeshNormalMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<MeshNormalMaterial>(new MeshNormalMaterial());
    m->setValues(values);

    return m;
}

void MeshNormalMaterial::copyInto(Material& material) const {

    Material::copyInto(material);

    auto m = material.as<MeshNormalMaterial>();

    m->normalMap = normalMap;
    m->normalMapType = normalMapType;
    m->normalScale.copy(normalScale);

    m->displacementMap = displacementMap;
    m->displacementScale = displacementScale;
    m->displacementBias = displacementBias;

    m->wireframe = wireframe;
    m->wireframeLinewidth = wireframeLinewidth;

    m->flatShading = flatShading;
}

bool MeshNormalMaterial::setValue(const std::string& key, const MaterialValue& value) {

    if (key == "wireframe") {

        wireframe = std::get<bool>(value);
        return true;

    } else if (key == "wireframeLinewidth") {

        wireframeLinewidth = extractFloat(value);
        return true;

    } else if (key == "flatShading") {

        flatShading = std::get<bool>(value);
        return true;

    } else if (key == "normalMap") {

        normalMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "normalMapType") {

        normalMapType = std::get<NormalMapType>(value);
        return true;

    } else if (key == "displacementMap") {

        displacementMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "displacementBias") {

        displacementBias = extractFloat(value);
        return true;

    } else if (key == "displacementScale") {

        displacementScale = extractFloat(value);
        return true;
    }

    return false;
}
