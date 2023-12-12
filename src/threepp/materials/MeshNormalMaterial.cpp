
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

std::shared_ptr<MeshNormalMaterial> MeshNormalMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<MeshNormalMaterial>(new MeshNormalMaterial());
    m->setValues(values);

    return m;
}

std::shared_ptr<Material> MeshNormalMaterial::clone() const {

    auto m = create();
    copyInto(m.get());

    m->normalMap = normalMap;
    m->normalMapType = normalMapType;
    m->normalScale.copy(normalScale);

    m->displacementMap = displacementMap;
    m->displacementScale = displacementScale;
    m->displacementBias = displacementBias;

    m->wireframe = wireframe;
    m->wireframeLinewidth = wireframeLinewidth;

    m->flatShading = flatShading;

    return m;
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
