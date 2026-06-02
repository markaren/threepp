
#include "threepp/materials/MeshNormalMaterial.hpp"

using namespace threepp;

MeshNormalMaterial::MeshNormalMaterial()
    : MaterialWithBumpMap(1),
      MaterialWithNormalMap(NormalMapType::TangentSpace, {1, 1}),
      MaterialWithDisplacementMap(1, 0),
      MaterialWithWireframe(false, 1),
      MaterialWithFlatShading(false) {

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

std::shared_ptr<MeshNormalMaterial> MeshNormalMaterial::create(const Params& p) {

    auto m = std::shared_ptr<MeshNormalMaterial>(new MeshNormalMaterial());

    p.applyBaseTo(*m);

    // Apply only the fields the caller set; everything else keeps the constructor default.
    // Params stores each value in a `field_` member; the material's field is `field`.
#define TPP_SET(field) \
    if (p.field##_) m->field = *p.field##_;
#define TPP_TEX(field) \
    if (p.field##_) m->field = p.field##_;

    TPP_SET(wireframe)
    TPP_SET(wireframeLinewidth)
    TPP_SET(flatShading)
    TPP_TEX(normalMap)
    TPP_SET(normalMapType)
    TPP_TEX(displacementMap)
    TPP_SET(displacementBias)
    TPP_SET(displacementScale)

#undef TPP_SET
#undef TPP_TEX

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
