
#include "threepp/materials/MeshPhongMaterial.hpp"

using namespace threepp;

MeshPhongMaterial::MeshPhongMaterial()
    : MaterialWithColor(0xffffff),
      MaterialWithCombine(MultiplyOperation),
      MaterialWithFlatShading(false),
      MaterialWithSpecular(0x111111, 30),
      MaterialWithLightMap(1),
      MaterialWithAoMap(1),
      MaterialWithEmissive(0x000000, 1),
      MaterialWithBumpMap(1),
      MaterialWithNormalMap(TangentSpaceNormalMap, {1, 1}),
      MaterialWithDisplacementMap(1, 0),
      MaterialWithReflectivity(1, 0.98f),
      MaterialWithWireframe(false, 1) {}


std::string MeshPhongMaterial::type() const {

    return "MeshPhongMaterial";
}

std::shared_ptr<Material> MeshPhongMaterial::clone() const {

    auto m = create();
    copyInto(m.get());

    m->color.copy(color);
    m->specular.copy(specular);
    m->shininess = shininess;

    m->map = map;

    m->lightMap = lightMap;
    m->lightMapIntensity = lightMapIntensity;

    m->aoMap = aoMap;
    m->aoMapIntensity = aoMapIntensity;

    m->emissive.copy(emissive);
    m->emissiveMap = emissiveMap;
    m->emissiveIntensity = emissiveIntensity;

    m->bumpMap = bumpMap;
    m->bumpScale = bumpScale;

    m->normalMap = normalMap;
    m->normalMapType = normalMapType;
    m->normalScale.copy(normalScale);

    m->displacementMap = displacementMap;
    m->displacementScale = displacementScale;
    m->displacementBias = displacementBias;

    m->specularMap = specularMap;

    m->alphaMap = alphaMap;

    m->envMap = envMap;
    m->combine = combine;
    m->reflectivity = reflectivity;
    m->refractionRatio = refractionRatio;

    m->wireframe = wireframe;
    m->wireframeLinewidth = wireframeLinewidth;

    m->flatShading = flatShading;

    return m;
}

std::shared_ptr<MeshPhongMaterial> MeshPhongMaterial::create(const std::unordered_map<std::string, MaterialValue>& values) {

    auto m = std::shared_ptr<MeshPhongMaterial>(new MeshPhongMaterial());
    m->setValues(values);

    return m;
}

bool MeshPhongMaterial::setValue(const std::string& key, const MaterialValue& value) {

    if (key == "color") {

        if (std::holds_alternative<int>(value)) {
            color = std::get<int>(value);
        } else {
            color.copy(std::get<Color>(value));
        }

        return true;

    } else if (key == "wireframe") {

        wireframe = std::get<bool>(value);
        return true;

    } else if (key == "flatShading") {

        flatShading = std::get<bool>(value);
        return true;

    } else if (key == "map") {

        map = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "bumpMap") {

        bumpMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "normalMap") {

        normalMap = std::get<std::shared_ptr<Texture>>(value);
        return true;

    } else if (key == "alphaMap") {

        alphaMap = std::get<std::shared_ptr<Texture>>(value);
        return true;
    }

    return false;
}
