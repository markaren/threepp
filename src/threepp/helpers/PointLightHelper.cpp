
#include "threepp/helpers/PointLightHelper.hpp"

#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"

using namespace threepp;


PointLightHelper::PointLightHelper(PointLight& light, float sphereSize, std::optional<Color> color)
    : Mesh(nullptr, nullptr), color(color), light(&light) {

    geometry_ = SphereGeometry::create(sphereSize, 4, 2);

    auto material = MeshBasicMaterial::create();
    material->wireframe = true;
    material->fog = false;
    material->toneMapped = false;
    this->materials_[0] = std::move(material);

    this->light->updateMatrixWorld();

    this->matrix = this->light->matrixWorld;
    this->matrixAutoUpdate = false;

    update();
}

std::shared_ptr<PointLightHelper> PointLightHelper::create(PointLight& light, float sphereSize, std::optional<Color> color) {

    return std::shared_ptr<PointLightHelper>(new PointLightHelper(light, sphereSize, color));
}

void PointLightHelper::update() {

    if (this->color) {

        this->material()->as<MaterialWithColor>()->color.copy(*this->color);

    } else {

        this->material()->as<MaterialWithColor>()->color.copy(this->light->color);
    }
}
