
#include "threepp/helpers/DirectionalLightHelper.hpp"

#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/objects/Line.hpp"

using namespace threepp;

DirectionalLightHelper::DirectionalLightHelper(DirectionalLight& light, float size, std::optional<Color> color)
    : color(color), light(light) {

    this->light.updateMatrixWorld();

    this->matrix = this->light.matrixWorld;
    this->matrixAutoUpdate = false;

    auto geometry = BufferGeometry::create();
    geometry->setAttribute("position", FloatBufferAttribute::create(
                                               {-size, size, 0,
                                                size, size, 0,
                                                size, -size, 0,
                                                -size, -size, 0,
                                                -size, size, 0},
                                               3));

    auto material = LineBasicMaterial::create();
    material->fog = false;
    material->toneMapped = false;

    this->lightPlane = Line::create(geometry, material);
    this->add(this->lightPlane);

    geometry = BufferGeometry::create();
    geometry->setAttribute("position", FloatBufferAttribute::create({0, 0, 0, 0, 0, 1}, 3));

    this->targetLine = Line::create(geometry, material);
    this->targetLine->frustumCulled = false;
    this->add(this->targetLine);

    this->update();
}

void DirectionalLightHelper::update() {

    static Vector3 _v1;
    static Vector3 _v2;
    static Vector3 _v3;

    _v1.setFromMatrixPosition(*this->light.matrixWorld);
    _v2.setFromMatrixPosition(*this->light.target().matrixWorld);
    _v3.subVectors(_v2, _v1);

    this->lightPlane->lookAt(_v2);

    if (this->color) {

        this->lightPlane->material()->as<MaterialWithColor>()->color.copy(*this->color);
        this->targetLine->material()->as<MaterialWithColor>()->color.copy(*this->color);

    } else {

        this->lightPlane->material()->as<MaterialWithColor>()->color.copy(this->light.color);
        this->targetLine->material()->as<MaterialWithColor>()->color.copy(this->light.color);
    }

    this->targetLine->lookAt(_v2);
    this->targetLine->scale.z = _v3.length();
}

std::shared_ptr<DirectionalLightHelper> DirectionalLightHelper::create(DirectionalLight& light, float size, std::optional<Color> color) {

    return std::shared_ptr<DirectionalLightHelper>(new DirectionalLightHelper(light, size, color));
}
