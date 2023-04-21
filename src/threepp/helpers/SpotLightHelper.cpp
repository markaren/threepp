
#include "threepp/helpers/SpotLightHelper.hpp"

#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/objects/LineSegments.hpp"

#include <cmath>

using namespace threepp;


SpotLightHelper::SpotLightHelper(SpotLight& light, std::optional<Color> color)
    : light(light), color(color) {

    this->light.updateMatrixWorld();

    this->matrix = this->light.matrixWorld;
    this->matrixAutoUpdate = false;

    auto geometry = BufferGeometry::create();

    std::vector<float> positions{
            0, 0, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 1,
            0, 0, 0, -1, 0, 1,
            0, 0, 0, 0, 1, 1,
            0, 0, 0, 0, -1, 1};

    for (unsigned i = 0, j = 1, l = 32; i < l; i++, j++) {

        const auto p1 = (static_cast<float>(i) / static_cast<float>(l)) * math::PI * 2;
        const auto p2 = (static_cast<float>(j) / static_cast<float>(l)) * math::PI * 2;

        positions.insert(positions.end(),
                         {std::cos(p1), std::sin(p1), 1,
                          std::cos(p2), std::sin(p2), 1});
    }

    geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    auto material = LineBasicMaterial::create();
    material->fog = false;
    material->toneMapped = false;

    this->cone = LineSegments::create(geometry, material);
    this->add(this->cone);

    this->update();
}

std::shared_ptr<SpotLightHelper> SpotLightHelper::create(SpotLight& light, std::optional<Color> color) {

    return std::shared_ptr<SpotLightHelper>(new SpotLightHelper(light, color));
}

void SpotLightHelper::update() {

    this->light.updateMatrixWorld();

    const auto coneLength = (this->light.distance > 0) ? this->light.distance : 1000;
    const auto coneWidth = coneLength * std::tan(this->light.angle);

    this->cone->scale.set(coneWidth, coneWidth, coneLength);

    static Vector3 _vector;
    _vector.setFromMatrixPosition(*this->light.target->matrixWorld);

    this->cone->lookAt(_vector);

    if (this->color) {

        this->material()->as<MaterialWithColor>()->color.copy(*this->color);

    } else {

        this->cone->material()->as<MaterialWithColor>()->color.copy(this->light.color);
    }
}

SpotLightHelper::~SpotLightHelper() {

    this->cone->geometry()->dispose();
    this->cone->material()->dispose();
}
