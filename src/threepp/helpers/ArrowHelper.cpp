
#include "threepp/helpers/ArrowHelper.hpp"

#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"

#include <cmath>

using namespace threepp;

ArrowHelper::ArrowHelper(Vector3 dir, Vector3 origin, float length, const Color& color, std::optional<float> headLength_, std::optional<float> headWidth_) {

    float headLength = headLength_.value_or(length * 0.2f);
    float headWidth = headWidth_.value_or(headLength * 0.2f);

    auto _lineGeometry = std::make_shared<BufferGeometry>();
    _lineGeometry->setAttribute("position", FloatBufferAttribute::create({0, 0, 0, 0, 1, 0}, 3));

    auto _coneGeometry = CylinderGeometry::create(0, 0.5f, 1, 5, 1);
    _coneGeometry->translate(0, -0.5f, 0);

    this->position.copy(origin);

    auto lineMaterial = LineBasicMaterial::create();
    lineMaterial->color.copy(color);
    lineMaterial->toneMapped = false;

    this->line = Line::create(_lineGeometry, lineMaterial);
    this->line->matrixAutoUpdate = false;
    this->add(this->line);

    auto coneMaterial = MeshBasicMaterial::create();
    coneMaterial->color.copy(color);
    coneMaterial->toneMapped = false;

    this->cone = Mesh::create(_coneGeometry, coneMaterial);
    this->cone->matrixAutoUpdate = false;
    this->add(this->cone);

    this->setDirection(dir);
    this->setLength(length, headLength, headWidth);
}

void ArrowHelper::setDirection(const Vector3& dir) {

    // dir is assumed to be normalized

    if (dir.y > 0.99999f) {

        this->quaternion.set(0, 0, 0, 1);

    } else if (dir.y < -0.99999f) {

        this->quaternion.set(1, 0, 0, 0);

    } else {

        Vector3 _axis;
        _axis.set(dir.z, 0, -dir.x).normalize();

        float radians = std::acos(dir.y);

        this->quaternion.setFromAxisAngle(_axis, radians);
    }
}

void ArrowHelper::setLength(float length, std::optional<float> headLength_, std::optional<float> headWidth_) {

    float headLength = headLength_.value_or(length * 0.2f);
    float headWidth = headWidth_.value_or(headLength * 0.2f);

    this->line->scale.set(1, std::max(0.0001f, length - headLength), 1);// see #17458
    this->line->updateMatrix();

    this->cone->scale.set(headWidth, headLength, headWidth);
    this->cone->position.y = length;
    this->cone->updateMatrix();
}

void ArrowHelper::setColor(const Color& color) {

    this->line->material()->as<MaterialWithColor>()->color.copy(color);
    this->cone->material()->as<MaterialWithColor>()->color.copy(color);
}

std::shared_ptr<ArrowHelper> ArrowHelper::create(Vector3 dir, Vector3 origin, float length, const Color& color, std::optional<float> headLength, std::optional<float> headWidth) {

    return std::shared_ptr<ArrowHelper>(new ArrowHelper(dir, origin, length, color, headLength, headWidth));
}
