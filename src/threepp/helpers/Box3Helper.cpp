
#include "threepp/helpers/Box3Helper.hpp"

#include "threepp/materials/LineBasicMaterial.hpp"

using namespace threepp;


Box3Helper::Box3Helper(const Box3& box, const Color& color)
    : LineSegments(BufferGeometry::create(), LineBasicMaterial::create()), box(box) {

    std::vector<unsigned int> indices{0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7};

    std::vector<float> positions{1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, -1};

    auto lineMaterial = material()->as<LineBasicMaterial>();
    lineMaterial->color.copy(color);
    lineMaterial->toneMapped = false;

    geometry_->setIndex(indices);

    geometry_->setAttribute("position", FloatBufferAttribute::create(positions, 3));

    geometry_->computeBoundingSphere();
}

std::shared_ptr<Box3Helper> Box3Helper::create(const Box3& box, const Color& color) {

    return std::shared_ptr<Box3Helper>(new Box3Helper(box, color));
}

void Box3Helper::updateMatrixWorld(bool force) {
    if (box.isEmpty()) return;

    box.getCenter(this->position);

    box.getSize(this->scale);

    this->scale.multiplyScalar(0.5);

    LineSegments::updateMatrixWorld(force);
}
