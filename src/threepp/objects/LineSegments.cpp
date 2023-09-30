
#include "threepp/objects/LineSegments.hpp"

#include <iostream>
#include <memory>

using namespace threepp;

LineSegments::LineSegments(
        const std::shared_ptr<BufferGeometry>& geometry,
        const std::shared_ptr<Material>& material)
    : Line(geometry, material) {}


std::string LineSegments::type() const {

    return "LineSegments";
}

void LineSegments::computeLineDistances() {

    Vector3 _start;
    Vector3 _end;

    // we assume non-indexed geometry

    if (geometry_->getIndex() == nullptr) {

        const auto positionAttribute = geometry_->getAttribute<float>("position");
        std::vector<float> lineDistances;

        for (int i = 0, l = positionAttribute->count(); i < l; i += 2) {

            positionAttribute->setFromBufferAttribute(_start, i);
            positionAttribute->setFromBufferAttribute(_end, i + 1);

            lineDistances[i] = (i == 0) ? 0 : lineDistances[i - 1];
            lineDistances[i + 1] = lineDistances[i] + _start.distanceTo(_end);
        }

        geometry_->setAttribute("lineDistance", FloatBufferAttribute::create(lineDistances, 1));

    } else {

        std::cerr << "THREE.LineSegments.computeLineDistances(): Computation only possible with non-indexed BufferGeometry." << std::endl;
    }
}

std::shared_ptr<Object3D> LineSegments::clone(bool recursive) {
    auto clone = create();
    clone->copy(*this, recursive);

    return clone;
}

std::shared_ptr<LineSegments> LineSegments::create(
        const std::shared_ptr<BufferGeometry>& geometry,
        const std::shared_ptr<Material>& material) {

    return std::make_shared<LineSegments>(geometry, (material));
}
