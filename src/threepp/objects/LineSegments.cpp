
#include "threepp/objects/LineSegments.hpp"

#include <iostream>
#include <memory>

#include "threepp/utils/Log.hpp"
using namespace threepp;

namespace {
    constinit log::Category kLog{"core.geometry"};
}

LineSegments::LineSegments(
        const std::shared_ptr<BufferGeometry>& geometry,
        const std::shared_ptr<Material>& material)
    : Line(geometry, material) {}


const std::string& LineSegments::type() const {

    static const std::string typeName = "LineSegments";
    return typeName;
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

        THREEPP_LOG(kLog, Warn) << "LineSegments::computeLineDistances needs a non-indexed geometry; no distances are computed";
    }
}

std::shared_ptr<LineSegments> LineSegments::create(
        const std::shared_ptr<BufferGeometry>& geometry,
        const std::shared_ptr<Material>& material) {

    return std::make_shared<LineSegments>(geometry, (material));
}

std::shared_ptr<Object3D> LineSegments::createDefault() {

    return create();
}
