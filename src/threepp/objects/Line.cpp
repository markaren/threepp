
#include "threepp/objects/Line.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"

using namespace threepp;


Line::Line(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material)
        : geometry_(geometry ? std::move(geometry) : BufferGeometry::create()),
          material_(material ? std::move(material) : LineBasicMaterial::create()) {}

void Line::computeLineDistances() {

    // we assume non-indexed geometry

    if (!geometry_->hasIndex()) {

        const auto positionAttribute = geometry_->getAttribute<float>("position");
        std::vector<float> lineDistances{0};

        for (int i = 1, l = positionAttribute->count(); i < l; i++) {

            positionAttribute->setFromBufferAttribute(_start, i - 1);
            positionAttribute->setFromBufferAttribute(_end, i);

            lineDistances[i] = lineDistances[i - 1];
            lineDistances[i] += _start.distanceTo(_end);
        }

        geometry_->setAttribute("lineDistance", TypedBufferAttribute<float>::create(lineDistances, 1));

    } else {

        std::cerr << "THREE.Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry." << std::endl;
    }
}
