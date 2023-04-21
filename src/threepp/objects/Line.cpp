
#include "threepp/objects/Line.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"

#include "threepp/core/Raycaster.hpp"

#include <iostream>

using namespace threepp;

namespace {

    Sphere _sphere;
    Matrix4 _inverseMatrix;
    Ray _ray;

}// namespace


Line::Line(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material)
    : geometry_(geometry ? std::move(geometry) : BufferGeometry::create()),
      material_(material ? std::move(material) : LineBasicMaterial::create()) {}

std::string Line::type() const {

    return "Line";
}

BufferGeometry* Line::geometry() {

    return geometry_.get();
}

Material* Line::material() {

    return material_.get();
}

std::vector<Material*> Line::materials() {

    return {material_.get()};
}

std::shared_ptr<Object3D> Line::clone(bool recursive) {
    auto clone = create(geometry_, material_);
    clone->copy(*this, recursive);

    return clone;
}

std::shared_ptr<Line> Line::create(const std::shared_ptr<BufferGeometry>& geometry, const std::shared_ptr<Material>& material) {

    return std::shared_ptr<Line>(new Line(geometry, (material)));
}

void Line::computeLineDistances() {

    Vector3 _start;
    Vector3 _end;

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

        geometry_->setAttribute("lineDistance", FloatBufferAttribute::create(lineDistances, 1));

    } else {

        std::cerr << "THREE.Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry." << std::endl;
    }
}

void Line::raycast(Raycaster& raycaster, std::vector<Intersection>& intersects) {

    auto geometry = this->geometry();
    auto threshold = raycaster.params.lineThreshold;
    auto drawRange = geometry->drawRange;

    // Checking boundingSphere distance to ray

    if (!geometry->boundingSphere) geometry->computeBoundingSphere();

    _sphere.copy(*geometry->boundingSphere);
    _sphere.applyMatrix4(*matrixWorld);
    _sphere.radius += threshold;

    if (!raycaster.ray.intersectsSphere(_sphere)) return;

    //

    _inverseMatrix.copy(*matrixWorld).invert();
    _ray.copy(raycaster.ray).applyMatrix4(_inverseMatrix);

    const auto localThreshold = threshold / ((this->scale.x + this->scale.y + this->scale.z) / 3);
    const auto localThresholdSq = localThreshold * localThreshold;

    Vector3 vStart;
    Vector3 vEnd;
    Vector3 interSegment;
    Vector3 interRay;
    const auto step = type() == "LineSegments" ? 2 : 1;

    auto index = geometry->getIndex();
    auto positionAttribute = geometry->getAttribute<float>("position");

    if (index) {

        const auto start = std::max(0, drawRange.start);
        const auto end = std::min(index->count(), (drawRange.start + drawRange.count));

        for (unsigned i = start, l = end - 1; i < l; i += step) {

            const auto a = index->getX(i);
            const auto b = index->getX(i + 1);

            positionAttribute->setFromBufferAttribute(vStart, a);
            positionAttribute->setFromBufferAttribute(vEnd, b);

            const auto distSq = _ray.distanceSqToSegment(vStart, vEnd, &interRay, &interSegment);

            if (distSq > localThresholdSq) continue;

            interRay.applyMatrix4(*this->matrixWorld);//Move back to world space for distance calculation

            const auto distance = raycaster.ray.origin.distanceTo(interRay);

            if (distance < raycaster.near || distance > raycaster.far) continue;

            Intersection intersection;
            intersection.distance = distance;
            intersection.point = interSegment.clone().applyMatrix4(*this->matrixWorld);
            intersection.index = i;
            intersection.object = this;

            intersects.emplace_back(intersection);
        }

    } else {

        const auto start = std::max(0, drawRange.start);
        const auto end = std::min(positionAttribute->count(), (drawRange.start + drawRange.count));

        for (unsigned i = start, l = end - 1; i < l; i += step) {

            positionAttribute->setFromBufferAttribute(vStart, i);
            positionAttribute->setFromBufferAttribute(vEnd, i + 1);

            const auto distSq = _ray.distanceSqToSegment(vStart, vEnd, &interRay, &interSegment);

            if (distSq > localThresholdSq) continue;

            interRay.applyMatrix4(*this->matrixWorld);//Move back to world space for distance calculation

            const auto distance = raycaster.ray.origin.distanceTo(interRay);

            if (distance < raycaster.near || distance > raycaster.far) continue;

            Intersection intersection;
            intersection.distance = distance;
            intersection.point = interSegment.clone().applyMatrix4(*this->matrixWorld);
            intersection.index = i;
            intersection.object = this;

            intersects.emplace_back(intersection);
        }
    }
}
