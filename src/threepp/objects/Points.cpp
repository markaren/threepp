
#include "threepp/objects/Points.hpp"
#include "threepp/core/Raycaster.hpp"

#include <cmath>
#include <memory>

using namespace threepp;

namespace {

    Sphere _sphere;
    Vector3 _position;
    Matrix4 _inverseMatrix;
    Ray _ray;

    void testPoint(
            const Vector3& point,
            unsigned int index,
            float localThresholdSq,
            const Matrix4& matrixWorld,
            const Raycaster& raycaster,
            std::vector<Intersection>& intersects,
            Object3D* object) {

        const auto rayPointDistanceSq = _ray.distanceSqToPoint(point);

        if (rayPointDistanceSq < localThresholdSq) {

            Vector3 intersectPoint;

            _ray.closestPointToPoint(point, intersectPoint);
            intersectPoint.applyMatrix4(matrixWorld);

            const auto distance = raycaster.ray.origin.distanceTo(intersectPoint);

            if (distance < raycaster.nearPlane || distance > raycaster.farPlane) return;

            Intersection intersection;
            intersection.distance = distance;
            intersection.distanceToRay = std::sqrt(rayPointDistanceSq);
            intersection.index = index;
            intersection.object = object;

            intersects.emplace_back(intersection);
        }
    }

}// namespace

Points::Points(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material)
    : geometry_(std::move(geometry)), ObjectWithMaterials({std::move(material)}) {}

std::string Points::type() const {

    return "Points";
}

std::shared_ptr<BufferGeometry> Points::geometry() const {

    return geometry_;
}

void Points::setGeometry(const std::shared_ptr<BufferGeometry>& geometry) {

    this->geometry_ = geometry;
}

std::shared_ptr<Points> Points::create(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material) {

    return std::make_shared<Points>(std::move(geometry), std::move(material));
}

void Points::raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) {

    const auto geometry = this->geometry();
    const auto threshold = raycaster.params.pointsThreshold;
    const auto drawRange = geometry->drawRange;

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

    const auto index = geometry->getIndex();
    const auto positionAttribute = geometry->getAttribute<float>("position");

    if (index) {

        const auto start = std::max(0, drawRange.start);
        const auto end = std::min(index->count(), (drawRange.start + drawRange.count));

        for (auto i = start, il = end; i < il; i++) {

            const auto a = index->getX(i);

            positionAttribute->setFromBufferAttribute(_position, a);

            testPoint(_position, a, localThresholdSq, *matrixWorld, raycaster, intersects, this);
        }

    } else {

        const auto start = std::max(0, drawRange.start);
        const auto end = std::min(positionAttribute->count(), (drawRange.start + drawRange.count));

        for (unsigned i = start, l = end; i < l; i++) {

            positionAttribute->setFromBufferAttribute(_position, i);

            testPoint(_position, i, localThresholdSq, *matrixWorld, raycaster, intersects, this);
        }
    }
}

void Points::copy(const Object3D& source, bool recursive) {
    Object3D::copy(source, recursive);

    if (auto p = source.as<Points>()) {

        materials_ = p->materials_;
        geometry_ = p->geometry_;
    }
}

std::shared_ptr<Object3D> Points::createDefault() {

    return create();
}
