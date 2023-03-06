
#include "threepp/objects/Mesh.hpp"

#include "threepp/core/Face3.hpp"
#include "threepp/core/Raycaster.hpp"

using namespace threepp;

namespace {

    std::optional<Intersection> checkIntersection(Object3D* object, Material* material, Raycaster& raycaster, Ray& ray, const Vector3& pA, const Vector3& pB, const Vector3& pC, Vector3& point) {

        static Vector3 _intersectionPointWorld{};

        if (material->side == BackSide) {

            ray.intersectTriangle(pC, pB, pA, true, point);

        } else {

            ray.intersectTriangle(pA, pB, pC, material->side != DoubleSide, point);
        }

        if (std::isnan(point.x)) return std::nullopt;

        _intersectionPointWorld.copy(point);
        _intersectionPointWorld.applyMatrix4(*object->matrixWorld);

        const auto distance = raycaster.ray.origin.distanceTo(_intersectionPointWorld);

        if (distance < raycaster.near || distance > raycaster.far) return std::nullopt;

        Intersection intersection{};
        intersection.distance = distance;
        intersection.point = _intersectionPointWorld;
        intersection.object = object;

        return intersection;
    }

    std::optional<Intersection> checkBufferGeometryIntersection(
            Object3D* object, Material* material, Raycaster& raycaster, Ray& ray,
            const FloatBufferAttribute& position, const FloatBufferAttribute* uv, const FloatBufferAttribute* uv2,
            unsigned int a, unsigned int b, unsigned int c) {

        static Vector3 _vA{};
        static Vector3 _vB{};
        static Vector3 _vC{};
        static Vector3 _intersectionPoint{};

        position.setFromBufferAttribute(_vA, a);
        position.setFromBufferAttribute(_vB, b);
        position.setFromBufferAttribute(_vC, c);

        auto intersection = checkIntersection(object, material, raycaster, ray, _vA, _vB, _vC, _intersectionPoint);

        if (intersection) {

            static Vector2 _uvA{};
            static Vector2 _uvB{};
            static Vector2 _uvC{};

            if (uv) {

                uv->setFromBufferAttribute(_uvA, a);
                uv->setFromBufferAttribute(_uvB, c);
                uv->setFromBufferAttribute(_uvC, c);

                Vector2 uvTarget{};
                Triangle::getUV(_intersectionPoint, _vA, _vB, _vC, _uvA, _uvB, _uvC, uvTarget);
                intersection->uv = uvTarget;
            }

            if (uv2) {

                uv2->setFromBufferAttribute(_uvA, a);
                uv2->setFromBufferAttribute(_uvB, c);
                uv2->setFromBufferAttribute(_uvC, c);

                Vector2 uv2Target{};
                Triangle::getUV(_intersectionPoint, _vA, _vB, _vC, _uvA, _uvB, _uvC, uv2Target);
                intersection->uv2 = uv2Target;
            }

            Face3 face{a, b, c, {}, 0};

            Triangle::getNormal(_vA, _vB, _vC, face.normal);

            intersection->face = face;
        }

        return intersection;
    }

}// namespace

void Mesh::raycast(Raycaster& raycaster, std::vector<Intersection>& intersects) {

    if (material() == nullptr) return;

    static Sphere _sphere{};

    // Checking boundingSphere distance to ray

    if (!geometry_->boundingSphere) geometry_->computeBoundingSphere();

    _sphere.copy(*geometry_->boundingSphere);
    _sphere.applyMatrix4(*matrixWorld);

    if (!raycaster.ray.intersectsSphere(_sphere)) return;

    //

    static Ray _ray{};
    static Matrix4 _inverseMatrix{};

    _inverseMatrix.copy(*matrixWorld).invert();
    _ray.copy(raycaster.ray).applyMatrix4(_inverseMatrix);

    // Check boundingBox before continuing

    if (geometry_->boundingBox) {

        if (!_ray.intersectsBox(*geometry_->boundingBox)) return;
    }

    std::optional<Intersection> intersection;

    const auto index = geometry_->getIndex();
    const auto position = geometry_->getAttribute<float>("position");
    const auto uv = geometry_->getAttribute<float>("uv");
    const auto uv2 = geometry_->getAttribute<float>("uv2");
    const auto groups = geometry_->groups;
    const auto drawRange = geometry_->drawRange;

    if (index != nullptr) {

        // indexed buffer geometry

        if (numMaterials() > 1) {

            for (auto& group : groups) {

                auto groupMaterial = materials_[group.materialIndex].get();

                const auto start = std::max(group.start, drawRange.start);
                const auto end = std::min((group.start + group.count), (drawRange.start + drawRange.count));

                for (int j = start, jl = end; j < jl; j += 3) {

                    const auto a = index->getX(j);
                    const auto b = index->getX(j + 1);
                    const auto c = index->getX(j + 2);

                    intersection = checkBufferGeometryIntersection(this, groupMaterial, raycaster, _ray, *position, uv, uv2, a, b, c);

                    if (intersection) {

                        intersection->faceIndex = j / 3;// triangle number in indexed buffer semantics
                        intersection->face->materialIndex = group.materialIndex;
                        intersects.emplace_back(*intersection);
                    }
                }
            }

        } else {

            const int start = std::max(0, drawRange.start);
            const int end = std::min(index->count(), (drawRange.start + drawRange.count));

            for (int i = start, il = end; i < il; i += 3) {

                const auto a = index->getX(i);
                const auto b = index->getX(i + 1);
                const auto c = index->getX(i + 2);

                intersection = checkBufferGeometryIntersection(this, material(), raycaster, _ray, *position, uv, uv2, a, b, c);

                if (intersection) {

                    intersection->faceIndex = (int) std::floor(i / 3);// triangle number in indexed buffer semantics
                    intersects.emplace_back(*intersection);
                }
            }
        }

    } else if (position != nullptr) {

        // non-indexed buffer geometry

        if (numMaterials() > 1) {

            for (auto& group : groups) {

                auto groupMaterial = materials_[group.materialIndex].get();

                const auto start = std::max(group.start, drawRange.start);
                const auto end = std::min((group.start + group.count), (drawRange.start + drawRange.count));

                for (int j = start, jl = end; j < jl; j += 3) {

                    const auto a = j;
                    const auto b = j + 1;
                    const auto c = j + 2;

                    intersection = checkBufferGeometryIntersection(this, groupMaterial, raycaster, _ray, *position, uv, uv2, a, b, c);

                    if (intersection) {

                        intersection->faceIndex = (int) std::floor(j / 3);// triangle number in non-indexed buffer semantics
                        intersection->face->materialIndex = group.materialIndex;
                        intersects.emplace_back(*intersection);
                    }
                }
            }

        } else {

            const int start = std::max(0, drawRange.start);
            const int end = std::min(position->count(), (drawRange.start + drawRange.count));

            for (int i = start, il = end; i < il; i += 3) {

                const int a = i;
                const int b = i + 1;
                const int c = i + 2;

                intersection = checkBufferGeometryIntersection(this, material(), raycaster, _ray, *position, uv, uv2, a, b, c);

                if (intersection) {

                    intersection->faceIndex = (int) std::floor(i / 3);// triangle number in non-indexed buffer semantics
                    intersects.emplace_back(*intersection);
                }
            }
        }
    }
}

std::shared_ptr<Object3D> Mesh::clone(bool recursive) {

    auto clone = create(geometry_, materials_);
    clone->copy(*this, recursive);

    return clone;
}
