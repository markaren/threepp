
#include "threepp/objects/Mesh.hpp"

#include "threepp/objects/SkinnedMesh.hpp"

#include "threepp/core/Face3.hpp"
#include "threepp/core/Raycaster.hpp"

#include "threepp/materials/MeshBasicMaterial.hpp"

#include "threepp/math/Triangle.hpp"

#include <algorithm>
#include <memory>


using namespace threepp;

namespace {

    std::optional<Intersection> checkIntersection(
            Object3D* object, Material* material, Raycaster& raycaster, Ray& ray,
            const Vector3& pA, const Vector3& pB, const Vector3& pC, Vector3& point) {

        static Vector3 _intersectionPointWorld{};

        if (material->side == Side::Back) {

            ray.intersectTriangle(pC, pB, pA, true, point);

        } else {

            ray.intersectTriangle(pA, pB, pC, material->side != Side::Double, point);
        }

        if (point.isNan()) return std::nullopt;

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
            Object3D* object, Material* material,
            Raycaster& raycaster, Ray& ray,
            const FloatBufferAttribute& position,
            const std::vector<std::shared_ptr<BufferAttribute>>* morphPosition,
            bool morphTargetsRelative,
            const FloatBufferAttribute* uv,
            const FloatBufferAttribute* uv2,
            unsigned int a, unsigned int b, unsigned int c) {

        static Vector3 _vA{};
        static Vector3 _vB{};
        static Vector3 _vC{};
        static Vector3 _intersectionPoint{};

        position.setFromBufferAttribute(_vA, a);
        position.setFromBufferAttribute(_vB, b);
        position.setFromBufferAttribute(_vC, c);

        if (morphPosition) {

            if (auto morphObject = object->as<ObjectWithMorphTargetInfluences>()) {

                if (auto morphMaterial = material->as<MaterialWithMorphTargets>()) {

                    if (morphMaterial->morphTargets) {

                        const auto& morphInfluences = morphObject->morphTargetInfluences();

                        if (!morphInfluences.empty()) {

                            Vector3 _morphA;
                            Vector3 _morphB;
                            Vector3 _morphC;
                            Vector3 _tempA;
                            Vector3 _tempB;
                            Vector3 _tempC;

                            for (unsigned i = 0, il = morphPosition->size(); i < il; i++) {

                                float influence = morphInfluences[i];

                                if (influence == 0) continue;

                                auto morphAttribute = morphPosition->at(i)->typed<float>();

                                morphAttribute->setFromBufferAttribute(_tempA, a);
                                morphAttribute->setFromBufferAttribute(_tempB, b);
                                morphAttribute->setFromBufferAttribute(_tempC, c);

                                if (morphTargetsRelative) {

                                    _morphA.addScaledVector(_tempA, influence);
                                    _morphB.addScaledVector(_tempB, influence);
                                    _morphC.addScaledVector(_tempC, influence);

                                } else {

                                    _morphA.addScaledVector(_tempA.sub(_vA), influence);
                                    _morphB.addScaledVector(_tempB.sub(_vB), influence);
                                    _morphC.addScaledVector(_tempC.sub(_vC), influence);
                                }
                            }

                            _vA.add(_morphA);
                            _vB.add(_morphB);
                            _vC.add(_morphC);
                        }
                    }
                }
            }
        }

        if (auto skinned = object->as<SkinnedMesh>()) {
            skinned->boneTransform(a, _vA);
            skinned->boneTransform(b, _vB);
            skinned->boneTransform(c, _vC);
        }

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


Mesh::Mesh(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material)
    : geometry_(geometry ? std::move(geometry) : BufferGeometry::create()),
      materials_{material ? std::move(material) : MeshBasicMaterial::create()} {
}

Mesh::Mesh(std::shared_ptr<BufferGeometry> geometry, std::vector<std::shared_ptr<Material>> materials)
    : geometry_(std::move(geometry)), materials_{std::move(materials)} {
}

Mesh::Mesh(Mesh&& other) noexcept: Object3D(std::move(other)) {
    geometry_ = std::move(other.geometry_);
    materials_ = std::move(other.materials_);
}

std::vector<Material*> Mesh::materials() {

    std::vector<Material*> res(materials_.size());
    std::transform(materials_.begin(), materials_.end(), res.begin(), [](auto& m) { return m.get(); });

    return res;
}

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
    const auto morphPosition = geometry_->getMorphAttribute("position");
    const auto morphTargetsRelative = geometry_->morphTargetsRelative;
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

                    intersection = checkBufferGeometryIntersection(
                            this, groupMaterial, raycaster, _ray, *position,
                            morphPosition, morphTargetsRelative, uv, uv2, a, b, c);

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

                intersection = checkBufferGeometryIntersection(
                        this, material(), raycaster, _ray, *position,
                        morphPosition, morphTargetsRelative, uv, uv2, a, b, c);

                if (intersection) {

                    intersection->faceIndex = i / 3;// triangle number in indexed buffer semantics
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

                    intersection = checkBufferGeometryIntersection(
                            this, groupMaterial, raycaster, _ray, *position,
                            morphPosition, morphTargetsRelative, uv, uv2, a, b, c);

                    if (intersection) {

                        intersection->faceIndex = j / 3;// triangle number in non-indexed buffer semantics
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

                intersection = checkBufferGeometryIntersection(
                        this, material(), raycaster, _ray, *position,
                        morphPosition, morphTargetsRelative, uv, uv2, a, b, c);

                if (intersection) {

                    intersection->faceIndex = i / 3;// triangle number in non-indexed buffer semantics
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

std::string Mesh::type() const {

    return "Mesh";
}

BufferGeometry* Mesh::geometry() {

    return geometry_.get();
}

const BufferGeometry* Mesh::geometry() const {

    return geometry_.get();
}

void Mesh::setGeometry(const std::shared_ptr<BufferGeometry>& geometry) {

    geometry_ = geometry;
}

Material* Mesh::material() {

    return materials_.front().get();
}

void Mesh::setMaterial(const std::shared_ptr<Material>& material) {

    setMaterials({material});
}

void Mesh::setMaterials(const std::vector<std::shared_ptr<Material>>& materials) {

    materials_ = materials;
}

size_t Mesh::numMaterials() const {

    return materials_.size();
}

std::shared_ptr<Mesh> Mesh::create(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material) {

    return std::make_shared<Mesh>(std::move(geometry), std::move(material));
}

std::shared_ptr<Mesh> Mesh::create(std::shared_ptr<BufferGeometry> geometry, std::vector<std::shared_ptr<Material>> materials) {

    return std::make_shared<Mesh>(std::move(geometry), std::move(materials));
}
