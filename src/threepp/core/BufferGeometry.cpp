
#include "threepp/core/BufferGeometry.hpp"

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"

#include <cmath>
#include <iostream>
#include <utility>

using namespace threepp;

namespace {

    std::unique_ptr<BufferAttribute> convertBufferAttribute(BufferAttribute& _attribute, const std::vector<unsigned int>& indices) {

        if (_attribute.typed<float>()) {

            auto attribute = _attribute.typed<float>();

            const auto& array = attribute->array();
            const auto itemSize = attribute->itemSize();
            const auto normalized = attribute->normalized();

            auto array2 = std::vector<float>(indices.size() * itemSize);

            unsigned index = 0, index2 = 0;

            for (unsigned i = 0, l = indices.size(); i < l; i++) {

                index = indices[i] * itemSize;

                for (auto j = 0; j < itemSize; j++) {

                    array2[index2++] = array[index++];
                }
            }

            return FloatBufferAttribute::create(array2, itemSize, normalized);
        }

        throw std::runtime_error("Unsupported operation");
    }

}// namespace

BufferGeometry::BufferGeometry()
    : uuid(math::generateUUID()) {}

std::string BufferGeometry::type() const {

    return "BufferGeometry";
}

bool BufferGeometry::hasIndex() const {

    return index_ != nullptr;
}

IntBufferAttribute* BufferGeometry::getIndex() {

    if (!index_) return nullptr;

    return this->index_.get();
}

const IntBufferAttribute* BufferGeometry::getIndex() const {

    if (!index_) return nullptr;

    return this->index_.get();
}

BufferAttribute* BufferGeometry::getAttribute(const std::string& name) {

    if (!hasAttribute(name)) return nullptr;

    return attributes_.at(name).get();
}

std::vector<std::shared_ptr<BufferAttribute>>* BufferGeometry::getMorphAttribute(const std::string& name) {

    if (!morphAttributes_.count(name)) return nullptr;

    return &morphAttributes_.at(name);
}

std::vector<std::shared_ptr<BufferAttribute>>* BufferGeometry::getOrCreateMorphAttribute(const std::string& name) {

    return &morphAttributes_[name];
}

const std::unordered_map<std::string, std::shared_ptr<BufferAttribute>>& BufferGeometry::getAttributes() const {

    return attributes_;
}

const std::unordered_map<std::string, std::vector<std::shared_ptr<BufferAttribute>>>& BufferGeometry::getMorphAttributes() const {

    return morphAttributes_;
}

void BufferGeometry::setAttribute(const std::string& name, std::shared_ptr<BufferAttribute> attribute) {

    attributes_[name] = std::move(attribute);
}

void BufferGeometry::deleteAttribute(const std::string& name) {

    if (attributes_.count(name)) {

        attributes_.erase(name);
    }
}

bool BufferGeometry::hasAttribute(const std::string& name) const {

    return attributes_.contains(name);
}

void BufferGeometry::addGroup(int start, int count, unsigned int materialIndex) {

    groups.emplace_back(GeometryGroup{start, count, materialIndex});
}

void BufferGeometry::clearGroups() {

    groups.clear();
}

BufferGeometry& BufferGeometry::setDrawRange(int start, int count) {

    this->drawRange.start = start;
    this->drawRange.count = count;

    return *this;
}

BufferGeometry& BufferGeometry::applyMatrix4(const Matrix4& matrix) {

    if (hasAttribute("position")) {

        auto position = dynamic_cast<TypedBufferAttribute<float>*>(this->attributes_.at("position").get());

        position->applyMatrix4(matrix);

        position->needsUpdate();
    }


    if (hasAttribute("normal")) {

        auto normal = dynamic_cast<TypedBufferAttribute<float>*>(this->attributes_.at("normal").get());

        auto normalMatrix = Matrix3().getNormalMatrix(matrix);

        normal->applyNormalMatrix(normalMatrix);

        normal->needsUpdate();
    }


    if (hasAttribute("tangent")) {

        auto tangent = dynamic_cast<TypedBufferAttribute<float>*>(this->attributes_.at("tangent").get());

        tangent->transformDirection(matrix);

        tangent->needsUpdate();
    }

    if (!this->boundingBox) {

        this->computeBoundingBox();
    }

    if (!this->boundingSphere) {

        this->computeBoundingSphere();
    }

    return *this;
}

BufferGeometry& BufferGeometry::applyQuaternion(const Quaternion& q) {

    Matrix4 _m1{};
    _m1.makeRotationFromQuaternion(q);

    this->applyMatrix4(_m1);

    return *this;
}

BufferGeometry& BufferGeometry::rotateX(float angle) {

    // rotate geometry around world x-axis

    Matrix4 _m1{};
    _m1.makeRotationX(angle);

    this->applyMatrix4(_m1);

    return *this;
}

BufferGeometry& BufferGeometry::rotateY(float angle) {

    // rotate geometry around world y-axis

    Matrix4 _m1{};
    _m1.makeRotationY(angle);

    this->applyMatrix4(_m1);

    return *this;
}

BufferGeometry& BufferGeometry::rotateZ(float angle) {

    // rotate geometry around world z-axis

    Matrix4 _m1{};
    _m1.makeRotationZ(angle);

    this->applyMatrix4(_m1);

    return *this;
}

BufferGeometry& BufferGeometry::translate(float x, float y, float z) {

    // translate geometry

    Matrix4 _m1{};
    _m1.makeTranslation(x, y, z);

    this->applyMatrix4(_m1);

    return *this;
}

BufferGeometry& BufferGeometry::scale(float x, float y, float z) {

    // scale geometry

    Matrix4 _m1{};
    _m1.makeScale(x, y, z);

    this->applyMatrix4(_m1);

    return *this;
}

BufferGeometry& BufferGeometry::center() {

    this->computeBoundingBox();

    Vector3 _offset{};
    this->boundingBox->getCenter(_offset);
    _offset.negate();

    this->translate(_offset.x, _offset.y, _offset.z);

    return *this;
}

BufferGeometry& BufferGeometry::setFromPoints(const std::vector<Vector2>& points) {

    std::vector<float> position(points.size() * 3);

    for (unsigned i = 0, j = 0, l = points.size(); i < l; i++) {

        const auto& point = points[i];
        position[j++] = point.x;
        position[j++] = point.y;
        position[j++] = 0;
    }

    this->setAttribute("position", FloatBufferAttribute::create(position, 3));

    return *this;
}

BufferGeometry& BufferGeometry::setFromPoints(const std::vector<Vector3>& points) {

    std::vector<float> position(points.size() * 3);

    for (unsigned i = 0, j = 0, l = points.size(); i < l; i++) {

        const auto& point = points[i];
        position[j++] = point.x;
        position[j++] = point.y;
        position[j++] = point.z;
    }

    this->setAttribute("position", FloatBufferAttribute::create(position, 3));

    return *this;
}

void BufferGeometry::computeBoundingBox() {

    if (!this->boundingBox) {

        this->boundingBox = Box3();
    }

    if (const auto position = this->getAttribute<float>("position")) {

        position->setFromBufferAttribute(*this->boundingBox);

        if (const auto morphAttributesPosition = this->getMorphAttribute("position")) {

            Box3 _box;
            Vector3 _vector;

            for (unsigned i = 0, il = morphAttributesPosition->size(); i < il; i++) {

                auto morphAttribute = morphAttributesPosition->at(i)->typed<float>();
                morphAttribute->setFromBufferAttribute(_box);

                if (morphTargetsRelative) {

                    _vector.addVectors(this->boundingBox->min(), _box.min());
                    this->boundingBox->expandByPoint(_vector);

                    _vector.addVectors(this->boundingBox->max(), _box.max());
                    this->boundingBox->expandByPoint(_vector);

                } else {

                    this->boundingBox->expandByPoint(_box.min());
                    this->boundingBox->expandByPoint(_box.max());
                }
            }
        }

    } else {

        this->boundingBox->makeEmpty();
    }

    if (this->boundingBox->min().isNan()) {

        std::cerr << "THREE.BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The 'position' attribute is likely to have NaN values." << std::endl;
    }
}

void BufferGeometry::computeBoundingSphere() {

    if (!this->boundingSphere) {

        this->boundingSphere = Sphere();
    }

    Vector3 _vector;

    if (const auto position = this->getAttribute<float>("position")) {

        // first, find the center of the bounding sphere

        auto& center = this->boundingSphere->center;

        Box3 _box;
        position->setFromBufferAttribute(_box);

        // process morph attributes if present

        if (const auto morphAttributesPosition = getMorphAttribute("position")) {

            Box3 _boxMorphTargets;

            for (unsigned i = 0, il = morphAttributesPosition->size(); i < il; i++) {

                auto morphAttribute = morphAttributesPosition->at(i)->typed<float>();
                morphAttribute->setFromBufferAttribute(_boxMorphTargets);

                if (this->morphTargetsRelative) {
                    _vector.addVectors(_box.min(), _boxMorphTargets.min());
                    _box.expandByPoint(_vector);

                    _vector.addVectors(_box.max(), _boxMorphTargets.max());
                    _box.expandByPoint(_vector);
                } else {

                    _box.expandByPoint(_boxMorphTargets.min());
                    _box.expandByPoint(_boxMorphTargets.max());
                }
            }
        }

        _box.getCenter(center);

        // second, try to find a boundingSphere with a radius smaller than the
        // boundingSphere of the boundingBox: sqrt(3) smaller in the best case

        float maxRadiusSq = 0;

        for (unsigned i = 0, il = position->count(); i < il; i++) {

            position->setFromBufferAttribute(_vector, i);

            maxRadiusSq = std::max(maxRadiusSq, center.distanceToSquared(_vector));
        }

        // process morph attributes if present

        if (const auto morphAttributesPosition = getMorphAttribute("position")) {

            Vector3 _offset;

            for (unsigned i = 0, il = morphAttributesPosition->size(); i < il; i++) {

                auto morphAttribute = morphAttributesPosition->at(i)->typed<float>();

                for (unsigned j = 0, jl = morphAttribute->count(); j < jl; j++) {

                    morphAttribute->setFromBufferAttribute(_vector, j);

                    if (morphTargetsRelative) {

                        position->setFromBufferAttribute(_offset, j);
                        _vector.add(_offset);
                    }

                    maxRadiusSq = std::max(maxRadiusSq, center.distanceToSquared(_vector));
                }
            }
        }

        this->boundingSphere->radius = std::sqrt(maxRadiusSq);

        if (std::isnan(this->boundingSphere->radius)) {

            std::cerr << "THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The 'position' attribute is likely to have NaN values." << std::endl;
        }
    }
}

void BufferGeometry::normalizeNormals() {

    auto normals = getAttribute<float>("normal");

    Vector3 _vector{};
    for (int i = 0, il = normals->count(); i < il; i++) {

        normals->setFromBufferAttribute(_vector, i);

        _vector.normalize();

        normals->setXYZ(i, _vector.x, _vector.y, _vector.z);
    }
}

void BufferGeometry::copy(const BufferGeometry& source) {
    // reset

    this->index_ = nullptr;
    this->attributes_.clear();
    this->groups.clear();
    this->boundingBox = std::nullopt;
    this->boundingSphere = std::nullopt;

    // name

    this->name = source.name;

    // index

    auto& index = source.index_;

    if (index) {

        this->index_ = index->clone();
    }

    // attributes

    auto& attributes = source.attributes_;

    for (const auto& [name, attribute] : attributes) {

        if (attribute->typed<unsigned int>()) {
            this->setAttribute(name, attribute->typed<unsigned int>()->clone());
        } else if (attribute->typed<float>()) {
            this->setAttribute(name, attribute->typed<float>()->clone());
        } else {
            throw std::runtime_error("TODO");
        }
    }


    // groups

    for (auto& group : source.groups) {

        this->addGroup(group.start, group.count, group.materialIndex);
    }

    // bounding box

    auto& boundingBox = source.boundingBox;

    if (boundingBox) {

        this->boundingBox = boundingBox;
    }

    // bounding sphere

    auto& boundingSphere = source.boundingSphere;

    if (boundingSphere) {

        this->boundingSphere = boundingSphere;
    }

    // draw range

    this->drawRange.start = source.drawRange.start;
    this->drawRange.count = source.drawRange.count;
}

std::shared_ptr<BufferGeometry> BufferGeometry::toNonIndexed() const {

    if (!this->hasIndex()) {

        std::cerr << "THREE.BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed." << std::endl;
        return nullptr;
    }

    auto geometry2 = BufferGeometry::create();

    const auto& indices = this->index_->array();
    const auto& attributes = this->attributes_;

    // attributes

    for (const auto& [name, attribute] : attributes) {

        auto newAttribute = convertBufferAttribute(*attribute, indices);

        geometry2->setAttribute(name, std::move(newAttribute));
    }

    // morph attributes

    for (auto& [name, attr] : morphAttributes_) {

        std::vector<std::shared_ptr<BufferAttribute>> morphArray;
        auto& morphAttribute = morphAttributes_.at(name);// morphAttribute: array of Float32BufferAttributes

        for (const auto& attribute : morphAttribute) {

            auto newAttribute = convertBufferAttribute(*attribute, indices);

            morphArray.emplace_back(std::move(newAttribute));
        }

        geometry2->morphAttributes_[name] = morphArray;
    }

    geometry2->morphTargetsRelative = this->morphTargetsRelative;

    // groups

    const auto& groups = this->groups;

    for (unsigned i = 0, l = groups.size(); i < l; i++) {

        const auto group = groups[i];
        geometry2->addGroup(group.start, group.count, group.materialIndex);
    }

    return geometry2;
}


void BufferGeometry::computeVertexNormals() {

    auto index = getIndex();

    const auto positionAttribute = this->getAttribute<float>("position");

    if (positionAttribute) {

        auto normalAttribute = this->getAttribute<float>("normal");

        if (!normalAttribute) {

            this->setAttribute("normal", FloatBufferAttribute::create(std::vector<float>(positionAttribute->count() * 3), 3));
            normalAttribute = this->getAttribute<float>("normal");

        } else {

            // reset existing normals to zero

            for (unsigned i = 0, il = normalAttribute->count(); i < il; i++) {

                normalAttribute->setXYZ(i, 0, 0, 0);
            }
        }

        Vector3 pA, pB, pC;
        Vector3 nA, nB, nC;
        Vector3 cb, ab;

        // indexed elements

        if (index) {

            for (unsigned i = 0, il = index->count(); i < il; i += 3) {

                auto vA = index->getX(i + 0);
                auto vB = index->getX(i + 1);
                auto vC = index->getX(i + 2);

                positionAttribute->setFromBufferAttribute(pA, vA);
                positionAttribute->setFromBufferAttribute(pB, vB);
                positionAttribute->setFromBufferAttribute(pC, vC);

                cb.subVectors(pC, pB);
                ab.subVectors(pA, pB);
                cb.cross(ab);

                normalAttribute->setFromBufferAttribute(nA, vA);
                normalAttribute->setFromBufferAttribute(nB, vB);
                normalAttribute->setFromBufferAttribute(nC, vC);

                nA.add(cb);
                nB.add(cb);
                nC.add(cb);

                normalAttribute->setXYZ(vA, nA.x, nA.y, nA.z);
                normalAttribute->setXYZ(vB, nB.x, nB.y, nB.z);
                normalAttribute->setXYZ(vC, nC.x, nC.y, nC.z);
            }

        } else {

            // non-indexed elements (unconnected triangle soup)

            for (unsigned i = 0, il = positionAttribute->count(); i < il; i += 3) {

                positionAttribute->setFromBufferAttribute(pA, i + 0);
                positionAttribute->setFromBufferAttribute(pB, i + 1);
                positionAttribute->setFromBufferAttribute(pC, i + 2);

                cb.subVectors(pC, pB);
                ab.subVectors(pA, pB);
                cb.cross(ab);

                normalAttribute->setXYZ(i + 0, cb.x, cb.y, cb.z);
                normalAttribute->setXYZ(i + 1, cb.x, cb.y, cb.z);
                normalAttribute->setXYZ(i + 2, cb.x, cb.y, cb.z);
            }
        }

        this->normalizeNormals();

        normalAttribute->needsUpdate();
    }
}

std::shared_ptr<BufferGeometry> BufferGeometry::clone() const {
    auto g = std::make_shared<BufferGeometry>();
    g->copy(*this);

    return g;
}

void BufferGeometry::dispose() {

    if (!disposed_) {
        disposed_ = true;
        this->dispatchEvent("dispose", this);
    }
}

BufferGeometry::~BufferGeometry() {
    dispose();
}

std::shared_ptr<BufferGeometry> BufferGeometry::create() {

    return std::make_shared<BufferGeometry>();
}
