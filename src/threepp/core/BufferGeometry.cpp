
#include "threepp/core/BufferGeometry.hpp"

using namespace threepp;

unsigned int BufferGeometry::_id = 0;

namespace {

    Matrix4 _m1;
    Vector3 _offset;
    Box3 _box;
    Vector3 _vector;

}// namespace


std::vector<int> &BufferGeometry::getIndex() {

    return this->index_;
}
BufferGeometry &BufferGeometry::setIndex(const std::vector<int> &index) {

    this->index_ = index;

    return *this;
}

void BufferGeometry::setAttribute(const std::string &name, std::unique_ptr<BufferAttribute> attribute) {

    attributes_[name] = std::move(attribute);
}

bool BufferGeometry::hasAttribute(const std::string &name) {

    return attributes_.count(name);
}

void BufferGeometry::addGroup(int start, int count, int materialIndex) {

    groups.emplace_back(GeometryGroup{start, count, materialIndex});
}

void BufferGeometry::clearGroups() {

    groups.clear();
}

BufferGeometry &BufferGeometry::setDrawRange(int start, int count) {

    this->drawRange.start = start;
    this->drawRange.count = count;

    return *this;
}

BufferGeometry &BufferGeometry::applyMatrix4(const Matrix4 &matrix) {

    if (hasAttribute("position")) {

        auto position = dynamic_cast<TypedBufferAttribute<float> *>(this->attributes_.at("position").get());

        position->applyMatrix4(matrix);

        position->needsUpdate();
    }


    if (hasAttribute("normal")) {

        auto normal = dynamic_cast<TypedBufferAttribute<float> *>(this->attributes_.at("normal").get());

        auto normalMatrix = Matrix3().getNormalMatrix(matrix);

        normal->applyNormalMatrix(normalMatrix);

        normal->needsUpdate();
    }


    if (hasAttribute("tangent")) {

        auto tangent = dynamic_cast<TypedBufferAttribute<float> *>(this->attributes_.at("tangent").get());

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

BufferGeometry &BufferGeometry::applyQuaternion(const Quaternion &q) {

    _m1.makeRotationFromQuaternion(q);

    this->applyMatrix4(_m1);

    return *this;
}

BufferGeometry &BufferGeometry::rotateX(float angle) {

    // rotate geometry around world x-axis

    _m1.makeRotationX(angle);

    this->applyMatrix4(_m1);

    return *this;
}

BufferGeometry &BufferGeometry::rotateY(float angle) {

    // rotate geometry around world y-axis

    _m1.makeRotationY(angle);

    this->applyMatrix4(_m1);

    return *this;
}

BufferGeometry &BufferGeometry::rotateZ(float angle) {

    // rotate geometry around world z-axis

    _m1.makeRotationZ(angle);

    this->applyMatrix4(_m1);

    return *this;
}
BufferGeometry &BufferGeometry::translate(float x, float y, float z) {

    // translate geometry

    _m1.makeTranslation(x, y, z);

    this->applyMatrix4(_m1);

    return *this;
}
BufferGeometry &BufferGeometry::scale(float x, float y, float z) {

    // scale geometry

    _m1.makeScale(x, y, z);

    this->applyMatrix4(_m1);

    return *this;
}
BufferGeometry &BufferGeometry::center() {

    this->computeBoundingBox();

    this->boundingBox->getCenter(_offset);
    _offset.negate();

    this->translate(_offset.x, _offset.y, _offset.z);

    return *this;
}
void BufferGeometry::computeBoundingBox() {

    if (!this->boundingBox) {

        this->boundingBox = Box3();
    }

    if (this->attributes_.count("position") != 0) {

        const auto position = dynamic_cast<TypedBufferAttribute<float> *>(this->attributes_.at("position").get());

        position->setFromBufferAttribute(*this->boundingBox);

    } else {

        this->boundingBox->makeEmpty();
    }

    if (std::isnan(this->boundingBox->min().x) || std::isnan(this->boundingBox->min().y) || std::isnan(this->boundingBox->min().z)) {

        std::cerr << "THREE.BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The 'position' attribute is likely to have NaN values." << std::endl;
    }
}
void BufferGeometry::computeBoundingSphere() {

    if (!this->boundingSphere) {

        this->boundingSphere = Sphere();
    }

    if (this->attributes_.count("position") != 0) {

        const auto &position = dynamic_cast<TypedBufferAttribute<float> *>(this->attributes_.at("position").get());

        // first, find the center of the bounding sphere

        auto center = this->boundingSphere->center;

        position->setFromBufferAttribute(_box);

        // process morph attributes if present

        _box.getCenter(center);

        // second, try to find a boundingSphere with a radius smaller than the
        // boundingSphere of the boundingBox: sqrt(3) smaller in the best case

        float maxRadiusSq = 0;

        for (auto i = 0, il = position->count(); i < il; i++) {

            position->setFromBufferAttribute(_vector, i);

            maxRadiusSq = std::max(maxRadiusSq, center.distanceToSquared(_vector));
        }

        this->boundingSphere->radius = std::sqrt(maxRadiusSq);

        if (std::isnan(this->boundingSphere->radius)) {

            std::cerr << "THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The 'position' attribute is likely to have NaN values." << std::endl;
        }
    }
}

const std::unordered_map<std::string, std::unique_ptr<BufferAttribute>> &BufferGeometry::getAttributes() const {
    return attributes_;
}
