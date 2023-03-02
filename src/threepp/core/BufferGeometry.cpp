
#include "threepp/core/BufferGeometry.hpp"

#include <utility>

using namespace threepp;

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

const std::unordered_map<std::string, std::unique_ptr<BufferAttribute>>& BufferGeometry::getAttributes() const {
    return attributes_;
}

void BufferGeometry::setAttribute(const std::string& name, std::unique_ptr<BufferAttribute> attribute) {

    attributes_[name] = std::move(attribute);
}

bool BufferGeometry::hasAttribute(const std::string& name) const {

    return attributes_.count(name);
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

void BufferGeometry::computeBoundingBox() {

    if (!this->boundingBox) {

        this->boundingBox = Box3();
    }

    if (this->attributes_.count("position") != 0) {

        const auto position = this->attributes_.at("position")->typed<float>();

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

    if (this->attributes_.count("position")) {

        const auto& position = this->attributes_.at("position")->typed<float>();

        // first, find the center of the bounding sphere

        auto& center = this->boundingSphere->center;

        Box3 _box;
        position->setFromBufferAttribute(_box);

        _box.getCenter(center);

        // second, try to find a boundingSphere with a radius smaller than the
        // boundingSphere of the boundingBox: sqrt(3) smaller in the best case

        float maxRadiusSq = 0;

        for (unsigned i = 0, il = position->count(); i < il; i++) {

            Vector3 _vector;
            position->setFromBufferAttribute(_vector, i);

            maxRadiusSq = std::max(maxRadiusSq, center.distanceToSquared(_vector));
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

    //        this->name = source.name;

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

void BufferGeometry::computeVertexNormals() {

    auto index = getIndex();

    const auto positionAttribute = this->getAttribute<float>( "position" );

    if ( positionAttribute ) {

        auto normalAttribute = this->getAttribute<float>("normal");

        if ( !normalAttribute ) {

            this->setAttribute("normal", FloatBufferAttribute::create( std::vector<float>( positionAttribute->count() * 3 ), 3 ) );
            normalAttribute = this->getAttribute<float>("normal");

        } else {

            // reset existing normals to zero

            for ( unsigned i = 0, il = normalAttribute->count(); i < il; i ++ ) {

                normalAttribute->setXYZ( i, 0, 0, 0 );

            }

        }

        Vector3 pA, pB, pC;
        Vector3 nA, nB, nC;
        Vector3 cb, ab;

        // indexed elements

        if ( index ) {

            for ( unsigned i = 0, il = index->count(); i < il; i += 3 ) {

                auto vA = index->getX( i + 0 );
                auto vB = index->getX( i + 1 );
                auto vC = index->getX( i + 2 );

                positionAttribute->setFromBufferAttribute(pA, vA);
                positionAttribute->setFromBufferAttribute(pB, vB);
                positionAttribute->setFromBufferAttribute(pC, vC);

                cb.subVectors( pC, pB );
                ab.subVectors( pA, pB );
                cb.cross( ab );

                normalAttribute->setFromBufferAttribute(nA, vA);
                normalAttribute->setFromBufferAttribute(nB, vB);
                normalAttribute->setFromBufferAttribute(nC, vC);

                nA.add( cb );
                nB.add( cb );
                nC.add( cb );

                normalAttribute->setXYZ( vA, nA.x, nA.y, nA.z );
                normalAttribute->setXYZ( vB, nB.x, nB.y, nB.z );
                normalAttribute->setXYZ( vC, nC.x, nC.y, nC.z );

            }

        } else {

            // non-indexed elements (unconnected triangle soup)

            for ( unsigned i = 0, il = positionAttribute->count(); i < il; i += 3 ) {

                positionAttribute->setFromBufferAttribute(pA, i + 0 );
                positionAttribute->setFromBufferAttribute(pB, i + 1 );
                positionAttribute->setFromBufferAttribute(pC, i + 2 );

                cb.subVectors( pC, pB );
                ab.subVectors( pA, pB );
                cb.cross( ab );

                normalAttribute->setXYZ( i + 0, cb.x, cb.y, cb.z );
                normalAttribute->setXYZ( i + 1, cb.x, cb.y, cb.z );
                normalAttribute->setXYZ( i + 2, cb.x, cb.y, cb.z );

            }

        }

        this->normalizeNormals();

        normalAttribute->needsUpdate();

    }

}

void BufferGeometry::dispose() {

    if (!disposed_) {
        disposed_ = true;
        this->dispatchEvent("dispose", this);
    }

}
