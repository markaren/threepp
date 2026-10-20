
#include "threepp/objects/InstancedMesh.hpp"

#include "threepp/core/Raycaster.hpp"

#include <algorithm>
#include <stdexcept>

using namespace threepp;

namespace {

    // Bounds are the buffer capacity (maxCount), not the live count: setCount()
    // may shrink the drawn range, but staging data for instances that will be
    // re-enabled later is legal, exactly like writing instanceMatrix in three.js.
    void checkInstanceIndex(size_t index, size_t maxCount) {

        if (index >= maxCount) {

            throw std::out_of_range("InstancedMesh: instance index " + std::to_string(index) +
                                    " out of range (capacity " + std::to_string(maxCount) + ")");
        }
    }

}// namespace


InstancedMesh::InstancedMesh(
        std::shared_ptr<BufferGeometry> geometry,
        std::shared_ptr<Material> material,
        size_t count)
    : Mesh(std::move(geometry), std::move(material)),
      count_(count), maxCount_(count), instanceMatrix_(FloatBufferAttribute::create(std::vector<float>(count * 16), 16)) {

    Matrix4 identity;
    for (unsigned i = 0; i < count; i++) {

        this->setMatrixAt(i, identity);
    }
}

size_t InstancedMesh::count() const {

    return count_;
}

void InstancedMesh::setCount(size_t count) {

    const size_t next = std::min(maxCount_, count);
    if (next == count_) return;
    count_ = next;
    structureChanged();
}

FloatBufferAttribute* InstancedMesh::instanceMatrix() const {

    return instanceMatrix_.get();
}

FloatBufferAttribute* InstancedMesh::instanceColor() const {

    return instanceColor_ ? instanceColor_.get() : nullptr;
}


const std::string& InstancedMesh::type() const {

    static const std::string typeName = "InstancedMesh";
    return typeName;
}

void InstancedMesh::getColorAt(size_t index, Color& color) const {

    if (!this->instanceColor_) {

        throw std::runtime_error("InstancedMesh: no instance colors set (call setColorAt first)");
    }
    checkInstanceIndex(index, maxCount_);

    color.fromArray(this->instanceColor_->array(), index * 3);
}

void InstancedMesh::getMatrixAt(size_t index, Matrix4& matrix) const {

    checkInstanceIndex(index, maxCount_);

    matrix.fromArray(this->instanceMatrix_->array(), index * 16);
}

void InstancedMesh::setColorAt(size_t index, const Color& color) {

    checkInstanceIndex(index, maxCount_);

    if (!this->instanceColor_) {

        this->instanceColor_ = FloatBufferAttribute ::create(std::vector<float>(maxCount_ * 3), 3);
    }

    color.toArray(this->instanceColor_->array(), index * 3);
}

void InstancedMesh::setColors(size_t offset, size_t n, const float* rgb) {

    if (n == 0) return;
    checkInstanceIndex(offset + n - 1, maxCount_);

    if (!this->instanceColor_) {

        this->instanceColor_ = FloatBufferAttribute ::create(std::vector<float>(maxCount_ * 3), 3);
    }

    std::copy_n(rgb, n * 3, this->instanceColor_->array().data() + offset * 3);
    this->instanceColor_->needsUpdate();
}

void InstancedMesh::setMatrixAt(size_t index, const Matrix4& matrix) const {

    checkInstanceIndex(index, maxCount_);

    matrix.toArray(this->instanceMatrix_->array(), index * 16);
}

namespace {

    // Same reading as Mesh::setFade: NaN is fully drawn.
    float clampFade(float f) {

        return !(f < 1.f) ? 1.f : (f > 0.f ? f : 0.f);
    }

}// namespace

void InstancedMesh::ensureInstanceFade() {

    if (instanceFade_) return;

    std::vector<float> data(maxCount_ * 2, 0.f);
    for (size_t i = 0; i < maxCount_; ++i) data[i * 2] = 1.f;
    instanceFade_ = FloatBufferAttribute::create(std::move(data), 2);
}

void InstancedMesh::setFadeAt(size_t index, float fade, bool complement) {

    checkInstanceIndex(index, maxCount_);
    ensureInstanceFade();

    auto& a = instanceFade_->array();
    a[index * 2] = clampFade(fade);
    a[index * 2 + 1] = complement ? 1.f : 0.f;
    instanceFade_->needsUpdate();
}

void InstancedMesh::setFades(size_t offset, size_t n, const float* fades, const bool* complement) {

    if (n == 0) return;
    checkInstanceIndex(offset + n - 1, maxCount_);
    ensureInstanceFade();

    auto& a = instanceFade_->array();
    for (size_t i = 0; i < n; ++i) {
        a[(offset + i) * 2] = clampFade(fades[i]);
        a[(offset + i) * 2 + 1] = (complement && complement[i]) ? 1.f : 0.f;
    }
    instanceFade_->needsUpdate();
}

float InstancedMesh::getFadeAt(size_t index) const {

    checkInstanceIndex(index, maxCount_);
    return instanceFade_ ? instanceFade_->array()[index * 2] : fade();
}

bool InstancedMesh::getFadeComplementAt(size_t index) const {

    checkInstanceIndex(index, maxCount_);
    return instanceFade_ ? instanceFade_->array()[index * 2 + 1] != 0.f : fadeComplement();
}

FloatBufferAttribute* InstancedMesh::instanceFade() const {

    return instanceFade_.get();
}

void InstancedMesh::dispose() {

    // Every call, as in three.js and Texture::dispose; see Material::dispose.
    dispatchEvent("dispose", this);
}

void InstancedMesh::raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) {

    const auto& matrixWorld = this->matrixWorld;
    const auto raycastTimes = this->count_;

    _mesh.setGeometry(geometry_);
    _mesh.setMaterials(materials_);

    if (!_mesh.material()) return;

    for (unsigned instanceId = 0; instanceId < raycastTimes; instanceId++) {

        // calculate the world matrix for each instance

        this->getMatrixAt(instanceId, _instanceLocalMatrix);

        _instanceWorldMatrix.multiplyMatrices(*matrixWorld, _instanceLocalMatrix);

        // the mesh represents this single instance

        _mesh.matrixWorld->copy(_instanceWorldMatrix);

        _mesh.raycast(raycaster, _instanceIntersects);

        // process the result of raycast

        for (auto& intersect : _instanceIntersects) {

            intersect.instanceId = instanceId;
            intersect.object = this;
            intersects.emplace_back(intersect);
        }

        _instanceIntersects.clear();
    }
}

void InstancedMesh::copy(const Object3D& source, bool recursive) {
    Mesh::copy(source, recursive);

    if (const auto m = source.as<InstancedMesh>()) {

        instanceMatrix_->copy(*m->instanceMatrix_);
        instanceMatrix_->needsUpdate();

        if (m->instanceColor_) {

            if (instanceColor_) {

                instanceColor_->copy(*m->instanceColor_);
                instanceColor_->needsUpdate();
            } else {

                instanceColor_ = m->instanceColor_->clone();
            }
        }

        if (m->instanceFade_) {

            if (instanceFade_) {

                instanceFade_->copy(*m->instanceFade_);
                instanceFade_->needsUpdate();
            } else {

                instanceFade_ = m->instanceFade_->clone();
            }
        }

        count_ = m->count_;
        maxCount_ = m->maxCount_;

        boundingBox = m->boundingBox;
        boundingSphere = m->boundingSphere;
    }
}

InstancedMesh::~InstancedMesh() {
    dispose();
}

std::shared_ptr<InstancedMesh> InstancedMesh::create(
        std::shared_ptr<BufferGeometry> geometry,
        std::shared_ptr<Material> material,
        size_t count) {

    return std::make_shared<InstancedMesh>(std::move(geometry), std::move(material), count);
}

std::shared_ptr<Object3D> InstancedMesh::createDefault() {

    return create(nullptr, nullptr, maxCount_);
}

void InstancedMesh::computeBoundingBox() {

    const auto geometry = this->geometry();
    const auto count = this->count_;

    if (!this->boundingBox) {

        this->boundingBox = Box3();
    }

    if (!geometry->boundingBox) {

        geometry->computeBoundingBox();
    }

    this->boundingBox->makeEmpty();

    for (unsigned i = 0; i < count; i++) {

        this->getMatrixAt(i, _instanceLocalMatrix);

        _box3.copy(*geometry->boundingBox).applyMatrix4(_instanceLocalMatrix);

        this->boundingBox->union_(_box3);
    }
}

void InstancedMesh::computeBoundingSphere() {

    const auto geometry = this->geometry();
    const auto count = this->count_;

    if (!this->boundingSphere) {

        this->boundingSphere = Sphere();
    }

    if (!geometry->boundingSphere) {

        geometry->computeBoundingSphere();
    }

    this->boundingSphere->makeEmpty();

    for (unsigned i = 0; i < count; i++) {

        this->getMatrixAt(i, _instanceLocalMatrix);

        _sphere.copy(*geometry->boundingSphere).applyMatrix4(_instanceLocalMatrix);

        this->boundingSphere->union_(_sphere);
    }
}
