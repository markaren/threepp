
#include "threepp/objects/InstancedMesh.hpp"

#include "threepp/core/Raycaster.hpp"

using namespace threepp;


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

    count_ = std::min(maxCount_, count);
}

FloatBufferAttribute* InstancedMesh::instanceMatrix() const {

    return instanceMatrix_.get();
}

FloatBufferAttribute* InstancedMesh::instanceColor() const {

    return instanceColor_ ? instanceColor_.get() : nullptr;
}


std::string InstancedMesh::type() const {

    return "InstancedMesh";
}

void InstancedMesh::getColorAt(size_t index, Color& color) const {

    color.fromArray(this->instanceColor_->array(), index * 3);
}

void InstancedMesh::getMatrixAt(size_t index, Matrix4& matrix) const {

    matrix.fromArray(this->instanceMatrix_->array(), index * 16);
}

void InstancedMesh::setColorAt(size_t index, const Color& color) {

    if (!this->instanceColor_) {

        this->instanceColor_ = FloatBufferAttribute ::create(std::vector<float>(maxCount_ * 3), 3);
    }

    color.toArray(this->instanceColor_->array(), index * 3);
}

void InstancedMesh::setMatrixAt(size_t index, const Matrix4& matrix) const {

    matrix.toArray(this->instanceMatrix_->array(), index * 16);
}

void InstancedMesh::dispose() {

    if (!disposed) {
        disposed = true;
        dispatchEvent("dispose", this);
    }
}

void InstancedMesh::raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) {

    const auto& matrixWorld = this->matrixWorld;
    const auto raycastTimes = this->count_;

    _mesh.setGeometry(geometry_);
    _mesh.setMaterials(materials_);

    if (!_mesh.material()) return;

    for (int instanceId = 0; instanceId < raycastTimes; instanceId++) {

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

InstancedMesh::~InstancedMesh() {
    dispose();
}

std::shared_ptr<InstancedMesh> InstancedMesh::create(
        std::shared_ptr<BufferGeometry> geometry,
        std::shared_ptr<Material> material,
        size_t count) {

    return std::make_shared<InstancedMesh>(std::move(geometry), std::move(material), count);
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
