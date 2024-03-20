
#include <memory>

#include "threepp/objects/InstancedMesh.hpp"

#include "threepp/core/Raycaster.hpp"

using namespace threepp;

namespace {

    Matrix4 _instanceLocalMatrix;
    Matrix4 _instanceWorldMatrix;

    std::vector<Intersection> _instanceIntersects;

}// namespace


InstancedMesh::InstancedMesh(
        std::shared_ptr<BufferGeometry> geometry,
        std::shared_ptr<Material> material,
        size_t capacity)
    : Mesh(std::move(geometry), std::move(material)),
      capacity_(capacity), drawInstanceCount_(capacity), instanceMatrix_(FloatBufferAttribute::create(std::vector<float>(capacity * 16), 16)) {

    this->frustumCulled = false;
}

size_t InstancedMesh::drawInstanceCount() const {
    return drawInstanceCount_;
}

void InstancedMesh::setDrawInstanceCount(size_t drawInstanceCount) {
    drawInstanceCount_ = capacity_ < drawInstanceCount ? capacity_ : drawInstanceCount;
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

        this->instanceColor_ = FloatBufferAttribute ::create(std::vector<float>(capacity_ * 3), 3);
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
    const auto raycastTimes = this->drawInstanceCount();

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
        size_t capacity) {

    return std::make_shared<InstancedMesh>(std::move(geometry), std::move(material), capacity);
}
