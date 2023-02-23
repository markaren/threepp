
#include "threepp/objects/InstancedMesh.hpp"

using namespace threepp;

namespace {

    Matrix4 _instanceLocalMatrix;
    Matrix4 _instanceWorldMatrix;

    std::vector<Intersection> _instanceIntersects;

    auto _mesh = Mesh::create(BufferGeometry::create(), MeshBasicMaterial::create());

}// namespace

void InstancedMesh::getColorAt(size_t index, Color &color) const {

    color.fromArray(this->instanceColor->array(), index * 3);
}

void InstancedMesh::getMatrixAt(size_t index, Matrix4 &matrix) const {

    matrix.fromArray(this->instanceMatrix->array(), index * 16);
}


void InstancedMesh::setColorAt(size_t index, const Color &color) {

    if (!this->instanceColor) {

        this->instanceColor = FloatBufferAttribute ::create(std::vector<float>(count * 3), 3);
    }

    color.toArray(this->instanceColor->array(), index * 3);
}

void InstancedMesh::setMatrixAt(size_t index, const Matrix4 &matrix) const {

    matrix.toArray(this->instanceMatrix->array(), index * 16);
}

void InstancedMesh::dispose() {

    dispatchEvent("dispose", this);
}

void InstancedMesh::raycast(Raycaster &raycaster, std::vector<Intersection> &intersects) {

    const auto matrixWorld = this->matrixWorld;
    const auto raycastTimes = this->count;

    _mesh->setGeometry(geometry_);
    _mesh->setMaterial(materials_);

    if (!_mesh->material()) return;

    for (int instanceId = 0; instanceId < raycastTimes; instanceId++) {

        // calculate the world matrix for each instance

        this->getMatrixAt(instanceId, _instanceLocalMatrix);

        _instanceWorldMatrix.multiplyMatrices(*matrixWorld, _instanceLocalMatrix);

        // the mesh represents this single instance

        _mesh->matrixWorld->copy(_instanceWorldMatrix);

        _mesh->raycast(raycaster, _instanceIntersects);

        // process the result of raycast

        for (auto intersect : _instanceIntersects) {

            intersect.instanceId = instanceId;
            intersect.object = this;
            intersects.emplace_back(intersect);
        }

        _instanceIntersects.clear();
    }
}
