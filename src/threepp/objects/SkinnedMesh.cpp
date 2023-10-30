
#include "threepp/objects/SkinnedMesh.hpp"

using namespace threepp;

namespace {

    Vector3 _basePosition;

    Vector4 _skinIndex;
    Vector4 _skinWeight;

    Vector3 _vector;
    Matrix4 _matrix;

}// namespace


SkinnedMesh::SkinnedMesh(const std::shared_ptr<BufferGeometry>& geometry, const std::shared_ptr<Material>& material)
    : Mesh(geometry, material) {}


std::string SkinnedMesh::type() const {

    return "SkinnedMesh";
}

void SkinnedMesh::bind(const std::shared_ptr<Skeleton>& skeleton, std::optional<Matrix4> bindMatrix) {

    this->skeleton = skeleton;

    if (!bindMatrix) {

        this->updateMatrixWorld(true);

        this->skeleton->calculateInverses();

        bindMatrix = *this->matrixWorld;
    }

    this->bindMatrix.copy(*bindMatrix);
    this->bindMatrixInverse.copy(*bindMatrix).invert();
}

void SkinnedMesh::pose() const {

    this->skeleton->pose();
}

void SkinnedMesh::normalizeSkinWeights() {

    Vector4 vector;

    auto skinWeight = this->geometry_->getAttribute<float>("skinWeight");

    for (unsigned i = 0, l = skinWeight->count(); i < l; i++) {

        vector.x = skinWeight->getX(i);
        vector.y = skinWeight->getY(i);
        vector.z = skinWeight->getZ(i);
        vector.w = skinWeight->getW(i);

        auto scale = 1.f / vector.manhattanLength();

        if (scale != std::numeric_limits<float>::infinity()) {

            vector.multiplyScalar(scale);

        } else {

            vector.set(1, 0, 0, 0);// do something reasonable
        }

        skinWeight->setXYZW(i, vector.x, vector.y, vector.z, vector.w);
    }
}

void SkinnedMesh::updateMatrixWorld(bool force) {
    Object3D::updateMatrixWorld(force);

    if (this->bindMode == BindMode::Attached) {
        this->bindMatrixInverse.copy(*this->matrixWorld).invert();
    } else {
        this->bindMatrixInverse.copy(this->bindMatrix).invert();
    }
}

void SkinnedMesh::boneTransform(size_t index, Vector3& target) {

    geometry_->getAttribute<int>("skinIndex")->setFromBufferAttribute(_skinIndex, index);
    geometry_->getAttribute<float>("skinWeight")->setFromBufferAttribute(_skinWeight, index);

    geometry_->getAttribute<float>("position")->setFromBufferAttribute(_basePosition, index);
    _basePosition.applyMatrix4(this->bindMatrix);

    target.set(0, 0, 0);

    for (unsigned i = 0; i < 4; i++) {

        const auto& weight = _skinWeight[i];

        if (weight != 0) {

            auto boneIndex = static_cast<int>(_skinIndex[i]);

            _matrix.multiplyMatrices(*skeleton->bones[boneIndex]->matrixWorld, skeleton->boneInverses[boneIndex]);

            target.addScaledVector(_vector.copy(_basePosition).applyMatrix4(_matrix), weight);
        }
    }

    target.applyMatrix4(this->bindMatrixInverse);
}
