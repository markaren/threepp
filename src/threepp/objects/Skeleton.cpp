
#include "threepp/objects/Skeleton.hpp"

#include <cmath>
#include <iostream>

using namespace threepp;

namespace {

    Matrix4 _identityMatrix;
    Matrix4 _offsetMatrix;

}// namespace


Skeleton::Skeleton(const std::vector<std::shared_ptr<Bone>>& bones, const std::vector<Matrix4>& boneInverses)
    : uuid_(math::generateUUID()), bones(bones), boneMatrices(bones.size() * 16), boneInverses(boneInverses) {

    init();
}

std::string Skeleton::uuid() const {

    return uuid_;
}

void Skeleton::init() {

    // calculate inverse bone matrices if necessary

    if (boneInverses.empty()) {

        this->calculateInverses();

    } else {

        // handle special case

        if (bones.size() != boneInverses.size()) {

            std::cerr << "THREE.Skeleton: Number of inverse bone matrices does not match amount of bones." << std::endl;

            this->boneInverses = {};

            for (unsigned i = 0, il = this->bones.size(); i < il; i++) {

                this->boneInverses.emplace_back();
            }
        }
    }
}

void Skeleton::calculateInverses() {

    this->boneInverses.clear();

    for (const auto& bone : this->bones) {

        Matrix4 inverse;

        if (bone) {

            inverse.copy(*bone->matrixWorld).invert();
        }

        this->boneInverses.emplace_back(inverse);
    }
}

void Skeleton::pose() {

    // recover the bind-time world matrices

    for (unsigned i = 0, il = this->bones.size(); i < il; i++) {

        const auto& bone = this->bones[i];

        if (bone) {

            bone->matrixWorld->copy(this->boneInverses[i]).invert();
        }
    }

    // compute the local matrices, positions, rotations and scales

    for (const auto& bone : this->bones) {

        if (bone) {

            if (bone->parent && bone->parent->is<Bone>()) {

                bone->matrix->copy(*bone->parent->matrixWorld).invert();
                bone->matrix->multiply(*bone->matrixWorld);

            } else {

                bone->matrix->copy(*bone->matrixWorld);
            }

            bone->matrix->decompose(bone->position, bone->quaternion, bone->scale);
        }
    }
}

void Skeleton::update() {

    // flatten bone matrices to array

    for (unsigned i = 0, il = bones.size(); i < il; i++) {

        // compute the offset between the current and the original transform

        const auto& matrix = bones[i] ? *bones[i]->matrixWorld : _identityMatrix;

        _offsetMatrix.multiplyMatrices(matrix, boneInverses[i]);
        _offsetMatrix.toArray(boneMatrices, i * 16);
    }

    if (boneTexture) {

        boneTexture->setData(boneMatrices);
        boneTexture->needsUpdate();
    }
}

Skeleton& Skeleton::computeBoneTexture() {

    // layout (1 matrix = 4 pixels)
    //      RGBA RGBA RGBA RGBA (=> column1, column2, column3, column4)
    //  with  8x8  pixel texture max   16 bones * 4 pixels =  (8 * 8)
    //       16x16 pixel texture max   64 bones * 4 pixels = (16 * 16)
    //       32x32 pixel texture max  256 bones * 4 pixels = (32 * 32)
    //       64x64 pixel texture max 1024 bones * 4 pixels = (64 * 64)

    auto size = float(std::sqrt(this->bones.size() * 4));// 4 pixels needed for 1 matrix
    int sizei = math::ceilPowerOfTwo(size);
    sizei = std::max(sizei, 4);

    auto boneMatrices = std::vector<float>(sizei * sizei * 4);// 4 floats per RGBA pixel
    //                boneMatrices.set(this->boneMatrices);                   // copy current values
    for (unsigned i = 0; i < this->boneMatrices.size(); ++i) {
        boneMatrices[i] = this->boneMatrices[i];
    }

    auto boneTexture = DataTexture::create(boneMatrices, sizei, sizei);
    boneTexture->format = Format::RGBA;
    boneTexture->type = Type::Float;

    this->boneMatrices = boneMatrices;
    this->boneTexture = boneTexture;
    this->boneTextureSize = sizei;

    return *this;
}

Bone* Skeleton::getBoneByName(const std::string& name) {

    for (const auto& bone : this->bones) {

        if (bone && bone->name == name) {

            return bone.get();
        }
    }

    return nullptr;
}

void Skeleton::dispose() {
    if (this->boneTexture) {

        this->boneTexture->dispose();

        this->boneTexture = nullptr;
    }
}

Skeleton::~Skeleton() {

    dispose();
}

std::shared_ptr<Skeleton> Skeleton::create(const std::vector<std::shared_ptr<Bone>>& bones, const std::vector<Matrix4>& boneInverses) {

    return std::shared_ptr<Skeleton>(new Skeleton(bones, boneInverses));
}
