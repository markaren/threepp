
#include "threepp/helpers/SkeletonHelper.hpp"

#include "threepp/materials/LineBasicMaterial.hpp"

using namespace threepp;

namespace {

    Matrix4 _matrixWorldInv;
    Matrix4 _boneMatrix;
    Vector3 _vector;

    std::vector<Bone*> getBoneList(Object3D* object) {

        std::vector<Bone*> boneList;

        if (object && object->is<Bone>()) {

            boneList.emplace_back(object->as<Bone>());
        }

        for (const auto& child : object->children) {

            auto list = getBoneList(child);
            boneList.insert(boneList.end(), list.begin(), list.end());
        }

        return boneList;
    }

}// namespace

SkeletonHelper::SkeletonHelper(Object3D& object)
    : LineSegments(BufferGeometry::create(), LineBasicMaterial::create()),
      root(object) {

    bones = getBoneList(&object);

    std::vector<float> vertices;
    std::vector<float> colors;

    Color color1(0, 0, 1);
    Color color2(0, 1, 0);

    for (const auto& bone : bones) {

        if (bone->parent && bone->parent->is<Bone>()) {

            vertices.insert(vertices.end(), {0, 0, 0, 0, 0, 0});
            colors.insert(colors.end(), {color1.r, color1.g, color1.b, color2.r, color2.g, color2.b});
        }
    }

    geometry_->setAttribute("position", FloatBufferAttribute::create(vertices, 3));
    geometry_->setAttribute("color", FloatBufferAttribute::create(colors, 3));

    auto m = material()->as<LineBasicMaterial>();
    m->vertexColors = true;
    m->depthTest = false;
    m->depthWrite = false;
    m->toneMapped = false;
    m->transparent = true;

    this->matrix = object.matrixWorld;
    this->matrixAutoUpdate = false;
}

const std::vector<Bone*>& SkeletonHelper::getBones() const {

    return bones;
}

void SkeletonHelper::updateMatrixWorld(bool force) {

    auto position = geometry_->getAttribute<float>("position");

    _matrixWorldInv.copy(*this->root.matrixWorld).invert();

    int j = 0;
    for (auto& bone : bones) {

        if (bone->parent && bone->parent->is<Bone>()) {

            _boneMatrix.multiplyMatrices(_matrixWorldInv, *bone->matrixWorld);
            _vector.setFromMatrixPosition(_boneMatrix);
            position->setXYZ(j, _vector.x, _vector.y, _vector.z);

            _boneMatrix.multiplyMatrices(_matrixWorldInv, *bone->parent->matrixWorld);
            _vector.setFromMatrixPosition(_boneMatrix);
            position->setXYZ(j + 1, _vector.x, _vector.y, _vector.z);

            j += 2;
        }
    }

    geometry_->getAttribute("position")->needsUpdate();

    Object3D::updateMatrixWorld(force);
}
