
#include "threepp/helpers/PlaneHelper.hpp"

#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

using namespace threepp;

PlaneHelper::PlaneHelper(const Plane& plane, float size, const Color& color)
    : Line(BufferGeometry::create(), LineBasicMaterial::create()), plane(plane), size(size) {

    std::vector<float> positions{1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0};

    geometry_->setAttribute("position", FloatBufferAttribute::create(positions, 3));
    geometry_->computeBoundingSphere();

    auto _material = material()->as<MaterialWithColor>();
    _material->color.copy(color);
    _material->toneMapped = false;

    std::vector<float> positions2{1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1};

    auto geometry2 = BufferGeometry::create();
    geometry2->setAttribute("position", FloatBufferAttribute::create(positions2, 3));
    geometry2->computeBoundingSphere();

    auto material2 = MeshBasicMaterial::create();
    material2->color.copy(color);
    material2->opacity = 0.2f;
    material2->transparent = true;
    material2->depthWrite = false;
    material2->toneMapped = false;

    mesh_ = Mesh::create(geometry2, material2);
    this->add(mesh_);
}

std::shared_ptr<PlaneHelper> PlaneHelper::create(const Plane& plane, float size, const Color& color) {

    return std::shared_ptr<PlaneHelper>(new PlaneHelper(plane, size, color));
}

void PlaneHelper::updateMatrixWorld(bool force) {

    auto scale = -plane.constant;
    if (std::abs(scale) < 1e-8) scale = 1e-8f;

    this->scale.set(0.5f * this->size, 0.5f * this->size, scale);

    this->children[0]->material()->side = (scale < 0) ? Side::Back : Side::Front;// renderer flips side when determinant < 0; flipping not wanted here

    this->lookAt(this->plane.normal);

    Object3D::updateMatrixWorld(force);
}
