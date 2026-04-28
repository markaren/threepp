
#include "threepp/lights/RectAreaLight.hpp"

#include "threepp/constants.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"

using namespace threepp;


RectAreaLight::RectAreaLight(const Color& color, std::optional<float> intensity, float width, float height)
    : Light(color, intensity), width(width), height(height) {

    auto geometry = PlaneGeometry::create(width, height);
    auto material = MeshStandardMaterial::create();
    material->color = Color(0x000000);
    material->emissive = this->color;
    material->emissiveIntensity = this->intensity;
    material->roughness = 1.0f;
    material->metalness = 0.0f;
    material->side = Side::Double;

    mesh_ = Mesh::create(geometry, material);
    mesh_->name = "RectAreaLight_quad";
    this->add(mesh_);
}


std::string RectAreaLight::type() const {

    return "RectAreaLight";
}

float RectAreaLight::getPower() const {

    return this->intensity * math::PI * this->width * this->height;
}

void RectAreaLight::setPower(float power) {

    this->intensity = power / (math::PI * this->width * this->height);
}

const std::shared_ptr<Mesh>& RectAreaLight::mesh() const {

    return mesh_;
}

void RectAreaLight::updateMatrixWorld(bool force) {

    if (mesh_) {

        if (auto m = std::dynamic_pointer_cast<MeshStandardMaterial>(mesh_->material())) {

            m->emissive.copy(this->color);
            m->emissiveIntensity = this->intensity;
        }
    }

    Object3D::updateMatrixWorld(force);
}

void RectAreaLight::copy(const Object3D& source, bool recursive) {
    Light::copy(source, recursive);
}

std::shared_ptr<RectAreaLight> RectAreaLight::create(const Color& color, std::optional<float> intensity, float width, float height) {

    return std::shared_ptr<RectAreaLight>(new RectAreaLight(color, intensity, width, height));
}
