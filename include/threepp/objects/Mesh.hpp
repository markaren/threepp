// https://github.com/mrdoob/three.js/blob/r129/src/objects/Mesh.js

#ifndef THREEPP_MESH_HPP
#define THREEPP_MESH_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"


namespace threepp {

    class Mesh: public Object3D {

    public:

        [[nodiscard]] std::string type() const override {

            return "Mesh";
        }

        BufferGeometry* geometry() override {

            return geometry_.get();
        }

        [[nodiscard]] const BufferGeometry* geometry() const {

            return geometry_.get();
        }

        void setGeometry(const std::shared_ptr<BufferGeometry>& geometry) {

            geometry_ = geometry;
        }

        Material* material() override {

            return materials_.front().get();
        }

        [[nodiscard]] std::vector<Material*> materials() override;

        void setMaterial(const std::shared_ptr<Material>& material) {

            setMaterials({material});
        }

        void setMaterials(const std::vector<std::shared_ptr<Material>>& materials) {

            materials_ = materials;
        }

        [[nodiscard]] size_t numMaterials() const {

            return materials_.size();
        }

        void raycast(Raycaster& raycaster, std::vector<Intersection>& intersects) override;

        std::shared_ptr<Object3D> clone(bool recursive = true) override;

        static std::shared_ptr<Mesh> create(
                std::shared_ptr<BufferGeometry> geometry = nullptr,
                std::shared_ptr<Material> material = nullptr) {

            return std::shared_ptr<Mesh>(new Mesh(std::move(geometry), std::move(material)));
        }

        static std::shared_ptr<Mesh> create(
                std::shared_ptr<BufferGeometry> geometry,
                std::vector<std::shared_ptr<Material>> materials) {

            return std::shared_ptr<Mesh>(new Mesh(std::move(geometry), std::move(materials)));
        }

        ~Mesh() override = default;

    protected:

        std::shared_ptr<BufferGeometry> geometry_;
        std::vector<std::shared_ptr<Material>> materials_;

        Mesh(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material)
            : geometry_(geometry ? std::move(geometry) : BufferGeometry::create()),
              materials_{material ? std::move(material) : MeshBasicMaterial::create()} {
        }

        Mesh(std::shared_ptr<BufferGeometry> geometry, std::vector<std::shared_ptr<Material>> materials)
            : geometry_(std::move(geometry)), materials_{std::move(materials)} {
        }
    };

}// namespace threepp

#endif//THREEPP_MESH_HPP
