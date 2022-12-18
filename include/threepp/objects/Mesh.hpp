// https://github.com/mrdoob/three.js/blob/r129/src/objects/Mesh.js

#ifndef THREEPP_MESH_HPP
#define THREEPP_MESH_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"

#include <memory>

namespace threepp {

    class Mesh : public Object3D {

    public:
        std::shared_ptr<BufferGeometry> geometry() override {

            return geometry_;
        }

        [[nodiscard]] std::shared_ptr<const BufferGeometry> geometry() const override {

            return geometry_;
        }

        std::shared_ptr<Material> material() override {

            return materials_[0];
        }

        [[nodiscard]] std::shared_ptr<const Material> material() const override {

            return materials_[0];
        }

        template<class T>
        std::shared_ptr<T> material() {

            return std::dynamic_pointer_cast<T>(material());
        }

        template<class T>
        std::shared_ptr<const T> material() const {

            return std::dynamic_pointer_cast<T>(material());
        }

        [[nodiscard]] std::vector<std::shared_ptr<Material>> materials() override {

            return materials_;
        }

        [[nodiscard]] size_t numMaterials() const {

            return materials_.size();
        }

        void raycast(Raycaster &raycaster, std::vector<Intersection> &intersects) override;

        std::shared_ptr<Mesh> clone(bool recursive = false);

        static std::shared_ptr<Mesh> create(
                std::shared_ptr<BufferGeometry> geometry = BufferGeometry::create(),
                std::shared_ptr<Material> material = MeshBasicMaterial::create()) {

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
            : geometry_(std::move(geometry)), materials_{std::move(material)} {
        }

        Mesh(std::shared_ptr<BufferGeometry> geometry, std::vector<std::shared_ptr<Material>> materials)
            : geometry_(std::move(geometry)), materials_{std::move(materials)} {
        }
    };

}// namespace threepp

#endif//THREEPP_MESH_HPP
