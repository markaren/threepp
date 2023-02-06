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
        std::shared_ptr<BufferGeometry> geometry_;
        std::vector<std::shared_ptr<Material>> materials_;

        BufferGeometry* geometry() override {

            return geometry_.get();
        }

        [[nodiscard]] const BufferGeometry* geometry() const {

            return geometry_.get();
        }

        Material* material() override {

            return materials_.front().get();
        }

        [[nodiscard]] const Material* material() const override {

            return materials_.front().get();
        }

        template<class T>
        T* material() {

            return dynamic_cast<T*>(material());
        }

        template<class T>
        const T* material() const {

            return dynamic_cast<T*>(material());
        }

        [[nodiscard]] std::vector<Material*> materials() override {
            std::vector<Material*> res(materials_.size());
            std::transform(materials_.begin(), materials_.end(), res.begin(), [](auto& m) {return m.get();});
            return res;
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

        Mesh(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material)
            : geometry_(std::move(geometry)), materials_{std::move(material)} {
        }

        Mesh(std::shared_ptr<BufferGeometry> geometry, std::vector<std::shared_ptr<Material>> materials)
            : geometry_(std::move(geometry)), materials_{std::move(materials)} {
        }
    };

}// namespace threepp

#endif//THREEPP_MESH_HPP
