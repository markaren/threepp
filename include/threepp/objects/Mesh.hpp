// https://github.com/mrdoob/three.js/blob/r129/src/objects/Mesh.js

#ifndef THREEPP_MESH_HPP
#define THREEPP_MESH_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"


namespace threepp {

    class Mesh: public Object3D {

    public:
        [[nodiscard]] std::string type() const override;

        BufferGeometry* geometry() override;

        [[nodiscard]] const BufferGeometry* geometry() const;

        void setGeometry(const std::shared_ptr<BufferGeometry>& geometry);

        Material* material() override;

        [[nodiscard]] std::vector<Material*> materials() override;

        void setMaterial(const std::shared_ptr<Material>& material);

        void setMaterials(const std::vector<std::shared_ptr<Material>>& materials);

        [[nodiscard]] size_t numMaterials() const;

        void raycast(Raycaster& raycaster, std::vector<Intersection>& intersects) override;

        std::shared_ptr<Object3D> clone(bool recursive = true) override;

        static std::shared_ptr<Mesh> create(
                std::shared_ptr<BufferGeometry> geometry = nullptr,
                std::shared_ptr<Material> material = nullptr);

        static std::shared_ptr<Mesh> create(
                std::shared_ptr<BufferGeometry> geometry,
                std::vector<std::shared_ptr<Material>> materials);

        ~Mesh() override = default;

    protected:
        std::shared_ptr<BufferGeometry> geometry_;
        std::vector<std::shared_ptr<Material>> materials_;

        Mesh(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material);
        Mesh(std::shared_ptr<BufferGeometry> geometry, std::vector<std::shared_ptr<Material>> materials);
    };

}// namespace threepp

#endif//THREEPP_MESH_HPP
