// https://github.com/mrdoob/three.js/blob/r129/src/objects/Mesh.js

#ifndef THREEPP_MESH_HPP
#define THREEPP_MESH_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"
#include "threepp/objects/ObjectWithMorphTargetInfluences.hpp"


namespace threepp {

    // Class representing triangular polygon mesh based objects.
    class Mesh: public virtual Object3D, public ObjectWithMorphTargetInfluences, public ObjectWithMaterials {

    public:
        explicit Mesh(std::shared_ptr<BufferGeometry> geometry = nullptr, std::shared_ptr<Material> material = nullptr);
        Mesh(std::shared_ptr<BufferGeometry> geometry, std::vector<std::shared_ptr<Material>> materials);

        [[nodiscard]] std::string type() const override;

        [[nodiscard]] std::shared_ptr<BufferGeometry> geometry() const override;

        void setGeometry(const std::shared_ptr<BufferGeometry>& geometry);

        void raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) override;

        void copy(const Object3D& source, bool recursive = true) override;

        static std::shared_ptr<Mesh> create(
                std::shared_ptr<BufferGeometry> geometry = nullptr,
                std::shared_ptr<Material> material = nullptr);

        static std::shared_ptr<Mesh> create(
                std::shared_ptr<BufferGeometry> geometry,
                std::vector<std::shared_ptr<Material>> materials);

        ~Mesh() override = default;

    protected:
        std::shared_ptr<BufferGeometry> geometry_;

        std::shared_ptr<Object3D> createDefault() override;
    };

}// namespace threepp

#endif//THREEPP_MESH_HPP
