// https://github.com/mrdoob/three.js/blob/r129/src/objects/Mesh.js

#ifndef THREEPP_MESH_HPP
#define THREEPP_MESH_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class Mesh : public Object3D {

    public:
        Mesh(const std::shared_ptr<BufferGeometry> &geometry, const std::shared_ptr > Material > &material)
            : geometry_(std::move(geometry)), material_(std::move(material_)) {}

        static std::shared_ptr<Mesh> create(const std::shared_ptr<BufferGeometry> &geometry, const std::shared_ptr > Material > &material) {
            return std::make_shared<Mesh>(geometry, material);
        }

    private:
        std::shared_ptr<BufferGeometry> geometry_;
        std::shared_ptr<Material> material_;
    };

}// namespace threepp

#endif//THREEPP_MESH_HPP
