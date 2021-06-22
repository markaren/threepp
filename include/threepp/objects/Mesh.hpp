// https://github.com/mrdoob/three.js/blob/r129/src/objects/Mesh.js

#ifndef THREEPP_MESH_HPP
#define THREEPP_MESH_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"

#include <memory>

namespace threepp {

    class Mesh : public Object3D {

    public:

        Mesh(const Mesh&) = delete;

        BufferGeometry *geometry() override {
            return geometry_.get();
        }

        Material *material() {
            return material_.get();
        }

        std::string type() const override {
            return "Mesh";
        }

        static std::shared_ptr<Mesh> create(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material) {
            return std::shared_ptr<Mesh>(new Mesh(std::move(geometry), std::move(material)));
        }

        ~Mesh() = default;

    protected:
        Mesh(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material)
            : geometry_(std::move(geometry)), material_(std::move(material)) {

        }

    private:
        std::shared_ptr<BufferGeometry> geometry_;
        std::shared_ptr<Material> material_;
    };

}// namespace threepp

#endif//THREEPP_MESH_HPP
