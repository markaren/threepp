#include "threepp/objects/GrassMesh.hpp"

namespace threepp {

    GrassMesh::GrassMesh(const std::shared_ptr<BufferGeometry>& geometry,
                         const std::shared_ptr<Material>& material)
        : Mesh(geometry, material) {}

    std::string GrassMesh::type() const {
        return "GrassMesh";
    }

}// namespace threepp
