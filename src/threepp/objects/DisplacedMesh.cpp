#include "threepp/objects/DisplacedMesh.hpp"

namespace threepp {

    DisplacedMesh::DisplacedMesh(const std::shared_ptr<BufferGeometry>& geometry,
                                 const std::shared_ptr<Material>& material)
        : Mesh(geometry, material) {}

    std::string DisplacedMesh::type() const {
        return "DisplacedMesh";
    }

}// namespace threepp
