#include "threepp/objects/GrassMesh.hpp"

namespace threepp {

    GrassMesh::GrassMesh(const std::shared_ptr<BufferGeometry>& geometry,
                         const std::shared_ptr<Material>& material)
        : Mesh(geometry, material) {}

    const std::string& GrassMesh::type() const {
        static const std::string typeName = "GrassMesh";
        return typeName;
    }

}// namespace threepp
