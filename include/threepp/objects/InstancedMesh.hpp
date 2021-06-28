// https://github.com/mrdoob/three.js/blob/r129/src/objects/InstancedMesh.js

#ifndef THREEPP_INSTANCEDMESH_HPP
#define THREEPP_INSTANCEDMESH_HPP

#include "Mesh.hpp"

#include <optional>

namespace threepp {

    class InstancedMesh : public Mesh {

    public:
        std::unique_ptr<FloatBufferAttribute> instanceMatrix;
        std::unique_ptr<FloatBufferAttribute> instanceColor = nullptr;

        int count;

        static std::shared_ptr<InstancedMesh> create(
                std::shared_ptr<BufferGeometry> geometry,
                std::shared_ptr<Material> material,
                int count) {

            return std::shared_ptr<InstancedMesh>(new InstancedMesh(std::move(geometry), std::move(material), count));
        }


    protected:
        InstancedMesh(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material, int count)
            : Mesh(std::move(geometry), std::move(material)), count(count), instanceMatrix(FloatBufferAttribute::create(std::vector<float>(count * 16), 16)) {

            this->frustumCulled = false;
        }
    };

}// namespace threepp

#endif//THREEPP_INSTANCEDMESH_HPP
