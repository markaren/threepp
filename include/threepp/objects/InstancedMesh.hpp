// https://github.com/mrdoob/three.js/blob/r129/src/objects/InstancedMesh.js

#ifndef THREEPP_INSTANCEDMESH_HPP
#define THREEPP_INSTANCEDMESH_HPP

#include "Mesh.hpp"

#include <memory>

namespace threepp {

    class InstancedMesh: public Mesh {

    public:

        InstancedMesh(
                std::shared_ptr<BufferGeometry> geometry,
                std::shared_ptr<Material> material,
                size_t count);

        [[nodiscard]] size_t count() const;

        FloatBufferAttribute* instanceMatrix() const;

        FloatBufferAttribute* instanceColor() const;

        [[nodiscard]] std::string type() const override;

        void getColorAt(size_t index, Color& color) const;

        void getMatrixAt(size_t index, Matrix4& matrix) const;

        void setColorAt(size_t index, const Color& color);

        void setMatrixAt(size_t index, const Matrix4& matrix) const;

        void dispose();

        void raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) override;

        static std::shared_ptr<InstancedMesh> create(
                std::shared_ptr<BufferGeometry> geometry,
                std::shared_ptr<Material> material,
                size_t count);

        ~InstancedMesh() override;

    private:
        Mesh _mesh;
        bool disposed{false};

        size_t count_;
        std::unique_ptr<FloatBufferAttribute> instanceMatrix_;
        std::unique_ptr<FloatBufferAttribute> instanceColor_ = nullptr;

    };

}// namespace threepp

#endif//THREEPP_INSTANCEDMESH_HPP
