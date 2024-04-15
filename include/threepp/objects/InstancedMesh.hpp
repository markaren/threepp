// https://github.com/mrdoob/three.js/blob/r129/src/objects/InstancedMesh.js

#ifndef THREEPP_INSTANCEDMESH_HPP
#define THREEPP_INSTANCEDMESH_HPP

#include "Mesh.hpp"

#include <memory>

namespace threepp {

    class InstancedMesh: public Mesh {

    public:
        std::optional<Sphere> boundingSphere;
        std::optional<Box3> boundingBox;

        InstancedMesh(
                std::shared_ptr<BufferGeometry> geometry,
                std::shared_ptr<Material> material,
                size_t count);

        [[nodiscard]] size_t count() const;

        void setCount(size_t count);

        [[nodiscard]] FloatBufferAttribute* instanceMatrix() const;

        [[nodiscard]] FloatBufferAttribute* instanceColor() const;

        [[nodiscard]] std::string type() const override;

        void getColorAt(size_t index, Color& color) const;

        void getMatrixAt(size_t index, Matrix4& matrix) const;

        void setColorAt(size_t index, const Color& color);

        void setMatrixAt(size_t index, const Matrix4& matrix) const;

        void computeBoundingBox();

        void computeBoundingSphere();

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
        size_t maxCount_;

        std::unique_ptr<FloatBufferAttribute> instanceMatrix_;
        std::unique_ptr<FloatBufferAttribute> instanceColor_ = nullptr;

        Matrix4 _instanceLocalMatrix;
        Matrix4 _instanceWorldMatrix;

        Sphere _sphere;
        Box3 _box3;

        std::vector<Intersection> _instanceIntersects;
    };

}// namespace threepp

#endif//THREEPP_INSTANCEDMESH_HPP
