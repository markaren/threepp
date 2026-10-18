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

        [[nodiscard]] const std::string& type() const override;

        void getColorAt(size_t index, Color& color) const;

        void getMatrixAt(size_t index, Matrix4& matrix) const;

        void setColorAt(size_t index, const Color& color);

        void setMatrixAt(size_t index, const Matrix4& matrix) const;

        // Per-instance LOD crossfade: Mesh::setFade for one instance. The first
        // call creates instanceFade() (two floats per instance: the fade and
        // the complement flag as 0 or 1) with every instance fully drawn; from
        // then on it replaces the mesh-wide Mesh::setFade for all instances.
        // Unlike setColorAt these mark the attribute for upload themselves.
        void setFadeAt(size_t index, float fade, bool complement = false);

        // Instances offset .. offset + n - 1 in one call. `complement` holds n
        // flags, or is null for all plain.
        void setFades(size_t offset, size_t n, const float* fades, const bool* complement = nullptr);

        // 1 / false for an instance (or a mesh) that was never given a fade.
        [[nodiscard]] float getFadeAt(size_t index) const;
        [[nodiscard]] bool getFadeComplementAt(size_t index) const;

        // Null until the first setFadeAt / setFades.
        [[nodiscard]] FloatBufferAttribute* instanceFade() const;

        void computeBoundingBox();

        void computeBoundingSphere();

        void dispose();

        void raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) override;

        void copy(const Object3D& source, bool recursive = true) override;

        static std::shared_ptr<InstancedMesh> create(
                std::shared_ptr<BufferGeometry> geometry,
                std::shared_ptr<Material> material,
                size_t count);

        ~InstancedMesh() override;

    protected:
        // Sized from this mesh's capacity, so copy() can fill the instance
        // buffers in place.
        std::shared_ptr<Object3D> createDefault() override;

    private:
        Mesh _mesh;

        size_t count_;
        size_t maxCount_;

        std::unique_ptr<FloatBufferAttribute> instanceMatrix_;
        std::unique_ptr<FloatBufferAttribute> instanceColor_ = nullptr;
        std::unique_ptr<FloatBufferAttribute> instanceFade_ = nullptr;

        void ensureInstanceFade();

        Matrix4 _instanceLocalMatrix;
        Matrix4 _instanceWorldMatrix;

        Sphere _sphere;
        Box3 _box3;

        std::vector<Intersection> _instanceIntersects;
    };

}// namespace threepp

#endif//THREEPP_INSTANCEDMESH_HPP
