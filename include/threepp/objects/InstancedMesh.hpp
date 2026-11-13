// https://github.com/mrdoob/three.js/blob/r129/src/objects/InstancedMesh.js

#ifndef THREEPP_INSTANCEDMESH_HPP
#define THREEPP_INSTANCEDMESH_HPP

#include "Mesh.hpp"

#include <array>
#include <cstdint>
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

        // Colours for instances offset .. offset + n - 1 in one call: `rgb`
        // holds 3 floats per instance (the components Color::toArray writes).
        // Creates instanceColor() on the first call, as setColorAt does, and,
        // unlike setColorAt, marks it for upload (once for the run).
        void setColors(size_t offset, size_t n, const float* rgb);

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

        // Rows an edit wrote. A renderer that last saw an attribute at some
        // version asks which rows changed since (matrixRowsSince /
        // fadeRowsSince) and refreshes only those: a pool of a few hundred
        // thousand rows that changes a few each frame is otherwise walked
        // whole every frame. `placed`: the rows were given to another object
        // (a pool's slot re-used, a row parked), not moved; markMatrixRows.
        struct RowEdits {
            uint32_t lo = 0, hi = 0;          // every row written, [lo, hi)
            uint32_t movedLo = 0, movedHi = 0;// the rows written as moved (placed = false)
            [[nodiscard]] bool placedOnly(uint32_t row) const {
                return row >= lo && row < hi && !(row >= movedLo && row < movedHi);
            }
        };

        // After writing rows offset .. offset + n - 1 of instanceMatrix()
        // directly: marks them for upload (instanceMatrix()->needsUpdate())
        // and records which rows they were, so a renderer refreshes those
        // alone. placed = true: these rows now hold a different object than
        // before (a pool's slot handed on, or parked out of sight); a renderer
        // takes the new place as where the row always was: no motion vector,
        // not a moving object for its temporal histories. Default false: the
        // rows moved, exactly as a bare needsUpdate() says.
        void markMatrixRows(size_t offset, size_t n, bool placed = false);

        // Matrices for instances offset .. offset + n - 1 in one call (16
        // floats each in Matrix4::elements order), then markMatrixRows.
        void setMatrices(size_t offset, size_t n, const float* elements, bool placed = false);

        // The rows the instance matrices' (fades') edits since version `seen`
        // wrote, when every edit since was recorded (markMatrixRows,
        // setMatrices, setFadeAt, setFades); false when any was not (a bare
        // needsUpdate()), when more were made than are kept, or when `seen`
        // is not an earlier version: the caller then takes every row.
        [[nodiscard]] bool matrixRowsSince(unsigned int seen, RowEdits& out) const;
        [[nodiscard]] bool fadeRowsSince(unsigned int seen, RowEdits& out) const;

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

        // The last kEditLog edits of each attribute: the version each left
        // the attribute at, its rows, and whether they were placed.
        struct RowEdit {
            unsigned int version = 0;
            uint32_t lo = 0, hi = 0;
            bool placed = false;
        };
        static constexpr size_t kEditLog = 64;
        std::array<RowEdit, kEditLog> matrixEdits_{};
        size_t matrixEditCount_ = 0;
        std::array<RowEdit, kEditLog> fadeEdits_{};
        size_t fadeEditCount_ = 0;
        static bool rowsSince(const std::array<RowEdit, kEditLog>& log, size_t logged, unsigned int seen,
                              unsigned int now, RowEdits& out);

        Matrix4 _instanceLocalMatrix;
        Matrix4 _instanceWorldMatrix;

        Sphere _sphere;
        Box3 _box3;

        std::vector<Intersection> _instanceIntersects;
    };

}// namespace threepp

#endif//THREEPP_INSTANCEDMESH_HPP
