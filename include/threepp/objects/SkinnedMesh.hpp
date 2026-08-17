
#ifndef THREEPP_SKINNEDMESH_HPP
#define THREEPP_SKINNEDMESH_HPP

#include "threepp/objects/Mesh.hpp"

#include "threepp/objects/Skeleton.hpp"

namespace threepp {

    class SkinnedMesh: public Mesh {

    public:
        enum class BindMode {
            Attached,
            Detached
        };

        BindMode bindMode{BindMode::Attached};
        Matrix4 bindMatrix;
        Matrix4 bindMatrixInverse;

        std::shared_ptr<Skeleton> skeleton = nullptr;

        SkinnedMesh(const std::shared_ptr<BufferGeometry>& geometry, const std::shared_ptr<Material>& material);

        [[nodiscard]] std::string type() const override;

        void bind(const std::shared_ptr<Skeleton>& skeleton, std::optional<Matrix4> bindMatrix = {});

        void pose() const;

        void normalizeSkinWeights();

        void updateMatrixWorld(bool force) override;

        void boneTransform(size_t index, Vector3& target);

        // This mesh's bounds AS POSED, in its own local space — the frame
        // boneTransform answers in, and the one a raycaster's local ray lives
        // in. Recomputed from the skeleton on demand and cached until the next
        // updateMatrixWorld, so a moving character pays for it at most once per
        // frame and only if something actually asks.
        //
        // LOCAL space, and on a glTF rig that is not metres: the node transform
        // the skin cancels is still on matrixWorld, so a 1.8 m character under
        // a 0.01 armature measures 180 here. Push the result through
        // matrixWorld for world bounds — which is what raycast() does.
        //
        // Not the same thing as geometry()->boundingBox, and that is the point:
        // the geometry describes the BIND pose, which for any glTF rig is not
        // where the vertices are drawn (see raycastBoundingSphere).
        const Box3& posedBoundingBox();
        const Sphere& posedBoundingSphere();

    protected:
        const Sphere* raycastBoundingSphere() override;
        const Box3* raycastBoundingBox() override;

    private:
        // Marked stale by updateMatrixWorld; filled by the two accessors above.
        bool posedBoundsDirty_ = true;
        Box3 posedBox_;
        Sphere posedSphere_;

        void computePosedBounds();

    public:
        static std::shared_ptr<SkinnedMesh> create(const std::shared_ptr<BufferGeometry>& geometry, const std::shared_ptr<Material>& material) {

            return std::make_shared<SkinnedMesh>(geometry, material);
        }
    };

}// namespace threepp

#endif//THREEPP_SKINNEDMESH_HPP
