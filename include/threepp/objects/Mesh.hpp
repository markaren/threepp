// https://github.com/mrdoob/three.js/blob/r129/src/objects/Mesh.js

#ifndef THREEPP_MESH_HPP
#define THREEPP_MESH_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/objects/ObjectWithMaterials.hpp"
#include "threepp/objects/ObjectWithMorphTargetInfluences.hpp"


namespace threepp {

    // Class representing triangular polygon mesh based objects.
    class Mesh: public virtual Object3D, public ObjectWithMorphTargetInfluences, public ObjectWithMaterials {

    public:
        explicit Mesh(std::shared_ptr<BufferGeometry> geometry = nullptr, std::shared_ptr<Material> material = nullptr);
        Mesh(std::shared_ptr<BufferGeometry> geometry, std::vector<std::shared_ptr<Material>> materials);

        [[nodiscard]] const std::string& type() const override;

        [[nodiscard]] std::shared_ptr<BufferGeometry> geometry() const override;

        void setGeometry(const std::shared_ptr<BufferGeometry>& geometry);

        void raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) override;

        void copy(const Object3D& source, bool recursive = true) override;

        // Dithered LOD crossfade (Vulkan G-buffer; the GL renderer ignores it).
        // `fade` in [0, 1] is the share of this mesh's pixels that are drawn:
        // 1 (the default) all of them, 0 none. Which pixels comes from a
        // screen-space hash that changes every frame, so the temporal resolve
        // (TAA, DLSS, FSR) turns the pattern into a blend. Two levels of one
        // object crossfade with no holes and no overlap when the outgoing one
        // has fade f and the incoming one fade 1 - f with complement = true:
        // the complement draws exactly the pixels the plain side leaves out.
        // Shadows and rays see ONE of the pair, never both: the plain side
        // while its fade is at least 0.5, the complement side while its fade
        // is above 0.5. On an InstancedMesh this applies to every instance
        // until InstancedMesh::setFadeAt gives them their own.
        void setFade(float fade, bool complement = false);
        [[nodiscard]] float fade() const { return fade_; }
        [[nodiscard]] bool fadeComplement() const { return fadeComplement_; }

        static std::shared_ptr<Mesh> create(
                std::shared_ptr<BufferGeometry> geometry = nullptr,
                std::shared_ptr<Material> material = nullptr);

        static std::shared_ptr<Mesh> create(
                std::shared_ptr<BufferGeometry> geometry,
                std::vector<std::shared_ptr<Material>> materials);

        ~Mesh() override = default;

    protected:
        std::shared_ptr<BufferGeometry> geometry_;

        // The volumes raycast() uses to reject a ray before touching a single
        // triangle, in this object's LOCAL space.
        //
        // Virtual because a SkinnedMesh's GEOMETRY bounds are its BIND pose,
        // and glTF says a skinned mesh's node transform is ignored — the joints
        // place every vertex — so the geometry sphere pushed through
        // matrixWorld describes a volume nothing is drawn in. It overrides
        // both with bounds measured off the posed skeleton.
        //
        // Null means "no early-out available"; the caller then tests triangles.
        virtual const Sphere* raycastBoundingSphere();
        virtual const Box3* raycastBoundingBox();

        std::shared_ptr<Object3D> createDefault() override;

    private:
        float fade_ = 1.f;
        bool fadeComplement_ = false;
    };

}// namespace threepp

#endif//THREEPP_MESH_HPP
