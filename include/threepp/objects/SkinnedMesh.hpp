
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
        Matrix4* bindMatrix = nullptr;
        Matrix4* bindMatrixInverse = nullptr;

        std::shared_ptr<Skeleton> skeleton = nullptr;

        SkinnedMesh(const std::shared_ptr<BufferGeometry>& geometry, const std::shared_ptr<Material>& material);

        [[nodiscard]] std::string type() const override;

        void bind(const std::shared_ptr<Skeleton>& skeleton, Matrix4* bindMatrix = nullptr);

        void pose() const;

        void normalizeSkinWeights();

        void updateMatrixWorld(bool force) override;

        void boneTransform(size_t index, Vector3& target);

        static std::shared_ptr<SkinnedMesh> create(const std::shared_ptr<BufferGeometry>& geometry, const std::shared_ptr<Material>& material) {

            return std::make_shared<SkinnedMesh>(geometry, material);
        }

    private:
        Matrix4 _defaultBindMatrix;
        Matrix4 _defaultBindMatrixInverse;
    };

}// namespace threepp

#endif//THREEPP_SKINNEDMESH_HPP
