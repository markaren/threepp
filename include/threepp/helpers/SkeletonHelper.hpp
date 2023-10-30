
#ifndef THREEPP_SKELETONHELPER_HPP
#define THREEPP_SKELETONHELPER_HPP

#include "threepp/objects/Bone.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Skeleton.hpp"


namespace threepp {

    class SkeletonHelper: public LineSegments {

    public:
        explicit SkeletonHelper(Object3D& skeleton);

        [[nodiscard]] const std::vector<Bone*>& getBones() const;

        void updateMatrixWorld(bool force) override;

        static std::shared_ptr<SkeletonHelper> create(Object3D& skeleton) {

            return std::make_shared<SkeletonHelper>(skeleton);
        }

    private:
        Object3D& root;
        std::vector<Bone*> bones;
    };

}// namespace threepp

#endif//THREEPP_SKELETONHELPER_HPP
