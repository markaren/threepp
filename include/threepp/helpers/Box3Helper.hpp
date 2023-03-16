// https://github.com/mrdoob/three.js/blob/dev/src/helpers/Box3Helper.js

#ifndef THREEPP_BOX3HELPER_HPP
#define THREEPP_BOX3HELPER_HPP

#include "threepp/objects/LineSegments.hpp"

namespace threepp {

    class Box3Helper: public LineSegments {

    public:
        void updateMatrixWorld(bool force) override;

        static std::shared_ptr<Box3Helper> create(const Box3& box, const Color& color = 0xffff00);

    protected:
        Box3Helper(const Box3& box, const Color& color);

    private:
        const Box3& box;
    };

}// namespace threepp

#endif//THREEPP_BOX3HELPER_HPP
