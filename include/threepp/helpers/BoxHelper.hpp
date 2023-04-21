// https://github.com/mrdoob/three.js/blob/r151/src/helpers/Box3Helper.js

#ifndef THREEPP_BOXHELPER_HPP
#define THREEPP_BOXHELPER_HPP

#include "threepp/objects/LineSegments.hpp"

namespace threepp {

    class BoxHelper: public LineSegments {

    public:
        std::string type() const override;

        void update();

        BoxHelper& setFromObject(Object3D& object);

        static std::shared_ptr<BoxHelper> create(Object3D& object, const Color& color = 0xffff00);

    private:
        Object3D* object;

        BoxHelper(Object3D& object, const Color& color);
    };

}// namespace threepp

#endif//THREEPP_BOXHELPER_HPP
