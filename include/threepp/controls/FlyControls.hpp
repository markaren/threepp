// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/controls/FlyControls.js

#ifndef THREEPP_FLYCONTROLS_HPP
#define THREEPP_FLYCONTROLS_HPP

#include <memory>

namespace threepp {

    class Object3D;
    class Canvas;

    class FlyControls {

    public:
        float movementSpeed = 1.0;
        float rollSpeed = 0.005;

        bool dragToLook = false;
        bool autoForward = false;

        FlyControls(Object3D& object, Canvas& canvas);

        void update(float delta);

        ~FlyControls();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_FLYCONTROLS_HPP
