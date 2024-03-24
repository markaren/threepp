// https://github.com/mrdoob/three.js/blob/r162/examples/jsm/controls/DragControls.js

#ifndef THREEPP_DRAGCONTROLS_HPP
#define THREEPP_DRAGCONTROLS_HPP

#include "threepp/core/EventDispatcher.hpp"

#include <memory>
#include <vector>

namespace threepp {

    class Camera;
    class Object3D;
    class PeripheralsEventSource;

    class DragControls {

    public:
        enum class Mode {
            Translate,
            Rotate
        };

        bool enabled = true;
        bool recursive = true;
        bool transformGroup = false;

        float rotateSpeed = 1;

        Mode mode{Mode::Translate};

        DragControls(const std::vector<Object3D*>& objects, Camera& camera, PeripheralsEventSource& eventSource);

        void setObjects(const std::vector<Object3D*>& objects);

        ~DragControls();

        EventDispatcher HoverOn;
        EventDispatcher HoverOff;
        EventDispatcher Drag;
        EventDispatcher Dragstart;
        EventDispatcher Dragend;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_DRAGCONTROLS_HPP
