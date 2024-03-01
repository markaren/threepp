
#ifndef THREEPP_MOUSELISTENER_HPP
#define THREEPP_MOUSELISTENER_HPP

#include "threepp/math/Vector2.hpp"

#include <functional>
#include <utility>

namespace threepp {

    struct MouseEvent : Event {
        MouseEvent(Vector2 pos) :
            pos(pos) {}
        Vector2 pos;
    };

    struct MouseMoveEvent : MouseEvent
    {
        MouseMoveEvent(Vector2 pos, Vector2 delta):
            MouseEvent(pos), delta(delta)
        {}

        Vector2 delta;
    };

    struct MouseButtonEvent : MouseEvent {
        MouseButtonEvent(int button, Vector2 pos)
            :MouseEvent(pos), button(button)
        {}
        int button;
    };

    struct MouseWheelEvent : Event {
        MouseWheelEvent(Vector2 offset_) :offset(offset_) {}
        Vector2 offset;
    };

    struct Mouse {
        TEventDispatcher<MouseMoveEvent> Move;
        TEventDispatcher<MouseWheelEvent> Wheel;
        TEventDispatcher<MouseButtonEvent> Down;
        TEventDispatcher<MouseButtonEvent> Up;
    };


}// namespace threepp


#endif//THREEPP_MOUSELISTENER_HPP
