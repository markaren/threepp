
#ifndef THREEPP_MOUSELISTENER_HPP
#define THREEPP_MOUSELISTENER_HPP

#include "threepp/math/Vector2.hpp"

#include <functional>
#include <utility>

namespace threepp {

    struct MouseEvent {

        const int button;
        const int action;
        const int mods;

        MouseEvent(const int button, const int action, const int mods)
            : button(button), action(action), mods(mods) {}
    };

    struct MouseListener {

        virtual void onMouseDown(int button, Vector2 pos) {}
        virtual void onMouseUp(int button, Vector2 pos) {}
        virtual void onMouseMove(Vector2 pos) {}
        virtual void onMouseWheel(Vector2 delta) {}

        virtual ~MouseListener() = default;

    };

}// namespace threepp


#endif//THREEPP_MOUSELISTENER_HPP
