
#ifndef THREEPP_MOUSELISTENER_HPP
#define THREEPP_MOUSELISTENER_HPP

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

}// namespace threepp


#endif//THREEPP_MOUSELISTENER_HPP
