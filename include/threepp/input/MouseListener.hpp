
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

        virtual void onMouseDown(int button, const Vector2& pos) {}
        virtual void onMouseUp(int button, const Vector2& pos) {}
        virtual void onMouseMove(const Vector2& pos) {}
        virtual void onMouseWheel(const Vector2& delta) {}

        virtual ~MouseListener() = default;
    };

    struct MouseMoveListener: MouseListener {

        explicit MouseMoveListener(std::function<void(const Vector2&)> f)
            : f_(std::move(f)) {}

        void onMouseMove(const Vector2& pos) override {
            f_(pos);
        }

    private:
        std::function<void(Vector2)> f_;
    };

}// namespace threepp


#endif//THREEPP_MOUSELISTENER_HPP
