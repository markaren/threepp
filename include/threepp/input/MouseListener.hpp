
#ifndef THREEPP_MOUSELISTENER_HPP
#define THREEPP_MOUSELISTENER_HPP

#include "threepp/math/Vector2.hpp"

#include <functional>
#include <utility>

namespace threepp {

    struct MouseListener {

        virtual void onMouseDown(int /*button*/, const Vector2& /*pos*/) {}
        virtual void onMouseUp(int /*button*/, const Vector2& /*pos*/) {}
        virtual void onMouseMove(const Vector2& /*pos*/) {}
        virtual void onMouseWheel(const Vector2& /*delta*/) {}

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

    struct MouseUpListener: MouseListener {

        explicit MouseUpListener(std::function<void(int, const Vector2&)> f)
            : f_(std::move(f)) {}

        void onMouseUp(int button, const Vector2& pos) override {
            f_(button, pos);
        }

    private:
        std::function<void(int, Vector2)> f_;
    };

    struct MouseDownListener: MouseListener {

        explicit MouseDownListener(std::function<void(int, const Vector2&)> f)
            : f_(std::move(f)) {}

        void onMouseDown(int button, const Vector2& pos) override {
            f_(button, pos);
        }

    private:
        std::function<void(int, Vector2)> f_;
    };

}// namespace threepp


#endif//THREEPP_MOUSELISTENER_HPP
