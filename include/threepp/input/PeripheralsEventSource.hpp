
#ifndef THREEPP_PERIPHERALSEVENTSOURCE_HPP
#define THREEPP_PERIPHERALSEVENTSOURCE_HPP

#include "threepp/input/IOCapture.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/input/MouseListener.hpp"

#include <vector>

namespace threepp {

    class PeripheralsEventSource {

    public:
        void setIOCapture(IOCapture* capture);

        void addKeyListener(KeyListener* listener);

        bool removeKeyListener(const KeyListener* listener);

        void addMouseListener(MouseListener* listener);

        bool removeMouseListener(const MouseListener* listener);

        virtual ~PeripheralsEventSource() = default;

    protected:
        enum class KeyAction {
            PRESS,
            RELEASE,
            REPEAT
        };

        enum class MouseAction {
            PRESS,
            RELEASE
        };

        void onMousePressedEvent(int button, const Vector2& pos, MouseAction action);

        void onMouseMoveEvent(const Vector2& pos);

        void onMouseWheelEvent(const Vector2& eventData);

        void onKeyEvent(KeyEvent evt, KeyAction action);

    private:
        IOCapture* ioCapture_ = nullptr;
        std::vector<KeyListener*> keyListeners_;
        std::vector<MouseListener*> mouseListeners_;
    };

}// namespace threepp

#endif//THREEPP_PERIPHERALSEVENTSOURCE_HPP
