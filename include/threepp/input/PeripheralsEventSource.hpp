
#ifndef THREEPP_PERIPHERALSEVENTSOURCE_HPP
#define THREEPP_PERIPHERALSEVENTSOURCE_HPP

#include "threepp/input/IOCapture.hpp"
#include "threepp/input/KeyListener.hpp"
#include "threepp/input/MouseListener.hpp"

#include "threepp/canvas/WindowSize.hpp"

#include <vector>

namespace threepp {

    class PeripheralsEventSource {

    public:
        [[nodiscard]] virtual WindowSize size() const = 0;

        void setIOCapture(IOCapture* capture);

        void addKeyListener(KeyListener& listener);

        bool removeKeyListener(const KeyListener& listener);

        void addMouseListener(MouseListener& listener);

        bool removeMouseListener(const MouseListener& listener);

        void onDrop(std::function<void(std::vector<std::string>)> paths);

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

        void onDropEvent(std::vector<std::string> paths) {
            if (dropListener_ && !paths.empty()) {
                dropListener_(std::move(paths));
            }
        }

    private:
        IOCapture* ioCapture_ = nullptr;
        std::vector<KeyListener*> keyListeners_;
        std::vector<MouseListener*> mouseListeners_;
        std::function<void(std::vector<std::string>)> dropListener_;
    };

}// namespace threepp

#endif//THREEPP_PERIPHERALSEVENTSOURCE_HPP
