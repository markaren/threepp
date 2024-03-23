
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

        bool preventMouseEvent() const {
            return ioCapture_ && ioCapture_->preventMouseEvent();
        }

        bool preventKeyboardEvent() const {
            return ioCapture_ && ioCapture_->preventKeyboardEvent();
        }

        bool preventScrollEvent() const {
            return ioCapture_ && ioCapture_->preventScrollEvent();
        }

        // Events for keys
        Keys keys;

        // Events for mouse
        Mouse mouse;

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

        void onDropEvent(std::vector<std::string> paths) {
            if (dropListener_ && !paths.empty()) {
                dropListener_(std::move(paths));
            }
        }

    private:
        IOCapture* ioCapture_ = nullptr;
        std::function<void(std::vector<std::string>)> dropListener_;
    };

}// namespace threepp

#endif//THREEPP_PERIPHERALSEVENTSOURCE_HPP
