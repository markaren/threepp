
#include "threepp/input/PeripheralsEventSource.hpp"

#include <algorithm>

using namespace threepp;

void PeripheralsEventSource::setIOCapture(IOCapture* capture) {
    ioCapture_ = capture;
}

void PeripheralsEventSource::addKeyListener(KeyListener& listener) {
    auto find = std::find(keyListeners_.begin(), keyListeners_.end(), &listener);
    if (find == keyListeners_.end()) {
        keyListeners_.emplace_back(&listener);
    }
}

bool PeripheralsEventSource::removeKeyListener(const KeyListener& listener) {
    auto find = std::find(keyListeners_.begin(), keyListeners_.end(), &listener);
    if (find != keyListeners_.end()) {
        keyListeners_.erase(find);
        return true;
    }
    return false;
}

void PeripheralsEventSource::addMouseListener(MouseListener& listener) {
    auto find = std::find(mouseListeners_.begin(), mouseListeners_.end(), &listener);
    if (find == mouseListeners_.end()) {
        mouseListeners_.emplace_back(&listener);
    }
}

bool PeripheralsEventSource::removeMouseListener(const MouseListener& listener) {
    auto find = std::find(mouseListeners_.begin(), mouseListeners_.end(), &listener);
    if (find != mouseListeners_.end()) {
        mouseListeners_.erase(find);
        return true;
    }
    return false;
}

void PeripheralsEventSource::onMousePressedEvent(int button, const Vector2& pos, PeripheralsEventSource::MouseAction action) {
    if (ioCapture_ && ioCapture_->preventMouseEvent()) return;

    auto listeners = mouseListeners_;//copy - IMPORTANT
    for (auto l : listeners) {
        switch (action) {
            case MouseAction::RELEASE: {
                l->onMouseUp(button, pos);
                break;
            }
            case MouseAction::PRESS: {
                l->onMouseDown(button, pos);
                break;
            }
        }
    }
}

void PeripheralsEventSource::onMouseMoveEvent(const Vector2& pos) {
    if (ioCapture_ && ioCapture_->preventMouseEvent()) return;

    auto listeners = mouseListeners_;//copy - IMPORTANT
    for (auto l : listeners) {
        l->onMouseMove(pos);
    }
}

void PeripheralsEventSource::onMouseWheelEvent(const Vector2& eventData) {
    if (ioCapture_ && ioCapture_->preventScrollEvent()) return;

    auto listeners = mouseListeners_;//copy - IMPORTANT
    for (auto l : listeners) {

        l->onMouseWheel(eventData);
    }
}

void PeripheralsEventSource::onKeyEvent(KeyEvent evt, PeripheralsEventSource::KeyAction action) {
    if (ioCapture_ && ioCapture_->preventKeyboardEvent()) return;

    auto listeners = keyListeners_;//copy - IMPORTANT
    for (auto l : listeners) {
        switch (action) {
            case KeyAction::PRESS: {
                l->onKeyPressed(evt);
                break;
            }
            case KeyAction::RELEASE: {
                l->onKeyReleased(evt);
                break;
            }
            case KeyAction::REPEAT: {
                l->onKeyRepeat(evt);
                break;
            }
        }
    }
}

void PeripheralsEventSource::onDrop(std::function<void(std::vector<std::string>)> paths) {
    dropListener_ = std::move(paths);
}
