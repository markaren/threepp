
#include "threepp/input/PeripheralsEventSource.hpp"

#include <algorithm>

using namespace threepp;

void PeripheralsEventSource::setIOCapture(IOCapture* capture) {
    ioCapture_ = capture;
}

void PeripheralsEventSource::addKeyListener(KeyListener& listener) {
    if (const auto find = std::ranges::find(keyListeners_, &listener); find == keyListeners_.end()) {
        keyListeners_.emplace_back(&listener);
    }
}

bool PeripheralsEventSource::removeKeyListener(const KeyListener& listener) {
    if (const auto find = std::ranges::find(keyListeners_, &listener); find != keyListeners_.end()) {
        keyListeners_.erase(find);
        return true;
    }
    return false;
}

int PeripheralsEventSource::onKeyPressed(std::function<void(KeyEvent)> callback) {
    const int id = nextListenerId_++;
    ownedKeyListeners_.push_back({id, KeyAction::PRESS, std::move(callback)});
    return id;
}

int PeripheralsEventSource::onKeyReleased(std::function<void(KeyEvent)> callback) {
    const int id = nextListenerId_++;
    ownedKeyListeners_.push_back({id, KeyAction::RELEASE, std::move(callback)});
    return id;
}

int PeripheralsEventSource::onKeyRepeat(std::function<void(KeyEvent)> callback) {
    const int id = nextListenerId_++;
    ownedKeyListeners_.push_back({id, KeyAction::REPEAT, std::move(callback)});
    return id;
}

void PeripheralsEventSource::removeKeyListener(int id) {
    std::erase_if(ownedKeyListeners_, [id](const OwnedKeyListener& l) { return l.id == id; });
}

bool PeripheralsEventSource::isKeyDown(Key key) const {
    return keysDown_.contains(key);
}

void PeripheralsEventSource::addMouseListener(MouseListener& listener) {
    if (const auto find = std::ranges::find(mouseListeners_, &listener); find == mouseListeners_.end()) {
        mouseListeners_.emplace_back(&listener);
    }
}

bool PeripheralsEventSource::removeMouseListener(const MouseListener& listener) {
    if (const auto find = std::ranges::find(mouseListeners_, &listener); find != mouseListeners_.end()) {
        mouseListeners_.erase(find);
        return true;
    }
    return false;
}

int PeripheralsEventSource::addOwnedMouseListener(OwnedMouseListener listener) {
    listener.id = nextListenerId_++;
    ownedMouseListeners_.push_back(std::move(listener));
    return ownedMouseListeners_.back().id;
}

int PeripheralsEventSource::onMouseDown(std::function<void(int, Vector2)> callback) {
    return addOwnedMouseListener({0, MouseCallbackKind::DOWN, std::move(callback), {}});
}

int PeripheralsEventSource::onMouseUp(std::function<void(int, Vector2)> callback) {
    return addOwnedMouseListener({0, MouseCallbackKind::UP, std::move(callback), {}});
}

int PeripheralsEventSource::onMouseMove(std::function<void(Vector2)> callback) {
    return addOwnedMouseListener({0, MouseCallbackKind::MOVE, {}, std::move(callback)});
}

int PeripheralsEventSource::onMouseWheel(std::function<void(Vector2)> callback) {
    return addOwnedMouseListener({0, MouseCallbackKind::WHEEL, {}, std::move(callback)});
}

void PeripheralsEventSource::removeMouseListener(int id) {
    std::erase_if(ownedMouseListeners_, [id](const OwnedMouseListener& l) { return l.id == id; });
}

Vector2 PeripheralsEventSource::mousePosition() const {
    return mousePosition_;
}

void PeripheralsEventSource::onMousePressedEvent(int button, const Vector2& pos, MouseAction action) {
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

    // Owning callbacks. Copy first so a callback may add/remove listeners during
    // dispatch without invalidating iterators.
    const auto kind = action == MouseAction::PRESS ? MouseCallbackKind::DOWN : MouseCallbackKind::UP;
    auto owned = ownedMouseListeners_;
    for (const auto& l : owned) {
        if (l.kind == kind) l.buttonCb(button, pos);
    }
}

void PeripheralsEventSource::onMouseMoveEvent(const Vector2& pos) {
    // Track the cursor regardless of IOCapture so mousePosition() stays accurate
    // while UI has mouse focus (same reasoning as keysDown_ in onKeyEvent).
    mousePosition_ = pos;

    if (ioCapture_ && ioCapture_->preventMouseEvent()) return;

    auto listeners = mouseListeners_;//copy - IMPORTANT
    for (auto l : listeners) {
        l->onMouseMove(pos);
    }

    // Owning callbacks; copy first, see onMousePressedEvent.
    auto owned = ownedMouseListeners_;
    for (const auto& l : owned) {
        if (l.kind == MouseCallbackKind::MOVE) l.vecCb(pos);
    }
}

void PeripheralsEventSource::onMouseWheelEvent(const Vector2& eventData) {
    if (ioCapture_ && ioCapture_->preventScrollEvent()) return;

    auto listeners = mouseListeners_;//copy - IMPORTANT
    for (auto l : listeners) {

        l->onMouseWheel(eventData);
    }

    // Owning callbacks; copy first, see onMousePressedEvent.
    auto owned = ownedMouseListeners_;
    for (const auto& l : owned) {
        if (l.kind == MouseCallbackKind::WHEEL) l.vecCb(eventData);
    }
}

void PeripheralsEventSource::onKeyEvent(KeyEvent evt, PeripheralsEventSource::KeyAction action) {
    // Track held state regardless of IOCapture so isKeyDown() stays physically accurate
    // and never sticks (e.g. a RELEASE arriving while UI has keyboard focus).
    if (action == KeyAction::PRESS) keysDown_.insert(evt.key);
    else if (action == KeyAction::RELEASE) keysDown_.erase(evt.key);

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

    // Owning callbacks (registered via onKeyPressed/Released/Repeat). Copy first so a
    // callback may add/remove listeners during dispatch without invalidating iterators.
    auto owned = ownedKeyListeners_;
    for (const auto& l : owned) {
        if (l.action == action) l.cb(evt);
    }
}

void PeripheralsEventSource::onDrop(std::function<void(std::vector<std::string>)> paths) {
    dropListener_ = std::move(paths);
}

void PeripheralsEventSource::onDropEvent(std::vector<std::string> paths) {
    if (dropListener_ && !paths.empty()) {
        dropListener_(std::move(paths));
    }
}
