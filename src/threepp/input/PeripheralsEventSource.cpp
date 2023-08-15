
#include "threepp/input/PeripheralsEventSource.hpp"

#include <algorithm>

using namespace threepp;
void PeripheralsEventSource::addKeyListener(KeyListener* listener) {
    auto find = std::find(keyListeners.begin(), keyListeners.end(), listener);
    if (find == keyListeners.end()) {
        keyListeners.emplace_back(listener);
    }
}

bool PeripheralsEventSource::removeKeyListener(const KeyListener* listener) {
    auto find = std::find(keyListeners.begin(), keyListeners.end(), listener);
    if (find != keyListeners.end()) {
        keyListeners.erase(find);
        return true;
    }
    return false;
}

void PeripheralsEventSource::addMouseListener(MouseListener* listener) {
    auto find = std::find(mouseListeners.begin(), mouseListeners.end(), listener);
    if (find == mouseListeners.end()) {
        mouseListeners.emplace_back(listener);
    }
}

bool PeripheralsEventSource::removeMouseListener(const MouseListener* listener) {
    auto find = std::find(mouseListeners.begin(), mouseListeners.end(), listener);
    if (find != mouseListeners.end()) {
        mouseListeners.erase(find);
        return true;
    }
    return false;
}
