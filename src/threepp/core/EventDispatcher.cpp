
#include "threepp/core/EventDispatcher.hpp"

#include <algorithm>

using namespace threepp;


void EventDispatcher::addEventListener(const std::string& type, EventListener* listener) {

    listeners_[type].emplace_back(listener);
}

bool EventDispatcher::hasEventListener(const std::string& type, const EventListener* listener) {

    if (!listeners_.count(type)) return false;

    auto& listenerArray = listeners_.at(type);
    return std::find(listenerArray.begin(), listenerArray.end(), listener) != listenerArray.end();
}

void EventDispatcher::removeEventListener(const std::string& type, const EventListener* listener) {

    if (!listeners_.count(type)) return;

    auto& listenerArray = listeners_.at(type);
    if (listenerArray.empty()) return;

    auto find = std::find(listenerArray.begin(), listenerArray.end(), listener);
    if (find != listenerArray.end()) {
        listenerArray.erase(find);
    }
}

void EventDispatcher::dispatchEvent(const std::string& type, void* target) {

    if (listeners_.count(type)) {

        Event e{type, target};

        auto listenersOfType = listeners_.at(type);//copy
        for (auto l : listenersOfType) {
            if (l) {
                l->onEvent(e);
            }
        }
    }
}
