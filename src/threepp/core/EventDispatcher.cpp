
#include "threepp/core/EventDispatcher.hpp"

#include <algorithm>

using namespace threepp;


void EventDispatcher::addEventListener(const std::string& type, EventListener& listener) {

    listeners_[type].emplace_back(&listener);
}

bool EventDispatcher::hasEventListener(const std::string& type, const EventListener& listener) {

    if (!listeners_.contains(type)) return false;

    const auto& listenerArray = listeners_.at(type);
    return std::ranges::find(listenerArray, &listener) != listenerArray.end();
}

void EventDispatcher::removeEventListener(const std::string& type, const EventListener& listener) {

    if (!listeners_.contains(type)) return;

    auto& listenerArray = listeners_.at(type);
    if (listenerArray.empty()) return;

    if (const auto find = std::ranges::find(listenerArray, &listener); find != listenerArray.end()) {
        listenerArray.erase(find);
    }
}

void EventDispatcher::dispatchEvent(const std::string& type, void* target) {

    if (listeners_.contains(type)) {

        Event e{type, target};

        auto listenersOfType = listeners_.at(type);//copy
        for (auto l : listenersOfType) {
            if (l) {
                l->onEvent(e);
            }
        }
    }
}
