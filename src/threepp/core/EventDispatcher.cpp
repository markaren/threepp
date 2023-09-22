
#include "threepp/core/EventDispatcher.hpp"

#include <algorithm>
#include <ranges>

using namespace threepp;


void EventDispatcher::addEventListener(const std::string& type, std::weak_ptr<EventListener> listener) {

    listeners_[type].emplace_back(std::move(listener));
}

bool EventDispatcher::hasEventListener(const std::string& type, const EventListener* listener) {

    if (!listeners_.contains(type)) return false;

    auto& listenerArray = listeners_.at(type);
    return std::ranges::find_if(listenerArray, [listener](const std::weak_ptr<EventListener>& l) {
               return !l.expired() && l.lock().get() == listener;
           }) != listenerArray.end();
}

void EventDispatcher::removeEventListener(const std::string& type, const EventListener* listener) {

    if (!listeners_.contains(type)) return;

    auto& listenerArray = listeners_.at(type);
    if (listenerArray.empty()) return;

    auto find = std::ranges::find_if(listenerArray, [listener](const std::weak_ptr<EventListener>& l) {
        return !l.expired() && l.lock().get() == listener;
    });
    if (find != listenerArray.end()) {
        listenerArray.erase(find);
    }
}

void EventDispatcher::dispatchEvent(const std::string& type, void* target) {

    if (listeners_.contains(type)) {

        Event e{type, target};

        auto listenersOfType = listeners_.at(type);//copy
        for (auto& l : listenersOfType) {
            if (!l.expired()) {
                l.lock()->onEvent(e);
            }
        }
    }
}
