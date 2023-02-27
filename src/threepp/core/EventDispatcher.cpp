
#include "threepp/core/EventDispatcher.hpp"


using namespace threepp;


void EventDispatcher::addEventListener(const std::string& type, std::weak_ptr<EventListener> listener) {

    listeners_[type].push_back(std::move(listener));
}

bool EventDispatcher::hasEventListener(const std::string& type, const EventListener* listener) {

    if (!listeners_.count(type)) return false;

    auto& listenerArray = listeners_.at(type);
    return std::find_if(listenerArray.begin(), listenerArray.end(), [listener](const std::weak_ptr<EventListener>& l) {
               return !l.expired() && l.lock().get() == listener;
           }) != listenerArray.end();
}

void EventDispatcher::removeEventListener(const std::string& type, const EventListener* listener) {

    if (!listeners_.count(type)) return;

    auto& listenerArray = listeners_.at(type);
    if (listenerArray.empty()) return;

    auto find = std::find_if(listenerArray.begin(), listenerArray.end(), [listener](const std::weak_ptr<EventListener>& l) {
        return !l.expired() && l.lock().get() == listener;
    });
    if (find != listenerArray.end()) {
        listenerArray.erase(find);
    }
}

void EventDispatcher::dispatchEvent(const std::string& type, void* target) {

    if (listeners_.count(type)) {

        Event e{type, target};

        auto listenersOfType = listeners_.at(type);//copy
        for (auto& l : listenersOfType) {
            if (!l.expired()) {
                l.lock()->onEvent(e);
            }
        }
    }
}
