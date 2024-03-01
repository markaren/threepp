
#include "threepp/core/EventDispatcher.hpp"

#include <algorithm>
#include <atomic>

using namespace threepp;


static std::atomic<size_t> id = 0;

threepp::Subscription EventDispatcher::addEventListener(const std::string& type, EventListener listener) {
    size_t current_id = id.load();
    listeners_[type].insert({current_id, listener});
    Subscription disposer((void*)nullptr, [this, type, current_id](void*) {listeners_[type].erase(current_id); });
    id=id+1;
    return disposer;
}

void EventDispatcher::addEventListenerOwned(const std::string& type, EventListener listener) {
    size_t current_id = id.load();
    listeners_[type].insert({current_id, listener});
    id=id+1;
}

void EventDispatcher::addEventListenerOneShot(const std::string& type, EventListener listener) {
    this->addEventListenerOwned(type, [l=std::move(listener)](Event& e) { l(e); e.unsubscribe = true; });
}

void EventDispatcher::dispatchEvent(const std::string& type, void* target) {

    if (listeners_.count(type)) {
        Event e{type, target};
        std::unordered_map<size_t, EventListener> & listenersOfType = listeners_.at(type);//copy
        std::vector<size_t> toUnsubscribe;
        for (auto const & item : listenersOfType) {
            item.second(e);
            if (e.unsubscribe) {
                e.unsubscribe = false;
                toUnsubscribe.push_back(item.first);
            }
        }
        for (size_t id : toUnsubscribe)
            listenersOfType.erase(id);
    }
}
