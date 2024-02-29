// https://github.com/mrdoob/three.js/blob/r129/src/core/EventDispatcher.js

#ifndef THREEPP_EVENTDISPATCHER_HPP
#define THREEPP_EVENTDISPATCHER_HPP

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>


namespace threepp {

    struct Event {

        const std::string type;
        void* target;
    };

    struct EventListener {

        virtual void onEvent(Event& event) = 0;

        virtual ~EventListener() = default;
    };

    struct FunctionalEventListener: EventListener {

        FunctionalEventListener(): callback_([](const Event&) {}) {}

        void setListener(std::function<void(Event)> listener) {
            callback_ = std::move(listener);
        }

        void onEvent(Event& event) override {
            return callback_(event);
        }

    private:
        std::function<void(Event)> callback_;
    };

    class EventDispatcher {

    public:
        void addEventListener(const std::string& type, EventListener& listener);

        bool hasEventListener(const std::string& type, const EventListener& listener);

        void removeEventListener(const std::string& type, const EventListener& listener);

        void dispatchEvent(const std::string& type, void* target = nullptr);

        virtual ~EventDispatcher() = default;

    private:
        std::unordered_map<std::string, std::vector<EventListener*>> listeners_;
    };

}// namespace threepp

#endif//THREEPP_EVENTDISPATCHER_HPP
