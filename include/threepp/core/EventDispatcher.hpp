// https://github.com/mrdoob/three.js/blob/r129/src/core/EventDispatcher.js

#ifndef THREEPP_EVENTDISPATCHER_HPP
#define THREEPP_EVENTDISPATCHER_HPP

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "threepp/utils/uuid.hpp"

namespace threepp {

    struct Event {

        const std::string type;
        void* target;
    };

    struct EventListener {

        virtual void onEvent(Event& event) = 0;

        virtual ~EventListener() = default;
    };

    class EventDispatcher {

    public:
        void addEventListener(const std::string& type, EventListener* listener);

        bool hasEventListener(const std::string& type, const EventListener* listener);

        void removeEventListener(const std::string& type, const EventListener* listener);

        void dispatchEvent(const std::string& type, void* target = nullptr);

        virtual ~EventDispatcher() = default;

    private:
        inline static bool shutdown = false;
        std::unordered_map<std::string, std::vector<EventListener*>> listeners_;

        friend class GLRenderer;
    };

}// namespace threepp

#endif//THREEPP_EVENTDISPATCHER_HPP
