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

        const std::string uuid = utils::generateUUID();

        virtual void onEvent(Event& event) = 0;

        virtual ~EventListener() = default;
    };

    class EventDispatcher {

    public:
        void addEventListener(const std::string& type, EventListener* listener) {

            listeners_[type].push_back(listener);
        }

        bool hasEventListener(const std::string& type, const EventListener* listener) {

            if (!listeners_.count(type)) return false;

            auto& listenerArray = listeners_.at(type);
            return std::find(listenerArray.begin(), listenerArray.end(), listener) != listenerArray.end();
        }

        void removeEventListener(const std::string& type, const EventListener* listener) {

            if (!listeners_.count(type)) return;

            auto& listenerArray = listeners_.at(type);
            if (listenerArray.empty()) return;

            auto find = std::find(listenerArray.begin(), listenerArray.end(), listener);
            if (find != listenerArray.end()) {
                listenerArray.erase(find);
            }
        }

        void dispatchEvent(const std::string& type, void* target = nullptr) {

            if (shutdown) return;

            if (listeners_.count(type)) {

                Event e{type, target};

                auto listenersOfType = listeners_.at(type);//copy
                for (auto& l : listenersOfType) {
                    l->onEvent(e);
                }
            }
        }

    private:
        inline static bool shutdown = false;
        std::unordered_map<std::string, std::vector<EventListener*>> listeners_;

        friend class GLRenderer;
    };

}// namespace threepp

#endif//THREEPP_EVENTDISPATCHER_HPP
