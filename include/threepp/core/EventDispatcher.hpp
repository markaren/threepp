// https://github.com/mrdoob/three.js/blob/r129/src/core/EventDispatcher.js

#ifndef THREEPP_EVENTDISPATCHER_HPP
#define THREEPP_EVENTDISPATCHER_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>


namespace threepp {

    struct Event {
        const std::string type;
        void* target;
        bool unsubscribe = false;
    };

    using EventListener = std::function<void(Event&)>;
    using Subscription = std::shared_ptr<void>;

    class EventDispatcher {

    public:
        /// Adds an event listener and returns a subscription
        [[nodiscard]]
        Subscription addEventListener(const std::string& type, EventListener listener);

        /// Adds an event listener and assumes event.unsubscribe = true will be
        /// used to unsubscribe. Useful for one shot events
        void addEventListenerOwned(const std::string& type, EventListener listener);

        /// Adds an event listener  and unsubscribes after a single shot
		void addEventListenerOneShot(const std::string& type, EventListener listener);

        void dispatchEvent(const std::string& type, void* target = nullptr);


        virtual ~EventDispatcher() = default;

    private:
        std::unordered_map<std::string, std::unordered_map<size_t, EventListener>> listeners_;
    };

}// namespace threepp

#endif//THREEPP_EVENTDISPATCHER_HPP
