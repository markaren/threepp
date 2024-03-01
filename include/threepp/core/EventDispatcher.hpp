// https://github.com/mrdoob/three.js/blob/r129/src/core/EventDispatcher.js

#ifndef THREEPP_EVENTDISPATCHER_HPP
#define THREEPP_EVENTDISPATCHER_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>
#include <algorithm>
#include <atomic>


namespace threepp {


    template<typename TEvent>
    using TEventListener = std::function<void(TEvent&)>;

    using Subscription = std::shared_ptr<void>;

    template <typename TEvent>
    class TEventDispatcher {
    public:
        using EventListener = TEventListener<TEvent>;

        /// Adds an event listener and returns a subscription
        [[nodiscard]] Subscription addEventListener(EventListener listener) {
            size_t current_id = id_.load();
            listeners_.insert({current_id, listener});
            Subscription disposer((void*) nullptr, [this, current_id](void*) { listeners_.erase(current_id); });
            id_ = id_ + 1;
            return disposer;
        }

        /// Adds an event listener and assumes event.unsubscribe = true will be
        /// used to unsubscribe. Useful for one shot events
        void addEventListenerOwned( EventListener listener) {
                size_t current_id = id_.load();
                listeners_.insert({current_id, listener});
                id_ = id_ + 1;
            }

        /// Adds an event listener  and unsubscribes after a single shot
		void addEventListenerOneShot( EventListener listener) {
			this->addEventListenerOwned([l = std::move(listener)](TEvent& e) { l(e); e.unsubscribe = true; });
		}

        void dispatchEvent(TEvent & e){
            std::vector<size_t> toUnsubscribe;
            for (auto const& item : listeners_) {
                item.second(e);
                if (e.unsubscribe) {
                    e.unsubscribe = false;
                    toUnsubscribe.push_back(item.first);
                }
            }
            for (size_t id : toUnsubscribe)
                listeners_.erase(id);
        }

        virtual ~TEventDispatcher() = default;

    private:
        std::unordered_map<size_t, EventListener> listeners_;
		std::atomic<size_t> id_ = 0;
    };


    struct Event {
        void* target;
        bool unsubscribe = false;
    };
    class EventDispatcher : public TEventDispatcher<Event> {

    };

}// namespace threepp

#endif//THREEPP_EVENTDISPATCHER_HPP
