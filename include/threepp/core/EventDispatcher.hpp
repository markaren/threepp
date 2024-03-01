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

    namespace concepts {
#if defined(__cpp_concepts) && (__cpp_concepts >= 201907L) 
        template<typename TEvent>
        concept Event = requires(TEvent e) {
            { e.target } -> std::convertible_to<void*>;
            { e.unsubscribe } -> std::convertible_to<bool>;
        };
#endif
    }

    /// An event listener is just a function that takes an argument of type TEvent
    template<typename TEvent>
    using TEventListener = std::function<void(TEvent&)>;

    /// A single subscription for an event
    using Subscription = std::shared_ptr<void>;

    /// For holding a large number of subscriptions to events
    using Subscriptions= std::vector<Subscription>;

    /// Allows one to use the << to push subscriptions onto the vector
    inline
    void operator<<(std::vector<Subscription>& subs, Subscription const & sub) {
        subs.push_back(sub);
    }

    /// Generic event dispatch class
    template <typename TEvent>
#if defined(__cpp_concepts) && (__cpp_concepts >= 201907L) 
    requires concepts::Event<TEvent>
#endif
    // C++20 (and later) code
    class TEventDispatcher {
    public:
        using EventListener = TEventListener<TEvent>;

        /// Adds an event listener and returns a subscription
        [[nodiscard]] Subscription subscribe(EventListener listener) {
            size_t current_id = id_.load();
            listeners_.insert({ current_id, listener });
            Subscription disposer((void*) nullptr, [this, current_id](void*) { listeners_.erase(current_id); });
            id_ = id_ + 1;
            return disposer;
        }

        /// Adds an event listener and never automatically unsubscribes. You
        /// can set event.unsubscribe = true and the subscription will be
        /// cancelled. Not recommended to be used directly. Build other
        /// tools on this.
        void subscribeForever(EventListener listener) {
            size_t current_id = id_.load();
            listeners_.insert({ current_id, listener });
            id_ = id_ + 1;
        }

        /// Adds an event listener that unsubscribes after a single shot
        void subscribeOnce(EventListener listener) {
            this->subscribeForever([l = std::move(listener)](TEvent& e) { l(e); e.unsubscribe = true; });
        }

        /// Hold onto the other subscription until the second source fires then
        /// dispose the subscription.
        template <typename T>
        void subscribeUntil(TEventDispatcher<T>& s, EventListener listener) {
            auto sub = subscribe(listener);
            s.subscribeOnce([sub](auto&) {});
        }


        /// Send an event to all listeners.
        void send(TEvent & e){
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

        /// Handle r-value versions of send
        void send(TEvent && e) {
            send(e);
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
