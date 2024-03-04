// https://github.com/mrdoob/three.js/blob/r129/src/core/EventDispatcher.js

#ifndef THREEPP_EVENTDISPATCHER_HPP
#define THREEPP_EVENTDISPATCHER_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>
#include <algorithm>
#include <threepp/utils/Scope.hpp>


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
    using Subscription = threepp::utils::ScopeExit;

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
    // C++20 (and later) code
    requires concepts::Event<TEvent>
#endif
    class TEventDispatcher {
    private:
        void unsubscribe(size_t id) {
            if (!sending_)
                listeners_.erase(id);
            else
                to_unsubscribe_.push_back(id);
        }
    public:
        using EventListener = TEventListener<TEvent>;

        /// Adds an event listener and returns a subscription
        [[nodiscard]] Subscription subscribe(EventListener listener) {
            size_t current_id = id_;
            id_++;
            listeners_.insert({current_id, listener});
            return utils::at_scope_exit([this, current_id]() { unsubscribe(current_id); });
        }

        /// Adds an event listener and never automatically unsubscribes. You
        /// can set event.unsubscribe = true and the subscription will be
        /// cancelled. Not recommended to be used directly. Build other
        /// tools on this.
        void subscribeForever(EventListener listener) {
            size_t current_id = id_;
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
            /// Mark that we are in the sending state.
            auto tmp = utils::reset_at_scope_exit(sending_, true);

            for(auto it = listeners_.begin(); it != listeners_.end();)
            {
                it->second(e);
                if (e.unsubscribe) {
                    e.unsubscribe = false;
                    it = listeners_.erase(it);
                }
                else
                    it++;
            }

            // Unsubscribe listeners that disposed of their
            // subscription by calling unsubscribe directly
            // during the event sending phase above. 
            for(size_t id : to_unsubscribe_)
                listeners_.erase(id);

            if (to_unsubscribe_.size() > 0)
                to_unsubscribe_.clear();
        }

        /// Handle r-value versions of send
        void send(TEvent && e) {
            send(e);
        }

        virtual ~TEventDispatcher() = default;

    private:
        std::unordered_map<size_t, EventListener> listeners_;
        /// The id of the current listener
		size_t id_ = 0;
        /// Are we sending event? Used to detect unsubscriptions during sending.
        bool sending_ = false;
        /// Subscriptions to be delay unsubscribed.
        std::vector<size_t> to_unsubscribe_;
    };


    struct Event {
        void* target;
        bool unsubscribe = false;
    };
    class EventDispatcher : public TEventDispatcher<Event> {

    };

}// namespace threepp

#endif//THREEPP_EVENTDISPATCHER_HPP
