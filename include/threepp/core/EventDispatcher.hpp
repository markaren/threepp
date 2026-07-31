// https://github.com/mrdoob/three.js/blob/r129/src/core/EventDispatcher.js

#ifndef THREEPP_EVENTDISPATCHER_HPP
#define THREEPP_EVENTDISPATCHER_HPP

#include <any>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>


namespace threepp {

    struct Event {

        const std::string type;
        std::any target;
    };

    struct EventListener {

        virtual void onEvent(Event& event) = 0;

        virtual ~EventListener() = default;
    };

    struct LambdaEventListener: EventListener {

        explicit LambdaEventListener(std::function<void(Event&)> f): f_(std::move(f)) {}

        void onEvent(Event& event) override {
            f_(event);
        }

    private:
        std::function<void(Event&)> f_;
    };

    namespace detail {
        struct ListenerRegistry;
    }

    // Scope-bound listener registration: destroying (or unsubscribe()-ing) the
    // handle removes the listener. Inert when default-constructed or moved-from,
    // and safe to outlive the dispatcher it came from.
    class Subscription {

    public:
        Subscription() = default;
        Subscription(const Subscription&) = delete;
        Subscription& operator=(const Subscription&) = delete;
        Subscription(Subscription&& other) noexcept;
        Subscription& operator=(Subscription&& other) noexcept;
        ~Subscription();

        [[nodiscard]] bool active() const;

        void unsubscribe();

    private:
        friend class EventDispatcher;
        Subscription(std::weak_ptr<detail::ListenerRegistry> registry, std::string type, std::uint64_t id);

        std::weak_ptr<detail::ListenerRegistry> registry_;
        std::string type_;
        std::uint64_t id_{0};
    };

    class EventDispatcher {

    public:
        EventDispatcher();
        // Copying an object does not copy who is listening to it: a copy starts
        // with no listeners, and assigning new content to an object leaves that
        // object's listeners alone.
        EventDispatcher(const EventDispatcher&);
        EventDispatcher& operator=(const EventDispatcher&);
        // Listeners follow a move; subscriptions keep tracking the moved-to object.
        EventDispatcher(EventDispatcher&&) noexcept;
        EventDispatcher& operator=(EventDispatcher&&) noexcept;

        // The registry owns the callable and the returned handle owns the
        // registration, so neither side can dangle.
        [[nodiscard]] Subscription subscribe(const std::string& type, std::function<void(Event&)> fn);

        // three.js-style identity API: the caller keeps `listener` alive while it
        // is registered. Removing it from inside onEvent is safe.
        void addEventListener(const std::string& type, EventListener& listener);

        bool hasEventListener(const std::string& type, const EventListener& listener) const;

        void removeEventListener(const std::string& type, const EventListener& listener);

        // Listeners removed during dispatch are not called (three.js still calls
        // them; here that stale call would be a use-after-free). Listeners added
        // during dispatch run from the next dispatch on.
        void dispatchEvent(const std::string& type, std::any target = {});

        virtual ~EventDispatcher() = default;

    private:
        std::shared_ptr<detail::ListenerRegistry> registry_;
    };

}// namespace threepp

#endif//THREEPP_EVENTDISPATCHER_HPP
