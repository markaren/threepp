
#include "threepp/core/EventDispatcher.hpp"

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace threepp;


namespace threepp::detail {

    struct ListenerRegistry {

        struct Entry {
            std::uint64_t id{0};
            EventListener* raw = nullptr;   // identity API: caller-owned
            std::function<void(Event&)> fn; // subscribe(): registry-owned
        };

        std::unordered_map<std::string, std::vector<Entry>> byType;
        std::uint64_t nextId{1};

        [[nodiscard]] bool contains(const std::string& type, std::uint64_t id) const {
            const auto it = byType.find(type);
            if (it == byType.end()) return false;
            return std::ranges::any_of(it->second, [id](const Entry& e) { return e.id == id; });
        }

        void remove(const std::string& type, std::uint64_t id) {
            const auto it = byType.find(type);
            if (it == byType.end()) return;
            auto& entries = it->second;
            const auto found = std::ranges::find_if(entries, [id](const Entry& e) { return e.id == id; });
            if (found != entries.end()) entries.erase(found);
        }
    };

}// namespace threepp::detail

using threepp::detail::ListenerRegistry;


Subscription::Subscription(std::weak_ptr<ListenerRegistry> registry, std::string type, std::uint64_t id)
    : registry_(std::move(registry)), type_(std::move(type)), id_(id) {}

Subscription::Subscription(Subscription&& other) noexcept
    : registry_(std::move(other.registry_)), type_(std::move(other.type_)), id_(std::exchange(other.id_, 0)) {}

Subscription& Subscription::operator=(Subscription&& other) noexcept {
    if (this != &other) {
        unsubscribe();
        registry_ = std::move(other.registry_);
        type_ = std::move(other.type_);
        id_ = std::exchange(other.id_, 0);
    }
    return *this;
}

Subscription::~Subscription() {
    unsubscribe();
}

bool Subscription::active() const {
    if (id_ == 0) return false;
    const auto registry = registry_.lock();
    return registry && registry->contains(type_, id_);
}

void Subscription::unsubscribe() {
    if (id_ == 0) return;
    if (const auto registry = registry_.lock()) {
        registry->remove(type_, id_);
    }
    registry_.reset();
    id_ = 0;
}


EventDispatcher::EventDispatcher(): registry_(std::make_shared<ListenerRegistry>()) {}

EventDispatcher::EventDispatcher(const EventDispatcher&): EventDispatcher() {}

EventDispatcher& EventDispatcher::operator=(const EventDispatcher&) {
    return *this;
}

// A moved-from dispatcher is left with no registry; the mutating entry points
// below recreate one on demand, the const ones treat it as "no listeners".
EventDispatcher::EventDispatcher(EventDispatcher&&) noexcept = default;

EventDispatcher& EventDispatcher::operator=(EventDispatcher&&) noexcept = default;

Subscription EventDispatcher::subscribe(const std::string& type, std::function<void(Event&)> fn) {

    if (!registry_) registry_ = std::make_shared<ListenerRegistry>();

    const auto id = registry_->nextId++;
    registry_->byType[type].push_back({id, nullptr, std::move(fn)});
    return Subscription{registry_, type, id};
}

void EventDispatcher::addEventListener(const std::string& type, EventListener& listener) {

    if (!registry_) registry_ = std::make_shared<ListenerRegistry>();

    registry_->byType[type].push_back({registry_->nextId++, &listener, {}});
}

bool EventDispatcher::hasEventListener(const std::string& type, const EventListener& listener) const {

    if (!registry_) return false;

    const auto it = registry_->byType.find(type);
    if (it == registry_->byType.end()) return false;

    return std::ranges::any_of(it->second, [&listener](const auto& e) { return e.raw == &listener; });
}

void EventDispatcher::removeEventListener(const std::string& type, const EventListener& listener) {

    if (!registry_) return;

    const auto it = registry_->byType.find(type);
    if (it == registry_->byType.end()) return;

    auto& entries = it->second;
    const auto found = std::ranges::find_if(entries, [&listener](const auto& e) { return e.raw == &listener; });
    if (found != entries.end()) entries.erase(found);
}

void EventDispatcher::dispatchEvent(const std::string& type, std::any target) {

    // A listener may remove itself, remove a later listener, or destroy this
    // dispatcher outright: past the snapshot, only these locals are touched.
    const auto registry = registry_;
    if (!registry) return;

    const auto it = registry->byType.find(type);
    if (it == registry->byType.end() || it->second.empty()) return;

    Event e{type, std::move(target)};

    const auto snapshot = it->second;
    for (const auto& entry : snapshot) {
        if (!registry->contains(type, entry.id)) continue;// removed mid-dispatch
        if (entry.raw) {
            entry.raw->onEvent(e);
        } else {
            entry.fn(e);
        }
    }
}
