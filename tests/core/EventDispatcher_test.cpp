
#include "threepp/core/EventDispatcher.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"

#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <memory>

using namespace threepp;

namespace {


    struct MyEventListener: EventListener {

        int numCalled = 0;

        void onEvent(Event& e) override {
            ++numCalled;
        }
    };

    struct OnMaterialDispose: EventListener {

        void onEvent(Event& event) override {
            auto* material = std::any_cast<Material*>(event.target);
            material->removeEventListener("dispose", *this);
        }
    };

}// namespace

TEST_CASE("Test events") {

    EventDispatcher evt;

    MyEventListener l;

    bool l1Called = false;
    LambdaEventListener l1([&l1Called](Event& e) {
        l1Called = true;
    });

    evt.addEventListener("test1", l);
    evt.addEventListener("test2", l1);

    evt.dispatchEvent("test1");
    evt.dispatchEvent("test1");

    REQUIRE(2 == l.numCalled);

    evt.removeEventListener("test1", l);

    evt.dispatchEvent("test1");
    evt.dispatchEvent("test2");

    REQUIRE(2 == l.numCalled);
    REQUIRE(l1Called);

    REQUIRE(!evt.hasEventListener("test1", l));
    REQUIRE(evt.hasEventListener("test2", l1));

    OnMaterialDispose onDispose;
    auto material = MeshBasicMaterial::create();
    material->addEventListener("dispose", onDispose);

    REQUIRE(material->hasEventListener("dispose", onDispose));
    material->dispose();
    REQUIRE(!material->hasEventListener("dispose", onDispose));
}

TEST_CASE("subscribe: handle going out of scope removes the listener") {

    EventDispatcher evt;
    int calls = 0;
    {
        auto sub = evt.subscribe("test", [&calls](Event&) { ++calls; });
        REQUIRE(sub.active());
        evt.dispatchEvent("test");
        REQUIRE(calls == 1);
    }
    evt.dispatchEvent("test");
    REQUIRE(calls == 1);
}

TEST_CASE("subscribe: unsubscribe is early and idempotent") {

    EventDispatcher evt;
    int calls = 0;
    auto sub = evt.subscribe("test", [&calls](Event&) { ++calls; });

    sub.unsubscribe();
    sub.unsubscribe();
    REQUIRE(!sub.active());

    evt.dispatchEvent("test");
    REQUIRE(calls == 0);
}

TEST_CASE("subscription safely outlives its dispatcher") {

    Subscription sub;
    {
        EventDispatcher evt;
        sub = evt.subscribe("test", [](Event&) {});
        REQUIRE(sub.active());
    }
    REQUIRE(!sub.active());
    sub.unsubscribe();// no-op, no crash
}

TEST_CASE("listener removed during dispatch is not called") {

    EventDispatcher evt;
    int calls = 0;
    Subscription second;
    auto first = evt.subscribe("test", [&second](Event&) { second.unsubscribe(); });
    second = evt.subscribe("test", [&calls](Event&) { ++calls; });

    evt.dispatchEvent("test");
    REQUIRE(calls == 0);
}

TEST_CASE("listener destroyed during dispatch is not called") {

    EventDispatcher evt;
    auto doomed = std::make_unique<LambdaEventListener>([](Event&) { FAIL("stale call to a destroyed listener"); });
    auto first = evt.subscribe("test", [&](Event&) {
        evt.removeEventListener("test", *doomed);
        doomed.reset();
    });
    evt.addEventListener("test", *doomed);

    evt.dispatchEvent("test");
}

TEST_CASE("listener added during dispatch runs from the next dispatch") {

    EventDispatcher evt;
    int calls = 0;
    Subscription inner;
    auto outer = evt.subscribe("test", [&](Event&) {
        if (!inner.active()) {
            inner = evt.subscribe("test", [&calls](Event&) { ++calls; });
        }
    });

    evt.dispatchEvent("test");
    REQUIRE(calls == 0);
    evt.dispatchEvent("test");
    REQUIRE(calls == 1);
}

TEST_CASE("copying does not copy listeners; assignment keeps the target's") {

    EventDispatcher a;
    int calls = 0;
    auto sub = a.subscribe("test", [&calls](Event&) { ++calls; });

    EventDispatcher copy{a};
    copy.dispatchEvent("test");
    REQUIRE(calls == 0);

    a = copy;// assigning new content does not unhook a's listeners
    a.dispatchEvent("test");
    REQUIRE(calls == 1);
}

TEST_CASE("move transfers listeners and subscriptions stay valid") {

    EventDispatcher a;
    int calls = 0;
    auto sub = a.subscribe("test", [&calls](Event&) { ++calls; });

    EventDispatcher b{std::move(a)};
    b.dispatchEvent("test");
    REQUIRE(calls == 1);
    REQUIRE(sub.active());

    sub.unsubscribe();
    b.dispatchEvent("test");
    REQUIRE(calls == 1);

    // the moved-from dispatcher is empty but usable
    a.dispatchEvent("test");
    auto sub2 = a.subscribe("test", [&calls](Event&) { ++calls; });
    a.dispatchEvent("test");
    REQUIRE(calls == 2);
}

TEST_CASE("dispatcher destroyed during dispatch") {

    auto evt = std::make_unique<EventDispatcher>();
    int calls = 0;
    auto first = evt->subscribe("test", [&evt](Event&) { evt.reset(); });
    auto second = evt->subscribe("test", [&calls](Event&) { ++calls; });

    evt->dispatchEvent("test");
    REQUIRE(!evt);
    REQUIRE(calls == 1);
    REQUIRE(!second.active());
}
