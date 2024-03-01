
#include "threepp/core/EventDispatcher.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace threepp;


TEST_CASE("Test subscribe") {

    EventDispatcher evt;

    int nCalls = 0;
    {
        auto s0 = evt.subscribe( [&](Event& e) {nCalls++; });

        REQUIRE(nCalls == 0);
        evt.send(Event{});
        REQUIRE(nCalls == 1);
        evt.send(Event{});
        REQUIRE(nCalls == 2);
    }
        evt.send(Event{});
	REQUIRE(nCalls == 2);

}

TEST_CASE("Test addEventListenerOneOwned") {

    EventDispatcher evt;

    int nCalls = 0;
    evt.subscribeForever([&](Event& e) {
        nCalls++;
        if (nCalls == 2) { e.unsubscribe = true; } });

    REQUIRE(nCalls == 0);
    evt.send(Event{});
    REQUIRE(nCalls == 1);
    evt.send(Event{});
    REQUIRE(nCalls == 2);
    evt.send(Event{});
    REQUIRE(nCalls == 2);
}
TEST_CASE("Test subscribeOnce") {

    EventDispatcher evt;

    int nCalls = 0;
	evt.subscribeOnce([&](Event& e) {nCalls++; });
	REQUIRE(nCalls == 0);
    evt.send(Event{});
	REQUIRE(nCalls == 1);
	evt.send(Event{});
	REQUIRE(nCalls == 1);
}
