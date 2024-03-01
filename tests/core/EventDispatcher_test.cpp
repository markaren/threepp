
#include "threepp/core/EventDispatcher.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace threepp;


TEST_CASE("Test addEventListener") {

    EventDispatcher evt;

    int nCalls = 0;
    {
        auto s0 = evt.addEventListener( [&](Event& e) {nCalls++; });

        REQUIRE(nCalls == 0);
        evt.dispatchEvent(Event{});
        REQUIRE(nCalls == 1);
        evt.dispatchEvent(Event{});
        REQUIRE(nCalls == 2);
    }
        evt.dispatchEvent(Event{});
	REQUIRE(nCalls == 2);

}

TEST_CASE("Test addEventListenerOneOwned") {

    EventDispatcher evt;

    int nCalls = 0;
    evt.addEventListenerOwned([&](Event& e) {
        nCalls++;
        if (nCalls == 2) { e.unsubscribe = true; } });

    REQUIRE(nCalls == 0);
    evt.dispatchEvent(Event{});
    REQUIRE(nCalls == 1);
    evt.dispatchEvent(Event{});
    REQUIRE(nCalls == 2);
    evt.dispatchEvent(Event{});
    REQUIRE(nCalls == 2);
}
TEST_CASE("Test addEventListenerOneShot") {

    EventDispatcher evt;

    int nCalls = 0;
	evt.addEventListenerOneShot([&](Event& e) {nCalls++; });
	REQUIRE(nCalls == 0);
    evt.dispatchEvent(Event{});
	REQUIRE(nCalls == 1);
	evt.dispatchEvent(Event{});
	REQUIRE(nCalls == 1);
}
