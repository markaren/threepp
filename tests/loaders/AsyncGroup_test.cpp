// AsyncGroup on a loader that fails.
//
// loadAsync runs the loader on a detached thread. A loader that threw used to
// be caught by an empty `catch (...)`, and a loader that returned nullptr was
// never delivered: the group stayed "not loaded" forever, its onLoaded
// callbacks were never released, and nothing was printed.

#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/AsyncGroup.hpp"
#include "threepp/objects/Group.hpp"

#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>

using namespace threepp;

namespace {

    void waitUntilSettled(const AsyncGroup& group) {
        for (int i = 0; i < 2000 && group.isLoading(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

}// namespace

TEST_CASE("AsyncGroup adopts a loaded group and fires onLoaded") {

    auto group = loadAsync([]() -> std::shared_ptr<Group> {
        auto g = Group::create();
        g->add(Group::create());
        return g;
    });

    bool fired = false;
    group->onLoaded([&](AsyncGroup&) { fired = true; });

    waitUntilSettled(*group);
    REQUIRE_FALSE(group->isLoading());

    group->updateMatrixWorld();

    CHECK(group->isLoaded());
    CHECK_FALSE(group->hasFailed());
    CHECK(fired);
    CHECK(group->children.size() == 1);
}

TEST_CASE("AsyncGroup reports a loader that throws instead of staying unloaded forever") {

    auto group = loadAsync([]() -> std::shared_ptr<Group> {
        throw std::runtime_error("no such file");
    });

    bool fired = false;
    group->onLoaded([&](AsyncGroup&) { fired = true; });

    waitUntilSettled(*group);
    REQUIRE_FALSE(group->isLoading());

    group->updateMatrixWorld();

    CHECK(group->hasFailed());
    CHECK_FALSE(group->isLoaded());
    CHECK_FALSE(fired);
    CHECK(group->children.empty());

    // A callback registered after the failure is dropped too, not parked.
    bool late = false;
    group->onLoaded([&](AsyncGroup&) { late = true; });
    group->updateMatrixWorld();
    CHECK_FALSE(late);
}

TEST_CASE("AsyncGroup treats a loader that returns nothing as a failed load") {

    auto group = loadAsync([]() -> std::shared_ptr<Group> { return nullptr; });

    waitUntilSettled(*group);
    REQUIRE_FALSE(group->isLoading());

    group->updateMatrixWorld();

    CHECK(group->hasFailed());
    CHECK_FALSE(group->isLoaded());
}
