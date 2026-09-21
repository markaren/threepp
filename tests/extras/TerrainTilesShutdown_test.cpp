// TileTerrain::drainBakes — an owner can finish the terrain's bake workers while the
// data their provider reads is still alive.
//
// The provider is a bag of std::functions and what they capture is the owner's
// business. GeoScene's read its terrain pack and its road network by reference, and
// GeoScene declares those two BEFORE tiles_ so that they are destroyed after it. That
// holds among the members and no further: the TileTerrain is also a CHILD of the
// owner, and Object3D's child list is a base-class member, released only after every
// member of the owning class is already gone. So the last reference to the terrain
// dropped after the pack and the network had been destroyed, its destructor sat
// waiting for bake workers (a std::async future blocks in its destructor), and those
// workers went on calling a provider whose data was freed. Closing a --terrain window
// while tiles were still streaming in killed the process with an access violation
// inside RoadNetwork::pavedWeight, on a bake thread.
//
// The Owner below has the same shape. It cannot test for a use-after-free directly,
// so the data announces its own death through a flag that lives outside it, and the
// provider counts every call it receives after that.

#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/terrain/TerrainTiles.hpp"
#include "threepp/objects/Group.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

using namespace threepp;
using namespace threepp::terrain;

namespace {

    std::atomic<bool> gDataAlive{false};
    std::atomic<int> gCallsAfterDeath{0};
    std::atomic<int> gCalls{0};

    struct Data {
        float height = 3.f;
        Data() { gDataAlive = true; }
        ~Data() { gDataAlive = false; }
    };

    class Owner: public Group {

    public:
        explicit Owner(bool drain): drain_(drain) {
            TerrainProvider prov;
            const Data* d = &data_;
            prov.height = [d](float, float) {
                ++gCalls;
                // Slow on purpose: a tile is thousands of samples, so a bake is
                // certain to be still running when the owner is destroyed. A
                // spin, not a sleep: sleep_for rounds up to the scheduler tick
                // (1 to 15 ms on Windows), which turns one tile into minutes.
                for (volatile int spin = 0; spin < 4000; ++spin) {}
                if (!gDataAlive) {
                    ++gCallsAfterDeath;
                    return 0.f;
                }
                return d->height;
            };

            TileTerrainOptions o;
            o.worldSize = 400.f;
            o.rootGrid = 2;
            o.maxDepth = 2;
            o.tileRes = 32;
            o.asyncBake = true;
            tiles_ = TileTerrain::create(prov, o);
            add(tiles_);
        }

        ~Owner() override {
            if (drain_) tiles_->drainBakes();
        }

        void update() { tiles_->update(Vector3(0, 50, 0)); }
        [[nodiscard]] int pendingBakes() const { return tiles_->pendingBakes(); }

    private:
        // The GeoScene order: the data first, so it is destroyed after tiles_ ...
        Data data_;
        std::shared_ptr<TileTerrain> tiles_;
        bool drain_;
    };

    int destroyMidBake(bool drain) {
        gCallsAfterDeath = 0;
        gCalls = 0;
        {
            Owner owner(drain);
            owner.update();
            REQUIRE(owner.pendingBakes() > 0);
            // Let the workers get going, and destroy the owner under them.
            while (gCalls < 50) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return gCallsAfterDeath.load();
    }

}// namespace

TEST_CASE("the member order alone does not keep the provider's data alive for the workers") {

    // Documents the hazard the fix exists for: without drainBakes the workers
    // outlive the data. If this ever reads 0, the scenario below proves nothing.
    CHECK(destroyMidBake(/*drain=*/false) > 0);
}

TEST_CASE("drainBakes in the owner's destructor ends the workers before its data dies") {

    CHECK(destroyMidBake(/*drain=*/true) == 0);
}
