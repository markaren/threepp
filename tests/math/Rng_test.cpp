
#include <catch2/catch_test_macros.hpp>

#include "threepp/math/Rng.hpp"

#include <bit>
#include <set>

using namespace threepp;

// The whole point of Rng is that these numbers can never change — not across
// runs, threads, platforms, or standard libraries. The golden values below
// were produced by the reference implementation; if a toolchain disagrees,
// the type has failed its one job and this test must fail loudly.

TEST_CASE("Rng replays and streams are isolated") {

    // Same seed, same sequence; copy replays the future.
    math::Rng a(42), b(42);
    math::Rng c = a;
    for (int i = 0; i < 100; ++i) {
        const auto x = a.nextUint();
        CHECK(x == b.nextUint());
        CHECK(x == c.nextUint());
    }

    // Forks are order-independent: forking before or after draws, or from a
    // sibling with the same seed, gives the same stream.
    math::Rng fresh(42);
    CHECK(a.fork(7).nextUint() == fresh.fork(7).nextUint());
    CHECK(fresh.fork(7).nextUint() != fresh.fork(8).nextUint());

    // Counter draws are pure functions of their inputs, and slot/stream/seed
    // each matter.
    CHECK(math::Rng::hash01(1, 2, 3) == math::Rng::hash01(1, 2, 3));
    const std::set<float> distinct{
            math::Rng::hash01(1, 2, 3), math::Rng::hash01(2, 2, 3),
            math::Rng::hash01(1, 3, 3), math::Rng::hash01(1, 2, 4)};
    CHECK(distinct.size() == 4);
}

TEST_CASE("Rng bit-level golden values") {

    // Reference values from MSVC 19.41 x64, 2026-08-15. Any platform or
    // toolchain that prints different bits has broken the type's one
    // guarantee; the float is compared by bit pattern, not tolerance.
    math::Rng r(1337);
    REQUIRE(r.nextUint() == 3064506803u);
    REQUIRE(std::bit_cast<std::uint32_t>(r.nextFloat()) == 1061912360u);
    REQUIRE(r.nextInt(0, 99) == 20);
    REQUIRE(std::bit_cast<std::uint32_t>(math::Rng::hash01(1, 2, 3)) == std::bit_cast<std::uint32_t>(0.700296521f));
    REQUIRE(math::Rng(42).fork(7).nextUint() == 1953563u);
}

TEST_CASE("Rng ranges hold") {

    math::Rng r(7);
    for (int i = 0; i < 10000; ++i) {
        const float f = r.nextFloat();
        REQUIRE(f >= 0.f);
        REQUIRE(f < 1.f);
        const int n = r.nextInt(-3, 3);
        REQUIRE(n >= -3);
        REQUIRE(n <= 3);
    }
    CHECK(r.nextInt(5, 5) == 5);
    CHECK(r.nextInt(5, 4) == 5);// degenerate range answers lo, not UB
}
