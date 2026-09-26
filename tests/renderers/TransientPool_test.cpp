// TransientPool placement: which ranges images of different spans and slots
// share. CPU only — AliasPacker is the pool's bookkeeping without a device.

#include <catch2/catch_test_macros.hpp>

#include "threepp/renderers/vulkan/AliasPacker.hpp"

using namespace threepp::vulkan;

namespace {
    constexpr uint32_t kAll = ~0u;
    // Single-phase spans, as transientGroup makes them.
    constexpr AliasPacker::Span kA = 1u << 0, kB = 1u << 1, kC = 1u << 2;
}

TEST_CASE("disjoint spans overlap each other, images of one span do not") {
    AliasPacker p;
    const uint32_t b = p.addBlock(1000, 3);
    // Span A: two images, packed one after the other.
    auto a0 = p.place(kA, kAll, 300, 1, 1u << 3);
    REQUIRE(a0.block == b);
    CHECK(a0.offset == 0);
    p.occupy(a0.block, kA, kAll, a0.offset, 300);
    auto a1 = p.place(kA, kAll, 300, 1, 1u << 3);
    CHECK(a1.offset == 300);
    p.occupy(a1.block, kA, kAll, a1.offset, 300);
    // Span B starts at the bottom of the same block again.
    auto c0 = p.place(kB, kAll, 500, 1, 1u << 3);
    REQUIRE(c0.block == b);
    CHECK(c0.offset == 0);
    p.occupy(c0.block, kB, kAll, c0.offset, 500);
    CHECK(p.liveBytes() == 1000);
}

TEST_CASE("an image live across two phases avoids both, and a third phase avoids neither") {
    AliasPacker p;
    const uint32_t b = p.addBlock(1000, 0);
    p.occupy(b, kA, kAll, 0, 200);  // live in A only
    p.occupy(b, kB, kAll, 0, 300);  // live in B only, over the A image
    // Live in A and B (sceneHdr across the shade and the post chain): above both.
    auto ab = p.place(kA | kB, kAll, 100, 1, 1u);
    CHECK(ab.offset == 300);
    p.occupy(b, kA | kB, kAll, ab.offset, 100);
    // Live in C only (the tail): back at the bottom, over all of them.
    auto c = p.place(kC, kAll, 400, 1, 1u);
    CHECK(c.offset == 0);
    p.occupy(b, kC, kAll, c.offset, 400);
    // Live in B and C: above the B image, the A+B image and the C image.
    auto bc = p.place(kB | kC, kAll, 50, 1, 1u);
    CHECK(bc.offset == 400);
    // Releasing the A+B image frees its range for another A+B image only when
    // nothing else it meets sits there.
    p.release(b, kA | kB, kAll, 300);
    auto ab2 = p.place(kA | kB, kAll, 100, 1, 1u);
    CHECK(ab2.offset == 300);
}

TEST_CASE("transientSpan covers the phases from first to last, per view") {
    CHECK(transientSpan(TransientPhase::Light, TransientPhase::Post) ==
          (transientGroup(TransientPhase::Light) | transientGroup(TransientPhase::Filter) |
           transientGroup(TransientPhase::Post)));
    // Different views' spans never meet; views fold modulo 8 (only ever more
    // conflicts, never fewer).
    CHECK((transientGroup(TransientPhase::Post, 0) & transientGroup(TransientPhase::Post, 1)) == 0);
    CHECK(transientGroup(TransientPhase::Post, 1) == transientGroup(TransientPhase::Post, 9));
}

TEST_CASE("within a span, images of different slots overlap, images a slot shares do not") {
    AliasPacker p;
    const uint32_t b = p.addBlock(1000, 0);
    // A shared image (every slot) at the bottom.
    p.occupy(b, kA, kAll, 0, 200);
    // Slot 0's and slot 1's copies of a per-slot image both land above it, at
    // the same offset.
    auto s0 = p.place(kA, 1u, 300, 1, 1u);
    CHECK(s0.offset == 200);
    p.occupy(b, kA, 1u, s0.offset, 300);
    auto s1 = p.place(kA, 2u, 300, 1, 1u);
    CHECK(s1.offset == 200);
    p.occupy(b, kA, 2u, s1.offset, 300);
    // A second shared image avoids both.
    auto s2 = p.place(kA, kAll, 100, 1, 1u);
    CHECK(s2.offset == 500);
    // Releasing slot 1's copy leaves slot 0's in place.
    p.release(b, kA, 2u, 200);
    auto s3 = p.place(kA, 1u, 100, 1, 1u);
    CHECK(s3.offset == 500);
    auto s4 = p.place(kA, 2u, 100, 1, 1u);
    CHECK(s4.offset == 200);
}

TEST_CASE("alignment, freed gaps, and blocks that run out") {
    AliasPacker p;
    const uint32_t b = p.addBlock(1024, 0);
    p.occupy(b, kC, kAll, 0, 100);
    // Aligned past the first image.
    auto x = p.place(kC, kAll, 200, 256, 1u);
    CHECK(x.offset == 256);
    p.occupy(b, kC, kAll, x.offset, 200);
    // A gap at the front fits a small aligned image once the first is freed.
    p.release(b, kC, kAll, 0);
    auto y = p.place(kC, kAll, 100, 64, 1u);
    CHECK(y.offset == 0);
    // Too big for what is left, or for any block: no placement.
    CHECK(p.place(kC, kAll, 800, 1, 1u).block == AliasPacker::kNone);
    CHECK(p.place(kB, kAll, 2000, 1, 1u).block == AliasPacker::kNone);
    // Wrong memory type: no placement either.
    CHECK(p.place(kB, kAll, 10, 1, 1u << 5).block == AliasPacker::kNone);
}

TEST_CASE("an emptied block can be removed and is not used again") {
    AliasPacker p;
    const uint32_t b = p.addBlock(100, 0);
    p.occupy(b, kA, kAll, 0, 50);
    CHECK_FALSE(p.blockEmpty(b));
    p.release(b, kA, kAll, 0);
    CHECK(p.blockEmpty(b));
    p.removeBlock(b);
    CHECK(p.liveBytes() == 0);
    CHECK(p.place(kA, kAll, 10, 1, 1u).block == AliasPacker::kNone);
}

TEST_CASE("closed blocks take no new images but keep the ones they hold") {
    AliasPacker p;
    const uint32_t b = p.addBlock(100, 0);
    p.occupy(b, kA, kAll, 0, 50);
    p.closeAll();
    CHECK(p.place(kB, kAll, 10, 1, 1u).block == AliasPacker::kNone);
    CHECK_FALSE(p.blockEmpty(b));
    const uint32_t c = p.addBlock(200, 0);
    CHECK(p.place(kB, kAll, 10, 1, 1u).block == c);
}
