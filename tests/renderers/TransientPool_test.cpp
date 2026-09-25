// TransientPool placement: which ranges images of different groups and slots
// share. CPU only — AliasPacker is the pool's bookkeeping without a device.

#include <catch2/catch_test_macros.hpp>

#include "threepp/renderers/vulkan/AliasPacker.hpp"

using namespace threepp::vulkan;

namespace {
    constexpr uint32_t kAll = ~0u;
}

TEST_CASE("groups overlap each other, members of a group do not") {
    AliasPacker p;
    const uint32_t b = p.addBlock(1000, 3);
    // Group 0: two images, packed one after the other.
    auto a0 = p.place(0, kAll, 300, 1, 1u << 3);
    REQUIRE(a0.block == b);
    CHECK(a0.offset == 0);
    p.occupy(a0.block, 0, kAll, a0.offset, 300);
    auto a1 = p.place(0, kAll, 300, 1, 1u << 3);
    CHECK(a1.offset == 300);
    p.occupy(a1.block, 0, kAll, a1.offset, 300);
    // Group 1 starts at the bottom of the same block again.
    auto c0 = p.place(1, kAll, 500, 1, 1u << 3);
    REQUIRE(c0.block == b);
    CHECK(c0.offset == 0);
    p.occupy(c0.block, 1, kAll, c0.offset, 500);
    CHECK(p.liveBytes() == 1000);
}

TEST_CASE("within a group, images of different slots overlap, images a slot shares do not") {
    AliasPacker p;
    const uint32_t b = p.addBlock(1000, 0);
    // A shared image (every slot) at the bottom.
    p.occupy(b, 0, kAll, 0, 200);
    // Slot 0's and slot 1's copies of a per-slot image both land above it, at
    // the same offset.
    auto s0 = p.place(0, 1u, 300, 1, 1u);
    CHECK(s0.offset == 200);
    p.occupy(b, 0, 1u, s0.offset, 300);
    auto s1 = p.place(0, 2u, 300, 1, 1u);
    CHECK(s1.offset == 200);
    p.occupy(b, 0, 2u, s1.offset, 300);
    // A second shared image avoids both.
    auto s2 = p.place(0, kAll, 100, 1, 1u);
    CHECK(s2.offset == 500);
    // Releasing slot 1's copy leaves slot 0's in place.
    p.release(b, 0, 2u, 200);
    auto s3 = p.place(0, 1u, 100, 1, 1u);
    CHECK(s3.offset == 500);
    auto s4 = p.place(0, 2u, 100, 1, 1u);
    CHECK(s4.offset == 200);
}

TEST_CASE("alignment, freed gaps, and blocks that run out") {
    AliasPacker p;
    const uint32_t b = p.addBlock(1024, 0);
    p.occupy(b, 7, kAll, 0, 100);
    // Aligned past the first image.
    auto x = p.place(7, kAll, 200, 256, 1u);
    CHECK(x.offset == 256);
    p.occupy(b, 7, kAll, x.offset, 200);
    // A gap at the front fits a small aligned image once the first is freed.
    p.release(b, 7, kAll, 0);
    auto y = p.place(7, kAll, 100, 64, 1u);
    CHECK(y.offset == 0);
    // Too big for what is left, or for any block: no placement.
    CHECK(p.place(7, kAll, 800, 1, 1u).block == AliasPacker::kNone);
    CHECK(p.place(9, kAll, 2000, 1, 1u).block == AliasPacker::kNone);
    // Wrong memory type: no placement either.
    CHECK(p.place(9, kAll, 10, 1, 1u << 5).block == AliasPacker::kNone);
}

TEST_CASE("an emptied block can be removed and is not used again") {
    AliasPacker p;
    const uint32_t b = p.addBlock(100, 0);
    p.occupy(b, 0, kAll, 0, 50);
    CHECK_FALSE(p.blockEmpty(b));
    p.release(b, 0, kAll, 0);
    CHECK(p.blockEmpty(b));
    p.removeBlock(b);
    CHECK(p.liveBytes() == 0);
    CHECK(p.place(0, kAll, 10, 1, 1u).block == AliasPacker::kNone);
}

TEST_CASE("closed blocks take no new images but keep the ones they hold") {
    AliasPacker p;
    const uint32_t b = p.addBlock(100, 0);
    p.occupy(b, 0, kAll, 0, 50);
    p.closeAll();
    CHECK(p.place(1, kAll, 10, 1, 1u).block == AliasPacker::kNone);
    CHECK_FALSE(p.blockEmpty(b));
    const uint32_t c = p.addBlock(200, 0);
    CHECK(p.place(1, kAll, 10, 1, 1u).block == c);
}
