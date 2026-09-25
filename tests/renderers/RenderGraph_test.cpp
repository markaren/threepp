// RenderGraph barrier planning. CPU only: compile() plans barriers without a
// device or command buffer, so these run everywhere.

#include <catch2/catch_test_macros.hpp>

#include "threepp/renderers/vulkan/RenderGraph.hpp"

using namespace threepp::vulkan;

namespace {

    // Distinct fake handles; the planner never dereferences them.
    VkImage fakeImage(uintptr_t v) { return reinterpret_cast<VkImage>(v); }
    VkBuffer fakeBuffer(uintptr_t v) { return reinterpret_cast<VkBuffer>(v); }

    constexpr auto CS      = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    constexpr auto FS      = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    constexpr auto GENERAL = VK_IMAGE_LAYOUT_GENERAL;
    constexpr auto RO      = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

}// namespace

TEST_CASE("write then read gets one barrier, repeated reads none") {
    rg::RenderGraph g;
    auto img = g.importImage("hdr", fakeImage(1), VK_IMAGE_ASPECT_COLOR_BIT, 1, GENERAL);
    g.addPass("shade", {}).use(img, rg::storageWrite());
    g.addPass("bloom", {}).use(img, rg::sampled(GENERAL));
    g.addPass("post", {}).use(img, rg::sampled(GENERAL));
    g.compile();

    CHECK(g.hasEntryMemoryBarrier());
    CHECK(g.entryBarriers().empty());// GENERAL at entry and at first use
    CHECK(g.barriersBefore(0).empty());// first write after the entry sync

    REQUIRE(g.barriersBefore(1).size() == 1);
    const auto& b = g.barriersBefore(1)[0];
    CHECK(b.srcStages == CS);
    CHECK(b.srcAccess == VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    CHECK(b.dstStages == CS);
    CHECK(b.dstAccess == VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    CHECK(b.oldLayout == GENERAL);
    CHECK(b.newLayout == GENERAL);

    CHECK(g.barriersBefore(2).empty());// same access already visible
    CHECK(g.exitBarriers().empty());
    CHECK(g.hasExitMemoryBarrier());
}

TEST_CASE("a read at a new stage needs its own barrier") {
    rg::RenderGraph g;
    auto img = g.importImage("img", fakeImage(1), VK_IMAGE_ASPECT_COLOR_BIT, 1, GENERAL);
    g.addPass("w", {}).use(img, rg::storageWrite());
    g.addPass("r_cs", {}).use(img, rg::sampled(GENERAL, CS));
    g.addPass("r_fs", {}).use(img, rg::sampled(GENERAL, FS));
    g.compile();
    REQUIRE(g.barriersBefore(2).size() == 1);
    CHECK(g.barriersBefore(2)[0].srcStages == CS);
    CHECK(g.barriersBefore(2)[0].dstStages == FS);
}

TEST_CASE("write after read orders against the readers") {
    rg::RenderGraph g;
    auto img = g.importImage("img", fakeImage(1), VK_IMAGE_ASPECT_COLOR_BIT, 1, GENERAL);
    g.addPass("read", {}).use(img, rg::sampled(GENERAL, FS));
    g.addPass("write", {}).use(img, rg::storageWrite());
    g.compile();
    CHECK(g.barriersBefore(0).empty());// read of entry-synced data
    REQUIRE(g.barriersBefore(1).size() == 1);
    const auto& b = g.barriersBefore(1)[0];
    CHECK(b.srcStages == FS);
    CHECK(b.srcAccess == 0);
    CHECK(b.dstStages == CS);
}

TEST_CASE("layouts: first use in the entry barrier, later changes in the pass, exit restores") {
    rg::RenderGraph g;
    auto img = g.importImage("img", fakeImage(1), VK_IMAGE_ASPECT_COLOR_BIT, 1, RO);
    g.addPass("write", {}).use(img, rg::storageWrite());
    g.addPass("read", {}).use(img, rg::sampled(RO));
    g.compile();

    REQUIRE(g.entryBarriers().size() == 1);
    CHECK(g.entryBarriers()[0].oldLayout == RO);
    CHECK(g.entryBarriers()[0].newLayout == GENERAL);
    CHECK(g.barriersBefore(0).empty());

    REQUIRE(g.barriersBefore(1).size() == 1);
    CHECK(g.barriersBefore(1)[0].oldLayout == GENERAL);
    CHECK(g.barriersBefore(1)[0].newLayout == RO);
    CHECK(g.barriersBefore(1)[0].srcAccess == VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    CHECK(g.exitBarriers().empty());// ends in RO, its entry layout
}

TEST_CASE("boundary barriers are scoped to the stages the graph declared") {
    rg::RenderGraph g;
    auto img = g.importImage("img", fakeImage(1), VK_IMAGE_ASPECT_COLOR_BIT, 1, RO, RO);
    g.addPass("write", {}).use(img, rg::storageWrite());
    g.addPass("copy", {}).use(img, rg::transferSrc(GENERAL));
    g.compile();
    const auto used = CS | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    REQUIRE(g.entryBarriers().size() == 1);
    CHECK(g.entryBarriers()[0].srcStages == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    CHECK(g.entryBarriers()[0].dstStages == used);
    REQUIRE(g.exitBarriers().size() == 1);
    CHECK(g.exitBarriers()[0].srcStages == used);
    CHECK(g.exitBarriers()[0].dstStages == VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    CHECK(g.exitBarriers()[0].newLayout == RO);
}

TEST_CASE("an explicit exit layout is transitioned to after the last pass") {
    rg::RenderGraph g;
    auto img = g.importImage("swap", fakeImage(1), VK_IMAGE_ASPECT_COLOR_BIT, 1, GENERAL,
                             VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    g.addPass("write", {}).use(img, rg::storageWrite());
    g.compile();
    REQUIRE(g.exitBarriers().size() == 1);
    CHECK(g.exitBarriers()[0].oldLayout == GENERAL);
    CHECK(g.exitBarriers()[0].newLayout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
}

TEST_CASE("an untouched import gets no transitions") {
    rg::RenderGraph g;
    auto used   = g.importImage("used", fakeImage(1), VK_IMAGE_ASPECT_COLOR_BIT, 1, GENERAL);
    (void) g.importImage("unused", fakeImage(2), VK_IMAGE_ASPECT_COLOR_BIT, 1, RO, GENERAL);
    g.addPass("w", {}).use(used, rg::storageWrite());
    g.compile();
    CHECK(g.entryBarriers().empty());
    CHECK(g.exitBarriers().empty());
}

TEST_CASE("mips are tracked separately and merged when they match") {
    rg::RenderGraph g;
    auto pyr = g.importImage("pyr", fakeImage(1), VK_IMAGE_ASPECT_COLOR_BIT, 4, GENERAL);
    g.addPass("all", {}).use(pyr, rg::storageWrite());
    g.addPass("down1", {}).use(pyr, rg::sampled(GENERAL), 0, 1).use(pyr, rg::storageWrite(), 1, 1);
    g.addPass("readAll", {}).use(pyr, rg::sampled(GENERAL));
    g.compile();

    // down1: mip 0 write->read, mip 1 write->write.
    const auto& d = g.barriersBefore(1);
    REQUIRE(d.size() == 2);
    CHECK(d[0].baseMip == 0);
    CHECK(d[0].dstAccess == VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    CHECK(d[1].baseMip == 1);
    CHECK(d[1].dstAccess == VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // readAll: mip 0 is already visible to a sampled CS read; mips 1..3 were
    // all last written by a CS storage write and merge into one barrier.
    const auto& r = g.barriersBefore(2);
    REQUIRE(r.size() == 1);
    CHECK(r[0].baseMip == 1);
    CHECK(r[0].mipCount == 3);
}

TEST_CASE("two declarations of one image in a pass become one access") {
    rg::RenderGraph g;
    auto img = g.importImage("img", fakeImage(1), VK_IMAGE_ASPECT_COLOR_BIT, 1, GENERAL);
    g.addPass("w", {}).use(img, rg::storageWrite());
    g.addPass("rw", {}).use(img, rg::sampled(GENERAL)).use(img, rg::storageWrite());
    g.compile();
    REQUIRE(g.barriersBefore(1).size() == 1);
    const auto& b = g.barriersBefore(1)[0];
    CHECK(b.dstAccess == (VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT));
}

TEST_CASE("buffers: write, indirect read, then compute read") {
    rg::RenderGraph g;
    auto buf = g.importBuffer("args", fakeBuffer(7));
    g.addPass("cull", {}).use(buf, rg::storageWrite());
    g.addPass("draw", {}).use(buf, rg::indirectRead());
    g.addPass("stats", {}).use(buf, rg::storageRead());
    g.compile();
    REQUIRE(g.barriersBefore(1).size() == 1);
    CHECK_FALSE(g.barriersBefore(1)[0].isImage);
    CHECK(g.barriersBefore(1)[0].dstStages == VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
    REQUIRE(g.barriersBefore(2).size() == 1);
    CHECK(g.barriersBefore(2)[0].dstStages == CS);
}

TEST_CASE("re-importing an image returns the same handle") {
    rg::RenderGraph g;
    auto a = g.importImage("a", fakeImage(3), VK_IMAGE_ASPECT_COLOR_BIT, 1, GENERAL);
    auto b = g.importImage("a again", fakeImage(3), VK_IMAGE_ASPECT_COLOR_BIT, 1, GENERAL);
    CHECK(a.index == b.index);
    CHECK_FALSE(g.importImage("null", VK_NULL_HANDLE, VK_IMAGE_ASPECT_COLOR_BIT, 1, GENERAL).valid());
}

TEST_CASE("a render pass that transitions its attachment itself") {
    // initialLayout UNDEFINED, finalLayout SHADER_RO: the graph orders the
    // pass against earlier readers but records no transition of its own, and
    // the image is in SHADER_RO afterwards.
    rg::RenderGraph g;
    auto img = g.importImage("gbuf", fakeImage(1), VK_IMAGE_ASPECT_COLOR_BIT, 1, RO);
    g.addPass("read", {}).use(img, rg::sampled(RO));
    rg::Access att = rg::colorAttachment(VK_IMAGE_LAYOUT_UNDEFINED, false);
    att.finalLayout = RO;
    g.addPass("raster", {}).use(img, att);
    g.addPass("shade", {}).use(img, rg::sampled(RO));
    g.compile();

    CHECK(g.entryBarriers().empty());
    REQUIRE(g.barriersBefore(1).size() == 1);// WAR against the read
    const auto& b = g.barriersBefore(1)[0];
    CHECK(b.srcStages == CS);
    CHECK(b.oldLayout == RO);
    CHECK(b.newLayout == RO);
    REQUIRE(g.barriersBefore(2).size() == 1);// attachment write -> sampled read, no transition
    CHECK(g.barriersBefore(2)[0].srcStages == VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
    CHECK(g.barriersBefore(2)[0].oldLayout == RO);
    CHECK(g.barriersBefore(2)[0].newLayout == RO);
    CHECK(g.exitBarriers().empty());// already back in its resting layout
}

TEST_CASE("a pass that flips an image and leaves it flipped") {
    rg::RenderGraph g;
    auto img = g.importImage("img", fakeImage(1), VK_IMAGE_ASPECT_COLOR_BIT, 1, RO);
    rg::Access flip = rg::sampled(RO);
    flip.finalLayout = GENERAL;// e.g. the pass's own barrier at its end
    g.addPass("flip", {}).use(img, flip);
    g.addPass("store", {}).use(img, rg::storageWrite());
    g.compile();
    REQUIRE(g.barriersBefore(1).size() == 1);
    CHECK(g.barriersBefore(1)[0].oldLayout == GENERAL);// no second transition
    CHECK(g.barriersBefore(1)[0].newLayout == GENERAL);
    REQUIRE(g.exitBarriers().size() == 1);
    CHECK(g.exitBarriers()[0].oldLayout == GENERAL);
    CHECK(g.exitBarriers()[0].newLayout == RO);
}

TEST_CASE("a reset graph plans like a fresh one") {
    const auto build = [](rg::RenderGraph& g) {
        auto a = g.importImage("a", fakeImage(1), VK_IMAGE_ASPECT_COLOR_BIT, 3, RO);
        auto b = g.importBuffer("b", fakeBuffer(2));
        g.addPass("w", {}).use(a, rg::storageWrite()).use(b, rg::storageWrite());
        g.addPass("r", {}).use(a, rg::sampled(RO), 1, 1).use(b, rg::storageRead());
        g.compile();
    };
    rg::RenderGraph reused;
    // A different graph first, so the reused one starts from dirty pools.
    auto x = reused.importImage("x", fakeImage(9), VK_IMAGE_ASPECT_COLOR_BIT, 5, GENERAL);
    reused.addPass("junk", {}).use(x, rg::storageWrite()).use(x, rg::sampled(GENERAL));
    reused.compile();
    reused.reset();
    build(reused);

    rg::RenderGraph fresh;
    build(fresh);

    REQUIRE(reused.passCount() == fresh.passCount());
    const auto same = [](const std::vector<rg::PlannedBarrier>& p, const std::vector<rg::PlannedBarrier>& q) {
        if (p.size() != q.size()) return false;
        for (size_t i = 0; i < p.size(); ++i) {
            if (p[i].resource != q[i].resource || p[i].baseMip != q[i].baseMip || p[i].mipCount != q[i].mipCount ||
                p[i].srcStages != q[i].srcStages || p[i].dstStages != q[i].dstStages ||
                p[i].oldLayout != q[i].oldLayout || p[i].newLayout != q[i].newLayout || p[i].isImage != q[i].isImage)
                return false;
        }
        return true;
    };
    CHECK(same(reused.entryBarriers(), fresh.entryBarriers()));
    CHECK(same(reused.exitBarriers(), fresh.exitBarriers()));
    for (size_t i = 0; i < fresh.passCount(); ++i) CHECK(same(reused.barriersBefore(i), fresh.barriersBefore(i)));
}

TEST_CASE("an empty graph records nothing") {
    rg::RenderGraph g;
    g.compile();
    CHECK_FALSE(g.hasEntryMemoryBarrier());
    CHECK_FALSE(g.hasExitMemoryBarrier());
}
