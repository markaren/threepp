// SogLoader: round trips, the seven conventions, the padding tripwire, errors.
//
// Every convention in this format decodes to a plausible-looking cloud when you
// get it wrong, and SplatData::validate() catches none of them — so most of the
// cases below are named after the specific WRONG implementation they exist to
// fail. See the gotcha list in SogLoader.hpp; each numbered gotcha there has at
// least one case here.

#include "sog_writer.hpp"

#include "threepp/loaders/SogLoader.hpp"
#include "threepp/splats/SplatData.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace threepp;
using Catch::Approx;

namespace {

    // A scratch directory per case, removed on the way in so a previous run's
    // planes can never be mistaken for this one's.
    std::filesystem::path scratch(const std::string& name) {

        auto dir = std::filesystem::temp_directory_path() / "threepp_sog_test" / name;
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir);
        return dir;
    }

    SplatData makeCloud(int degree, std::size_t count = 64, float extent = 4.f) {

        SplatGenerator::Options options;
        options.count = count;
        options.shDegree = degree;
        options.seed = 20260806u;
        options.extent = {extent, extent, extent};
        // The format stores log(scale), so a zero-scale splat is not
        // representable — the same reason splat_ply_writer.hpp excludes them.
        options.includeDegenerates = false;
        return SplatGenerator::generate(options);
    }

    // Tolerances are the quantisation steps, not fudge factors. Means carry a
    // 16-bit code over the cloud's own log-space range; everything else is an
    // 8-bit codebook index. See the tolerance note in sog_writer.hpp.
    void compare(const SplatData& a, const SplatData& b, float meanTol, float codebookTol) {

        REQUIRE(a.count() == b.count());
        REQUIRE(a.shDegree == b.shDegree);

        for (std::size_t i = 0; i < a.count(); ++i) {

            INFO("splat " << i);

            CHECK(a.means[i].x == Approx(b.means[i].x).margin(meanTol));
            CHECK(a.means[i].y == Approx(b.means[i].y).margin(meanTol));
            CHECK(a.means[i].z == Approx(b.means[i].z).margin(meanTol));

            CHECK(a.scales[i].x == Approx(b.scales[i].x).epsilon(codebookTol));
            CHECK(a.scales[i].y == Approx(b.scales[i].y).epsilon(codebookTol));
            CHECK(a.scales[i].z == Approx(b.scales[i].z).epsilon(codebookTol));

            CHECK(a.opacities[i] == Approx(b.opacities[i]).margin(1.f / 255.f));

            // Up to sign: q and -q are the same rotation, and smallest-three
            // normalises the omitted component positive, so a round trip may
            // legitimately flip the whole quaternion.
            const auto& qa = a.rotations[i];
            const auto& qb = b.rotations[i];
            const float dot = qa.x * qb.x + qa.y * qb.y + qa.z * qb.z + qa.w * qb.w;
            const float s = dot < 0 ? -1.f : 1.f;
            CHECK(qa.x == Approx(s * qb.x).margin(0.01f));
            CHECK(qa.y == Approx(s * qb.y).margin(0.01f));
            CHECK(qa.z == Approx(s * qb.z).margin(0.01f));
            CHECK(qa.w == Approx(s * qb.w).margin(0.01f));

            const float* ca = a.shAt(i);
            const float* cb = b.shAt(i);
            for (int k = 0; k < a.coeffCount() * 3; ++k) {

                INFO("coefficient " << k);
                CHECK(ca[k] == Approx(cb[k]).margin(codebookTol));
            }
        }
    }

    // For these cases the MESSAGE is the deliverable, not just the throw: a
    // mode-byte failure reported as "bad file" would leave the next reader no
    // better off than the silent misdecode it exists to prevent.
    void throwsContaining(const std::function<void()>& body, const std::string& needle) {

        try {

            body();

        } catch (const std::runtime_error& e) {

            const std::string what = e.what();
            INFO("message: " << what);
            CHECK(what.find(needle) != std::string::npos);
            return;
        }
        FAIL("expected a std::runtime_error containing '" << needle << "'");
    }

    // Rewrites one integer member of a chunk's meta.json. Used to forge the
    // corrupt inputs a writer cannot produce.
    void patchMetaCount(const std::filesystem::path& dir, std::size_t newCount) {

        const auto file = dir / "meta.json";
        std::ifstream in(file);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();

        const auto key = text.find("\"count\"");
        REQUIRE(key != std::string::npos);
        const auto colon = text.find(':', key);
        const auto comma = text.find(',', colon);
        REQUIRE(comma != std::string::npos);

        text = text.substr(0, colon + 1) + std::to_string(newCount) + text.substr(comma);

        std::ofstream out(file, std::ios::trunc);
        out << text;
    }

    constexpr const char* scanSkipReason =
            "set THREEPP_SOG_SCAN to a SOG asset directory (or archive) to run this test";

    std::optional<std::filesystem::path> scanRoot() {

        const char* env = std::getenv("THREEPP_SOG_SCAN");
        if (!env || !*env) return std::nullopt;

        std::filesystem::path root(env);
        std::error_code ec;
        if (!std::filesystem::exists(root, ec)) return std::nullopt;
        return root;
    }

}// namespace


TEST_CASE("SogLoader round trips a chunk at every SH degree") {

    for (int degree = 0; degree <= 3; ++degree) {

        DYNAMIC_SECTION("degree " << degree) {

            const auto cloud = makeCloud(degree);
            const auto dir = scratch("roundtrip" + std::to_string(degree));
            (void) splattest::writeSogChunk(dir, cloud);

            const auto loaded = SogLoader::load(dir);

            REQUIRE(loaded.shDegree == degree);
            compare(cloud, loaded, 1e-3f, 0.02f);

            std::string why;
            CHECK(loaded.validate(&why));
        }
    }
}

TEST_CASE("SogLoader reads a chunk named by its meta.json, not just its directory") {

    const auto cloud = makeCloud(1, 16);
    const auto dir = scratch("bymeta");
    (void) splattest::writeSogChunk(dir, cloud);

    const auto loaded = SogLoader::load(dir / "meta.json");
    CHECK(loaded.count() == cloud.count());
}

// Gotcha 2. The planes are padded and the padding is NOT inert: a loader that
// iterates width * height instead of `count` reads it. Forging that is a matter
// of lying about the count, which is exactly what a mis-sized image would do.
TEST_CASE("SogLoader ignores padding, and says so when asked to read it") {

    const auto cloud = makeCloud(1, 20);

    SECTION("padding is invisible to a correct read") {

        const auto tight = scratch("padtight");
        const auto padded = scratch("padpadded");

        (void) splattest::writeSogChunk(tight, cloud);

        splattest::SogWriteOptions options;
        options.padTo = 200;// far more pixels than splats
        options.poisonPadding = true;
        (void) splattest::writeSogChunk(padded, cloud, options);

        const auto a = SogLoader::load(tight);
        const auto b = SogLoader::load(padded);

        REQUIRE(a.count() == cloud.count());
        REQUIRE(b.count() == cloud.count());
        compare(a, b, 1e-3f, 0.02f);
    }

    SECTION("reading into the padding trips the quaternion mode guard") {

        const auto dir = scratch("padoverrun");
        splattest::SogWriteOptions options;
        options.padTo = 200;
        options.poisonPadding = true;
        const unsigned int width = splattest::writeSogChunk(dir, cloud, options);

        // Claim the whole image is splats. A correct decoder would now walk off
        // into poisoned padding — and the mode guard is what stops it.
        const std::size_t rows = (200 + width - 1) / width;
        patchMetaCount(dir, static_cast<std::size_t>(width) * rows);

        throwsContaining([&] { (void) SogLoader::load(dir); }, "mode byte");
    }
}

// Gotcha 3. means.mins/maxs are in LOG space. Skipping the expansion renders a
// scan roughly a hundred times too small — and renders it, rather than failing.
// A cloud spread over hundreds of units is what makes the difference visible.
TEST_CASE("SogLoader expands the log-space means") {

    const auto cloud = makeCloud(0, 128, 800.f);

    float wanted = 0.f;
    for (std::size_t i = 0; i < cloud.count(); ++i) wanted = std::max(wanted, std::abs(cloud.means[i].x));
    REQUIRE(wanted > 100.f);

    const auto dir = scratch("logmeans");
    (void) splattest::writeSogChunk(dir, cloud);
    const auto loaded = SogLoader::load(dir);

    float got = 0.f;
    for (std::size_t i = 0; i < loaded.count(); ++i) got = std::max(got, std::abs(loaded.means[i].x));

    // A decoder that skipped sign(n) * (exp(|n|) - 1) would land near log(800),
    // which is under 7 — two orders of magnitude adrift.
    CHECK(got == Approx(wanted).epsilon(0.01f));
    CHECK(got > 100.f);
}

// Gotcha 5. sh0's codebook holds RAW DC coefficients. Applying the spec's
// render-time colour = 0.5 + c * SH_C0 here would put this loader at odds with
// SplatLoader on the same scan, and the error is a smooth rescale that looks
// like nothing worse than a contrast change.
TEST_CASE("SogLoader stores the DC coefficient raw, not as a colour") {

    auto cloud = makeCloud(0, 8);
    for (std::size_t i = 0; i < cloud.count(); ++i) {

        float* c = cloud.shAt(i);
        c[0] = 3.f;
        c[1] = -2.f;
        c[2] = 0.5f;
    }

    const auto dir = scratch("dcraw");
    (void) splattest::writeSogChunk(dir, cloud);
    const auto loaded = SogLoader::load(dir);

    const float* c = loaded.shAt(0);
    CHECK(c[0] == Approx(3.f).margin(0.05f));
    CHECK(c[1] == Approx(-2.f).margin(0.05f));
    CHECK(c[2] == Approx(0.5f).margin(0.05f));

    // What the colour conversion would have produced, so the case fails loudly
    // rather than merely drifting if someone applies it.
    CHECK(c[0] != Approx(0.5f + 3.f * splats::SH_C0).margin(0.05f));
}

// Gotcha 5 again: opacity is already activated. A sigmoid here would map 0.5 to
// 0.62 and every value into (0,1) — never out of range, so nothing downstream
// would complain.
TEST_CASE("SogLoader takes opacity as-is, with no sigmoid") {

    auto cloud = makeCloud(0, 8);
    cloud.opacities[0] = 0.25f;
    cloud.opacities[1] = 0.5f;
    cloud.opacities[2] = 0.75f;

    const auto dir = scratch("opacity");
    (void) splattest::writeSogChunk(dir, cloud);
    const auto loaded = SogLoader::load(dir);

    CHECK(loaded.opacities[0] == Approx(0.25f).margin(1.f / 255.f));
    CHECK(loaded.opacities[1] == Approx(0.5f).margin(1.f / 255.f));
    CHECK(loaded.opacities[2] == Approx(0.75f).margin(1.f / 255.f));

    CHECK(loaded.opacities[1] != Approx(splats::sigmoid(0.5f)).margin(0.01f));
}

// Gotcha 6. The palette is coefficient-major with rgb in the pixel, which is
// already SplatData's layout — so the channel-major transpose SplatLoader needs
// for the INRIA PLY must NOT be copied here. A transpose keeps every value and
// only moves it, so the cloud still renders; it just shades wrongly while
// orbiting.
TEST_CASE("SogLoader does not transpose the higher-order coefficients") {

    auto cloud = makeCloud(3, 4);
    // Give every (coefficient, channel) slot a distinct value, so any permutation
    // shows up as a mismatch rather than cancelling.
    for (std::size_t i = 0; i < cloud.count(); ++i) {

        float* c = cloud.shAt(i);
        for (int k = 0; k < cloud.coeffCount(); ++k) {

            for (int ch = 0; ch < 3; ++ch) c[k * 3 + ch] = static_cast<float>(k) + 0.1f * ch;
        }
    }

    const auto dir = scratch("shorder");
    (void) splattest::writeSogChunk(dir, cloud);
    const auto loaded = SogLoader::load(dir);

    const float* c = loaded.shAt(0);
    for (int k = 0; k < loaded.coeffCount(); ++k) {

        INFO("coefficient " << k);
        for (int ch = 0; ch < 3; ++ch) {

            CHECK(c[k * 3 + ch] == Approx(static_cast<float>(k) + 0.1f * ch).margin(0.03f));
        }
    }
}

TEST_CASE("SogLoader describes a chunk without decoding it") {

    const auto cloud = makeCloud(2, 32);
    const auto dir = scratch("describe");
    (void) splattest::writeSogChunk(dir, cloud);

    const auto info = SogLoader::describe(dir);
    CHECK(info.lodLevels == 1);
    CHECK(info.shDegree == 2);
    REQUIRE(info.levels.size() == 1);
    CHECK(info.levels[0].count == 32);
}

TEST_CASE("SogLoader recognises a SOG asset by content") {

    const auto cloud = makeCloud(0, 8);
    const auto dir = scratch("issog");
    (void) splattest::writeSogChunk(dir, cloud);

    CHECK(SogLoader::isSog(dir));
    CHECK(SogLoader::isSog(dir / "meta.json"));

    SECTION("and declines everything else without throwing") {

        const auto empty = scratch("issog_empty");
        CHECK_FALSE(SogLoader::isSog(empty));
        CHECK_FALSE(SogLoader::isSog(empty / "nothing_here"));

        const auto notJson = empty / "meta.json";
        std::ofstream(notJson) << "this is not json";
        CHECK_FALSE(SogLoader::isSog(empty));
    }
}

TEST_CASE("SogLoader rejects what it cannot represent") {

    const auto cloud = makeCloud(1, 8);

    SECTION("a missing asset") {

        const auto dir = scratch("err_missing");
        throwsContaining([&] { (void) SogLoader::load(dir); }, "meta.json");
    }

    SECTION("an unsupported version") {

        const auto dir = scratch("err_version");
        (void) splattest::writeSogChunk(dir, cloud);

        const auto file = dir / "meta.json";
        std::ifstream in(file);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();
        const auto at = text.find("\"version\"");
        REQUIRE(at != std::string::npos);
        text[text.find(':', at) + 2] = '9';
        std::ofstream(file, std::ios::trunc) << text;

        throwsContaining([&] { (void) SogLoader::load(dir); }, "version");
    }

    SECTION("a lod requested from a lone chunk") {

        const auto dir = scratch("err_lod");
        (void) splattest::writeSogChunk(dir, cloud);

        SogLoader::Options options;
        options.lod = 1;
        throwsContaining([&] { (void) SogLoader::load(dir, options); }, "single chunk");
    }
}

// The real thing. Everything above is generated by a writer that shares this
// loader's understanding of the format, so a symmetric misreading would cancel;
// only a file written by splat-transform can catch that.
TEST_CASE("SogLoader reads a real SOG scan") {

    const auto root = scanRoot();
    if (!root) SKIP(scanSkipReason);

    const auto info = SogLoader::describe(*root);
    CHECK(info.lodLevels >= 1);
    CHECK(info.shDegree >= 0);

    REQUIRE(!info.levels.empty());
    for (const auto& level : info.levels) {

        INFO("level " << level.lod);
        CHECK(level.count > 0);
        CHECK(!level.chunks.empty());

        std::size_t summed = 0;
        for (const auto& chunk : level.chunks) summed += chunk.count;
        CHECK(summed == level.count);
    }

    // The tree, which per-node LOD addresses splats through. Nothing in the
    // container states that a leaf's (offset, count) ranges TILE their chunk —
    // it is a property of every writer seen, and the property the whole scheme
    // rests on, so it is asserted rather than assumed. A tree that only nearly
    // tiled would render a subset of the scan and look merely thin.
    if (!info.nodes.empty()) {

        // level -> chunk -> the ranges landing in it
        std::vector<std::vector<std::vector<std::pair<std::size_t, std::size_t>>>> tiles(
                info.levels.size());
        for (std::size_t l = 0; l < info.levels.size(); ++l) tiles[l].resize(info.levels[l].chunks.size());

        for (const auto& node : info.nodes) {

            CHECK(!node.lods.empty());
            for (const auto& r : node.lods) {

                REQUIRE(r.lod >= 0);
                REQUIRE(static_cast<std::size_t>(r.lod) < info.levels.size());
                REQUIRE(r.chunk < info.levels[static_cast<std::size_t>(r.lod)].chunks.size());
                tiles[static_cast<std::size_t>(r.lod)][r.chunk].emplace_back(r.offset, r.count);
            }
        }

        for (std::size_t l = 0; l < tiles.size(); ++l) {

            for (std::size_t c = 0; c < tiles[l].size(); ++c) {

                INFO("level " << l << " chunk " << c);
                auto& mine = tiles[l][c];
                std::sort(mine.begin(), mine.end());

                std::size_t at = 0;
                bool contiguous = true;
                for (const auto& [off, count] : mine) {

                    if (off != at) contiguous = false;
                    at += count;
                }
                CHECK(contiguous);
                CHECK(at == info.levels[l].chunks[c].count);
            }
        }
    }

    // The coarsest level, so the case stays affordable: on the Sanctuaire scan
    // that is 625k splats and about a quarter of a second, against 5.0M and
    // ~1.2 GB resident for level 0.
    SogLoader::Options options;
    options.lod = info.lodLevels - 1;
    const auto data = SogLoader::load(*root, options);

    CHECK(data.count() == info.levels[static_cast<std::size_t>(options.lod)].count);

    std::string why;
    CHECK(data.validate(&why));

    float maxNormErr = 0.f;
    float minOpacity = 1.f, maxOpacity = 0.f;
    for (std::size_t i = 0; i < data.count(); ++i) {

        const auto& q = data.rotations[i];
        const float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        maxNormErr = std::max(maxNormErr, std::abs(n - 1.f));
        minOpacity = std::min(minOpacity, data.opacities[i]);
        maxOpacity = std::max(maxOpacity, data.opacities[i]);
    }
    CHECK(maxNormErr < 1e-5f);
    CHECK(minOpacity >= 0.f);
    CHECK(maxOpacity <= 1.f);

    // Every splat inside the asset's own declared bound — the check that fails
    // if the log expansion, the endpoint divisor or the axis order is wrong.
    if (!info.bound.isEmpty()) {

        const auto lo = info.bound.min();
        const auto hi = info.bound.max();
        std::size_t outside = 0;
        for (std::size_t i = 0; i < data.count(); ++i) {

            const auto& m = data.means[i];
            if (m.x < lo.x || m.x > hi.x || m.y < lo.y || m.y > hi.y || m.z < lo.z || m.z > hi.z) ++outside;
        }
        CHECK(outside == 0);
    }
}

// The same asset through the archive path. Points at the .zip superspl.at
// serves, or at a bundled .sog — one container, recognised by its magic.
TEST_CASE("SogLoader reads a real SOG scan out of its archive") {

    const char* env = std::getenv("THREEPP_SOG_ARCHIVE");
    if (!env || !*env) SKIP("set THREEPP_SOG_ARCHIVE to a .zip/.sog holding a SOG asset to run this test");

    const std::filesystem::path archive(env);
    std::error_code ec;
    if (!std::filesystem::is_regular_file(archive, ec)) SKIP("THREEPP_SOG_ARCHIVE is not a file");

    CHECK(SogLoader::isSog(archive));

    const auto info = SogLoader::describe(archive);
    REQUIRE(!info.levels.empty());

    SogLoader::Options options;
    options.lod = info.lodLevels - 1;
    const auto data = SogLoader::load(archive, options);

    CHECK(data.count() == info.levels[static_cast<std::size_t>(options.lod)].count);

    std::string why;
    CHECK(data.validate(&why));

    // If the extracted form is also on hand, the two must agree exactly: the
    // archive is a container, not a transformation.
    if (const auto root = scanRoot()) {

        const auto fromDir = SogLoader::load(*root, options);
        REQUIRE(fromDir.count() == data.count());

        bool identical = true;
        for (std::size_t i = 0; i < data.count() && identical; ++i) {

            identical = data.means[i].x == fromDir.means[i].x &&
                        data.means[i].y == fromDir.means[i].y &&
                        data.means[i].z == fromDir.means[i].z &&
                        data.opacities[i] == fromDir.opacities[i];
        }
        CHECK(identical);
    }
}
