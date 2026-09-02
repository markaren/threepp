// The splat data model and its generator: determinism, activations, the
// covariance, bounds and validation.

#include "threepp/math/MathUtils.hpp"
#include "threepp/splats/SplatData.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using namespace threepp;
using Catch::Approx;

namespace {

    SplatGenerator::Options options(unsigned int seed, int degree = 0, size_t count = 128) {

        SplatGenerator::Options o;
        o.seed = seed;
        o.shDegree = degree;
        o.count = count;
        return o;
    }

}// namespace


TEST_CASE("SplatGenerator: the same seed gives the same cloud") {

    const auto a = SplatGenerator::generate(options(42u, 3));
    const auto b = SplatGenerator::generate(options(42u, 3));

    REQUIRE(a.count() == b.count());
    REQUIRE(a.sh.size() == b.sh.size());

    for (size_t i = 0; i < a.count(); ++i) {

        INFO("splat " << i);
        // Bit-identical, not approximately equal: the generator promises
        // reproducibility, and "close enough" would hide a drifting RNG.
        CHECK(a.means[i].x == b.means[i].x);
        CHECK(a.means[i].y == b.means[i].y);
        CHECK(a.means[i].z == b.means[i].z);
        CHECK(a.scales[i].x == b.scales[i].x);
        CHECK(a.opacities[i] == b.opacities[i]);
        CHECK(a.rotations[i].w == b.rotations[i].w);
    }

    for (size_t k = 0; k < a.sh.size(); ++k) CHECK(a.sh[k] == b.sh[k]);
}

TEST_CASE("SplatGenerator: different seeds give different clouds") {

    const auto a = SplatGenerator::generate(options(1u));
    const auto b = SplatGenerator::generate(options(2u));

    int differing = 0;
    for (size_t i = 0; i < a.count(); ++i) {

        if (a.means[i].x != b.means[i].x) ++differing;
    }

    CHECK(differing > static_cast<int>(a.count()) / 2);
}

TEST_CASE("SplatGenerator: produces a valid, in-range cloud") {

    for (int degree = 0; degree <= 3; ++degree) {

        INFO("degree " << degree);
        const auto data = SplatGenerator::generate(options(7u, degree, 200));

        std::string why;
        REQUIRE(data.validate(&why));
        CHECK(why.empty());

        REQUIRE(data.count() == 200);
        REQUIRE(data.shDegree == degree);
        REQUIRE(data.sh.size() == 200 * static_cast<size_t>(splats::shCoeffCount(degree)) * 3);

        for (size_t i = 0; i < data.count(); ++i) {

            CHECK(data.opacities[i] >= 0.f);
            CHECK(data.opacities[i] <= 1.f);
            CHECK(data.scales[i].x > 0.f);
            CHECK(data.scales[i].y > 0.f);
            CHECK(data.scales[i].z > 0.f);
            CHECK(data.rotations[i].length() == Approx(1.f).margin(1e-5f));
        }
    }
}

TEST_CASE("SplatGenerator: unnormalised rotations are emitted, and normalise cleanly") {

    auto o = options(11u);
    o.unnormalizedRotations = true;

    auto data = SplatGenerator::generate(o);

    int nonUnit = 0;
    for (const auto& q : data.rotations) {

        if (std::abs(q.length() - 1.f) > 1e-3f) ++nonUnit;
    }
    CHECK(nonUnit > static_cast<int>(data.count()) / 2);

    data.normalizeRotations();
    for (const auto& q : data.rotations) CHECK(q.length() == Approx(1.f).margin(1e-5f));
}

TEST_CASE("SplatGenerator: degenerates are injected on request") {

    auto o = options(3u, 0, 200);
    o.includeDegenerates = true;

    const auto data = SplatGenerator::generate(o);

    int zeroScale = 0, zeroOpacity = 0;
    for (size_t i = 0; i < data.count(); ++i) {

        if (data.scales[i].x == 0.f) ++zeroScale;
        if (data.opacities[i] < 1e-4f) ++zeroOpacity;
    }

    CHECK(zeroScale > 0);
    CHECK(zeroOpacity > 0);
}

TEST_CASE("SplatData: a zero quaternion normalises to identity, not to NaN") {

    SplatData data;
    data.resize(1, 0);
    data.rotations[0].set(0, 0, 0, 0);

    data.normalizeRotations();

    CHECK(data.rotations[0].w == Approx(1.f));
    CHECK(std::isfinite(data.rotations[0].x));
}

TEST_CASE("SplatData: covariance of an unrotated splat is diag(scale^2)") {

    SplatData data;
    data.resize(1, 0);
    data.scales[0].set(2.f, 3.f, 4.f);
    data.rotations[0].set(0, 0, 0, 1);

    float cov[6];
    data.computeCovariance(0, cov);

    CHECK(cov[0] == Approx(4.f));  // xx
    CHECK(cov[1] == Approx(0.f).margin(1e-6f)); // xy
    CHECK(cov[2] == Approx(0.f).margin(1e-6f)); // xz
    CHECK(cov[3] == Approx(9.f));  // yy
    CHECK(cov[4] == Approx(0.f).margin(1e-6f)); // yz
    CHECK(cov[5] == Approx(16.f)); // zz
}

TEST_CASE("SplatData: covariance trace is rotation invariant") {

    // trace(R S S^T R^T) == trace(S S^T): a cheap but genuinely discriminating
    // check on the quaternion-to-matrix conversion, since almost any error in
    // it (transposed, wrong sign, w/x swapped) breaks orthogonality and moves
    // the trace.
    const float expected = 1.f + 4.f + 0.25f;

    for (const auto& q : {Quaternion(0, 0, 0, 1),
                          Quaternion(0.5f, 0.5f, 0.5f, 0.5f),
                          Quaternion(0.1826f, 0.3651f, 0.5477f, 0.7303f)}) {

        SplatData data;
        data.resize(1, 0);
        data.scales[0].set(1.f, 2.f, 0.5f);
        data.rotations[0] = q;
        data.normalizeRotations();

        float cov[6];
        data.computeCovariance(0, cov);

        INFO("q = " << q);
        CHECK(cov[0] + cov[3] + cov[5] == Approx(expected).epsilon(1e-4f));

        // Positive semi-definite: every diagonal entry non-negative and the
        // determinant of each 2x2 principal minor non-negative.
        CHECK(cov[0] >= 0.f);
        CHECK(cov[3] >= 0.f);
        CHECK(cov[5] >= 0.f);
        CHECK(cov[0] * cov[3] - cov[1] * cov[1] >= -1e-5f);
    }
}

TEST_CASE("SplatData: a 90-degree rotation about z swaps the x and y extents") {

    SplatData data;
    data.resize(1, 0);
    data.scales[0].set(3.f, 1.f, 1.f);
    // Quaternion for +90 degrees about z.
    data.rotations[0].set(0.f, 0.f, std::sin(math::PI / 4.f), std::cos(math::PI / 4.f));

    float cov[6];
    data.computeCovariance(0, cov);

    CHECK(cov[0] == Approx(1.f).margin(1e-4f));// xx: was the y axis
    CHECK(cov[3] == Approx(9.f).margin(1e-4f));// yy: was the x axis
    CHECK(cov[5] == Approx(1.f).margin(1e-4f));
}

TEST_CASE("SplatData: bounds cover the means dilated by 3 sigma") {

    SplatData data;
    data.resize(2, 0);
    data.means[0].set(0.f, 0.f, 0.f);
    data.means[1].set(10.f, 0.f, 0.f);
    data.scales[0].set(1.f, 1.f, 1.f);
    data.scales[1].set(0.5f, 0.5f, 2.f);
    data.normalizeRotations();

    const auto box = data.computeBounds(3.f);

    CHECK(box.min().x == Approx(-3.f));
    CHECK(box.max().x == Approx(16.f));// 10 + 3 * 2
    CHECK(box.min().y == Approx(-6.f));// splat 1 reaches -6 in y as well
}

TEST_CASE("SplatData: validate catches every length mismatch") {

    auto data = SplatGenerator::generate(options(5u, 1, 8));
    REQUIRE(data.validate());

    SECTION("scales") {

        data.scales.pop_back();
        std::string why;
        CHECK_FALSE(data.validate(&why));
        CHECK(why.find("scales") != std::string::npos);
    }

    SECTION("sh") {

        data.sh.pop_back();
        CHECK_FALSE(data.validate());
    }

    SECTION("extras") {

        data.extras["odd"] = std::vector<float>(3, 0.f);
        std::string why;
        CHECK_FALSE(data.validate(&why));
        CHECK(why.find("odd") != std::string::npos);
    }

    SECTION("degree") {

        data.shDegree = 9;
        CHECK_FALSE(data.validate());
    }
}

TEST_CASE("SplatData: an empty cloud is valid") {

    SplatData data;
    CHECK(data.validate());
    CHECK(data.count() == 0);
    CHECK(data.computeBounds().isEmpty());
}


// --------------------------------------------------------------------------
// removeOutliers
// --------------------------------------------------------------------------

namespace {

    // A tidy cloud: 200 splats scattered inside a unit-ish box, all about the
    // same size. Nothing in here is an outlier by any percentile rule.
    SplatData cleanCloud(int degree = 1) {

        SplatData data;
        data.resize(200, degree);

        for (size_t i = 0; i < data.count(); ++i) {

            const auto f = static_cast<float>(i);
            data.means[i].set(std::sin(f) * 0.5f, std::cos(f) * 0.5f, std::sin(f * 1.7f) * 0.5f);
            // Deliberately not identical: a rule that only survives a
            // perfectly uniform cloud is not a rule.
            data.scales[i].set(0.02f + 0.01f * (f / 200.f), 0.03f, 0.025f);
            data.rotations[i].set(0.f, 0.f, 0.f, 1.f);
            data.opacities[i] = 0.5f;
            data.setDcColor(i, Vector3{f / 200.f, 0.5f, 1.f - f / 200.f});
        }

        return data;
    }

}// namespace

TEST_CASE("SplatData::removeOutliers: a clean cloud loses nothing") {

    // The one-sided-guard doctrine: a cull that fires on input with no tail
    // is not conservative, it is destructive. This is the case that matters.
    auto data = cleanCloud();
    const auto before = data.count();

    CHECK(data.removeOutliers() == 0);
    CHECK(data.count() == before);
    CHECK(data.validate());
}

TEST_CASE("SplatData::removeOutliers: generated clouds lose nothing either") {

    // Same claim against the generator, including the awkward cases it can
    // produce (zero scale, near-zero opacity, extreme anisotropy).
    auto o = options(7u, 3, 512);
    o.anisotropy = 6.f;
    o.includeDegenerates = true;

    auto data = SplatGenerator::generate(o);
    const auto before = data.count();

    CHECK(data.removeOutliers() == 0);
    CHECK(data.count() == before);
}

TEST_CASE("SplatData::removeOutliers: exactly the planted outliers go") {

    auto data = cleanCloud(1);
    const auto total = data.count();

    // Tag every splat so survivors can be identified after the compaction.
    // The two planted outliers get negative tags.
    data.extras["tag"].resize(total);
    for (size_t i = 0; i < total; ++i) data.extras["tag"][i] = static_cast<float>(i);

    // A giant near-opaque smear at the centre of the scene, and a stray point
    // far outside it: one for the size rule, one for the distance rule. Both
    // sit in the MIDDLE of the cloud, so "order preserved" is a real claim
    // rather than an artefact of only ever removing from the end.
    data.scales[7].set(40.f, 40.f, 40.f);
    data.opacities[7] = 0.95f;
    data.extras["tag"][7] = -1.f;

    data.means[150].set(900.f, 0.f, 0.f);
    data.extras["tag"][150] = -2.f;

    REQUIRE(data.validate());
    CHECK(data.removeOutliers() == 2);
    CHECK(data.count() == total - 2);
    REQUIRE(data.validate());

    // Neither planted splat survived...
    const auto& tags = data.extras.at("tag");
    for (float t : tags) CHECK(t >= 0.f);

    // ...and the survivors are still in their original relative order.
    for (size_t i = 1; i < tags.size(); ++i) CHECK(tags[i] > tags[i - 1]);
}

TEST_CASE("SplatData::removeOutliers: the SH block travels with its splat") {

    SplatData data;
    data.resize(60, 2);

    for (size_t i = 0; i < data.count(); ++i) {

        data.means[i].set(0.1f * static_cast<float>(i % 5), 0.f, 0.f);
        data.scales[i].set(0.05f, 0.05f, 0.05f);
        data.rotations[i].set(0.f, 0.f, 0.f, 1.f);
        data.opacities[i] = 0.6f;

        // Every coefficient of splat i carries i, so a mis-strided compaction
        // shows up as a coefficient that belongs to somebody else.
        float* c = data.shAt(i);
        for (int k = 0; k < data.coeffCount() * 3; ++k) c[k] = static_cast<float>(i);
    }

    // One giant in the middle.
    data.scales[30].set(20.f, 20.f, 20.f);

    REQUIRE(data.removeOutliers() == 1);
    REQUIRE(data.count() == 59);
    REQUIRE(data.validate());

    for (size_t i = 0; i < data.count(); ++i) {

        const auto expected = static_cast<float>(i < 30 ? i : i + 1);
        const float* c = data.shAt(i);
        for (int k = 0; k < data.coeffCount() * 3; ++k) {

            REQUIRE(c[k] == Approx(expected));
        }
    }
}

TEST_CASE("SplatData::removeOutliers: an empty cloud is a no-op") {

    SplatData data;
    CHECK(data.removeOutliers() == 0);
    CHECK(data.validate());
}

TEST_CASE("SplatData::removeOutliers: the rule is scale-free") {

    // The same cloud measured in different units must lose the same splats.
    auto build = [](float unit) {
        auto d = cleanCloud(0);
        for (auto& m : d.means) m.multiplyScalar(unit);
        for (auto& s : d.scales) s.multiplyScalar(unit);
        d.means.emplace_back(0.f, 0.f, 0.f);
        d.scales.emplace_back(40.f * unit, 40.f * unit, 40.f * unit);
        d.rotations.emplace_back(0.f, 0.f, 0.f, 1.f);
        d.opacities.push_back(0.9f);
        d.sh.insert(d.sh.end(), 3, 0.5f);
        return d;
    };

    auto metres = build(1.f);
    auto millimetres = build(1000.f);

    CHECK(metres.removeOutliers() == 1);
    CHECK(millimetres.removeOutliers() == 1);
}


// --------------------------------------------------------------------------
// reorderMorton
//
// The permutation is documented as NEW INDEX -> OLD INDEX: after the call the
// splat at i is the one that used to be at perm[i]. Every test below reads it
// in that direction, so a silent flip of the convention fails here first.
//
// The key interleaves x -> y -> z with x MOST significant inside each 3-bit
// group. The cube-corner case hand-computes the consequence.
// --------------------------------------------------------------------------

namespace {

    // A cloud where every array is a distinct function of the splat index, so
    // an attribute that lands beside the wrong splat is visible rather than
    // plausible. The positions are deliberately not a lattice walk: input that
    // already arrived in Morton order would let the identity permutation pass.
    SplatData taggedCloud(size_t n, int degree) {

        SplatData data;
        data.resize(n, degree);
        data.extras["confidence"].assign(n, 0.f);
        data.extras["tag"].assign(n, 0.f);

        for (size_t i = 0; i < n; ++i) {

            const auto f = static_cast<float>(i);
            data.means[i].set(std::sin(f * 1.3f), std::cos(f * 0.7f), std::sin(f * 2.1f));
            data.scales[i].set(0.01f + f * 1e-4f, 0.02f, 0.03f);
            data.rotations[i].set(0.f, 0.f, 0.f, 1.f);
            data.opacities[i] = 0.25f + 0.5f * f / static_cast<float>(n);

            data.extras["tag"][i] = f;
            data.extras["confidence"][i] = 1000.f - f;

            float* c = data.shAt(i);
            for (int k = 0; k < data.coeffCount() * 3; ++k) c[k] = f * 100.f + static_cast<float>(k);
        }

        return data;
    }

    bool isIdentity(const std::vector<std::uint32_t>& perm) {

        for (size_t i = 0; i < perm.size(); ++i) {

            if (perm[i] != static_cast<std::uint32_t>(i)) return false;
        }
        return true;
    }

}// namespace

TEST_CASE("SplatData::reorderMorton: the permutation is a bijection and every tuple survives") {

    const auto before = taggedCloud(300, 3);
    auto data = before;

    const auto perm = data.reorderMorton();

    REQUIRE(perm.size() == before.count());
    REQUIRE(data.count() == before.count());
    REQUIRE(data.validate());

    // Sorting the permutation yields 0..n-1: nothing dropped, nothing doubled.
    auto sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (size_t i = 0; i < sorted.size(); ++i) REQUIRE(sorted[i] == static_cast<std::uint32_t>(i));

    // ...and it is not the trivial answer.
    REQUIRE_FALSE(isIdentity(perm));

    for (size_t i = 0; i < data.count(); ++i) {

        const size_t old = perm[i];
        INFO("new " << i << " <- old " << old);

        // The tag is the fingerprint: it says which splat this used to be, and
        // the permutation has to agree with it.
        REQUIRE(data.extras.at("tag")[i] == Approx(static_cast<float>(old)));
        REQUIRE(data.extras.at("confidence")[i] == Approx(before.extras.at("confidence")[old]));

        REQUIRE(data.means[i].x == before.means[old].x);
        REQUIRE(data.means[i].y == before.means[old].y);
        REQUIRE(data.means[i].z == before.means[old].z);
        REQUIRE(data.scales[i].x == before.scales[old].x);
        REQUIRE(data.scales[i].y == before.scales[old].y);
        REQUIRE(data.scales[i].z == before.scales[old].z);
        REQUIRE(data.opacities[i] == before.opacities[old]);
        REQUIRE(data.rotations[i].w == before.rotations[old].w);

        const float* got = data.shAt(i);
        const float* want = before.shAt(old);
        for (int k = 0; k < data.coeffCount() * 3; ++k) REQUIRE(got[k] == want[k]);
    }
}

TEST_CASE("SplatData::reorderMorton: the corners of a cube come out in Z-order") {

    // Eight splats at the corners of the unit cube, handed in scrambled order.
    // One bit per axis is all this case exercises, and with x most significant
    // the key is (x, y, z) read as a 3-bit number — so the Morton sequence is
    // plain lexicographic:
    //
    //   (0,0,0) (0,0,1) (0,1,0) (0,1,1) (1,0,0) (1,0,1) (1,1,0) (1,1,1)
    //
    // The robust bounds do not disturb it: four corners sit at 0 and four at 1
    // on every axis, so the percentile interval is [0, 1] whichever order they
    // arrived in, and the two ends quantise to cells 0 and 1023.
    const std::vector<Vector3> scrambled{
            {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}, {0, 0, 0}, {1, 0, 0}, {0, 1, 0}};

    const std::vector<Vector3> expected{
            {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1}, {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}};

    SplatData data;
    data.resize(scrambled.size(), 0);
    for (size_t i = 0; i < scrambled.size(); ++i) {

        data.means[i] = scrambled[i];
        data.scales[i].set(0.1f, 0.1f, 0.1f);
    }

    const auto perm = data.reorderMorton();
    REQUIRE(data.validate());

    for (size_t i = 0; i < expected.size(); ++i) {

        INFO("position " << i);
        CHECK(data.means[i].x == Approx(expected[i].x));
        CHECK(data.means[i].y == Approx(expected[i].y));
        CHECK(data.means[i].z == Approx(expected[i].z));

        // The same claim through the permutation, which is the part a caller
        // remapping its own arrays depends on.
        CHECK(scrambled[perm[i]].x == Approx(expected[i].x));
        CHECK(scrambled[perm[i]].y == Approx(expected[i].y));
        CHECK(scrambled[perm[i]].z == Approx(expected[i].z));
    }
}

TEST_CASE("SplatData::reorderMorton: one cell keeps file order, and a second call is a no-op") {

    auto data = taggedCloud(200, 1);

    // Twelve splats on exactly the same point, planted through the middle of
    // the cloud. They share a cell, so nothing but the sort's stability decides
    // their order — and it has to leave it alone.
    for (size_t i = 40; i < 52; ++i) data.means[i].set(0.25f, -0.5f, 0.75f);

    const auto perm = data.reorderMorton();
    REQUIRE(data.validate());
    REQUIRE_FALSE(isIdentity(perm));

    std::vector<float> duplicates;
    for (size_t i = 0; i < data.count(); ++i) {

        const float tag = data.extras.at("tag")[i];
        if (tag >= 40.f && tag < 52.f) duplicates.push_back(tag);
    }

    REQUIRE(duplicates.size() == 12);
    for (size_t i = 1; i < duplicates.size(); ++i) CHECK(duplicates[i] > duplicates[i - 1]);

    // Idempotent. The cloud is already Morton-ordered, the bounds are exact
    // order statistics of the same (unchanged) set of positions, and the sort
    // is stable — so the second call has nothing left to do.
    const auto settled = data;
    const auto again = data.reorderMorton();

    CHECK(isIdentity(again));

    for (size_t i = 0; i < data.count(); ++i) {

        INFO("splat " << i);
        REQUIRE(data.means[i].x == settled.means[i].x);
        REQUIRE(data.means[i].y == settled.means[i].y);
        REQUIRE(data.means[i].z == settled.means[i].z);
        REQUIRE(data.opacities[i] == settled.opacities[i]);
        REQUIRE(data.extras.at("tag")[i] == settled.extras.at("tag")[i]);
    }

    for (size_t k = 0; k < data.sh.size(); ++k) REQUIRE(data.sh[k] == settled.sh[k]);
}

TEST_CASE("SplatData::reorderMorton: one stray does not flatten the grid") {

    // 200 splats filling a unit cube, plus a single stray a thousand units out.
    // The cluster is handed in DESCENDING x, which is emphatically not Morton
    // order, so a grid that resolves the cluster has to move nearly everything.
    //
    // Why this catches naive min/max bounds: those would make the interval
    // ~1000 units wide, one cell ~0.98 units, and the entire cluster would fall
    // into one or two cells per axis — at most eight distinct keys between all
    // 200 splats. The stable sort would then hand back file order almost
    // untouched. Percentile bounds put the interval on the cluster's own unit
    // span instead, ~1024 cells across it, and the cluster resolves into
    // hundreds of keys.
    //
    // The assertion counts maximal ascending runs of the permutation rather
    // than keys, which reorderMorton does not expose. A stable sort emits one
    // ascending run per key group, and adjacent groups can only merge runs, so
    // runs <= distinct keys: more than 100 runs proves more than 100 keys, and
    // the naive-bounds cloud could not produce more than about 9.
    //
    // That last claim is asserted rather than reasoned about, because a
    // boundsPercentile of 1 IS min/max bounds: the percentile of rank 1 is the
    // maximum and the percentile of rank 0 is the minimum. The same cloud is
    // run both ways below and the difference is the whole point of the design.
    constexpr size_t CLUSTER = 200;

    auto strayCloud = [] {
        SplatData d;
        d.resize(CLUSTER + 1, 0);

        for (size_t i = 0; i < CLUSTER; ++i) {

            const auto f = static_cast<float>(i);
            d.means[i].set(1.f - f / static_cast<float>(CLUSTER),
                           std::fmod(f * 0.618034f, 1.f),
                           std::fmod(f * 0.381966f, 1.f));
            d.scales[i].set(0.005f, 0.005f, 0.005f);
        }

        d.means[CLUSTER].set(1000.f, 1000.f, 1000.f);
        d.scales[CLUSTER].set(0.005f, 0.005f, 0.005f);
        return d;
    };

    auto ascendingRuns = [](const std::vector<std::uint32_t>& p) {
        size_t runs = p.empty() ? 0 : 1;
        for (size_t i = 1; i < p.size(); ++i) {

            if (p[i] < p[i - 1]) ++runs;
        }
        return runs;
    };

    auto data = strayCloud();
    const auto perm = data.reorderMorton();
    REQUIRE(data.validate());

    const size_t runs = ascendingRuns(perm);
    INFO("robust-bounds ascending runs " << runs);
    CHECK(runs > 100);

    // The counterfactual, on the same cloud: min/max bounds hand the stray the
    // whole grid and the cluster collapses into a corner of it.
    auto naive = strayCloud();
    const size_t naiveRuns = ascendingRuns(naive.reorderMorton(1.f));

    INFO("min/max-bounds ascending runs " << naiveRuns);
    CHECK(naiveRuns < 10);
    CHECK(runs > naiveRuns * 10);

    // The stray clamps into the top cell on all three axes, which is the
    // largest key there is, so it ends up at one end rather than scattered
    // through the middle of the cloud.
    CHECK(perm.back() == static_cast<std::uint32_t>(CLUSTER));
    CHECK(data.means[data.count() - 1].x == Approx(1000.f));
}

TEST_CASE("SplatData::reorderMorton: SH degree 3 and extras follow their splat") {

    // The spot check the bijection test generalises: a full 48-float degree-3
    // block, read coefficient by coefficient, against named extras.
    const auto before = taggedCloud(64, 3);
    REQUIRE(before.coeffCount() == 16);
    REQUIRE(before.sh.size() == 64 * 16 * 3);

    auto data = before;
    const auto perm = data.reorderMorton();

    REQUIRE(data.sh.size() == before.sh.size());
    REQUIRE(data.extras.size() == 2);
    REQUIRE(data.validate());

    for (size_t i : {size_t{0}, size_t{1}, data.count() / 2, data.count() - 2, data.count() - 1}) {

        const size_t old = perm[i];
        INFO("new " << i << " <- old " << old);

        // Every coefficient of the block, in order — a mis-strided permute
        // shows up as a coefficient belonging to somebody else.
        const float* c = data.shAt(i);
        for (int k = 0; k < 16 * 3; ++k) {

            REQUIRE(c[k] == Approx(static_cast<float>(old) * 100.f + static_cast<float>(k)));
        }

        REQUIRE(data.extras.at("tag")[i] == Approx(static_cast<float>(old)));
        REQUIRE(data.extras.at("confidence")[i] == Approx(1000.f - static_cast<float>(old)));
    }
}

TEST_CASE("SplatData::reorderMorton: degenerate clouds are no-ops") {

    SECTION("empty") {

        SplatData data;
        CHECK(data.reorderMorton().empty());
        CHECK(data.validate());
    }

    SECTION("one splat") {

        auto data = taggedCloud(1, 2);
        const auto perm = data.reorderMorton();

        REQUIRE(perm.size() == 1);
        CHECK(perm[0] == 0u);
        CHECK(data.validate());
    }

    SECTION("every splat on the same point") {

        // No spread on any axis: the grid collapses, every key is 0, and the
        // stable sort has to hand back file order rather than something
        // arbitrary or a division by zero.
        auto data = taggedCloud(32, 1);
        for (auto& m : data.means) m.set(2.f, 3.f, 4.f);

        const auto perm = data.reorderMorton();

        CHECK(isIdentity(perm));
        for (size_t i = 0; i < data.count(); ++i) {

            CHECK(data.extras.at("tag")[i] == Approx(static_cast<float>(i)));
        }
    }

    SECTION("a non-finite coordinate does not reach the cast") {

        // Same doctrine as the depth sort: a NaN quantises to cell 0 instead
        // of to whatever a float-to-int conversion happens to produce, and the
        // cloud comes back valid with every tuple intact.
        auto data = taggedCloud(48, 1);
        data.means[10].x = std::numeric_limits<float>::quiet_NaN();
        data.means[20].y = std::numeric_limits<float>::infinity();

        const auto perm = data.reorderMorton();

        REQUIRE(data.validate());
        auto sorted = perm;
        std::sort(sorted.begin(), sorted.end());
        for (size_t i = 0; i < sorted.size(); ++i) REQUIRE(sorted[i] == static_cast<std::uint32_t>(i));

        for (size_t i = 0; i < data.count(); ++i) {

            const auto old = static_cast<float>(perm[i]);
            REQUIRE(data.extras.at("tag")[i] == Approx(old));
            REQUIRE(data.opacities[i] == Approx(0.25f + 0.5f * old / 48.f));
        }
    }
}


TEST_CASE("splats::medianNeighbourSpacing: a regular grid reports its pitch") {

    SECTION("a 10x10x10 lattice at pitch 0.25") {

        std::vector<Vector3> pts;
        for (int x = 0; x < 10; ++x)
            for (int y = 0; y < 10; ++y)
                for (int z = 0; z < 10; ++z) pts.emplace_back(x * 0.25f, y * 0.25f, z * 0.25f);

        CHECK(splats::medianNeighbourSpacing(pts) == Approx(0.25f).margin(1e-5f));
    }

    SECTION("a flat 30x30 sheet at pitch 1 (one extent is zero)") {

        std::vector<Vector3> pts;
        for (int x = 0; x < 30; ++x)
            for (int y = 0; y < 30; ++y) pts.emplace_back(static_cast<float>(x), static_cast<float>(y), 0.f);

        CHECK(splats::medianNeighbourSpacing(pts) == Approx(1.f).margin(1e-5f));
    }

    SECTION("a jittered lattice lands near the pitch") {

        std::vector<Vector3> pts;
        uint32_t s = 7u;
        auto jitter = [&] {
            s = s * 1664525u + 1013904223u;
            return (static_cast<float>(s >> 8) / 16777216.f - 0.5f) * 0.2f;
        };
        for (int x = 0; x < 12; ++x)
            for (int y = 0; y < 12; ++y)
                for (int z = 0; z < 12; ++z)
                    pts.emplace_back(x + jitter(), y + jitter(), z + jitter());

        const float spacing = splats::medianNeighbourSpacing(pts);
        CHECK(spacing > 0.8f);
        CHECK(spacing < 1.05f);
    }

    SECTION("the sample stride does not change a uniform answer") {

        std::vector<Vector3> pts;
        for (int x = 0; x < 20; ++x)
            for (int y = 0; y < 20; ++y)
                for (int z = 0; z < 20; ++z) pts.emplace_back(x * 2.f, y * 2.f, z * 2.f);

        CHECK(splats::medianNeighbourSpacing(pts, 100) == Approx(2.f).margin(1e-5f));
        CHECK(splats::medianNeighbourSpacing(pts, 1000000) == Approx(2.f).margin(1e-5f));
    }

    SECTION("degenerate inputs") {

        CHECK(splats::medianNeighbourSpacing({}) == 0.f);
        CHECK(splats::medianNeighbourSpacing({Vector3{1.f, 2.f, 3.f}}) == 0.f);
        CHECK(splats::medianNeighbourSpacing({Vector3{0.f, 0.f, 0.f}, Vector3{0.f, 0.f, 0.f}}) == 0.f);
        CHECK(splats::medianNeighbourSpacing({Vector3{0.f, 0.f, 0.f}, Vector3{3.f, 0.f, 0.f}}) ==
              Approx(3.f).margin(1e-5f));

        // A non-finite point is skipped, not counted.
        const float nan = std::numeric_limits<float>::quiet_NaN();
        CHECK(splats::medianNeighbourSpacing({Vector3{0.f, 0.f, 0.f}, Vector3{nan, 0.f, 0.f},
                                              Vector3{0.f, 5.f, 0.f}}) == Approx(5.f).margin(1e-5f));
    }
}
