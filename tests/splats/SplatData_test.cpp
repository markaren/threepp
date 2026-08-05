// The splat data model and its generator: determinism, activations, the
// covariance, bounds and validation.

#include "threepp/math/MathUtils.hpp"
#include "threepp/splats/SplatData.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
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
