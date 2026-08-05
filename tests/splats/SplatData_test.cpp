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
