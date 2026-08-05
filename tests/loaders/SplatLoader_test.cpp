// SplatLoader: round trips, the three activation conventions, extras, errors.

#include "splat_ply_writer.hpp"

#include "threepp/loaders/SplatLoader.hpp"
#include "threepp/splats/SplatData.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace threepp;
using Catch::Approx;

namespace {

    SplatData load(const std::string& buffer) {

        std::istringstream in(buffer, std::ios::binary);
        return SplatLoader::parsePly(in);
    }

    SplatData makeCloud(int degree, size_t count = 64) {

        SplatGenerator::Options options;
        options.count = count;
        options.shDegree = degree;
        options.seed = 20260805u;
        // Degenerates are deliberately absent: the format stores log(scale),
        // so a zero-scale splat is not representable. See splat_ply_writer.hpp.
        options.includeDegenerates = false;
        return SplatGenerator::generate(options);
    }

    void compare(const SplatData& a, const SplatData& b, float tol = 1e-5f) {

        REQUIRE(a.count() == b.count());
        REQUIRE(a.shDegree == b.shDegree);

        for (size_t i = 0; i < a.count(); ++i) {

            INFO("splat " << i);
            CHECK(a.means[i].x == Approx(b.means[i].x).margin(tol));
            CHECK(a.means[i].y == Approx(b.means[i].y).margin(tol));
            CHECK(a.means[i].z == Approx(b.means[i].z).margin(tol));

            CHECK(a.scales[i].x == Approx(b.scales[i].x).epsilon(tol));
            CHECK(a.scales[i].y == Approx(b.scales[i].y).epsilon(tol));
            CHECK(a.scales[i].z == Approx(b.scales[i].z).epsilon(tol));

            CHECK(a.opacities[i] == Approx(b.opacities[i]).margin(tol));

            // Quaternions are compared up to sign: q and -q are the same
            // rotation, and nothing in the pipeline promises which one comes
            // back. (The writer does not flip it, so this is belt and braces.)
            const auto& qa = a.rotations[i];
            const auto& qb = b.rotations[i];
            const float dot = qa.x * qb.x + qa.y * qb.y + qa.z * qb.z + qa.w * qb.w;
            CHECK(std::abs(dot) == Approx(1.f).margin(1e-4f));

            const float* ca = a.shAt(i);
            const float* cb = b.shAt(i);
            for (int k = 0; k < a.coeffCount() * 3; ++k) {

                INFO("sh coefficient slot " << k);
                CHECK(ca[k] == Approx(cb[k]).margin(tol));
            }
        }
    }

}// namespace


TEST_CASE("SplatLoader: degree-3 cloud survives a PLY round trip") {

    // The one that matters: with 45 f_rest properties, a channel-major /
    // coefficient-major mix-up is silent. Nothing else in the suite catches it.
    const auto original = makeCloud(3);
    const auto reloaded = load(splattest::writeSplatPly(original));

    REQUIRE(reloaded.shDegree == 3);
    REQUIRE(reloaded.coeffCount() == 16);
    compare(original, reloaded);
}

TEST_CASE("SplatLoader: degree 0, 1 and 2 clouds survive a PLY round trip") {

    for (int degree : {0, 1, 2}) {

        INFO("degree " << degree);
        const auto original = makeCloud(degree);
        const auto reloaded = load(splattest::writeSplatPly(original));

        REQUIRE(reloaded.shDegree == degree);
        compare(original, reloaded);
    }
}

TEST_CASE("SplatLoader: f_rest is read channel-major") {

    // Hand-built, one splat, degree 1: three higher-order coefficients per
    // channel. Distinct values everywhere, so any transposition shows.
    SplatData data;
    data.resize(1, 1);
    data.rotations[0].set(0, 0, 0, 1);
    data.scales[0].set(1, 1, 1);
    data.opacities[0] = 0.5f;

    float* c = data.shAt(0);
    for (int k = 0; k < 4; ++k) {

        c[k * 3 + 0] = 100.f + static_cast<float>(k);// red
        c[k * 3 + 1] = 200.f + static_cast<float>(k);// green
        c[k * 3 + 2] = 300.f + static_cast<float>(k);// blue
    }

    const auto buffer = splattest::writeSplatPly(data);
    const auto reloaded = load(buffer);

    const float* r = reloaded.shAt(0);
    CHECK(r[0] == Approx(100.f));
    CHECK(r[1] == Approx(200.f));
    CHECK(r[2] == Approx(300.f));

    for (int k = 1; k < 4; ++k) {

        INFO("coefficient " << k);
        CHECK(r[k * 3 + 0] == Approx(100.f + static_cast<float>(k)));
        CHECK(r[k * 3 + 1] == Approx(200.f + static_cast<float>(k)));
        CHECK(r[k * 3 + 2] == Approx(300.f + static_cast<float>(k)));
    }

    // And prove the buffer really is channel-major on disk, so the test above
    // is testing the reorder rather than the writer and loader agreeing on a
    // shared mistake. Record layout for degree 1 is 23 floats:
    //   0..2 xyz | 3..5 f_dc | 6..14 f_rest | 15 opacity | 16..18 scale | 19..22 rot
    constexpr int RECORD_FLOATS = 3 + 3 + 9 + 1 + 3 + 4;
    const size_t body = buffer.size() - static_cast<size_t>(RECORD_FLOATS) * 4;
    auto diskFloat = [&](int index) {
        float v;
        std::memcpy(&v, buffer.data() + body + static_cast<size_t>(index) * 4, 4);
        return v;
    };
    CHECK(diskFloat(3) == Approx(100.f)); // f_dc_0
    CHECK(diskFloat(6) == Approx(101.f)); // f_rest_0 = red, coefficient 1
    CHECK(diskFloat(7) == Approx(102.f)); // f_rest_1 = red, coefficient 2
    CHECK(diskFloat(8) == Approx(103.f)); // f_rest_2 = red, coefficient 3
    CHECK(diskFloat(9) == Approx(201.f)); // f_rest_3 = GREEN, coefficient 1
    CHECK(diskFloat(12) == Approx(301.f));// f_rest_6 = BLUE, coefficient 1
}

TEST_CASE("SplatLoader: opacity, scale and rotation activations are applied") {

    // Hand-built header + body, exact values in, exact values expected out.
    // Nothing here comes from the writer, so this pins the loader on its own.
    std::string ply =
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 1\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
            "property float opacity\n"
            "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
            "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
            "end_header\n";

    using splattest::putFloat;
    putFloat(ply, 1.f);
    putFloat(ply, 2.f);
    putFloat(ply, 3.f);
    putFloat(ply, 0.1f);
    putFloat(ply, 0.2f);
    putFloat(ply, 0.3f);
    putFloat(ply, 0.f);   // logit 0 -> sigmoid 0.5
    putFloat(ply, 0.f);   // log 0 -> exp 1
    putFloat(ply, -1.f);  // log -1 -> exp ~0.3679
    putFloat(ply, 1.f);   // log 1 -> exp ~2.7183
    putFloat(ply, 2.f);   // rot_0 is W ...
    putFloat(ply, 0.f);   // ... then X
    putFloat(ply, 0.f);   // Y
    putFloat(ply, 0.f);   // Z

    const auto data = load(ply);
    REQUIRE(data.count() == 1);
    REQUIRE(data.shDegree == 0);

    CHECK(data.means[0].x == Approx(1.f));
    CHECK(data.means[0].y == Approx(2.f));
    CHECK(data.means[0].z == Approx(3.f));

    // Gotcha 2a: opacity is a logit.
    CHECK(data.opacities[0] == Approx(0.5f));

    // Gotcha 2b: scale is a log.
    CHECK(data.scales[0].x == Approx(1.f));
    CHECK(data.scales[0].y == Approx(std::exp(-1.f)));
    CHECK(data.scales[0].z == Approx(std::exp(1.f)));

    // Gotcha 2c: rot_0 is w, and the file's quaternion is unnormalised —
    // (w=2, x=y=z=0) must come back as identity, not as w=2.
    CHECK(data.rotations[0].w == Approx(1.f));
    CHECK(data.rotations[0].x == Approx(0.f));
    CHECK(data.rotations[0].y == Approx(0.f));
    CHECK(data.rotations[0].z == Approx(0.f));

    // f_dc is stored raw: no 0.5 + C0*dc baked in at import.
    CHECK(data.shAt(0)[0] == Approx(0.1f));
    CHECK(data.shAt(0)[1] == Approx(0.2f));
    CHECK(data.shAt(0)[2] == Approx(0.3f));
}

TEST_CASE("SplatLoader: unnormalised quaternions come back unit") {

    auto original = makeCloud(0, 16);

    splattest::PlyWriteOptions options;
    options.quatScale = 7.5f;// what the optimiser leaves behind

    const auto reloaded = load(splattest::writeSplatPly(original, options));

    for (size_t i = 0; i < reloaded.count(); ++i) {

        INFO("splat " << i);
        CHECK(reloaded.rotations[i].length() == Approx(1.f).margin(1e-5f));
    }

    // Same rotation as the (already unit) original.
    compare(original, reloaded);
}

TEST_CASE("SplatLoader: normals are tolerated and ignored") {

    const auto original = makeCloud(1, 16);

    splattest::PlyWriteOptions options;
    options.withNormals = true;

    const auto reloaded = load(splattest::writeSplatPly(original, options));

    compare(original, reloaded);
    CHECK(reloaded.extras.empty());// nx/ny/nz are dropped, not kept as extras
}

TEST_CASE("SplatLoader: unknown properties are preserved, not fatal") {

    auto original = makeCloud(0, 16);

    // The robotics escape hatch: a downstream tool bolted a semantic label and
    // a confidence onto every splat.
    std::vector<float> labels(original.count());
    std::vector<float> confidence(original.count());
    for (size_t i = 0; i < original.count(); ++i) {

        labels[i] = static_cast<float>(i % 5);
        confidence[i] = 0.01f * static_cast<float>(i);
    }
    original.extras["semantic_label"] = labels;
    original.extras["confidence"] = confidence;

    const auto reloaded = load(splattest::writeSplatPly(original));

    REQUIRE(reloaded.extras.size() == 2);
    REQUIRE(reloaded.extras.count("semantic_label") == 1);
    REQUIRE(reloaded.extras.count("confidence") == 1);

    for (size_t i = 0; i < reloaded.count(); ++i) {

        CHECK(reloaded.extras.at("semantic_label")[i] == Approx(labels[i]));
        CHECK(reloaded.extras.at("confidence")[i] == Approx(confidence[i]));
    }

    compare(original, reloaded);
}

TEST_CASE("SplatLoader: stride is taken from the header, not assumed") {

    // Same cloud, three different property layouts. If the parser had a
    // hardcoded stride, only one of them could work.
    const auto original = makeCloud(2, 24);

    auto withExtras = original;
    withExtras.extras["junk_a"] = std::vector<float>(original.count(), 1.f);
    withExtras.extras["junk_b"] = std::vector<float>(original.count(), 2.f);

    splattest::PlyWriteOptions normals;
    normals.withNormals = true;

    compare(original, load(splattest::writeSplatPly(original)));
    compare(original, load(splattest::writeSplatPly(original, normals)));
    compare(original, load(splattest::writeSplatPly(withExtras)));
    compare(original, load(splattest::writeSplatPly(withExtras, normals)));
}

TEST_CASE("SplatLoader: loads from a file path") {

    const auto original = makeCloud(1, 32);

    const auto path = std::filesystem::temp_directory_path() / "threepp_splat_roundtrip.ply";
    splattest::writeSplatPlyFile(original, path);

    const auto reloaded = SplatLoader::loadPly(path);
    compare(original, reloaded);

    std::filesystem::remove(path);
}

TEST_CASE("SplatLoader: rejects what it cannot represent") {

    SECTION("not a PLY") {

        CHECK_THROWS_AS(load("not a ply at all\n"), std::runtime_error);
    }

    SECTION("ASCII PLY") {

        CHECK_THROWS_AS(load("ply\nformat ascii 1.0\nelement vertex 1\nend_header\n"),
                        std::runtime_error);
    }

    SECTION("missing a required property") {

        CHECK_THROWS_AS(load("ply\nformat binary_little_endian 1.0\n"
                             "element vertex 1\n"
                             "property float x\nproperty float y\nproperty float z\n"
                             "end_header\n"),
                        std::runtime_error);
    }

    SECTION("an f_rest count matching no SH degree") {

        std::string ply =
                "ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n";
        for (int i = 0; i < 7; ++i) ply += "property float f_rest_" + std::to_string(i) + "\n";
        ply += "property float opacity\n"
               "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
               "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
               "end_header\n";

        CHECK_THROWS_AS(load(ply), std::runtime_error);
    }

    SECTION("a non-float required property") {

        CHECK_THROWS_AS(load("ply\nformat binary_little_endian 1.0\n"
                             "element vertex 1\n"
                             "property double x\nproperty double y\nproperty double z\n"
                             "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
                             "property float opacity\n"
                             "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
                             "property float rot_0\nproperty float rot_1\n"
                             "property float rot_2\nproperty float rot_3\n"
                             "end_header\n"),
                        std::runtime_error);
    }

    SECTION("a truncated body") {

        auto ply = splattest::writeSplatPly(makeCloud(0, 8));
        ply.resize(ply.size() - 17);

        CHECK_THROWS_AS(load(ply), std::runtime_error);
    }
}
