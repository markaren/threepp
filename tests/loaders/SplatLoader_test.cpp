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
#include <algorithm>
#include <cstdint>
#include <iterator>

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


// ---------------------------------------------------------------------------
// Colour-only point clouds (SplatLoader::loadPointCloudPly)
// ---------------------------------------------------------------------------

namespace {

    struct PointSpec {
        float x, y, z;
        unsigned char r = 255, g = 255, b = 255;
        float nx = 0.f, ny = 0.f, nz = 0.f;
    };

    struct PointPlyOptions {
        bool color = true;
        bool normals = false;
        bool ascii = false;
        bool bigEndian = false;
        bool faces = false;         // a mesh: one triangle after the vertices
        bool doubles = false;       // x/y/z as double
        bool floatColor = false;    // red/green/blue as float in [0, 255]
        bool intensity = false;     // ushort intensity instead of a colour
        bool leadingElement = false;// an unrelated element before the vertices
    };

    void putBytes(std::string& out, const void* p, size_t n, bool bigEndian) {

        const auto* b = static_cast<const unsigned char*>(p);
        if (bigEndian)
            for (size_t i = n; i-- > 0;) out.push_back(static_cast<char>(b[i]));
        else
            out.append(reinterpret_cast<const char*>(b), n);
    }

    std::string pointCloudPly(const std::vector<PointSpec>& pts, const PointPlyOptions& o = {}) {

        std::ostringstream h;
        h << "ply\nformat "
          << (o.ascii ? "ascii" : o.bigEndian ? "binary_big_endian" : "binary_little_endian")
          << " 1.0\ncomment made by SplatLoader_test\n";
        if (o.leadingElement) h << "element camera 1\nproperty float fx\nproperty float fy\n";
        h << "element vertex " << pts.size() << "\n";
        const char* ct = o.doubles ? "double" : "float";
        h << "property " << ct << " x\nproperty " << ct << " y\nproperty " << ct << " z\n";
        if (o.normals) h << "property float nx\nproperty float ny\nproperty float nz\n";
        if (o.color) {
            const char* cc = o.floatColor ? "float" : "uchar";
            h << "property " << cc << " red\nproperty " << cc << " green\nproperty " << cc << " blue\n";
        }
        if (o.intensity) h << "property ushort intensity\n";
        h << "property float confidence\n";
        if (o.faces) h << "element face 1\nproperty list uchar int vertex_indices\n";
        h << "end_header\n";

        std::string out = h.str();

        if (o.ascii) {
            if (o.leadingElement) out += "1.5 2.5\n";
            for (size_t i = 0; i < pts.size(); ++i) {
                const auto& p = pts[i];
                std::ostringstream l;
                l << p.x << " " << p.y << " " << p.z;
                if (o.normals) l << " " << p.nx << " " << p.ny << " " << p.nz;
                if (o.color) {
                    if (o.floatColor) l << " " << float(p.r) << " " << float(p.g) << " " << float(p.b);
                    else l << " " << int(p.r) << " " << int(p.g) << " " << int(p.b);
                }
                if (o.intensity) l << " " << (i * 100);
                l << " " << 0.5f * static_cast<float>(i) << "\n";
                out += l.str();
            }
            if (o.faces) out += "3 0 1 2\n";
            return out;
        }

        const bool be = o.bigEndian;
        const auto putF = [&](float v) { putBytes(out, &v, 4, be); };
        if (o.leadingElement) {
            putF(1.5f);
            putF(2.5f);
        }
        for (size_t i = 0; i < pts.size(); ++i) {
            const auto& p = pts[i];
            if (o.doubles) {
                const double d[3] = {p.x, p.y, p.z};
                for (double v : d) putBytes(out, &v, 8, be);
            } else {
                putF(p.x);
                putF(p.y);
                putF(p.z);
            }
            if (o.normals) {
                putF(p.nx);
                putF(p.ny);
                putF(p.nz);
            }
            if (o.color) {
                if (o.floatColor) {
                    putF(float(p.r));
                    putF(float(p.g));
                    putF(float(p.b));
                } else {
                    out.push_back(static_cast<char>(p.r));
                    out.push_back(static_cast<char>(p.g));
                    out.push_back(static_cast<char>(p.b));
                }
            }
            if (o.intensity) {
                const uint16_t v = static_cast<uint16_t>(i * 100);
                putBytes(out, &v, 2, be);
            }
            putF(0.5f * static_cast<float>(i));
        }
        if (o.faces) {
            out.push_back(3);
            const int32_t idx[3] = {0, 1, 2};
            for (int32_t v : idx) putBytes(out, &v, 4, be);
        }
        return out;
    }

    std::vector<PointSpec> lattice(int n, float pitch) {

        std::vector<PointSpec> pts;
        for (int x = 0; x < n; ++x)
            for (int y = 0; y < n; ++y)
                for (int z = 0; z < n; ++z) {
                    PointSpec p{x * pitch, y * pitch, z * pitch};
                    p.r = static_cast<unsigned char>(255 * x / std::max(1, n - 1));
                    p.g = 0;
                    p.b = static_cast<unsigned char>(255 * z / std::max(1, n - 1));
                    pts.push_back(p);
                }
        return pts;
    }

    SplatData loadPoints(const std::string& buffer, const SplatLoader::PointCloudOptions& o = {},
                         SplatLoader::PointCloudInfo* info = nullptr) {

        std::istringstream in(buffer, std::ios::binary);
        return SplatLoader::parsePointCloudPly(in, o, info);
    }

    bool isPointCloud(const std::string& buffer) {

        std::istringstream in(buffer, std::ios::binary);
        return SplatLoader::isPointCloudPly(in);
    }

    PointPlyOptions plyOpts(bool color = true, bool normals = false, bool ascii = false,
                            bool bigEndian = false, bool faces = false, bool doubles = false,
                            bool floatColor = false, bool intensity = false,
                            bool leadingElement = false) {

        PointPlyOptions o;
        o.color = color;
        o.normals = normals;
        o.ascii = ascii;
        o.bigEndian = bigEndian;
        o.faces = faces;
        o.doubles = doubles;
        o.floatColor = floatColor;
        o.intensity = intensity;
        o.leadingElement = leadingElement;
        return o;
    }

}// namespace


TEST_CASE("SplatLoader point cloud: the discriminator tells a point cloud from a splat and a mesh") {

    const auto pts = lattice(3, 0.5f);
    CHECK(isPointCloud(pointCloudPly(pts)));
    CHECK(isPointCloud(pointCloudPly(pts, plyOpts(true, false, /*ascii*/ true))));
    CHECK(isPointCloud(pointCloudPly(pts, plyOpts(true, false, false, false, false, false, false, false,
                                                  /*leadingElement*/ true))));
    CHECK_FALSE(isPointCloud(pointCloudPly(pts, plyOpts(true, false, false, false, /*faces*/ true))));
    CHECK_FALSE(isPointCloud(splattest::writeSplatPly(makeCloud(0, 8))));
    CHECK_FALSE(isPointCloud("not a ply\n"));
    CHECK_FALSE(SplatLoader::isPointCloudPly(std::filesystem::path("does-not-exist.ply")));

    // The splat discriminator says no to a point cloud, so an importer that
    // asks both gets exactly one yes.
    std::istringstream in(pointCloudPly(pts), std::ios::binary);
    CHECK_FALSE(SplatLoader::isSplatPly(in));
}

TEST_CASE("SplatLoader point cloud: uchar colour becomes the DC colour, sigma comes from the spacing") {

    const auto pts = lattice(4, 0.5f);
    SplatLoader::PointCloudInfo info;
    const auto data = loadPoints(pointCloudPly(pts), {}, &info);

    REQUIRE(data.count() == pts.size());
    CHECK(data.shDegree == 0);
    CHECK(info.count == pts.size());
    CHECK(info.hadColor);
    CHECK_FALSE(info.hadNormals);
    CHECK(info.spacing == Approx(0.5f).margin(1e-5f));
    CHECK(info.sigma == Approx(0.5f).margin(1e-5f));// 1.0 * spacing

    for (size_t i = 0; i < data.count(); ++i) {

        INFO("point " << i);
        CHECK(data.means[i].x == Approx(pts[i].x));
        CHECK(data.means[i].y == Approx(pts[i].y));
        CHECK(data.means[i].z == Approx(pts[i].z));
        CHECK(data.scales[i].x == Approx(0.5f).margin(1e-5f));
        CHECK(data.scales[i].y == Approx(0.5f).margin(1e-5f));
        CHECK(data.scales[i].z == Approx(0.5f).margin(1e-5f));
        CHECK(data.rotations[i] == SplatQuat{});
        CHECK(data.opacities[i] == 1.f);

        const auto c = data.colorAt(i, Vector3{0, 0, 1});
        CHECK(c.x == Approx(pts[i].r / 255.f).margin(2e-3f));
        CHECK(c.y == Approx(0.f).margin(2e-3f));
        CHECK(c.z == Approx(pts[i].b / 255.f).margin(2e-3f));
    }

    // The unconsumed scalar rides in extras, as the splat loader's do.
    REQUIRE(data.extras.count("confidence") == 1);
    CHECK(data.extras.at("confidence")[3] == Approx(1.5f));
}

TEST_CASE("SplatLoader point cloud: ascii, big-endian and double coordinates read the same cloud") {

    const auto pts = lattice(3, 1.f);
    const auto ref = loadPoints(pointCloudPly(pts));

    const std::string variants[] = {
            pointCloudPly(pts, plyOpts(true, false, /*ascii*/ true)),
            pointCloudPly(pts, plyOpts(true, false, false, /*bigEndian*/ true)),
            pointCloudPly(pts, plyOpts(true, false, false, false, false, /*doubles*/ true)),
            pointCloudPly(pts, plyOpts(true, false, false, true, false, true)),
            pointCloudPly(pts, plyOpts(true, false, false, false, false, false, false, false, true)),
            pointCloudPly(pts, plyOpts(true, false, true, false, false, false, false, false, true)),
    };
    for (size_t v = 0; v < std::size(variants); ++v) {

        INFO("variant " << v);
        const auto data = loadPoints(variants[v]);
        REQUIRE(data.count() == ref.count());
        for (size_t i = 0; i < data.count(); ++i) {
            CHECK(data.means[i].x == Approx(ref.means[i].x).margin(1e-5f));
            CHECK(data.means[i].y == Approx(ref.means[i].y).margin(1e-5f));
            CHECK(data.means[i].z == Approx(ref.means[i].z).margin(1e-5f));
            CHECK(data.shAt(i)[0] == Approx(ref.shAt(i)[0]).margin(1e-5f));
            CHECK(data.shAt(i)[2] == Approx(ref.shAt(i)[2]).margin(1e-5f));
        }
    }
}

TEST_CASE("SplatLoader point cloud: float colours in [0, 255] are normalised, intensity is grey") {

    const auto pts = lattice(2, 1.f);

    SECTION("float [0, 255]") {

        const auto data = loadPoints(pointCloudPly(pts, plyOpts(true, false, false, false, false, false,
                                                                /*floatColor*/ true)));
        const auto c = data.colorAt(7, Vector3{0, 0, 1});// x = 1, z = 1 corner: (255, 0, 255)
        CHECK(c.x == Approx(1.f).margin(2e-3f));
        CHECK(c.y == Approx(0.f).margin(2e-3f));
        CHECK(c.z == Approx(1.f).margin(2e-3f));
    }

    SECTION("intensity, scaled by the cloud's maximum") {

        SplatLoader::PointCloudInfo info;
        const auto data = loadPoints(pointCloudPly(pts, plyOpts(/*color*/ false, false, false, false, false,
                                                                false, false, /*intensity*/ true)),
                                     {}, &info);
        CHECK_FALSE(info.hadColor);
        CHECK(info.hadIntensity);
        // Intensities 0, 100, ..., 700: the last point is white, the first black.
        const auto last = data.colorAt(7, Vector3{0, 0, 1});
        const auto first = data.colorAt(0, Vector3{0, 0, 1});
        CHECK(last.x == Approx(1.f).margin(2e-3f));
        CHECK(last.x == Approx(last.y).margin(1e-6f));
        CHECK(first.x == Approx(0.f).margin(2e-3f));
    }

    SECTION("no colour at all is white") {

        const auto data = loadPoints(pointCloudPly(pts, plyOpts(/*color*/ false)));
        const auto c = data.colorAt(0, Vector3{0, 0, 1});
        CHECK(c.x == Approx(1.f).margin(2e-3f));
        CHECK(c.y == Approx(1.f).margin(2e-3f));
        CHECK(c.z == Approx(1.f).margin(2e-3f));
    }
}

TEST_CASE("SplatLoader point cloud: normals orient a thin disc, and can be ignored") {

    auto pts = lattice(3, 1.f);
    for (auto& p : pts) {
        p.nx = 1.f;
        p.ny = 0.f;
        p.nz = 0.f;
    }
    pts[0].nx = 0.f;// one zero normal stays isotropic

    SplatLoader::PointCloudInfo info;
    const auto data = loadPoints(pointCloudPly(pts, plyOpts(true, /*normals*/ true)), {}, &info);
    CHECK(info.hadNormals);
    REQUIRE(data.count() == pts.size());

    // The local +Z axis, rotated, must land on the normal.
    const Vector3 z{0.f, 0.f, 1.f};
    for (size_t i = 1; i < data.count(); ++i) {
        INFO("point " << i);
        Vector3 axis = z;
        axis.applyQuaternion(data.rotations[i].toQuaternion());
        CHECK(axis.x == Approx(1.f).margin(1e-5f));
        CHECK(axis.y == Approx(0.f).margin(1e-5f));
        CHECK(axis.z == Approx(0.f).margin(1e-5f));
        CHECK(data.scales[i].z == Approx(data.scales[i].x * 0.15f).margin(1e-6f));
    }
    CHECK(data.rotations[0] == SplatQuat{});
    CHECK(data.scales[0].z == Approx(data.scales[0].x).margin(1e-6f));

    SECTION("a normal pointing at -Z is a half turn, not a NaN") {

        auto flipped = lattice(2, 1.f);
        for (auto& p : flipped) p.nz = -1.f;
        const auto d = loadPoints(pointCloudPly(flipped, plyOpts(true, true)));
        Vector3 axis = z;
        axis.applyQuaternion(d.rotations[0].toQuaternion());
        CHECK(axis.z == Approx(-1.f).margin(1e-5f));
    }

    SECTION("useNormals = false keeps every point isotropic") {

        SplatLoader::PointCloudOptions o;
        o.useNormals = false;
        const auto d = loadPoints(pointCloudPly(pts, plyOpts(true, true)), o);
        for (size_t i = 0; i < d.count(); ++i) {
            CHECK(d.rotations[i] == SplatQuat{});
            CHECK(d.scales[i].z == Approx(d.scales[i].x).margin(1e-6f));
        }
    }
}

TEST_CASE("SplatLoader point cloud: options override the spacing, the opacity and the thickness") {

    const auto pts = lattice(3, 1.f);
    SplatLoader::PointCloudOptions o;
    o.sigma = 0.07f;
    o.opacity = 0.4f;

    SplatLoader::PointCloudInfo info;
    const auto data = loadPoints(pointCloudPly(pts), o, &info);
    CHECK(info.spacing == 0.f);// not measured when sigma is given
    CHECK(info.sigma == Approx(0.07f));
    for (size_t i = 0; i < data.count(); ++i) {
        CHECK(data.scales[i].x == Approx(0.07f));
        CHECK(data.opacities[i] == Approx(0.4f));
    }

    SECTION("sigmaPerSpacing scales the measured spacing") {

        SplatLoader::PointCloudOptions s;
        s.sigmaPerSpacing = 1.5f;
        const auto d = loadPoints(pointCloudPly(pts), s, &info);
        CHECK(info.spacing == Approx(1.f).margin(1e-5f));
        CHECK(d.scales[0].x == Approx(1.5f).margin(1e-5f));
    }

    SECTION("a single point has no spacing and gets the documented fallback") {

        const auto d = loadPoints(pointCloudPly({PointSpec{1.f, 2.f, 3.f}}), {}, &info);
        CHECK(d.count() == 1);
        CHECK(info.spacing == 0.f);
        CHECK(d.scales[0].x == Approx(0.01f));
    }
}

TEST_CASE("SplatLoader point cloud: a mesh's vertices load and its faces are skipped") {

    const auto pts = lattice(2, 1.f);
    for (const bool ascii : {false, true}) {

        INFO("ascii " << ascii);
        const auto data = loadPoints(pointCloudPly(pts, plyOpts(true, false, ascii, false, /*faces*/ true)));
        REQUIRE(data.count() == pts.size());
        CHECK(data.means[7].x == Approx(1.f));
    }
}

TEST_CASE("SplatLoader point cloud: rejects what it cannot represent") {

    SECTION("no x/y/z") {

        CHECK_THROWS_AS(loadPoints("ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                                   "property float u\nproperty float v\nend_header\n"),
                        std::runtime_error);
    }

    SECTION("a list property on the vertex element") {

        CHECK_THROWS_AS(loadPoints("ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
                                   "property float x\nproperty float y\nproperty float z\n"
                                   "property list uchar float weights\nend_header\n"),
                        std::runtime_error);
    }

    SECTION("a truncated body") {

        auto ply = pointCloudPly(lattice(2, 1.f));
        ply.resize(ply.size() - 5);
        CHECK_THROWS_AS(loadPoints(ply), std::runtime_error);
    }

    SECTION("a truncated ascii body") {

        auto ply = pointCloudPly(lattice(2, 1.f), plyOpts(true, false, /*ascii*/ true));
        ply.resize(ply.size() - 20);
        CHECK_THROWS_AS(loadPoints(ply), std::runtime_error);
    }

    SECTION("loading a missing file names it") {

        CHECK_THROWS_AS(SplatLoader::loadPointCloudPly(std::filesystem::path("does-not-exist.ply")),
                        std::runtime_error);
    }
}
