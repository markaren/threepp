// What the scene format does with a splat cloud: a reference, never the splats.
//
// A SplatCloud is written as a `threeppSplat` block — the file the splats come
// from, relative to the document when possible, plus the import ops to replay
// and the point-mode look — and ObjectLoader re-imports the file on the way
// back. A cloud with no file gets a sidecar .ply next to a loose document or a
// member of a .tpz; inside an archive a sourced cloud's file is copied in, so
// the archive is self-contained. A document that can hold neither (the play
// snapshot's toJson) writes a pathless block, and the loader's resolver hook
// hands the live object back.
//
// The size assertion is kept from the days the type was refused outright: a
// cloud must never cost the document anything like its splat count.

#include "splat_ply_writer.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/loaders/ObjectExporter.hpp"
#include "threepp/loaders/ObjectLoader.hpp"
#include "threepp/loaders/SplatLoader.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/splats/SplatData.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>

using namespace threepp;
using Catch::Approx;
namespace fs = std::filesystem;

namespace {

    bool mentions(const std::vector<std::string>& warnings, const char* needle) {

        for (const auto& w : warnings) {
            if (w.find(needle) != std::string::npos) return true;
        }
        return false;
    }

    std::string userDataString(const Object3D& o, const char* key) {

        const auto it = o.userData.find(key);
        if (it == o.userData.end()) return {};
        if (const auto* s = std::any_cast<std::string>(&it->second)) return *s;
        return {};
    }

    SplatData generated(size_t count, unsigned seed = 7u) {

        SplatGenerator::Options options;
        options.count = count;
        options.seed = seed;
        options.shDegree = 1;
        options.includeDegenerates = false;
        return SplatGenerator::generate(options);
    }

    // A fresh directory per test, so sidecars and archives cannot collide.
    fs::path scratch(const char* tag) {

        const auto dir = fs::temp_directory_path() / ("threepp_splat_doc_" + std::string(tag));
        fs::remove_all(dir);
        fs::create_directories(dir);
        return dir;
    }

    // The cloud an editor import produces: the data, the source mark, the ops.
    std::shared_ptr<SplatCloud> importedCloud(const fs::path& ply, const SplatData& data, bool cull) {

        auto cloud = SplatCloud::create(data);
        cloud->name = "atlas scan";
        const auto u8 = ply.u8string();
        cloud->userData["splatSource"] = std::string(u8.begin(), u8.end());
        cloud->userData["splatImportOps"] = std::string(cull ? "cull=1;removed=0;flipX=1;lod=-1;points=0"
                                                             : "cull=0;removed=0;flipX=1;lod=-1;points=0");
        return cloud;
    }

    void checkSameSplats(const SplatData& a, const SplatData& b) {

        REQUIRE(a.count() == b.count());
        REQUIRE(a.shDegree == b.shDegree);
        for (size_t i = 0; i < a.count(); i += std::max<size_t>(1, a.count() / 16)) {
            CHECK(a.means[i].x == Approx(b.means[i].x).margin(1e-5f));
            CHECK(a.means[i].y == Approx(b.means[i].y).margin(1e-5f));
            CHECK(a.scales[i].x == Approx(b.scales[i].x).epsilon(1e-4f));
            CHECK(a.opacities[i] == Approx(b.opacities[i]).margin(1e-4f));
            CHECK(a.shAt(i)[0] == Approx(b.shAt(i)[0]).margin(1e-5f));
        }
    }

}// namespace


TEST_CASE("SplatLoader::writePly round-trips through loadPly", "[splats]") {

    auto data = generated(300);
    data.extras["confidence"].assign(data.count(), 0.f);
    for (size_t i = 0; i < data.count(); ++i) data.extras["confidence"][i] = static_cast<float>(i) * 0.5f;

    const auto dir = scratch("writeply");
    const auto path = dir / "cloud.ply";
    SplatLoader::writePly(data, path);

    CHECK(SplatLoader::isSplatPly(path));
    const auto back = SplatLoader::loadPly(path);
    checkSameSplats(data, back);
    REQUIRE(back.extras.count("confidence") == 1);
    CHECK(back.extras.at("confidence")[10] == Approx(5.f));
    // Quaternions come back unit, up to sign.
    for (size_t i = 0; i < back.count(); i += 37) {
        const auto& p = data.rotations[i];
        const auto& q = back.rotations[i];
        const float dot = p.x * q.x + p.y * q.y + p.z * q.z + p.w * q.w;
        CHECK(std::abs(dot) == Approx(1.f).margin(1e-4f));
    }

    // Degree 0 and 3 write the right column count.
    for (int degree : {0, 3}) {
        SplatGenerator::Options o;
        o.count = 20;
        o.shDegree = degree;
        o.includeDegenerates = false;
        const auto d = SplatGenerator::generate(o);
        const auto p = dir / ("d" + std::to_string(degree) + ".ply");
        SplatLoader::writePly(d, p);
        const auto b = SplatLoader::loadPly(p);
        CHECK(b.shDegree == degree);
        CHECK(b.count() == 20);
    }

    CHECK_THROWS_AS(SplatLoader::writePly(data, dir / "nope" / "deeper" / "x.ply"), std::runtime_error);
}

TEST_CASE("a sourced splat cloud is written as a relative reference and reloads", "[splats]") {

    const auto dir = scratch("reference");
    const auto data = generated(4000);
    const auto ply = dir / "scan.ply";
    SplatLoader::writePly(data, ply);

    auto scene = Scene::create();
    auto cloud = importedCloud(ply, data, /*cull=*/false);
    cloud->position.set(1.f, 2.f, 3.f);
    cloud->rotation.x = 3.14159f;
    cloud->setPointMix(1.f);
    cloud->setPointSize(4.f);
    scene->add(cloud);
    auto box = Object3D::create();
    box->name = "keep me";
    scene->add(box);

    ObjectExporter exporter;
    ObjectExporterOptions options;
    options.resourcePath = dir;
    const auto json = exporter.toJson(*scene, options);

    INFO("document is " << json.size() << " bytes");
    CHECK(exporter.warnings().empty());
    CHECK(json.find("\"SplatCloud\"") != std::string::npos);
    CHECK(json.find("threeppSplat") != std::string::npos);
    CHECK(json.find("\"scan.ply\"") != std::string::npos);// relative, not absolute
    CHECK(json.find("instanceMatrix") == std::string::npos);
    // A 4000-splat cloud costs the document a few hundred bytes, not its splats.
    CHECK(json.size() < 8192);

    ObjectLoader loader;
    loader.setResourcePath(dir);
    auto restored = loader.parse(json);
    REQUIRE(restored);
    CHECK(loader.warnings().empty());
    REQUIRE(restored->children.size() == 2);

    auto* back = restored->children.front()->as<SplatCloud>();
    REQUIRE(back);
    CHECK(back->name == "atlas scan");
    CHECK(back->splatCount() == 4000);
    CHECK(back->position.equals({1.f, 2.f, 3.f}));
    CHECK(back->rotation.x == Approx(3.14159f).margin(1e-4f));
    CHECK(back->pointMix() == 1.f);
    CHECK(back->pointSize() == 4.f);
    checkSameSplats(data, back->data());
    CHECK(restored->children.back()->name == "keep me");

    // The source mark is this machine's absolute path to the file, so the
    // inspector shows it and a re-save writes the same reference.
    const auto source = userDataString(*back, "splatSource");
    CHECK(fs::exists(fs::u8path(source)));
    CHECK(fs::equivalent(fs::u8path(source), ply));
    CHECK(userDataString(*back, "splatImportOps").find("flipX=1") != std::string::npos);

    SECTION("and the cull op is replayed on reload") {

        auto culled = importedCloud(ply, data, /*cull=*/true);
        auto scene2 = Scene::create();
        scene2->add(culled);
        const auto json2 = exporter.toJson(*scene2, options);
        CHECK((json2.find("\"cull\":true") != std::string::npos || json2.find("\"cull\": true") != std::string::npos));
        auto restored2 = loader.parse(json2);
        REQUIRE(restored2);
        auto* back2 = restored2->children.front()->as<SplatCloud>();
        REQUIRE(back2);
        auto expected = data;
        expected.removeOutliers();
        CHECK(back2->splatCount() == expected.count());
    }
}

TEST_CASE("a cloud with no source file gets a sidecar next to the document", "[splats]") {

    const auto dir = scratch("sidecar");
    const auto data = generated(500, 11u);

    auto scene = Scene::create();
    auto cloud = SplatCloud::create(data);
    cloud->name = "procedural";
    scene->add(cloud);

    ObjectExporter exporter;
    ObjectExporterOptions options;
    options.resourcePath = dir;
    const auto json = exporter.toJson(*scene, options);

    CHECK(exporter.warnings().empty());
    const auto sidecar = dir / "splats" / (cloud->uuid + ".ply");
    CHECK(fs::is_regular_file(sidecar));
    CHECK(json.find("splats/" + cloud->uuid + ".ply") != std::string::npos);

    ObjectLoader loader;
    loader.setResourcePath(dir);
    auto restored = loader.parse(json);
    REQUIRE(restored);
    CHECK(loader.warnings().empty());
    auto* back = restored->children.front()->as<SplatCloud>();
    REQUIRE(back);
    CHECK(back->name == "procedural");
    checkSameSplats(data, back->data());

    // An empty cloud is a legal document too.
    auto empty = SplatCloud::create(SplatData{});
    empty->name = "empty";
    auto scene2 = Scene::create();
    scene2->add(empty);
    const auto json2 = exporter.toJson(*scene2, options);
    auto restored2 = loader.parse(json2);
    REQUIRE(restored2);
    auto* backEmpty = restored2->children.front()->as<SplatCloud>();
    REQUIRE(backEmpty);
    CHECK(backEmpty->splatCount() == 0);
}

TEST_CASE("a pathless block: the resolver hands the live cloud back, a plain load gets a placeholder", "[splats]") {

    const auto data = generated(200, 3u);
    auto scene = Scene::create();
    auto cloud = SplatCloud::create(data);
    cloud->name = "snapshot me";
    cloud->position.set(4.f, 5.f, 6.f);
    cloud->setPointMix(0.25f);
    scene->add(cloud);

    // No resourcePath, no archive: what the play snapshot does.
    ObjectExporter exporter;
    const auto json = exporter.toJson(*scene);
    CHECK(json.find("threeppSplat") != std::string::npos);
    CHECK((json.find("\"path\":\"\"") != std::string::npos || json.find("\"path\": \"\"") != std::string::npos));

    SECTION("plain load") {

        ObjectLoader loader;
        auto restored = loader.parse(json);
        REQUIRE(restored);
        CHECK(mentions(loader.warnings(), "snapshot me"));
        CHECK(mentions(loader.warnings(), "could not be restored"));
        REQUIRE(restored->children.size() == 1);
        // Placed and named, just not a cloud.
        CHECK(restored->children.front()->as<SplatCloud>() == nullptr);
        CHECK(restored->children.front()->name == "snapshot me");
        CHECK(restored->children.front()->position.equals({4.f, 5.f, 6.f}));
    }

    SECTION("with the resolver") {

        // The "session" moves the cloud and changes its look; the restore must
        // put the captured state back onto the SAME object.
        cloud->position.set(0.f, 0.f, 0.f);
        cloud->setPointMix(1.f);

        ObjectLoader loader;
        loader.setSplatCloudResolver([&](const std::string& uuid) -> std::shared_ptr<SplatCloud> {
            return uuid == cloud->uuid ? cloud : nullptr;
        });
        auto restored = loader.parse(json);
        REQUIRE(restored);
        CHECK(loader.warnings().empty());
        REQUIRE(restored->children.size() == 1);
        CHECK(restored->children.front() == cloud.get());
        CHECK(cloud->parent == restored.get());
        CHECK(scene->children.empty());// moved, not copied
        CHECK(cloud->position.equals({4.f, 5.f, 6.f}));
        CHECK(cloud->pointMix() == Approx(0.25f));
        CHECK(cloud->splatCount() == 200);
    }
}

TEST_CASE("a .tpz carries the scan's bytes and reloads self-contained", "[splats]") {

    const auto dir = scratch("archive");
    const auto data = generated(1500, 5u);
    const auto ply = dir / "source.ply";
    SplatLoader::writePly(data, ply);

    auto scene = Scene::create();
    auto sourced = importedCloud(ply, data, false);
    sourced->name = "sourced";
    scene->add(sourced);
    auto loose = SplatCloud::create(generated(80, 9u));
    loose->name = "loose";
    scene->add(loose);

    const auto archive = dir / "scene.tpz";
    ObjectExporter exporter;
    ObjectExporterOptions options;
    options.resourcePath = dir;
    exporter.save(*scene, archive, options);
    CHECK(exporter.warnings().empty());

    // The source file is not needed any more: the archive has a copy.
    fs::remove(ply);

    ObjectLoader loader;
    auto restored = loader.load(archive);
    REQUIRE(restored);
    CHECK(loader.warnings().empty());
    REQUIRE(restored->children.size() == 2);
    auto* a = restored->children[0]->as<SplatCloud>();
    auto* b = restored->children[1]->as<SplatCloud>();
    REQUIRE(a);
    REQUIRE(b);
    CHECK(a->name == "sourced");
    CHECK(a->splatCount() == 1500);
    CHECK(b->name == "loose");
    CHECK(b->splatCount() == 80);
    // The mark says where the bytes live now: this archive, that member.
    CHECK(userDataString(*a, "splatSource").find(".tpz|splats/") != std::string::npos);
}

TEST_CASE("a missing source leaves a placed placeholder and says why", "[splats]") {

    const auto dir = scratch("missing");
    const auto data = generated(50);
    auto scene = Scene::create();
    auto cloud = importedCloud(dir / "gone.ply", data, false);
    cloud->name = "lost scan";
    cloud->position.set(7.f, 8.f, 9.f);
    scene->add(cloud);

    ObjectExporter exporter;
    ObjectExporterOptions options;
    options.resourcePath = dir;
    const auto json = exporter.toJson(*scene, options);

    ObjectLoader loader;
    loader.setResourcePath(dir);
    auto restored = loader.parse(json);
    REQUIRE(restored);
    CHECK(mentions(loader.warnings(), "lost scan"));
    CHECK(mentions(loader.warnings(), "does not exist"));
    REQUIRE(restored->children.size() == 1);
    CHECK(restored->children.front()->as<SplatCloud>() == nullptr);
    CHECK(restored->children.front()->name == "lost scan");
    CHECK(restored->children.front()->position.equals({7.f, 8.f, 9.f}));
}

TEST_CASE("a splat cloud exported as the document root round-trips", "[splats]") {

    const auto dir = scratch("root");
    const auto data = generated(64);
    auto cloud = SplatCloud::create(data);
    cloud->name = "lone scan";
    cloud->position.set(1.f, 2.f, 3.f);

    ObjectExporter exporter;
    ObjectExporterOptions options;
    options.resourcePath = dir;
    const auto json = exporter.toJson(*cloud, options);
    CHECK(json.find("instanceMatrix") == std::string::npos);

    ObjectLoader loader;
    loader.setResourcePath(dir);
    auto restored = loader.parse(json);
    REQUIRE(restored);
    CHECK(restored->type() == "SplatCloud");
    CHECK(restored->name == "lone scan");
    CHECK(restored->position.equals({1.f, 2.f, 3.f}));
    CHECK(restored->as<SplatCloud>()->splatCount() == 64);
}
