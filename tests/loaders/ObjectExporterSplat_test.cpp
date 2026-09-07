// What the scene format does with a splat cloud: nothing, loudly.
//
// SplatCloud is not serializable yet. It is an InstancedMesh, though, so
// without a deliberate refusal the exporter takes the instancing branch and
// writes a 16-float instanceMatrix per splat — matrices SplatCloud keeps at
// identity on purpose, because the per-splat payload lives in DataTextures
// hanging off a RawShaderMaterial's uniforms, which are not serialized either.
// ObjectLoader has no "SplatCloud" case and drops the type on the way back in,
// so all of it is written to be discarded.
//
// These tests pin the refusal, and the size assertion is the part that matters:
// it is what turns "the exporter quietly got expensive again" into a failure
// rather than a slow save nobody attributes to this.

#include "threepp/core/Object3D.hpp"
#include "threepp/loaders/ObjectExporter.hpp"
#include "threepp/loaders/ObjectLoader.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/splats/SplatData.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <string>

using namespace threepp;

namespace {

    bool mentions(const std::vector<std::string>& warnings, const char* needle) {

        for (const auto& w : warnings) {
            if (w.find(needle) != std::string::npos) return true;
        }
        return false;
    }

}// namespace


TEST_CASE("a splat cloud is skipped, with a warning naming it", "[splats]") {

    auto scene = Scene::create();

    // 4000 splats: enough that the instanceMatrix the old path wrote (4000 * 16
    // floats, as JSON numbers) dwarfs everything else in the document, so the
    // size assertion below has something to catch.
    SplatGenerator::Options options;
    options.count = 4000;
    auto cloud = SplatCloud::create(SplatGenerator::generate(options));
    cloud->name = "atlas scan";
    scene->add(cloud);

    auto box = Object3D::create();
    box->name = "keep me";
    scene->add(box);

    ObjectExporter exporter;
    const auto json = exporter.toJson(*scene);

    INFO("document is " << json.size() << " bytes");

    CHECK(mentions(exporter.warnings(), "atlas scan"));
    CHECK(mentions(exporter.warnings(), "not serialized yet"));

    // The cloud is absent, the sibling is not.
    CHECK(json.find("SplatCloud") == std::string::npos);
    CHECK(json.find("atlas scan") == std::string::npos);
    CHECK(json.find("keep me") != std::string::npos);

    // No instanceMatrix, and nothing the size of one. 4000 identity matrices
    // alone are ~500 kB of JSON; a scene of two empty nodes is well under 4 kB.
    CHECK(json.find("instanceMatrix") == std::string::npos);
    CHECK(json.size() < 4096);

    // And it loads back, with the sibling intact.
    ObjectLoader loader;
    auto restored = loader.parse(json);
    REQUIRE(restored);
    CHECK(restored->children.size() == 1);
    CHECK(restored->children.front()->name == "keep me");
}

TEST_CASE("an empty splat cloud is skipped too", "[splats]") {

    // Nothing about the refusal depends on the cloud having splats in it: the
    // type is what cannot be written, not the size.
    auto scene = Scene::create();
    auto cloud = SplatCloud::create(SplatData{});
    cloud->name = "empty";
    scene->add(cloud);

    ObjectExporter exporter;
    const auto json = exporter.toJson(*scene);

    CHECK(mentions(exporter.warnings(), "not serialized yet"));
    CHECK(json.find("SplatCloud") == std::string::npos);
}

TEST_CASE("a splat cloud exported as the document root leaves a placeholder", "[splats]") {

    // Nothing in the editor does this — a document is rooted at a Scene — but
    // the entry point is public, and the answer must not be "write the payload
    // anyway because it happens to be the root".
    SplatGenerator::Options options;
    options.count = 64;
    auto cloud = SplatCloud::create(SplatGenerator::generate(options));
    cloud->name = "lone scan";
    cloud->position.set(1.f, 2.f, 3.f);

    ObjectExporter exporter;
    const auto json = exporter.toJson(*cloud);

    CHECK(mentions(exporter.warnings(), "not serialized yet"));
    CHECK(json.find("instanceMatrix") == std::string::npos);

    // A valid document that loads, keeping the identity and the placement.
    ObjectLoader loader;
    auto restored = loader.parse(json);
    REQUIRE(restored);
    CHECK(restored->type() == "Object3D");
    CHECK(restored->name == "lone scan");
    CHECK(restored->position.equals({1.f, 2.f, 3.f}));
}
