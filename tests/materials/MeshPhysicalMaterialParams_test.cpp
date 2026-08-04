// The five thin-film / KHR-specular properties, end to end through the two
// APIs that are easy to leave half-wired.
//
// They are all plain data on MeshPhysicalMaterial and all already understood by
// GLTFLoader, ObjectExporter and ObjectLoader — but nothing exercised the
// serialization pair, and the fluent Params builder was missing
// specularIntensity/specularColor entirely (the fields existed, the setters did
// not, so `Params{}.specularIntensity(...)` simply did not compile).
//
// MaterialClone_test already covers clone()/copyInto() for these mixins via its
// mirror tripwire; this file covers the other two paths.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/loaders/ObjectExporter.hpp"
#include "threepp/loaders/ObjectLoader.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/scenes/Scene.hpp"

using namespace threepp;
using Catch::Matchers::WithinAbs;

TEST_CASE("MeshPhysicalMaterial::Params carries the specular and iridescence fields") {

    auto m = MeshPhysicalMaterial::create(
            MeshPhysicalMaterial::Params{}
                    .specularIntensity(0.25f)
                    .specularColor(Color(1.f, 0.5f, 0.25f))
                    .iridescence(0.75f)
                    .iridescenceIOR(1.8f)
                    .iridescenceThicknessNm(275.f));

    CHECK_THAT(m->specularIntensity, WithinAbs(0.25f, 1e-6f));
    CHECK(m->specularColor.equals(Color(1.f, 0.5f, 0.25f)));
    CHECK_THAT(m->iridescence, WithinAbs(0.75f, 1e-6f));
    CHECK_THAT(m->iridescenceIOR, WithinAbs(1.8f, 1e-6f));
    CHECK_THAT(m->iridescenceThicknessNm, WithinAbs(275.f, 1e-4f));

    // Params only writes what was set: the untouched fields keep their defaults.
    auto def = MeshPhysicalMaterial::create(MeshPhysicalMaterial::Params{}.roughness(0.3f));
    CHECK_THAT(def->specularIntensity, WithinAbs(1.f, 1e-6f));
    CHECK(def->specularColor.equals(Color(1, 1, 1)));
    CHECK_THAT(def->iridescence, WithinAbs(0.f, 1e-6f));
}

TEST_CASE("Specular and iridescence survive an ObjectExporter/ObjectLoader round-trip") {

    auto scene = Scene::create();
    auto mat = MeshPhysicalMaterial::create();
    mat->specularIntensity = 0.4f;
    mat->specularColor = Color(0.2f, 0.6f, 0.9f);
    mat->iridescence = 0.65f;
    mat->iridescenceIOR = 2.1f;
    mat->iridescenceThicknessNm = 512.f;
    scene->add(Mesh::create(BoxGeometry::create(), mat));

    ObjectExporter exporter;
    const auto text = exporter.toJson(*scene);

    ObjectLoader loader;
    auto parsed = loader.parse(text);
    REQUIRE(parsed != nullptr);
    REQUIRE(parsed->children.size() == 1);

    auto* mesh = parsed->children.front()->as<Mesh>();
    REQUIRE(mesh != nullptr);
    auto* out = dynamic_cast<MeshPhysicalMaterial*>(mesh->material().get());
    REQUIRE(out != nullptr);

    CHECK_THAT(out->specularIntensity, WithinAbs(0.4f, 1e-5f));

    // Colours serialize as a packed 24-bit hex, so a channel comes back
    // quantized to 1/255. Floats round-trip exactly.
    constexpr float lsb = 1.f / 255.f;
    CHECK_THAT(out->specularColor.r, WithinAbs(0.2f, lsb));
    CHECK_THAT(out->specularColor.g, WithinAbs(0.6f, lsb));
    CHECK_THAT(out->specularColor.b, WithinAbs(0.9f, lsb));
    CHECK_THAT(out->iridescence, WithinAbs(0.65f, 1e-5f));
    CHECK_THAT(out->iridescenceIOR, WithinAbs(2.1f, 1e-5f));
    CHECK_THAT(out->iridescenceThicknessNm, WithinAbs(512.f, 1e-3f));
}
