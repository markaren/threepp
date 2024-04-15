#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#undef near
#undef far
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/renderers/gl/GLProperties.hpp"
#include "threepp/renderers/gl/GLRenderLists.hpp"

using namespace threepp;
using namespace threepp::gl;

class DummyMaterial: public Material {
public:
    [[nodiscard]] std::string type() const override {
        return "";
    }

protected:
    std::shared_ptr<Material> createDefault() const override {

        return {};
    }
};

class DummyProgram: public GLProgram {};

TEST_CASE("init") {

    GLProperties properties;
    GLRenderList list(properties);

    REQUIRE(list.transparent.empty());
    REQUIRE(list.opaque.empty());

    Object3D o;
    BufferGeometry g1;
    BufferGeometry g2;

    DummyMaterial m1;
    m1.transparent = true;
    DummyMaterial m2;
    m2.transparent = false;
    list.push(&o, &g1, &m1, 0, 0, std::nullopt);
    list.push(&o, &g2, &m2, 0, 0, std::nullopt);

    REQUIRE(list.transparent.size() == 1);
    REQUIRE(list.opaque.size() == 1);

    list.init();

    REQUIRE(list.transparent.empty());
    REQUIRE(list.opaque.empty());
}

TEST_CASE("push") {

    GLProperties properties;
    GLRenderList list(properties);

    Object3D objA;
    objA.id = 'A';
    objA.renderOrder = 0;
    DummyMaterial matA;
    matA.transparent = true;
    DummyProgram proA;
    BufferGeometry geoA;

    Object3D objB;
    objB.id = 'B';
    objB.renderOrder = 0;
    DummyMaterial matB;
    matB.transparent = true;
    DummyProgram proB;
    BufferGeometry geoB;

    Object3D objC;
    objC.id = 'C';
    objC.renderOrder = 0;
    DummyMaterial matC;
    matC.transparent = false;
    DummyProgram proC;
    BufferGeometry geoC;

    Object3D objD;
    objD.id = 'D';
    objD.renderOrder = 0;
    DummyMaterial matD;
    matD.transparent = false;
    DummyProgram proD;
    BufferGeometry geoD;

    auto materialProperties = properties.materialProperties.get(&matA);
    materialProperties->program = &proA;

    materialProperties = properties.materialProperties.get(&matB);
    materialProperties->program = &proB;

    materialProperties = properties.materialProperties.get(&matC);
    materialProperties->program = &proC;

    materialProperties = properties.materialProperties.get(&matD);
    materialProperties->program = &proD;

    // A
    {
        list.push(&objA, &geoA, &matA, 0, 0.5f, std::nullopt);
        CHECK(list.transparent.size() == 1);
        CHECK(list.opaque.empty());

        auto o = list.transparent[0];
        CHECK(o->id == 'A');
        CHECK(o->object == &objA);
        CHECK(o->geometry == &geoA);
        CHECK(o->material == &matA);
        CHECK(o->program == &proA);
        CHECK(o->groupOrder == 0);
        CHECK(o->renderOrder == 0);
        CHECK_THAT(o->z, Catch::Matchers::WithinRel(0.5f));
        CHECK(!o->group.has_value());
    }

    // B
    {
        list.push(&objB, &geoB, &matB, 1, 1.5f, std::nullopt);
        CHECK(list.transparent.size() == 2);
        CHECK(list.opaque.empty());

        auto o = list.transparent[1];
        CHECK(o->id == 'B');
        CHECK(o->object == &objB);
        CHECK(o->geometry == &geoB);
        CHECK(o->material == &matB);
        CHECK(o->program == &proB);
        CHECK(o->groupOrder == 1);
        CHECK(o->renderOrder == 0);
        CHECK_THAT(o->z, Catch::Matchers::WithinRel(1.5f));
        CHECK(!o->group.has_value());
    }

    // C
    {
        list.push(&objC, &geoC, &matC, 2, 2.5f, std::nullopt);
        CHECK(list.transparent.size() == 2);
        CHECK(list.opaque.size() == 1);

        auto o = list.opaque[0];
        CHECK(o->id == 'C');
        CHECK(o->object == &objC);
        CHECK(o->geometry == &geoC);
        CHECK(o->material == &matC);
        CHECK(o->program == &proC);
        CHECK(o->groupOrder == 2);
        CHECK(o->renderOrder == 0);
        CHECK_THAT(o->z, Catch::Matchers::WithinRel(2.5f));
        CHECK(!o->group.has_value());
    }

    // D
    {
        list.push(&objD, &geoD, &matD, 3, 3.5f, std::nullopt);
        CHECK(list.transparent.size() == 2);
        CHECK(list.opaque.size() == 2);

        auto o = list.opaque[1];
        CHECK(o->id == 'D');
        CHECK(o->object == &objD);
        CHECK(o->geometry == &geoD);
        CHECK(o->material == &matD);
        CHECK(o->program == &proD);
        CHECK(o->groupOrder == 3);
        CHECK(o->renderOrder == 0);
        CHECK_THAT(o->z, Catch::Matchers::WithinRel(3.5f));
        CHECK(!o->group.has_value());
    }
}
