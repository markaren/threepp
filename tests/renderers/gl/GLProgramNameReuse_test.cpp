// Regression tests for the GLState program-cache name-reuse hazard, the
// sibling of the texture one in GLTextureNameReuse_test.cpp.
//
// GLState::useProgram caches the current program name and skips the
// glUseProgram when it matches. glDeleteProgram on the program that is current
// only flags it: it stays current and keeps its name, so while threepp is the
// only GL user the stale cache entry cannot false-hit. Any other GL code in
// the context breaks that. Dear ImGui's GL backend binds its own program and,
// finding the previous one deleted, does not restore it; the name is then
// free, the next program threepp links is handed it, the cache says it is
// already current, and the draw runs with ImGui's program.
//
// The other GL user is what arms the trap and a test here cannot play it
// without raw GL calls, so the first two tests inspect the cache directly and
// assert the dispose path let go of the name (GLState::purgeProgram). The
// end-to-end case, with the real ImGui backend, is in python/tests/test_imgui.py.

#include "gl_test_helpers.hpp"
#include "threepp/materials/MeshNormalMaterial.hpp"

TEST_CASE("GL: disposing the current program's material evicts it from the program cache") {
    GLRenderer renderer(glCanvas());
    renderer.setClearColor(Color(0.0f, 0.0f, 1.0f));

    auto scene = Scene::create();
    auto camera = OrthographicCamera::create(-1, 1, 1, -1, 0.1f, 10);
    camera->position.z = 1;

    auto material = MeshBasicMaterial::create();
    material->color = Color(1.0f, 0.0f, 0.0f);
    auto mesh = Mesh::create(PlaneGeometry::create(2, 2), material);
    scene->add(mesh);

    renderer.render(*scene, *camera);
    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);
    CHECK(allPixelsMatch(pixels, 255, 0, 0, 2));

    // The one mesh in the scene was the last thing drawn, so its program is
    // the cached one. Dispose deletes it; a stale entry here is the recycled-
    // name trap armed.
    REQUIRE(renderer.state().currentProgram.has_value());

    scene->remove(*mesh);
    material->dispose();

    CHECK_FALSE(renderer.state().currentProgram.has_value());

    renderer.dispose();
}

TEST_CASE("GL: disposing a material whose program is not current leaves the program cache alone") {
    GLRenderer renderer(glCanvas());
    renderer.setClearColor(Color(0.0f, 0.0f, 1.0f));

    auto camera = OrthographicCamera::create(-1, 1, 1, -1, 0.1f, 10);
    camera->position.z = 1;

    auto geometry = PlaneGeometry::create(2, 2);

    auto sceneA = Scene::create();
    auto matA = MeshBasicMaterial::create();
    sceneA->add(Mesh::create(geometry, matA));

    auto sceneB = Scene::create();
    auto matB = MeshNormalMaterial::create();
    sceneB->add(Mesh::create(geometry, matB));

    renderer.render(*sceneA, *camera);
    REQUIRE(renderer.state().currentProgram.has_value());
    const auto programA = renderer.state().currentProgram;

    renderer.render(*sceneB, *camera);
    REQUIRE(renderer.state().currentProgram.has_value());
    const auto programB = renderer.state().currentProgram;
    REQUIRE(programA != programB);

    // B's program is current, A's is not: deleting A's must not unbind B's.
    matA->dispose();
    CHECK(renderer.state().currentProgram == programB);

    matB->dispose();
    CHECK_FALSE(renderer.state().currentProgram.has_value());

    renderer.dispose();
}

TEST_CASE("GL: material created after a dispose is drawn with its own program") {
    GLRenderer renderer(glCanvas());
    renderer.setClearColor(Color(0.0f, 0.0f, 1.0f));

    auto scene = Scene::create();
    auto camera = OrthographicCamera::create(-1, 1, 1, -1, 0.1f, 10);
    camera->position.z = 1;

    auto geometry = PlaneGeometry::create(2, 2);

    auto matA = MeshBasicMaterial::create();
    matA->color = Color(1.0f, 0.0f, 0.0f);
    auto meshA = Mesh::create(geometry, matA);
    scene->add(meshA);

    renderer.render(*scene, *camera);
    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);
    CHECK(allPixelsMatch(pixels, 255, 0, 0, 2));

    // Delete the current program, then draw with one that has to be linked
    // anew. Its first glUseProgram must go through whichever name it gets.
    scene->remove(*meshA);
    matA->dispose();

    auto matB = MeshBasicMaterial::create();
    matB->color = Color(0.0f, 1.0f, 0.0f);
    auto meshB = Mesh::create(geometry, matB);
    scene->add(meshB);

    renderer.render(*scene, *camera);
    pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);
    CHECK(allPixelsMatch(pixels, 0, 255, 0, 2));

    renderer.dispose();
}
