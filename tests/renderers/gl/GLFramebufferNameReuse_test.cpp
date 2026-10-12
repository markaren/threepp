// Regression test for the GLState framebuffer-cache name-reuse hazard, the
// sibling of the texture and program ones (GLTextureNameReuse_test.cpp,
// GLProgramNameReuse_test.cpp).
//
// GLState::bindFramebuffer caches the bound name per target and skips the
// glBindFramebuffer when it matches. Deleting the bound framebuffer reverts
// the binding to 0 and frees the name, and a driver that recycles promptly
// (Mesa) hands it to the next render target. Its set-up then binds "the same"
// framebuffer, the bind is skipped, and the colour attachment is made on the
// default framebuffer: the target has no colour buffer and nothing drawn to it
// lands. GLState::purgeFramebuffer drops the name on delete.
//
// On a driver that does not recycle names promptly the test passes either way.

#include "gl_test_helpers.hpp"

TEST_CASE("GL: render target created after the bound one was disposed is rendered to") {
    GLRenderer renderer(glCanvas());

    auto scene = Scene::create();
    auto camera = OrthographicCamera::create(-1, 1, 1, -1, 0.1f, 10);
    camera->position.z = 1;

    auto first = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(first.get());
    renderer.setClearColor(Color(1.0f, 0.0f, 0.0f));
    renderer.render(*scene, *camera);

    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);
    CHECK(allPixelsMatch(pixels, 255, 0, 0, 2));

    // Disposed while it is the bound target, so its name is the cached one.
    first->dispose();

    auto second = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(second.get());
    renderer.setClearColor(Color(0.0f, 1.0f, 0.0f));
    renderer.render(*scene, *camera);

    pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);
    CHECK(allPixelsMatch(pixels, 0, 255, 0, 2));

    renderer.setRenderTarget(nullptr);
    renderer.dispose();
}
