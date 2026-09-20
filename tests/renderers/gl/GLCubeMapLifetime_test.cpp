// The cube-map / PMREM caches are keyed on a raw Texture*.
//
// An equirect environment gets converted into a cube render target and prefiltered
// into a PMREM atlas, both cached against the source texture's ADDRESS. Nothing
// listened for that texture's disposal, so the entries outlived it: the render
// target and the atlas leaked, and — the part that is not merely wasteful — a
// later Texture allocated at the same address would find the dead entry and
// silently render the PREVIOUS environment. r129 keys on a WeakMap and still
// subscribes to dispose (WebGLCubeMaps.js:51, 73).
//
// Adding that subscription creates a second hazard which the fix also has to
// handle: the listener holds a raw pointer to the cache, and a source texture
// routinely outlives the renderer — a scene holds it — so failing to unsubscribe
// on teardown turns the leak into a use-after-free in the texture's destructor.
//
// What these two cases do NOT do is prove either bug. Address reuse is not
// deterministic, and the stale-listener call is undefined behaviour that a
// release build happily executes against freed memory without faulting —
// measured: this file passes with the unsubscribe removed. They are here to
// exercise both lifetime orderings end to end, so that a future change which
// makes the failure loud is caught rather than merely latent.

#include "gl_test_helpers.hpp"

namespace {

    std::shared_ptr<Texture> equirectEnv(float le) {
        constexpr int W = 8, H = 4;
        std::vector<float> data(W * H * 4, le);
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    std::shared_ptr<Scene> sceneWith(const std::shared_ptr<Texture>& env) {
        auto scene = Scene::create();
        scene->environment = env;
        auto mat = MeshStandardMaterial::create();
        mat->roughness = 0.4f;
        mat->metalness = 1.f;
        scene->add(Mesh::create(SphereGeometry::create(1.f, 32, 24), mat));
        return scene;
    }

    std::shared_ptr<PerspectiveCamera> camera() {
        auto c = PerspectiveCamera::create(45, 1.f, 0.1f, 100.f);
        c->position.set(0, 0, 3.f);
        c->lookAt(Vector3{0, 0, 0});
        return c;
    }

}// namespace

TEST_CASE("an environment texture may outlive the renderer that cached it") {

    auto env = equirectEnv(0.5f);
    auto scene = sceneWith(env);
    auto cam = camera();

    {
        GLRenderer renderer(glCanvas());
        renderer.render(*scene, *cam);// builds the cube target and the PMREM
    }// renderer, and its cube-map cache, go away here

    // `env` is still alive and still subscribed if nothing unsubscribed it. Its
    // destructor dispatches "dispose", which would run the cache's listener
    // against freed memory.
    scene.reset();
    REQUIRE_NOTHROW(env.reset());
}

TEST_CASE("disposing an environment texture mid-session leaves the renderer usable") {

    auto first = equirectEnv(0.5f);
    auto cam = camera();

    GLRenderer renderer(glCanvas());

    auto scene = sceneWith(first);
    renderer.render(*scene, *cam);

    // Evicts the entries while the cache is alive — the path the listener exists
    // for.
    first->dispose();

    auto second = equirectEnv(0.25f);
    scene->environment = second;

    REQUIRE_NOTHROW(renderer.render(*scene, *cam));

    const auto px = renderer.readRGBPixels();
    REQUIRE(px.size() == DATA_SIZE);
    CHECK(maxPixelBrightness(px) > 0);
}
