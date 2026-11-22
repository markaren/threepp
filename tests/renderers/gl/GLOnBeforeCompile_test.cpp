// Material::onBeforeCompile on the GL renderer.
//
// The hook edits a built-in material's GLSL before its program is compiled, and the
// uniforms it declares live on the material (customUniforms) so they can change each
// frame. Programs are keyed by customProgramCacheKey: two hooked materials with no key
// must not share a program (each would draw with whichever hook compiled first), and two
// with the same key must.

#include "gl_test_helpers.hpp"

namespace {

    // Paint the surface with uTint after the material has done its own shading.
    std::function<void(Shader&)> tintHook(const Color& initial) {
        return [initial](Shader& s) {
            s.uniforms["uTint"] = Uniform(initial);
            s.fragmentShader = "uniform vec3 uTint;\n" + s.fragmentShader;
            const std::string at = "#include <opaque_fragment>";
            const auto p = s.fragmentShader.find(at);
            REQUIRE(p != std::string::npos);
            s.fragmentShader.replace(p, at.size(), at + "\n\tgl_FragColor.rgb = uTint;");
        };
    }

    struct Pair {
        std::shared_ptr<Scene> scene;
        std::shared_ptr<PerspectiveCamera> camera;
        std::shared_ptr<MeshBasicMaterial> left, right;
    };

    // Two planes, each filling one half of the frame.
    Pair makePair() {
        Pair p;
        p.scene = Scene::create();
        p.left = MeshBasicMaterial::create();
        p.right = MeshBasicMaterial::create();
        auto a = Mesh::create(PlaneGeometry::create(10, 20), p.left);
        auto b = Mesh::create(PlaneGeometry::create(10, 20), p.right);
        a->position.x = -5;
        b->position.x = 5;
        p.scene->add(a);
        p.scene->add(b);
        p.camera = PerspectiveCamera::create(60, 1.f, 0.1f, 100.f);
        p.camera->position.set(0, 0, 6);
        return p;
    }

    AvgColor pixelAt(const std::vector<unsigned char>& px, int x, int y) {
        const size_t i = (static_cast<size_t>(y) * gltest::RT_WIDTH + x) * 3;
        return {static_cast<double>(px[i]), static_cast<double>(px[i + 1]), static_cast<double>(px[i + 2])};
    }

    bool isRed(const AvgColor& c) { return c.r > 200 && c.g < 40 && c.b < 40; }
    bool isGreen(const AvgColor& c) { return c.g > 200 && c.r < 40 && c.b < 40; }
    bool isBlue(const AvgColor& c) { return c.b > 200 && c.r < 40 && c.g < 40; }

}// namespace

TEST_CASE("onBeforeCompile's GLSL runs, and its uniform can change between frames") {

    auto p = makePair();
    p.left->onBeforeCompile = tintHook(Color(1, 0, 0));
    p.right->onBeforeCompile = tintHook(Color(1, 0, 0));

    GLRenderer renderer(glCanvas());
    renderer.render(*p.scene, *p.camera);
    auto px = renderer.readRGBPixels();
    CHECK(isRed(pixelAt(px, 16, 32)));
    REQUIRE(p.left->customUniforms.contains("uTint"));

    p.left->customUniforms["uTint"].setValue(Color(0, 1, 0));
    renderer.render(*p.scene, *p.camera);
    px = renderer.readRGBPixels();
    CHECK(isGreen(pixelAt(px, 16, 32)));
    CHECK(isRed(pixelAt(px, 48, 32)));// the other material keeps its own value
}

TEST_CASE("Hooked materials without a key do not share a program") {

    auto p = makePair();
    p.left->onBeforeCompile = tintHook(Color(1, 0, 0));
    p.right->onBeforeCompile = [](Shader& s) {
        const std::string at = "#include <opaque_fragment>";
        s.fragmentShader.replace(s.fragmentShader.find(at), at.size(), at + "\n\tgl_FragColor.rgb = vec3( 0.0, 0.0, 1.0 );");
    };

    GLRenderer renderer(glCanvas());
    renderer.render(*p.scene, *p.camera);
    const auto px = renderer.readRGBPixels();
    CHECK(isRed(pixelAt(px, 16, 32)));
    CHECK(isBlue(pixelAt(px, 48, 32)));
}

TEST_CASE("Hooked materials with the same key share one program") {

    // The right-hand hook writes different GLSL (half the tint). With a shared key the
    // program compiled for the left is used for both, so the right still draws full
    // strength; the hook runs for both (as in three.js) and so registers both uniforms.
    auto halfTint = [](Shader& s) {
        s.uniforms["uTint"] = Uniform(Color(0, 0, 1));
        s.fragmentShader = "uniform vec3 uTint;\n" + s.fragmentShader;
        const std::string at = "#include <opaque_fragment>";
        s.fragmentShader.replace(s.fragmentShader.find(at), at.size(), at + "\n\tgl_FragColor.rgb = uTint * 0.5;");
    };

    GLRenderer renderer(glCanvas());

    auto shared = makePair();
    shared.left->onBeforeCompile = tintHook(Color(1, 0, 0));
    shared.right->onBeforeCompile = halfTint;
    shared.left->customProgramCacheKey = shared.right->customProgramCacheKey = "tint";
    renderer.render(*shared.scene, *shared.camera);
    auto px = renderer.readRGBPixels();
    CHECK(isRed(pixelAt(px, 16, 32)));
    CHECK(pixelAt(px, 48, 32).b > 240);// the left's program, the right's own uTint

    auto separate = makePair();
    separate.left->onBeforeCompile = tintHook(Color(1, 0, 0));
    separate.right->onBeforeCompile = halfTint;
    renderer.render(*separate.scene, *separate.camera);
    px = renderer.readRGBPixels();
    INFO("separate programs, right blue " << pixelAt(px, 48, 32).b);
    CHECK(isRed(pixelAt(px, 16, 32)));
    CHECK(pixelAt(px, 48, 32).b < 220);// its own half-strength program
    CHECK(pixelAt(px, 48, 32).b > 120);
}

TEST_CASE("clone() carries the hook, its key and its uniforms") {

    auto m = MeshBasicMaterial::create();
    m->onBeforeCompile = tintHook(Color(0, 1, 0));
    m->customProgramCacheKey = "tint";
    m->customUniforms["uTint"] = Uniform(Color(0, 1, 0));

    auto c = m->clone<MeshBasicMaterial>();
    CHECK(static_cast<bool>(c->onBeforeCompile));
    CHECK(c->customProgramCacheKey == "tint");
    REQUIRE(c->customUniforms.contains("uTint"));

    auto scene = Scene::create();
    scene->add(Mesh::create(PlaneGeometry::create(40, 40), c));
    auto camera = PerspectiveCamera::create(60, 1.f, 0.1f, 100.f);
    camera->position.set(0, 0, 6);
    GLRenderer renderer(glCanvas());
    renderer.render(*scene, *camera);
    CHECK(isGreen(pixelAt(renderer.readRGBPixels(), 32, 32)));
}
