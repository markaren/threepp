// ParticleSprites on the GL renderer: sprites are drawn where they are set, square on
// screen at any aspect ratio, hidden ones draw nothing, and the anisotropy decides from
// which side a lit sprite is brightest, and self-shadowing dims the sun on a sprite but not
// the light from all round, and a scene with a point light still draws them.
// VulkanParticleSprites_test checks the same on Vulkan.

#include "gl_test_helpers.hpp"

#include "threepp/objects/ParticleSprites.hpp"

namespace {

    int lum(const std::vector<unsigned char>& px, int x, int y) {
        const size_t i = (static_cast<size_t>(y) * gltest::RT_WIDTH + x) * 3;
        return px[i] + px[i + 1] + px[i + 2];
    }

    // The run of lit pixels through (x, y), along a row and along a column.
    std::pair<int, int> extent(const std::vector<unsigned char>& px, int x, int y, int threshold) {
        int w = 0, h = 0;
        for (int i = 0; i < gltest::RT_WIDTH; ++i) w += lum(px, i, y) > threshold;
        for (int j = 0; j < gltest::RT_HEIGHT; ++j) h += lum(px, x, j) > threshold;
        return {w, h};
    }

    struct Setup {
        std::shared_ptr<Scene> scene = Scene::create();
        std::shared_ptr<PerspectiveCamera> camera;
        std::shared_ptr<ParticleSprites> sprites = ParticleSprites::create(4);

        explicit Setup(float aspect) {
            camera = PerspectiveCamera::create(50, aspect, 0.1f, 100.f);
            camera->position.set(0, 0, 5);
            camera->lookAt(Vector3());
            sprites->hideAll();
            scene->add(sprites);
        }
    };

}// namespace

TEST_CASE("A sprite is drawn square on a wide view") {

    // A 2:1 view in the middle of the 64x64 target: one scale for both axes stretched
    // the sprite to twice its height.
    Setup s(2.f);
    s.scene->add(AmbientLight::create(Color(1, 1, 1), math::PI));// lit colour = albedo
    s.sprites->set(0, Vector3(), 1.6f, 1.f);
    s.sprites->commit();

    GLRenderer renderer(glCanvas());
    renderer.setViewport(0, 16, 64, 32);
    renderer.render(*s.scene, *s.camera);
    const auto px = renderer.readRGBPixels();

    const auto [w, h] = extent(px, 32, 32, 60);
    INFO("lit run " << w << " px wide, " << h << " px tall");
    REQUIRE(h > 4);
    CHECK(std::abs(w - h) <= 2);
}

TEST_CASE("Hidden sprites draw nothing") {

    Setup s(1.f);
    s.scene->add(AmbientLight::create(Color(1, 1, 1), math::PI));
    s.sprites->set(0, Vector3(), 2.f, 1.f);
    s.sprites->set(1, Vector3(0.5f, 0, 0), 2.f, 0.f);// opacity 0 hides it too
    s.sprites->hide(0);
    s.sprites->commit();

    GLRenderer renderer(glCanvas());
    renderer.render(*s.scene, *s.camera);
    CHECK(gltest::countNonBlack(renderer.readRGBPixels()) == 0);
}

TEST_CASE("Anisotropy decides from which side a lit sprite is brightest") {

    // centre brightness of one grey sprite lit by a sun at `sunZ`, for a given g
    auto centre = [](float sunZ, float g) {
        Setup s(1.f);
        auto sun = DirectionalLight::create(Color(1, 1, 1), 0.6f);
        sun->position.set(0, 0, sunZ);
        s.scene->add(sun);
        s.sprites->setAnisotropy(g);
        s.sprites->set(0, Vector3(), 2.f, 1.f, Color(0.5f, 0.5f, 0.5f));
        s.sprites->commit();
        GLRenderer renderer(glCanvas());
        renderer.render(*s.scene, *s.camera);
        return lum(renderer.readRGBPixels(), 32, 32);
    };

    // sun behind the camera: a dense billow (g < 0) shows its lit face
    const int behindDense = centre(10.f, -0.2f), behindThin = centre(10.f, 0.35f);
    // looking into the sun: thin haze (g > 0) glows
    const int aheadDense = centre(-10.f, -0.2f), aheadThin = centre(-10.f, 0.35f);
    INFO("sun behind: dense " << behindDense << ", thin " << behindThin
                              << "; sun ahead: dense " << aheadDense << ", thin " << aheadThin);
    CHECK(behindDense > behindThin + 30);
    CHECK(aheadThin > aheadDense + 30);
}

TEST_CASE("Self-shadowing dims the sun on a sprite, not the ambient light") {

    // centre brightness of one grey sprite, lit by a sun (or an ambient light) alone
    auto centre = [](bool sunLit, float k) {
        Setup s(1.f);
        if (sunLit) {
            auto sun = DirectionalLight::create(Color(1, 1, 1), 0.6f);
            sun->position.set(0, 0, 10);
            s.scene->add(sun);
        } else {
            s.scene->add(AmbientLight::create(Color(1, 1, 1), 1.5f));
        }
        s.sprites->setSelfShadowing(k);
        s.sprites->set(0, Vector3(), 2.f, 1.f, Color(0.5f, 0.5f, 0.5f));
        s.sprites->commit();
        GLRenderer renderer(glCanvas());
        renderer.render(*s.scene, *s.camera);
        return lum(renderer.readRGBPixels(), 32, 32);
    };

    const int sunOpen = centre(true, 0.f), sunDense = centre(true, 0.5f);
    const int ambOpen = centre(false, 0.f), ambDense = centre(false, 0.9f);
    INFO("sun: " << sunOpen << " -> " << sunDense << "; ambient: " << ambOpen << " -> " << ambDense);
    CHECK(sunOpen > sunDense + 30);
    CHECK(std::abs(ambOpen - ambDense) <= 3);
    CHECK(ambOpen > 30);
}

TEST_CASE("A scene with a point light still draws its sprites") {

    // The sprite's shader takes lights_pars_begin, whose point and spot lights need bsdfs'
    // distance falloff: without it a scene with such a light failed to compile the program
    // and drew no sprites at all. (The sprites take their light from the suns, the ambient
    // and the hemisphere lights; a point light does not light them.)
    auto centre = [](bool withLamp) {
        Setup s(1.f);
        s.scene->add(AmbientLight::create(Color(1, 1, 1), math::PI));
        if (withLamp) {
            auto lamp = PointLight::create(Color(1, 1, 1), 6.f, 0.f, 2.f);
            lamp->position.set(0, 0, 1.5f);
            s.scene->add(lamp);
        }
        s.sprites->set(0, Vector3(), 2.f, 1.f, Color(0.5f, 0.5f, 0.5f));
        s.sprites->commit();
        GLRenderer renderer(glCanvas());
        renderer.render(*s.scene, *s.camera);
        return lum(renderer.readRGBPixels(), 32, 32);
    };

    const int without = centre(false), with = centre(true);
    INFO("without a point light " << without << ", with one " << with);
    CHECK(without > 30);
    CHECK(with == without);
}

TEST_CASE("ParticleSprites keeps its settings in range") {

    auto s = ParticleSprites::create(3);
    CHECK(s->capacity() == 3);
    CHECK(s->anisotropy() == 0.35f);
    s->setAnisotropy(-4.f);
    CHECK(s->anisotropy() == -0.9f);
    CHECK(s->selfShadowing() == 0.f);
    s->setSelfShadowing(2.f);
    CHECK(s->selfShadowing() == 0.95f);
    s->setSelfShadowing(-1.f);
    CHECK(s->selfShadowing() == 0.f);
    s->set(3, Vector3(), 1.f, 1.f);// past the end: ignored
    s->hide(7);
    CHECK_THROWS(ParticleSprites::create(0));
}
