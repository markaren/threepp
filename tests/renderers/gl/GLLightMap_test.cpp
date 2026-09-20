// A light map must reach the shader — and must not take the frame down.
//
// UniformsLib registered the light map's intensity under the key
// "lightMapIntesity" (no 'n'), while GLMaterials reads it as
// uniforms.at("lightMapIntensity"). std::unordered_map::at on a missing key
// throws, so ANY material carrying a lightMap killed the render with
// "invalid unordered_map<K, T> key" the first time it was drawn. Not a wrong
// pixel — no pixels at all.
//
// The first case is the crash gate. The second is the one that keeps a future
// "fix" honest: silencing the throw by switching to operator[] would insert a
// default-constructed uniform and quietly ignore lightMapIntensity, which passes
// a does-it-render check and fails this one.

#include "gl_test_helpers.hpp"

#include "threepp/materials/MeshLambertMaterial.hpp"

namespace {

    // The light map samples uv2, which PlaneGeometry does not carry.
    std::shared_ptr<BufferGeometry> planeWithUv2() {
        auto g = PlaneGeometry::create(2.f, 2.f);
        const auto uv = g->getAttribute<float>("uv");
        std::vector<float> uv2(uv->array().begin(), uv->array().end());
        g->setAttribute("uv2", FloatBufferAttribute::create(uv2, 2));
        return g;
    }

    std::shared_ptr<Texture> whiteLightMap() {
        auto tex = Texture::create(Image(std::vector<unsigned char>{255, 255, 255, 255}, 1, 1));
        tex->colorSpace = ColorSpace::Linear;
        tex->needsUpdate();
        return tex;
    }

    // Average brightness of a plane filling the frame, lit ONLY by its light map.
    double renderLightMapped(const std::shared_ptr<Material>& mat) {
        auto scene = Scene::create();
        scene->add(Mesh::create(planeWithUv2(), mat));

        PerspectiveCamera camera(50, 1.f, 0.1f, 100.f);
        camera.position.set(0, 0, 1.6f);
        camera.lookAt({0, 0, 0});

        return avgBrightness(renderWithGL(*scene, camera, Color(0x000000)));
    }

}// namespace

TEST_CASE("a light map does not throw, and lights the surface") {

    auto basic = MeshBasicMaterial::create();
    basic->color = Color(1, 1, 1);
    basic->lightMap = whiteLightMap();
    basic->lightMapIntensity = 1.f;

    double lit = 0;
    REQUIRE_NOTHROW(lit = renderLightMapped(basic));
    INFO("mean brightness with a white light map: " << lit);
    CHECK(lit > 10.0);

    // The other shader that consumes a light map.
    auto lambert = MeshLambertMaterial::create();
    lambert->color = Color(1, 1, 1);
    lambert->lightMap = whiteLightMap();
    lambert->lightMapIntensity = 1.f;

    double litLambert = 0;
    REQUIRE_NOTHROW(litLambert = renderLightMapped(lambert));
    INFO("mean brightness, lambert: " << litLambert);
    CHECK(litLambert > 10.0);
}

TEST_CASE("lightMapIntensity scales the light map's contribution") {

    auto make = [](float intensity) {
        auto m = MeshBasicMaterial::create();
        m->color = Color(1, 1, 1);
        m->lightMap = whiteLightMap();
        m->lightMapIntensity = intensity;
        return m;
    };

    const double dim = renderLightMapped(make(0.15f));
    const double bright = renderLightMapped(make(0.6f));

    INFO("intensity 0.15 -> " << dim << ", intensity 0.6 -> " << bright);
    REQUIRE(dim > 1.0);// both actually drew something
    CHECK(bright > dim * 1.5);
}
