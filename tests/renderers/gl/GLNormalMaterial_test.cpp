// MeshNormalMaterial must reach the shader.
//
// refreshMaterialUniforms dispatches on the exact string Material::type() and had
// no "MeshNormalMaterial" case, so nothing was ever uploaded for one. Two distinct
// consequences, and the second is the severe one:
//
//   - opacity stayed at the ShaderLib default of 1, so a transparent normal
//     material rendered fully opaque;
//   - the surface-detail map samplers were left UNASSIGNED, and the upload path
//     reads an empty variant for those, so a MeshNormalMaterial carrying a
//     normalMap did not render wrong — it threw "bad variant access" and took the
//     frame with it.
//
// r129 handles this as refreshUniformsCommon + refreshUniformsNormal
// (WebGLMaterials.js:72).

#include "gl_test_helpers.hpp"

#include "threepp/materials/MeshNormalMaterial.hpp"

namespace {

    // A pale backdrop, so blending a normal material over it moves the pixels
    // toward a value no opaque normal material would produce.
    const Color kBackdrop{0xdddddd};

    std::vector<unsigned char> renderSphere(const std::shared_ptr<MeshNormalMaterial>& mat) {
        auto scene = Scene::create();
        scene->add(Mesh::create(SphereGeometry::create(1.1f, 48, 32), mat));

        PerspectiveCamera camera(50, 1.f, 0.1f, 100.f);
        camera.position.set(0, 0, 3.2f);
        camera.lookAt({0, 0, 0});

        return renderWithGL(*scene, camera, kBackdrop);
    }

    std::shared_ptr<Texture> checkerNormalMap() {
        constexpr int S = 32;
        std::vector<unsigned char> d(static_cast<size_t>(S) * S * 4);
        for (int y = 0; y < S; ++y)
            for (int x = 0; x < S; ++x) {
                const size_t i = (static_cast<size_t>(y) * S + x) * 4;
                const bool bump = ((x / 4) + (y / 4)) % 2 == 0;
                d[i + 0] = bump ? 220 : 35;
                d[i + 1] = bump ? 35 : 220;
                d[i + 2] = 255;
                d[i + 3] = 255;
            }
        auto tex = Texture::create(Image(std::move(d), S, S));
        tex->colorSpace = ColorSpace::Linear;
        tex->needsUpdate();
        return tex;
    }

}// namespace

TEST_CASE("MeshNormalMaterial honours opacity") {

    auto opaque = MeshNormalMaterial::create();
    const auto solid = renderSphere(opaque);

    auto faint = MeshNormalMaterial::create();
    faint->transparent = true;
    faint->opacity = 0.25f;
    const auto ghost = renderSphere(faint);

    REQUIRE(solid.size() == DATA_SIZE);
    REQUIRE(ghost.size() == DATA_SIZE);

    // Blended toward a pale backdrop, the sphere must end up markedly brighter
    // and markedly less colourful than the opaque one. Left inert, the two
    // renders are identical.
    const double solidBright = avgBrightness(solid);
    const double ghostBright = avgBrightness(ghost);

    INFO("mean brightness: opaque " << solidBright << ", opacity 0.25 " << ghostBright);
    CHECK(ghostBright > solidBright + 8.0);
}

TEST_CASE("MeshNormalMaterial with a normalMap renders instead of throwing") {

    auto plain = MeshNormalMaterial::create();
    const auto flat = renderSphere(plain);

    auto mapped = MeshNormalMaterial::create();
    mapped->normalMap = checkerNormalMap();
    mapped->normalScale.set(1.f, 1.f);

    std::vector<unsigned char> bumpy;
    REQUIRE_NOTHROW(bumpy = renderSphere(mapped));
    REQUIRE(bumpy.size() == DATA_SIZE);

    // And the map has to actually perturb the encoded normals, not just survive.
    int differing = 0;
    for (int i = 0; i < PIXEL_COUNT; ++i) {
        const int d = std::abs(int(bumpy[i * 3]) - int(flat[i * 3])) +
                      std::abs(int(bumpy[i * 3 + 1]) - int(flat[i * 3 + 1])) +
                      std::abs(int(bumpy[i * 3 + 2]) - int(flat[i * 3 + 2]));
        if (d > 24) ++differing;
    }
    INFO("pixels changed by the normal map: " << differing << " / " << PIXEL_COUNT);
    CHECK(differing > PIXEL_COUNT / 10);
}
