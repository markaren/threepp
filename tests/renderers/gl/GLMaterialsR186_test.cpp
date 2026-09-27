// Material behaviours brought up to three.js r186 on the GL path (phase 3 of
// the GL uplift): per-fragment Lambert, per-map UV transforms and channels,
// two-pass double-sided transparency, bounding-sphere transparent sort, flat
// shading for geometry without normals, and alphaHash.

#include "gl_test_helpers.hpp"

#include "threepp/extras/environments/RoomEnvironment.hpp"
#include "threepp/materials/MeshLambertMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/renderers/gl/PMREMGenerator.hpp"

using namespace gltest;

namespace {

    std::shared_ptr<PerspectiveCamera> makeCamera(float z) {
        auto camera = PerspectiveCamera::create(50, 1.f, 0.1f, 100.f);
        camera->position.set(0, 0, z);
        camera->lookAt({0, 0, 0});
        camera->updateProjectionMatrix();
        return camera;
    }

    int red(const std::vector<unsigned char>& px, int x, int y) {
        return px[(y * RT_WIDTH + x) * 3];
    }

}// namespace

TEST_CASE("a close point light peaks where the coarse Lambert sphere is nearest, not on a vertex", "[gl][lambert]") {

    auto scene = Scene::create();
    auto mat = MeshLambertMaterial::create();
    mat->color = Color(0.8f, 0.8f, 0.8f);
    // widthSegments 5, heightSegments 3: no vertex anywhere near (0, 0, 1).
    scene->add(Mesh::create(SphereGeometry::create(1.f, 5, 3), mat));
    auto light = PointLight::create(0xffffff, 0.5f, 0, 2);
    light->position.set(0, 0, 1.6f);
    scene->add(light);

    auto camera = makeCamera(4.f);
    const auto px = renderWithGL(*scene, *camera, Color(0x000000));

    int maxR = 0;
    for (int i = 0; i < PIXEL_COUNT; ++i) maxR = std::max(maxR, static_cast<int>(px[i * 3]));
    REQUIRE(maxR > 40);
    REQUIRE(maxR < 255);// not clipped, so the peak is meaningful

    double sx = 0, sy = 0;
    int n = 0;
    for (int y = 0; y < RT_HEIGHT; ++y) {
        for (int x = 0; x < RT_WIDTH; ++x) {
            if (red(px, x, y) >= maxR - 2) {
                sx += x;
                sy += y;
                ++n;
            }
        }
    }
    const double dx = sx / n - (RT_WIDTH - 1) / 2.0;
    const double dy = sy / n - (RT_HEIGHT - 1) / 2.0;
    INFO("peak " << maxR << " centroid offset (" << dx << ", " << dy << ") px over " << n << " px");
    // Gouraud puts the peak on a vertex ~10 px off centre at this size.
    CHECK(std::sqrt(dx * dx + dy * dy) < 3.0);
}

TEST_CASE("a Lambert sphere is lit by scene.environment alone", "[gl][lambert]") {

    GLRenderer renderer(glCanvas());
    renderer.setClearColor(Color(0x000000));
    auto scene = Scene::create();
    {
        PMREMGenerator pmrem(renderer);
        RoomEnvironment room;
        scene->environment = pmrem.fromScene(room, 0.04f);
    }
    auto mat = MeshLambertMaterial::create();
    mat->color = Color(0.8f, 0.8f, 0.8f);
    scene->add(Mesh::create(SphereGeometry::create(1.f, 32, 16), mat));

    auto camera = makeCamera(4.f);
    renderer.render(*scene, *camera);
    const auto c = centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT);
    INFO("centre " << c.r << ", " << c.g << ", " << c.b);
    CHECK(c.r > 60);// black before r184 #32791
}

namespace {

    // Tangent-space normal map of one dome per tile.
    std::shared_ptr<Texture> domeNormalMap() {
        constexpr int N = 32;
        std::vector<unsigned char> nrm(static_cast<size_t>(N) * N * 4);
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / N * 2.f - 1.f;
                const float v = (static_cast<float>(y) + 0.5f) / N * 2.f - 1.f;
                const float nx = u * 0.8f, ny = v * 0.8f;
                const float nz = std::sqrt(std::max(0.f, 1.f - nx * nx - ny * ny));
                const size_t i = (static_cast<size_t>(y) * N + x) * 4;
                nrm[i + 0] = static_cast<unsigned char>((nx * 0.5f + 0.5f) * 255);
                nrm[i + 1] = static_cast<unsigned char>((ny * 0.5f + 0.5f) * 255);
                nrm[i + 2] = static_cast<unsigned char>((nz * 0.5f + 0.5f) * 255);
                nrm[i + 3] = 255;
            }
        }
        auto tex = Texture::create(Image(std::move(nrm), N, N));
        tex->wrapS = tex->wrapT = TextureWrapping::Repeat;
        tex->needsUpdate();
        return tex;
    }

    std::shared_ptr<Texture> twoTexelMap() {
        // Left texel red, right texel green.
        auto tex = Texture::create(Image(std::vector<unsigned char>{255, 0, 0, 255, 0, 255, 0, 255}, 2, 1));
        tex->magFilter = Filter::Nearest;
        tex->minFilter = Filter::Nearest;
        tex->generateMipmaps = false;
        tex->needsUpdate();
        return tex;
    }

}// namespace

TEST_CASE("a normalMap with repeat 8 shades at 8x the frequency of a repeat-1 map", "[gl][uv]") {

    auto scene = Scene::create();
    auto mat = MeshStandardMaterial::create();
    auto white = Texture::create(Image(std::vector<unsigned char>{255, 255, 255, 255}, 1, 1));
    white->needsUpdate();
    mat->map = white;// repeat 1; before r152 its transform drove every map
    mat->normalMap = domeNormalMap();
    mat->normalMap->repeat.set(8, 8);
    mat->roughness = 1.f;
    scene->add(Mesh::create(PlaneGeometry::create(2, 2), mat));
    auto light = DirectionalLight::create(0xffffff, 2.f);
    light->position.set(-1, 0, 0.4f);
    scene->add(light);

    auto camera = makeCamera(1.1f / std::tan(math::degToRad(25.f)));
    const auto px = renderWithGL(*scene, *camera, Color(0x000000));

    // Count bright-to-dark transitions along the middle row.
    const int y = RT_HEIGHT / 2;
    int mean = 0;
    for (int x = 0; x < RT_WIDTH; ++x) mean += red(px, x, y);
    mean /= RT_WIDTH;
    int crossings = 0;
    for (int x = 1; x < RT_WIDTH; ++x) {
        if (red(px, x - 1, y) >= mean && red(px, x, y) < mean) ++crossings;
    }
    INFO("mean " << mean << ", falling crossings " << crossings);
    CHECK(crossings >= 6);// one dome per 8 px; repeat 1 gives at most 1
}

TEST_CASE("Texture::channel 1 samples the uv2 attribute", "[gl][uv]") {

    auto render = [](int channel) {
        auto scene = Scene::create();
        auto geo = PlaneGeometry::create(2, 2);
        const auto count = geo->getAttribute<float>("uv")->count();
        std::vector<float> uv2;
        for (int i = 0; i < count; ++i) {
            uv2.push_back(0.25f);// every vertex at the red texel
            uv2.push_back(0.5f);
        }
        geo->setAttribute("uv2", FloatBufferAttribute::create(uv2, 2));
        auto mat = MeshBasicMaterial::create();
        mat->map = twoTexelMap();
        mat->map->channel = channel;
        scene->add(Mesh::create(geo, mat));
        auto camera = makeCamera(1.1f / std::tan(math::degToRad(25.f)));
        return renderWithGL(*scene, *camera, Color(0x000000));
    };

    const auto uv = render(0);
    const auto uv2 = render(1);
    const int x0 = RT_WIDTH / 4, x1 = 3 * RT_WIDTH / 4, y = RT_HEIGHT / 2;
    // channel 0: left red, right green; channel 1: red everywhere.
    CHECK((red(uv, x0, y) > 200 && red(uv, x1, y) < 50 && red(uv2, x0, y) > 200 && red(uv2, x1, y) > 200));
}

TEST_CASE("a transparent double-sided sphere draws its back faces under its front faces everywhere", "[gl][transparency]") {

    auto scene = Scene::create();
    auto mat = MeshBasicMaterial::create();
    mat->color = Color(1, 0, 0);
    mat->transparent = true;
    mat->opacity = 0.5f;
    mat->side = Side::Double;
    scene->add(Mesh::create(SphereGeometry::create(1.f, 32, 16), mat));

    auto camera = makeCamera(4.f);
    const auto px = renderWithGL(*scene, *camera, Color(0x000000));

    // Both faces blended: 0.5 + 0.5 * 0.5 = 0.75 (191 in this linear-output
    // harness); a front face drawn before its back face leaves 0.5 (128).
    const double rPx = RT_HEIGHT / 2.0 / (4.f * std::tan(math::degToRad(25.f)));
    int minR = 255;
    for (int y = 0; y < RT_HEIGHT; ++y) {
        for (int x = 0; x < RT_WIDTH; ++x) {
            const double dx = x - (RT_WIDTH - 1) / 2.0, dy = y - (RT_HEIGHT - 1) / 2.0;
            if (std::sqrt(dx * dx + dy * dy) < 0.7 * rPx) minR = std::min(minR, red(px, x, y));
        }
    }
    INFO("min red inside the sphere " << minR);
    CHECK(minR >= 180);
}

TEST_CASE("transparent meshes sort on their geometry's bounding sphere, not their origin", "[gl][transparency]") {

    auto scene = Scene::create();
    auto redMat = MeshBasicMaterial::create();
    redMat->color = Color(1, 0, 0);
    redMat->transparent = true;
    redMat->opacity = 0.6f;
    auto redGeo = PlaneGeometry::create(2, 2);
    redGeo->translate(0, 0, -6.f);
    auto redMesh = Mesh::create(redGeo, redMat);
    redMesh->position.z = 3.f;// origin near, geometry at z = -3
    scene->add(redMesh);

    auto blueMat = MeshBasicMaterial::create();
    blueMat->color = Color(0, 0, 1);
    blueMat->transparent = true;
    blueMat->opacity = 0.6f;
    auto blueGeo = PlaneGeometry::create(2, 2);
    blueGeo->translate(0, 0, 5.f);
    auto blueMesh = Mesh::create(blueGeo, blueMat);
    blueMesh->position.z = -5.f;// origin far, geometry at z = 0
    scene->add(blueMesh);

    auto camera = makeCamera(6.f);
    const auto c = centerPixel(renderWithGL(*scene, *camera, Color(0x000000)), RT_WIDTH, RT_HEIGHT);
    INFO("centre " << c.r << ", " << c.g << ", " << c.b);
    // Blue over red: red keeps 0.4 * 0.6 = 0.24 (61 in this linear-output
    // harness). Origin order draws blue first and depth-rejects red (0).
    CHECK(c.r > 40);
}

TEST_CASE("a lit mesh whose geometry has no normals is shaded, not black", "[gl][flat]") {

    auto scene = Scene::create();
    const std::vector<float> p{
            0, 1, 0, -1, -0.6f, 0.8f, 1, -0.6f, 0.8f,
            0, 1, 0, 1, -0.6f, 0.8f, 0, -0.6f, -1,
            0, 1, 0, 0, -0.6f, -1, -1, -0.6f, 0.8f};
    auto geo = BufferGeometry::create();
    geo->setAttribute("position", FloatBufferAttribute::create(p, 3));
    auto mat = MeshStandardMaterial::create();
    scene->add(Mesh::create(geo, mat));
    auto light = DirectionalLight::create(0xffffff, 2.f);
    light->position.set(0.3f, 0.5f, 1);
    scene->add(light);

    auto camera = makeCamera(4.f);
    const auto px = renderWithGL(*scene, *camera, Color(0x000000));
    CHECK(countNonBlack(px, 20) > 100);
}

TEST_CASE("alphaHash coverage tracks opacity", "[gl][alphahash]") {

    auto coverage = [](float opacity) {
        auto scene = Scene::create();
        auto mat = MeshBasicMaterial::create();
        mat->color = Color(1, 1, 1);
        mat->opacity = opacity;
        mat->alphaHash = true;
        scene->add(Mesh::create(PlaneGeometry::create(2, 2), mat));
        auto camera = makeCamera(1.f / std::tan(math::degToRad(25.f)));
        const auto px = renderWithGL(*scene, *camera, Color(0x000000));
        int on = 0;
        for (int i = 0; i < PIXEL_COUNT; ++i) on += px[i * 3] > 128 ? 1 : 0;
        return static_cast<double>(on) / PIXEL_COUNT;
    };

    const double c30 = coverage(0.3f);
    const double c70 = coverage(0.7f);
    INFO("coverage at 0.3: " << c30 << ", at 0.7: " << c70);
    CHECK((std::abs(c30 - 0.3) < 0.1 && std::abs(c70 - 0.7) < 0.1));
}
