// White furnace parity tests — Phase 0 of the color/tone/lighting unification.
//
// Setup: a diffuse-white sphere (ρ=1, roughness=1, metalness=0) lit only by a
// uniform-radiance environment (Le=1 in every direction). For a Lambertian
// surface, L_o = (ρ/π) · ∫ L_i cos θ dω = (1/π) · Le · π = Le = 1.0, so every
// pixel on the sphere must read 1.0.
//
// outputColorSpace=NoColorSpace + toneMapping=None → readback bytes are raw
// linear, so 1.0 ↔ 255. Center pixel is always inside the sphere's projection
// (camera framing keeps it covered).
//
// Why two tests, not three: the path-tracer furnace has its own runner
// (wgpu_furnace_env_test.cpp) that needs many frames to converge; bringing
// that into a unit test would slow CI. PT parity is checked by Phase 2's gate
// against this raster reference instead.

#include "CrossRenderer_helpers.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/textures/Texture.hpp"

namespace {

    // 8x4 equirect, Le = (1,1,1,1) per texel. Same construction as the
    // wgpu_furnace_env_test runner so any IBL math difference is exposed.
    std::shared_ptr<Texture> makeConstantEnv() {
        constexpr int W = 8, H = 4;
        std::vector<float> data(W * H * 4, 1.f);
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    // Top-lit hemisphere env: Le = 1 above the horizon, 0 below (equirect v
    // maps y=+1 to v=1, i.e. the LAST data rows). A constant env cannot catch
    // a diffuse-IBL path that has lost its directionality (any normalized
    // average of a constant is the constant); this one can. Analytic
    // reference for a Lambertian ρ=1 sphere under a uniform hemispherical
    // source: E(γ) = πL(1+cosγ)/2 with γ = angle(normal, up), so
    // L_o = (1+cosγ)/2 — sphere top ≈ 1.0, equator ≈ 0.5, bottom ≈ 0.
    // 64x32 so the prefilters resolve the horizon without the whole env
    // collapsing into one bilinear footprint.
    std::shared_ptr<Texture> makeTopLitEnv() {
        constexpr int W = 64, H = 32;
        std::vector<float> data(static_cast<size_t>(W) * H * 4, 0.f);
        for (int y = H / 2; y < H; ++y)// v >= 0.5 → upper hemisphere
            for (int x = 0; x < W; ++x) {
                const size_t i = (static_cast<size_t>(y) * W + x) * 4;
                data[i + 0] = data[i + 1] = data[i + 2] = 1.f;
                data[i + 3] = 1.f;
            }
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    std::shared_ptr<Scene> makeFurnaceScene() {
        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);
        scene->environment = makeConstantEnv();

        auto mat = MeshStandardMaterial::create();
        mat->color = Color(1, 1, 1);
        mat->roughness = 1.f;
        mat->metalness = 0.f;
        auto sphere = Mesh::create(SphereGeometry::create(1.f, 32, 32), mat);
        scene->add(sphere);
        return scene;
    }

    std::shared_ptr<PerspectiveCamera> makeFurnaceCamera() {
        auto camera = PerspectiveCamera::create(45, 1.0f, 0.1f, 100.f);
        camera->position.set(0, 0, 2.4f);
        camera->lookAt(Vector3{0, 0, 0});
        return camera;
    }

}// namespace

TEST_CASE("Furnace: GL diffuse-white sphere in unit env reads ~1.0", "[furnace]") {
    auto scene = makeFurnaceScene();
    auto camera = makeFurnaceCamera();

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.f;
    renderer.setClearColor(Color(0, 0, 0));
    renderer.render(*scene, *camera);

    auto pixels = renderer.readRGBPixels();
    REQUIRE(pixels.size() == DATA_SIZE);

    auto c = centerPixel(pixels, RT_WIDTH, RT_HEIGHT);
    INFO("GL center RGB: " << c.r << ", " << c.g << ", " << c.b);

    // Phase 0 target is 255 ± 1 on all three renderers; today GL is the
    // reference so we check it tightly. Raster IBL has small numerical drift
    // from PMREM sampling — 5 LSB is comfortable headroom.
    CHECK(std::abs(c.r - 255.0) < 5.0);
    CHECK(std::abs(c.g - 255.0) < 5.0);
    CHECK(std::abs(c.b - 255.0) < 5.0);
}

TEST_CASE("Furnace: Wgpu diffuse-white sphere in unit env reads ~1.0", "[furnace][wgpu]") {
    REQUIRE_WGPU();
    SKIP_ON_SOFTWARE_ADAPTER();

    auto scene = makeFurnaceScene();
    auto camera = makeFurnaceCamera();

    WgpuRenderer renderer(wgpuCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.f;
    renderer.setClearColor(Color(0, 0, 0));

    auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());
    renderer.render(*scene, *camera);

    auto pixels = renderer.readRGBPixels();
    renderer.setRenderTarget(nullptr);
    renderer.dispose();

    REQUIRE(pixels.size() == DATA_SIZE);

    auto c = centerPixel(pixels, RT_WIDTH, RT_HEIGHT);
    INFO("Wgpu center RGB: " << c.r << ", " << c.g << ", " << c.b);

    // Parity gate met (GL 255, Wgpu 254 after the diffuse-IBL π fix): both
    // backends now carry the full E = π·PMREM-mean irradiance. Tight like GL.
    CHECK(std::abs(c.r - 255.0) < 5.0);
    CHECK(std::abs(c.g - 255.0) < 5.0);
    CHECK(std::abs(c.b - 255.0) < 5.0);
}

// Directional complement to the constant furnace: catches diffuse-IBL paths
// that lost their directionality (e.g. sampling a 1×1 top mip — a single
// convolved direction sprayed onto every normal — reads fine in a constant
// env but flattens E(n) to a constant here). Probes three rows of the sphere
// (camera framing makes the sphere cover the full 64² frame): top ≈ (1+cosγ)/2
// near 1, centre ≈ 0.5, bottom ≈ 0, and GL/Wgpu must agree per-probe.
TEST_CASE("Furnace: top-lit env diffuse irradiance is directional and matches", "[furnace][wgpu]") {
    REQUIRE_WGPU();
    SKIP_ON_SOFTWARE_ADAPTER();

    auto makeScene = [] {
        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);
        scene->environment = makeTopLitEnv();
        auto mat = MeshStandardMaterial::create();
        mat->color = Color(1, 1, 1);
        mat->roughness = 1.f;
        mat->metalness = 0.f;
        scene->add(Mesh::create(SphereGeometry::create(1.f, 32, 32), mat));
        return scene;
    };
    auto camera = makeFurnaceCamera();

    // Readback row order differs by backend: glReadPixels returns scanlines
    // BOTTOM-UP (writeFramebuffer flips before encoding for exactly this
    // reason), Wgpu readback is top-down. Probe rows are screen-space
    // (from-top); the helper converts per backend.
    auto probe = [](const std::vector<unsigned char>& px, int rowFromTop, bool bottomUp) {
        const int row = bottomUp ? RT_HEIGHT - 1 - rowFromTop : rowFromTop;
        const int i = (row * RT_WIDTH + RT_WIDTH / 2) * 3;
        return static_cast<double>(px[i]);
    };
    const int topRow = RT_HEIGHT / 8;          // normal ≈ up
    const int midRow = RT_HEIGHT / 2;          // normal ≈ toward camera (horizontal)
    const int botRow = RT_HEIGHT - RT_HEIGHT / 8;// normal ≈ down

    double gl[3], wg[3];
    {
        auto scene = makeScene();
        GLRenderer renderer(glCanvas());
        renderer.outputColorSpace = ColorSpace::NoColorSpace;
        renderer.toneMapping = ToneMapping::None;
        renderer.setClearColor(Color(0, 0, 0));
        renderer.render(*scene, *camera);
        auto px = renderer.readRGBPixels();
        REQUIRE(px.size() == DATA_SIZE);
        gl[0] = probe(px, topRow, true);
        gl[1] = probe(px, midRow, true);
        gl[2] = probe(px, botRow, true);
    }
    {
        auto scene = makeScene();
        WgpuRenderer renderer(wgpuCanvas());
        renderer.outputColorSpace = ColorSpace::NoColorSpace;
        renderer.toneMapping = ToneMapping::None;
        renderer.setClearColor(Color(0, 0, 0));
        auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        renderer.render(*scene, *camera);
        auto px = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
        REQUIRE(px.size() == DATA_SIZE);
        wg[0] = probe(px, topRow, false);
        wg[1] = probe(px, midRow, false);
        wg[2] = probe(px, botRow, false);
    }

    INFO("GL   top/mid/bot: " << gl[0] << ", " << gl[1] << ", " << gl[2]);
    INFO("Wgpu top/mid/bot: " << wg[0] << ", " << wg[1] << ", " << wg[2]);

    // Physics: strong top-to-bottom gradient on BOTH backends. The absolute
    // windows are looser than the analytic (1+cosγ)/2 because both prefilters
    // approximate the cosine lobe with GGX(roughness=1), which is wider and
    // compresses the gradient toward the mean — identically on both, so the
    // parity checks stay tight.
    CHECK(gl[0] > 170.0);
    CHECK(std::abs(gl[1] - 128.0) < 35.0);
    CHECK(gl[2] < 85.0);
    CHECK(gl[0] - gl[2] > 90.0);
    CHECK(wg[0] > 170.0);
    CHECK(std::abs(wg[1] - 128.0) < 35.0);
    CHECK(wg[2] < 85.0);
    CHECK(wg[0] - wg[2] > 90.0);

    // Parity: the two rasters agree per-probe.
    CHECK(std::abs(gl[0] - wg[0]) < 20.0);
    CHECK(std::abs(gl[1] - wg[1]) < 20.0);
    CHECK(std::abs(gl[2] - wg[2]) < 20.0);
}

// Sun-disc furnace: synthetic env with a CLOSED-FORM up-facing irradiance —
// uniform upper hemisphere L0 = 0.25 plus a 4°-radius sun disc (L = 40, below
// the prefilter firefly clamp of 50) at 45° elevation.
//   E(up)/π = L0 + L_s·Ωs·cos(45°)/π,  Ωs = 2π(1−cos 4°)
//           = 0.25 + 40·0.015308·0.7071/π = 0.3878  → byte ≈ 99.
// Guards the failure modes a constant or smooth env cannot see: a prefilter
// that under-samples a small bright source reads ≈ 64 (sun missed — Wgpu's
// old LOD-0 sampling); one that amplifies it reads high (e.g. f16-Inf texels
// going sticky through box mips). Probed on an up-facing white Lambertian
// plate, centre pixel.
TEST_CASE("Furnace: analytic sun-disc irradiance matches closed form", "[furnace][wgpu]") {
    REQUIRE_WGPU();
    SKIP_ON_SOFTWARE_ADAPTER();

    constexpr int W = 512, H = 256;
    const Vector3 sunDir = Vector3(0.f, std::sin(math::PI / 4), std::cos(math::PI / 4)).normalize();
    const float cosSun = std::cos(4.f * math::DEG2RAD);
    std::vector<float> data(static_cast<size_t>(W) * H * 4, 0.f);
    for (int y = 0; y < H; ++y) {
        const float elev = ((y + 0.5f) / H - 0.5f) * math::PI;
        for (int x = 0; x < W; ++x) {
            const float az = ((x + 0.5f) / W - 0.5f) * 2.f * math::PI;
            const Vector3 dir(std::cos(elev) * std::cos(az), std::sin(elev), std::cos(elev) * std::sin(az));
            float L = elev > 0.f ? 0.25f : 0.f;
            if (dir.dot(sunDir) > cosSun) L = 40.f;
            const size_t i = (static_cast<size_t>(y) * W + x) * 4;
            data[i + 0] = data[i + 1] = data[i + 2] = L;
            data[i + 3] = 1.f;
        }
    }
    Image img{std::move(data), W, H, 0};
    auto sunEnv = Texture::create(img);
    sunEnv->format = Format::RGBA;
    sunEnv->type = Type::Float;
    sunEnv->colorSpace = ColorSpace::Linear;
    sunEnv->mapping = Mapping::EquirectangularReflection;
    sunEnv->needsUpdate();

    auto makeScene = [&] {
        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);
        scene->environment = sunEnv;
        auto plate = Mesh::create(PlaneGeometry::create(4, 4),
                                  MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(1, 1, 1)).roughness(1.f).metalness(0.f)));
        plate->rotation.x = -math::PI / 2.f;// face +Y
        scene->add(plate);
        return scene;
    };
    auto camera = PerspectiveCamera::create(60, 1.f, 0.1f, 100.f);
    camera->position.set(0, 2.5f, -1.2f);
    camera->lookAt(Vector3{0, 0, 0});

    double gl, wg;
    {
        auto scene = makeScene();
        GLRenderer renderer(glCanvas());
        renderer.outputColorSpace = ColorSpace::NoColorSpace;
        renderer.toneMapping = ToneMapping::None;
        renderer.setClearColor(Color(0, 0, 0));
        renderer.render(*scene, *camera);
        auto px = renderer.readRGBPixels();
        REQUIRE(px.size() == DATA_SIZE);
        gl = centerPixel(px, RT_WIDTH, RT_HEIGHT).r;
    }
    {
        auto scene = makeScene();
        WgpuRenderer renderer(wgpuCanvas());
        renderer.outputColorSpace = ColorSpace::NoColorSpace;
        renderer.toneMapping = ToneMapping::None;
        renderer.setClearColor(Color(0, 0, 0));
        auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
        renderer.setRenderTarget(target.get());
        renderer.render(*scene, *camera);
        auto px = renderer.readRGBPixels();
        renderer.setRenderTarget(nullptr);
        renderer.dispose();
        REQUIRE(px.size() == DATA_SIZE);
        wg = centerPixel(px, RT_WIDTH, RT_HEIGHT).r;
    }

    INFO("sun-plate analytic 99   GL: " << gl << "   Wgpu: " << wg);
    CHECK(std::abs(gl - 99.0) < 10.0);
    CHECK(std::abs(wg - 99.0) < 10.0);
    CHECK(std::abs(gl - wg) < 8.0);
}

// Hidden diagnostic ([.] = not run by default): the gltf_loader example setup
// (real HDR sky + lightgray floor) printed as GL vs Wgpu region averages.
// Run explicitly when chasing env-lit scene divergence:
//   CrossRenderer_furnace_test "[hdrparity]"
TEST_CASE("Furnace: real-HDR floor/sky parity (diagnostic)", "[.][hdrparity]") {
    REQUIRE_WGPU();
    SKIP_ON_SOFTWARE_ADAPTER();

    RGBELoader loader;
    auto hdr = loader.load(std::string(DATA_FOLDER) + "/textures/env/autumn_field_puresky_2k.hdr");
    REQUIRE(hdr);

    auto makeScene = [&] {
        auto scene = Scene::create();
        scene->background = hdr;
        scene->environment = hdr;
        auto floor = Mesh::create(BoxGeometry::create(10, 0.1f, 10),
                                  MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color::lightgray)));
        scene->add(floor);
        // mid-rough grey sphere ~ the character's material regime; probed
        // per-channel to catch hue (not just brightness) divergence
        auto sphere = Mesh::create(SphereGeometry::create(0.8f, 32, 32),
                                   MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0.12f, 0.12f, 0.12f)).roughness(1.f).metalness(0.f)));
        sphere->position.set(0, 1.2f, 0);
        scene->add(sphere);
        return scene;
    };
    auto camera = PerspectiveCamera::create(60, 1.f, 0.1f, 100.f);
    camera->position.set(0, 2, -4);
    camera->lookAt(Vector3{0, 0, 0});

    // Screen-space halves; `bottomUp` converts for GL's glReadPixels row order.
    auto regionAvg = [](const std::vector<unsigned char>& px, bool topHalf, bool bottomUp) {
        double sum = 0;
        int n = 0;
        for (int y = 0; y < RT_HEIGHT / 2; ++y) {
            const int rowFromTop = topHalf ? y : RT_HEIGHT / 2 + y;
            const int row = bottomUp ? RT_HEIGHT - 1 - rowFromTop : rowFromTop;
            for (int x = 0; x < RT_WIDTH; ++x) {
                const int i = (row * RT_WIDTH + x) * 3;
                sum += (px[i] + px[i + 1] + px[i + 2]) / 3.0;
                ++n;
            }
        }
        return sum / n;
    };

    // rawLinear=true → outputColorSpace NoColorSpace + toneMapping None on
    // both, so the bytes are the linear shading result: separates IBL energy
    // differences (visible here) from output-transform differences (visible
    // only in the default-output pass).
    auto renderBoth = [&](bool rawLinear, double out[4]) {
        {
            auto scene = makeScene();
            GLRenderer renderer(glCanvas());
            if (rawLinear) {
                renderer.outputColorSpace = ColorSpace::NoColorSpace;
                renderer.toneMapping = ToneMapping::None;
            }
            renderer.render(*scene, *camera);
            auto px = renderer.readRGBPixels();
            out[0] = regionAvg(px, true, true); // sky
            out[1] = regionAvg(px, false, true);// floor
        }
        {
            auto scene = makeScene();
            WgpuRenderer renderer(wgpuCanvas());
            if (rawLinear) {
                renderer.outputColorSpace = ColorSpace::NoColorSpace;
                renderer.toneMapping = ToneMapping::None;
            }
            renderer.setClearColor(Color(0, 0, 0));
            auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
            renderer.setRenderTarget(target.get());
            renderer.render(*scene, *camera);
            auto px = renderer.readRGBPixels();
            renderer.setRenderTarget(nullptr);
            renderer.dispose();
            out[2] = regionAvg(px, true, false);
            out[3] = regionAvg(px, false, false);
        }
    };

    // Column scan over the sphere, per backend, raw linear.
    auto sphereProbe = [&](bool isGL, std::string& rgb) {
        auto scene = makeScene();
        std::vector<unsigned char> px;
        if (isGL) {
            GLRenderer renderer(glCanvas());
            renderer.outputColorSpace = ColorSpace::NoColorSpace;
            renderer.toneMapping = ToneMapping::None;
            renderer.render(*scene, *camera);
            px = renderer.readRGBPixels();
        } else {
            WgpuRenderer renderer(wgpuCanvas());
            renderer.outputColorSpace = ColorSpace::NoColorSpace;
            renderer.toneMapping = ToneMapping::None;
            renderer.setClearColor(Color(0, 0, 0));
            auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
            renderer.setRenderTarget(target.get());
            renderer.render(*scene, *camera);
            px = renderer.readRGBPixels();
            renderer.setRenderTarget(nullptr);
            renderer.dispose();
        }
        // The sphere spans screen rows ~2..23 (centre column); scan a few,
        // converting per backend row order (GL readback is bottom-up).
        std::ostringstream os;
        for (int rowFromTop = 6; rowFromTop <= 22; rowFromTop += 4) {
            const int row = isGL ? RT_HEIGHT - 1 - rowFromTop : rowFromTop;
            const int i = (row * RT_WIDTH + RT_WIDTH / 2) * 3;
            os << " [" << rowFromTop << "] " << (int) px[i] << "," << (int) px[i + 1] << "," << (int) px[i + 2];
        }
        rgb = os.str();
    };

    double def[4], lin[4];
    std::string glRgb, wgRgb;
    renderBoth(false, def);
    renderBoth(true, lin);
    sphereProbe(true, glRgb);
    sphereProbe(false, wgRgb);

    WARN("default  GL sky/floor: " << def[0] << " / " << def[1]
         << "   Wgpu sky/floor: " << def[2] << " / " << def[3]
         << "\nlinear   GL sky/floor: " << lin[0] << " / " << lin[1]
         << "   Wgpu sky/floor: " << lin[2] << " / " << lin[3]
         << "\nsphere linear RGB rows  GL:" << glRgb
         << "\nsphere linear RGB rows  Wgpu:" << wgRgb);

    // Transfer-function probe: constant 0.5 float equirect as background only.
    // Raw linear output must read exactly 127.5 on a correct path; an sRGB
    // encode shows ~188, a decode ~54.
    {
        constexpr int W = 64, H = 32;
        std::vector<float> data(static_cast<size_t>(W) * H * 4, 0.5f);
        for (size_t i = 3; i < data.size(); i += 4) data[i] = 1.f;
        Image img{std::move(data), W, H, 0};
        auto flat = Texture::create(img);
        flat->format = Format::RGBA;
        flat->type = Type::Float;
        flat->colorSpace = ColorSpace::Linear;
        flat->mapping = Mapping::EquirectangularReflection;
        flat->needsUpdate();

        auto scene = Scene::create();
        scene->background = flat;

        double glBg, wgBg;
        {
            GLRenderer renderer(glCanvas());
            renderer.outputColorSpace = ColorSpace::NoColorSpace;
            renderer.toneMapping = ToneMapping::None;
            renderer.render(*scene, *camera);
            auto px = renderer.readRGBPixels();
            glBg = avgBrightness(px);
        }
        {
            WgpuRenderer renderer(wgpuCanvas());
            renderer.outputColorSpace = ColorSpace::NoColorSpace;
            renderer.toneMapping = ToneMapping::None;
            renderer.setClearColor(Color(0, 0, 0));
            auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
            renderer.setRenderTarget(target.get());
            renderer.render(*scene, *camera);
            auto px = renderer.readRGBPixels();
            renderer.setRenderTarget(nullptr);
            renderer.dispose();
            wgBg = avgBrightness(px);
        }
        WARN("flat-0.5 bg, raw linear   GL: " << glBg << "   Wgpu: " << wgBg << "   (expect 127.5)");
    }

    CHECK(std::abs(def[0] - def[2]) < 25.0);
    CHECK(std::abs(def[1] - def[3]) < 25.0);

    // ── Analytic sun referee (duplicated as a gating test below) ───────────
    {
        constexpr int W = 512, H = 256;
        const Vector3 sunDir = Vector3(0.f, std::sin(math::PI / 4), std::cos(math::PI / 4)).normalize();
        const float cosSun = std::cos(4.f * math::DEG2RAD);
        std::vector<float> data(static_cast<size_t>(W) * H * 4, 0.f);
        for (int y = 0; y < H; ++y) {
            const float elev = ((y + 0.5f) / H - 0.5f) * math::PI;
            for (int x = 0; x < W; ++x) {
                const float az = ((x + 0.5f) / W - 0.5f) * 2.f * math::PI;
                const Vector3 dir(std::cos(elev) * std::cos(az), std::sin(elev), std::cos(elev) * std::sin(az));
                float L = elev > 0.f ? 0.25f : 0.f;
                if (dir.dot(sunDir) > cosSun) L = 40.f;
                const size_t i = (static_cast<size_t>(y) * W + x) * 4;
                data[i + 0] = data[i + 1] = data[i + 2] = L;
                data[i + 3] = 1.f;
            }
        }
        Image img{std::move(data), W, H, 0};
        auto sunEnv = Texture::create(img);
        sunEnv->format = Format::RGBA;
        sunEnv->type = Type::Float;
        sunEnv->colorSpace = ColorSpace::Linear;
        sunEnv->mapping = Mapping::EquirectangularReflection;
        sunEnv->needsUpdate();

        auto plateScene = Scene::create();
        plateScene->background = Color(0, 0, 0);
        plateScene->environment = sunEnv;
        auto plate = Mesh::create(PlaneGeometry::create(4, 4),
                                  MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(1, 1, 1)).roughness(1.f).metalness(0.f)));
        plate->rotation.x = -math::PI / 2.f;// face +Y
        plateScene->add(plate);
        auto plateCam = PerspectiveCamera::create(60, 1.f, 0.1f, 100.f);
        plateCam->position.set(0, 2.5f, -1.2f);
        plateCam->lookAt(Vector3{0, 0, 0});

        double glPlate, wgPlate;
        {
            GLRenderer renderer(glCanvas());
            renderer.outputColorSpace = ColorSpace::NoColorSpace;
            renderer.toneMapping = ToneMapping::None;
            renderer.setClearColor(Color(0, 0, 0));
            renderer.render(*plateScene, *plateCam);
            auto px = renderer.readRGBPixels();
            glPlate = centerPixel(px, RT_WIDTH, RT_HEIGHT).r;
        }
        {
            WgpuRenderer renderer(wgpuCanvas());
            renderer.outputColorSpace = ColorSpace::NoColorSpace;
            renderer.toneMapping = ToneMapping::None;
            renderer.setClearColor(Color(0, 0, 0));
            auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
            renderer.setRenderTarget(target.get());
            renderer.render(*plateScene, *plateCam);
            auto px = renderer.readRGBPixels();
            renderer.setRenderTarget(nullptr);
            renderer.dispose();
            wgPlate = centerPixel(px, RT_WIDTH, RT_HEIGHT).r;
        }
        WARN("sun-plate (analytic 99)   GL: " << glPlate << "   Wgpu: " << wgPlate);
    }

    // ── E(n) hue probe under the user-reported env ──────────────────────────
    // White roughness-1 plates: one facing up, one tilted 45° toward the
    // camera. Centre-pixel RGB ≈ E(n)/π per backend — direct comparison of
    // the diffuse irradiance the two PMREMs deliver, including its hue.
    {
        auto bridge = loader.load(std::string(DATA_FOLDER) + "/textures/env/san_giuseppe_bridge/san_giuseppe_bridge_4k.hdr");
        auto envTex = bridge ? bridge : hdr;
        WARN("E(n) env: " << (bridge ? "san_giuseppe_bridge_4k" : "autumn fallback"));

        auto probePlate = [&](float tiltX, double glRgb[3], double wgRgb[3]) {
            auto makePlateScene = [&] {
                auto scene = Scene::create();
                scene->background = Color(0, 0, 0);
                scene->environment = envTex;
                auto plate = Mesh::create(PlaneGeometry::create(4, 4),
                                          MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0.5f, 0.5f, 0.5f)).roughness(1.f).metalness(0.f)));
                plate->rotation.x = tiltX;
                scene->add(plate);
                return scene;
            };
            auto plateCam = PerspectiveCamera::create(60, 1.f, 0.1f, 100.f);
            plateCam->position.set(0, 2.5f, -1.2f);
            plateCam->lookAt(Vector3{0, 0, 0});
            {
                auto scene = makePlateScene();
                GLRenderer renderer(glCanvas());
                renderer.outputColorSpace = ColorSpace::NoColorSpace;
                renderer.toneMapping = ToneMapping::None;
                renderer.setClearColor(Color(0, 0, 0));
                renderer.render(*scene, *plateCam);
                auto px = renderer.readRGBPixels();
                auto c = centerPixel(px, RT_WIDTH, RT_HEIGHT);
                glRgb[0] = c.r; glRgb[1] = c.g; glRgb[2] = c.b;
            }
            {
                auto scene = makePlateScene();
                WgpuRenderer renderer(wgpuCanvas());
                renderer.outputColorSpace = ColorSpace::NoColorSpace;
                renderer.toneMapping = ToneMapping::None;
                renderer.setClearColor(Color(0, 0, 0));
                auto target = RenderTarget::create(RT_WIDTH, RT_HEIGHT, RenderTarget::Options{});
                renderer.setRenderTarget(target.get());
                renderer.render(*scene, *plateCam);
                auto px = renderer.readRGBPixels();
                renderer.setRenderTarget(nullptr);
                renderer.dispose();
                auto c = centerPixel(px, RT_WIDTH, RT_HEIGHT);
                wgRgb[0] = c.r; wgRgb[1] = c.g; wgRgb[2] = c.b;
            }
        };

        double glUp[3], wgUp[3], glTilt[3], wgTilt[3];
        probePlate(-math::PI / 2.f, glUp, wgUp);          // facing up
        probePlate(-3.f * math::PI / 4.f, glTilt, wgTilt);// 45° toward camera
        WARN("E(up)/pi   GL: " << glUp[0] << "," << glUp[1] << "," << glUp[2]
             << "   Wgpu: " << wgUp[0] << "," << wgUp[1] << "," << wgUp[2]
             << "\nE(45)/pi   GL: " << glTilt[0] << "," << glTilt[1] << "," << glTilt[2]
             << "   Wgpu: " << wgTilt[0] << "," << wgTilt[1] << "," << wgTilt[2]);
    }
}
