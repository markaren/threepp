// White furnace tests — energy conservation of the GL raster IBL path.
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
// Each case pins an ANALYTIC target, not a cross-backend reference, so they
// stand alone as a regression net on PMREM prefiltering and the diffuse-IBL
// π bookkeeping.

#include "gl_test_helpers.hpp"

#include "threepp/materials/MeshPhysicalMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/textures/Texture.hpp"

namespace {

    // 8x4 equirect, Le = (1,1,1,1) per texel.
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
    // 64x32 so the prefilter resolves the horizon without the whole env
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

    // glReadPixels returns scanlines BOTTOM-UP (writeFramebuffer flips before
    // encoding for exactly this reason). Probe rows are screen-space
    // (from-top), so convert here.
    double probeRow(const std::vector<unsigned char>& px, int rowFromTop) {
        const int row = RT_HEIGHT - 1 - rowFromTop;
        const int i = (row * RT_WIDTH + RT_WIDTH / 2) * 3;
        return static_cast<double>(px[i]);
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

    // Raster IBL has small numerical drift from PMREM sampling — 5 LSB is
    // comfortable headroom around the analytic 255.
    CHECK(std::abs(c.r - 255.0) < 5.0);
    CHECK(std::abs(c.g - 255.0) < 5.0);
    CHECK(std::abs(c.b - 255.0) < 5.0);
}

// Directional complement to the constant furnace: catches a diffuse-IBL path
// that lost its directionality (e.g. sampling a 1×1 top mip — a single
// convolved direction sprayed onto every normal — reads fine in a constant env
// but flattens E(n) to a constant here). Probes three rows of the sphere
// (camera framing makes the sphere cover the full 64² frame): top ≈ (1+cosγ)/2
// near 1, centre ≈ 0.5, bottom ≈ 0.
TEST_CASE("Furnace: GL top-lit env diffuse irradiance is directional", "[furnace]") {
    auto scene = Scene::create();
    scene->background = Color(0, 0, 0);
    scene->environment = makeTopLitEnv();
    auto mat = MeshStandardMaterial::create();
    mat->color = Color(1, 1, 1);
    mat->roughness = 1.f;
    mat->metalness = 0.f;
    scene->add(Mesh::create(SphereGeometry::create(1.f, 32, 32), mat));

    auto camera = makeFurnaceCamera();

    const int topRow = RT_HEIGHT / 8;            // normal ≈ up
    const int midRow = RT_HEIGHT / 2;            // normal ≈ toward camera
    const int botRow = RT_HEIGHT - RT_HEIGHT / 8;// normal ≈ down

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.toneMapping = ToneMapping::None;
    renderer.setClearColor(Color(0, 0, 0));
    renderer.render(*scene, *camera);
    auto px = renderer.readRGBPixels();
    REQUIRE(px.size() == DATA_SIZE);

    const double top = probeRow(px, topRow);
    const double mid = probeRow(px, midRow);
    const double bot = probeRow(px, botRow);
    INFO("GL top/mid/bot: " << top << ", " << mid << ", " << bot);

    // The absolute windows are looser than the analytic (1+cosγ)/2 because the
    // prefilter approximates the cosine lobe with GGX(roughness=1), which is
    // wider and compresses the gradient toward the mean. The gradient itself
    // is the real assertion.
    CHECK(top > 170.0);
    CHECK(std::abs(mid - 128.0) < 35.0);
    CHECK(bot < 85.0);
    CHECK(top - bot > 90.0);
}

// Sun-disc furnace: synthetic env with a CLOSED-FORM up-facing irradiance —
// uniform upper hemisphere L0 = 0.25 plus a 4°-radius sun disc (L = 40, below
// the prefilter firefly clamp of 50) at 45° elevation.
//   E(up)/π = L0 + L_s·Ωs·cos(45°)/π,  Ωs = 2π(1−cos 4°)
//           = 0.25 + 40·0.015308·0.7071/π = 0.3878  → byte ≈ 99.
// Guards the failure modes a constant or smooth env cannot see: a prefilter
// that under-samples a small bright source reads ≈ 64 (sun missed); one that
// amplifies it reads high (e.g. f16-Inf texels going sticky through box mips).
// Probed on an up-facing white Lambertian plate, centre pixel.
TEST_CASE("Furnace: GL analytic sun-disc irradiance matches closed form", "[furnace]") {
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

    auto scene = Scene::create();
    scene->background = Color(0, 0, 0);
    scene->environment = sunEnv;
    auto plate = Mesh::create(PlaneGeometry::create(4, 4),
                              MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(1, 1, 1)).roughness(1.f).metalness(0.f)));
    plate->rotation.x = -math::PI / 2.f;// face +Y
    scene->add(plate);

    auto camera = PerspectiveCamera::create(60, 1.f, 0.1f, 100.f);
    camera->position.set(0, 2.5f, -1.2f);
    camera->lookAt(Vector3{0, 0, 0});

    GLRenderer renderer(glCanvas());
    renderer.outputColorSpace = ColorSpace::NoColorSpace;
    renderer.toneMapping = ToneMapping::None;
    renderer.setClearColor(Color(0, 0, 0));
    renderer.render(*scene, *camera);
    auto px = renderer.readRGBPixels();
    REQUIRE(px.size() == DATA_SIZE);

    const double c = centerPixel(px, RT_WIDTH, RT_HEIGHT).r;
    INFO("sun-plate analytic 99, GL: " << c);
    CHECK(std::abs(c - 99.0) < 10.0);
}

// The METAL furnace — the case none of the above covers, and the one that shows
// whether the indirect specular lobe conserves energy.
//
// Every case above uses metalness 0, where the diffuse lobe carries almost all
// the energy and hides what the specular lobe is doing. Worse, at Le = 1 an
// energy-conserving answer lands exactly on the byte ceiling, so 255 cannot be
// told apart from an energy GAIN. This case fixes both: metalness 1 (no diffuse
// lobe at all, so the reading IS the specular lobe) at Le = 0.5 (so the correct
// answer, 127.5, has headroom on both sides).
//
// A white metal in a white furnace must return the furnace, at every roughness —
// that is the definition of energy conservation, independent of the BRDF. With
// the single-scatter split-sum alone this swept 0.96 of the furnace at roughness
// 0 down to 0.45 at roughness 1; <bsdfs> computeMultiscattering adds the energy
// of the later bounces back.
//
// The flatness check is the load-bearing one: a term that merely rescales the
// specular lobe could hit 127 at one roughness, but only a correct compensation
// holds it across the whole sweep.
namespace {

    std::shared_ptr<Texture> makeConstantEnvAt(float le) {
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

    double furnaceRead(float roughness, float metalness, float le) {
        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);
        scene->environment = makeConstantEnvAt(le);

        auto mat = MeshStandardMaterial::create();
        mat->color = Color(1, 1, 1);
        mat->roughness = roughness;
        mat->metalness = metalness;
        scene->add(Mesh::create(SphereGeometry::create(1.f, 64, 48), mat));

        auto camera = PerspectiveCamera::create(45, 1.f, 0.1f, 100.f);
        camera->position.set(0, 0, 2.4f);
        camera->lookAt(Vector3{0, 0, 0});

        GLRenderer renderer(glCanvas());
        renderer.outputColorSpace = ColorSpace::NoColorSpace;
        renderer.toneMapping = ToneMapping::None;
        renderer.toneMappingExposure = 1.f;
        renderer.setClearColor(Color(0, 0, 0));
        renderer.render(*scene, *camera);
        return centerPixel(renderer.readRGBPixels(), RT_WIDTH, RT_HEIGHT).r;
    }

}// namespace

TEST_CASE("Furnace: GL white METAL returns the furnace at every roughness", "[furnace]") {

    constexpr float kLe = 0.5f;
    const double target = kLe * 255.0;// 127.5

    double lo = 1e9, hi = -1e9;
    for (int i = 0; i <= 8; ++i) {
        const float r = static_cast<float>(i) / 8.f;
        const double v = furnaceRead(r, 1.f, kLe);
        INFO("roughness " << r << " -> " << v << " (furnace " << target << ")");
        CHECK(std::abs(v - target) < 4.0);
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }

    INFO("across the roughness sweep: min " << lo << ", max " << hi);
    CHECK(hi - lo < 4.0);
}

// The dielectric complement, measured rather than clipped. A white albedo-1
// dielectric must also return exactly the furnace: the diffuse lobe has to be
// reduced by whatever the specular lobe took, or the surface emits more light
// than falls on it. At Le = 1 this reads 255 whether the answer is 1.0 or 1.3.
TEST_CASE("Furnace: GL white dielectric does not exceed the furnace", "[furnace]") {

    constexpr float kLe = 0.5f;
    const double target = kLe * 255.0;

    for (float r : {0.f, 0.5f, 1.f}) {
        const double v = furnaceRead(r, 0.f, kLe);
        INFO("roughness " << r << " -> " << v << " (furnace " << target << ")");
        CHECK(std::abs(v - target) < 4.0);
    }
}

// The two LAYERED lobes, clearcoat and sheen, in the same furnace.
namespace {

    // Fraction of the furnace returned at pixel (x, y); the centre of the sphere
    // by default. The sphere overfills the 64 px frame (projected radius 35 px), so
    // (9, 9) on the diagonal sits at 0.90 of the radius: the RIM, which is where a
    // Fresnel- or albedo-weighted layer differs most from its reading at the centre.
    double layerRead(const std::shared_ptr<MeshPhysicalMaterial>& mat, float le,
                     int x = RT_WIDTH / 2, int y = RT_HEIGHT / 2) {
        auto scene = Scene::create();
        scene->background = Color(0, 0, 0);
        scene->environment = makeConstantEnvAt(le);
        scene->add(Mesh::create(SphereGeometry::create(1.f, 64, 48), mat));

        auto camera = makeFurnaceCamera();

        GLRenderer renderer(glCanvas());
        renderer.outputColorSpace = ColorSpace::NoColorSpace;
        renderer.toneMapping = ToneMapping::None;
        renderer.setClearColor(Color(0, 0, 0));
        renderer.render(*scene, *camera);
        const auto px = renderer.readRGBPixels();
        return px[(static_cast<size_t>(y) * RT_WIDTH + x) * 3] / (le * 255.0);
    }

}// namespace

// A coat can only reflect light or pass it on to the base, so a white albedo-1
// dielectric UNDER a clearcoat still returns at most the furnace. r129's model
// dimmed the base by a per-light reflectance guess smaller than what the coat
// reflected, and did not dim the indirect diffuse at all: 1.043 of the furnace at
// coat roughness 0.05. r155 and r185 attenuate everything beneath the coat by its
// Fresnel term, once, in <meshphysical_frag>.
TEST_CASE("Furnace: GL clearcoat does not add light", "[furnace]") {

    for (float baseRoughness : {0.2f, 1.f}) {
        for (float coatRoughness : {0.05f, 0.5f, 1.f}) {
            auto mat = MeshPhysicalMaterial::create();
            mat->color = Color(1, 1, 1);
            mat->metalness = 0.f;
            mat->roughness = baseRoughness;
            mat->clearcoat = 1.f;
            mat->clearcoatRoughness = coatRoughness;

            const double v = layerRead(mat, 0.5f);
            INFO("base roughness " << baseRoughness << ", coat roughness " << coatRoughness << " -> " << v << " of the furnace");
            CHECK(v < 1.012);// one byte of headroom at Le = 0.5
            CHECK(v > 0.93); // and the coat must not eat the base either
        }
    }
}

// Black base, white sheen, so what is left after subtracting the same material
// without sheen IS the environment sheen lobe, and in a uniform furnace that is the
// sheen's directional albedo. Integrating BRDF_Specular_Sheen over the hemisphere
// gives 0.141 at sheenRoughness 1 and 0.056 at 0.6. The r136 fit paired with an
// irradiance already divided by PI returned 0.027 and 0.013; r155's pairing of the
// same fit would give 0.084 and 0.042. Le = 3 because the lobe is a few percent of
// the furnace, which at Le = 0.5 is one or two bytes.
TEST_CASE("Furnace: GL environment sheen returns the sheen albedo", "[furnace]") {

    const auto sheenLobe = [](float sheenRoughness) {
        auto bare = MeshPhysicalMaterial::create();
        bare->color = Color(0, 0, 0);
        bare->metalness = 0.f;
        bare->roughness = 1.f;

        auto sheen = MeshPhysicalMaterial::create();
        sheen->color = Color(0, 0, 0);
        sheen->metalness = 0.f;
        sheen->roughness = 1.f;
        sheen->sheenColor = Color(1, 1, 1);
        sheen->sheenRoughness = sheenRoughness;

        return layerRead(sheen, 3.f) - layerRead(bare, 3.f);
    };

    const double at1 = sheenLobe(1.f);
    INFO("sheenRoughness 1.0 -> " << at1 << " (integrated albedo 0.141)");
    CHECK(at1 > 0.11);
    CHECK(at1 < 0.19);

    const double at06 = sheenLobe(0.6f);
    INFO("sheenRoughness 0.6 -> " << at06 << " (integrated albedo 0.056)");
    CHECK(at06 > 0.035);
    CHECK(at06 < 0.075);
}

// The sheen lies ON the base, so the base can only receive what the sheen did not
// reflect. Adding the lobe without taking that out of the base made a WHITE base
// with a white sheen return more than the furnace: 1.16 at the centre and 1.34 at
// the rim at sheen roughness 1, where the sheen's albedo is largest. r185 scales
// every base lobe by 1 - max3( sheenColor ) * albedo; in a furnace that is exact
// for a white sheen, base * ( 1 - albedo ) + albedo = 1 wherever base = 1.
//
// The black-base case above cannot see this, because there the base has nothing
// to give back. Read at the centre AND the rim: the centre alone passed a
// clearcoat that was still wrong at grazing angles once before.
TEST_CASE("Furnace: GL sheen does not add light to a white base", "[furnace]") {

    for (float sheenRoughness : {0.3f, 0.6f, 1.f}) {
        auto mat = MeshPhysicalMaterial::create();
        mat->color = Color(1, 1, 1);
        mat->metalness = 0.f;
        mat->roughness = 1.f;
        mat->sheenColor = Color(1, 1, 1);
        mat->sheenRoughness = sheenRoughness;

        const double centre = layerRead(mat, 0.5f);
        const double rim = layerRead(mat, 0.5f, 9, 9);
        INFO("sheenRoughness " << sheenRoughness << " -> centre " << centre << ", rim " << rim << " of the furnace");
        CHECK(centre < 1.012);// one byte of headroom at Le = 0.5
        CHECK(centre > 0.93);
        CHECK(rim < 1.012);
        CHECK(rim > 0.93);
    }
}
