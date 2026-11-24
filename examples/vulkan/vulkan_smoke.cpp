// Vulkan deferred — a SOLVED smoke plume that a sphere stirs (ParticleField::GridSmoke).
//
// The smoke is not particles. It is a stable-fluids solve on the device — velocity,
// density and temperature on a 128^3 grid over a 5 m box, advected with a MacCormack
// scheme, kept lively by vorticity confinement, made divergence-free by a Jacobi
// pressure projection — and the solver's last pass writes the field's density
// volume. Everything downstream is the code a particle dust cloud already uses: the
// froxel passes, the shade's march (sun, sky, the clustered lights), the LIDAR's
// majorant. The CPU's whole part is one number a frame:
//
//     smoke->gridSmoke().time = t;
//
// plus the capsule the sphere occupies, which is how a body stirs the air: a collider
// carries the smoke with it and leaves a wake.
//
// Controls:  drag = orbit   scroll = zoom   SPACE = the sphere stops / goes again
// Headless:  vulkan_smoke --shot out.png [--t SECONDS] [--res N] [--cam x,y,z --look x,y,z]
//            vulkan_smoke --film DIR [--seconds S] [--fps F]    consecutive PNG frames
//            vulkan_smoke --bench                              the solver's GPU cost

#include "capture_util.hpp"

#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/ParticleField.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>

using namespace threepp;

namespace {

    // The sphere's track: a slow figure-eight through the plume at about head height, as a closed form of t so
    // its velocity is exact (the solver needs it: a collider's velocity is what the air takes on).
    struct Track {
        Vector3 position;
        Vector3 velocity;
    };
    Track sphereAt(double t) {
        const double w = 0.55, A = 1.5, B = 0.55;
        Track        k;
        k.position.set(float(A * std::sin(w * t)), float(1.55 + 0.35 * std::sin(2.0 * w * t)), float(B * std::sin(2.0 * w * t)));
        k.velocity.set(float(A * w * std::cos(w * t)), float(0.35 * 2.0 * w * std::cos(2.0 * w * t)), float(B * 2.0 * w * std::cos(2.0 * w * t)));
        return k;
    }

}// namespace

int main(int argc, char** argv) {

    std::string shotPath, filmDir;
    float       shotT   = 9.f;
    float       seconds = 12.f;
    int         fps     = 60;
    int         res     = 128;
    bool        bench   = false;
    int         iters   = 28;
    std::string upscaler = "dlss";
    bool        noBall  = false;// the A leg of the stir test: no collider, no sphere
    float       sigma = 3.2f, vort = 7.f, buoy = 4.5f, srcRate = 7.f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--film" && i + 1 < argc) filmDir = argv[++i];
        else if (a == "--t" && i + 1 < argc) shotT = float(std::atof(argv[++i]));
        else if (a == "--seconds" && i + 1 < argc) seconds = float(std::atof(argv[++i]));
        else if (a == "--fps" && i + 1 < argc) fps = std::atoi(argv[++i]);
        else if (a == "--res" && i + 1 < argc) res = std::atoi(argv[++i]);
        else if (a == "--bench") bench = true;
        else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--upscaler" && i + 1 < argc) upscaler = argv[++i];
        else if (a == "--no-ball") noBall = true;
        else if (a == "--sigma" && i + 1 < argc) sigma = float(std::atof(argv[++i]));
        else if (a == "--vort" && i + 1 < argc) vort = float(std::atof(argv[++i]));
        else if (a == "--buoy" && i + 1 < argc) buoy = float(std::atof(argv[++i]));
        else if (a == "--rate" && i + 1 < argc) srcRate = float(std::atof(argv[++i]));
    }
    const capture::Args cap = capture::parseArgs(argc, argv);
    const bool          headless = !shotPath.empty() || !filmDir.empty() || bench;

    Canvas canvas("Vulkan Deferred - Smoke",
                  {{"size", std::pair<int, int>{1600, 900}}, {"vsync", false}, {"headless", headless}});
    VulkanRenderer renderer(canvas);
    renderer.setDlss(upscaler == "dlss");
    renderer.setFsr(upscaler == "fsr");

    Scene scene;

    // ── a day: sun, sky, a little haze ──────────────────────────────────────
    const Color sky(0.58f, 0.72f, 0.92f);
    renderer.setClearColor(sky);
    scene.background = sky;
    renderer.toneMapping         = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;
    renderer.setAutoExposure(false);
    renderer.setDeferredStarfield(0.f);
    VulkanRenderer::HeightFogSettings hf;
    hf.baseY       = 0.f;
    hf.density     = 0.004f;
    hf.falloff     = 3.f;
    hf.noiseAmount = 0.3f;
    renderer.setHeightFog(hf);

    auto sun = DirectionalLight::create();
    sun->color.setRGB(1.00f, 0.96f, 0.88f);
    sun->intensity = 2.6f;
    sun->position.set(4.f, 6.5f, 3.f);
    scene.add(sun);
    auto ambient = AmbientLight::create();
    ambient->color.setRGB(0.55f, 0.66f, 0.86f);
    ambient->intensity = 1.1f;
    scene.add(ambient);

    auto ground = Mesh::create(PlaneGeometry::create(120.f, 120.f, 1, 1),
                               MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0.30f, 0.27f, 0.22f)).roughness(0.95f).metalness(0.f)));
    ground->rotateX(-math::PI / 2.f);
    scene.add(ground);

    // ── the smoke ───────────────────────────────────────────────────────────
    // A 5 m box standing on the ground, the bottom face a wall.
    const Vector3 boxCenter(0.f, 2.5f, 0.f), boxHalf(2.5f, 2.5f, 2.5f);
    auto          smoke = ParticleField::createGridSmoke(boxCenter, boxHalf, static_cast<std::uint32_t>(res));
    auto&         gs    = smoke->gridSmoke();
    gs.solidFloor       = true;
    gs.buoyancy         = buoy;
    gs.dissipation      = 0.10f;
    gs.cooling          = 0.55f;
    gs.vorticity        = vort;
    gs.damping          = 0.02f;
    gs.sigma            = sigma;
    gs.pressureIterations = std::uint32_t(iters);
    gs.wind.set(0.30f, 0.f, 0.08f);
    {
        ParticleField::GridSmoke::Source s;
        s.position.set(0.f, 0.22f, 0.f);
        s.radius        = 0.30f;
        s.density       = srcRate;
        s.temperature   = 4.f;
        s.velocity.set(0.f, 1.9f, 0.f);
        s.velocityBlend = 5.f;
        s.noise         = 0.7f;
        gs.sources.push_back(s);
    }
    gs.colliders.resize(noBall ? 0 : 1);
    auto& dr             = smoke->densityRepr();
    dr.albedo            = Color(0.62f, 0.62f, 0.64f);
    dr.anisotropy        = 0.35f;
    scene.add(smoke);

    // The sphere the collider stands for.
    constexpr float kBall = 0.42f;
    auto ball = Mesh::create(SphereGeometry::create(kBall, 64, 40),
                             MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0.80f, 0.12f, 0.08f)).roughness(0.28f).metalness(0.1f)));
    if (!noBall) scene.add(ball);
    bool ballMoving = true;
    double ballT    = 0.0;

    // ONE place the clock is read, so every loop below (live, --shot, --film, --bench) steps the smoke and the
    // sphere together.
    const auto tick = [&](double t) {
        const Track k = sphereAt(t);
        ball->position.copy(k.position);
        if (!gs.colliders.empty()) {
            auto& c    = gs.colliders[0];
            c.a        = k.position;
            c.b        = k.position;
            c.radius   = kBall;
            c.velocity = k.velocity;
        }
        gs.time = t;
    };

    PerspectiveCamera camera(42.f, canvas.aspect(), 0.05f, 400.f);
    camera.position.set(5.6f, 1.9f, 6.4f);
    if (cap.camPos) camera.position.copy(*cap.camPos);
    const Vector3 target = cap.camTarget.value_or(Vector3(0.f, 2.0f, 0.f));
    camera.lookAt(target);
    canvas.onWindowResize([&](WindowSize s) {
        camera.aspect = s.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(s);
    });

    constexpr double kDt = 1.0 / 60.0;

    if (bench) {
        for (int i = 0; i < 240; ++i) {
            tick(double(i) * kDt);
            canvas.animateOnce([&] { renderer.render(scene, camera); });
        }
        double ms = 0, gpu = 0;
        for (int i = 240; i < 400; ++i) {
            tick(double(i) * kDt);
            canvas.animateOnce([&] { renderer.render(scene, camera); });
            const auto t = renderer.lastFrameTimings();
            ms += t.particleDensityMs;
            gpu += t.gpuTotalMs;
        }
        std::printf("[bench] %d^3, %u pressure sweeps: density block %.3f ms, whole frame %.3f ms (per frame, 1 solver step each)\n",
                    res, gs.pressureIterations, ms / 160.0, gpu / 160.0);
        return 0;
    }

    if (!filmDir.empty()) {
        namespace fs = std::filesystem;
        fs::create_directories(filmDir);
        const double step = 1.0 / double(fps);
        const int    n    = int(seconds * float(fps));
        // The solver runs fixed 1/60 s steps; a film at another rate asks for whole steps per frame.
        gs.maxStepsPerFrame = std::max<std::uint32_t>(8u, std::uint32_t(std::ceil(step / kDt)) + 1u);
        for (int i = 0; i < n; ++i) {
            tick(double(i) * step);
            canvas.animateOnce([&] { renderer.render(scene, camera); });
            char name[64];
            std::snprintf(name, sizeof(name), "f%05d.png", i);
            renderer.writeFramebuffer((fs::path(filmDir) / name).string());
        }
        std::printf("[film] %d frames -> %s\n", n, filmDir.c_str());
        return 0;
    }

    if (!shotPath.empty()) {
        // Warm the solve up at 60 steps a second to shotT, then hold the clock so the density stops moving and TAA converges.
        const int warm   = int(shotT / kDt);
        const int settle = cap.frames.value_or(48);
        for (int i = 0; i < warm + settle; ++i) {
            tick(std::min(double(i), double(warm)) * kDt);
            canvas.animateOnce([&] { renderer.render(scene, camera); });
        }
        renderer.writeFramebuffer(shotPath);
        std::printf("[shot] %s (smoke at t=%.1f s)\n", shotPath.c_str(), double(shotT));
        return 0;
    }

    OrbitControls controls(camera, canvas);
    controls.target.copy(target);
    controls.update();
    KeyAdapter keys(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent evt) {
        if (evt.key == Key::SPACE) ballMoving = !ballMoving;
    });
    canvas.addKeyListener(keys);

    Clock clock;
    canvas.animate([&] {
        const double dt = clock.getDelta();
        if (ballMoving) ballT += dt;
        // The solver keeps its own clock: the sphere holding still still lets the smoke run.
        static double simT = 0.0;
        simT += dt;
        tick(ballT);
        if (!ballMoving && !gs.colliders.empty()) gs.colliders[0].velocity.set(0.f, 0.f, 0.f);
        gs.time = simT;
        renderer.render(scene, camera);
    });
    return 0;
}
