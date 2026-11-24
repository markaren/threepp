// A pelican riding a bicycle, built entirely in code: a steel roadster with spoked wheels, a running chain, a leather
// saddle, mud guards, lamp and bell, and a white pelican on it, pedalling, with its wings on the bars. The bird's
// body is lofted from sections, its plumage is thousands of individual feathers, its bill and pouch are modelled.
// The path runs along a sandy coast. Every texture is painted per pixel here; nothing is loaded.
//
//   orbit: left mouse, pan: right mouse, wheel: closer / further.   Space or P: pause.   1 .. 9: the views
//   The panel: the cadence (and the speed it makes), ambient occlusion.
//
// Headless:
//   PelicanBike --shot out.png [--view side|hero|face|wing|legs|rear|chain|...] [--eye x,y,z] [--look x,y,z]
//               [--fov deg] [--size WxH] [--phase deg] [--time T] [--no-ao] [--vulkan] [--pathtrace SPP]
//               [--exposure E] [--ride] [--bench N]
//   PelicanBike --film dir [--seconds 12] [--fps 60] [--sub 4] [--size WxH]   frames of a camera ride, each averaged over
//               --sub moments of a half-frame shutter (encode with ffmpeg: tmix over the sub-frames)

#include "threepp/threepp.hpp"

#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/terrain/MountainBackdrop.hpp"
#include "threepp/postprocessing/EffectComposer.hpp"
#include "threepp/postprocessing/GTAOPass.hpp"
#include "threepp/postprocessing/OutputPass.hpp"
#include "threepp/postprocessing/RenderPass.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include "Bicycle.hpp"
#include "Pelican.hpp"
#include "Scenery.hpp"
#include "SkyBake.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <string>

using namespace threepp;

namespace {

    struct View {
        Vector3 eye, target;
        float fov;
    };

    const std::map<std::string, View> views{
            {"side", {{0.5f, 0.85f, 3.6f}, {0.5f, 0.75f, 0.f}, 34.f}},
            {"hero", {{1.9f, 0.95f, 2.7f}, {0.4f, 0.85f, 0.f}, 38.f}},
            {"front", {{3.2f, 1.2f, 1.0f}, {0.5f, 0.85f, 0.f}, 34.f}},
            {"head", {{1.6f, 1.55f, 1.4f}, {0.55f, 1.35f, 0.f}, 24.f}},
            {"face", {{1.30f, 1.45f, 0.95f}, {0.60f, 1.33f, 0.03f}, 20.f}},
            {"basket", {{1.65f, 1.30f, 1.15f}, {0.85f, 1.0f, 0.0f}, 30.f}},
            {"front34", {{3.0f, 0.95f, 2.0f}, {0.55f, 0.95f, 0.0f}, 36.f}},
            {"low", {{-0.7f, 0.42f, 2.4f}, {0.55f, 0.95f, 0.0f}, 38.f}},
            {"hips", {{0.25f, 0.55f, 1.5f}, {0.22f, 0.72f, 0.05f}, 30.f}},
            {"hipsfront", {{1.6f, 0.45f, 0.9f}, {0.25f, 0.72f, 0.0f}, 34.f}},
            {"basketfish", {{0.95f, 1.45f, 1.2f}, {0.74f, 1.02f, 0.0f}, 28.f}},
            {"wing", {{0.2f, 0.95f, 2.1f}, {0.15f, 0.85f, 0.f}, 30.f}},
            {"wingclose", {{0.05f, 0.95f, 1.05f}, {0.05f, 0.88f, 0.1f}, 28.f}},
            {"legs", {{1.25f, 0.5f, 1.7f}, {0.38f, 0.45f, 0.1f}, 30.f}},
            {"feet", {{0.9f, 0.35f, 1.5f}, {0.43f, 0.3f, 0.f}, 34.f}},
            {"chain", {{0.5f, 0.38f, 1.1f}, {0.25f, 0.3f, 0.05f}, 32.f}},
            {"rear", {{-1.7f, 1.0f, 1.3f}, {0.2f, 0.85f, 0.f}, 36.f}},
            {"three", {{1.9f, 0.95f, 2.7f}, {0.4f, 0.85f, 0.f}, 38.f}},
            {"rearguard", {{-1.0f, 0.55f, 1.2f}, {0.0f, 0.42f, 0.0f}, 38.f}},
            {"frontguard", {{1.9f, 0.55f, 1.3f}, {0.9f, 0.5f, 0.0f}, 38.f}},
            {"seatbridge", {{0.55f, 0.80f, 0.75f}, {0.25f, 0.55f, 0.0f}, 30.f}},
            {"kick", {{0.15f, 0.12f, 0.95f}, {0.27f, 0.28f, 0.0f}, 32.f}},
            {"tail", {{-1.1f, 0.40f, 0.45f}, {-0.30f, 0.30f, 0.0f}, 32.f}},
    };
    const std::array<const char*, 9> viewKeys{"hero", "side", "front", "face", "wing", "legs", "chain", "rear", "head"};

    double secondsSince(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

}// namespace

int main(int argc, char** argv) {

    std::string shotPath, viewName = "hero";
    float shotTime = 1.f;
    int width = 1600, height = 1000;
    Vector3 eye, lookAt;
    bool freeEye = false, freeLook = false, vulkan = false, noAo = false, ride = false;
    float fov = 0, exposureArg = 0, phaseDeg = 20.f, cadence = 52.f;
    int ptSamples = 0, benchFrames = 0, filmSub = 4;
    std::string filmDir;
    float filmSeconds = 12.f, filmFps = 60.f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--view" && i + 1 < argc) viewName = argv[++i];
        else if (a == "--time" && i + 1 < argc) shotTime = std::stof(argv[++i]);
        else if (a == "--size" && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &width, &height);
        else if (a == "--eye" && i + 1 < argc) std::sscanf(argv[++i], "%f,%f,%f", &eye.x, &eye.y, &eye.z), freeEye = true;
        else if (a == "--look" && i + 1 < argc) std::sscanf(argv[++i], "%f,%f,%f", &lookAt.x, &lookAt.y, &lookAt.z), freeLook = true;
        else if (a == "--fov" && i + 1 < argc) fov = std::stof(argv[++i]);
        else if (a == "--phase" && i + 1 < argc) phaseDeg = std::stof(argv[++i]);
        else if (a == "--cadence" && i + 1 < argc) cadence = std::stof(argv[++i]);
        else if (a == "--exposure" && i + 1 < argc) exposureArg = std::stof(argv[++i]);
        else if (a == "--vulkan") vulkan = true;
        else if (a == "--no-ao") noAo = true;
        else if (a == "--film" && i + 1 < argc) filmDir = argv[++i];
        else if (a == "--seconds" && i + 1 < argc) filmSeconds = std::stof(argv[++i]);
        else if (a == "--fps" && i + 1 < argc) filmFps = std::stof(argv[++i]);
        else if (a == "--sub" && i + 1 < argc) filmSub = std::max(1, std::atoi(argv[++i]));
        else if (a == "--ride") ride = true;   // headless: ride for --time seconds first
        else if (a == "--bench" && i + 1 < argc) benchFrames = std::max(1, std::atoi(argv[++i]));
        else if (a == "--pathtrace" && i + 1 < argc) ptSamples = std::max(1, std::atoi(argv[++i])), vulkan = true;
    }
    const bool film = !filmDir.empty();
    const bool headless = !shotPath.empty() || film;
    if (!views.count(viewName)) viewName = "hero";
    const auto t0 = std::chrono::steady_clock::now();

    Canvas canvas(Canvas::Parameters().title("Pelican on a bicycle").size(width, height).antialiasing(4).vsync(!headless).headless(headless));
    std::unique_ptr<Renderer> rendererOwner;
    GLRenderer* gl = nullptr;
    VulkanRenderer* vk = nullptr;
    if (vulkan) {
        auto r = std::make_unique<VulkanRenderer>(canvas);
        vk = r.get();
        rendererOwner = std::move(r);
    } else {
        auto r = std::make_unique<GLRenderer>(canvas);
        r->shadowMap().enabled = true;
        r->shadowMap().type = ShadowMap::PFC;
        gl = r.get();
        rendererOwner = std::move(r);
    }
    Renderer& renderer = *rendererOwner;
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = exposureArg > 0 ? exposureArg : 0.85f;

    // ---- light: a bright late-afternoon sun over a coast path ---------------------------------------------------------
    mill::SkyLook look;
    look.sun = Vector3(-0.36f, 0.40f, 0.84f).normalize();
    look.cloudCover = 0.3f;
    auto sky = mill::bakeSky(look);
    auto scene = Scene::create();
    scene->background = std::shared_ptr<Texture>(sky);
    scene->environment = std::shared_ptr<Texture>(sky);

    auto sun = DirectionalLight::create(Color(1.f, 0.9f, 0.74f), 3.6f);
    sun->castShadow = true;
    sun->shadow->mapSize.set(4096, 4096);
    sun->shadow->bias = -0.0002f;
    sun->shadow->normalBias = 0.01f;
    if (auto* sc = sun->shadow->camera->as<OrthographicCamera>()) {
        sc->left = sc->bottom = -3.f;
        sc->right = sc->top = 3.f;
        sc->nearPlane = 1.f;
        sc->farPlane = 80.f;
        sc->updateProjectionMatrix();
    }
    auto sunTarget = Object3D::create();
    scene->add(sunTarget);
    sun->setTarget(*sunTarget);
    scene->add(sun);
    const auto aimSun = [&](float x) {
        sunTarget->position.set(x + 0.5f, 0.8f, 0.f);
        sun->position.copy(look.sun).multiplyScalar(30.f).add(sunTarget->position);
    };
    aimSun(0.f);

    // ---- the setting ------------------------------------------------------------------------------------------------------
    const auto tScene = std::chrono::steady_clock::now();
    pb::Scenery scenery;
    scene->add(scenery.root);
    // ranges across the water, standing in the sea
    std::shared_ptr<Object3D> mountains;
    {
        terrain::MountainBackdropOptions r;
        r.ground = [](float, float) { return -1.7f; };
        r.sunDirection = look.sun;
        r.hazeColor = Color(0.58f, 0.68f, 0.80f);
        r.innerRadius = 4200.f;
        r.fullRadius = 7500.f;
        r.outerRadius = 15000.f;
        r.height = 650.f;
        r.crestHeight = 180.f;
        r.snowSlope = 0.04f;
        r.hazeDistance = 9000.f;
        mountains = terrain::MountainBackdrop::create(r);
        scene->add(mountains);
    }
    std::cout << "scenery " << secondsSince(tScene) << " s" << std::endl;

    // ---- the machine and its rider ------------------------------------------------------------------------------------------
    const auto tBuild = std::chrono::steady_clock::now();
    pb::Bicycle bike;
    scene->add(bike.root);
    std::cout << "bicycle " << secondsSince(tBuild) << " s" << std::endl;
    const auto tBird = std::chrono::steady_clock::now();
    pb::Pelican bird;
    bike.pose(0.f, 0.f);
    bike.root->updateMatrixWorld(true);
    bird.mount(bike, Vector3(0.2f, 0.835f, 0.f), 0.14f, 1.1f);
    bike.root->add(bird.root);
    std::cout << "pelican " << secondsSince(tBird) << " s" << std::endl;

    auto camera = PerspectiveCamera::create(36.f, static_cast<float>(width) / static_cast<float>(height), 0.05f, 20000.f);
    const View& v = views.at(viewName);
    camera->position.copy(freeEye ? eye : v.eye);
    camera->fov = fov > 0 ? fov : v.fov;
    camera->updateProjectionMatrix();
    camera->lookAt(freeLook ? lookAt : v.target);

    // ---- post: ambient occlusion over the forward render ---------------------------------------------------------------
    std::unique_ptr<EffectComposer> composer;
    std::shared_ptr<GTAOPass> gtao;
    if (gl) {
        EffectComposer::Options options;
        options.samples = 4;
        composer = std::make_unique<EffectComposer>(*gl, options);
        composer->addPass(std::make_shared<RenderPass>(*scene, *camera));
        GTAOParameters aoParameters;
        aoParameters.radius = 0.35f;
        aoParameters.distanceExponent = 1.f;
        aoParameters.thickness = 1.f;
        aoParameters.scale = 1.f;
        aoParameters.samples = 16;
        aoParameters.distanceFallOff = 1.f;
        aoParameters.screenSpaceRadius = false;
        PoissonDenoiseParameters pdParameters;
        pdParameters.lumaPhi = 10.f;
        pdParameters.depthPhi = 2.f;
        pdParameters.normalPhi = 3.f;
        pdParameters.radius = 4.f;
        pdParameters.radiusExponent = 1.f;
        pdParameters.rings = 2.f;
        pdParameters.samples = 16;
        gtao = std::make_shared<GTAOPass>(*scene, *camera, width, height, aoParameters, pdParameters);
        gtao->blendIntensity = 0.8f;
        gtao->enabled = !noAo;
        composer->addPass(gtao);
        composer->addPass(std::make_shared<OutputPass>());
    }

    OrbitControls controls(*camera, canvas);
    controls.target.copy(freeLook ? lookAt : v.target);
    controls.update();

    // ---- the ride ---------------------------------------------------------------------------------------------------------
    float phase = phaseDeg * pb::PI / 180.f;
    float riderX = 0.f;
    bool paused = false;
    const float rolling = 0.325f;// the tyre's rolling radius
    const float gearing = static_cast<float>(pb::Bicycle::ringTeeth) / static_cast<float>(pb::Bicycle::cogTeeth);
    const auto goTo = [&](const char* name) {
        const View& w = views.at(name);
        camera->position.set(w.eye.x + riderX, w.eye.y, w.eye.z);
        camera->fov = w.fov;
        camera->updateProjectionMatrix();
        controls.target.set(w.target.x + riderX, w.target.y, w.target.z);
        camera->lookAt(controls.target);
        controls.update();
    };

    std::unique_ptr<ImguiFunctionalContext> ui;
    IOCapture capture{};
    if (!headless) {
        ui = std::make_unique<ImguiFunctionalContext>(canvas, renderer, [&] {
            ImGui::SetNextWindowPos({10, 10}, 0, {0, 0});
            ImGui::SetNextWindowSize({320, 0}, 0);
            ImGui::Begin("Pelican on a bicycle");
            ImGui::SliderFloat("cadence (rpm)", &cadence, 0.f, 110.f, "%.0f");
            ImGui::Text("%.1f km/h, %.0f m ridden", cadence * 2.f * pb::PI / 60.f * gearing * rolling * 3.6f, riderX);
            ImGui::Checkbox("pause", &paused);
            if (gtao) {
                bool ao = gtao->enabled;
                if (ImGui::Checkbox("ambient occlusion", &ao)) gtao->enabled = ao;
            }
            ImGui::Text("orbit; 1-9: views; space: pause");
            ImGui::End();
        });
        capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
        capture.preventScrollEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
        canvas.setIOCapture(&capture);
    }
    canvas.onKeyPressed([&](KeyEvent e) {
        if (e.key == Key::SPACE || e.key == Key::P) paused = !paused;
        else if (e.key >= Key::NUM_1 && e.key <= Key::NUM_9) goTo(viewKeys[static_cast<size_t>(static_cast<int>(e.key) - static_cast<int>(Key::NUM_1))]);
    });
    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
        if (composer) composer->setSize(size.width(), size.height());
    });

    // the film's camera: a ride-along, round the front and in on the face, then back out low beside the wheels. Each key is
    // an eye and a target relative to the bicycle, and a field of view.
    struct Key {
        Vector3 eye, target;
        float fov;
    };
    const std::vector<Key> filmKeys{
            {{0.9f, 0.80f, 3.4f}, {0.5f, 0.80f, 0.f}, 34.f},
            {{2.2f, 0.95f, 3.0f}, {0.45f, 0.90f, 0.f}, 36.f},
            {{3.1f, 0.70f, 1.3f}, {0.5f, 0.95f, 0.f}, 36.f},
            {{1.9f, 1.30f, 1.5f}, {0.6f, 1.20f, 0.f}, 28.f},
            {{1.1f, 0.55f, 2.2f}, {0.35f, 0.60f, 0.f}, 32.f},
            {{-0.9f, 0.85f, 3.0f}, {0.45f, 0.85f, 0.f}, 36.f}};
    const auto filmCamera = [&](float t) {
        std::vector<Vector3> eyes, targets;
        std::vector<float> fovs;
        for (const auto& k : filmKeys) {
            eyes.push_back(k.eye);
            targets.push_back(k.target);
            fovs.push_back(k.fov);
        }
        CatmullRomCurve3 ce(eyes), ct(targets);
        float u = std::clamp(t / filmSeconds, 0.f, 1.f);
        u = u * u * (3.f - 2.f * u) * 0.35f + u * 0.65f;   // a soft start and finish
        Vector3 e, tg;
        ce.getPoint(u, e);
        ct.getPoint(u, tg);
        const float f = u * static_cast<float>(fovs.size() - 1);
        const size_t i0 = std::min(static_cast<size_t>(f), fovs.size() - 2);
        const float w = f - static_cast<float>(i0);
        const float sm = w * w * (3.f - 2.f * w);
        camera->position.set(e.x + riderX, e.y, e.z);
        camera->fov = fovs[i0] + (fovs[i0 + 1] - fovs[i0]) * sm;
        camera->updateProjectionMatrix();
        camera->lookAt(Vector3(tg.x + riderX, tg.y, tg.z));
    };
    int filmFrame = 0, filmSubIndex = 0;
    const int filmTotal = static_cast<int>(filmSeconds * filmFps);

    std::cout << "ready " << secondsSince(t0) << " s" << std::endl;
    Clock clock;
    float simTime = 0, lastX = 0;
    int benchDone = 0;
    auto benchStart = std::chrono::steady_clock::now();
    canvas.animate([&] {
        const float dt = headless ? 1.f / 60 : std::min(clock.getDelta(), 0.05f);
        if (!paused) {
            simTime += dt;
            if (!headless || ride) {
                const float dPhase = cadence * 2.f * pb::PI / 60.f * dt;
                phase += dPhase;
                riderX += rolling * gearing * dPhase;
            }
        }
        if (film) {
            if (filmFrame >= filmTotal) std::exit(0);
            // this frame's moment: the shutter is open for half the frame, split into filmSub instants
            const float t = (static_cast<float>(filmFrame) + 0.5f * static_cast<float>(filmSubIndex) / static_cast<float>(filmSub)) / filmFps;
            const float omega = cadence * 2.f * pb::PI / 60.f;
            phase = phaseDeg * pb::PI / 180.f + omega * t;
            riderX = rolling * gearing * omega * t;
            filmCamera(t);
        }
        // the machine rolls along, leaning and steering a little with each pedal stroke
        bike.root->position.x = riderX;
        bike.root->rotation.x = -0.012f * std::sin(phase);
        bike.pose(phase, 0.012f * std::sin(phase));
        bird.root->position.y = 0.835f + 0.003f * std::sin(2.f * phase + 0.6f);
        bike.root->updateMatrixWorld(true);
        bird.pose(bike, phase);
        scenery.follow(riderX);
        mountains->position.x = riderX;
        aimSun(riderX);
        if (!headless || ride) {
            // the camera rides along
            const float dx = riderX - lastX;
            camera->position.x += dx;
            controls.target.x += dx;
            lastX = riderX;
            controls.update();
        }
        if (composer) composer->render(dt);
        else renderer.render(*scene, *camera);
        if (ui) ui->render();
        if (film) {
            char name[64];
            std::snprintf(name, sizeof name, "/f_%05d.jpg", filmFrame * filmSub + filmSubIndex);
            renderer.writeFramebuffer(filmDir + name);
            if (++filmSubIndex == filmSub) {
                filmSubIndex = 0;
                if (++filmFrame % 60 == 0) std::cout << "frame " << filmFrame << " / " << filmTotal << std::endl;
            }
            return;
        }
        if (headless && simTime >= shotTime && benchDone < benchFrames) {
            if (benchDone == 0) benchStart = std::chrono::steady_clock::now();
            ++benchDone;
            // keep the ride going so the frames measured are moving ones
            phase += cadence * 2.f * pb::PI / 60.f * dt;
            riderX += rolling * gearing * cadence * 2.f * pb::PI / 60.f * dt;
            return;
        }
        if (headless && simTime >= shotTime) {
            if (benchFrames > 0) std::cout << "bench " << benchFrames << " frames: " << 1000.0 * secondsSince(benchStart) / benchFrames << " ms a frame" << std::endl;
            if (vk && ptSamples > 0) {
                VulkanRenderer::PathTraceParams params;
                params.samplesPerFrame = 16;
                params.maxBounces = 6;
                params.fireflyClamp = 20.f;
                const auto tTrace = std::chrono::steady_clock::now();
                vk->pathTraceToFile(*scene, *camera, shotPath, static_cast<uint32_t>(ptSamples), params);
                std::cout << "path traced " << ptSamples << " spp in " << secondsSince(tTrace) << " s; wrote " << shotPath << std::endl;
                std::exit(0);
            }
            renderer.writeFramebuffer(shotPath);
            std::cout << "wrote " << shotPath << std::endl;
            std::exit(0);
        }
    });
}
