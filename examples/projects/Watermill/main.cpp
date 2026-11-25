// An old watermill on a forest stream, built entirely in code: the valley and its boulders, the stream (simulated),
// the timber mill on its stone foundation with its machinery inside, the overshot wheel turned by the water from
// its flume, the footbridge, and the summer woods round it. Every texture is painted here per pixel; nothing is
// loaded.
//
//   orbit (default)   left mouse: orbit, right mouse: pan, wheel: closer / further
//   Tab               walk instead: W A S D to move, Q / E down / up, Shift faster, left mouse drag to look
//   1 .. 9, 0         the views: hero, overview, side, bridge, stream, wheel, stone floor, basement, bin floor; falls
//   P                 pause the water
//   [ / ]             less / more water in the creek
//   The panel: the wheel's speed and the torque on it, and the flume's gate (how much water it lets through).
//
// Headless (dev):
//   Watermill --shot out.png [--view hero|overview|...] [--eye x,y,z] [--look x,y,z] [--fov deg] [--size WxH]
//             [--time T] [--no-ao] [--bench N]   (N frames timed after T, then the still)
//   Watermill --flow Q ...                       the creek's flow in m3/s (default 2.8; 0.3 a trickle, 8 a spate)
//   Watermill --vulkan ...                       the same on VulkanRenderer (ray-traced light)
//   Watermill --pathtrace SPP --shot out.png ... a path-traced still (Vulkan), SPP samples a pixel

#include "threepp/threepp.hpp"

#include "threepp/extras/terrain/MountainBackdrop.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

#include "threepp/postprocessing/EffectComposer.hpp"
#include "threepp/postprocessing/GTAOPass.hpp"
#include "threepp/postprocessing/OutputPass.hpp"
#include "threepp/postprocessing/RenderPass.hpp"
#include "threepp/renderers/PlanarReflection.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include "Atmosphere.hpp"
#include "Bridge.hpp"
#include "Flume.hpp"
#include "Interior.hpp"
#include "Materials.hpp"
#include "Mill.hpp"
#include "SkyBake.hpp"
#include "Site.hpp"
#include "Spray.hpp"
#include "Terrain.hpp"
#include "Trees.hpp"
#include "Water.hpp"
#include "Wheel.hpp"
#include "WheelWater.hpp"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <map>
#include <string>

using namespace threepp;
using namespace mill;

namespace {

    struct View {
        Vector3 eye, target;
        float fov;
    };

    const std::map<std::string, View> views{
            {"hero", {{-4.3f, 0.95f, 27.5f}, {2.4f, 3.9f, -8.f}, 52.f}},
            {"overview", {{3.f, 11.f, 36.f}, {2.f, 3.5f, -6.f}, 50.f}},
            {"side", {{-14.f, 4.f, 2.f}, {6.f, 4.f, 0.f}, 50.f}},
            {"bridge", {{-6.f, 2.5f, 2.f}, {-3.f, 3.5f, -13.f}, 55.f}},
            {"stream", {{-1.f, 0.9f, 8.f}, {0.f, 0.f, -8.f}, 55.f}},
            {"wheel", {{0.5f, 2.2f, 9.5f}, {5.f, 3.f, 0.5f}, 45.f}},
            {"stone", {{10.7f, 5.25f, 0.4f}, {7.9f, 6.f, 0.4f}, 62.f}},
            {"basement", {{10.9f, 2.3f, -1.0f}, {7.2f, 2.7f, 0.9f}, 62.f}},
            {"bin", {{11.6f, 8.5f, 3.6f}, {7.5f, 7.6f, -2.f}, 60.f}},
            {"falls", {{-1.2f, -2.2f, 50.f}, {3.f, 1.7f, 8.f}, 48.f}},
    };
    const std::array<const char*, 10> viewKeys{"falls", "hero", "overview", "side", "bridge", "stream", "wheel", "stone", "basement", "bin"};

    double secondsSince(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

}// namespace

int main(int argc, char** argv) {

    std::string shotPath, viewName = "hero";
    float shotTime = 2.f;
    int width = 1500, height = 1000;
    bool ao = true;
    Vector3 eye, lookAt;
    bool freeEye = false, freeLook = false;
    float fov = 0;
    int benchFrames = 0;
    Vector3 sunArg;
    float airScale = 1.f;
    bool vulkan = false;
    float flowArg = -1.f;
    int ptSamples = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--view" && i + 1 < argc) viewName = argv[++i];
        else if (a == "--time" && i + 1 < argc) shotTime = std::stof(argv[++i]);
        else if (a == "--size" && i + 1 < argc) std::sscanf(argv[++i], "%dx%d", &width, &height);
        else if (a == "--no-ao") ao = false;
        else if (a == "--eye" && i + 1 < argc) std::sscanf(argv[++i], "%f,%f,%f", &eye.x, &eye.y, &eye.z), freeEye = true;
        else if (a == "--look" && i + 1 < argc) std::sscanf(argv[++i], "%f,%f,%f", &lookAt.x, &lookAt.y, &lookAt.z), freeLook = true;
        else if (a == "--fov" && i + 1 < argc) fov = std::stof(argv[++i]);
        else if (a == "--bench" && i + 1 < argc) benchFrames = std::max(1, std::atoi(argv[++i]));
        else if (a == "--vulkan") vulkan = true;
        else if (a == "--flow" && i + 1 < argc) flowArg = std::stof(argv[++i]);
        else if (a == "--sun" && i + 1 < argc) std::sscanf(argv[++i], "%f,%f,%f", &sunArg.x, &sunArg.y, &sunArg.z);
        else if (a == "--air" && i + 1 < argc) airScale = std::stof(argv[++i]);
        else if (a == "--pathtrace" && i + 1 < argc) ptSamples = std::max(1, std::atoi(argv[++i])), vulkan = true;
    }
    const bool headless = !shotPath.empty();
    if (!views.count(viewName)) viewName = "hero";
    const auto t0 = std::chrono::steady_clock::now();

    Canvas canvas(Canvas::Parameters()
                          .title("Watermill")
                          .size(width, height)
                          .antialiasing(4)
                          .vsync(!headless)
                          .headless(headless));
    std::unique_ptr<Renderer> rendererOwner;
    GLRenderer* gl = nullptr;
    VulkanRenderer* vk = nullptr;
#ifndef THREEPP_WITH_VULKAN
    if (vulkan) std::cerr << "--vulkan: this build has no Vulkan backend, running on GL" << std::endl;
    vulkan = false;
#else
    if (vulkan) {
        auto r = std::make_unique<VulkanRenderer>(canvas);
        vk = r.get();
        rendererOwner = std::move(r);
    } else
#endif
    {
        auto r = std::make_unique<GLRenderer>(canvas);
        r->shadowMap().enabled = true;
        r->shadowMap().type = ShadowMap::PFC;
        gl = r.get();
        rendererOwner = std::move(r);
    }
    Renderer& renderer = *rendererOwner;
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.2f;
    float exposureArg = 0;
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--exposure") exposureArg = std::stof(argv[i + 1]);
    if (vk) renderer.toneMappingExposure = 1.5f;// the traced light is darker under the canopy than GL's
    if (exposureArg > 0) renderer.toneMappingExposure = exposureArg;

    // ---- light: a summer afternoon, the sun high to the left of the view ------------------------------------------
    SkyLook look;
    look.sun = sunArg.lengthSq() > 0 ? sunArg.normalize() : Vector3(-0.42f, 0.85f, 0.3f).normalize();
    auto sky = bakeSky(look);
    auto scene = Scene::create();
    scene->background = std::shared_ptr<Texture>(sky);
    scene->environment = bakeForestEnvironment(look);
    scene->fog = FogExp2(Color(0.62f, 0.70f, 0.74f), 0.0035f);
    // the valley's air: denser down by the water, thinning up the valley sides (GL: Atmosphere.hpp, applied below)
    AirLook airLook;
    airLook.sunDirection = look.sun;
    airLook.sun = Vector3(1.f, 0.94f, 0.82f) * 1.3f;
    airLook.ambient = Vector3(0.075f, 0.1f, 0.1f);// the canopy hides most of the sky from the valley's air
    airLook.density = 0.0065f * airScale;
    airLook.base = 0.f;
    airLook.falloff = 16.f;
    airLook.g = 0.62f;
    if (vk) {
        // Vulkan's air is one medium that thins with height (a uniform one would veil the whole sky); the path tracer
        // casts the canopy's shadows through it as shafts
        // thinner by the water than GL's, but thinning far more slowly with height, so the ranges kilometres off
        // stand in blue haze behind the valley
        scene->fog = FogExp2(Color(0.62f, 0.70f, 0.74f), 0.004f * airScale);
#ifdef THREEPP_WITH_VULKAN
        VulkanRenderer::HeightFogSettings air;
        air.density = 0.f;
        air.baseY = 0.f;
        air.falloff = 150.f;
        air.noiseAmount = 0.f;
        vk->setHeightFog(air);
        if (ptSamples > 0) vk->setAutoLod(false);
#endif
    }

    auto sun = DirectionalLight::create(Color(1.f, 0.94f, 0.82f), 3.4f);
    sun->position.copy(look.sun).multiplyScalar(80.f);
    sun->castShadow = true;
    sun->shadow->mapSize.set(4096, 4096);
    sun->shadow->bias = -0.0003f;
    sun->shadow->normalBias = 0.02f;
    if (auto* sc = sun->shadow->camera->as<OrthographicCamera>()) {
        sc->left = sc->bottom = -42.f;
        sc->right = sc->top = 42.f;
        sc->nearPlane = 1.f;
        sc->farPlane = 220.f;
        sc->updateProjectionMatrix();
    }
    auto sunTarget = Object3D::create();
    sunTarget->position.set(0, 0, -4);
    sun->position.add(sunTarget->position);
    scene->add(sunTarget);
    sun->setTarget(*sunTarget);
    scene->add(sun);

    // ---- surfaces -----------------------------------------------------------------------------------------------
    const auto tTex = std::chrono::steady_clock::now();
    const Materials materials;
    std::cout << "textures " << secondsSince(tTex) << " s" << std::endl;

    // ---- the valley ---------------------------------------------------------------------------------------------
    const auto tGeo = std::chrono::steady_clock::now();
    scene->add(makeTerrain(materials.groundMat));
    const auto rocks = placeRocks();
    std::vector<float> rockPoints;
    scene->add(makeRocks(rocks, materials.rockMat, &rockPoints));
    scene->add(makeGravel(rocks, materials.rockMat));
    std::cout << "valley " << secondsSince(tGeo) << " s, " << rocks.size() << " boulders" << std::endl;

    // ---- the mill, its wheel and flume, the bridge ----------------------------------------------------------------
    const auto tBuild = std::chrono::steady_clock::now();
    MillShell millShell(materials);
    scene->add(millShell.root);
    WheelModel wheel(materials);
    scene->add(wheel.root);
    {
        std::shared_ptr<Mesh> pierMortar;
        scene->add(wheelPier(materials, pierMortar));
        scene->add(pierMortar);
    }
    FlumeModel flume(materials);
    scene->add(flume.root);
    BridgeModel bridge(materials);
    scene->add(bridge.root);
    std::cout << "buildings " << secondsSince(tBuild) << " s" << std::endl;

    // ---- the water: the stream, the flume, the wheel's buckets and its jet, the spray --------------------------------
    const auto tWater = std::chrono::steady_clock::now();
    auto ripples = waterRipples();
    std::shared_ptr<MeshPhysicalMaterial> waterMat;
    std::shared_ptr<Material> tintMat;// GL only: the absorption pass
    if (vk) waterMat = makeWaterMaterialPhysical(ripples);
    else {
        waterMat = makeWaterMaterial(ripples);
        tintMat = makeWaterTint();
    }
    const TexSet fall = fallingWaterTexture(3);
    auto jetMat = fall.material(1.f);
    jetMat->transparent = true;
    jetMat->depthWrite = false;
    jetMat->side = Side::Double;
    jetMat->color.setRGB(0.9f, 0.9f, 0.88f);
    jetMat->metalnessMap = nullptr;
    jetMat->metalness = 0.f;
    jetMat->vertexColors = true;// RGBA: the sheets fade where they break up
    StreamWater stream(rockPoints, waterMat, tintMat, jetMat);
    if (flowArg >= 0.f) stream.setInflow(flowArg);
    scene->add(stream.mesh);
    if (stream.tint) scene->add(stream.tint);
    scene->add(stream.curtain);
    if (vk) {
        stream.addFoamSkin();
        scene->add(stream.foamSkin);
    }
    // the same water in the flume and in the buckets, each with its absorption pass on GL
    const auto withTint = [&](const std::shared_ptr<Mesh>& waterMesh, Object3D& parent) {
        waterMesh->renderOrder = 2;
        parent.add(waterMesh);
        if (!tintMat) return;
        auto t = Mesh::create(waterMesh->geometry(), tintMat);
        t->renderOrder = 1;
        t->frustumCulled = false;
        parent.add(t);
    };
    auto flumeWater = makeFlumeWater(waterMat, 0.12f, 1.6f);
    withTint(flumeWater, *scene);
    WheelSim sim;
    for (int k = 0; k < 60 * 45; ++k) sim.step(1.f / 60);// the wheel has been running a while
    BucketWater buckets(waterMat);
    withTint(buckets.mesh, *wheel.root);
    auto jet = makeJet(sim, jetMat);
    jet->renderOrder = 3;
    wheel.root->add(jet);
    PourSheets pours(jetMat, site::nominalWater(site::wheelZ + 1.f) - site::wheelY);
    wheel.root->add(pours.mesh);
    Spray spray;
    scene->add(spray.drops);
    scene->add(spray.mist);
    // after the water (its passes write over what is behind them)
    spray.drops->mesh().renderOrder = 4;
    spray.mist->mesh().renderOrder = 5;
    spray.veil->mesh().renderOrder = 6;
    scene->add(spray.veil);
    pours.mesh->renderOrder = 3;

    // GL: a planar mirror of the scene for the water, at the level of the water in front of the camera
    std::unique_ptr<PlanarReflection> mirror;
    if (gl) {
        PlanarReflection::Options o;
        o.resolutionScale = 0.4f;
        o.farDistance = 70.f;
        o.clipBias = 0.02f;
        mirror = std::make_unique<PlanarReflection>(o);
        mirror->hidden = {stream.mesh.get(), flumeWater.get(), buckets.mesh.get(), spray.drops.get(), spray.mist.get()};
        if (stream.tint) mirror->hidden.push_back(stream.tint.get());
    }
    // seams in the old flume that drip
    std::vector<Vector3> leaks;
    {
        Rng rng(55);
        for (int k = 0; k < 7; ++k) {
            const float z = rng.range(site::flumeStartZ + 1.5f, site::wheelZ - 3.5f);
            leaks.emplace_back(site::flumeX + rng.range(-0.42f, 0.42f), site::flumeFloor(z) - 0.06f, z);
        }
    }
    std::cout << "water " << secondsSince(tWater) << " s, wheel at " << sim.omega * 60.f / 6.2831853f << " rpm" << std::endl;

    // ---- the woods ----------------------------------------------------------------------------------------------
    const auto tWoods = std::chrono::steady_clock::now();
    scene->add(plantWoods(materials));
    std::cout << "woods " << secondsSince(tWoods) << " s" << std::endl;

    // ---- the mountains beyond the valley ---------------------------------------------------------------------------
    // The engine's MountainBackdrop, as the glacier walker has it: ranges from a kilometre and a quarter out to the
    // horizon, derivative-damped and warped so they have sharp crests, smooth flanks and gullies, rock in strata and
    // snow lying where it would, hazed toward the sky. A massif stands where the hero view's gap in the canopy looks
    // up the valley, far enough off to be only just there in the haze.
    {
        const Vector3 heroEye = views.at("hero").eye, heroAt = views.at("hero").target;
        Vector3 toward(heroAt.x - heroEye.x, 0.f, heroAt.z - heroEye.z);
        toward.normalize();
        const float turn = 0.1f;// a little left of the view's middle, where the canopy opens
        const Vector3 dir(toward.x * std::cos(turn) + toward.z * std::sin(turn), 0.f, toward.z * std::cos(turn) - toward.x * std::sin(turn));
        terrain::MountainBackdropOptions r;
        r.ground = [](float x, float z) { return 8.f + 0.12f * std::min(std::hypot(x, z * 0.5f), 400.f); };
        // summer: no snow. The "snow" rule is given the forest's colour and the steepest faces stay bare rock, so the
        // range reads as wooded ridges with crags; a long massif (a ridge, not a lone cone) fills the gap
        r.massifs.push_back({Vector2(heroEye.x + dir.x * 8000.f, heroEye.z + dir.z * 8000.f), 3200.f, 950.f});
        r.snowSlope = 0.45f;
        r.snowColor = Color(0.028f, 0.042f, 0.022f);
        r.rockColor = Color(0.04f, 0.038f, 0.034f);
        r.rockBandColor = Color(0.075f, 0.068f, 0.058f);
        // the air between: a blue a little darker than the sky at the horizon, so the far ridge is a faint silhouette
        r.hazeColor = Color(0.4f, 0.5f, 0.6f);
        r.hazeSunColor = Color(0.55f, 0.6f, 0.65f);
        r.hazeDistance = 5500;
        r.hazeColor = Color(0.6f, 0.72f, 0.82f);// the sky at the horizon
        r.hazeSunColor = Color(1.f, 0.9f, 0.74f);
        r.sunDirection = look.sun;
        if (vk) r.shading = terrain::MountainBackdropOptions::Shading::Textured;// Vulkan's air is the haze
        auto mountains = terrain::MountainBackdrop::create(r);
        mountains->mesh()->autoLod = false;// its rings are spaced for distance already
        scene->add(mountains);
    }

    // ---- the mill's inside --------------------------------------------------------------------------------------
    Interior interior(materials);
    scene->add(interior.root());
    interior.setSun(look.sun);
    // Light inside: daylight comes in at the windows and doors and is scattered round each room by its boards. GL has
    // no bounce light at all, and a trace through the small grimy windows under the canopy leaves the rooms near
    // black, so a dim, warm fill on each floor stands in for that scattered light (no shadow maps; the path tracer
    // shadows them itself, so none leaks out).
    for (const auto& [fy, power] : {std::pair{2.6f, 2.0f}, {5.7f, 2.0f}, {8.3f, 1.8f}, {10.9f, 1.4f}}) {
        auto fill = PointLight::create(Color(0.95f, 0.9f, 0.82f), power, 6.5f, 2.f);
        fill->position.set(0.5f * (site::millWest + site::millEast) + 0.4f, fy, 0.f);
        fill->castShadow = false;
        scene->add(fill);
    }
    if (gl) applyAir(*scene, airLook);

    // ---- the camera ---------------------------------------------------------------------------------------------
    View v = views.at(viewName);
    if (freeEye) v.eye = eye;
    if (freeLook) v.target = lookAt;
    if (fov > 0) v.fov = fov;
    auto camera = PerspectiveCamera::create(v.fov, canvas.aspect(), 0.08f, 16000.f);// the mountains stand out to 9 km
    camera->position.copy(v.eye);
    OrbitControls controls{*camera, canvas};
    controls.target.copy(v.target);
    controls.update();

    // ---- post: ambient occlusion over the forward render --------------------------------------------------------
    std::unique_ptr<EffectComposer> composer;
    EffectComposer::Options options;
    options.samples = 4;
    if (gl) composer = std::make_unique<EffectComposer>(*gl, options);
    if (composer) composer->addPass(std::make_shared<RenderPass>(*scene, *camera));
    if (gl) {
        GTAOParameters aoParameters;
        aoParameters.radius = 0.6f;
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
        auto gtao = std::make_shared<GTAOPass>(*scene, *camera, width, height, aoParameters, pdParameters);
        gtao->blendIntensity = 0.85f;
        gtao->enabled = ao;
        composer->addPass(gtao);
        composer->addPass(std::make_shared<OutputPass>());
    }

    // ---- the live window's controls: orbit or walk, the views on the number keys, a small panel -------------------
    bool walking = false, paused = false, dragging = false;
    float mirrorLevel = site::nominalWater(camera->position.z - 8.f);
    float streamMsShown = 0;
    float yaw = 0, pitch = 0;
    Vector2 lastMouse;
    float gate = 1.f;
    const float fullInflow = sim.inflow;
    const auto lookFromCamera = [&] {
        Vector3 d;
        camera->getWorldDirection(d);
        yaw = std::atan2(d.x, -d.z);
        pitch = std::asin(std::clamp(d.y, -1.f, 1.f));
    };
    const auto goTo = [&](const std::string& name) {
        const View& w = views.at(name);
        camera->position.copy(w.eye);
        camera->fov = w.fov;
        camera->updateProjectionMatrix();
        controls.target.copy(w.target);
        camera->lookAt(w.target);
        controls.update();
        lookFromCamera();
    };
    std::unique_ptr<ImguiFunctionalContext> ui;
    IOCapture capture{};
    if (!headless) {
        ui = std::make_unique<ImguiFunctionalContext>(canvas, renderer, [&] {
            ImGui::SetNextWindowPos({10, 10}, 0, {0, 0});
            ImGui::SetNextWindowSize({330, 0}, 0);
            ImGui::Begin("Watermill");
            ImGui::Text("wheel %.1f rpm, torque %.1f kN m", sim.omega * 60.f / 6.2831853f, sim.torque / 1000.f);
            ImGui::Text("water on the wheel %.0f kg, spilling %.3f m3/s", sim.waterMass, sim.spillRate);
            if (ImGui::SliderFloat("flume gate", &gate, 0.f, 1.5f, "%.2f")) sim.inflow = fullInflow * gate;
            float creek = stream.inflow();
            if (ImGui::SliderFloat("creek flow (m3/s)", &creek, 0.1f, 8.f, "%.1f", ImGuiSliderFlags_Logarithmic)) stream.setInflow(creek);
            ImGui::Text("stream %.1f ms a frame", streamMsShown);
            ImGui::Text(walking ? "walking: WASD, Q/E, Shift, drag to look; Tab: orbit" : "orbit; Tab: walk; 0-9: views; P: pause; [ ]: creek");
            ImGui::End();
        });
        capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
        capture.preventScrollEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
        canvas.setIOCapture(&capture);
    }
    canvas.onKeyPressed([&](KeyEvent e) {
        if (e.key == Key::TAB) {
            walking = !walking;
            controls.enabled = !walking;
            if (walking) lookFromCamera();
            else {
                Vector3 d;
                camera->getWorldDirection(d);
                controls.target.copy(camera->position).add(d.multiplyScalar(4.f));
                controls.update();
            }
        } else if (e.key == Key::P) {
            paused = !paused;
        } else if (e.key == Key::LEFT_BRACKET) {
            stream.setInflow(std::max(0.1f, stream.inflow() / 1.4f));
        } else if (e.key == Key::RIGHT_BRACKET) {
            stream.setInflow(std::min(8.f, stream.inflow() * 1.4f));
        } else if (e.key >= Key::NUM_0 && e.key <= Key::NUM_9) {
            goTo(viewKeys[static_cast<int>(e.key) - static_cast<int>(Key::NUM_0)]);
        }
    });
    canvas.onMouseDown([&](int button, Vector2 pos) {
        if (button == 0) {
            dragging = true;
            lastMouse = pos;
        }
    });
    canvas.onMouseUp([&](int button, Vector2) {
        if (button == 0) dragging = false;
    });
    canvas.onMouseMove([&](Vector2 pos) {
        if (walking && dragging) {
            yaw += 0.004f * (pos.x - lastMouse.x);
            pitch = std::clamp(pitch - 0.004f * (pos.y - lastMouse.y), -1.4f, 1.4f);
        }
        lastMouse = pos;
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
        if (composer) composer->setSize(size.width(), size.height());
    });

    std::cout << "ready " << secondsSince(t0) << " s" << std::endl;
    Clock clock;
    float simTime = 0;
    Rng fx(99);
    const auto surfaceAt = [&](float x, float z) { return std::max(stream.levelAt(x, z), site::groundHeight(x, z)); };
    const Vector3 axle(site::wheelX, site::wheelY, site::wheelZ);
    int benchDone = 0;
    double streamTotal = 0;
    auto benchStart = std::chrono::steady_clock::now();
    canvas.animate([&] {
        const float frameDt = headless ? 1.f / 60 : std::min(clock.getDelta(), 0.05f);
        if (walking) {
            const float speed = (canvas.isKeyDown(Key::LEFT_SHIFT) ? 6.f : 1.8f) * frameDt;
            const Vector3 forward(std::sin(yaw), 0, -std::cos(yaw)), right(std::cos(yaw), 0, std::sin(yaw));
            if (canvas.isKeyDown(Key::W)) camera->position.add(Vector3(forward).multiplyScalar(speed));
            if (canvas.isKeyDown(Key::S)) camera->position.sub(Vector3(forward).multiplyScalar(speed));
            if (canvas.isKeyDown(Key::D)) camera->position.add(Vector3(right).multiplyScalar(speed));
            if (canvas.isKeyDown(Key::A)) camera->position.sub(Vector3(right).multiplyScalar(speed));
            if (canvas.isKeyDown(Key::E)) camera->position.y += speed;
            if (canvas.isKeyDown(Key::Q)) camera->position.y -= speed;
            const Vector3 look(std::sin(yaw) * std::cos(pitch), std::sin(pitch), -std::cos(yaw) * std::cos(pitch));
            camera->lookAt(Vector3(camera->position).add(look));
        }
        const float dt = paused ? 0.f : frameDt;
        simTime += dt;
        // the wheel, turned by the water in its buckets; what spills falls into the pit and runs into the stream
        sim.step(dt);
        wheel.setAngle(sim.angle);
        interior.update(sim.angle, sim.omega, dt);
        buckets.update(sim);
        pours.update(sim, std::max(dt, 1e-4f));
        stream.setTailInflow(sim.spillRate);
        const auto tStream = std::chrono::steady_clock::now();
        stream.step(dt);
        stream.updateMesh(simTime);
        const double streamMs = 1000.0 * secondsSince(tStream);
        for (const auto& t : {fall.albedo, fall.normal, fall.orm}) t->offset.y = -simTime * 2.2f;
        // spray: drops off the spilling buckets, a splash where the jet lands, drips from the flume, mist off the
        // pit and off the stream's white water
        for (const auto& sp : sim.spills) {
            const int count = std::clamp(static_cast<int>(sp.volume * 2.f * 1e4f), 0, 14);
            for (int k = 0; k < count; ++k) {
                const Vector3 at = axle + Vector3(fx.range(-0.47f, 0.47f), sp.y, sp.z);
                spray.emitDrops(at, Vector3(0, sp.vy, sp.vz), 1, 0.1f);
            }
        }
        for (int k = 0; k < 10; ++k) {
            const Vector3 at = axle + Vector3(fx.range(-0.4f, 0.4f), sim.hitY, sim.hitZ);
            spray.emitDrops(at, Vector3(0, 1.1f, 0.6f), 1, 0.6f);
        }
        // the falls: mist and spray where the sheet lands in the pool, in proportion to what goes over
        for (const auto& p : stream.pours) {
            if (p.q < 0.01f) continue;
            const float zHit = site::fallsLipZ(p.x) + p.speed * std::sqrt(2.f * std::max(p.top - p.bottom, 0.05f) / 9.81f);
            if (fx() < 6.f * p.q) spray.emitMist(Vector3(p.x + fx.range(-0.1f, 0.1f), p.bottom + 0.15f, zHit + fx.range(-0.2f, 0.5f)), Vector3(0, 0.3f, 0.45f), 0.6f, 0.25f);
            if (fx() < 8.f * p.q) spray.emitDrops(Vector3(p.x, p.bottom + 0.05f, zHit), Vector3(0, 1.4f, 0.8f), 2, 0.7f, 0.08f);
        }
        for (const auto& l : leaks)
            if (fx() < 0.25f) spray.emitDrops(l, Vector3(0, -0.2f, 0), 1, 0.02f, 0.005f);
        if (fx() < 0.12f) spray.emitMist(axle + Vector3(fx.range(-0.5f, 0.5f), -2.85f, fx.range(0.3f, 2.2f)), Vector3(0, 0.3f, 0.2f), 0.35f, 0.5f);
        for (int k = 0; k < 40; ++k) {
            const int c = static_cast<int>(fx.next() % static_cast<uint32_t>(StreamWater::nx * StreamWater::nz));
            if (!stream.swe.activeAt(c) || stream.swe.depth(c) < 0.03f || stream.swe.foamAt(c) < 0.55f) continue;
            const float x = stream.swe.xOf(c % StreamWater::nx), z = stream.swe.zOf(c / StreamWater::nx);
            float vx, vz;
            stream.swe.velocityAt(x, z, vx, vz);
            if (vx * vx + vz * vz < 0.5f) continue;
            const float y = stream.swe.level(c);
            if (fx() < 0.3f) spray.emitMist(Vector3(x, y + 0.05f, z), Vector3(vx * 0.3f, 0.15f, vz * 0.3f), 0.18f, 0.6f);
            spray.emitDrops(Vector3(x, y + 0.02f, z), Vector3(vx * 0.6f, 0.9f, vz * 0.6f), 2, 0.35f, 0.05f);
        }
        // the veil of damp air over the water: thickest over the falls and the wheel's pit
        for (int k = 0; k < 2; ++k) {
            const int c = static_cast<int>(fx.next() % static_cast<uint32_t>(StreamWater::nx * StreamWater::nz));
            if (!stream.swe.activeAt(c) || stream.swe.depth(c) < 0.05f) continue;
            const float x = stream.swe.xOf(c % StreamWater::nx), z = stream.swe.zOf(c / StreamWater::nx);
            const float nearFalls = smooth(12.f, 2.f, std::abs(z - site::fallsZ - 3.f)), nearWheel = smooth(8.f, 2.f, std::hypot(x - site::wheelX, z - site::wheelZ));
            if (fx() < 0.25f + 0.75f * std::max(nearFalls, nearWheel)) spray.emitVeil(Vector3(x, stream.swe.level(c) + fx.range(0.2f, 0.9f), z), fx.range(2.f, 3.5f));
        }
        spray.step(dt, simTime, surfaceAt);
        spray.commit();
        if (!walking) controls.update();
        streamMsShown += 0.05f * (static_cast<float>(streamMs) - streamMsShown);
        if (mirror) {
            // the mirror lies on the water the camera is looking at: march its view forward to the first water
            Vector3 d;
            camera->getWorldDirection(d);
            Vector3 flat(d.x, 0, d.z);
            if (flat.lengthSq() < 1e-6f) flat.set(0, 0, -1);
            flat.normalize();
            for (float t = 2.f; t < 40.f; t += 1.f) {
                const float level = stream.levelAt(camera->position.x + flat.x * t, camera->position.z + flat.z * t);
                if (level > -1e8f && level < camera->position.y) {
                    mirrorLevel += (headless ? 1.f : 0.2f) * (level - mirrorLevel);
                    break;
                }
            }
            mirror->setPlane(Vector3(0, mirrorLevel, 0), Vector3(0, 1, 0));
            if (mirror->render(renderer, *scene, *camera)) {
                const bool first = !waterMat->planarReflection;
                mirror->apply(*waterMat);
                if (first) waterMat->needsUpdate();
            }
        }
        if (composer) composer->render(frameDt);
        else renderer.render(*scene, *camera);
        if (ui) ui->render();
        if (headless && simTime >= shotTime && benchDone < benchFrames) {
            if (benchDone == 0) benchStart = std::chrono::steady_clock::now();
            streamTotal += streamMs;
            ++benchDone;
            return;
        }
#ifdef THREEPP_WITH_VULKAN
        if (headless && simTime >= shotTime && vk && ptSamples > 0) {
            // the trace converges over a still scene: the clock stops here
            VulkanRenderer::PathTraceParams params;
            params.samplesPerFrame = 16;
            params.maxBounces = 6;
            params.fireflyClamp = 20.f;
            const auto tTrace = std::chrono::steady_clock::now();
            vk->pathTraceToFile(*scene, *camera, shotPath, static_cast<uint32_t>(ptSamples), params);
            std::cout << "path traced " << ptSamples << " spp in " << secondsSince(tTrace) << " s; wrote " << shotPath << std::endl;
            std::exit(0);
        }
#endif
        if (headless && simTime >= shotTime) {
            if (benchFrames > 0) {
                const double ms = 1000.0 * secondsSince(benchStart) / benchFrames;
                std::cout << "bench " << benchFrames << " frames: " << ms << " ms a frame, stream " << streamTotal / benchFrames << " ms; wheel "
                          << sim.omega * 60.f / 6.2831853f << " rpm, " << spray.aliveDrops() << " drops" << std::endl;
            }
            renderer.writeFramebuffer(shotPath);
            std::cout << "wrote " << shotPath << std::endl;
            std::exit(0);
        }
    });
}
