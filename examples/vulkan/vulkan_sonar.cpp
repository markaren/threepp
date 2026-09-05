// Imaging sonar on an ROV inside a net pen — SonarSensor on the Vulkan
// renderer's ray tracer.
//
// The scene is the acoustic version of the thing a camera sees badly: a
// cylindrical net (twine to the eye, a wall to sound), a steel collar, a
// seabed, and a school of fish. The ROV orbits inside the pen with a
// 130 deg x 20 deg forward-looking sonar tilted down, and the fan on the
// right is its echogram: range up, bearing across, brightness = echo.
//
// The net is the point. Its visible mesh is a wireframe, which the renderer
// draws as an overlay and never puts in the acceleration structure — so the
// sonar would trace straight through it. A second, solid cylinder sits on
// VulkanRenderer::kSensorOnlyLayer: with setSensorOnlySurfaces(true) the
// sonar's rays strike it and the camera never renders it. That is how an
// acoustic proxy is authored: geometry present for the sensors, absent from
// the picture. The "sonar sees proxies" checkbox flips the master switch and
// the net vanishes from the fan while the collar and fish remain.
//
// Reflectivity is per target, keyed on the stable instance id: net 1.0,
// collar 1.0, seabed 0.6, fish 0.35 — the sonar ignores optical material.
//
//     vulkan_sonar                   interactive
//     vulkan_sonar --shot out.png    headless: render + scan a few frames,
//                                    write the frame, print echo statistics

#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/helpers/SonarSensor.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kFishCount = 64;
    constexpr float kPenRadius = 8.f;
    constexpr float kPenDepth = 12.f;
    constexpr int kFishIdBase = 10;

    // ── Fan display ──────────────────────────────────────────────────────
    // A polar echogram drawn into a bottom-up RGBA image (the DataTexture
    // convention): apex near the bottom edge, range grows upward, bearing
    // across. Every pixel is a gather from (beam, bin), precomputed once.
    struct FanDisplay {
        unsigned int w = 512, h = 384;
        std::vector<int> beamOf, binOf;   // -1 outside the fan
        std::vector<unsigned char> overlay;// range rings + swath edge, alpha 0 elsewhere
        std::vector<unsigned char> bytes;
        std::array<std::array<unsigned char, 3>, 256> lut{};

        void build(const SonarModel& m) {
            beamOf.assign(w * h, -1);
            binOf.assign(w * h, -1);
            overlay.assign(w * h * 4, 0);
            bytes.assign(w * h * 4, 0);
            const float apexY = 12.f;
            const float scale = static_cast<float>(h) - 24.f;// pixels per maxRange
            for (unsigned int y = 0; y < h; ++y) {
                for (unsigned int x = 0; x < w; ++x) {
                    const float dx = static_cast<float>(x) - 0.5f * static_cast<float>(w);
                    const float dy = static_cast<float>(y) - apexY;
                    if (dy <= 0.f) continue;
                    const float rPx = std::hypot(dx, dy);
                    const float r = rPx / scale * m.maxRange;
                    const float bearingDeg = std::atan2(dx, dy) * math::RAD2DEG;
                    const bool inside = r < m.maxRange && std::abs(bearingDeg) < 0.5f * m.horizontalFov;
                    const std::size_t i = static_cast<std::size_t>(y) * w + x;
                    if (inside) {
                        // Beam 0 is the sensor's left-most beam (negative azimuth),
                        // which is negative bearing on the fan.
                        beamOf[i] = std::clamp(static_cast<int>((bearingDeg + 0.5f * m.horizontalFov) /
                                                                m.horizontalFov * static_cast<float>(m.beams)),
                                               0, static_cast<int>(m.beams) - 1);
                        binOf[i] = std::clamp(static_cast<int>(r / m.maxRange * static_cast<float>(m.rangeBins)),
                                              0, static_cast<int>(m.rangeBins) - 1);
                    }
                    // Range rings every 5 m, the swath edge, and the outer arc.
                    const float ringStep = m.maxRange >= 50.f ? 20.f : 5.f;
                    const float ringDist = std::abs(r - ringStep * std::round(r / ringStep));
                    const bool ring = inside && r > 0.5f && ringDist < 0.6f * m.maxRange / scale;
                    const bool edge = (std::abs(std::abs(bearingDeg) - 0.5f * m.horizontalFov) < 0.35f && r < m.maxRange) ||
                                      (std::abs(r - m.maxRange) < 0.03f * m.maxRange / (scale / 30.f) &&
                                       std::abs(bearingDeg) < 0.5f * m.horizontalFov);
                    if (ring || edge) {
                        overlay[i * 4 + 0] = 70;
                        overlay[i * 4 + 1] = 42;
                        overlay[i * 4 + 2] = 14;
                        overlay[i * 4 + 3] = 255;
                    }
                }
            }
            // Dark -> amber -> white, the classic echogram palette.
            for (int k = 0; k < 256; ++k) {
                const float t = static_cast<float>(k) / 255.f;
                float r, g, b;
                if (t < 0.5f) {
                    const float u = t / 0.5f;
                    r = 0.004f + (0.62f - 0.004f) * u;
                    g = 0.002f + (0.26f - 0.002f) * u;
                    b = 0.0f + (0.05f - 0.0f) * u;
                } else {
                    const float u = (t - 0.5f) / 0.5f;
                    r = 0.62f + (1.0f - 0.62f) * u;
                    g = 0.26f + (0.86f - 0.26f) * u;
                    b = 0.05f + (0.55f - 0.05f) * u;
                }
                // The texture is tagged sRGB, so store encoded values.
                auto enc = [](float v) {
                    v = std::clamp(v, 0.f, 1.f);
                    const float s = v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
                    return static_cast<unsigned char>(std::lround(s * 255.f));
                };
                lut[static_cast<std::size_t>(k)] = {enc(r), enc(g), enc(b)};
            }
        }

        // `a` is the (persisted) echogram; gain maps [0,1] echo to display.
        void draw(const SonarImage& img, const std::vector<float>& a, float gain) {
            for (std::size_t i = 0; i < beamOf.size(); ++i) {
                unsigned char* px = &bytes[i * 4];
                if (beamOf[i] < 0) {
                    px[0] = px[1] = px[2] = 6;
                    px[3] = 200;
                } else {
                    const float v = a[static_cast<std::size_t>(beamOf[i]) * img.bins + static_cast<std::size_t>(binOf[i])];
                    const float d = 1.f - std::exp(-gain * v);
                    const auto& c = lut[static_cast<std::size_t>(std::clamp(static_cast<int>(d * 255.f), 0, 255))];
                    px[0] = c[0];
                    px[1] = c[1];
                    px[2] = c[2];
                    px[3] = 230;
                }
                if (overlay[i * 4 + 3]) {
                    px[0] = std::max(px[0], overlay[i * 4 + 0]);
                    px[1] = std::max(px[1], overlay[i * 4 + 1]);
                    px[2] = std::max(px[2], overlay[i * 4 + 2]);
                }
            }
        }
    };

    struct Fish {
        std::shared_ptr<Mesh> mesh;
        float radius, depth, phase, speed, bob;
    };

}// namespace

int main(int argc, char** argv) {
    std::string shotPath;
    int shotFrames = 12;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot" && i + 1 < argc) shotPath = argv[++i];
        else if (a == "--frames" && i + 1 < argc) shotFrames = std::atoi(argv[++i]);
    }
    const bool headless = !shotPath.empty();

    Canvas canvas("Vulkan - imaging sonar in a net pen",
                  {{"vsync", false}, {"size", WindowSize{1600, 900}}, {"headless", headless}});
    VulkanRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;
    // The master switch for acoustic proxies: without it the sensor-only layer
    // is a mask nothing hits, and the net below is invisible to the sonar too.
    renderer.setSensorOnlySurfaces(true);

    Scene scene;
    scene.background = Color(0.02f, 0.07f, 0.11f);

    auto sun = DirectionalLight::create(Color(0.6f, 0.85f, 1.f), 2.5f);
    sun->position.set(6.f, 30.f, 10.f);
    scene.add(sun);

    auto camera = PerspectiveCamera::create(55.f, canvas.aspect(), 0.1f, 300.f);
    camera->position.set(-14.f, 9.f, 24.f);
    OrbitControls controls{*camera, canvas};
    controls.target.set(0.f, -6.f, 0.f);
    controls.update();

    // ── Seabed ───────────────────────────────────────────────────────────
    auto seabed = Mesh::create(PlaneGeometry::create(90.f, 90.f, 1, 1),
                               MeshStandardMaterial::create(
                                       MeshStandardMaterial::Params{}.color(Color(0x3a3d2b)).roughness(1.f)));
    seabed->rotation.x = -math::PI / 2;
    seabed->position.y = -kPenDepth - 3.f;
    scene.add(seabed);
    renderer.setObjectInstanceId(*seabed, 1);

    // ── Net: twine for the camera, a membrane for the sonar ──────────────
    auto netGeom = CylinderGeometry::create(kPenRadius, kPenRadius, kPenDepth, 96, 1, true);
    auto twineMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x9fbccc)));
    twineMat->wireframe = true;
    auto netVisible = Mesh::create(netGeom, twineMat);
    netVisible->position.y = -0.5f * kPenDepth;
    scene.add(netVisible);

    auto membraneMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x9fbccc)));
    membraneMat->side = Side::Double;
    auto netProxy = Mesh::create(netGeom, membraneMat);
    netProxy->position.y = -0.5f * kPenDepth;
    netProxy->layers.set(VulkanRenderer::kSensorOnlyLayer);
    scene.add(netProxy);
    renderer.setObjectInstanceId(*netProxy, 2);

    auto floorGeom = CircleGeometry::create(kPenRadius, 96);
    auto netFloorVisible = Mesh::create(floorGeom, twineMat);
    netFloorVisible->rotation.x = -math::PI / 2;
    netFloorVisible->position.y = -kPenDepth;
    scene.add(netFloorVisible);
    auto netFloorProxy = Mesh::create(floorGeom, membraneMat);
    netFloorProxy->rotation.x = -math::PI / 2;
    netFloorProxy->position.y = -kPenDepth;
    netFloorProxy->layers.set(VulkanRenderer::kSensorOnlyLayer);
    scene.add(netFloorProxy);
    renderer.setObjectInstanceId(*netFloorProxy, 2);

    // ── Collar: steel, visible, acoustically bright ──────────────────────
    auto collar = Mesh::create(TorusGeometry::create(kPenRadius, 0.28f, 12, 128),
                               MeshStandardMaterial::create(
                                       MeshStandardMaterial::Params{}.color(Color(0x8a8f96)).metalness(0.8f).roughness(0.4f)));
    collar->rotation.x = math::PI / 2;
    collar->position.y = 0.f;
    scene.add(collar);
    renderer.setObjectInstanceId(*collar, 3);

    // ── Fish ─────────────────────────────────────────────────────────────
    auto fishGeom = SphereGeometry::create(1.f, 16, 10);
    auto fishMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x6f7f8c)).roughness(0.5f));
    std::vector<Fish> fish;
    fish.reserve(kFishCount);
    for (int i = 0; i < kFishCount; ++i) {
        Fish f;
        f.mesh = Mesh::create(fishGeom, fishMat);
        // Salmon-sized, and schooling in the band the sonar sweeps (the ROV
        // sits at -5 m with a 20 deg aperture tilted 6 deg down).
        f.mesh->scale.set(0.36f, 0.09f, 0.075f);
        f.radius = 2.4f + 4.8f * static_cast<float>((i * 7) % 11) / 10.f;
        f.depth = -3.5f - 4.f * static_cast<float>((i * 5) % 9) / 8.f;
        f.phase = static_cast<float>(i) * 0.7f;
        f.speed = 0.25f + 0.1f * static_cast<float>(i % 4);
        f.bob = 0.3f + 0.2f * static_cast<float>(i % 3);
        scene.add(f.mesh);
        renderer.setObjectInstanceId(*f.mesh, kFishIdBase + i);
        fish.push_back(f);
    }

    // ── ROV with the sonar mounted on its nose ───────────────────────────
    auto rov = Mesh::create(BoxGeometry::create(0.6f, 0.35f, 0.8f),
                            MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0xf2c230))));
    scene.add(rov);
    renderer.setObjectInstanceId(*rov, 4);

    const std::array<const char*, 4> modelNames{"Wide 130 (20 m)", "Oculus M750d (120 m)",
                                                "BlueView M900 (100 m)", "Gemini 720is (120 m)"};
    int currentModel = 0, cachedModel = 0;
    auto makeModel = [&]() -> SonarModel {
        switch (currentModel) {
            case 1: return SonarModel::OculusM750d();
            case 2: return SonarModel::BlueViewM900();
            case 3: return SonarModel::Gemini720is();
            default: return SonarModel::Wide130();
        }
    };
    auto sonar = std::make_shared<SonarSensor>(makeModel());
    sonar->position.set(0.f, -0.1f, -0.42f);
    sonar->rotation.x = -6.f * math::DEG2RAD;// tilted down toward the school
    sonar->reflectivity.defaultValue = 1.f;
    sonar->reflectivity.set(1, 0.6f);// seabed
    sonar->reflectivity.set(2, 1.0f);// net
    sonar->reflectivity.set(3, 1.0f);// collar
    for (int i = 0; i < kFishCount; ++i) sonar->reflectivity.set(kFishIdBase + i, 0.35f);
    rov->add(sonar);

    // ── Echogram panel ───────────────────────────────────────────────────
    FanDisplay fan;
    fan.build(sonar->model());
    auto panelTex = DataTexture::create(ImageData{std::vector<unsigned char>(fan.w * fan.h * 4, 0u)}, fan.w, fan.h);
    panelTex->colorSpace = ColorSpace::sRGB;
    auto panelMat = SpriteMaterial::create();
    panelMat->map = panelTex;
    auto panel = Sprite::create(panelMat);
    panel->scale.set(static_cast<float>(fan.w), static_cast<float>(fan.h), 1.f);
    panel->screenSpace = true;
    panel->screenAnchor.set(1.f, 1.f);
    panel->center.set(1.f, 1.f);
    panel->position.set(-10.f, -10.f, 0.f);
    scene.add(panel);

    // ── Hit cloud: where the rays actually struck ────────────────────────
    const unsigned int kMaxHits = sonar->rayCount() * 4;
    auto cloudGeom = BufferGeometry::create();
    cloudGeom->setAttribute("position", FloatBufferAttribute::create(std::vector<float>(kMaxHits * 3), 3));
    cloudGeom->setAttribute("color", FloatBufferAttribute::create(std::vector<float>(kMaxHits * 3), 3));
    cloudGeom->getAttribute<float>("position")->setUsage(DrawUsage::Dynamic);
    cloudGeom->getAttribute<float>("color")->setUsage(DrawUsage::Dynamic);
    cloudGeom->setDrawRange(0, 0);
    auto cloudMat = PointsMaterial::create(PointsMaterial::Params{}.size(3.f).sizeAttenuation(false).vertexColors(true));
    auto cloud = Points::create(cloudGeom, cloudMat);
    cloud->frustumCulled = false;
    scene.add(cloud);

    // ── UI state ─────────────────────────────────────────────────────────
    float gain = 4.f;
    float speckle = 0.25f;
    bool persistence = true;
    bool sonarSeesProxies = true;
    bool showHits = true;
    bool orbit = true;
    float orbitSpeed = 0.18f;
    int maxReturns = 1;
    float lastScanMs = 0.f;
    int lastEchoes = 0, lastHits = 0;

    std::optional<ImguiFunctionalContext> ui;
    if (!headless) ui.emplace(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({360, 430});
        ImGui::Begin("Imaging sonar");
        ImGui::TextWrapped("Rays through the renderer's TLAS, folded into an echogram. "
                           "The net is a wireframe to the camera and a sensor-only membrane to the sonar.");
        ImGui::Separator();
        ImGui::Combo("Sonar", &currentModel, modelNames.data(), static_cast<int>(modelNames.size()));
        ImGui::Checkbox("ROV orbits", &orbit);
        ImGui::SliderFloat("Orbit speed", &orbitSpeed, 0.f, 0.6f);
        ImGui::Separator();
        ImGui::Text("Echo model");
        float atten = sonar->model().attenuation;
        if (ImGui::SliderFloat("Attenuation (1/m)", &atten, 0.f, 0.2f, "%.3f")) sonar->setAttenuation(atten);
        float floor = sonar->model().incidenceFloor;
        if (ImGui::SliderFloat("Incidence floor", &floor, 0.f, 1.f)) sonar->setIncidenceFloor(floor);
        ImGui::SliderFloat("Speckle", &speckle, 0.f, 0.6f);
        ImGui::SliderInt("Returns per ray", &maxReturns, 1, 3);
        ImGui::Separator();
        ImGui::Text("Scene");
        ImGui::Checkbox("Sonar sees proxies (net)", &sonarSeesProxies);
        ImGui::Separator();
        ImGui::Text("Display");
        ImGui::SliderFloat("Gain", &gain, 1.f, 12.f);
        ImGui::Checkbox("3-frame persistence", &persistence);
        ImGui::Checkbox("Show hit cloud", &showHits);
        ImGui::Separator();
        ImGui::Text("%u rays, %u beams x %u bins", sonar->rayCount(), sonar->model().beams, sonar->model().rangeBins);
        ImGui::Text("scan %.2f ms, %d hits, %d echo bins", double(lastScanMs), lastHits, lastEchoes);
        ImGui::End();
    });

    SonarImage image;
    std::array<std::vector<float>, 3> history;
    std::vector<float> persisted;
    int historyHead = 0;
    Clock clock;
    double simT = 0.0;
    double rovAngle = 0.0;

    auto stepScene = [&](float dt) {
        simT += dt;
        renderer.setSimTime(simT);
        sonar->setSimTime(simT);
        if (orbit) rovAngle += dt * orbitSpeed;
        const float rr = 3.6f;
        rov->position.set(rr * static_cast<float>(std::cos(rovAngle)), -5.f,
                          rr * static_cast<float>(std::sin(rovAngle)));
        // Nose along the tangent: forward is local -Z.
        rov->rotation.y = static_cast<float>(-rovAngle) + math::PI;
        for (auto& f : fish) {
            const float a = f.phase + static_cast<float>(simT) * f.speed;
            f.mesh->position.set(f.radius * std::cos(a), f.depth + f.bob * std::sin(a * 2.3f), f.radius * std::sin(a));
            f.mesh->rotation.y = -a + math::PI / 2;
        }
    };

    auto uploadHits = [&]() {
        auto* pos = cloudGeom->getAttribute<float>("position");
        auto* col = cloudGeom->getAttribute<float>("color");
        auto& p = pos->array();
        auto& c = col->array();
        unsigned int n = 0;
        for (const auto& r : sonar->lastReturns()) {
            if (r.returnNo <= 0 || n >= kMaxHits) continue;
            p[n * 3 + 0] = r.position.x;
            p[n * 3 + 1] = r.position.y;
            p[n * 3 + 2] = r.position.z;
            const float t = std::clamp(1.f - r.distance / sonar->model().maxRange, 0.f, 1.f);
            c[n * 3 + 0] = 0.2f + 0.8f * t;
            c[n * 3 + 1] = 0.5f * t + 0.1f;
            c[n * 3 + 2] = 0.05f;
            ++n;
        }
        cloudGeom->setDrawRange(0, static_cast<int>(n));
        pos->needsUpdate();
        col->needsUpdate();
        lastHits = static_cast<int>(n);
    };

    auto onImage = [&]() {
        // Persistence: the max over the last three frames, the way a
        // display keeps a wall from flickering as the ROV yaws.
        history[static_cast<std::size_t>(historyHead)] = image.intensity;
        historyHead = (historyHead + 1) % 3;
        persisted.assign(image.intensity.size(), 0.f);
        for (const auto& h : history) {
            if (h.size() != persisted.size()) continue;
            for (std::size_t i = 0; i < persisted.size(); ++i) persisted[i] = std::max(persisted[i], h[i]);
        }
        const auto& shown = persistence ? persisted : image.intensity;
        fan.draw(image, shown, gain);
        auto& px = panelTex->image().data<unsigned char>();
        std::copy(fan.bytes.begin(), fan.bytes.end(), px.begin());
        panelTex->needsUpdate();
        lastEchoes = static_cast<int>(std::count_if(image.intensity.begin(), image.intensity.end(),
                                                    [](float v) { return v > 0.f; }));
        uploadHits();
    };

    if (headless) {
        for (int f = 0; f < shotFrames; ++f) {
            stepScene(1.f / 60.f);
            sonar->speckle = speckle;
            canvas.animateOnce([&] { renderer.render(scene, *camera); });
            sonar->scan(renderer, image);
            onImage();
        }
        canvas.animateOnce([&] { renderer.render(scene, *camera); });
        renderer.writeFramebuffer(shotPath);
        int netBins = 0, fishBins = 0;
        for (const auto& r : sonar->lastReturns()) {
            if (r.returnNo <= 0) continue;
            if (r.hitInstanceId == 2) ++netBins;
            if (r.hitInstanceId >= kFishIdBase) ++fishBins;
        }
        std::printf("[shot] %s  rays %u  hits %d (net %d, fish %d)  echo bins %d\n",
                    shotPath.c_str(), sonar->rayCount(), lastHits, netBins, fishBins, lastEchoes);
        return 0;
    }

    canvas.animate([&] {
        const float dt = clock.getDelta();
        stepScene(dt);

        if (currentModel != cachedModel) {
            auto next = std::make_shared<SonarSensor>(makeModel());
            next->position.copy(sonar->position);
            next->rotation.copy(sonar->rotation);
            next->reflectivity = sonar->reflectivity;
            rov->remove(*sonar);
            sonar = next;
            rov->add(sonar);
            fan.build(sonar->model());
            for (auto& h : history) h.clear();
            cachedModel = currentModel;
        }
        sonar->speckle = speckle;
        sonar->params.maxReturns = static_cast<uint32_t>(maxReturns);
        renderer.setSensorOnlySurfaces(sonarSeesProxies);
        cloud->visible = showHits;

        renderer.render(scene, *camera);

        // Pipelined: collect last frame's scan, fire this frame's.
        if (sonar->scanFired() && sonar->scanReady(renderer)) {
            const auto t0 = std::chrono::steady_clock::now();
            if (sonar->scanCollect(renderer, image)) onImage();
            lastScanMs = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
        if (!sonar->scanFired()) sonar->scanBegin(renderer);

        if (ui) ui->render();
    });
}
