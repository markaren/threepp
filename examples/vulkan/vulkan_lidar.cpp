// Vulkan PT — physically correct LIDAR scanner.
//
// Demonstrates VulkanRenderer::scanLidar — a synchronous ray-tracing pass
// that re-uses the path tracer's TLAS, evaluates a back-scatter LIDAR
// equation in a custom closest-hit shader, and returns per-beam (position,
// normal, distance, intensity, hit-instance) tuples. The scan runs on the
// same GPU + acceleration structure the renderer uses, so geometry the
// path tracer sees is exactly the geometry the LIDAR sees — no
// CPU/Raycaster duplication, no cube-face raster approximation.
//
// The scene mixes matte concrete pillars (bright LIDAR returns), chrome
// spheres (almost invisible to a single-pulse LIDAR because the mirror
// lobe throws energy off-axis), and brushed-metal panels (in-between)
// so the material-dependent intensity falloff is visible. Returns are
// rendered as LineSegments overlaid on the path-traced image, each
// segment pointing along the surface normal and coloured by intensity.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/helpers/LidarModel.hpp"
#include "threepp/helpers/PathTracedLidarSensor.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/textures/DataTexture.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <array>
#include <cmath>

using namespace threepp;

namespace {

    constexpr int kMaxBeams = 250000;

    void setupScene(Scene& scene) {
        // Ground — slightly off-white concrete.
        auto groundMat = MeshStandardMaterial::create({
                {"color", Color(0xb0b0b0)},
                {"roughness", 0.95f},
        });
        auto ground = Mesh::create(BoxGeometry::create(60.f, 0.2f, 60.f), groundMat);
        ground->position.y = -0.1f;
        scene.add(ground);

        // Matte concrete pillars — bright LIDAR returns. Arranged in a
        // ring so a yaw sweep produces a clear angular signature.
        auto pillarMat = MeshStandardMaterial::create({
                {"color", Color(0xdedede)},
                {"roughness", 0.9f},
        });
        for (int i = 0; i < 8; ++i) {
            auto cyl = Mesh::create(CylinderGeometry::create(0.4f, 0.4f, 3.f), pillarMat);
            const float a = static_cast<float>(i) * math::TWO_PI / 8.f;
            cyl->position.set(7.f * std::cos(a), 1.5f, 7.f * std::sin(a));
            scene.add(cyl);
        }

        // Chrome spheres — smooth metal. The back-scatter damping
        // (1 - metalness · √(1-roughness)) kills the diffuse term, so the
        // sensor sees almost nothing here unless a beam strikes near-
        // perpendicular. Visible as "holes" in the point cloud.
        auto chromeMat = MeshStandardMaterial::create({
                {"color", Color(0xeeeeee)},
                {"roughness", 0.05f},
                {"metalness", 1.0f},
        });
        for (int i = 0; i < 4; ++i) {
            auto sph = Mesh::create(SphereGeometry::create(0.6f, 48, 32), chromeMat);
            const float a = static_cast<float>(i) * math::TWO_PI / 4.f + 0.3f;
            sph->position.set(3.5f * std::cos(a), 0.6f, 3.5f * std::sin(a));
            scene.add(sph);
        }

        // Brushed-metal boxes — rough metal, mid-strength returns.
        auto brushedMat = MeshStandardMaterial::create({
                {"color", Color(0xa9a9a9)},
                {"roughness", 0.55f},
                {"metalness", 0.8f},
        });
        for (int i = 0; i < 5; ++i) {
            auto box = Mesh::create(BoxGeometry::create(0.9f, 1.6f, 0.4f), brushedMat);
            const float a = static_cast<float>(i) * math::TWO_PI / 5.f + 0.7f;
            box->position.set(11.f * std::cos(a), 0.8f, 11.f * std::sin(a));
            box->rotation.y = -a;
            scene.add(box);
        }

        // Coloured target rods — verify intensity tracks albedo luminance.
        std::array<Color, 4> rodColors{Color(0xff5050), Color(0x60ff60), Color(0x6080ff), Color(0xffffff)};
        for (size_t i = 0; i < rodColors.size(); ++i) {
            auto mat = MeshStandardMaterial::create({{"color", rodColors[i]}, {"roughness", 0.9f}});
            auto rod = Mesh::create(CylinderGeometry::create(0.12f, 0.12f, 2.5f), mat);
            rod->position.set(-12.f + 2.f * static_cast<float>(i), 1.25f, 6.f);
            scene.add(rod);
        }

        // Back wall — long range reference.
        auto wallMat = MeshStandardMaterial::create({
                {"color", Color(0xc0c0c0)},
                {"roughness", 0.95f},
        });
        auto wall = Mesh::create(BoxGeometry::create(60.f, 5.f, 0.4f), wallMat);
        wall->position.set(0.f, 2.5f, -18.f);
        scene.add(wall);

        // Lighting — keep it simple so the scene visualisation is readable.
        scene.add(AmbientLight::create(Color(0xffffff), 0.35f));
        auto sun = DirectionalLight::create(Color(0xffe9c8), 2.0f);
        sun->position.set(8.f, 14.f, 6.f);
        scene.add(sun);
    }

    // Refresh the Points geometry from the return set. One vertex per
    // return, colour-mapped by intensity. drawRange limits the rendered
    // count so the over-allocated buffer's tail isn't drawn.
    void updateLidarVisualization(Points& cloud,
                                  const std::vector<LidarReturn>& returns,
                                  float colorGain) {
        auto& geom = *cloud.geometry();
        auto* posAttr = geom.getAttribute<float>("position");
        auto* colAttr = geom.getAttribute<float>("color");
        if (!posAttr || !colAttr) return;

        const int maxVerts = posAttr->count();
        int vi = 0;
        Color c;
        for (const auto& r : returns) {
            if (r.hitInstanceId < 0) continue;
            if (vi >= maxVerts) break;

            // Colour: intensity → hue. Low intensity = deep blue,
            // mid = green/yellow, high = red. Gain stretches the visible
            // range when most returns are dim (e.g. heavy atmospheric
            // extinction or smooth-metal scenes).
            const float t = std::clamp(r.intensity * colorGain, 0.f, 1.f);
            c.setHSL((1.f - t) * 0.66f, 1.f, 0.5f);

            posAttr->setXYZ(vi, r.position.x, r.position.y, r.position.z);
            colAttr->setXYZ(vi, c.r, c.g, c.b);
            ++vi;
        }
        geom.setDrawRange(0, vi);
        posAttr->needsUpdate();
        colAttr->needsUpdate();
    }

}// namespace


int main() {

    Canvas canvas("Vulkan PT - physically correct LIDAR",
                  {{"vsync", false}, {"size", WindowSize{1600, 900}}});
    VulkanRenderer renderer(canvas);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;

    Scene scene;
    scene.background = Color(0.03f, 0.04f, 0.06f);
    setupScene(scene);

    auto camera = PerspectiveCamera::create(60.f, canvas.aspect(), 0.1f, 200.f);
    camera->position.set(18.f, 14.f, 22.f);
    OrbitControls controls{*camera, canvas};

    // ── Sensor model picker ────────────────────────────────────────────
    const std::array<const char*, 4> modelNames{"VLP-16", "HDL-32E", "OS1-64", "OS0-128"};
    int currentModel = 0;
    auto makeModel = [&]() -> LidarModel {
        switch (currentModel) {
            case 0:  return LidarModel::VLP16();
            case 1:  return LidarModel::HDL32E();
            case 2:  return LidarModel::OS1_64();
            default: return LidarModel::OS0_128();
        }
    };
    LidarModel model = makeModel();

    // PathTracedLidarSensor mirrors LidarSensor's ergonomics — Object3D
    // pose, scan() each frame after render(). Rebuilt below when the user
    // picks a different model in the UI.
    auto sensor = std::make_unique<PathTracedLidarSensor>(model, /*maxRange=*/25.f);
    sensor->position.set(0.f, 1.6f, 0.f);
    sensor->params.referenceRange = 5.f;
    sensor->params.laserPower = 1.f;
    sensor->params.atmosphericExtinction = 0.f;
    sensor->params.detectorThreshold = 0.005f;
    scene.add(*sensor);

    // ── Visualisation: Points overlay ─────────────────────────────────
    // One vertex per LIDAR return, colour-mapped by intensity. Vulkan PT
    // routes Points to the POINT_LIST overlay pipeline (drawn after TAA,
    // depth-tested against the raster G-buffer) and excludes them from
    // the TLAS, so the LIDAR beams don't see their own visualisation.
    // PointsMaterial::size controls the sprite size in pixels.
    auto cloudGeom = BufferGeometry::create();
    cloudGeom->setAttribute("position",
                            FloatBufferAttribute::create(std::vector<float>(kMaxBeams * 3), 3));
    cloudGeom->setAttribute("color",
                            FloatBufferAttribute::create(std::vector<float>(kMaxBeams * 3), 3));
    cloudGeom->getAttribute<float>("position")->setUsage(DrawUsage::Dynamic);
    cloudGeom->getAttribute<float>("color")->setUsage(DrawUsage::Dynamic);
    cloudGeom->setDrawRange(0, 0);

    auto cloudMat = PointsMaterial::create({
            {"size", 4.f},
            {"vertexColors", true},
    });

    auto cloud = Points::create(cloudGeom, cloudMat);
    cloud->frustumCulled = false;
    scene.add(cloud);

    // ── Sensor readout panel (top-right corner) ────────────────────────
    // A 2-D image laid out as (azimuth × elevation), one pixel per beam,
    // coloured by return intensity. Mirrors the "range image" view that
    // real LIDAR sensor SDKs (Ouster Studio, Velodyne VeloView) show in
    // their debug panels. Uploaded once per scan via DataTexture →
    // SpriteMaterial → screen-space Sprite.
    constexpr unsigned int kPanelW = 720;
    constexpr unsigned int kPanelH = 128;
    constexpr float        kPanelDispW = 360.f;// pixels on screen
    constexpr float        kPanelDispH = 64.f;

    auto panelTex = DataTexture::create(
            ImageData{std::vector<unsigned char>(kPanelW * kPanelH * 4, 0u)},
            kPanelW, kPanelH);
    // Tag as sRGB so the Vulkan sampler decodes to linear before shading;
    // without this the bytes we write get treated as linear and the
    // post-pipeline sRGB encode washes the colours out.
    panelTex->colorSpace = ColorSpace::sRGB;

    auto panelMat = SpriteMaterial::create();
    panelMat->map = panelTex;

    auto panel = Sprite::create(panelMat);
    panel->scale.set(kPanelDispW, kPanelDispH, 1.f);
    panel->screenSpace = true;
    panel->screenAnchor.set(1.f, 1.f); // viewport top-right
    panel->center.set(1.f, 1.f);       // sprite's top-right pivots at anchor
    panel->position.set(-10.f, -10.f, 0.f);// inset 10 px from the corner
    scene.add(panel);

    // ── UI state ───────────────────────────────────────────────────────
    float yaw = 0.f;
    float pitch = -0.05f;
    bool animateYaw = true;
    float yawRate = 0.4f;
    float pointSize = 4.0f;
    float colorGain = 3.0f;
    bool showSceneVsCloudOnly = false;
    bool showStats = true;

    int lastBeams = 0;
    int lastReturns = 0;
    float lastScanMs = 0.f;

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({340, 460});
        ImGui::Begin("Vulkan PT LIDAR");
        ImGui::TextWrapped("Beams traced via a dedicated VK_KHR_ray_tracing_pipeline; "
                           "intensity computed from the same MaterialDesc the path tracer uses.");

        ImGui::Separator();
        ImGui::Combo("Sensor model", &currentModel, modelNames.data(),
                     static_cast<int>(modelNames.size()));
        ImGui::Checkbox("Animate yaw", &animateYaw);
        ImGui::SliderFloat("Yaw rate", &yawRate, -2.f, 2.f);
        ImGui::SliderFloat("Pitch", &pitch, -0.6f, 0.6f);

        ImGui::Separator();
        ImGui::Text("LIDAR equation");
        ImGui::SliderFloat("Max range (m)", &sensor->params.maxRange, 1.f, 50.f);
        ImGui::SliderFloat("Reference range (m)", &sensor->params.referenceRange, 0.5f, 20.f);
        ImGui::SliderFloat("Laser power", &sensor->params.laserPower, 0.1f, 10.f);
        ImGui::SliderFloat("Atmospheric ext (1/m)", &sensor->params.atmosphericExtinction, 0.f, 0.2f, "%.3f");
        ImGui::SliderFloat("Detector threshold", &sensor->params.detectorThreshold, 0.f, 0.05f, "%.4f");

        ImGui::Separator();
        ImGui::Text("Visualisation");
        ImGui::Checkbox("Show point cloud only", &showSceneVsCloudOnly);
        ImGui::SliderFloat("Point size (px)", &pointSize, 1.f, 12.f);
        ImGui::SliderFloat("Colour gain", &colorGain, 1.f, 10.f);

        if (showStats) {
            ImGui::Separator();
            ImGui::Text("Beams:    %d", lastBeams);
            ImGui::Text("Returns:  %d", lastReturns);
            const float ratio = lastBeams > 0 ? 100.f * static_cast<float>(lastReturns) / static_cast<float>(lastBeams) : 0.f;
            ImGui::Text("Yield:    %.1f %%", static_cast<double>(ratio));
            ImGui::Text("Scan:     %.2f ms", static_cast<double>(lastScanMs));
        }
        ImGui::End();
    });

    IOCapture capture;
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    canvas.setIOCapture(&capture);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    std::vector<LidarReturn> returns;
    Clock clock;
    int  cachedModelIndex = currentModel;

    canvas.animate([&] {
        const float dt = clock.getDelta();
        if (animateYaw) yaw += dt * yawRate;
        sensor->rotation.set(pitch, yaw, 0.f);

        if (currentModel != cachedModelIndex) {
            // Rebuild the sensor for the new beam pattern. Carry over the
            // current pose + tunable LIDAR params so the swap is invisible
            // beyond the change in beam density.
            model = makeModel();
            auto next = std::make_unique<PathTracedLidarSensor>(model, sensor->params.maxRange);
            next->position.copy(sensor->position);
            next->rotation.copy(sensor->rotation);
            next->params = sensor->params;
            scene.remove(*sensor);
            sensor = std::move(next);
            scene.add(*sensor);
            cachedModelIndex = currentModel;
        }

        // Apply the latest point-size from the UI.
        cloudMat->size = pointSize;

        // "Point cloud only" mode: drop the tone-map exposure to 0 so the
        // path-traced image goes pitch black. We deliberately do NOT toggle
        // mesh visibility — the overlay pass scans `traverseVisible`, and
        // hiding meshes there would also remove them from the TLAS, which
        // would leave scanLidar() with nothing to hit. The cloud + panel
        // overlays draw post-tonemap, so they stay fully visible.
        renderer.toneMappingExposure = showSceneVsCloudOnly ? 0.0f : 1.0f;

        // The renderer must have at least one render() call to build the
        // TLAS before scan can read it. Render → scan → update viz.
        renderer.render(scene, *camera);

        const auto t0 = std::chrono::steady_clock::now();
        sensor->scan(renderer, returns);
        const auto t1 = std::chrono::steady_clock::now();
        lastScanMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

        lastBeams = static_cast<int>(sensor->beamCount());

        lastReturns = 0;
        for (const auto& r : returns) {
            if (r.hitInstanceId >= 0) ++lastReturns;
        }

        updateLidarVisualization(*cloud, returns, colorGain);

        // Update the sensor readout panel. Lay out beams as
        // (azimuth column × elevation row); intensity → HSL hue. Misses
        // stay black (alpha = 0). Texture is sized for any sensor — we
        // scale the live beam grid into kPanelW × kPanelH and mark the
        // texture dirty so the renderer re-uploads it before the sprite
        // draw later this frame.
        {
            const int numAz   = std::max(1, static_cast<int>(std::round(
                                                    (model.azimuthMax - model.azimuthMin) /
                                                    model.azimuthResolution)));
            const int numElev = static_cast<int>(model.elevationAngles.size());

            auto& panelBytes = panelTex->image().data<unsigned char>();
            std::fill(panelBytes.begin(), panelBytes.end(), 0u);

            // Cell size in texture pixels: stretch each beam across its
            // proportional rectangle so no black gaps remain between rows
            // when the sensor has fewer elevations than the texture's
            // pixel height (e.g. VLP-16 = 16 rows into 128). The sprite
            // path samples with VK_FILTER_LINEAR, so gaps would otherwise
            // get blended into adjacent bright cells and wash the colours
            // out on screen.
            const int blockW = std::max(1, static_cast<int>(kPanelW) / numAz);
            const int blockH = std::max(1, static_cast<int>(kPanelH) / numElev);

            Color c;
            for (size_t b = 0; b < returns.size(); ++b) {
                const auto& r = returns[b];
                if (r.hitInstanceId < 0) continue;

                const int ai = static_cast<int>(b) / numElev;
                const int ei = static_cast<int>(b) % numElev;

                const int px0 = std::clamp(
                        ai * static_cast<int>(kPanelW) / numAz,
                        0, static_cast<int>(kPanelW) - 1);
                // Texture Y grows downward in image space; combined with the
                // sprite UV convention (V flipped), writing low elevations
                // to the top rows lands "up" at the top of the on-screen
                // panel. Sensor elevations are stored in ascending order in
                // LidarModel, so direct ei→py keeps positive elevation up.
                const int py0 = std::clamp(
                        ei * static_cast<int>(kPanelH) / numElev,
                        0, static_cast<int>(kPanelH) - 1);

                const float t = std::clamp(r.intensity * colorGain, 0.f, 1.f);
                c.setHSL((1.f - t) * 0.66f, 1.f, 0.5f);

                const unsigned char rByte = static_cast<unsigned char>(c.r * 255.f);
                const unsigned char gByte = static_cast<unsigned char>(c.g * 255.f);
                const unsigned char bByte = static_cast<unsigned char>(c.b * 255.f);
                for (int dy = 0; dy < blockH; ++dy) {
                    const int y = std::min(py0 + dy, static_cast<int>(kPanelH) - 1);
                    size_t row = static_cast<size_t>(y) * kPanelW * 4;
                    for (int dx = 0; dx < blockW; ++dx) {
                        const int x = std::min(px0 + dx, static_cast<int>(kPanelW) - 1);
                        const size_t idx = row + static_cast<size_t>(x) * 4;
                        panelBytes[idx + 0] = rByte;
                        panelBytes[idx + 1] = gByte;
                        panelBytes[idx + 2] = bByte;
                        panelBytes[idx + 3] = 255u;
                    }
                }
            }
            panelTex->needsUpdate();
        }

        ui.render();
    });
}
