// WgpuRenderer performance baseline test.
// Renders a WebTide-inspired scene (49 lit planes + ground + skybox) to a
// 256x256 render target for 100 timed frames and writes results to
// build/tests/wgpu_perf_baseline.txt.

#include "CrossRenderer_helpers.hpp"

#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/materials/MeshLambertMaterial.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>

TEST_CASE("WgpuRenderer performance baseline", "[wgpu][perf]") {
    REQUIRE_WGPU();
    SKIP_ON_SOFTWARE_ADAPTER();

    constexpr int PERF_RT_WIDTH = 256;
    constexpr int PERF_RT_HEIGHT = 256;
    constexpr int WARMUP_FRAMES = 10;
    constexpr int TIMED_FRAMES = 100;

    // --- Build scene ---
    auto scene = Scene::create();

    // Ambient + directional light
    auto ambient = AmbientLight::create(0x404040);
    scene->add(ambient);

    auto dirLight = DirectionalLight::create(0xffffff, 0.8f);
    dirLight->position.set(-1, 1, -3);
    scene->add(dirLight);

    // 7x7 grid of lit planes (49 meshes)
    for (int z = -3; z <= 3; z++) {
        for (int x = -3; x <= 3; x++) {
            auto geo = PlaneGeometry::create(20, 20, 64, 64);
            auto mat = MeshPhongMaterial::create();
            // Per-tile color variation
            float r = 0.2f + 0.1f * static_cast<float>(x + 3);
            float g = 0.3f + 0.05f * static_cast<float>(z + 3);
            float b = 0.5f;
            mat->color = Color(r, g, b);
            auto mesh = Mesh::create(geo, mat);
            mesh->rotation.x = -math::PI / 2.0f;
            mesh->position.set(static_cast<float>(x) * 20.0f, 0, static_cast<float>(z) * 20.0f);
            scene->add(mesh);
        }
    }

    // Ground plane
    {
        auto geo = PlaneGeometry::create(200, 200);
        auto mat = MeshLambertMaterial::create();
        mat->color = Color(0.4f, 0.4f, 0.35f);
        auto ground = Mesh::create(geo, mat);
        ground->rotation.x = -math::PI / 2.0f;
        ground->position.y = -0.1f;
        scene->add(ground);
    }

    // Skybox
    {
        auto geo = BoxGeometry::create(500, 500, 500);
        auto mat = MeshBasicMaterial::create();
        mat->color = Color(0.5f, 0.7f, 0.9f);
        mat->side = Side::Back;
        auto skybox = Mesh::create(geo, mat);
        scene->add(skybox);
    }

    // Camera
    auto camera = PerspectiveCamera::create(60, 1.0f, 0.1f, 1000);
    camera->position.set(0, 5, 15);
    camera->lookAt({0, 0, 0});

    // --- Create Wgpu renderer and render target ---
    Canvas perfCanvas(Canvas::Parameters()
            .size(PERF_RT_WIDTH, PERF_RT_HEIGHT)
            .headless(true));


    WgpuRenderer renderer(perfCanvas);
    renderer.setClearColor(Color(0.1f, 0.1f, 0.15f));

    auto target = RenderTarget::create(PERF_RT_WIDTH, PERF_RT_HEIGHT, RenderTarget::Options{});
    renderer.setRenderTarget(target.get());

    // --- Warm-up ---
    for (int i = 0; i < WARMUP_FRAMES; i++) {
        renderer.render(*scene, *camera);
    }

    // --- Timed frames ---
    std::vector<double> frameTimes(TIMED_FRAMES);
    std::vector<size_t> drawCalls(TIMED_FRAMES);
    std::vector<size_t> triangles(TIMED_FRAMES);

    for (int i = 0; i < TIMED_FRAMES; i++) {
        auto start = std::chrono::steady_clock::now();
        renderer.render(*scene, *camera);
        auto end = std::chrono::steady_clock::now();

        frameTimes[i] = std::chrono::duration<double, std::milli>(end - start).count();

        auto& info = renderer.info();
        drawCalls[i] = info.render.calls;
        triangles[i] = info.render.triangles;
    }

    // --- Compute statistics ---
    double totalMs = std::accumulate(frameTimes.begin(), frameTimes.end(), 0.0);
    double avgFrameMs = totalMs / TIMED_FRAMES;
    double minFrameMs = *std::min_element(frameTimes.begin(), frameTimes.end());
    double maxFrameMs = *std::max_element(frameTimes.begin(), frameTimes.end());
    double avgDrawCalls = static_cast<double>(std::accumulate(drawCalls.begin(), drawCalls.end(), size_t(0))) / TIMED_FRAMES;
    double avgTriangles = static_cast<double>(std::accumulate(triangles.begin(), triangles.end(), size_t(0))) / TIMED_FRAMES;

    // --- Write results ---
    {
        std::ofstream out("wgpu_perf_baseline.txt");
        out << "scene_objects=51\n";
        out << "warmup_frames=" << WARMUP_FRAMES << "\n";
        out << "timed_frames=" << TIMED_FRAMES << "\n";
        out << "rt_width=" << PERF_RT_WIDTH << "\n";
        out << "rt_height=" << PERF_RT_HEIGHT << "\n";
        out << "total_ms=" << totalMs << "\n";
        out << "avg_frame_ms=" << avgFrameMs << "\n";
        out << "min_frame_ms=" << minFrameMs << "\n";
        out << "max_frame_ms=" << maxFrameMs << "\n";
        out << "avg_draw_calls=" << avgDrawCalls << "\n";
        out << "avg_triangles=" << avgTriangles << "\n";
    }

    // --- Print summary to stdout ---
    std::cout << "\n=== Wgpu Perf Baseline ===\n";
    std::cout << "  avg_frame_ms: " << avgFrameMs << "\n";
    std::cout << "  min_frame_ms: " << minFrameMs << "\n";
    std::cout << "  max_frame_ms: " << maxFrameMs << "\n";
    std::cout << "  avg_draw_calls: " << avgDrawCalls << "\n";
    std::cout << "  avg_triangles: " << avgTriangles << "\n";
    std::cout << "  Results written to wgpu_perf_baseline.txt\n";

    // --- Sanity checks ---
    CHECK(avgFrameMs > 0);
    CHECK(avgDrawCalls >= 10);  // frustum culling reduces visible objects
    CHECK(avgTriangles > 1000);

    renderer.setRenderTarget(nullptr);
    renderer.dispose();
}
