// SensorBackendParity_test — a vision sensor's near/far must mean the same
// thing on GL as on Vulkan.
//
// Both backends bound what a ranging sensor reports, but they used to bound
// DIFFERENT quantities. The raster path clipped on the face camera's near/far
// PLANES, i.e. on perpendicular view-space Z; the ray-traced path bounded the
// ray's [tMin, tMax], i.e. the Euclidean RANGE. Those differ by the pixel's
// slant ratio sqrt(1 + u^2 + v^2) — 1 dead ahead, up to sqrt(3) at the corner of
// a 90-degree cube face — so the raster near plane was really an L-infinity
// blind CUBE while the tracer's was a blind SPHERE. An off-axis surface could
// sit outside the sphere and inside the cube, and the two backends then
// disagreed about whether the sensor could see it: a drone's rotor 1.35 m from
// a near=1.2 m LIDAR was invisible on GL and a self-return on Vulkan.
//
// The library now uses the sphere on both — near/far bound the RANGE, which is
// what LidarReturn::distance is measured in and what a scanner's datasheet
// quotes — and the raster near plane is pulled in by the worst-case slant ratio
// so it cannot clip anything the range test would keep. Both bounds are
// INCLUSIVE, matching traceRayEXT's [tMin, tMax].
//
// This lives in its own executable for the same reason CanvasApiOrder_test does:
// it needs BOTH backends in one process, and the Vulkan canvas has to come first
// (GLFW window hints are sticky). Plain exit-code program, not Catch2. Exits 42
// (→ CTest "Skipped") when no Vulkan-capable GPU is available.
//
// RT is never bit-exact and the two backends sample different beam geometry (the
// raster path quantises each beam to a cube-face pixel), so every cross-backend
// claim here is a tolerance, never a hash.

#include "threepp/threepp.hpp"

#include "threepp/helpers/DepthSensor.hpp"
#include "threepp/helpers/LidarSensor.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kSkipCode = 42;

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  %s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    // The sensor shell under test. near is deliberately larger than the
    // "housing" below and smaller than the "rotor", so the two objects sit on
    // opposite sides of the blind sphere.
    constexpr float kNear = 1.2f;
    constexpr float kFar = 6.f;

    // Sensor-local positions of the two probe objects. Both are inside the old
    // raster blind CUBE (max|coord| < kNear) — that is what makes the rotor the
    // regression case, because its RANGE is outside the blind sphere.
    //
    //   housing: range = sqrt(0.5^2 + 0.5^2) = 0.707  -> inside the sphere, never reported
    //   rotor:   range = sqrt(1.0^2 + 1.0^2) = 1.414  -> outside it, always reported
    //
    // Their L-infinity radii are 0.5 and 1.0, so a near PLANE at 1.2 clipped
    // both and a near SPHERE at 1.2 clips only the housing.
    const Vector3 kHousing{0.5f, 0.f, -0.5f};
    const Vector3 kRotor{1.0f, 0.f, -1.0f};
    constexpr float kProbeHalf = 0.12f;

    std::shared_ptr<Mesh> box(float sx, float sy, float sz, const Vector3& at) {
        auto material = MeshStandardMaterial::create();
        material->color = Color(0xffffff);
        material->roughness = 0.9f;
        material->metalness = 0.f;
        auto mesh = Mesh::create(BoxGeometry::create(sx, sy, sz), material);
        mesh->position.copy(at);
        return mesh;
    }

    // A four-walled room around the sensor plus the two probe objects. The room
    // gives every beam something to hit; its corners are past kFar, so the far
    // bound is exercised as well as the near one.
    std::shared_ptr<Scene> makeScene() {
        auto scene = Scene::create();
        scene->add(HemisphereLight::create(0xffffff, 0x404040, 1.f));
        auto key = DirectionalLight::create(0xffffff, 2.f);
        key->position.set(3, 5, 2);
        scene->add(key);

        constexpr float kRoom = 5.f;
        scene->add(box(24, 24, 0.4f, {0, 0, -kRoom}));
        scene->add(box(24, 24, 0.4f, {0, 0, kRoom}));
        scene->add(box(0.4f, 24, 24, {-kRoom, 0, 0}));
        scene->add(box(0.4f, 24, 24, {kRoom, 0, 0}));

        const float d = kProbeHalf * 2.f;
        scene->add(box(d, d, d, kHousing));
        scene->add(box(d, d, d, kRotor));
        return scene;
    }

    // What a scan is judged on. Ranges are measured from the sensor origin,
    // which sits at the world origin in this scene, so |position| IS the range.
    struct Stats {
        std::size_t count = 0;
        float minRange = 0.f;
        float maxRange = 0.f;
        std::size_t onHousing = 0;
        std::size_t onRotor = 0;
        float rotorMeanRange = 0.f;
    };

    bool near(const Vector3& p, const Vector3& centre) {
        // Generous enough to catch a hit anywhere on the probe box, tight enough
        // that no wall return can be mistaken for one.
        const float slack = kProbeHalf + 0.1f;
        return std::abs(p.x - centre.x) < slack &&
               std::abs(p.y - centre.y) < slack &&
               std::abs(p.z - centre.z) < slack;
    }

    Stats summarise(const std::vector<Vector3>& points) {
        Stats s;
        double rotorSum = 0;
        for (const auto& p : points) {
            const float r = p.length();
            if (s.count == 0 || r < s.minRange) s.minRange = r;
            if (s.count == 0 || r > s.maxRange) s.maxRange = r;
            ++s.count;
            if (near(p, kHousing)) ++s.onHousing;
            if (near(p, kRotor)) {
                ++s.onRotor;
                rotorSum += r;
            }
        }
        if (s.onRotor) s.rotorMeanRange = static_cast<float>(rotorSum / static_cast<double>(s.onRotor));
        return s;
    }

    void report(const char* sensor, const char* backend, const Stats& s) {
        std::printf("  [ .. ] %-6s %-6s: %6zu returns, range [%.3f, %.3f], housing %zu, rotor %zu @ %.3f m\n",
                    sensor, backend, s.count, s.minRange, s.maxRange, s.onHousing, s.onRotor, s.rotorMeanRange);
    }

    // Everything that must hold on a SINGLE backend, whichever it is.
    void checkShell(const char* sensor, const char* backend, const Stats& s) {
        const std::string tag = std::string(sensor) + " " + backend;
        check(s.count > 200, tag + ": the scan returned something to judge");
        // Inclusive bounds, so a hit exactly at near/far is legal; the epsilon
        // covers only float round-trips through the depth pack / the tracer.
        check(s.minRange >= kNear - 1e-3f, tag + ": nothing is reported inside the blind sphere");
        check(s.maxRange <= kFar + 1e-3f, tag + ": nothing is reported past the max range");
        check(s.onHousing == 0, tag + ": the housing (range 0.71 < near) produces no return");
        // The regression: this object is inside the old raster blind CUBE but
        // outside the blind SPHERE, so it must be SEEN, on both backends.
        check(s.onRotor > 0, tag + ": the rotor (range 1.41 > near) is seen, not blinded");
    }

    // ...and what must hold BETWEEN the two.
    void checkParity(const char* sensor, const Stats& gl, const Stats& vk) {
        const std::string tag = std::string(sensor) + " GL/VK";

        // Counts: the raster path quantises each beam to a cube-face pixel and
        // the tracer traces the exact direction, so silhouettes disagree by a
        // few beams. A shell that disagreed would be off by thousands.
        const auto lo = std::min(gl.count, vk.count);
        const auto hi = std::max(gl.count, vk.count);
        const double drift = hi == 0 ? 1.0 : static_cast<double>(hi - lo) / static_cast<double>(hi);
        check(drift < 0.10, tag + ": total return counts agree within 10%");

        check(gl.onRotor > 0 && vk.onRotor > 0, tag + ": both backends see the rotor");
        const auto rlo = std::min(gl.onRotor, vk.onRotor);
        const auto rhi = std::max(gl.onRotor, vk.onRotor);
        const double rdrift = rhi == 0 ? 1.0 : static_cast<double>(rhi - rlo) / static_cast<double>(rhi);
        check(rdrift < 0.25, tag + ": rotor return counts agree within 25%");

        // The ranging claim: the same surface must be reported at the same
        // distance. Noise is off, so the only spread is beam quantisation.
        check(std::abs(gl.rotorMeanRange - vk.rotorMeanRange) < 0.05f,
              tag + ": the rotor is ranged at the same distance (within 5 cm)");
        check(std::abs(gl.minRange - vk.minRange) < 0.05f,
              tag + ": the closest reported return matches");
        check(std::abs(gl.maxRange - vk.maxRange) < 0.05f,
              tag + ": the farthest reported return matches");
    }

    // ── the two sensors, scanned identically on whichever renderer is passed ──

    Stats scanLidar(Renderer& renderer, Scene& scene, Camera& camera) {
        renderer.render(scene, camera);// GL: warm; Vulkan: builds the TLAS the scan traces

        LidarSensor lidar(LidarModel::VLP16(), /*faceSize*/ 256, kNear, kFar);
        lidar.rangeNoise = {};// a ground-truth capture: the shell is the only filter
        lidar.resetNoise();

        std::vector<LidarReturn> returns;
        lidar.scan(renderer, scene, returns);

        std::vector<Vector3> points;
        points.reserve(returns.size());
        for (const auto& r : returns) points.push_back(r.position);
        return summarise(points);
    }

    Stats scanDepth(Renderer& renderer, Scene& scene, Camera& camera) {
        renderer.render(scene, camera);

        // A wide FOV so the probe objects, which sit at 45 degrees off the view
        // axis, are inside the image — that is where the plane/sphere split is
        // widest and where the backends used to part company.
        DepthSensor depth(/*fovY*/ 120.f, 128, 128, kNear, kFar);
        depth.rangeNoise = {};
        depth.resetNoise();

        std::vector<Vector3> cloud;
        depth.scan(renderer, scene, cloud);
        return summarise(cloud);
    }

}// namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    constexpr int kW = 160, kH = 120;

    auto scene = makeScene();
    auto camera = PerspectiveCamera::create(60, static_cast<float>(kW) / kH, 0.1f, 100.f);
    camera->position.set(0, 2, 4);
    camera->lookAt(0, 0, 0);

    // ── Vulkan first: a Vulkan canvas must precede the GL one (CanvasApiOrder) ──
    Stats vkLidar, vkDepth;
    try {
        Canvas vkCanvas(Canvas::Parameters().title("SensorParity-vk").size(kW, kH).vsync(false).headless(true));
        VulkanRenderer vkRenderer(vkCanvas);

        vkLidar = scanLidar(vkRenderer, *scene, *camera);
        vkDepth = scanDepth(vkRenderer, *scene, *camera);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }

    // ── GL second ───────────────────────────────────────────────────────────
    Stats glLidar, glDepth;
    try {
        Canvas glCanvas(Canvas::Parameters().title("SensorParity-gl").size(kW, kH).headless(true));
        GLRenderer glRenderer(glCanvas);
        glRenderer.setClearColor(Color(0x000000));

        glLidar = scanLidar(glRenderer, *scene, *camera);
        glDepth = scanDepth(glRenderer, *scene, *camera);
    } catch (const std::exception& e) {
        check(false, std::string("the GL scans ran (threw: ") + e.what() + ")");
        std::printf("\n%d CHECK(S) FAILED\n", failures);
        return 1;
    }

    report("lidar", "GL", glLidar);
    report("lidar", "VK", vkLidar);
    report("depth", "GL", glDepth);
    report("depth", "VK", vkDepth);

    checkShell("lidar", "GL", glLidar);
    checkShell("lidar", "VK", vkLidar);
    checkParity("lidar", glLidar, vkLidar);

    checkShell("depth", "GL", glDepth);
    checkShell("depth", "VK", vkDepth);
    checkParity("depth", glDepth, vkDepth);

    std::printf(failures == 0 ? "\nOK\n" : "\n%d CHECK(S) FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
