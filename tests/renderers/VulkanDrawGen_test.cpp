// VulkanDrawGen_test: the G-buffer draw records draw_gen.comp writes for
// instanced meshes are, byte for byte, the records the CPU writes.
//
// Stage 2 of plans/gpu-driven-instances.md. With setGpuDrawGeneration the
// primary view's records of eligible InstancedMeshes (static, no auto-LOD
// chain, no LOD fades) are written by compute from the expansion's world
// matrices, the frustum cull included (vertex count 0 for an instance outside
// the view). setGpuDrawGenerationCheck keeps the CPU building them too, and
// drawGenCheck compares: every visible instance's DrawInfo, every instance's
// command, and the cull, over the frame last rendered.
//
// The scene puts instanced meshes in three cull-mode buckets (front faces,
// double-sided, back faces) between plain meshes, so the GPU ranges sit
// between CPU records in every bucket; one mesh's instances move every frame
// (instanceMatrix()->needsUpdate()), one grows and shrinks through setCount (a
// slot edit), and the camera orbits through the forest so the cull changes
// every frame and instances leave and enter the view.
//
// Plain exit-code program, not Catch2, like the other Vulkan tests here.
// Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/ConeGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"
#include "threepp/scenes/Scene.hpp"
#include "vulkan_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>

using namespace threepp;
using vktest::kSkipCode;

namespace {

    constexpr int kW = 480, kH = 270;
    constexpr int kFrames = 48;

    std::shared_ptr<MeshStandardMaterial> paint(const Color& c, Side side = Side::Front) {
        auto m = MeshStandardMaterial::create();
        m->color = c;
        m->roughness = 0.7f;
        m->side = side;
        return m;
    }

    float hash(uint32_t i) {
        i ^= i >> 16;
        i *= 0x7feb352du;
        i ^= i >> 15;
        i *= 0x846ca68bu;
        i ^= i >> 16;
        return static_cast<float>(i & 0xffffffu) / 8388608.f - 1.f;
    }

    std::shared_ptr<InstancedMesh> field(std::shared_ptr<BufferGeometry> geom, std::shared_ptr<Material> mat,
                                         size_t capacity, size_t count, float y, uint32_t seed) {
        auto im = InstancedMesh::create(std::move(geom), std::move(mat), capacity);
        Matrix4 m;
        for (size_t i = 0; i < capacity; ++i) {
            m.makeTranslation(hash(seed + uint32_t(i) * 3u) * 40.f, y, hash(seed + uint32_t(i) * 3u + 1u) * 40.f);
            im->setMatrixAt(i, m);
        }
        im->instanceMatrix()->needsUpdate();
        im->setCount(count);
        return im;
    }

}// namespace

int main() {

    std::setvbuf(stdout, nullptr, _IONBF, 0);

    auto gpu = vktest::openHeadlessVulkan("VulkanDrawGen_test", kW, kH);
    if (!gpu) return kSkipCode;
    Canvas& canvas = *gpu.canvas;
    VulkanRenderer& renderer = *gpu.renderer;
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setAutoLod(false);// a LOD chain takes a span back to the CPU path

    Scene scene;
    scene.background = Color(0.4f, 0.5f, 0.6f);
    auto ground = Mesh::create(PlaneGeometry::create(120.f, 120.f), paint(Color(0.5f, 0.45f, 0.4f)));
    ground->rotation.x = -math::PI / 2;
    scene.add(ground);
    auto sun = DirectionalLight::create(0xffffff, 2.f);
    sun->position.set(20.f, 40.f, 10.f);
    scene.add(sun);

    // Bucket 0 (back-face culled), the forest, growing and shrinking.
    auto forest = field(ConeGeometry::create(0.6f, 2.5f, 8), paint(Color(0.1f, 0.5f, 0.15f)), 4000, 2500, 1.25f, 11u);
    scene.add(forest);
    // A plain mesh between the instanced ones (a CPU record in the same bucket).
    auto rock = Mesh::create(SphereGeometry::create(1.5f, 16, 8), paint(Color(0.5f, 0.5f, 0.5f)));
    rock->position.set(3.f, 1.5f, -4.f);
    scene.add(rock);
    // Bucket 2 (double-sided), moving every frame.
    auto movers = field(BoxGeometry::create(0.8f, 0.8f, 0.8f), paint(Color(0.8f, 0.2f, 0.1f), Side::Double), 600,
                        600, 3.f, 977u);
    scene.add(movers);
    // Bucket 1 (back faces drawn).
    auto shells = field(SphereGeometry::create(0.7f, 12, 6), paint(Color(0.2f, 0.3f, 0.9f), Side::Back), 400, 400,
                        0.7f, 4441u);
    scene.add(shells);
    auto crate = Mesh::create(BoxGeometry::create(2.f, 2.f, 2.f), paint(Color(0.7f, 0.6f, 0.3f), Side::Double));
    crate->position.set(-5.f, 1.f, 2.f);
    scene.add(crate);

    PerspectiveCamera camera(55.f, static_cast<float>(kW) / kH, 0.3f, 200.f);

    renderer.setGpuInstanceExpansion(true);
    renderer.setGpuDrawGeneration(true);
    renderer.setGpuDrawGenerationCheck(true);

    int failures = 0;
    const auto check = [&](const std::string& what, bool ok) {
        std::printf("  %-70s %s\n", what.c_str(), ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    };

    try {
        std::size_t checked = 0, records = 0, spansMax = 0, drawMis = 0, cmdMis = 0, cullMis = 0, culledSeen = 0;
        Matrix4 m;
        for (int f = 0; f < kFrames; ++f) {
            const float a = 0.13f * static_cast<float>(f);
            camera.position.set(std::sin(a) * 18.f, 6.f + 2.f * std::sin(a * 0.7f), std::cos(a) * 18.f);
            camera.lookAt(Vector3{0, 1.f, 0});
            camera.updateMatrixWorld();
            // The movers drift every frame.
            for (size_t i = 0; i < movers->count(); ++i) {
                m.makeTranslation(hash(977u + uint32_t(i) * 3u) * 40.f + std::sin(a + float(i)) * 0.5f, 3.f,
                                  hash(977u + uint32_t(i) * 3u + 1u) * 40.f);
                movers->setMatrixAt(i, m);
            }
            movers->instanceMatrix()->needsUpdate();
            // The forest grows and shrinks (slot edits).
            if (f % 12 == 4) forest->setCount(forest->count() + 400);
            if (f % 12 == 10) forest->setCount(forest->count() - 300);

            canvas.animateOnce([&] { renderer.render(scene, camera); });
            if (f < 3) continue;// let the first builds land
            VulkanRenderer::DrawGenCheck dc;
            if (!renderer.drawGenCheck(dc)) continue;
            ++checked;
            records += dc.records;
            spansMax = std::max(spansMax, dc.spans);
            drawMis += dc.drawInfoMismatches;
            cmdMis += dc.cmdMismatches;
            cullMis += dc.cullMismatches;
            if (dc.drawInfoMismatches || dc.cmdMismatches || dc.cullMismatches) {
                std::printf("  frame %d: %zu spans, %zu records, DrawInfo %zu, cmd %zu, cull %zu mismatched\n", f,
                            dc.spans, dc.records, dc.drawInfoMismatches, dc.cmdMismatches, dc.cullMismatches);
            }
        }
        std::printf("checked %zu frames: %zu records, up to %zu GPU spans; mismatches DrawInfo %zu, cmd %zu, cull %zu\n",
                    checked, records, spansMax, drawMis, cmdMis, cullMis);
        (void) culledSeen;
        check("the GPU wrote the records of the three instanced meshes", spansMax == 3u);
        check("frames compared", checked >= static_cast<std::size_t>(kFrames - 8));
        check("every visible instance's DrawInfo is the CPU's, byte for byte", drawMis == 0u);
        check("every command is the CPU's (vertex range, instance, record)", cmdMis == 0u);
        check("the GPU culls exactly the instances the CPU culls", cullMis == 0u);
        const auto stats = renderer.tlasStats();
        std::printf("structural rebuilds %llu, slot edits %llu\n",
                    static_cast<unsigned long long>(stats.structuralRebuilds),
                    static_cast<unsigned long long>(stats.slotEdits));
        if (vulkan::validationActive()) {
            check("no validation errors", vulkan::validationErrorCount() == 0u);
        }
    } catch (const std::exception& e) {
        std::printf("  FAIL: the renderer threw: %s\n", e.what());
        ++failures;
    }

    std::printf("draw gen: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
