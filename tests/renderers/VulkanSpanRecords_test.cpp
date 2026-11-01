// VulkanSpanRecords_test: a large instanced span's kept draw records stay true
// to the entries through everything that can change a row.
//
// buildIndirectDrawData keeps such a span's DrawInfo records and occlusion meta
// in each frame slot's buffers from frame to frame and writes only the rows
// that changed (SpanRecordPool). THREEPP_VK_SPAN_RECORDS_CHECK reads every kept
// row back against the entries on every frame and aborts on a stale one, so
// this program is the list of those changes, each rendered for enough frames
// that both frame slots catch up, with occlusion culling off and on:
//
//   * an instance moved, then back, and the whole array re-sent with one row
//     changed, as a pool re-sends its slots;
//   * a row faded to no pixels and back (it gets no command, keeps its record);
//   * the moved-sticky window running out, which changes a flag word only;
//   * a row moved while nothing of the span is in view: its records wait, and
//     the first drawn frame must bring them up to date;
//   * the mesh's own transform moved: every row;
//   * setCount down and up;
//   * a mesh added ahead of the span, which renumbers its entries;
//   * the span hidden and shown again.
//
// One big row in the middle of the view shows that the commands really do
// reach the records: the pixel under it follows the edits.
//
// Plain exit-code program, not Catch2, like the other Vulkan tests here.
// Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"
#include "vulkan_test_util.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>

using namespace threepp;
using vktest::kSkipCode;

namespace {

    constexpr int kW = 160, kH = 120;
    constexpr size_t kRows = 1024;
    constexpr size_t kBigRow = 7;// the row in the middle of the view
    constexpr int kSlots = 3;    // frames for every frame slot to have seen an edit
    constexpr int kSticky = 30;  // kMovedStickyFrames (VulkanCoreFrame.cpp)

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  %s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    void setCheckMode() {
#ifdef _WIN32
        _putenv_s("THREEPP_VK_SPAN_RECORDS_CHECK", "1");
#else
        setenv("THREEPP_VK_SPAN_RECORDS_CHECK", "1", 1);
#endif
    }

    // A wide field of small boxes low in the view, most of them out of it, so
    // the span straddles the frustum and draws in runs.
    Matrix4 smallRow(size_t i) {
        Matrix4 m;
        m.makeTranslation(static_cast<float>(i % 64) * 1.5f - 48.f, -3.f, -static_cast<float>(i / 64) * 1.5f);
        return m;
    }

    Matrix4 bigRow(float x) {
        Matrix4 m;
        m.makeScale(2.5f, 2.5f, 2.5f);
        m.setPosition(x, 0.f, 0.f);
        return m;
    }

}// namespace

int main() {

    std::setvbuf(stdout, nullptr, _IONBF, 0);
    setCheckMode();

    auto gpu = vktest::openHeadlessVulkan("VulkanSpanRecords_test", kW, kH);
    if (!gpu) return kSkipCode;
    Canvas& canvas = *gpu.canvas;
    VulkanRenderer& renderer = *gpu.renderer;

    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setRenderScale(1.0f);
    renderer.setAutoExposure(false);
    renderer.setBloomIntensity(0.f);
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;

    Scene scene;
    auto sun = DirectionalLight::create(0xffffff, 3.f);
    sun->position.set(1.f, 2.f, 6.f);
    scene.add(sun);

    // Ahead of the field in the traversal: what lands here renumbers its entries.
    auto ahead = Group::create();
    scene.add(ahead);

    auto material = MeshStandardMaterial::create();
    material->color = Color(0.9f, 0.9f, 0.9f);
    auto field = InstancedMesh::create(BoxGeometry::create(1.f, 1.f, 1.f), material, kRows);
    for (size_t i = 0; i < kRows; ++i) field->setMatrixAt(i, i == kBigRow ? bigRow(0.f) : smallRow(i));
    field->instanceMatrix()->needsUpdate();
    scene.add(field);

    PerspectiveCamera camera(45.f, static_cast<float>(kW) / kH, 0.1f, 300.f);
    camera.position.set(0, 0, 12.f);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    const auto frames = [&](int n) {
        for (int i = 0; i < n; ++i) canvas.animateOnce([&] { renderer.render(scene, camera); });
    };
    // The big row's pixel, on the sun-facing side of the box.
    const auto middleLit = [&] {
        const auto im = renderer.readRGBPixels();
        if (im.size() < static_cast<size_t>(kW) * kH * 3) {
            std::printf("readback too small: %zu\n", im.size());
            std::exit(1);
        }
        const size_t at = (static_cast<size_t>(kH / 2) * kW + kW / 2) * 3;
        return int(im[at]) + im[at + 1] + im[at + 2] > 60;
    };
    const auto moveBig = [&](float x) {
        field->setMatrixAt(kBigRow, bigRow(x));
        field->instanceMatrix()->needsUpdate();
    };

    int added = 0;
    try {
        for (const bool occlusion : {false, true}) {
            std::printf("occlusion culling %s:\n", occlusion ? "on" : "off");
            renderer.setOcclusionCulling(occlusion);
            frames(kSlots);
            check(middleLit(), "the big row is drawn");

            moveBig(60.f);
            frames(kSlots);
            check(!middleLit(), "moved away: gone from the middle");
            moveBig(0.f);
            frames(kSlots);
            check(middleLit(), "moved back");

            field->setMatrixAt(100, smallRow(101));
            field->instanceMatrix()->needsUpdate();
            frames(kSlots);
            field->setMatrixAt(100, smallRow(100));
            field->instanceMatrix()->needsUpdate();
            frames(kSlots);
            frames(kSticky);// the window runs out: flag words alone change

            field->setFadeAt(kBigRow, 0.f, false);
            frames(kSlots);
            check(!middleLit(), "faded to no pixels: not drawn");
            field->setFadeAt(kBigRow, 1.f, false);
            frames(kSlots);
            check(middleLit(), "faded back");

            // Nothing of the span in view: no command of its, and no write.
            camera.lookAt(Vector3{0, 0, 40.f});
            camera.updateMatrixWorld();
            frames(kSlots);
            moveBig(60.f);
            frames(kSlots);
            camera.lookAt(Vector3{0, 0, 0});
            camera.updateMatrixWorld();
            frames(kSlots);
            check(!middleLit(), "moved out of view: the first drawn frame has it");
            moveBig(0.f);
            frames(kSlots);
            check(middleLit(), "and back");

            field->position.x = 0.5f;
            frames(kSlots);
            field->position.x = 0.f;
            frames(kSlots);
            check(middleLit(), "the mesh moved and back");

            field->setCount(kRows / 2);
            frames(kSlots);
            field->setCount(kRows);
            frames(kSlots);
            check(middleLit(), "count down and up");

            auto plain = Mesh::create(BoxGeometry::create(0.2f, 0.2f, 0.2f), material);
            plain->position.set(-40.f, 0.f, 0.f);
            ahead->add(plain);
            ++added;
            frames(kSlots);
            check(middleLit(), "entries renumbered by a mesh ahead");

            field->visible = false;
            frames(kSlots);
            check(!middleLit(), "hidden");
            field->visible = true;
            frames(kSlots);
            check(middleLit(), "shown");
        }
    } catch (const std::exception& e) {
        std::printf("exception: %s\n", e.what());
        return 1;
    }

    std::printf("%s: %d failure(s), %d mesh(es) added\n", failures == 0 ? "PASS" : "FAIL", failures, added);
    return failures == 0 ? 0 : 1;
}
