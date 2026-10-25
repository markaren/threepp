// VulkanMovedInstances_test — an instance-matrix edit makes only the instances
// whose matrix changed "moving", not every instance of the InstancedMesh.
//
// Moving (the moved-sticky state, kGeomFlagMoved: moved, shown or hidden in
// the last 30 frames) keeps an entry's shadow, GI and reflection histories
// short and redraws it into the cached shadow maps every frame. An
// InstancedMesh used as a pool of slots (the pine forest's level meshes,
// whose matrices are re-sent whole whenever one slot changes owner) used to
// make all of its instances moving on every such edit: a third of a 113k-entry
// forest was moving on every frame of a camera orbit.
//
// Read through TlasStats::movingEntries:
//   * a settled scene: nothing moving;
//   * one instance moved (setMatrixAt + needsUpdate): one;
//   * the whole array re-sent with one more instance changed: two;
//   * the array re-sent unchanged: no more;
//   * the window running out: back to none;
//   * the mesh's own transform moved: every instance;
//   * the plain mesh beside it moved: one.
//
// Run standalone (plain exit-code program) or via CTest. Exits 42 (CTest
// "Skipped") when no Vulkan GPU is available.

#include "threepp/threepp.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kW = 160, kH = 120;
    constexpr int kSkipCode = 42;
    constexpr int kSticky = 30;// kMovedStickyFrames (VulkanCoreFrame.cpp)

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  %s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    void expectMoving(VulkanRenderer& r, uint32_t want, const std::string& what) {
        const uint32_t got = r.tlasStats().movingEntries;
        check(got == want, what + ": " + std::to_string(got) + " moving, want " + std::to_string(want));
    }

    Matrix4 slotMatrix(size_t i, float lift = 0.f) {
        Matrix4 m;
        m.makeTranslation(static_cast<float>(i % 32) * 0.5f - 8.f, lift, static_cast<float>(i / 32) * 0.5f - 8.f);
        return m;
    }

}// namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanMovedInstances_test").size(kW, kH).vsync(false).headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    VulkanRenderer& renderer = *rendererPtr;

    Scene scene;
    auto light = DirectionalLight::create(0xffffff, 2.f);
    light->position.set(4, 8, 6);
    scene.add(light);

    PerspectiveCamera camera(60.f, static_cast<float>(kW) / static_cast<float>(kH), 0.1f, 200.f);
    camera.position.set(0, 10, 18);
    camera.lookAt({0, 0, 0});

    constexpr size_t kN = 1024;
    auto field = InstancedMesh::create(BoxGeometry::create(0.2f, 0.2f, 0.2f), MeshStandardMaterial::create(), kN);
    std::vector<float> mats(kN * 16);
    for (size_t i = 0; i < kN; ++i) {
        const auto m = slotMatrix(i);
        std::copy(m.elements.begin(), m.elements.end(), mats.begin() + i * 16);
        field->setMatrixAt(i, m);
    }
    field->instanceMatrix()->needsUpdate();
    scene.add(field);

    auto plain = Mesh::create(SphereGeometry::create(0.5f), MeshStandardMaterial::create());
    plain->position.set(0, 2, 0);
    scene.add(plain);

    const auto settle = [&](int frames) {
        for (int k = 0; k < frames; ++k) renderer.render(scene, camera);
    };
    // The whole array written over and marked, as a pool re-sends its slots.
    const auto resend = [&] {
        std::copy(mats.begin(), mats.end(), field->instanceMatrix()->array().begin());
        field->instanceMatrix()->needsUpdate();
    };

    std::printf("settled:\n");
    settle(kSticky + 2);
    expectMoving(renderer, 0u, "settled scene");

    std::printf("one instance moved (setMatrixAt):\n");
    field->setMatrixAt(7, slotMatrix(7, 1.f));
    field->instanceMatrix()->needsUpdate();
    std::copy(field->instanceMatrix()->array().begin() + 7 * 16, field->instanceMatrix()->array().begin() + 8 * 16,
              mats.begin() + 7 * 16);
    renderer.render(scene, camera);
    expectMoving(renderer, 1u, "one of 1024 instances moved");

    std::printf("the whole array re-sent, one more instance changed:\n");
    {
        const auto m = slotMatrix(900, 2.f);
        std::copy(m.elements.begin(), m.elements.end(), mats.begin() + 900 * 16);
    }
    resend();
    renderer.render(scene, camera);
    expectMoving(renderer, 2u, "re-sent with one more changed: the two changed ones");

    std::printf("the array re-sent unchanged:\n");
    resend();
    renderer.render(scene, camera);
    expectMoving(renderer, 2u, "re-sent unchanged: no more");

    std::printf("the window runs out:\n");
    settle(kSticky + 1);
    expectMoving(renderer, 0u, "settled again");

    std::printf("the mesh's own transform moved:\n");
    field->position.x += 0.5f;
    renderer.render(scene, camera);
    expectMoving(renderer, static_cast<uint32_t>(kN), "the mesh moved: every instance");
    settle(kSticky + 1);
    expectMoving(renderer, 0u, "settled again");

    std::printf("the plain mesh moved:\n");
    plain->position.y += 0.5f;
    renderer.render(scene, camera);
    expectMoving(renderer, 1u, "the plain mesh");

    std::printf("%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
