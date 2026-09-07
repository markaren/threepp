// VulkanInstanceExpand_test — the GPU per-instance world matrices must equal
// the CPU's, bit for bit.
//
// Stage 1 of plans/gpu-driven-instances.md moves the expansion
// world = mesh.matrixWorld * instanceMatrix[i] onto the GPU
// (instance_expand.comp) while switching NO consumer: the draw list, motion
// vectors, frustum cull and ray-tracing instance descriptors all still read
// MeshEntry::worldMatrix. That is only worth doing if the GPU producer is
// provably interchangeable with the CPU one, and "provably" here means
// BITWISE — both sides sum the same four products in the same order (see the
// note on multiplyMatricesSse2 in math/Matrix4.cpp and the `precise` block in
// instance_expand.comp), so 0 mismatches is the achievable answer and an
// epsilon would only hide an ordering bug. This test is what lets stages 2-5
// swap a consumer over and blame the consumer when the pixels move.
//
// The scenes are chosen to make ordering mistakes visible rather than lucky:
//   * the InstancedMesh sits under a rotated + translated + non-uniformly
//     scaled Group, so spanWorld is a full affine matrix — an identity
//     spanWorld multiplies correctly in either operand order.
//   * two instanced spans with different parents, so the shader's span search
//     and the host's prefix sums are exercised, not just span 0.
//   * a plain Mesh in the same scene, which must NOT appear on the GPU path.
//   * mutation cases: instance matrices only, parent transform only, both, a
//     static frame, and structural add/remove — each of which drives a
//     different branch of the version-gated upload and the per-frame-in-flight
//     catch-up.
//
// Run standalone (plain exit-code program, not Catch2) or via CTest. Exits 42
// (→ CTest "Skipped") when no Vulkan/RT GPU is available.

#include "threepp/threepp.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>

using namespace threepp;

namespace {

    constexpr int kW = 320, kH = 240;
    constexpr int kSkipCode = 42;

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  %s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    // Deterministic per-instance transforms with rotation, non-uniform scale and
    // translation, so every element of the product depends on every column of
    // both operands. A pure-translation instance matrix would pass even with the
    // multiply transposed.
    void fillInstances(InstancedMesh& mesh, unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        Matrix4 m;
        Quaternion q;
        Euler e;
        Vector3 pos, scale;
        for (size_t i = 0; i < mesh.count(); ++i) {
            pos.set(u(rng) * 6.f, u(rng) * 3.f, u(rng) * 6.f);
            e.set(u(rng) * 3.f, u(rng) * 3.f, u(rng) * 3.f);
            q.setFromEuler(e);
            scale.set(0.4f + 0.3f * u(rng), 0.4f + 0.3f * u(rng), 0.4f + 0.3f * u(rng));
            m.compose(pos, q, scale);
            mesh.setMatrixAt(i, m);
        }
        mesh.instanceMatrix()->needsUpdate();
    }

    std::shared_ptr<InstancedMesh> makeField(size_t n, unsigned seed) {
        auto mesh = InstancedMesh::create(BoxGeometry::create(0.2f, 0.2f, 0.2f),
                                          MeshStandardMaterial::create(), n);
        fillInstances(*mesh, seed);
        return mesh;
    }

    // Every check funnels through here so a failure prints the numbers that
    // identify WHICH way the two sides diverged: an ULP or two is a rounding /
    // FMA-contraction story, a large maxAbsDiff is an order / indexing story.
    void expectExact(VulkanRenderer& r, size_t wantEntries, const std::string& what) {
        VulkanRenderer::InstanceExpandCheck c{};
        if (!r.instanceExpandCheck(c)) {
            check(false, what + " — instanceExpandCheck reported nothing to compare");
            return;
        }
        std::printf("  [info] %s: spans=%zu compared=%zu mismatch=%zu maxAbs=%.9g maxUlp=%u\n",
                    what.c_str(), c.spans, c.entriesCompared, c.mismatches,
                    static_cast<double>(c.maxAbsDiff), c.maxUlpDiff);
        check(c.entriesCompared == wantEntries,
              what + " — compared every instanced entry (" +
                      std::to_string(c.entriesCompared) + " of " +
                      std::to_string(wantEntries) + ")");
        check(c.mismatches == 0, what + " — GPU == CPU bitwise");
    }

}// namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanInstanceExpand_test").size(kW, kH).vsync(false).headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    VulkanRenderer& renderer = *rendererPtr;

    // OFF by default: the pass is a measured regression whose output no
    // consumer reads yet, so every app in the tree would pay for nothing.
    // This test is the opt-in caller, which is the point of the setter.
    check(!renderer.gpuInstanceExpansion(), "GPU instance expansion is off by default");
    renderer.setGpuInstanceExpansion(true);

    Scene scene;
    scene.background = Color(0.05f, 0.05f, 0.08f);
    auto light = DirectionalLight::create(0xffffff, 2.f);
    light->position.set(4, 8, 6);
    scene.add(light);

    PerspectiveCamera camera(60.f, static_cast<float>(kW) / static_cast<float>(kH), 0.1f, 200.f);
    camera.position.set(0, 6, 22);
    camera.lookAt({0, 0, 0});

    // Span A: 3000 instances under a rotated / translated / non-uniformly
    // scaled parent — a full affine spanWorld.
    constexpr size_t kNa = 3000;
    auto groupA = Group::create();
    groupA->position.set(-4.f, 1.5f, 0.5f);
    groupA->rotation.set(0.31f, -0.77f, 0.19f);
    groupA->scale.set(1.3f, 0.7f, 1.1f);
    auto fieldA = makeField(kNa, 1234u);
    groupA->add(fieldA);
    scene.add(groupA);

    // Span B: a second instanced mesh with a DIFFERENT parent, so the shader's
    // span search has to pick the right one and the host's prefix sums have to
    // be right for span index > 0.
    constexpr size_t kNb = 1500;
    auto groupB = Group::create();
    groupB->position.set(5.f, -0.5f, -2.f);
    groupB->rotation.set(-0.44f, 0.61f, 0.85f);
    auto fieldB = makeField(kNb, 4321u);
    groupB->add(fieldB);
    scene.add(groupB);

    // A plain, non-instanced mesh. It stays on the CPU path by design (the
    // stage's scope discipline), so it must not be counted below.
    auto plain = Mesh::create(SphereGeometry::create(1.f), MeshStandardMaterial::create());
    plain->position.set(0, 4, 0);
    scene.add(plain);

    std::printf("first frame (full expansion):\n");
    renderer.render(scene, camera);
    expectExact(renderer, kNa + kNb, "full expansion");

    std::printf("static frames (nothing moved; both frames-in-flight):\n");
    // Two renders so the OTHER frame-in-flight becomes current with no span
    // marked moved — the case where a naive "upload what moved this frame" gate
    // would leave that slot's matrix pool empty and the check would explode.
    renderer.render(scene, camera);
    renderer.render(scene, camera);
    expectExact(renderer, kNa + kNb, "static, slot alternation");

    std::printf("instance matrices only (setMatrixAt + needsUpdate):\n");
    fillInstances(*fieldA, 999u);
    renderer.render(scene, camera);
    expectExact(renderer, kNa + kNb, "instance matrices moved");
    // ...and again on the sibling slot, without a further edit.
    renderer.render(scene, camera);
    expectExact(renderer, kNa + kNb, "instance matrices moved, sibling slot");

    std::printf("parent transform only (spanWorld moves, instances do not):\n");
    groupA->rotation.y += 0.4f;
    groupA->position.x -= 1.2f;
    renderer.render(scene, camera);
    expectExact(renderer, kNa + kNb, "spanWorld moved");
    renderer.render(scene, camera);
    expectExact(renderer, kNa + kNb, "spanWorld moved, sibling slot");

    std::printf("both at once, on both spans:\n");
    groupB->rotation.z -= 0.25f;
    fillInstances(*fieldB, 777u);
    fillInstances(*fieldA, 778u);
    groupA->scale.set(0.9f, 1.4f, 0.8f);
    renderer.render(scene, camera);
    expectExact(renderer, kNa + kNb, "both spans moved");

    std::printf("structural add (new span; layout + pool growth):\n");
    constexpr size_t kNc = 5000;// larger than A and B together — forces the
                                // matrix pool and the world buffer to grow
    auto groupC = Group::create();
    groupC->position.set(0.f, -3.f, 4.f);
    groupC->rotation.set(0.9f, 0.2f, -0.5f);
    auto fieldC = makeField(kNc, 5150u);
    groupC->add(fieldC);
    scene.add(groupC);
    renderer.render(scene, camera);
    expectExact(renderer, kNa + kNb + kNc, "span added");
    renderer.render(scene, camera);
    expectExact(renderer, kNa + kNb + kNc, "span added, sibling slot");

    std::printf("structural remove (span list shrinks; slots re-stamped):\n");
    scene.remove(*groupA);
    renderer.render(scene, camera);
    expectExact(renderer, kNb + kNc, "span removed");
    renderer.render(scene, camera);
    expectExact(renderer, kNb + kNc, "span removed, sibling slot");

    std::printf("instance count change (setCount → structural re-expansion):\n");
    fieldC->setCount(kNc / 2);
    renderer.render(scene, camera);
    expectExact(renderer, kNb + kNc / 2, "instance count halved");

    std::printf("switch off:\n");
    renderer.setGpuInstanceExpansion(false);
    renderer.render(scene, camera);
    {
        VulkanRenderer::InstanceExpandCheck c{};
        check(!renderer.instanceExpandCheck(c),
              "instanceExpandCheck reports nothing while the pass is off");
    }
    // Back on: the pass must pick the scene up again from a cold per-slot state.
    renderer.setGpuInstanceExpansion(true);
    renderer.render(scene, camera);
    expectExact(renderer, kNb + kNc / 2, "re-enabled");

    std::printf("%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
