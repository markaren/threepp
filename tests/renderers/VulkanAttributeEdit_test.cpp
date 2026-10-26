// VulkanAttributeEdit_test: a version bump of any one geometry attribute the
// renderer tracks (position, normal, uv, uv2, color, index) is seen on a frame
// whose scene did not otherwise change.
//
// Such a frame takes the lean diff (VulkanCoreScene.cpp, scene.5_leanDiff),
// which reads the attribute versions through pointers the fingerprint cached
// rather than through geomVersionOf. Those pointers were a hand-kept copy of
// geomVersionOf's list and fell behind it twice: color, and then uv2, which
// joined the list for the aoMap's UVs and was never cached, so a uv2-only
// edit on a settled scene went unseen (2026-10-26). Both now read one list
// (GeomAttrRefs, VulkanGeometryState.hpp).
//
// "Seen" here: the edited mesh counts as moving on the frame after the edit
// (TlasStats::movingEntries), as any geometry edit makes it. Before each edit
// the count settles back to zero.
//
// Plain exit-code program, not Catch2, like the other Vulkan tests here.
// Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"
#include "vulkan_test_util.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace threepp;
using vktest::kSkipCode;

namespace {

    constexpr int kW = 160, kH = 96;
    // The moving count holds an entry for 30 frames after its last move.
    constexpr int kSettleFrames = 120;

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  %s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

}// namespace

int main() {

    std::setvbuf(stdout, nullptr, _IONBF, 0);

    auto gpu = vktest::openHeadlessVulkan("VulkanAttributeEdit_test", kW, kH);
    if (!gpu) return kSkipCode;
    Canvas& canvas = *gpu.canvas;
    VulkanRenderer& renderer = *gpu.renderer;
    renderer.setDlss(false);
    renderer.setFsr(false);

    Scene scene;
    auto light = DirectionalLight::create(Color(1, 1, 1), 2.f);
    light->position.set(1, 3, 2);
    scene.add(light);

    PerspectiveCamera camera(40.f, static_cast<float>(kW) / kH, 0.1f, 100.f);
    camera.position.set(0, 1.f, 4.f);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    // The mesh under test carries every tracked attribute.
    auto geom = PlaneGeometry::create(2.f, 2.f, 4, 4);
    geom->setAttribute("uv2", FloatBufferAttribute::create(geom->getAttribute<float>("uv")->array(), 2));
    std::vector<float> colors(static_cast<size_t>(geom->getAttribute<float>("position")->count()) * 3, 1.f);
    geom->setAttribute("color", FloatBufferAttribute::create(colors, 3));
    auto mat = MeshStandardMaterial::create();
    mat->vertexColors = true;
    scene.add(Mesh::create(geom, mat));
    // A bystander, so the edited mesh is not the whole scene.
    auto other = Mesh::create(SphereGeometry::create(0.5f), MeshStandardMaterial::create());
    other->position.set(1.5f, 0.5f, 0);
    scene.add(other);

    const auto frame = [&] { canvas.animateOnce([&] { renderer.render(scene, camera); }); };
    const auto settle = [&] {
        for (int i = 0; i < kSettleFrames && renderer.tlasStats().movingEntries != 0u; ++i) frame();
        return renderer.tlasStats().movingEntries == 0u;
    };

    for (int i = 0; i < 3; ++i) frame();
    check(settle(), "the scene settles: no entry moving");

    struct Edit {
        const char* name;
        BufferAttribute* attr;
    };
    const Edit edits[] = {
            {"position", geom->getAttribute("position")},
            {"normal", geom->getAttribute("normal")},
            {"uv", geom->getAttribute("uv")},
            {"uv2", geom->getAttribute("uv2")},
            {"color", geom->getAttribute("color")},
            {"index", geom->getIndex()},
    };
    for (const Edit& e : edits) {
        const std::string name = e.name;
        if (!e.attr) {
            check(false, "the plane has a " + name + " attribute");
            continue;
        }
        const auto flatBefore = renderer.tlasStats().flatChecks;
        e.attr->needsUpdate();
        frame();
        const auto stats = renderer.tlasStats();
        check(stats.flatChecks > flatBefore, name + " edited alone: the frame takes the lean diff");
        check(stats.movingEntries >= 1u, name + " edited alone: the edit is seen");
        check(settle(), "the scene settles again after the " + name + " edit");
    }

    std::printf("attribute edit: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
