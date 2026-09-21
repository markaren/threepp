// VulkanRecycledAddress_test — a replaced mesh must be seen as a NEW mesh even
// when the allocator gives it the dead one's address.
//
// The Vulkan scene sync used to recognise an unchanged node by the raw Object3D*,
// BufferGeometry* and Material* plus the two version counters. A mesh that is
// removed and freed, and then replaced by a freshly created one before the next
// render, routinely comes back at the SAME three addresses: the allocations are
// the same sizes, made in the same order, straight after the frees. Its versions
// are 0 again as well, so every identity the sync compared still matched, the
// node took the unchanged fast path, and the PREVIOUS material's parameters went
// on being rendered. First measured with a furnace harness that built a fresh
// sphere per case: three of sixteen cases read exactly the previous case's value.
//
// The ids (Object3D::id, BufferGeometry::id, Material::id) are process-unique and
// never repeat, so comparing them alongside the pointers tells the two apart.
//
// What this does: alternates a white metal and a black dielectric sphere in a
// uniform environment, replacing mesh + geometry + material every round with the
// old ones freed FIRST, and requires each reading to be the current material's.
// The two readings are an order of magnitude apart (about 0.9 and 0.02 of the
// environment), so nothing here depends on convergence or on the exact shading.
//
// Whether the allocator actually recycles the addresses is its own business. The
// test counts the rounds in which all three repeated and prints it; with none it
// cannot fail and says so. Measured on MSVC RelWithDebInfo with the id compare
// disabled: every round in which all three repeated was stale (2 of 2), and none
// of the other 190 was. With it enabled, 9 of 9 such rounds were correct.
//
// Plain exit-code program, not Catch2 — same shape as the other Vulkan tests
// here. Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <vector>

using namespace threepp;

namespace {

    constexpr int kSize = 128;
    // Many short rounds rather than a few long ones. All three addresses repeat
    // in about 3% of replacements here (measured over 600 rounds), so 96 rounds
    // make a run that exercises the defect likely without making it certain. 30
    // frames are plenty after a history reset, given how far apart the two
    // readings are.
    constexpr int kFrames = 30;
    constexpr int kRounds = 96;
    constexpr int kSkipCode = 42;

    std::shared_ptr<Texture> constantEnv(float le) {
        constexpr int W = 64, H = 32;
        std::vector<float> data(static_cast<size_t>(W) * H * 4, le);
        for (size_t i = 3; i < data.size(); i += 4) data[i] = 1.f;
        Image img{std::move(data), static_cast<unsigned>(W), static_cast<unsigned>(H), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->type = Type::Float;
        tex->colorSpace = ColorSpace::Linear;
        tex->mapping = Mapping::EquirectangularReflection;
        tex->needsUpdate();
        return tex;
    }

    double srgbToLinear(double c) {
        return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    }

    double patch(const std::vector<unsigned char>& px, int cx, int cy, int r) {
        double sum = 0;
        int n = 0;
        for (int y = cy - r; y <= cy + r; ++y) {
            for (int x = cx - r; x <= cx + r; ++x) {
                sum += srgbToLinear(px[(static_cast<size_t>(y) * kSize + x) * 3] / 255.0);
                ++n;
            }
        }
        return sum / n;
    }

}// namespace

int main() {

    std::unique_ptr<Canvas> canvasPtr;
    std::unique_ptr<VulkanRenderer> rendererPtr;
    try {
        canvasPtr = std::make_unique<Canvas>(
                Canvas::Parameters().title("VulkanRecycledAddress_test").size(kSize, kSize).vsync(false).headless(true));
        rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
    } catch (const std::exception& e) {
        std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
        return kSkipCode;
    }
    Canvas& canvas = *canvasPtr;
    VulkanRenderer& renderer = *rendererPtr;

    renderer.setDenoise(true);
    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setBloomIntensity(0.f);
    renderer.toneMapping = ToneMapping::None;
    renderer.toneMappingExposure = 1.0f;

    const auto env = constantEnv(0.5f);

    // The scene itself persists, so the snapshot replay starts from a root it
    // recognises and the only thing that changes is the one child.
    Scene scene;
    scene.background = env;
    scene.environment = env;

    PerspectiveCamera camera(45.f, 1.f, 0.1f, 100.f);
    camera.position.set(0, 0, 3.2f);
    camera.lookAt(Vector3{0, 0, 0});
    camera.updateMatrixWorld();

    std::shared_ptr<Mesh> mesh;
    std::uintptr_t prevMesh = 0, prevGeom = 0, prevMat = 0;
    int recycled = 0;
    bool recycledThisRound = false;

    const auto replace = [&](int round) {
        if (mesh) {
            prevMesh = reinterpret_cast<std::uintptr_t>(mesh.get());
            prevGeom = reinterpret_cast<std::uintptr_t>(mesh->geometry().get());
            prevMat = reinterpret_cast<std::uintptr_t>(mesh->material().get());
            scene.remove(*mesh);
            mesh.reset();// frees mesh, geometry and material BEFORE the next three allocations
        }

        const bool white = (round % 2) == 0;
        auto geometry = SphereGeometry::create(1.f, 48, 32);
        auto material = MeshStandardMaterial::create();
        material->color = white ? Color(1, 1, 1) : Color(0, 0, 0);
        material->metalness = white ? 1.f : 0.f;
        material->roughness = white ? 0.2f : 1.f;
        mesh = Mesh::create(geometry, material);
        scene.add(mesh);

        // Through the mesh's own accessors, as above and as the renderer sees them:
        // Material is a virtual base, so a MeshStandardMaterial* and the Material*
        // of the same object are different addresses and would never compare equal.
        const bool sameMesh = reinterpret_cast<std::uintptr_t>(mesh.get()) == prevMesh;
        const bool sameGeom = reinterpret_cast<std::uintptr_t>(mesh->geometry().get()) == prevGeom;
        const bool sameMat = reinterpret_cast<std::uintptr_t>(mesh->material().get()) == prevMat;
        recycledThisRound = sameMesh && sameGeom && sameMat;
        if (recycledThisRound) ++recycled;

        // Same sphere, same camera: nothing disoccludes, so the temporal
        // histories would otherwise carry the previous round's radiance.
        renderer.resetTemporalHistory();
    };

    int round = 0, frame = 0, failures = 0;
    replace(round);

    canvas.animate([&] {
        renderer.render(scene, camera);
        if (++frame < kFrames) return;
        frame = 0;

        const auto px = renderer.readRGBPixels();
        if (px.size() < static_cast<size_t>(kSize) * kSize * 3) {
            std::printf("readback too small: %zu\n", px.size());
            std::exit(1);
        }
        const double fraction = patch(px, kSize / 2, kSize / 2, 3) / patch(px, 6, 6, 3);
        const bool white = (round % 2) == 0;
        const bool ok = white ? fraction > 0.5 : fraction < 0.2;
        // Only the rounds that say something: a stale one, or one that ran at
        // recycled addresses and therefore could have been.
        if (!ok || recycledThisRound) {
            std::printf("[round %2d] %-16s centre = %.3f of the environment%s  ->  %s\n",
                        round, white ? "white metal" : "black dielectric", fraction,
                        recycledThisRound ? "  (all three addresses recycled)" : "", ok ? "ok" : "STALE");
        }
        if (!ok) ++failures;

        if (++round < kRounds) {
            replace(round);
            return;
        }

        std::printf("%d of %d replacements came back at all three recycled addresses%s\n",
                    recycled, kRounds - 1,
                    recycled == 0 ? "  [inconclusive: this run could not have failed]" : "");
        std::printf("recycled-address: %d/%d rounds stale\n", failures, kRounds);
        std::exit(failures == 0 ? 0 : 1);
    });
    return 0;
}
