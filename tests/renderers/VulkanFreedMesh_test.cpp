// VulkanFreedMesh_test — a mesh removed from the scene and freed between two
// frames is gone as far as the renderer is concerned: nothing reads it again,
// and nothing it owned on the GPU outlives it.
//
// Between two frames the renderer's entry list still names the meshes the LAST
// frame drew, by address. The dune buggy swapped the player's car between two
// frames and crashed with an access violation in set_emissive_casts_light: the
// old car was out of the scene and Python had dropped its last reference, so
// the walk that republishes the glow-only bit read a freed Mesh. The same walk
// lives in setStableCorrespondence and in enableVertexInterop's packed-record
// swap. And a deforming mesh's GPU state (skinned here) was keyed by the mesh's
// address but freed only with its GEOMETRY, so a clone sharing its template's
// geometry left a whole skinning state behind when it died.
//
// The doomed meshes are allocated through PoisonAllocator: when the last
// reference goes, their bytes are overwritten with 0xFE and never handed back
// to the heap, so a read of a dead mesh dereferences a non-canonical pointer
// and faults every time instead of reading whatever the allocator left there.
//
// Rounds, each with a fresh doomed mesh in a Group (the car) beside a live mesh
// sharing its geometry (the clone's template), four frames drawn with both, the
// Group removed and freed, then:
//
//   0. plain     nothing, the next frames only (the scene build's own path).
//   1. emissive  setEmissiveCastsLight on the live mesh.
//   2. stable    setStableCorrespondence on the live mesh.
//   3. interop   enableVertexInterop on the live mesh (packed record -> unpacked
//                swap), then disableVertexInterop. Skipped where the device
//                cannot export memory.
//   4. skinned   four skinned meshes sharing one geometry with a fifth that
//                stays: the four freed, the GPU allocation must come back down
//                to what it was before they were added.
//
// After each: the doomed allocation really was freed (the poison count moved:
// nothing kept the mesh alive, so the round tested what it says), three more
// frames render, the frame is not black, no validation errors.
// Rounds 1 to 3 crashed on the old code; round 4 kept about 6 MB per freed mesh.
// --only N runs round N alone (each round reproduces on its own).
//
// Plain exit-code program, like the other Vulkan tests here. Exits 42 (CTest
// "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/BufferAttribute.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"
#include "threepp/scenes/Scene.hpp"
#include "vulkan_test_util.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

using namespace threepp;
using vktest::kSkipCode;

namespace {

    constexpr int kW = 320, kH = 240;

    int failures = 0;

    void check(bool ok, const std::string& what) {
        std::printf("  %s %s\n", ok ? "[ ok ]" : "[FAIL]", what.c_str());
        if (!ok) ++failures;
    }

    // Blocks freed through PoisonAllocator so far.
    int poisonedBlocks = 0;

    // std::allocate_shared puts the control block and the object in one block
    // from this allocator, and returns it when the last shared_ptr and weak_ptr
    // are gone, right after the destructor. Poisoned and kept, never reused: a
    // dead Mesh's vtable and virtual-base pointers read 0xFEFE..., which no
    // platform maps.
    template<class T>
    struct PoisonAllocator {
        using value_type = T;
        PoisonAllocator() = default;
        template<class U>
        PoisonAllocator(const PoisonAllocator<U>&) {}

        T* allocate(std::size_t n) { return static_cast<T*>(::operator new(n * sizeof(T))); }
        void deallocate(T* p, std::size_t n) {
            std::memset(static_cast<void*>(p), 0xFE, n * sizeof(T));
            ++poisonedBlocks;// leaked on purpose: the address stays poisoned
        }
        template<class U>
        bool operator==(const PoisonAllocator<U>&) const { return true; }
        template<class U>
        bool operator!=(const PoisonAllocator<U>&) const { return false; }
    };

    std::shared_ptr<MeshStandardMaterial> emissiveMaterial(const Color& c) {
        auto m               = MeshStandardMaterial::create();
        m->color             = Color(0.1f, 0.1f, 0.1f);
        m->emissive          = c;
        m->emissiveIntensity = 4.f;
        return m;
    }

    struct Rig {
        std::shared_ptr<Bone> bone;
        std::shared_ptr<SkinnedMesh> mesh;
    };

    // A skinned mesh on `geometry` (skinIndex/skinWeight already on it) with a
    // one-bone skeleton of its own, both under `parent`.
    template<class Alloc>
    Rig addSkinned(Object3D& parent, const std::shared_ptr<BufferGeometry>& geometry,
                   const std::shared_ptr<Material>& material, const Alloc& alloc, float x) {
        Rig r;
        r.bone = Bone::create();
        r.bone->position.x = x;
        parent.add(r.bone);
        r.mesh = std::allocate_shared<SkinnedMesh>(alloc, geometry, material);
        r.mesh->position.x = x;
        parent.add(r.mesh);
        parent.updateMatrixWorld(true);
        Matrix4 inv(*r.bone->matrixWorld);
        inv.invert();
        r.mesh->bind(Skeleton::create({r.bone}, {inv}), Matrix4());
        return r;
    }

}// namespace

int main(int argc, char** argv) {
    std::printf("VulkanFreedMesh_test\n");
    int only = -1;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--only") == 0) only = std::atoi(argv[i + 1]);
    }
#ifdef _WIN32
    _putenv_s("THREEPP_VULKAN_VALIDATION", "1");
#else
    setenv("THREEPP_VULKAN_VALIDATION", "1", 1);
#endif

    auto vk = vktest::openHeadlessVulkan("VulkanFreedMesh_test", kW, kH);
    if (!vk) return kSkipCode;
    Canvas& canvas           = *vk.canvas;
    VulkanRenderer& renderer = *vk.renderer;
    renderer.setDlss(false);
    renderer.setFsr(false);

    Scene scene;
    scene.background = Color(0.05f, 0.06f, 0.08f);
    auto floor       = Mesh::create(PlaneGeometry::create(20.f, 20.f), MeshStandardMaterial::create());
    floor->rotateX(-math::PI / 2.f);
    scene.add(floor);
    auto sun = DirectionalLight::create(Color(1.f, 1.f, 1.f), 1.f);
    sun->position.set(2.f, 5.f, 3.f);
    scene.add(sun);

    PerspectiveCamera camera(50.f, float(kW) / float(kH), 0.1f, 100.f);
    camera.position.set(0.f, 2.5f, 6.f);
    camera.lookAt(Vector3(0.f, 0.5f, 0.f));

    double t         = 0.0;
    const auto frame = [&] {
        t += 1.0 / 60.0;
        renderer.setSimTime(t);
        canvas.animateOnce([&] { renderer.render(scene, camera); });
    };
    const auto frameIsDrawn = [&] {
        const auto px = renderer.readRGBPixels();
        if (px.empty()) return false;
        unsigned maxV = 0;
        for (const unsigned char v : px) maxV = v > maxV ? v : maxV;
        return maxV > 40u;
    };
    const std::uint32_t errorsBefore = vulkan::validationErrorCount();
    const PoisonAllocator<Mesh> poison;

    const char* names[] = {"plain", "emissive", "stable", "interop"};
    for (int round = 0; round < 4; ++round) {
        if (only >= 0 && only != round) continue;
        std::printf("round %d: %s\n", round, names[round]);
        // Its own geometry each round, so the interop round meets a record that
        // is still packed (the default for a plain static mesh).
        auto geometry = SphereGeometry::create(0.4f, 24, 16);
        auto keeper   = Mesh::create(geometry, emissiveMaterial(Color(1.f, 0.6f, 0.2f)));
        keeper->position.set(1.f, 0.5f, 0.f);
        scene.add(keeper);
        {
            auto car    = Group::create();
            auto doomed = std::allocate_shared<Mesh>(poison, geometry, emissiveMaterial(Color(0.2f, 0.6f, 1.f)));
            doomed->position.set(-1.f, 0.5f, 0.f);
            car->add(doomed);
            scene.add(car);
            for (int i = 0; i < 4; ++i) frame();
            const int before = poisonedBlocks;
            scene.remove(*car);
            car.reset();
            doomed.reset();
            check(poisonedBlocks == before + 1, "the doomed mesh was freed between the frames");
        }

        if (round == 1) {
            renderer.setEmissiveCastsLight(*keeper, false);
            renderer.setEmissiveCastsLight(*keeper, true);
            renderer.setEmissiveCastsLight(*keeper, false);
        } else if (round == 2) {
            renderer.setStableCorrespondence(*keeper, false);
        } else if (round == 3) {
            const auto h = renderer.enableVertexInterop(*keeper, [] {});
            if (!h.posHandle) {
                std::printf("  [info] no vertex interop on this device: the round only renders\n");
            } else {
                renderer.disableVertexInterop(*keeper);
            }
        }
        for (int i = 0; i < 3; ++i) frame();
        check(frameIsDrawn(), "the next frames render");
        scene.remove(*keeper);
    }

    if (only < 0 || only == 4) {
        std::printf("round 4: skinned\n");
        // About 66k vertices, so one skinning state (its vertex, normal, skin
        // index/weight buffers and BLAS) is megabytes, far above the noise of
        // the renderer's own per-frame buffers.
        auto geometry    = SphereGeometry::create(0.4f, 256, 256);
        const auto count = geometry->getAttribute<float>("position")->count();
        std::vector<float> idx(count * 4, 0.f), w(count * 4, 0.f);
        for (unsigned i = 0; i < count; ++i) w[i * 4] = 1.f;
        geometry->setAttribute("skinIndex", FloatBufferAttribute::create(std::move(idx), 4));
        geometry->setAttribute("skinWeight", FloatBufferAttribute::create(std::move(w), 4));
        auto material = MeshStandardMaterial::create();

        auto stays   = Group::create();
        auto stayRig = addSkinned(*stays, geometry, material, std::allocator<SkinnedMesh>(), 0.f);
        scene.add(stays);
        for (int i = 0; i < 6; ++i) frame();
        const auto m0 = renderer.gpuAllocatedBytes();

        constexpr int kClones = 4;
        const int before = poisonedBlocks;
        {
            auto clones = Group::create();
            std::vector<Rig> rigs;
            for (int c = 0; c < kClones; ++c) {
                rigs.push_back(addSkinned(*clones, geometry, material, PoisonAllocator<SkinnedMesh>(),
                                          -1.5f + float(c)));
            }
            scene.add(clones);
            for (int i = 0; i < 6; ++i) frame();
            const auto m1 = renderer.gpuAllocatedBytes();
            scene.remove(*clones);
            clones.reset();
            rigs.clear();
            for (int i = 0; i < 8; ++i) frame();
            // After the frames, not before: a block from allocate_shared is
            // returned when the last weak_ptr goes too, and the renderer may
            // hold one until its next scene build lets the state go.
            check(poisonedBlocks == before + kClones, "the skinned clones were freed");
            const auto m2    = renderer.gpuAllocatedBytes();
            const double mib = 1.0 / (1024.0 * 1024.0);
            const double add = double(m1) - double(m0), left = double(m2) - double(m0);
            std::printf("  [info] GPU allocation: %.1f MiB with one, %.1f with five, %.1f after the four "
                        "were freed\n",
                        double(m0) * mib, double(m1) * mib, double(m2) * mib);
            if (add < 4.0 * 1024 * 1024) {
                std::printf("  [info] the clones added only %.2f MiB: no skinning state to leak here\n",
                            add * mib);
            } else {
                check(left < 0.25 * add, "the freed clones' skinning states were released");
            }
        }
        check(frameIsDrawn(), "the next frames render");
        (void) stayRig;
    }

    const std::uint32_t errors = vulkan::validationErrorCount() - errorsBefore;
    check(errors == 0, "no validation errors (" + std::to_string(errors) + ")");

    if (failures) {
        std::printf("FAIL (%d)\n", failures);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
