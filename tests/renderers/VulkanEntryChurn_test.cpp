// VulkanEntryChurn_test: content streaming through the Vulkan renderer's entry
// list, the way terrain::CellStreamer and the pine forest's LODs drive it, and
// what each kind of change costs the frame that carries it.
//
// The renderer keeps one MeshEntry per mesh instance. Any change to that
// list's SHAPE is a structural rebuild in ensureSceneBuilt: the device is
// drained, and every descriptor, draw record and TLAS instance is re-derived
// (about 30 ms an event on the forest, 2026-10-18). This bench pins that
// defect before it is fixed (plans/gpu-driven-instances.md, stage 2a: stable
// entry slots), and is the gate every later stage is measured against.
//
// The scene: a ground plane, an InstancedMesh "forest" of cones, cells of
// plain meshes (boxes, spheres, cylinders from shared geometries and
// materials, one of them textured) and spheres hidden since before their first
// render. The camera orbits slowly, so TAA and the motion vectors are live.
// Every kGap frames one event:
//   remove cell        a cell's group detached (CellStreamer leaving a cell)
//   add cached cell    a detached cell re-attached (coming back to it)
//   add new cell       a cell built now from the shared assets and attached
//   add new geometry   a cell whose meshes carry geometries never seen
//                      (a BLAS admission; informational, never asserted)
//   instanced grow     InstancedMesh::setCount up by kInstStep
//   instanced shrink   InstancedMesh::setCount down by kInstStep
//   first show         a mesh hidden since before its first render is shown
// Per frame: the wall time of render(), FrameTimings (cpuFrameMs,
// cpuEnsureSceneMs, gpuTotalMs), TlasStats::structuralRebuilds and
// TlasStats::historyResets (temporal history wiped). It prints one table:
// median, p99 and max frame; per event kind the structural rebuilds and
// history resets per event and the event frame against the quiet median.
//
// The assertion is the per-kind expectation in kExpect: the structural
// rebuilds one event of that kind costs TODAY (one). A kind whose expectation
// becomes 0 must also bring its event frame within kHitchRatio of the quiet
// median: the row edit replaces the rebuild, it does not move it to the next
// frame. No history reset on any event, ever.
//
// Auto-LOD is off: its chains land asynchronously and their TLAS rebuilds
// would land on random frames of the table.
//
// THREEPP_CHURN_DUMP=<dir> writes the event frames and the frame after each as
// PPM there, for looking at; that readback stalls, so its timings are void.
// THREEPP_CHURN_INSTANCES=<n> sets the forest's starting count (default 8192;
// its capacity is twice that).
//
// Plain exit-code program, not Catch2, like the other Vulkan tests here.
// Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/ConeGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/renderers/vulkan/ValidationReport.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"
#include "vulkan_test_util.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using vktest::kSkipCode;

namespace {

    constexpr int kW = 640, kH = 360;
    constexpr int kWarmupFrames = 90;
    constexpr int kGap = 8;     // frames from one event to the next
    constexpr int kRounds = 5;  // every kind this many times
    constexpr int kCells = 40;  // cell slots on the ground
    constexpr int kCellsAtStart = 24;
    constexpr int kMeshesPerCell = 12;
    constexpr int kHiddenMeshes = 8;
    constexpr size_t kInstStep = 512;
    constexpr double kHitchRatio = 2.0;
    constexpr double kHitchSlackMs = 2.0;// floor for frames so short that 2x is noise

    enum Kind { kRemoveCell, kAddCachedCell, kAddNewCell, kAddNewGeom, kInstGrow, kInstShrink, kFirstShow, kKinds };
    constexpr const char* kKindName[kKinds] = {"remove cell", "add cached cell", "add new cell", "add new geometry",
                                               "instanced grow", "instanced shrink", "first show"};
    // Structural rebuilds one event of the kind costs. -1: informational, not
    // asserted. The baseline (2026-10-19): every kind is one.
    constexpr int kExpect[kKinds] = {1, 1, 1, -1, 1, 1, 1};

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

    std::shared_ptr<Texture> checker() {
        constexpr int N = 32;
        std::vector<unsigned char> data(static_cast<size_t>(N) * N * 4);
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                const bool on = ((x / 4) + (y / 4)) % 2 == 0;
                unsigned char* p = &data[(static_cast<size_t>(y) * N + x) * 4];
                p[0] = on ? 230 : 40;
                p[1] = on ? 200 : 40;
                p[2] = on ? 60 : 40;
                p[3] = 255;
            }
        }
        Image img{std::move(data), static_cast<unsigned>(N), static_cast<unsigned>(N), 0};
        auto tex = Texture::create(img);
        tex->format = Format::RGBA;
        tex->needsUpdate();
        return tex;
    }

    std::shared_ptr<MeshStandardMaterial> paint(const Color& c) {
        auto m = MeshStandardMaterial::create();
        m->color = c;
        m->metalness = 0.f;
        m->roughness = 0.7f;
        return m;
    }

    // Deterministic jitter in [-1, 1).
    float hash(uint32_t i) {
        i ^= i >> 16;
        i *= 0x7feb352du;
        i ^= i >> 15;
        i *= 0x846ca68bu;
        i ^= i >> 16;
        return static_cast<float>(i & 0xffffffu) / 8388608.f - 1.f;
    }

    double percentile(std::vector<double> v, double q) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        const size_t i = std::min(v.size() - 1, static_cast<size_t>(q * static_cast<double>(v.size() - 1) + 0.5));
        return v[i];
    }

    void writePpm(const std::string& path, const std::vector<unsigned char>& rgb) {
        if (FILE* f = std::fopen(path.c_str(), "wb")) {
            std::fprintf(f, "P6\n%d %d\n255\n", kW, kH);
            std::fwrite(rgb.data(), 1, static_cast<size_t>(kW) * kH * 3, f);
            std::fclose(f);
        }
    }

    struct FrameRow {
        int kind = -1;// the event this frame carries, -1 for a quiet frame
        int sinceEvent = 0;
        double wallMs = 0, cpuFrameMs = 0, cpuEnsureMs = 0, gpuMs = 0;
        uint64_t structural = 0, history = 0;// deltas over this frame
    };

}// namespace

int main() {

    std::setvbuf(stdout, nullptr, _IONBF, 0);

    size_t instStart = 8192;
    if (const char* e = std::getenv("THREEPP_CHURN_INSTANCES"); e && *e) {
        instStart = std::clamp<size_t>(std::strtoull(e, nullptr, 10), kInstStep * 2, size_t(1) << 20);
    }
    const size_t kInstCapacity = instStart * 2;
    const char* dumpDir = std::getenv("THREEPP_CHURN_DUMP");
    const bool dump = dumpDir && *dumpDir;

    auto gpu = vktest::openHeadlessVulkan("VulkanEntryChurn_test", kW, kH);
    if (!gpu) return kSkipCode;
    Canvas& canvas = *gpu.canvas;
    VulkanRenderer& renderer = *gpu.renderer;

    renderer.setRenderScale(1.0f);
    renderer.setDlss(false);
    renderer.setFsr(false);
    renderer.setAutoExposure(false);
    renderer.setAutoLod(false);
    renderer.setBloomIntensity(0.f);
    renderer.toneMapping = ToneMapping::ACESFilmic;
    renderer.toneMappingExposure = 1.0f;
    std::printf("ray query: %s\n", renderer.rayTracingSupported() ? "yes" : "no");

    const auto env = constantEnv(0.35f);
    Scene scene;
    scene.background = env;
    scene.environment = env;

    PerspectiveCamera camera(50.f, static_cast<float>(kW) / kH, 0.5f, 600.f);

    auto ground = Mesh::create(PlaneGeometry::create(400.f, 400.f), paint(Color(0.45f, 0.42f, 0.38f)));
    ground->rotation.x = -math::PI / 2;
    scene.add(ground);

    auto sun = DirectionalLight::create(0xffffff, 2.5f);
    sun->position.set(40.f, 80.f, 30.f);
    scene.add(sun);

    // The forest: one InstancedMesh, every capacity slot given a matrix up
    // front, so growing the count shows trees that were staged all along.
    auto forest = InstancedMesh::create(ConeGeometry::create(0.7f, 3.f, 8), paint(Color(0.12f, 0.45f, 0.15f)),
                                        kInstCapacity);
    {
        Matrix4 m;
        const int side = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(kInstCapacity))));
        for (size_t i = 0; i < kInstCapacity; ++i) {
            const float gx = static_cast<float>(i % side), gz = static_cast<float>(i / side);
            const float x = -90.f + gx * (180.f / side) + hash(static_cast<uint32_t>(i) * 2u) * 0.6f;
            const float z = -90.f + gz * (60.f / side) + hash(static_cast<uint32_t>(i) * 2u + 1u) * 0.3f;
            m.makeTranslation(x, 1.5f, z);
            forest->setMatrixAt(i, m);
        }
        forest->instanceMatrix()->needsUpdate();
    }
    forest->setCount(instStart);
    scene.add(forest);

    // Cells: shared geometries and materials, like a CellStreamer builder that
    // places props from one asset set.
    const std::array<std::shared_ptr<BufferGeometry>, 3> cellGeoms = {
            BoxGeometry::create(1.6f, 1.6f, 1.6f), SphereGeometry::create(0.9f, 16, 8),
            CylinderGeometry::create(0.6f, 0.6f, 2.f, 12)};
    auto texMat = paint(Color(1, 1, 1));
    texMat->map = checker();
    const std::array<std::shared_ptr<Material>, 3> cellMats = {paint(Color(0.7f, 0.15f, 0.1f)),
                                                               paint(Color(0.15f, 0.25f, 0.75f)), texMat};
    uint32_t meshSerial = 0;
    const auto cellOrigin = [](int c) {
        return Vector3(-75.f + static_cast<float>(c % 8) * 20.f, 0.f, 10.f + static_cast<float>(c / 8) * 16.f);
    };
    const auto buildCell = [&](int c, bool newGeometry) {
        auto g = Group::create();
        const Vector3 o = cellOrigin(c);
        for (int k = 0; k < kMeshesPerCell; ++k) {
            const uint32_t s = meshSerial++;
            std::shared_ptr<BufferGeometry> geom =
                    newGeometry ? BoxGeometry::create(1.f + 0.05f * static_cast<float>(k), 1.2f, 1.f)
                                : cellGeoms[s % cellGeoms.size()];
            auto m = Mesh::create(geom, cellMats[(s / 3) % cellMats.size()]);
            m->position.set(o.x + hash(s * 3u) * 7.f, 1.f, o.z + hash(s * 3u + 1u) * 6.f);
            m->rotation.y = hash(s * 3u + 2u) * math::PI;
            g->add(m);
        }
        return g;
    };
    std::vector<std::shared_ptr<Group>> active, cached;
    int nextFreshCell = 0;
    for (; nextFreshCell < kCellsAtStart; ++nextFreshCell) {
        active.push_back(buildCell(nextFreshCell, false));
        scene.add(active.back());
    }

    // Hidden since before their first render, not resident.
    std::vector<std::shared_ptr<Mesh>> hiddenMeshes;
    for (int k = 0; k < kHiddenMeshes; ++k) {
        auto m = Mesh::create(cellGeoms[1], cellMats[k % 2]);
        m->position.set(-35.f + static_cast<float>(k) * 10.f, 4.f, -20.f);
        m->visible = false;
        scene.add(m);
        hiddenMeshes.push_back(m);
    }
    size_t nextShow = 0;

    float orbit = 0.f;
    const auto placeCamera = [&] {
        camera.position.set(std::sin(orbit) * 70.f, 28.f, std::cos(orbit) * 70.f);
        camera.lookAt(Vector3{0, 0, 0});
        camera.updateMatrixWorld();
        orbit += 0.004f;
    };

    int failures = 0;
    const auto check = [&](const std::string& what, bool ok) {
        std::printf("  %-74s %s\n", what.c_str(), ok ? "ok" : "FAIL");
        if (!ok) ++failures;
    };

    std::vector<FrameRow> rows;
    try {
        const auto frame = [&](int kind, int sinceEvent) {
            placeCamera();
            const auto before = renderer.tlasStats();
            double wall = 0;
            canvas.animateOnce([&] {
                const auto t0 = std::chrono::steady_clock::now();
                renderer.render(scene, camera);
                wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            });
            const auto after = renderer.tlasStats();
            const auto ft = renderer.lastFrameTimings();
            FrameRow r;
            r.kind = kind;
            r.sinceEvent = sinceEvent;
            r.wallMs = wall;
            r.cpuFrameMs = ft.cpuFrameMs;
            r.cpuEnsureMs = ft.cpuEnsureSceneMs;
            r.gpuMs = ft.gpuTotalMs;
            r.structural = after.structuralRebuilds - before.structuralRebuilds;
            r.history = after.historyResets - before.historyResets;
            return r;
        };

        for (int i = 0; i < kWarmupFrames; ++i) frame(-1, kGap);

        const auto stats0 = renderer.tlasStats();
        const auto t0 = std::chrono::steady_clock::now();
        for (int round = 0; round < kRounds; ++round) {
            for (int kind = 0; kind < kKinds; ++kind) {
                switch (kind) {
                    case kRemoveCell: {
                        auto g = active.front();
                        active.erase(active.begin());
                        scene.remove(*g);
                        cached.push_back(g);
                        break;
                    }
                    case kAddCachedCell: {
                        auto g = cached.front();
                        cached.erase(cached.begin());
                        scene.add(g);
                        active.push_back(g);
                        break;
                    }
                    case kAddNewCell:
                    case kAddNewGeom: {
                        auto g = buildCell(nextFreshCell++ % kCells, kind == kAddNewGeom);
                        scene.add(g);
                        active.push_back(g);
                        break;
                    }
                    case kInstGrow:
                        forest->setCount(forest->count() + kInstStep);
                        break;
                    case kInstShrink:
                        forest->setCount(forest->count() - kInstStep);
                        break;
                    case kFirstShow:
                        hiddenMeshes[nextShow++ % hiddenMeshes.size()]->visible = true;
                        break;
                    default:
                        break;
                }
                for (int f = 0; f < kGap; ++f) {
                    rows.push_back(frame(f == 0 ? kind : -1, f));
                    if (dump && f < 2) {
                        const auto im = renderer.readRGBPixels();
                        if (im.size() >= static_cast<size_t>(kW) * kH * 3) {
                            char name[96];
                            std::snprintf(name, sizeof(name), "/churn_r%d_%02d_%s_f%d.ppm", round, kind,
                                          f == 0 ? "event" : "after", f);
                            std::string fname = name;
                            std::replace(fname.begin(), fname.end(), ' ', '_');
                            writePpm(std::string(dumpDir) + fname, im);
                        }
                    }
                }
            }
        }
        const double loopSec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const auto stats1 = renderer.tlasStats();

        // ── The table ───────────────────────────────────────────────────────
        std::vector<double> all, quiet;
        for (const auto& r : rows) {
            all.push_back(r.wallMs);
            if (r.kind < 0 && r.sinceEvent >= 3) quiet.push_back(r.wallMs);
        }
        const double quietMedian = percentile(quiet, 0.5);
        std::vector<double> quietCpu, quietGpu;
        for (const auto& r : rows) {
            if (r.kind < 0 && r.sinceEvent >= 3) {
                quietCpu.push_back(r.cpuFrameMs);
                quietGpu.push_back(r.gpuMs);
            }
        }
        std::printf("\nentries: forest %zu..%zu of %zu instances, %d cells of %d meshes, %d hidden meshes; "
                    "%zu frames in %.1f s\n",
                    instStart, instStart + kInstStep, kInstCapacity, kCellsAtStart, kMeshesPerCell, kHiddenMeshes,
                    rows.size(), loopSec);
        std::printf("frame wall ms: median %.2f  p99 %.2f  max %.2f   quiet median %.2f (cpuFrame %.2f, gpu %.2f)\n",
                    percentile(all, 0.5), percentile(all, 0.99), percentile(all, 1.0), quietMedian,
                    percentile(quietCpu, 0.5), percentile(quietGpu, 0.5));
        std::printf("TLAS: %llu full builds, %llu updates over the loop\n",
                    static_cast<unsigned long long>(stats1.fullRebuilds - stats0.fullRebuilds),
                    static_cast<unsigned long long>(stats1.updates - stats0.updates));
        std::printf("\n%-18s %3s %10s %9s %22s %12s %12s %8s\n", "event", "n", "structural", "history",
                    "event frame ms med/max", "ensure ms", "next ms med", "ratio");
        for (int kind = 0; kind < kKinds; ++kind) {
            std::vector<double> ev, evEnsure, next;
            uint64_t structural = 0, history = 0;
            int n = 0;
            for (size_t i = 0; i < rows.size(); ++i) {
                if (rows[i].kind != kind) continue;
                ++n;
                ev.push_back(rows[i].wallMs);
                evEnsure.push_back(rows[i].cpuEnsureMs);
                // What the event costs is charged to the event frame and the
                // one after it: a cost moved one frame later is not a fix.
                structural += rows[i].structural;
                history += rows[i].history;
                if (i + 1 < rows.size()) {
                    next.push_back(rows[i + 1].wallMs);
                    structural += rows[i + 1].structural;
                    history += rows[i + 1].history;
                }
            }
            const double evMax = std::max(percentile(ev, 1.0), percentile(next, 1.0));
            const double ratio = quietMedian > 0 ? evMax / quietMedian : 0.0;
            std::printf("%-18s %3d %10.2f %9.2f %11.2f / %8.2f %12.2f %12.2f %8.2f\n", kKindName[kind], n,
                        n ? static_cast<double>(structural) / n : 0.0, n ? static_cast<double>(history) / n : 0.0,
                        percentile(ev, 0.5), percentile(ev, 1.0), percentile(evEnsure, 0.5), percentile(next, 0.5),
                        ratio);

            if (kExpect[kind] >= 0) {
                check(std::string(kKindName[kind]) + ": " + std::to_string(kExpect[kind]) +
                              " structural rebuild(s) per event",
                      structural == static_cast<uint64_t>(kExpect[kind]) * static_cast<uint64_t>(n));
                if (kExpect[kind] == 0) {
                    check(std::string(kKindName[kind]) + ": event frame within " + std::to_string(kHitchRatio) +
                                  "x the quiet median",
                          evMax <= std::max(kHitchRatio * quietMedian, quietMedian + kHitchSlackMs));
                }
            }
            check(std::string(kKindName[kind]) + ": no temporal history reset", history == 0);
        }
        if (vulkan::validationActive()) {
            check("no validation errors", vulkan::validationErrorCount() == 0u);
        }
    } catch (const std::exception& e) {
        std::printf("  FAIL: the renderer threw: %s\n", e.what());
        ++failures;
    }

    std::printf("entry churn: %d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
