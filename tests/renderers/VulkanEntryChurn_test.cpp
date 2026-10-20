// VulkanEntryChurn_test: content streaming through the Vulkan renderer's entry
// list, the way terrain::CellStreamer and the pine forest's LODs drive it, and
// what each kind of change costs the frame that carries it.
//
// The renderer keeps one MeshEntry per mesh instance. Until 2026-10-19 any
// change to that list's SHAPE was a structural rebuild in ensureSceneBuilt: the
// device drained, every descriptor, draw record and TLAS instance re-derived
// (about 30 ms an event on the forest, 43-51 ms here at 33k entries). Now such
// a change is a slot edit (VulkanCoreSlots.cpp, plans/gpu-driven-instances.md
// stage 2a): rows of the entry index space written in place. This bench pinned
// the defect first and is the gate every later stage is measured against.
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
//                      (a BLAS admission: a slot edit whose BLAS builds are
//                      submitted ahead of the frame, never waited for)
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
// rebuilds one event of that kind costs (the baseline was one for every kind;
// now none). A kind expecting none must also, when kFrameAsserted says so,
// keep what an event costs (the slower of its frame and the frame after: a
// cost moved one frame later is no fix) within kHitchRatio of the quiet median
// over its events, and within kHitchMaxRatio on its worst one. Each ratio has
// an absolute floor: this bench's quiet frame is about a millisecond, where a
// ratio is noise, and the rebuild it replaces cost ten (8.9k entries) - the
// floors keep the gate sharp against that and blind to scheduler jitter. No
// history reset on any event, ever. Then a ParticleField (debris, dust, a
// wake) joins the scene, moving every frame, and every kind runs again around
// it: the field's arrival and departure may be rebuilds, the churn around a
// live field may not (until 2026-10-20 any field sent every slot edit back to
// the rebuild). Then the forest grows to its whole capacity at once, more
// entries than the slot pool the last rebuild laid out: the pool grows in place
// (no device wait, no rebuild), and a reservation (reserveEntrySlots) sizes it
// up front. GrassMesh fields (wind-swayed, each its own
// BLAS) are added, detached, re-attached, swapped, admitted hidden and dropped,
// each a slot edit. A last phase removes every mesh (no TLAS instance left),
// renders, and adds some back.
// THREEPP_VULKAN_NO_SLOTS=1 restores the rebuild, and with it the baseline:
// run with it, the bench expects one structural rebuild per event.
//
// Auto-LOD is off: its chains land asynchronously and their TLAS rebuilds
// would land on random frames of the table.
//
// THREEPP_CHURN_DUMP=<dir> writes the event frames and the frame after each as
// PPM there (churn_r* for the main rounds, churn_field_r* for the field's), for
// looking at and for comparing against a THREEPP_VULKAN_NO_SLOTS=1 run; that
// readback stalls, so its timings are void.
// THREEPP_CHURN_INSTANCES=<n> sets the forest's starting count (default 8192;
// its capacity is twice that). THREEPP_CHURN_VERBOSE=1 prints every event's
// frame and the frame after it. THREEPP_CHURN_DRAWGEN=1 turns on the GPU
// instance expansion, draw records and TLAS instances (stages 1, 2 and 5),
// for A/B runs. THREEPP_CHURN_MOVER=1 adds one small mesh that moves every
// frame (a car in the forest): every frame then refits the TLAS.
//
// Plain exit-code program, not Catch2, like the other Vulkan tests here.
// Exits 42 (CTest "Skipped") when no Vulkan GPU is available.

#include "threepp/canvas/Canvas.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/extras/vegetation/GrassTiles.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/ConeGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/GrassMesh.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/ParticleField.hpp"
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
#include <functional>
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
    constexpr int kFieldRounds = 2;      // every kind this many times around a live ParticleField
    constexpr uint32_t kFieldParticles = 256;
    constexpr double kHitchRatio = 2.0;     // median event against the quiet median
    constexpr double kHitchSlackMs = 2.5;   // ... or the quiet median + this, whichever is larger
    constexpr double kHitchMaxRatio = 4.0;  // the worst event
    constexpr double kHitchMaxSlackMs = 5.0;

    enum Kind { kRemoveCell, kAddCachedCell, kAddNewCell, kAddNewGeom, kInstGrow, kInstShrink, kFirstShow, kKinds };
    constexpr const char* kKindName[kKinds] = {"remove cell", "add cached cell", "add new cell", "add new geometry",
                                               "instanced grow", "instanced shrink", "first show"};
    // Structural rebuilds one event of the kind costs. The baseline before the
    // slots (081b3ead, 2026-10-19): one for every kind.
    constexpr int kExpect[kKinds] = {0, 0, 0, 0, 0, 0, 0};
    // Whether the event frame's time is asserted: every kind, now that new
    // geometry's BLAS builds are submitted without a host wait.
    constexpr bool kFrameAsserted[kKinds] = {true, true, true, true, true, true, true};

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

    // A terrain-tile-sized map of its own (512x512 noise over a colour): its
    // encode (mips + BC7) is real work, as a streamed tile's three maps are.
    std::shared_ptr<Texture> tileMap(uint32_t seed) {
        constexpr int N = 512;
        std::vector<unsigned char> data(static_cast<size_t>(N) * N * 4);
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                const uint32_t h = (static_cast<uint32_t>(x) * 73856093u) ^ (static_cast<uint32_t>(y) * 19349663u) ^
                                   (seed * 83492791u);
                unsigned char* p = &data[(static_cast<size_t>(y) * N + x) * 4];
                p[0] = static_cast<unsigned char>(90 + (h & 63u) + (seed * 37u) % 80u);
                p[1] = static_cast<unsigned char>(110 + ((h >> 6) & 63u));
                p[2] = static_cast<unsigned char>(60 + ((h >> 12) & 63u));
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
    const char* verboseEnv = std::getenv("THREEPP_CHURN_VERBOSE");
    const bool verbose = verboseEnv && *verboseEnv == '1';
    const char* noSlotsEnv = std::getenv("THREEPP_VULKAN_NO_SLOTS");
    const bool slotsOff = noSlotsEnv && *noSlotsEnv == '1';

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
    if (const char* e = std::getenv("THREEPP_CHURN_DRAWGEN"); e && *e == '1') {
        renderer.setGpuInstanceExpansion(true);
        renderer.setGpuDrawGeneration(true);
        renderer.setGpuTlasInstances(true);
        std::printf("GPU instance expansion + draw records + TLAS instances: on\n");
    }

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

    // One mesh moving every frame (THREEPP_CHURN_MOVER=1).
    const char* moverEnv = std::getenv("THREEPP_CHURN_MOVER");
    std::shared_ptr<Mesh> mover;
    if (moverEnv && *moverEnv == '1') {
        mover = Mesh::create(cellGeoms[0], cellMats[0]);
        mover->position.set(0.f, 1.f, 0.f);
        scene.add(mover);
    }

    float orbit = 0.f;
    const auto placeCamera = [&] {
        if (mover) mover->position.x = std::sin(orbit * 7.f) * 20.f;
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

    // The ParticleField phase's field, fed every frame it is in the scene: a
    // ring of boxes drifting around the scene's centre, a closed form of the
    // frame count so both lanes see the same positions.
    std::shared_ptr<ParticleField> field;
    std::vector<ParticlePos> fieldPos(kFieldParticles);
    uint32_t fieldTick = 0;
    const auto feedField = [&] {
        if (!field) return;
        const float t = 0.02f * static_cast<float>(fieldTick++);
        for (uint32_t i = 0; i < kFieldParticles; ++i) {
            const float a = 6.2831853f * static_cast<float>(i) / kFieldParticles + t;
            const float r = 18.f + 4.f * hash(i * 5u);
            fieldPos[i] = {std::cos(a) * r, 2.f + 1.5f * std::sin(3.f * a + t), std::sin(a) * r, 1.f};
        }
        field->submit(fieldPos.data(), kFieldParticles);
    };

    // The grass phase's fields, their wind clock a closed form of the frame
    // count (both lanes see the same poses).
    std::vector<std::shared_ptr<GrassMesh>> grassFields;
    uint32_t grassTick = 0;

    std::vector<FrameRow> rows;
    try {
        const auto frame = [&](int kind, int sinceEvent) {
            placeCamera();
            feedField();
            for (auto& g : grassFields) g->params.time = 0.05f * static_cast<float>(grassTick);
            ++grassTick;
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

        const auto applyEvent = [&](int kind) {
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
        };
        const auto dumpFrame = [&](const char* prefix, int round, int kind, int f) {
            const auto im = renderer.readRGBPixels();
            if (im.size() < static_cast<size_t>(kW) * kH * 3) return;
            char name[112];
            std::snprintf(name, sizeof(name), "/%s_r%d_%02d_%s_f%d.ppm", prefix, round, kind, f == 0 ? "event" : "after",
                          f);
            std::string fname = name;
            std::replace(fname.begin(), fname.end(), ' ', '_');
            writePpm(std::string(dumpDir) + fname, im);
        };

        const auto stats0 = renderer.tlasStats();
        const auto t0 = std::chrono::steady_clock::now();
        for (int round = 0; round < kRounds; ++round) {
            for (int kind = 0; kind < kKinds; ++kind) {
                applyEvent(kind);
                for (int f = 0; f < kGap; ++f) {
                    rows.push_back(frame(f == 0 ? kind : -1, f));
                    if (verbose && f < 2) {
                        const FrameRow& r = rows.back();
                        std::printf("  r%d %-17s f%d wall %6.2f ensure %6.2f cpu %6.2f gpu %5.2f structural %llu\n",
                                    round, kKindName[kind], f, r.wallMs, r.cpuEnsureMs, r.cpuFrameMs, r.gpuMs,
                                    static_cast<unsigned long long>(r.structural));
                    }
                    if (dump && f < 2) dumpFrame("churn", round, kind, f);
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
        std::printf("TLAS: %llu full builds, %llu updates over the loop; %llu slot edits, %llu structural rebuilds, "
                    "%llu BLASes compacted after their slot edit\n",
                    static_cast<unsigned long long>(stats1.fullRebuilds - stats0.fullRebuilds),
                    static_cast<unsigned long long>(stats1.updates - stats0.updates),
                    static_cast<unsigned long long>(stats1.slotEdits - stats0.slotEdits),
                    static_cast<unsigned long long>(stats1.structuralRebuilds - stats0.structuralRebuilds),
                    static_cast<unsigned long long>(stats1.asyncCompactions - stats0.asyncCompactions));
        // The geometries "add new geometry" brings are built by the slot edit
        // and compacted a frame or two later, with no wait (unless
        // THREEPP_VK_BLAS_COMPACT=0 turns compaction off).
        const char* compactEnv = std::getenv("THREEPP_VK_BLAS_COMPACT");
        if (!slotsOff && renderer.rayTracingSupported() && !(compactEnv && *compactEnv == '0'))
            check("the slot edits' new BLASes were compacted after their build",
                  stats1.asyncCompactions > stats0.asyncCompactions);
        std::printf("\n%-18s %3s %10s %9s %22s %12s %12s %8s\n", "event", "n", "structural", "history",
                    "event frame ms med/max", "ensure ms", "next ms med", "ratio");
        for (int kind = 0; kind < kKinds; ++kind) {
            std::vector<double> ev, evEnsure, next, cost;
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
                    cost.push_back(std::max(rows[i].wallMs, rows[i + 1].wallMs));
                } else {
                    cost.push_back(rows[i].wallMs);
                }
            }
            const double costMed = percentile(cost, 0.5), costMax = percentile(cost, 1.0);
            const double ratio = quietMedian > 0 ? costMed / quietMedian : 0.0;
            std::printf("%-18s %3d %10.2f %9.2f %11.2f / %8.2f %12.2f %12.2f %8.2f\n", kKindName[kind], n,
                        n ? static_cast<double>(structural) / n : 0.0, n ? static_cast<double>(history) / n : 0.0,
                        percentile(ev, 0.5), percentile(ev, 1.0), percentile(evEnsure, 0.5), percentile(next, 0.5),
                        ratio);

            const int expect = slotsOff ? 1 : kExpect[kind];
            if (expect >= 0) {
                check(std::string(kKindName[kind]) + ": " + std::to_string(expect) +
                              " structural rebuild(s) per event",
                      structural == static_cast<uint64_t>(expect) * static_cast<uint64_t>(n));
                if (expect == 0 && kFrameAsserted[kind]) {
                    char what[112];
                    std::snprintf(what, sizeof(what), "%s: events within %.0fx the quiet median (%.2f ms)",
                                  kKindName[kind], kHitchRatio, costMed);
                    check(what, costMed <= std::max(kHitchRatio * quietMedian, quietMedian + kHitchSlackMs));
                    std::snprintf(what, sizeof(what), "%s: worst event within %.0fx (%.2f ms)",
                                  kKindName[kind], kHitchMaxRatio, costMax);
                    check(what, costMax <= std::max(kHitchMaxRatio * quietMedian, quietMedian + kHitchMaxSlackMs));
                }
            }
            check(std::string(kKindName[kind]) + ": no temporal history reset", history == 0);
        }
        // ── The same churn around a live ParticleField ───────────────────────
        // A field publishes its entry index to its pass; its arrival and its
        // departure may each be a rebuild (THREEPP_VULKAN_SLOTS_LOG names them),
        // every event around it must be a slot edit like any other.
        {
            ParticleField::Config cfg;
            cfg.capacity = kFieldParticles;
            cfg.uniformRadius = 0.4f;
            cfg.orientations = true;
            field = ParticleField::create(cfg);
            field->setMeshRepr(BoxGeometry::create(0.8f, 0.8f, 0.8f), paint(Color(0.95f, 0.75f, 0.2f)));
            std::vector<float> quats(static_cast<size_t>(kFieldParticles) * 4u, 0.f);
            for (uint32_t i = 0; i < kFieldParticles; ++i) quats[static_cast<size_t>(i) * 4u + 3u] = 1.f;
            field->setOrientations(quats.data(), kFieldParticles);
            scene.add(field);
            for (int f = 0; f < kGap; ++f) frame(-1, kGap);// the field's arrival
            const auto f0 = renderer.tlasStats();
            int events = 0;
            std::vector<double> fieldQuiet, fieldCost;
            for (int round = 0; round < kFieldRounds; ++round) {
                for (int kind = 0; kind < kKinds; ++kind) {
                    applyEvent(kind);
                    ++events;
                    double eventMs = 0;
                    for (int f = 0; f < kGap; ++f) {
                        const FrameRow r = frame(f == 0 ? kind : -1, f);
                        if (f == 0) eventMs = r.wallMs;
                        if (f == 1) fieldCost.push_back(std::max(eventMs, r.wallMs));
                        if (f >= 3) fieldQuiet.push_back(r.wallMs);
                        if (dump && f < 2) dumpFrame("churn_field", round, kind, f);
                    }
                }
            }
            const auto f1 = renderer.tlasStats();
            const uint64_t structural = f1.structuralRebuilds - f0.structuralRebuilds;
            const double fq = percentile(fieldQuiet, 0.5), fMed = percentile(fieldCost, 0.5),
                         fMax = percentile(fieldCost, 1.0);
            std::printf("\naround a live ParticleField: %d events, %llu slot edits, %llu structural rebuilds, "
                        "%llu history resets; event ms med/max %.2f / %.2f, quiet median %.2f\n",
                        events, static_cast<unsigned long long>(f1.slotEdits - f0.slotEdits),
                        static_cast<unsigned long long>(structural),
                        static_cast<unsigned long long>(f1.historyResets - f0.historyResets), fMed, fMax, fq);
            const uint64_t expect = slotsOff ? static_cast<uint64_t>(events) : 0u;
            check("around a live ParticleField: " + std::to_string(expect) + " structural rebuilds",
                  structural == expect);
            if (!slotsOff) {
                check("around a live ParticleField: events within 2x the quiet median",
                      fMed <= std::max(kHitchRatio * fq, fq + kHitchSlackMs));
                check("around a live ParticleField: worst event within 4x",
                      fMax <= std::max(kHitchMaxRatio * fq, fq + kHitchMaxSlackMs));
            }
            check("around a live ParticleField: no temporal history reset", f1.historyResets == f0.historyResets);
            scene.remove(*field);
            for (int f = 0; f < kGap; ++f) frame(-1, kGap);// its departure
            field.reset();
        }

        // ── The slot pool outgrown ───────────────────────────────────────────
        // In place and, since 2026-10-20, with no device wait: the old buffers
        // go to the frames in flight's retire queue. A growth stays a hitch
        // (tens of MB of buffers made anew and the old ones freed), so a
        // reservation (reserveEntrySlots, a streamed world sizing its pool up
        // front) is taken by the next edit, and the forest at its whole
        // capacity again fits without another growth.
        {
            std::vector<double> quiet;
            for (int f = 0; f < 16; ++f) quiet.push_back(frame(-1, kGap).wallMs);
            const double q = percentile(quiet, 0.5);
            const auto g0 = renderer.tlasStats();
            const size_t countBefore = forest->count();
            forest->setCount(kInstCapacity);
            double growMs[3] = {};// the event frame and the two after it
            for (int f = 0; f < kGap; ++f) {
                const FrameRow r = frame(-1, f);
                if (f < 3) growMs[f] = r.wallMs;
                if (dump && f < 2) dumpFrame("churn_growth", 0, 0, f);
            }
            const auto g1 = renderer.tlasStats();
            std::printf("\nthe forest at its whole capacity (%zu -> %zu instances): %llu pool growths (pool %u -> %u), "
                        "%llu slot edits, %llu structural rebuilds, %llu history resets; frames %.2f, %.2f, %.2f ms "
                        "(quiet %.2f)\n",
                        countBefore, kInstCapacity,
                        static_cast<unsigned long long>(g1.slotPoolGrowths - g0.slotPoolGrowths), g0.slotCapacity,
                        g1.slotCapacity, static_cast<unsigned long long>(g1.slotEdits - g0.slotEdits),
                        static_cast<unsigned long long>(g1.structuralRebuilds - g0.structuralRebuilds),
                        static_cast<unsigned long long>(g1.historyResets - g0.historyResets), growMs[0], growMs[1],
                        growMs[2], q);
            if (!slotsOff) {
                check("the slot pool outgrown: it grows in place", g1.slotPoolGrowths > g0.slotPoolGrowths);
                check("the slot pool outgrown: no structural rebuild", g1.structuralRebuilds == g0.structuralRebuilds);
            }
            check("the slot pool outgrown: no temporal history reset", g1.historyResets == g0.historyResets);
            forest->setCount(countBefore);
            for (int f = 0; f < kGap; ++f) frame(-1, kGap);

            const uint32_t want = g1.slotCapacity + 20000u;
            renderer.reserveEntrySlots(want);
            auto extra = buildCell(nextFreshCell++ % kCells, false);
            scene.add(extra);
            for (int f = 0; f < kGap; ++f) frame(-1, kGap);
            const auto g2 = renderer.tlasStats();
            // The same forest event again, now with no growth in it: what the
            // growth adds is measured against this, not against a quiet frame
            // (the event writes 8192 rows either way).
            forest->setCount(kInstCapacity);
            double sameMs[3] = {};
            for (int f = 0; f < kGap; ++f) {
                const FrameRow r = frame(-1, f);
                if (f < 3) sameMs[f] = r.wallMs;
            }
            const auto g3 = renderer.tlasStats();
            const double growWorst = std::max({growMs[0], growMs[1], growMs[2]});
            const double sameWorst = std::max({sameMs[0], sameMs[1], sameMs[2]});
            std::printf("a reservation of %u entries: pool %u after the next edit (%llu growths), %u with the forest "
                        "at its whole capacity again (%llu growths; frames %.2f, %.2f, %.2f ms against %.2f, %.2f, "
                        "%.2f with the growth)\n",
                        want, g2.slotCapacity, static_cast<unsigned long long>(g2.slotPoolGrowths - g1.slotPoolGrowths),
                        g3.slotCapacity, static_cast<unsigned long long>(g3.slotPoolGrowths - g2.slotPoolGrowths),
                        sameMs[0], sameMs[1], sameMs[2], growMs[0], growMs[1], growMs[2]);
            if (!slotsOff) {
                check("a reservation: the next edit grows the pool to it", g2.slotCapacity >= want);
                check("a reservation: what fits in it grows nothing", g3.slotPoolGrowths == g2.slotPoolGrowths);
                // A growth is tens of MB of buffers made anew: bounded against
                // the same event without one, not against a quiet frame.
                check("the slot pool outgrown: the growth's worst frame within 2x the same event's + 5 ms",
                      growWorst <= 2.0 * sameWorst + 5.0);
            }
            forest->setCount(countBefore);
            scene.remove(*extra);
            for (int f = 0; f < kGap; ++f) frame(-1, kGap);
        }

        // ── A cell admitted hidden, shown later ──────────────────────────────
        // Meshes that arrive hidden (Object3D::resident; a CellStreamer
        // prefetch band attaches cells so) with geometries and terrain-tile-
        // sized maps never seen: their slot edit only places them, their
        // admission (BLAS, the maps' encode on worker threads, the uploads)
        // lands over the next frames while they stay hidden, and their show
        // is a flip. THREEPP_VULKAN_SYNC_ADMIT=1 admits them on the add frame
        // (the lane as it was); both lanes render the same frames, so the
        // dumped frame after the show (THREEPP_CHURN_DUMP, churn_ahead_*)
        // compares pixel for pixel across them.
        {
            const char* syncEnv = std::getenv("THREEPP_VULKAN_SYNC_ADMIT");
            const bool syncAdmit = syncEnv && *syncEnv == '1';
            for (int f = 0; f < kGap; ++f) frame(-1, kGap);
            std::vector<double> quiet;
            for (int f = 0; f < 16; ++f) quiet.push_back(frame(-1, kGap).wallMs);
            const double q = percentile(quiet, 0.5);
            const auto a0 = renderer.tlasStats();
            constexpr int kAhead = 12;
            auto cell = Group::create();
            for (int k = 0; k < kAhead; ++k) {
                auto mat = paint(Color(1, 1, 1));
                mat->map = tileMap(static_cast<uint32_t>(k) + 1u);
                auto m = Mesh::create(SphereGeometry::create(1.2f, 40 + k, 20), mat);
                m->position.set(-55.f + static_cast<float>(k) * 9.f, 2.5f, 20.f);
                m->resident = true;
                cell->add(m);
            }
            cell->visible = false;
            scene.add(cell);
            const auto admittedCount = [&] {
                int n = 0;
                for (const auto& c : cell->children) n += renderer.admitted(*c) ? 1 : 0;
                return n;
            };
            const double addMs = frame(-1, 0).wallMs;
            const int admittedAtAdd = admittedCount();
            const double afterAddMs = frame(-1, 1).wallMs;
            std::vector<double> between;// the frames the admission lands on
            for (int f = 0; f < 180; ++f) between.push_back(frame(-1, kGap).wallMs);
            const int admittedBeforeShow = admittedCount();
            const auto a1 = renderer.tlasStats();
            cell->visible = true;
            const double showMs = frame(-1, 0).wallMs;
            const double afterShowMs = frame(-1, 1).wallMs;
            for (int f = 0; f < 30; ++f) frame(-1, kGap);
            if (dump) dumpFrame("churn_ahead", 0, 0, 30);
            const auto a2 = renderer.tlasStats();
            const double betweenMax = percentile(between, 1.0);
            std::printf("\na cell of %d never-seen meshes with 512x512 maps, added hidden and shown 182 frames later "
                        "(%s): %llu admissions deferred, %llu landed before the show, %llu structural rebuilds, "
                        "%llu history resets; quiet %.2f ms, add frame %.2f / %.2f, frames between max %.2f, "
                        "show frame %.2f / %.2f\n",
                        kAhead, syncAdmit ? "synchronous admission" : "admitted over the frames after",
                        static_cast<unsigned long long>(a1.admissionsDeferred - a0.admissionsDeferred),
                        static_cast<unsigned long long>(a1.admissionsLanded - a0.admissionsLanded),
                        static_cast<unsigned long long>(a2.structuralRebuilds - a0.structuralRebuilds),
                        static_cast<unsigned long long>(a2.historyResets - a0.historyResets), q, addMs, afterAddMs,
                        betweenMax, showMs, afterShowMs);
            if (!slotsOff) {
                check("a cell admitted hidden: no structural rebuild", a2.structuralRebuilds == a0.structuralRebuilds);
                if (!syncAdmit) {
                    check("a cell admitted hidden: every mesh's admission deferred",
                          a1.admissionsDeferred - a0.admissionsDeferred == static_cast<uint64_t>(kAhead));
                    check("a cell admitted hidden: every admission landed before the show",
                          a1.admissionsLanded - a0.admissionsLanded == static_cast<uint64_t>(kAhead));
                    std::printf("admitted(): %d of %d meshes after the add frame, %d before the show\n",
                                admittedAtAdd, kAhead, admittedBeforeShow);
                    check("a cell admitted hidden: admitted() false for each mesh after its add frame",
                          admittedAtAdd == 0);
                    check("a cell admitted hidden: admitted() true for each mesh before its show",
                          admittedBeforeShow == kAhead);
                    check("a cell admitted hidden: the add frame within 4x the quiet median (one event)",
                          std::max(addMs, afterAddMs) <= std::max(kHitchMaxRatio * q, q + kHitchMaxSlackMs));
                    check("a cell admitted hidden: the frames it lands on within 4x",
                          betweenMax <= std::max(kHitchMaxRatio * q, q + kHitchMaxSlackMs));
                }
                check("a cell admitted hidden: the show within 4x the quiet median (one event)",
                      std::max(showMs, afterShowMs) <= std::max(kHitchMaxRatio * q, q + kHitchMaxSlackMs));
            }
            check("a cell admitted hidden: no temporal history reset", a2.historyResets == a0.historyResets);
            scene.remove(*cell);
            for (int f = 0; f < kGap; ++f) frame(-1, kGap);
        }

        // ── Swaps on meshes that stay ────────────────────────────────────────
        // A geometry, a material and a texture swapped on meshes of a live
        // cell, each kind twice (domain randomisation, a damaged prop): until
        // 2026-10-20 each was a structural rebuild; now the slot edit places
        // the mesh again with what it holds. THREEPP_CHURN_DUMP writes the
        // frame after each swap (churn_swap_*) for comparing the lanes.
        {
            for (int f = 0; f < kGap; ++f) frame(-1, kGap);
            std::vector<double> quiet;
            for (int f = 0; f < 16; ++f) quiet.push_back(frame(-1, kGap).wallMs);
            const double q = percentile(quiet, 0.5);
            const auto w0 = renderer.tlasStats();
            auto& cellGroup = active.front();
            std::vector<Mesh*> targets;
            std::vector<std::shared_ptr<BufferGeometry>> origGeom;
            for (size_t c = 0; c < cellGroup->children.size(); c += 3) {
                if (auto* mm = cellGroup->children[c]->as<Mesh>()) {
                    targets.push_back(mm);
                    origGeom.push_back(mm->geometry());
                }
            }
            const auto otherGeom = SphereGeometry::create(0.7f, 20, 10);
            // Materials of their own, textured: the material swaps put one on,
            // the texture swaps then change its map.
            std::array<std::shared_ptr<MeshStandardMaterial>, 2> swapMats = {paint(Color(1, 1, 1)),
                                                                             paint(Color(0.9f, 0.85f, 0.6f))};
            swapMats[0]->map = checker();
            swapMats[1]->map = checker();
            int swaps = 0;
            std::vector<double> swapCost;
            const char* kSwapName[3] = {"geometry", "material", "texture"};
            for (int round = 0; round < 2; ++round) {
                for (int kind = 0; kind < 3; ++kind) {
                    if (kind == 2) {// one map change, on the material every target holds
                        swapMats[round]->map = round == 0 ? tileMap(7u) : tileMap(9u);
                        swapMats[round]->needsUpdate();
                    }
                    for (size_t t = 0; t < targets.size(); ++t) {
                        Mesh& m = *targets[t];
                        if (kind == 0) m.setGeometry(round == 0 ? otherGeom : origGeom[t]);
                        if (kind == 1) m.setMaterial(swapMats[round]);
                        ++swaps;
                    }
                    const double ev = frame(-1, 0).wallMs;
                    swapCost.push_back(std::max(ev, frame(-1, 1).wallMs));
                    if (dump) dumpFrame("churn_swap", round, kind, 1);
                    for (int f = 2; f < kGap; ++f) frame(-1, kGap);
                    std::printf("%s swap, round %d: %.2f ms (quiet %.2f)\n", kSwapName[kind], round,
                                swapCost.back(), q);
                }
            }
            const auto w1 = renderer.tlasStats();
            std::printf("\nswaps on meshes that stay: %d swaps in 6 events, %llu taken as slot edits, %llu structural "
                        "rebuilds, %llu history resets; event ms med/max %.2f / %.2f, quiet median %.2f\n",
                        swaps, static_cast<unsigned long long>(w1.swapEdits - w0.swapEdits),
                        static_cast<unsigned long long>(w1.structuralRebuilds - w0.structuralRebuilds),
                        static_cast<unsigned long long>(w1.historyResets - w0.historyResets),
                        percentile(swapCost, 0.5), percentile(swapCost, 1.0), q);
            if (!slotsOff) {
                check("swaps: no structural rebuild", w1.structuralRebuilds == w0.structuralRebuilds);
                check("swaps: each a slot edit", w1.swapEdits - w0.swapEdits >= static_cast<uint64_t>(swaps));
            } else {
                check("swaps: a structural rebuild each (the rebuild lane)",
                      w1.structuralRebuilds - w0.structuralRebuilds == 6u);
            }
            check("swaps: no temporal history reset", w1.historyResets == w0.historyResets);
        }

        // ── Grass fields streamed ────────────────────────────────────────────
        // GrassMesh fields (one merged mesh of blades the wind compute pass
        // sways, its own BLAS refit every frame) come and go the way a
        // streamed heath's cells do: fields added shown, one detached and
        // attached again, one whose geometry is swapped (the old one kept
        // alive) and swapped back, one added hidden and shown later, and all
        // of them dropped for good. Until 2026-10-20 each was a structural
        // rebuild (a deformer's state was the rebuild's to build and to free);
        // now each is a slot edit: a field's record and inputs are built in the
        // edit's batch, or over the frames after for the hidden one, and its
        // first wind pose is recorded in the frame. THREEPP_CHURN_DUMP writes
        // the frame after each event (churn_grass_*) for comparing against the
        // rebuild lane: a field that did not sway, or swayed from the wrong
        // pose, shows there.
        {
            const char* syncEnv = std::getenv("THREEPP_VULKAN_SYNC_ADMIT");
            const bool syncAdmit = syncEnv && *syncEnv == '1';
            for (int f = 0; f < kGap; ++f) frame(-1, kGap);
            std::vector<double> quiet;
            for (int f = 0; f < 16; ++f) quiet.push_back(frame(-1, kGap).wallMs);
            const double q = percentile(quiet, 0.5);
            const auto r0 = renderer.tlasStats();
            const uint64_t mem0 = renderer.gpuAllocatedBytes();
            auto grassMat = paint(Color(1, 1, 1));
            grassMat->vertexColors = true;
            grassMat->side = Side::Double;
            const auto fieldGeometry = [](float cx, float cz, uint32_t seed) {
                constexpr int kBlades = 2400;
                std::vector<vegetation::GrassBlade> blades;
                blades.reserve(kBlades);
                for (int b = 0; b < kBlades; ++b) {
                    const uint32_t s = seed * 7919u + static_cast<uint32_t>(b);
                    vegetation::GrassBlade bl;
                    bl.position.set(cx + 4.f * hash(s * 4u), 0.f, cz + 4.f * hash(s * 4u + 1u));
                    bl.scale.set(1.5f, 0.7f + 0.35f * hash(s * 4u + 2u), 1.5f);
                    bl.yaw.setFromAxisAngle(Vector3(0, 1, 0), math::PI * hash(s * 4u + 3u));
                    blades.push_back(bl);
                }
                return vegetation::buildGrassGeometry(blades);
            };
            const auto makeField = [&](float cx, float cz, uint32_t seed) {
                auto g = GrassMesh::create(fieldGeometry(cx, cz, seed), grassMat);
                g->params.windStrength = 0.35f;
                grassFields.push_back(g);
                return g;
            };
            int events = 0;
            std::vector<double> cost;
            const auto settle = [&](const char* what, const std::function<void()>& afterEvent = {}) {
                const double ev = frame(-1, 0).wallMs;
                if (afterEvent) afterEvent();
                if (dump) dumpFrame("churn_grass", events, 0, 0);
                cost.push_back(std::max(ev, frame(-1, 1).wallMs));
                if (dump) dumpFrame("churn_grass", events, 0, 1);
                if (verbose) std::printf("  grass: %-34s %.2f ms\n", what, cost.back());
                for (int f = 2; f < kGap; ++f) {
                    frame(-1, kGap);
                    if (dump && f == kGap - 1) dumpFrame("churn_grass", events, 0, f);
                }
                ++events;
            };
            std::vector<std::shared_ptr<GrassMesh>> live;
            for (int k = 0; k < 4; ++k) {
                live.push_back(makeField(-18.f + 12.f * static_cast<float>(k), -4.f, 11u + static_cast<uint32_t>(k)));
                scene.add(live.back());
                settle("a new field");
            }
            scene.remove(*live[0]);
            settle("a field detached");
            scene.add(live[0]);
            settle("the field attached again");
            const auto keepGeom = live[1]->geometry();
            live[1]->setGeometry(fieldGeometry(-6.f, -4.f, 97u));
            settle("a field's geometry swapped");
            live[1]->setGeometry(keepGeom);
            settle("and swapped back");
            // A hidden field takes no deform and no BLAS refit; shown again it
            // takes one on its first frame (a prime: that frame's wind, no
            // motion from the pose it was left in).
            const auto deformsOver = [&](int n) {
                const uint64_t a = renderer.dynamicGeomStats().grassDeforms;
                for (int f = 0; f < n; ++f) frame(-1, kGap);
                return renderer.dynamicGeomStats().grassDeforms - a;
            };
            const uint64_t allShown = deformsOver(4);
            live[2]->visible = false;
            live[3]->visible = false;
            frame(-1, kGap);// the hide
            const uint64_t twoHidden = deformsOver(4);
            live[2]->visible = true;
            live[3]->visible = true;
            uint64_t atShow = 0;
            {
                const uint64_t a = renderer.dynamicGeomStats().grassDeforms;
                settle("two hidden fields shown again",
                       [&] { atShow = renderer.dynamicGeomStats().grassDeforms - a; });
            }
            std::printf("grass deforms: %llu in 4 frames with 4 fields shown, %llu with 2 of them hidden, %llu on "
                        "the frame they show again\n",
                        static_cast<unsigned long long>(allShown), static_cast<unsigned long long>(twoHidden),
                        static_cast<unsigned long long>(atShow));
            const auto a0 = renderer.tlasStats();
            auto ahead = makeField(0.f, 8.f, 41u);
            ahead->visible = false;
            ahead->resident = true;
            scene.add(ahead);
            bool aheadAdmittedAtAdd = true;
            settle("a field added hidden", [&] { aheadAdmittedAtAdd = renderer.admitted(*ahead); });
            for (int f = 0; f < 24; ++f) frame(-1, kGap);
            const bool aheadAdmittedBeforeShow = renderer.admitted(*ahead);
            const auto a1 = renderer.tlasStats();
            ahead->visible = true;
            settle("the hidden field shown");
            for (auto& g : live) scene.remove(*g);
            scene.remove(*ahead);
            live.clear();
            ahead.reset();
            grassFields.clear();// the last references: the fields and their geometries die
            settle("every field dropped");
            for (int f = 0; f < 8; ++f) frame(-1, kGap);// past the frames in flight: the retired go
            const auto r1 = renderer.tlasStats();
            const uint64_t mem1 = renderer.gpuAllocatedBytes();
            constexpr double MiB = 1024.0 * 1024.0;
            std::printf("\ngrass fields streamed: %d events, %llu structural rebuilds, %llu history resets, %llu "
                        "records pruned; event ms med/max %.2f / %.2f, quiet median %.2f; device memory %.1f MiB "
                        "before, %.1f after\n",
                        events, static_cast<unsigned long long>(r1.structuralRebuilds - r0.structuralRebuilds),
                        static_cast<unsigned long long>(r1.historyResets - r0.historyResets),
                        static_cast<unsigned long long>(r1.recordsPruned - r0.recordsPruned), percentile(cost, 0.5),
                        percentile(cost, 1.0), q, mem0 / MiB, mem1 / MiB);
            if (!slotsOff) {
                check("grass fields: no structural rebuild", r1.structuralRebuilds == r0.structuralRebuilds);
                if (!syncAdmit) {
                    check("grass fields: the hidden one admitted over the frames after, landed before its show",
                          a1.admissionsDeferred - a0.admissionsDeferred == 1u &&
                                  a1.admissionsLanded - a0.admissionsLanded == 1u);
                    // A field has no maps to encode: the admissions that land at the
                    // end of a lean frame take it on its add frame already.
                    std::printf("admitted(): the hidden field %s after its add frame, %s before its show\n",
                                aheadAdmittedAtAdd ? "yes" : "no", aheadAdmittedBeforeShow ? "yes" : "no");
                    check("grass fields: admitted() true for the hidden field before its show",
                          aheadAdmittedBeforeShow);
                }
                check("grass fields: every field's state freed with it", r1.recordsPruned - r0.recordsPruned >= 5u);
                check("grass fields: the worst event within 4x the quiet median",
                      percentile(cost, 1.0) <= std::max(kHitchMaxRatio * q, q + kHitchMaxSlackMs));
            }
            check("grass fields: a shown field deforms every frame", allShown == 16u);
            check("grass fields: a hidden field takes no deform", twoHidden == 8u);
            check("grass fields: shown again, each takes one on its first frame", atShow == 4u);
            check("grass fields: device memory back within 8 MiB of where it started",
                  mem1 <= mem0 + static_cast<uint64_t>(8 * MiB));
            check("grass fields: no temporal history reset", r1.historyResets == r0.historyResets);
        }

        // ── A streamed world frees what it drops ─────────────────────────────
        // Tiles made of geometries and textures nobody else holds, added and
        // dropped again and again, the way a terrain quadtree streams: their
        // BLAS records and bindless texture slots must go with them, without
        // a rebuild. Until 2026-10-20 they stayed until the next structural
        // rebuild, which the slot path never makes: a kilometre GeoScene
        // flight ran out of the 2048 texture slots and drew new tiles white.
        // 100 tiles of 24 meshes, each with a texture of its own, cycle 2400
        // textures through the 2048 slots; device memory must come back.
        {
            for (int f = 0; f < kGap; ++f) frame(-1, kGap);
            const auto p0 = renderer.tlasStats();
            const uint64_t mem0 = renderer.gpuAllocatedBytes();
            uint64_t memPeak = mem0;
            constexpr int kTiles = 100, kTileMeshes = 24;
            for (int t = 0; t < kTiles; ++t) {
                auto tile = Group::create();
                for (int k = 0; k < kTileMeshes; ++k) {
                    auto mat = paint(Color(1, 1, 1));
                    mat->map = checker();
                    auto m = Mesh::create(SphereGeometry::create(0.8f, 32, 16), mat);
                    m->position.set(-70.f + static_cast<float>(k) * 6.f, 2.f, -60.f + static_cast<float>(t % 5) * 4.f);
                    tile->add(m);
                }
                scene.add(tile);
                for (int f = 0; f < 2; ++f) frame(-1, kGap);
                memPeak = std::max(memPeak, renderer.gpuAllocatedBytes());
                scene.remove(*tile);
                tile.reset();// the last reference: its geometries, materials and textures die
                for (int f = 0; f < 2; ++f) frame(-1, kGap);
            }
            for (int f = 0; f < 8; ++f) frame(-1, kGap);// past the frames in flight: the retired go
            const auto p1 = renderer.tlasStats();
            const uint64_t mem1 = renderer.gpuAllocatedBytes();
            constexpr double MiB = 1024.0 * 1024.0;
            std::printf("\na streamed world, %d tiles of %d meshes with their own geometry and texture: "
                        "%llu records and %llu textures pruned, %llu structural rebuilds, %llu history resets; "
                        "device memory %.1f MiB before, %.1f at the peak, %.1f after\n",
                        kTiles, kTileMeshes, static_cast<unsigned long long>(p1.recordsPruned - p0.recordsPruned),
                        static_cast<unsigned long long>(p1.texturesPruned - p0.texturesPruned),
                        static_cast<unsigned long long>(p1.structuralRebuilds - p0.structuralRebuilds),
                        static_cast<unsigned long long>(p1.historyResets - p0.historyResets), mem0 / MiB,
                        memPeak / MiB, mem1 / MiB);
            if (!slotsOff) {
                check("a streamed world: no structural rebuild", p1.structuralRebuilds == p0.structuralRebuilds);
                check("a streamed world: every tile's records pruned",
                      p1.recordsPruned - p0.recordsPruned >= static_cast<uint64_t>(kTiles * kTileMeshes));
                check("a streamed world: every tile's textures pruned",
                      p1.texturesPruned - p0.texturesPruned >= static_cast<uint64_t>(kTiles * kTileMeshes));
            }
            // What stays is pool slack (staging, scratch, the slot pool), not
            // a tile's worth per tile.
            check("a streamed world: device memory back within 8 MiB of where it started",
                  mem1 <= mem0 + static_cast<uint64_t>(8 * MiB));
            std::printf("texture slots in use %u before, %u after; %llu textures found no slot\n",
                        p0.textureSlotsInUse, p1.textureSlotsInUse,
                        static_cast<unsigned long long>(p1.textureSlotFallbacks - p0.textureSlotFallbacks));
            check("a streamed world: no texture found the slots exhausted",
                  p1.textureSlotFallbacks == p0.textureSlotFallbacks);
            // The rebuild lane frees the last tile's textures at its next rebuild
            // (its prune runs while the old rows still hold them).
            if (!slotsOff)
                check("a streamed world: texture slots in use back to where they started",
                      p1.textureSlotsInUse <= p0.textureSlotsInUse);
            check("a streamed world: no temporal history reset", p1.historyResets == p0.historyResets);
        }

        // ── Everything removed, then some of it back ─────────────────────────
        // No TLAS instance at all: the TLAS must be rebuilt empty, never left
        // with the removed instances, whose desc rows are zeroed (a ray that
        // hits one reads its geometry at address 0 and loses the device).
        {
            const auto s0 = renderer.tlasStats();
            scene.remove(*ground);
            scene.remove(*forest);
            for (auto& g : active) scene.remove(*g);
            for (auto& m : hiddenMeshes) scene.remove(*m);
            if (mover) scene.remove(*mover);
            for (int f = 0; f < 6; ++f) frame(-1, kGap);
            const auto s1 = renderer.tlasStats();
            std::printf("\nevery mesh removed: %llu TLAS instances, %llu structural rebuilds\n",
                        static_cast<unsigned long long>(s1.instances),
                        static_cast<unsigned long long>(s1.structuralRebuilds - s0.structuralRebuilds));
            if (renderer.rayTracingSupported())
                check("every mesh removed: the TLAS holds no instance", s1.instances == 0u);
            scene.add(ground);
            scene.add(forest);
            scene.add(active.front());
            for (int f = 0; f < 6; ++f) frame(-1, kGap);
            const auto s2 = renderer.tlasStats();
            std::printf("ground, forest and a cell back: %llu TLAS instances, %llu structural rebuilds\n",
                        static_cast<unsigned long long>(s2.instances),
                        static_cast<unsigned long long>(s2.structuralRebuilds - s0.structuralRebuilds));
            if (renderer.rayTracingSupported())
                check("added back: the TLAS holds the ground, the forest and the cell",
                      s2.instances == 1u + forest->count() + static_cast<uint32_t>(kMeshesPerCell));
            if (!slotsOff)
                check("removing everything and adding back: no structural rebuild",
                      s2.structuralRebuilds == s0.structuralRebuilds);
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
