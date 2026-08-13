// AOV replay audit — part 2 of the determinism go/no-go (part 1: the
// proprioceptive replay_audit under examples/extras/sensors).
//
// Renders a fixed scene with scripted motion for N frames on the Vulkan
// deferred renderer (headless canvas, auto-exposure pinned, vsync off) and
// folds every byte of every G-buffer AOV readback into one FNV-1a chain per
// AOV. Two fresh processes must produce identical manifests for the
// ground-truth label claim ("lossless depth, stable ids, normals, motion")
// to be *measured* rather than assumed:
//
//     vulkan_aov_audit --frames 120 --out a.txt
//     vulkan_aov_audit --frames 120 --out b.txt
//     vulkan_aov_audit --compare a.txt b.txt      # exit 0 = bit-identical
//
// The G-buffer AOVs are raster-prepass products, so they are expected to be
// bit-exact per device — unlike the RT-fed beauty frame. The `rgb` row hashes
// the post-composite colour output too, deliberately: GI/ReSTIR are stochastic
// per frame *index* but seeded, so whether the full frame also replays is a
// question worth an answer per commit, and a DIFF on that row alone (AOVs OK)
// is itself a publishable data point, not a failure of this gate.

#include "threepp/threepp.hpp"

#include "threepp/renderers/VulkanRenderer.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // FNV-1a 64, chained: order is part of the contract.
    class Fnv {
    public:
        void bytes(const void* p, std::size_t n) {
            const auto* b = static_cast<const unsigned char*>(p);
            for (std::size_t i = 0; i < n; ++i) {
                h_ ^= b[i];
                h_ *= 0x100000001B3ULL;
            }
        }
        [[nodiscard]] std::uint64_t value() const { return h_; }

    private:
        std::uint64_t h_ = 0xCBF29CE484222325ULL;
    };

    struct Stream {
        Fnv hash;
        std::uint64_t frames = 0;
        std::uint64_t bytes = 0;
    };

    int compare(const std::string& pathA, const std::string& pathB) {
        auto load = [](const std::string& path) {
            std::map<std::string, std::string> rows;
            std::ifstream f(path);
            if (!f) throw std::runtime_error("cannot open " + path);
            std::string name, rest;
            while (f >> name && std::getline(f, rest)) rows[name] = rest;
            return rows;
        };
        const auto a = load(pathA);
        const auto b = load(pathB);
        int bad = 0;
        for (const auto& [name, row] : a) {
            const auto it = b.find(name);
            if (it == b.end()) {
                std::cout << "MISSING  " << name << " (only in " << pathA << ")\n";
                ++bad;
            } else if (it->second != row) {
                std::cout << "DIFF     " << name << "\n  " << pathA << ":" << row
                          << "\n  " << pathB << ":" << it->second << "\n";
                ++bad;
            } else {
                std::cout << "OK       " << name << row << "\n";
            }
        }
        for (const auto& [name, row] : b) {
            if (!a.count(name)) {
                std::cout << "MISSING  " << name << " (only in " << pathB << ")\n";
                ++bad;
            }
        }
        std::cout << (bad ? "NO-GO: " : "GO: ") << (a.size() - bad) << "/" << a.size()
                  << " streams bit-identical\n";
        return bad ? 1 : 0;
    }

}// namespace

int main(int argc, char** argv) {

    int frames = 120;
    std::string outPath;
    // Per-frame rgb hash trace. Diagnostic for a DIFF on the rgb row: if run
    // B's frame k hashes equal run A's frame k±1, the divergence is capture
    // timing (a stale-frame readback), not pixel content; if individual frames
    // differ at the same index, the renderer itself diverged there.
    std::string rgbTracePath;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--frames" && i + 1 < argc) frames = std::atoi(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) outPath = argv[++i];
        else if (arg == "--rgbtrace" && i + 1 < argc) rgbTracePath = argv[++i];
        else if (arg == "--compare" && i + 2 < argc) return compare(argv[i + 1], argv[i + 2]);
    }

    Canvas canvas("vulkan_aov_audit",
                  {{"vsync", false}, {"size", WindowSize{800, 600}}, {"headless", true}});
    VulkanRenderer renderer(canvas);
    // Pinned: auto-exposure adapts across frames from image statistics, which
    // makes the colour row depend on history length — exactly the kind of
    // confound this harness exists to exclude.
    renderer.setAutoExposure(false);
    // The rgb row reads the scene-only capture path (post-TAA, pre-overlay) —
    // the same pixels a CameraSensor consumes. Pin the temporal resolve to the
    // in-house TAA: DLSS (and FSR) are third-party black boxes we cannot make
    // determinism claims about, and DLSS measurably breaks fresh-process
    // rgb replay while the AOV rows stay bit-exact. The goldens pin TAA for
    // the same reason.
    renderer.setSceneCaptureEnabled(true);
    renderer.setDlss(false);
    renderer.setFsr(false);

    Scene scene;
    scene.background = Color(0x304050);

    auto sun = DirectionalLight::create(Color(0xffffff), 3.f);
    sun->position.set(20.f, 30.f, 15.f);
    scene.add(sun);

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.1f, 100.f);
    camera->position.set(0.f, 3.f, 9.f);
    camera->lookAt(Vector3(0.f, 1.f, 0.f));

    auto ground = Mesh::create(BoxGeometry::create(30.f, 0.5f, 30.f),
                               MeshStandardMaterial::create(
                                       MeshStandardMaterial::Params{}.color(Color(0x556b45))));
    ground->position.y = -0.25f;
    scene.add(ground);
    renderer.setObjectInstanceId(*ground, 1);
    renderer.setObjectClassId(*ground, 1);

    // One mover (translates + rotates on the frame clock: exercises Motion and
    // per-frame TLAS/G-buffer updates), one spinner, one static occluder.
    auto mover = Mesh::create(BoxGeometry::create(1.f, 1.f, 1.f),
                              MeshStandardMaterial::create(
                                      MeshStandardMaterial::Params{}.color(Color(0xc8783c))));
    scene.add(mover);
    renderer.setObjectInstanceId(*mover, 2);
    renderer.setObjectClassId(*mover, 2);

    auto spinner = Mesh::create(SphereGeometry::create(0.7f, 32, 16),
                                MeshStandardMaterial::create(
                                        MeshStandardMaterial::Params{}.color(Color(0x3c78c8))));
    spinner->position.set(-2.5f, 1.f, 0.f);
    scene.add(spinner);
    renderer.setObjectInstanceId(*spinner, 3);
    renderer.setObjectClassId(*spinner, 2);

    auto pillar = Mesh::create(BoxGeometry::create(0.8f, 4.f, 0.8f),
                               MeshStandardMaterial::create(
                                       MeshStandardMaterial::Params{}.color(Color(0x808890))));
    pillar->position.set(2.5f, 2.f, -1.f);
    scene.add(pillar);
    renderer.setObjectInstanceId(*pillar, 4);
    renderer.setObjectClassId(*pillar, 3);

    // ---- render + hash ------------------------------------------------------

    struct AovRow {
        VulkanRenderer::GBufferAOV aov;
        const char* name;
        Stream stream;
    };
    AovRow rows[] = {
            {VulkanRenderer::GBufferAOV::Depth, "aov.depth", {}},
            {VulkanRenderer::GBufferAOV::Normal, "aov.normal", {}},
            {VulkanRenderer::GBufferAOV::Motion, "aov.motion", {}},
            {VulkanRenderer::GBufferAOV::Ids, "aov.ids", {}},
            {VulkanRenderer::GBufferAOV::Albedo, "aov.albedo", {}},
    };
    Stream rgb;

    constexpr double kDt = 1.0 / 60.0;// scripted clock — never wall time
    std::vector<std::uint8_t> buf;
    std::ostringstream rgbTrace;
    int failures = 0;

    for (int f = 0; f < frames; ++f) {
        const double t = f * kDt;
        mover->position.set(static_cast<float>(2.0 * std::sin(t * 1.3)), 1.f,
                            static_cast<float>(1.5 * std::cos(t * 0.9)));
        mover->rotation.y = static_cast<float>(t * 1.7);
        spinner->rotation.x = static_cast<float>(t * 2.3);

        renderer.render(scene, *camera);

        for (auto& row : rows) {
            int w = 0, h = 0, bpp = 0;
            if (renderer.readGBufferAOV(row.aov, buf, w, h, bpp)) {
                row.stream.hash.bytes(buf.data(), buf.size());
                row.stream.bytes += buf.size();
                ++row.stream.frames;
            } else if (f > 0) {
                // The first frame legitimately has nothing to read yet
                // (readGBufferAOV documents it); anything later is a failure.
                ++failures;
            }
        }
        const auto pixels = renderer.readSceneRGBPixels();
        if (!pixels.empty()) {
            rgb.hash.bytes(pixels.data(), pixels.size());
            rgb.bytes += pixels.size();
            ++rgb.frames;
            if (!rgbTracePath.empty()) {
                Fnv one;
                one.bytes(pixels.data(), pixels.size());
                rgbTrace << "f" << f << " " << std::hex << one.value() << std::dec << "\n";
            }
        }
    }
    if (!rgbTracePath.empty()) {
        std::ofstream tf(rgbTracePath);
        tf << rgbTrace.str();
    }

    std::ostringstream manifest;
    auto emit = [&](const char* name, const Stream& s) {
        manifest << name << " frames=" << s.frames << " bytes=" << s.bytes
                 << " fnv=" << std::hex << s.hash.value() << std::dec << "\n";
    };
    for (auto& row : rows) emit(row.name, row.stream);
    emit("rgb", rgb);

    std::cout << "vulkan_aov_audit: " << frames << " frames";
    if (failures) {
        std::cout << ", READBACK FAILURES: " << failures;
    } else {
        std::cout << ", readbacks clean";
    }
    std::cout << "\n" << manifest.str();
    if (!outPath.empty()) {
        std::ofstream out(outPath);
        out << manifest.str();
        std::cout << "wrote " << outPath << "\n";
    }
    return failures ? 2 : 0;
}
