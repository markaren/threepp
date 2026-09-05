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
//     vulkan_aov_audit --fsr --out a_fsr.txt       # the rgb.fsr row instead of rgb
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
    // --dumprgb <prefix>: write raw rgb bytes of frames 2..9 to
    // <prefix>_f<N>.raw so two runs can be diffed spatially — WHERE pixels
    // differ says what class of pass diverged (edges = reprojection, scattered
    // singles = ray order, whole-frame LSB = a blend/exposure factor).
    std::string dumpPrefix;
    // --taasplit: additionally hash the TAA INPUT image (the shade→bloom→post
    // product) and the written HISTORY slot per frame, as manifest rows
    // taa.input / taa.history with per-frame traces on stdout. Splits "the
    // shading diverged" from "the temporal resolve diverged".
    bool taaSplit = false;
    // --hdrsplit: additionally hash the linear-HDR scene image (bloom's
    // sceneHdr — what the shade/denoise chain wrote, BEFORE bloom and post
    // touch it) as manifest row shade.hdr with a per-frame trace. Pairs with
    // --no-denoise to split the shade dispatch from the denoiser chain.
    bool hdrSplit = false;
    // --shadesplit: per-frame hash of each deferred-shade temporal image
    // (indirect / momentsSq / reflect / reflAux / shadowVis / directU) via
    // debugHashShadeImages. The first name to differ between two runs is the
    // pass the divergence enters at; directU (no rays, no history) is the
    // control that indicts the dispatch itself if it moves.
    bool shadeSplit = false;
    // Divergence-bisection toggles: each turns off one pass group suspected of
    // carrying run-varying state into the frame. The AOV rows are already
    // proven exact, so whatever breaks rgb replay enters downstream of the
    // G-buffer — find the minimal configuration that replays, then re-enable
    // one group at a time.
    bool noDenoise = false; // setDenoise(false): the inline deterministic shading path
    bool noRestir = false;  // setRestirDIEnabled(false): no reservoir feedback
    bool noOccl = false;    // setOcclusionCulling(false)
    bool noLod = false;     // setAutoLod(false)
    bool hardSun = false;   // setSunAngularRadius(0): no shadow-ray cone jitter
    bool staticScene = false;// --static: no scripted motion → no BLAS refit /
                             // TLAS update. Discriminates acceleration-structure
                             // rebuild nondeterminism from everything else.
    bool noProbes = false;   // --no-probes: setProbeGI(false). The falsification
                             // test for "probe_update is the carrier": with the
                             // atlas out of the chain, rgb must replay bit-exact.
    // --dumpprobes <prefix>: raw probeSh dump per frame (frames 0..5) for
    // byte-level forensics — which probe index, which SH band, what magnitude.
    std::string probeDumpPrefix;
    // --scene-edit: the lidar audit's entry-list churn, applied to the AOV
    // rows. A mesh is added at frame 45 and removed at frame 81, and at those
    // same frames an existing MID-LIST object (the spinner) is pulled out and
    // later re-added at the tail, so the post-revert scene holds the same
    // objects in a different order. The Ids AOV must not care: it carries the
    // stable per-object id, not the entry index.
    bool sceneEdit = false;
    int editAddFrame = 45, editRemoveFrame = 81;
    // --fsr: lift exactly one pin, the FSR 3.1 upscaler, in place of the
    // in-house TAA. The frame row is then emitted as `rgb.fsr`, its own
    // finding: the pip wheel turns FSR on by default wherever it has it, so
    // whether the shipped default replays is a question the matrix must
    // answer, but never by folding it into the TAA row's claim.
    bool fsr = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--frames" && i + 1 < argc) frames = std::atoi(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) outPath = argv[++i];
        else if (arg == "--scene-edit") sceneEdit = true;
        else if (arg == "--fsr") fsr = true;
        else if (arg == "--edit-add" && i + 1 < argc) editAddFrame = std::atoi(argv[++i]);
        else if (arg == "--edit-remove" && i + 1 < argc) editRemoveFrame = std::atoi(argv[++i]);
        else if (arg == "--rgbtrace" && i + 1 < argc) rgbTracePath = argv[++i];
        else if (arg == "--dumprgb" && i + 1 < argc) dumpPrefix = argv[++i];
        else if (arg == "--taasplit") taaSplit = true;
        else if (arg == "--hdrsplit") hdrSplit = true;
        else if (arg == "--shadesplit") shadeSplit = true;
        else if (arg == "--no-denoise") noDenoise = true;
        else if (arg == "--no-restir") noRestir = true;
        else if (arg == "--no-occl") noOccl = true;
        else if (arg == "--no-lod") noLod = true;
        else if (arg == "--hard-sun") hardSun = true;
        else if (arg == "--static") staticScene = true;
        else if (arg == "--no-probes") noProbes = true;
        else if (arg == "--dumpprobes" && i + 1 < argc) probeDumpPrefix = argv[++i];
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
    // the same reason. --fsr is the one deliberate exception, reported under
    // its own row name.
    renderer.setSceneCaptureEnabled(true);
    renderer.setDlss(false);
    renderer.setFsr(fsr);
    // fsr() reports the ACTIVE upscaler, which is decided at the first frame;
    // availability (compiled in, context created) is what can be checked here.
    if (fsr && !renderer.fsrAvailable()) {
        std::cout << "FSR requested but unavailable on this build/GPU\n";
        return 2;
    }
    if (noDenoise) renderer.setDenoise(false);
    if (noRestir) renderer.setRestirDIEnabled(false);
    if (noOccl) renderer.setOcclusionCulling(false);
    if (noLod) renderer.setAutoLod(false);
    if (hardSun) renderer.setSunAngularRadius(0.f);
    if (noProbes) renderer.setProbeGI(false);

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

    // A CPU deformer: a grid whose position+normal attributes are rewritten
    // and needsUpdate()ed EVERY frame, so it graduates to the per-frame
    // dynamic-geometry path (staging upload + frame-cb BLAS refit + the
    // vertex→prevVertex motion snapshot). Without this row the certificate
    // silently excludes the refit machinery the flock and CPU trails run on;
    // the dyn.geom manifest row below asserts the path actually engaged.
    // The wave is a pure function of the scripted clock — never wall time.
    auto waver = Mesh::create(PlaneGeometry::create(4.f, 4.f, 20, 20),
                              MeshStandardMaterial::create(
                                      MeshStandardMaterial::Params{}.color(Color(0xb03a48))));
    waver->rotation.x = -math::PI / 2;
    waver->position.set(-0.5f, 0.6f, 3.0f);
    scene.add(waver);
    renderer.setObjectInstanceId(*waver, 5);
    renderer.setObjectClassId(*waver, 4);
    auto* wavPos = waver->geometry()->getAttribute<float>("position");
    auto* wavNrm = waver->geometry()->getAttribute<float>("normal");
    wavPos->setUsage(DrawUsage::Dynamic);
    wavNrm->setUsage(DrawUsage::Dynamic);
    const std::vector<float> wavRest = wavPos->array();// rest-pose x,y copy
    auto deform = [&](double t) {
        auto& p = wavPos->array();
        auto& n = wavNrm->array();
        for (std::size_t v = 0; v < p.size() / 3; ++v) {
            const float x = wavRest[3 * v + 0];
            const float y = wavRest[3 * v + 1];
            // Local z wave with analytic partials, so the normals are exact
            // rather than re-derived from the mesh (cheaper, and one fewer
            // spot for cross-run arithmetic to hide in).
            const float ax = 2.0f * x + static_cast<float>(t) * 3.0f;
            const float ay = 2.0f * y + static_cast<float>(t) * 2.0f;
            const float z = 0.25f * std::sin(ax) * std::cos(ay);
            const float dzdx = 0.50f * std::cos(ax) * std::cos(ay);
            const float dzdy = -0.50f * std::sin(ax) * std::sin(ay);
            p[3 * v + 2] = z;
            const float inv = 1.f / std::sqrt(dzdx * dzdx + dzdy * dzdy + 1.f);
            n[3 * v + 0] = -dzdx * inv;
            n[3 * v + 1] = -dzdy * inv;
            n[3 * v + 2] = inv;
        }
        wavPos->needsUpdate();
        wavNrm->needsUpdate();
    };

    // The edit subject (see --scene-edit above). Built up front so its
    // geometry upload is not itself the event.
    auto editBox = Mesh::create(BoxGeometry::create(1.f, 1.f, 1.f),
                                MeshStandardMaterial::create(
                                        MeshStandardMaterial::Params{}.color(Color(0xd0c060))));
    editBox->position.set(3.5f, 1.0f, 3.5f);

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
    Stream taaIn, taaHist, shadeHdr;
    std::vector<std::uint8_t> taaInBuf, taaHistBuf, hdrBuf;
    int failures = 0;

    for (int f = 0; f < frames; ++f) {
        const double t = f * kDt;
        // The deterministic frame clock: with it, the TAA blend weights and
        // every other formerly-wall-clock input advance on this scripted time.
        renderer.setSimTime(t);
        if (!staticScene) {
            mover->position.set(static_cast<float>(2.0 * std::sin(t * 1.3)), 1.f,
                                static_cast<float>(1.5 * std::cos(t * 0.9)));
            mover->rotation.y = static_cast<float>(t * 1.7);
            spinner->rotation.x = static_cast<float>(t * 2.3);
            deform(t);
        }
        if (sceneEdit) {
            if (f == editAddFrame) {
                scene.add(editBox);
                renderer.setObjectInstanceId(*editBox, 6);
                renderer.setObjectClassId(*editBox, 2);
                scene.remove(*spinner);
            } else if (f == editRemoveFrame) {
                scene.remove(*editBox);
                scene.add(spinner);
                renderer.setObjectInstanceId(*spinner, 3);
                renderer.setObjectClassId(*spinner, 2);
            }
        }

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
            if (taaSplit) {
                int iw = 0, ih = 0, hw = 0, hh = 0;
                if (renderer.readTaaDebugImages(taaInBuf, iw, ih, taaHistBuf, hw, hh)) {
                    Fnv i1, h1;
                    i1.bytes(taaInBuf.data(), taaInBuf.size());
                    h1.bytes(taaHistBuf.data(), taaHistBuf.size());
                    std::cout << "taasplit f" << f << " in=" << std::hex << i1.value()
                              << " hist=" << h1.value() << std::dec << "\n";
                    taaIn.hash.bytes(taaInBuf.data(), taaInBuf.size());
                    taaIn.bytes += taaInBuf.size();
                    ++taaIn.frames;
                    taaHist.hash.bytes(taaHistBuf.data(), taaHistBuf.size());
                    taaHist.bytes += taaHistBuf.size();
                    ++taaHist.frames;
                }
            }
            if (hdrSplit) {
                int hw = 0, hh = 0;
                if (renderer.readSceneHdrDebug(hdrBuf, hw, hh)) {
                    Fnv h1;
                    h1.bytes(hdrBuf.data(), hdrBuf.size());
                    std::cout << "hdrsplit f" << f << " hdr=" << std::hex << h1.value()
                              << std::dec << "\n";
                    shadeHdr.hash.bytes(hdrBuf.data(), hdrBuf.size());
                    shadeHdr.bytes += hdrBuf.size();
                    ++shadeHdr.frames;
                }
            }
            if (shadeSplit) {
                for (const auto& [nm, hsh] : renderer.debugHashShadeImages()) {
                    std::cout << "shadesplit f" << f << " " << nm << "=" << std::hex << hsh
                              << std::dec << "\n";
                }
            }
            if (!probeDumpPrefix.empty() && f <= 5) {
                std::vector<std::uint8_t> shRaw;
                if (renderer.readProbeShDebug(shRaw)) {
                    std::ofstream pf(probeDumpPrefix + "_f" + std::to_string(f) + ".raw",
                                     std::ios::binary);
                    pf.write(reinterpret_cast<const char*>(shRaw.data()),
                             static_cast<std::streamsize>(shRaw.size()));
                }
            }
            if (!dumpPrefix.empty() && f >= 2 && f <= 9) {
                std::ofstream df(dumpPrefix + "_f" + std::to_string(f) + ".raw",
                                 std::ios::binary);
                df.write(reinterpret_cast<const char*>(pixels.data()),
                         static_cast<std::streamsize>(pixels.size()));
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
    emit(fsr ? "rgb.fsr" : "rgb", rgb);
    if (taaSplit) {
        emit("taa.input", taaIn);
        emit("taa.history", taaHist);
    }
    if (hdrSplit) emit("shade.hdr", shadeHdr);

    // The graduated-path proof: a manifest row both runs must agree on, and a
    // hard failure if the deformer never graduated — a certificate that reads
    // "deterministic" while the refit path sat idle would be a false claim.
    const auto dyn = renderer.dynamicGeomStats();
    manifest << "dyn.geom graduated=" << dyn.graduated
             << " refits=" << dyn.refitsRecorded
             << " rebuilds=" << dyn.fullRebuilds << "\n";
    // Which experiment this was: two runs that disagree here were not running
    // the same script, and the compare must say so rather than diff pixels.
    const auto tl = renderer.tlasStats();
    manifest << "script frames=" << frames << " sceneEdit=" << (sceneEdit ? 1 : 0)
             << " editAdd=" << (sceneEdit ? editAddFrame : -1)
             << " editRemove=" << (sceneEdit ? editRemoveFrame : -1)
             << " static=" << (staticScene ? 1 : 0)
             << " resolve=" << (fsr ? "fsr" : "taa")
             << " tlasRebuilds=" << tl.fullRebuilds << " tlasInstances=" << tl.instances << "\n";
    if (!staticScene && dyn.graduated == 0) {
        std::cout << "DYNAMIC-GEOM PATH NEVER ENGAGED (deformer failed to graduate)\n";
        ++failures;
    }

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
