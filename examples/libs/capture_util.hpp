// capture_util.hpp — shared, committed machinery for the headless capture /
// profile / diff loop used by the Vulkan examples and the gitignored scratch
// harnesses (see scratch/README.md). The split is deliberate: this is the
// durable *capability*; the per-debug scene/pose/flags live in scratch.
//
// Four things, none of which should be re-implemented per demo:
//   • parseArgs()           — camera pose + frames + output overridable from the
//                             CLI, so reframing a capture needs NO rebuild.
//   • Shot/finishShot()     — the "warm up N frames, write one PNG, exit"
//                             recorder every headless capture is built on.
//   • writeFrameTimings()   — dump VulkanRenderer::lastFrameTimings() as JSON
//                             (--profile) so per-pass cost is measurable headless
//                             / in CI instead of eyeballed.
//   • imageDiff()/diffStats — MSE/PSNR + banded delta stats for objective
//                             before/after and convergence checks.
//
// The window/camera resize boilerplate lives next door in window_util.hpp — it
// is not a capture concern, and most demos want it without wanting this.
//
// Header-only and dependency-free (just threepp + the stdlib). Lives in
// examples/libs (on every example target's include path), so ANY demo — GL or
// Vulkan — can `#include "capture_util.hpp"` with no build-system changes; the
// Vulkan-only pieces (frame timings) are guarded by THREEPP_WITH_VULKAN.
#pragma once

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Vector3.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace capture {

    // ── CLI parsing ─────────────────────────────────────────────────────────
    // All optional; unknown args are ignored so a demo keeps its own flags.
    //   --cam x,y,z   --look x,y,z   --frames N   --out name.png   --profile [path]
    struct Args {
        std::optional<threepp::Vector3> camPos;     // --cam:  eye position
        std::optional<threepp::Vector3> camTarget;  // --look: look-at target
        std::optional<int>              frames;     // --frames
        std::optional<std::string>      shot;       // --shot: capture name -> aaa_caps/
        std::optional<std::string>      out;        // --out
        bool                            profile = false;// --profile (dump timings)
        std::string                     profilePath;    // optional JSON-lines file
    };

    inline std::optional<threepp::Vector3> parseVec3(const char* s) {
        float v[3];
        if (std::sscanf(s, "%f,%f,%f", &v[0], &v[1], &v[2]) == 3 ||
            std::sscanf(s, "%f %f %f", &v[0], &v[1], &v[2]) == 3)
            return threepp::Vector3{v[0], v[1], v[2]};
        return std::nullopt;
    }

    inline Args parseArgs(int argc, char** argv) {
        Args a;
        for (int i = 1; i < argc; ++i) {
            auto val = [&](const char* flag) -> const char* {
                return (std::strcmp(argv[i], flag) == 0 && i + 1 < argc) ? argv[++i] : nullptr;
            };
            if (const char* s = val("--cam")) a.camPos = parseVec3(s);
            else if (const char* s = val("--look")) a.camTarget = parseVec3(s);
            else if (const char* s = val("--frames")) a.frames = std::atoi(s);
            else if (const char* s = val("--shot")) a.shot = std::string(s);
            else if (const char* s = val("--out")) a.out = std::string(s);
            else if (std::strcmp(argv[i], "--profile") == 0) {
                a.profile = true;
                if (i + 1 < argc && argv[i + 1][0] != '-') a.profilePath = argv[++i];
            }
        }
        return a;
    }

    // ── The --shot warm-up counter ────────────────────────────────────────────
    // A headless capture renders N frames so the path tracer / TAA can converge,
    // then writes one PNG and exits (see finishShot below). Every demo grew its
    // own `shotPath` + `shotFrames` + `shotFrame` triple to drive that; Shot IS
    // that triple, so the loop tail is one line and the default warm-up is stated
    // in exactly one place.
    struct Shot {
        // Enough for the path tracer and TAA to converge on a typical scene; a
        // demo that settles slower passes its own default to the constructor.
        static constexpr int kDefaultFrames = 240;

        std::string name;  // --shot <name.png>; empty = interactive run
        int         frames = kDefaultFrames;// --frames N: warm-up before the write
        int         frame = 0;              // frames rendered so far

        Shot() = default;
        explicit Shot(const Args& a, int defaultFrames = kDefaultFrames)
            : name(a.shot.value_or(std::string{})),
              frames(a.frames.value_or(defaultFrames)) {}

        // Headless capture requested? (Demos gate their ImGui panel on this: UI
        // must not be drawn into the measured frames.)
        [[nodiscard]] bool active() const { return !name.empty(); }

        // Count one rendered frame; true once the warm-up is complete, i.e. this
        // framebuffer is the one to write. Stays true afterwards — harmless,
        // because the only sensible response is the [[noreturn]] finishShot():
        //     if (shot.ready()) capture::finishShot(renderer, shot.name);
        bool ready() { return active() && ++frame >= frames; }

        // True on the single frame at `fraction` through the warm-up — for
        // mid-capture state changes ("switch the wind halfway", "toggle to
        // night"), which is how the live-update paths get exercised headlessly.
        // Read it BEFORE ready() in the frame body; ready() is what advances.
        [[nodiscard]] bool at(float fraction) const {
            return active() && frame == int(float(frames) * fraction);
        }
    };

#ifdef THREEPP_WITH_VULKAN
    // ── Frame-timing dump ─────────────────────────────────────────────────────
    inline std::string frameTimingsJson(const threepp::VulkanRenderer::FrameTimings& t) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "{\"pathTraceMs\":%.3f,\"denoiseMs\":%.3f,"
            "\"taaMs\":%.3f,\"rasterGbufMs\":%.3f,\"overlayMs\":%.3f,"
            "\"cpuEnsureSceneMs\":%.3f,\"cpuRecordMs\":%.3f,\"cpuFrameMs\":%.3f}",
            t.pathTraceMs, t.denoiseMs, t.taaMs, t.rasterGbufMs,
            t.overlayMs, t.cpuEnsureSceneMs, t.cpuRecordMs, t.cpuFrameMs);
        return std::string(buf);
    }

    // Prints one JSON object to stdout (tagged with the frame index) and, if a
    // path was given, appends it — a multi-frame run yields a JSON-lines profile.
    inline void writeFrameTimings(const threepp::VulkanRenderer::FrameTimings& t,
                                  const Args& a, int frame) {
        if (!a.profile) return;
        const std::string js = frameTimingsJson(t);
        std::printf("[profile] frame %d %s\n", frame, js.c_str());
        std::fflush(stdout);
        if (!a.profilePath.empty()) {
            std::ofstream f(a.profilePath, std::ios::app);
            f << js << "\n";
        }
    }
#endif// THREEPP_WITH_VULKAN

#ifdef PROJECT_FOLDER
    // ── The --shot tail ──────────────────────────────────────────────────────
    // Every demo ends a headless capture the same way: resolve the --shot name
    // into the repo's aaa_caps/ directory, write the framebuffer, log the path
    // (plus any per-demo stats) and exit. These two helpers ARE that tail —
    // it was previously copy-pasted across seven demos, drifting on whether
    // the directory got created first.

    inline std::filesystem::path shotOutputPath(const std::string& name) {
        const auto p = std::filesystem::path(PROJECT_FOLDER) / "aaa_caps" / name;
        std::filesystem::create_directories(p.parent_path());
        return p;
    }

    // A sequence run (--seqn / --seqframes) writes consecutive frames as
    // <stem>_000.png, _001.png, … beside where finishShot() would put
    // <stem>.png. Built on shotOutputPath() so "where captures go" stays a
    // single fact: the demos each hand-rolled this <stem>_%03d splice, and a
    // demo that spells the aaa_caps/ path itself is one that can drift from it.
    inline std::filesystem::path shotSequencePath(const std::string& name, int index) {
        const std::filesystem::path p(name);
        char suffix[16];
        std::snprintf(suffix, sizeof suffix, "_%03d", index);
        return shotOutputPath(
                (p.parent_path() / (p.stem().string() + suffix + p.extension().string())).string());
    }

    // Write one frame of a sequence; returns the path it went to. Deliberately
    // silent: callers differ on whether they log per frame or one summary line
    // at the end, and that is a presentation choice, not shared machinery.
    template<class Renderer>
    std::filesystem::path writeShotSequenceFrame(Renderer& renderer,
                                                 const std::string& name, int index) {
        const auto p = shotSequencePath(name, index);
        renderer.writeFramebuffer(p);
        return p;
    }

    // `stats` is appended verbatim to the "wrote <path>" line — build it with
    // an ostringstream so float formatting matches operator<<.
    template<class Renderer>
    [[noreturn]] inline void finishShot(Renderer& renderer, const std::string& name,
                                        const std::string& stats = {}) {
        const auto p = shotOutputPath(name);
        renderer.writeFramebuffer(p);
        std::cout << "wrote " << p.string() << stats << std::endl;
        std::exit(0);
    }
#endif// PROJECT_FOLDER

    // finishShot()'s sibling for demos whose --shot is a PATH written exactly as
    // given, rather than a name resolved into the repo's aaa_caps/ directory.
    // The detector demos work that way (they take an output filename beside their
    // input image), so they must NOT go through finishShot — the two are kept
    // adjacent here so the choice is visible instead of re-derived per demo.
    template<class Renderer>
    [[noreturn]] inline void finishShotAtPath(Renderer& renderer, const std::string& path) {
        renderer.writeFramebuffer(path);
        std::cout << "wrote " << path << std::endl;
        std::exit(0);
    }

    // ── The scripted-orbit sequence (--seq) ──────────────────────────────────
    // Some defects are invisible in a still and only appear while the camera
    // moves — view-anchored quantisation is the whole class. The evidence has to
    // be a SEQUENCE: warm-up frames to bring every temporal history (TAA,
    // froxel EMA) to its steady state, then N consecutive frames written out.
    //
    // The camera path is closed-form in the FRAME INDEX, not in wall-clock time.
    // That is the property that makes the output evidence: two runs of the same
    // command produce the same poses, so frames can be diffed across builds.
    struct OrbitSequence {
        std::string dir;             // output directory (created if absent)
        int         frames = 8;      // consecutive frames written
        int         warm = 100;      // frames rendered before the first write
        float       orbitDegPerSec = 22.f;// azimuth rate; 0 holds the pose
        float       dt = 1.f / 60.f; // fixed step — a capture has no wall clock
    };

    // Orbits `camera` about `target` at its current radius and height, calling
    // `step(t)` to advance the effect to sequence time t and `renderFrame()` to
    // draw. Writes dir/f00.png, f01.png, … once the warm-up is done.
    template<class Renderer, class Camera, class Step, class RenderFrame>
    void runOrbitSequence(const OrbitSequence& seq, Renderer& renderer, Camera& camera,
                          const threepp::Vector3& target, Step&& step, RenderFrame&& renderFrame) {
        std::filesystem::create_directories(seq.dir);

        const threepp::Vector3 eye0 = camera.position;
        const float r0 = std::hypot(eye0.x - target.x, eye0.z - target.z);
        const float a0 = std::atan2(eye0.z - target.z, eye0.x - target.x);
        const float rate = seq.orbitDegPerSec * threepp::math::DEG2RAD;// rad/s

        for (int i = 0; i < seq.warm + seq.frames; ++i) {
            const float t = float(i) * seq.dt;
            const float a = a0 + rate * t;
            camera.position.set(target.x + r0 * std::cos(a), eye0.y,
                                target.z + r0 * std::sin(a));
            camera.lookAt(target);
            step(t);
            renderFrame();
            if (i >= seq.warm) {
                char name[16];
                std::snprintf(name, sizeof name, "f%02d.png", i - seq.warm);
                renderer.writeFramebuffer(std::filesystem::path(seq.dir) / name);
            }
        }
    }


    // ── Image diff (objective before/after + convergence) ─────────────────────
    struct DiffResult {
        double mse = 0.0;
        double psnr = 0.0;// dB; 99 when identical
        int    maxD = 0;
        double hotPct = 0.0;// fraction of channels differing by > thresh
    };

    // 8-bit images of equal size (any channel count).
    inline DiffResult imageDiff(const std::vector<unsigned char>& a,
                                const std::vector<unsigned char>& b, int thresh = 4) {
        DiffResult r;
        if (a.empty() || a.size() != b.size()) return r;
        double sumSq = 0.0;
        long long hot = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const int d = std::abs(int(a[i]) - int(b[i]));
            sumSq += double(d) * double(d);
            if (d > thresh) ++hot;
            r.maxD = std::max(r.maxD, d);
        }
        r.mse = sumSq / double(a.size());
        r.psnr = r.mse > 0.0 ? 10.0 * std::log10(255.0 * 255.0 / r.mse) : 99.0;
        r.hotPct = 100.0 * double(hot) / double(a.size());
        return r;
    }

    // Banded delta stats (promoted from the scratch harness): 3 horizontal bands
    // so *where* the difference lives is visible (top = sky/far … bot = near).
    // Used for A-vs-B and consecutive-frame settling/convergence checks.
    inline void diffStats(const std::vector<unsigned char>& a,
                          const std::vector<unsigned char>& b,
                          int w, int h, int channels, const char* tag) {
        if (a.size() != b.size() || a.size() != size_t(w) * size_t(h) * size_t(channels)) {
            std::printf("[%s] size mismatch\n", tag);
            return;
        }
        static const char* bands[3] = {"top", "mid", "bot"};
        for (int band = 0; band < 3; ++band) {
            const int y0 = h * band / 3, y1 = h * (band + 1) / 3;
            double sum = 0.0;
            long long hot = 0, n = 0;
            int mx = 0;
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < w; ++x) {
                    const size_t i = (size_t(y) * size_t(w) + size_t(x)) * size_t(channels);
                    int d = 0;
                    for (int c = 0; c < channels; ++c)
                        d = std::max(d, std::abs(int(a[i + c]) - int(b[i + c])));
                    sum += d;
                    if (d > 4) ++hot;
                    mx = std::max(mx, d);
                    ++n;
                }
            std::printf("[%s] %s meanD=%6.3f maxD=%3d hot=%6.3f%%\n",
                        tag, bands[band], n ? sum / double(n) : 0.0, mx,
                        n ? 100.0 * double(hot) / double(n) : 0.0);
        }
    }

}// namespace capture
