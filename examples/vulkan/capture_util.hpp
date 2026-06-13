// capture_util.hpp — shared, committed machinery for the headless capture /
// profile / diff loop used by the Vulkan examples and the gitignored scratch
// harnesses (see scratch/README.md). The split is deliberate: this is the
// durable *capability*; the per-debug scene/pose/flags live in scratch.
//
// Three things, none of which should be re-implemented per demo:
//   • parseArgs()           — camera pose + frames + output overridable from the
//                             CLI, so reframing a capture needs NO rebuild.
//   • writeFrameTimings()   — dump VulkanRenderer::lastFrameTimings() as JSON
//                             (--profile) so per-pass cost is measurable headless
//                             / in CI instead of eyeballed.
//   • imageDiff()/diffStats — MSE/PSNR + banded delta stats for objective
//                             before/after and convergence checks.
//
// Header-only and dependency-free (just threepp + the stdlib) so any scratch
// file can `#include "../capture_util.hpp"` with no build-system changes.
#pragma once

#include "threepp/math/Vector3.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
            else if (const char* s = val("--out")) a.out = std::string(s);
            else if (std::strcmp(argv[i], "--profile") == 0) {
                a.profile = true;
                if (i + 1 < argc && argv[i + 1][0] != '-') a.profilePath = argv[++i];
            }
        }
        return a;
    }

    // ── Frame-timing dump ─────────────────────────────────────────────────────
    inline std::string frameTimingsJson(const threepp::VulkanRenderer::FrameTimings& t) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "{\"photonEmitMs\":%.3f,\"pathTraceMs\":%.3f,\"denoiseMs\":%.3f,"
            "\"taaMs\":%.3f,\"rasterGbufMs\":%.3f,\"overlayMs\":%.3f,"
            "\"cpuEnsureSceneMs\":%.3f,\"cpuRecordMs\":%.3f,\"cpuFrameMs\":%.3f}",
            t.photonEmitMs, t.pathTraceMs, t.denoiseMs, t.taaMs, t.rasterGbufMs,
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
