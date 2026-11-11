// ocean_metrics: a headless audit that measures the FFT ocean against
// published field measurements, so "the sea is physical" is a number with a
// source instead of a claim.
//
// It builds an Ocean (no terrain, no boat) for one wind speed and fetch, steps
// it on a pinned renderer clock (setSimTime, as rg_ab does), reads the wave
// fields back to the CPU every frame and prints one table row per metric:
// measured, target, tolerance, PASS / FAIL / INFO. With --json each metric is
// also appended as one JSON object per line.
//
//   ocean_metrics                         the sweep: U10 = 7, 10, 15 m/s at
//                                         fetch 0 (fully developed) and 20 km,
//                                         then two wakes (30 m at 10 kn, 4 m
//                                         at 15 kn) and the engine's foam
//                                         interval, each in its own process
//   ocean_metrics --wind 10 --fetch 0     one case
//   ocean_metrics --json metrics.jsonl    also append JSON lines to this file
//   ocean_metrics --wake 5.14 --hull 30   one wake case: a hull of 30 m at
//                                         5.14 m/s on a calm sea
//   ocean_metrics --foam-default          the foam interval the engine runs
//                                         under the current environment
//
//   ocean_metrics --compare base.jsonl    run, then print each metric's change
//                                         against a saved run
//   ocean_metrics --compare base.jsonl --against new.jsonl
//                                         compare two saved runs, no GPU
//
// Comparing another water branch against dev
//   1. Bring the audit to the branch: it lives in examples/vulkan/audits and
//      needs readOceanFoam and readWakeRipples (VulkanRenderer); cherry-pick
//      this audit's commits if the branch lacks them, then build the
//      ocean_metrics target in that branch's own build tree.
//   2. Check the whitecap mask first. The whitecap area is graded through
//      FoamShade below, a CPU copy of dev's water-shade foam mask
//      (deferred_shade_50_water_glass.glsl, the foam block; sampleFoamBicubic;
//      generateFoamDetailTile_). A branch that changes that mask, the detail
//      tile or how foam_world.comp fills the accumulator must mirror its
//      change into FoamShade before running, or the whitecap rows measure
//      dev's mask on the branch's foam and mean nothing.
//   3. Run the sweep with dev's saved run as the baseline:
//        ocean_metrics --json branch.jsonl --compare dev_baseline_v3.jsonl
//      (about 6 minutes; one case is --wind 15 --fetch 0 with the same flags).
//      Rows print base, now, change, the choppiness of both runs and the
//      verdicts; '*' marks a verdict that changed.
//   4. Read the chop column. '!' marks a row whose choppiness differs from
//      the baseline's: the dispersion, slope, crest and whitecap rows depend
//      on it, so rerun with --chop at the baseline's value to split the
//      branch's change from its change of default.
//   The wave-model switches are read from the environment, as the engine
//   reads them (THREEPP_OCEAN_CWM,
//   THREEPP_OCEAN_BOUND_WAVES, THREEPP_OCEAN_BREAKING_CRESTS = 1 / 0), so a
//   sweep's child processes inherit them. Under THREEPP_OCEAN_BREAKING_CRESTS
//   the whitecap row grades the breaking-crest mask (see section 5), not the
//   accumulator. Under the crest foam (with
//   the chop normals, which it needs) it grades the crest-foam
//   texture itself, read back with VulkanRenderer::readOceanCrestFoam.
//   5. MARGINAL means the estimators of that row fall on both sides of a
//      limit; it is neither a PASS nor a FAIL (and not a FAIL under
//      --strict). Two saved runs compare with no GPU: --compare a --against b.
//
// Flags
//   --wind U      wind speed at 10 m (m/s)                 default 10
//   --fetch m     fetch (m), 0 = fully developed            default 0
//   --chop c      choppiness                                default Ocean's 0.55
//   --theta rad   wind direction                            default Ocean's 0.6
//   --seconds s   recorded sim time                         default 409.6
//   --hz f        sample rate of the pinned clock           default 10
//   --gauges n    n x n virtual wave gauges over the tile   default 16
//   --disp-grid n n x n Eulerian height grid for dispersion default 128
//   --disp-span m its side (m), 0 = one cascade-1 tile      default 0
//   --foam-hz f   frame rate of the whitecap segment        default 30
//   --foam-seconds s  whitecap measurement after a 20 s spin-up, default 30
//   --json path   append one JSON object per metric
//   --compare p   print each metric against the saved run p (JSON lines)
//   --against p   with --compare: the run to compare, read from p; no GPU
//   --wake U      the wake case at hull speed U (m/s); with --hull L (30),
//                 --beam B (8), --window S (512, the ripples' period),
//                 --ripple-res n (2 S, a 0.5 m texel), --track-z z (0.137 S,
//                 off the patch's mirror line), --wake-seconds (0.8 S / U, so
//                 no wake wraps round the periodic patch into the measured
//                 region) and --wake-scalar auto | slope | height (auto reads
//                 height when the ripple image's alpha carries it)
//   --dump prefix for a look: a sea case writes the gauge PSD to
//                 <prefix>_psd.txt, the dispersion diagram to <prefix>_kw.txt
//                 and the graded dispersion rings to <prefix>_rings.txt; a
//                 wake case writes the image and the ridge maxima to
//                 <prefix>.f32 and <prefix>.txt
//   --strict      exit 1 when any graded metric FAILs (MARGINAL does not)
//   --foam-default  measure the engine's foam interval (see above)
// Exit codes: 0 ran, 1 a FAIL under --strict or a renderer that failed to
// start, 2 bad arguments, 42 no Vulkan device (the ctest skip code of the
// other Vulkan tests). The sweep returns the worst of its children's codes.
//
// How each metric is read
//   Wave gauges and the dispersion grid: DisplacedMesh::sampleHeight at fixed
//     world points, so the series is Eulerian (what a buoy or a lidar sees,
//     chop and the B-spline reconstruction of cascades 0 and 1 included).
//   Spectra: the per-cascade spatial fields the renderer copies back for
//     sampleHeight (DisplacedMesh::heightFields), FFT'd here. Each cascade is
//     periodic over its own tile, so its spatial FFT has no leakage.
//   The data in heightFields lags the render by the renderer's frames in
//   flight; every series here is uniformly spaced, so the lag is a constant
//   time offset and changes nothing measured.
//
//   Whitecaps: VulkanRenderer::readOceanFoam, the foam accumulator itself,
//     put through a CPU copy of the water shade's foam mask, in a segment run
//     at a game frame rate with the foam stepped every frame. The engine's
//     own default steps it less often; the 'engine defaults' case reports
//     that interval (INFO).
//
// Not measured here: the shade's view-dependent LEAN banking (reported only
// as its footprint floor) and the analytic wake (bow bump and V-wedge in
// water_displace.comp); the wake case measures the ripples of the wake field
// (wake_ripple.comp), read back with VulkanRenderer::readWakeRipples. On dev
// they carry slopes only, so the wake metrics are taken from the slope image;
// a ripple image whose alpha carries the height is read as height.

#include "threepp/objects/Ocean.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#endif

using namespace threepp;

namespace {

    constexpr int kSkipCode = 42;
    constexpr double kG = 9.81;// the engine's g (dynamic_spectrum.comp)
    constexpr double kPi = 3.14159265358979323846;
    // Cascade 1's sample domain is rotated by this angle (vulkan_shared.h,
    // kOceanCascade1RotTheta = atan(1/2)).
    constexpr double kC1Rot = 0.4636476090008061;

    using cfloat = std::complex<float>;

    // ── Radix-2 FFT ──────────────────────────────────────────────────────────
    // Forward is X[m] = sum x[n] e^(-2 pi i m n / N); inverse is unnormalised.
    struct Fft {
        int n = 0;
        std::vector<int> rev;
        std::vector<cfloat> tw;// e^(-2 pi i j / n), j < n/2

        explicit Fft(int size): n(size), rev(size), tw(size / 2) {
            int bits = 0;
            while ((1 << bits) < n) ++bits;
            for (int i = 0; i < n; ++i) {
                int r = 0;
                for (int b = 0; b < bits; ++b) r |= ((i >> b) & 1) << (bits - 1 - b);
                rev[i] = r;
            }
            for (int j = 0; j < n / 2; ++j) {
                const double a = -2.0 * kPi * j / n;
                tw[j] = cfloat(float(std::cos(a)), float(std::sin(a)));
            }
        }

        void run(cfloat* a, bool inverse) const {
            for (int i = 0; i < n; ++i)
                if (i < rev[i]) std::swap(a[i], a[rev[i]]);
            for (int len = 2; len <= n; len <<= 1) {
                const int half = len / 2, step = n / len;
                for (int i = 0; i < n; i += len) {
                    for (int j = 0; j < half; ++j) {
                        cfloat w = tw[j * step];
                        if (inverse) w = std::conj(w);
                        const cfloat u = a[i + j];
                        const cfloat v = a[i + j + half] * w;
                        a[i + j] = u + v;
                        a[i + j + half] = u - v;
                    }
                }
            }
        }
    };

    // In-place 2D FFT of a row-major n x n array (row = z, column = x).
    void fft2(const Fft& f, std::vector<cfloat>& a, bool inverse) {
        const int n = f.n;
        for (int r = 0; r < n; ++r) f.run(a.data() + size_t(r) * n, inverse);
        std::vector<cfloat> col(n);
        for (int c = 0; c < n; ++c) {
            for (int r = 0; r < n; ++r) col[r] = a[size_t(r) * n + c];
            f.run(col.data(), inverse);
            for (int r = 0; r < n; ++r) a[size_t(r) * n + c] = col[r];
        }
    }

    // Signed frequency index of FFT bin m.
    inline int signedIndex(int m, int n) { return m < n / 2 ? m : m - n; }

    // ── Metric table ─────────────────────────────────────────────────────────
    struct Metric {
        std::string name;
        double measured = 0;
        double lo = 0, hi = 0;// pass band; lo > hi means INFO (not graded)
        std::string target;   // the published value, readable
        std::string tolerance;
        std::string source;
        // Other estimators of the same quantity. When one of them falls on
        // the other side of a limit from `measured`, the verdict is MARGINAL:
        // the estimator, not the sea, decides PASS or FAIL.
        std::vector<double> alt;
        bool graded() const { return lo <= hi; }
        bool pass() const { return measured >= lo && measured <= hi; }
        int side(double v) const { return v < lo ? -1 : (v > hi ? 1 : 0); }
        bool marginal() const {
            if (!graded()) return false;
            for (double a: alt)
                if (side(a) != side(measured)) return true;
            return false;
        }
        const char* verdict() const {
            return !graded() ? "INFO" : marginal() ? "MARGINAL" : pass() ? "PASS" : "FAIL";
        }
    };

    std::string fmt(const char* f, double v) {
        char b[64];
        std::snprintf(b, sizeof(b), f, v);
        return b;
    }

    // ── One case ─────────────────────────────────────────────────────────────
    struct Config {
        float wind = 10.f;
        float fetch = 0.f;
        float chop = -1.f; // < 0: Ocean's default
        float theta = -1e9f;// unset: Ocean's default
        double seconds = 409.6;
        double hz = 10.0;
        int gauges = 16;
        int dispGrid = 128;
        double dispSpan = 0.0;// 0: one cascade-1 tile
        double foamHz = 30.0;
        double foamSeconds = 30.0;
        std::string json;
        std::string compare, against;
        bool strict = false;
        // --wake: one pressure source (a hull) at speed U on a calm sea.
        float wakeSpeed = 0.f;     // m/s; > 0 selects the wake case
        float hullLength = 30.f;   // m
        float hullBeam = 8.f;      // m
        float wakeWindow = 512.f;  // m, the ripples' period and window
        int rippleRes = 0;         // 0: 2 texels a metre
        double trackZ = -1e9;      // unset: 0.137 S
        double wakeSeconds = 0.0;  // 0: 0.8 S / U
        std::string wakeScalar = "auto";
        std::string dump;          // --dump prefix: the wake image and ridge, for a look
        double wakeWeight = 0.0;   // N; 0: the box hull's (draft 0.4 B, block 0.5)
        std::string wakeStrips;    // a file of "x z share radius half" lines (usv_rig.Wash.weight_strips)
        float wakeDraft = 0.f;     // m, each source's draft (WakeSource::draft; 0 = none)
        bool foamDefault = false;  // --foam-default: measure the engine's foam interval
    };

    // The foam accumulator is stepped every frame here, so a foam step is one
    // frame; foam_world's decay clamps a step to 0.25 s
    // (VulkanCoreGeometry.cpp, the foamDecay block), so a slower frame rate
    // would decay too little and read high.
    constexpr double kMaxFoamStep = 0.25;

    // ── Shared: a hidden-window renderer and the table ───────────────────────
    constexpr int kW = 320, kH = 180;
    struct Gpu {
        std::unique_ptr<Canvas> canvas;
        std::unique_ptr<VulkanRenderer> renderer;
    };

    // True when a renderer start-up failure means "this machine has no usable
    // Vulkan device" (VulkanContext's own messages for a missing driver, no
    // GPU, or no surface it can present to), the case the other Vulkan tests
    // skip with 42. Any other failure is a real error.
    bool isNoDevice(const std::string& what) {
        for (const char* s: {"no Vulkan-capable GPU", "no GPU with swapchain support", "vkCreateInstance failed",
                             "VK_EXT_headless_surface", "required queue families not present"})
            if (what.find(s) != std::string::npos) return true;
        return false;
    }

    // 0 started, kSkipCode no Vulkan device, 1 any other start-up failure.
    int makeGpu(Gpu& g) {
        std::unordered_map<std::string, Canvas::ParameterValue> canvasParams{
                {"vsync", false}, {"size", WindowSize{kW, kH}}, {"headless", true}};
        try {
            g.canvas = std::make_unique<Canvas>("ocean_metrics", canvasParams);
            g.renderer = std::make_unique<VulkanRenderer>(*g.canvas);
        } catch (const std::exception& e) {
            if (isNoDevice(e.what())) {
                std::printf("[skip] no Vulkan device: %s\n", e.what());
                return kSkipCode;
            }
            std::printf("[error] the renderer failed to start: %s\n", e.what());
            return 1;
        }
        return 0;
    }

    // ── Saved runs and --compare ─────────────────────────────────────────────
    struct Row {
        std::string kase, metric, verdict;
        double measured = 0;
        double chop = -1;// the case's choppiness; < 0 where it does not apply
    };

    // The value of "key":"..." or "key":number in one JSON line written by
    // report() (no escapes are ever written, so a plain scan is enough).
    std::string jsonField(const std::string& line, const std::string& key) {
        const std::string k = "\"" + key + "\":";
        const size_t p = line.find(k);
        if (p == std::string::npos) return {};
        size_t a = p + k.size();
        if (a < line.size() && line[a] == '"') {
            const size_t b = line.find('"', a + 1);
            return b == std::string::npos ? std::string{} : line.substr(a + 1, b - a - 1);
        }
        size_t b = a;
        while (b < line.size() && line[b] != ',' && line[b] != '}') ++b;
        return line.substr(a, b - a);
    }

    std::vector<Row> loadRows(const std::string& path, std::streamoff from = 0) {
        std::vector<Row> rows;
        std::ifstream in(path);
        if (!in) return rows;
        in.seekg(from);
        std::string line;
        while (std::getline(in, line)) {
            Row r;
            r.kase = jsonField(line, "case");
            r.metric = jsonField(line, "metric");
            r.verdict = jsonField(line, "verdict");
            const std::string m = jsonField(line, "measured");
            if (r.metric.empty() || m.empty()) continue;
            r.measured = std::atof(m.c_str());
            const std::string c = jsonField(line, "chop");
            r.chop = c.empty() ? -1.0 : std::atof(c.c_str());
            rows.push_back(std::move(r));
        }
        return rows;
    }

    // The chop column of --compare: the choppiness both runs used, or
    // "base->now !" where they differ.
    std::string chopCell(double base, double now, bool& differs) {
        differs = false;
        if (base < 0 && now < 0) return "-";
        if (base < 0 || now < 0) return fmt("%.2f", base < 0 ? now : base) + "?";
        if (std::fabs(base - now) <= 1e-3) return fmt("%.2f", now);
        differs = true;
        return fmt("%.2f", base) + fmt("->%.2f !", now);
    }

    // One line per metric of `now`: the saved value, the new one, the change,
    // the choppiness of both runs and the verdicts; a '*' marks a verdict
    // that changed, a '!' a row whose choppiness differs from the baseline.
    void printCompare(const std::vector<Row>& base, const std::vector<Row>& now) {
        std::printf("\n%-24s %-44s %12s %12s %10s %-12s %s\n", "case", "metric", "base", "now", "change", "chop",
                    "verdict");
        std::string last;
        int chopRows = 0;
        for (const auto& r: now) {
            const Row* b = nullptr;
            for (const auto& x: base)
                if (x.kase == r.kase && x.metric == r.metric) b = &x;
            if (r.kase != last) std::printf("\n"), last = r.kase;
            if (!b) {
                std::printf("%-24s %-44s %12s %12.5g %10s %-12s %s (new)\n", r.kase.c_str(), r.metric.c_str(), "-",
                            r.measured, "", r.chop < 0 ? "-" : fmt("%.2f", r.chop).c_str(), r.verdict.c_str());
                continue;
            }
            char change[32] = "";
            if (b->measured != 0.0) std::snprintf(change, sizeof(change), "%+.1f%%", 100.0 * (r.measured / b->measured - 1.0));
            else if (r.measured != 0.0) std::snprintf(change, sizeof(change), "%+.3g", r.measured);
            else std::snprintf(change, sizeof(change), "0");
            bool differs = false;
            const std::string chop = chopCell(b->chop, r.chop, differs);
            chopRows += differs ? 1 : 0;
            std::printf("%-24s %-44s %12.5g %12.5g %10s %-12s %s -> %s%s\n", r.kase.c_str(), r.metric.c_str(),
                        b->measured, r.measured, change, chop.c_str(), b->verdict.c_str(), r.verdict.c_str(),
                        b->verdict != r.verdict ? " *" : "");
        }
        for (const auto& b: base) {
            bool seen = false;
            for (const auto& r: now) seen = seen || (r.kase == b.kase && r.metric == b.metric);
            if (!seen && std::any_of(now.begin(), now.end(), [&](const Row& r) { return r.kase == b.kase; }))
                std::printf("%-24s %-44s %12.5g %12s %10s %-12s %s (gone)\n", b.kase.c_str(), b.metric.c_str(),
                            b.measured, "-", "", "", b.verdict.c_str());
        }
        if (chopRows > 0)
            std::printf("\n! %d rows ran at a different choppiness from the baseline. The dispersion, slope, crest and\n"
                        "  whitecap rows depend on it: rerun the branch with --chop at the baseline's value to\n"
                        "  separate its change from the change of default.\n",
                        chopRows);
        std::fflush(stdout);
    }

    // Prints the table and appends one JSON object per metric; `tag` names
    // the case in the JSON (wind and fetch, or the wake; never a tunable, so
    // a run on another branch matches its baseline row by row).
    int report(const std::vector<Metric>& out, const Config& cfg, const std::string& tag, double chop = -1.0) {
        std::printf("\n%-42s %12s  %-14s %-16s %-8s %s\n", "metric", "measured", "target", "tolerance", "", "source");
        // MARGINAL is not a FAIL under --strict: the estimators disagree on
        // the side of the limit, so the row cannot decide either way.
        bool anyFail = false;
        for (const auto& m: out) {
            const char* v = m.verdict();
            if (std::strcmp(v, "FAIL") == 0) anyFail = true;
            std::printf("%-42s %12.5g  %-14s %-16s %-8s %s\n", m.name.c_str(), m.measured, m.target.c_str(),
                        m.tolerance.c_str(), v, m.source.c_str());
        }
        std::fflush(stdout);
        std::vector<Row> rows;
        for (const auto& m: out) {
            std::string name = m.name;
            name.erase(0, name.find_first_not_of(' '));
            rows.push_back({tag, name, m.verdict(), m.measured, chop});
        }
        if (!cfg.json.empty()) {
            std::ofstream js(cfg.json, std::ios::app);
            for (size_t i = 0; i < out.size(); ++i) {
                const auto& m = out[i];
                char b[1024];
                std::snprintf(b, sizeof(b),
                              "{\"case\":\"%s\",\"metric\":\"%s\",\"measured\":%.6g,\"lo\":%.6g,\"hi\":%.6g,"
                              "\"target\":\"%s\",\"tolerance\":\"%s\",\"verdict\":\"%s\",\"source\":\"%s\",\"chop\":%.3g}",
                              tag.c_str(), rows[i].metric.c_str(), m.measured, m.lo, m.hi, m.target.c_str(),
                              m.tolerance.c_str(), rows[i].verdict.c_str(), m.source.c_str(), chop);
                js << b << "\n";
            }
        }
        // A single case compares itself; the sweep compares once at its end.
        if (!cfg.compare.empty()) printCompare(loadRows(cfg.compare), rows);
        return (cfg.strict && anyFail) ? 1 : 0;
    }

    // ── The water shade's foam mask, on the CPU ──────────────────────────────
    // deferred_shade_50_water_glass.glsl turns the foam accumulator into the
    // whitewater it blends over the water:
    //   coverage = clamp(1.3 B(acc), 0, 1), B the accumulator's bicubic
    //     B-spline (line 907; sampleFoamBicubic, deferred_shade_10_lighting_
    //     utils.glsl:904);
    //   sheet = smoothstep(0.03, 0.42, coverage - pools mix(0.50, 0.22,
    //     coverage) - (edge - 0.5) 0.30), pools = fbm4(P.xz 0.18 + drift),
    //     edge = detail.r at P.xz / 8 (lines 929-933);
    //   web = smoothstep(0.55, 0.80, lace (0.55 + 0.45 coverage)) smoothstep(
    //     0.03, 0.22, coverage), lace = detail.g at P.xz / 12 (lines 937-939);
    //   mask = max(sheet, 0.8 web) (line 945), seen from above.
    // The detail tile is VulkanRenderer's generateFoamDetailTile_
    // (VulkanCoreTextures.cpp), rebuilt here bit for bit. "Near" samples it at
    // mip 0, as the shade does for water within about 8 m of the camera;
    // "far" at its mean, the top of its mip chain.
    inline float fractf(float x) { return x - std::floor(x); }
    inline float smoothstepf(float a, float b, float x) {
        const float t = std::clamp((x - a) / (b - a), 0.f, 1.f);
        return t * t * (3.f - 2.f * t);
    }
    // deferred_shade_10_lighting_utils.glsl hash21 / vnoise21 / fbm4, in fp32.
    inline float hash21(float px, float py) {
        px = fractf(px * 123.34f), py = fractf(py * 456.21f);
        const float d = px * (px + 45.32f) + py * (py + 45.32f);
        return fractf((px + d) * (py + d));
    }
    inline float vnoise21(float x, float y) {
        const float ix = std::floor(x), iy = std::floor(y);
        float fx = x - ix, fy = y - iy;
        fx = fx * fx * (3.f - 2.f * fx), fy = fy * fy * (3.f - 2.f * fy);
        const float a = hash21(ix, iy), b = hash21(ix + 1.f, iy);
        const float c = hash21(ix, iy + 1.f), d = hash21(ix + 1.f, iy + 1.f);
        const float ab = a + (b - a) * fx, cd = c + (d - c) * fx;
        return ab + (cd - ab) * fy;
    }
    inline float fbm4(float x, float y) {
        float a = 0.f, w = 0.5f;
        for (int i = 0; i < 4; ++i) {
            a += w * vnoise21(x, y);
            x *= 2.03f, y *= 2.03f, w *= 0.5f;
        }
        return a;
    }

    struct FoamShade {
        static constexpr int kRes = 512;
        std::vector<uint8_t> tile;// RG8, row-major
        float mean[2] = {0, 0};

        FoamShade(): tile(size_t(kRes) * kRes * 2) {
            auto hashf = [](int x, int y) {
                uint32_t h = static_cast<uint32_t>(x) * 374761393u + static_cast<uint32_t>(y) * 668265263u;
                h = (h ^ (h >> 13)) * 1274126177u;
                h ^= h >> 16;
                return static_cast<float>(h) * (1.0f / 4294967296.0f);
            };
            auto vnoiseT = [&](float u, float v, int cells, int seed) {
                const float x = u * float(cells), y = v * float(cells);
                const int xi = int(std::floor(x)), yi = int(std::floor(y));
                float tx = x - float(xi), ty = y - float(yi);
                tx = tx * tx * (3.f - 2.f * tx);
                ty = ty * ty * (3.f - 2.f * ty);
                auto at = [&](int ix, int iy) {
                    return hashf(((ix % cells) + cells) % cells + seed * 7919, ((iy % cells) + cells) % cells);
                };
                const float a = at(xi, yi), b = at(xi + 1, yi), c = at(xi, yi + 1), d = at(xi + 1, yi + 1);
                return a + (b - a) * tx + (c - a) * ty + (a - b - c + d) * tx * ty;
            };
            double sum[2] = {0, 0};
            for (int y = 0; y < kRes; ++y) {
                const float v = (float(y) + 0.5f) / float(kRes);
                for (int x = 0; x < kRes; ++x) {
                    const float u = (float(x) + 0.5f) / float(kRes);
                    const float micro = 0.55f * vnoiseT(u, v, 26, 1) + 0.30f * vnoiseT(u, v, 60, 2) +
                                        0.15f * vnoiseT(u, v, 132, 3);
                    const float nf = 0.62f * vnoiseT(u, v, 10, 4) + 0.38f * vnoiseT(u, v, 23, 5);
                    const float lace = 1.f - std::fabs(nf - 0.5f) * 2.f;
                    const size_t i = (size_t(y) * kRes + x) * 2;
                    tile[i + 0] = uint8_t(std::lround(std::clamp(micro, 0.f, 1.f) * 255.f));
                    tile[i + 1] = uint8_t(std::lround(std::clamp(lace, 0.f, 1.f) * 255.f));
                    sum[0] += tile[i] / 255.0, sum[1] += tile[i + 1] / 255.0;
                }
            }
            mean[0] = float(sum[0] / (double(kRes) * kRes)), mean[1] = float(sum[1] / (double(kRes) * kRes));
        }

        // Bilinear, REPEAT, texel centres at (i + 0.5) / kRes.
        float sample(float u, float v, int ch) const {
            const float x = u * kRes - 0.5f, y = v * kRes - 0.5f;
            const float x0 = std::floor(x), y0 = std::floor(y);
            const float tx = x - x0, ty = y - y0;
            const int ix = wrap(int(x0)), iy = wrap(int(y0)), ix1 = wrap(ix + 1), iy1 = wrap(iy + 1);
            auto t = [&](int a, int b) { return tile[(size_t(b) * kRes + a) * 2 + ch] / 255.f; };
            return (t(ix, iy) * (1 - tx) + t(ix1, iy) * tx) * (1 - ty) + (t(ix, iy1) * (1 - tx) + t(ix1, iy1) * tx) * ty;
        }
        static int wrap(int i) { return ((i % kRes) + kRes) % kRes; }

        // Coverage at world (px, pz): clamp(1.3 B(acc), 0, 1).
        static float coverage(const std::vector<float>& acc, int res, float tileSize, float px, float pz) {
            const float sx = px / tileSize * res - 0.5f, sz = pz / tileSize * res - 0.5f;
            const float ix = std::floor(sx), iz = std::floor(sz);
            const float fx = sx - ix, fz = sz - iz;
            auto w = [](float f, float* o) {
                const float f2 = f * f, f3 = f2 * f;
                o[0] = (-f3 + 3 * f2 - 3 * f + 1) / 6.f;
                o[1] = (3 * f3 - 6 * f2 + 4) / 6.f;
                o[2] = (-3 * f3 + 3 * f2 + 3 * f + 1) / 6.f;
                o[3] = f3 / 6.f;
            };
            float wx[4], wz[4];
            w(fx, wx), w(fz, wz);
            float v = 0.f;
            for (int j = 0; j < 4; ++j) {
                const int z = ((int(iz) - 1 + j) % res + res) % res;
                float row = 0.f;
                for (int i = 0; i < 4; ++i) row += wx[i] * acc[size_t(z) * res + size_t(((int(ix) - 1 + i) % res + res) % res)];
                v += wz[j] * row;
            }
            return std::clamp(1.3f * v, 0.f, 1.f);
        }

        // The mask at world (px, pz) at time t; `near` picks mip 0 of the
        // detail tile, else its mean.
        float mask(float cov, float px, float pz, float t, bool near) const {
            if (cov <= 0.f) return 0.f;
            const float dx = 0.42f * t * 0.18f, dz = 0.71f * t * 0.18f;
            const float pools = fbm4(px * 0.18f + dx, pz * 0.18f + dz);
            const float edge = near ? sample(px * 0.125f + dx * 0.07f, pz * 0.125f + dz * 0.07f, 0) : mean[0];
            const float sheet = smoothstepf(0.03f, 0.42f, cov - pools * (0.50f + (0.22f - 0.50f) * cov) - (edge - 0.5f) * 0.30f);
            const float lace = near ? sample(px / 12.f + dx * 0.12f, pz / 12.f + dz * 0.12f, 1) : mean[1];
            const float web = smoothstepf(0.55f, 0.80f, lace * (0.55f + 0.45f * cov)) * smoothstepf(0.03f, 0.22f, cov);
            return std::max(sheet, web * 0.8f);
        }
    };

    // Per-cascade spectral state kept across frames.
    struct Cascade {
        int index = 0;
        int n = 0;
        double tile = 0;
        double rot = 0;// sample-domain rotation (world -> domain)
        std::unique_ptr<Fft> fft;

        // Dispersion: full spectra of the last three frames (Prony fit) and
        // energy / lag-one accumulators per mode.
        std::vector<cfloat> spec[3];
        std::vector<double> e0, c1;
        int specCount = 0;

        // Dispersion, 3D FFT part: the time series of the strongest modes.
        std::vector<int> modes;
        std::vector<std::vector<cfloat>> series;

        // Snapshot derivative fields, world metres: dh/dx, dh/dz and the
        // horizontal displacement gradient (before choppiness).
        std::vector<cfloat> gradH, gradDx, gradDz;
        // Breaking crests: the world-frame displacement gradient as the
        // engine's stencil sees it (central differences at the mesh spacing,
        // water_displace.comp), packed d/dx + i d/dz of Dx_world and Dz_world.
        std::vector<cfloat> gradSx, gradSz;
        // Height variance per wavenumber bin, summed over snapshots (the
        // omnidirectional k-spectrum).
        std::vector<double> eSum;
    };

    void readHeight(const DisplacedMesh::CascadeField& cf, double scale, std::vector<cfloat>& out) {
        const size_t cells = size_t(cf.dim) * cf.dim;
        out.resize(cells);
        for (size_t i = 0; i < cells; ++i) out[i] = cfloat(float(cf.data[2 * i] * scale), 0.f);
    }

    // Spectral gradient of a real field given its spectrum: returns the
    // complex field (d/dx) + i (d/dz), the packing that keeps both real.
    void spectralGradient(const Cascade& c, const std::vector<cfloat>& spec, std::vector<cfloat>& out) {
        const int n = c.n;
        out.assign(size_t(n) * n, cfloat(0.f, 0.f));
        const double dk = 2.0 * kPi / c.tile;
        for (int mz = 0; mz < n; ++mz) {
            for (int mx = 0; mx < n; ++mx) {
                if (mx == n / 2 || mz == n / 2) continue;// Nyquist: no real derivative
                const float kx = float(signedIndex(mx, n) * dk);
                const float kz = float(signedIndex(mz, n) * dk);
                const cfloat s = spec[size_t(mz) * n + mx];
                const cfloat i1(0.f, 1.f);
                // i kx S + i (i kz S)
                out[size_t(mz) * n + mx] = i1 * kx * s - kz * s;
            }
        }
        fft2(*c.fft, out, true);
        const float inv = 1.f / float(size_t(n) * n);
        for (auto& v: out) v *= inv;
    }

    // The displacement gradient water_displace.comp's breaking-crest stencil
    // measures: world-frame components of D (cascade 1's domain rotated back),
    // differenced along the world axes at +-eps, i.e. each mode's i k_w
    // replaced by i sin(k_w eps) / eps. sdx, sdz: the domain-frame spectra of
    // the displacement before choppiness. Outputs packed d/dx + i d/dz.
    void stencilGradients(const Cascade& c, const std::vector<cfloat>& sdx, const std::vector<cfloat>& sdz,
                          double eps, std::vector<cfloat>& gx, std::vector<cfloat>& gz) {
        const int n = c.n;
        gx.assign(size_t(n) * n, cfloat(0.f, 0.f));
        gz.assign(size_t(n) * n, cfloat(0.f, 0.f));
        const double dk = 2.0 * kPi / c.tile;
        const double cr = std::cos(c.rot), sr = std::sin(c.rot);
        const cfloat i1(0.f, 1.f);
        for (int mz = 0; mz < n; ++mz) {
            for (int mx = 0; mx < n; ++mx) {
                if (mx == n / 2 || mz == n / 2) continue;
                const double kx = signedIndex(mx, n) * dk, kz = signedIndex(mz, n) * dk;
                // World wavevector R^T k and world displacement R^T D.
                const double kwx = cr * kx + sr * kz, kwz = -sr * kx + cr * kz;
                const size_t m = size_t(mz) * n + mx;
                const cfloat dxw = float(cr) * sdx[m] + float(sr) * sdz[m];
                const cfloat dzw = float(-sr) * sdx[m] + float(cr) * sdz[m];
                const float sx = float(std::sin(kwx * eps) / eps), sz = float(std::sin(kwz * eps) / eps);
                gx[m] = i1 * sx * dxw - sz * dxw;
                gz[m] = i1 * sx * dzw - sz * dzw;
            }
        }
        fft2(*c.fft, gx, true);
        fft2(*c.fft, gz, true);
        const float inv = 1.f / float(size_t(n) * n);
        for (auto& v: gx) v *= inv;
        for (auto& v: gz) v *= inv;
    }

    // water_displace.comp's cwmLimit: the largest l <= lMax keeping
    // det(I + l A) >= kCwmMinJacobian on [0, l].
    double cwmLimit(double trA, double detA, double lMax) {
        const double c = 1.0 - double(DisplacedMesh::kCwmMinJacobian);
        double r = 1e30;
        if (std::abs(detA) < 1e-8) {
            if (trA < 0.0) r = -c / trA;
        } else {
            const double disc = trA * trA - 4.0 * detA * c;
            if (disc >= 0.0) {
                const double q = -0.5 * (trA + (trA >= 0.0 ? 1.0 : -1.0) * std::sqrt(disc));
                const double r1 = q / detA;
                const double r2 = q != 0.0 ? c / q : 1e30;
                if (r1 > 0.0) r = std::min(r, r1);
                if (r2 > 0.0) r = std::min(r, r2);
            }
        }
        return std::min(lMax, r);
    }

    // Bilinear read of a periodic complex field at domain point (px, pz).
    cfloat sampleBilinear(const std::vector<cfloat>& f, int n, double tile, double px, double pz) {
        const double fx = px / tile * n - 0.5, fz = pz / tile * n - 0.5;
        const double x0 = std::floor(fx), z0 = std::floor(fz);
        const float tx = float(fx - x0), tz = float(fz - z0);
        auto wrap = [n](long long v) { v %= n; return int(v < 0 ? v + n : v); };
        const int ix = wrap((long long) x0), iz = wrap((long long) z0);
        const int ix1 = (ix + 1) % n, iz1 = (iz + 1) % n;
        const cfloat a = f[size_t(iz) * n + ix], b = f[size_t(iz) * n + ix1];
        const cfloat c = f[size_t(iz1) * n + ix], d = f[size_t(iz1) * n + ix1];
        return (a * (1.f - tx) + b * tx) * (1.f - tz) + (c * (1.f - tx) + d * tx) * tz;
    }

    // Whitecap fraction (not percent) by U10: Monahan and O'Muircheartaigh
    // 1980, and Callaghan et al. 2008's two pieces joined at 11.25 m/s.
    double monahan(double U) { return 3.84e-6 * std::pow(U, 3.41); }
    double callaghan(double U) {
        if (U <= 3.7) return 0.0;
        return U <= 11.25 ? 3.18e-5 * std::pow(U - 3.70, 3.0) : 4.82e-6 * std::pow(U + 1.98, 3.0);
    }

    int runCase(const Config& cfg) {
        Gpu gpu;
        if (const int rc = makeGpu(gpu)) return rc;
        Canvas& canvas = *gpu.canvas;
        VulkanRenderer& renderer = *gpu.renderer;

        Scene scene;
        PerspectiveCamera camera(55.f, float(kW) / kH, 0.5f, 4000.f);
        camera.position.set(0.f, 40.f, 120.f);
        camera.lookAt(Vector3{0.f, 0.f, 0.f});
        auto sun = DirectionalLight::create(Color::white, 2.f);
        sun->position.set(100.f, 200.f, 50.f);
        scene.add(sun);

        Ocean::Options opt;
        opt.windSpeed = cfg.wind;
        opt.fetch = cfg.fetch;
        if (cfg.chop >= 0.f) opt.choppiness = cfg.chop;
        if (cfg.theta > -1e8f) opt.windTheta = cfg.theta;
        auto ocean = Ocean::create(opt);
        scene.add(ocean);
        // The first sampleHeight opts the mesh in to the CPU copy of its wave
        // fields (DisplacedMesh::wantsHeightReadback).
        (void) ocean->sampleHeight(0.f, 0.f);

        const auto& P = ocean->params;
        const double U = cfg.wind;
        // The choppiness the surface is drawn with: 1 under the Choppy Wave
        // Model (DisplacedMesh::effectiveChoppiness).
        const double chop = ocean->effectiveChoppiness();
        const bool cwmOn = ocean->cwmOn();
        // Crest sharpness: cascades 0+1 drawn at choppiness 1 + s under the limiter.
        const double sharp = ocean->effectiveCrestSharpness();
        const bool breakOn = ocean->breakingCrestsOn();
        // The engine's crest stencil: the vertex grid's spacing.
        const double meshEps = double(opt.size) / double(std::max(2u, opt.resolution) - 1u);
        const double waveScale = P.waveScale;
        const double theta = P.windTheta;
        const double tile0 = P.tileSize0;
        const double dt = 1.0 / cfg.hz;
        const int nFrames = std::max(64, int(std::lround(cfg.seconds * cfg.hz)));
        // The 3D FFT and the Prony fit run over the first nDisp frames.
        int nDisp = 1024;
        while (nDisp > nFrames) nDisp /= 2;
        // Spatial snapshots (slopes, Jacobian) every snapEvery frames.
        const int snapEvery = std::max(1, nFrames / 32);
        const double t0 = 10.0;// setSimTime needs t >= 0, warm-up included
        const int nWarm = 8;

        // Eulerian dispersion grid: D x D sampleHeight points over a square of
        // side W (one cascade-1 tile by default), sampled every frame of the
        // first nDisp.
        int D = 8;
        while (D * 2 <= std::max(8, cfg.dispGrid)) D *= 2;
        const double W = cfg.dispSpan > 0 ? cfg.dispSpan : double(P.tileSize1 > 0 ? P.tileSize1 : P.tileSize0);

        std::printf("case U10=%.1f m/s fetch=%.0f m choppiness=%.2f theta=%.3f tiles=%.1f/%.2f/%.2f fft=%u/%u/%u "
                    "frames=%d at %.1f Hz, dispersion grid %d^2 over %.1f m, foam at %.0f Hz\n",
                    U, double(cfg.fetch), chop, theta, P.tileSize0, P.tileSize1, P.tileSize2,
                    P.textureSize0, P.textureSize1, P.textureSize2, nFrames, cfg.hz, D, W, cfg.foamHz);
        std::printf("wave model: cwm %d, bound waves %d, breaking crests %d, crest sharpness %.2f\n",
                    int(cwmOn), int(ocean->boundWavesOn()), int(breakOn), sharp);
        std::fflush(stdout);

        auto renderAt = [&](double t) {
            renderer.setSimTime(t);
            canvas.animateOnce([&] { renderer.render(scene, camera); });
        };
        for (int i = 0; i < nWarm; ++i) renderAt(t0 - (nWarm - i) * dt);

        // Gauges: a G x G grid over the cascade-0 tile, offset off the texel
        // lattice.
        const int G = std::max(2, cfg.gauges);
        std::vector<std::pair<float, float>> gauges;
        for (int gz = 0; gz < G; ++gz)
            for (int gx = 0; gx < G; ++gx)
                gauges.emplace_back(float((gx + 0.37) * tile0 / G - tile0 * 0.5),
                                    float((gz + 0.61) * tile0 / G - tile0 * 0.5));
        std::vector<std::vector<float>> eta(gauges.size(), std::vector<float>(nFrames));

        std::vector<Cascade> cas;
        {
            const float tiles[3] = {P.tileSize0, P.tileSize1, P.tileSize2};
            for (int i = 0; i < 3; ++i) {
                if (!(tiles[i] > 0.f)) continue;
                Cascade c;
                c.index = i;
                c.tile = tiles[i];
                c.rot = i == 1 ? kC1Rot : 0.0;
                cas.push_back(std::move(c));
            }
        }

        // Slope / Jacobian accumulators over snapshots.
        double lin[3][3] = {};// per cascade: <sx^2>, <sx sz>, <sz^2> in WORLD axes
        int linCount = 0;
        double geoSum[3] = {}, geoW = 0;// area-weighted <su^2>, <sc^2>, <su sc>
        long long jSamples = 0, jNeg = 0, jNegC[3] = {}, jNeg01 = 0, jLimited = 0;
        // Breaking crests: drawn area (weight J) with the stenciled steepness
        // above the threshold, and a weighted histogram of the steepness for
        // the calibration rows.
        constexpr int kEpsBins = 400;
        constexpr double kEpsBin = 0.0025;
        std::vector<double> epsHist(kEpsBins + 1, 0.0);
        double breakArea = 0, breakW = 0;
        double restVar[3] = {};
        double divCorr[3][3] = {};// per cascade: <h div D>, <h^2>, <(div D)^2>
        int staleFrames = 0;

        // Eulerian dispersion: the grid's spatial spectrum every frame; the
        // time series of each bin up to half the grid's Nyquist wavenumber.
        // The cut does not keep aliases out: the grid takes point samples,
        // so a wave with a component in [1.5, 2.5] kNyq folds to within
        // 0.5 kNyq of zero, inside the graded band, and carries its own
        // frequency sqrt(g k) there. What it brings in is the sea's energy
        // at k >= 1.5 kNyq: waves under 1.3 m at the default 128 points over
        // a cascade-1 tile (a 0.99 m grid, against cascade 1's 0.5 m texel),
        // under 0.66 m at 256 (a 0.5 m grid). Measured at U15 with the chop
        // off: the rings in the spectrum's holes peak at 7-8.5 rad/s, the
        // aliases of 0.9-1.2 m waves, but 256 points lift the 4-8.5 m band's
        // in-band share by only 0.7 points; the rest of its 2-4 % floor is
        // the Hann window's main lobe (+-2 bins) reaching past the 5 % band
        // at the band's long end. The graded peak position moves by 0.3
        // points between 128 and 256, so 128 stays the default.
        const double hE = W / D, dkE = 2.0 * kPi / W, kNyqE = kPi / hE;
        Fft fftE(D);
        std::vector<float> hannE(D);
        for (int i = 0; i < D; ++i) hannE[i] = float(0.5 - 0.5 * std::cos(2.0 * kPi * (i + 0.5) / D));
        std::vector<int> eBins;
        for (int mz = 0; mz < D; ++mz)
            for (int mx = 0; mx < D; ++mx) {
                const double k = dkE * std::hypot(signedIndex(mx, D), signedIndex(mz, D));
                if (k > 0 && k <= 0.5 * kNyqE) eBins.push_back(mz * D + mx);
            }
        std::vector<std::vector<cfloat>> eSeries(eBins.size(), std::vector<cfloat>(nDisp));
        double eGridAll = 0;// grid height variance, all bins but k = 0, all frames
        std::vector<cfloat> eg(size_t(D) * D);
        std::vector<std::pair<float, float>> ePts;
        for (int iz = 0; iz < D; ++iz)
            for (int ix = 0; ix < D; ++ix)
                ePts.emplace_back(float(-0.5 * W + (ix + 0.29) * hE), float(-0.5 * W + (iz + 0.71) * hE));
        std::vector<float> prevC0;
        std::mt19937 rng(1234);
        const int kPointsPerSnap = 1 << 18;

        for (int f = 0; f < nFrames; ++f) {
            renderAt(t0 + f * dt);

            // Gauges (Eulerian heights, chop inverted by sampleHeight).
            for (size_t g = 0; g < gauges.size(); ++g)
                eta[g][f] = ocean->sampleHeight(gauges[g].first, gauges[g].second);

            // Stale-frame guard: the readback must advance every frame (all
            // cascades together; a fetch-limited sea can leave one empty).
            {
                std::vector<float> now;
                for (const auto& cf: ocean->heightFields) now.insert(now.end(), cf.data.begin(), cf.data.end());
                if (!prevC0.empty() && now == prevC0) ++staleFrames;
                prevC0 = std::move(now);
            }

            const bool doDisp = f < nDisp;
            const bool doSnap = (f % snapEvery) == 0;

            if (doDisp) {
                double mean = 0;
                for (size_t i = 0; i < ePts.size(); ++i) {
                    const float h = ocean->sampleHeight(ePts[i].first, ePts[i].second);
                    eg[i] = cfloat(h, 0.f);
                    mean += h;
                }
                mean /= double(ePts.size());
                for (int iz = 0; iz < D; ++iz)
                    for (int ix = 0; ix < D; ++ix) {
                        cfloat& v = eg[size_t(iz) * D + ix];
                        v = cfloat(float((v.real() - mean) * hannE[ix] * hannE[iz]), 0.f);
                    }
                fft2(fftE, eg, false);
                for (size_t m = 1; m < eg.size(); ++m) eGridAll += std::norm(eg[m]);
                for (size_t j = 0; j < eBins.size(); ++j) eSeries[j][f] = eg[eBins[j]];
            }
            if (!doDisp && !doSnap) continue;

            for (auto& c: cas) {
                const auto& cf = ocean->heightFields[c.index];
                if (cf.dim == 0) continue;
                if (c.n == 0) {
                    c.n = int(cf.dim);
                    c.fft = std::make_unique<Fft>(c.n);
                    const size_t cells = size_t(c.n) * c.n;
                    for (auto& s: c.spec) s.assign(cells, cfloat(0.f, 0.f));
                    c.e0.assign(cells, 0.0);
                    c.c1.assign(cells, 0.0);
                    c.eSum.assign(cells, 0.0);
                }
                const int n = c.n;
                const size_t cells = size_t(n) * n;
                // h = data / tile * waveScale (DisplacedMesh::CascadeField).
                std::vector<cfloat> hs;
                readHeight(cf, waveScale / c.tile, hs);
                if (doSnap) {
                    double v = 0;
                    for (const auto& x: hs) v += double(x.real()) * x.real();
                    restVar[c.index] += v / double(cells);
                }
                fft2(*c.fft, hs, false);

                if (doDisp) {
                    // Prony: for a mode A e^(iwt) + B e^(-iwt), H(t+dt) + H(t-dt)
                    // = 2 cos(w dt) H(t) exactly, whatever A and B are.
                    std::swap(c.spec[0], c.spec[1]);
                    std::swap(c.spec[1], c.spec[2]);
                    c.spec[2] = hs;
                    ++c.specCount;
                    if (c.specCount >= 3) {
                        for (size_t m = 0; m < cells; ++m) {
                            const cfloat h1 = c.spec[1][m];
                            c.e0[m] += std::norm(h1);
                            c.c1[m] += double((std::conj(h1) * (c.spec[0][m] + c.spec[2][m])).real());
                        }
                    }
                    // 3D FFT: on the first frame pick the strongest modes
                    // holding 99.5 % of this frame's energy (at most 4096),
                    // then record their complex amplitude every frame.
                    if (f == 0) {
                        std::vector<int> idx(cells);
                        std::iota(idx.begin(), idx.end(), 0);
                        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return std::norm(hs[a]) > std::norm(hs[b]); });
                        double tot = 0;
                        for (const auto& x: hs) tot += std::norm(x);
                        double acc = 0;
                        for (int m: idx) {
                            if (int(c.modes.size()) >= 4096 || acc >= 0.995 * tot) break;
                            const int mx = m % n, mz = m / n;
                            if ((mx == 0 && mz == 0) || mx == n / 2 || mz == n / 2) continue;
                            c.modes.push_back(m);
                            acc += std::norm(hs[m]);
                        }
                        c.series.assign(c.modes.size(), std::vector<cfloat>(nDisp));
                    }
                    for (size_t j = 0; j < c.modes.size(); ++j) c.series[j][f] = hs[c.modes[j]];
                }

                if (doSnap) {
                    // Height-gradient (linear) slope covariance by Parseval:
                    // <grad h grad h^T> = (1/N^4) sum k k^T |H(k)|^2, in the
                    // cascade's domain, rotated back to world.
                    const double dk = 2.0 * kPi / c.tile;
                    double sxx = 0, sxz = 0, szz = 0;
                    for (int mz = 0; mz < n; ++mz)
                        for (int mx = 0; mx < n; ++mx) {
                            const double kx = signedIndex(mx, n) * dk, kz = signedIndex(mz, n) * dk;
                            const double e = std::norm(hs[size_t(mz) * n + mx]);
                            c.eSum[size_t(mz) * n + mx] += e / (double(cells) * double(cells));
                            sxx += kx * kx * e;
                            sxz += kx * kz * e;
                            szz += kz * kz * e;
                        }
                    const double norm4 = double(cells) * double(cells);
                    sxx /= norm4, sxz /= norm4, szz /= norm4;
                    // World gradient = R^T domain gradient, R = rotation by c.rot.
                    const double cr = std::cos(c.rot), sr = std::sin(c.rot);
                    // R^T = [[cr, sr], [-sr, cr]]; C_w = R^T C_d R.
                    const double a = cr * cr * sxx + 2 * cr * sr * sxz + sr * sr * szz;
                    const double b = -cr * sr * sxx + (cr * cr - sr * sr) * sxz + cr * sr * szz;
                    const double d = sr * sr * sxx - 2 * cr * sr * sxz + cr * cr * szz;
                    lin[c.index][0] += a, lin[c.index][1] += b, lin[c.index][2] += d;

                    // Derivative fields for the Jacobian and the chop-correct
                    // slope: grad h, and grad of each displacement component,
                    // the displacement taken from the renderer's own field.
                    spectralGradient(c, hs, c.gradH);
                    std::vector<cfloat> dsp(cells);
                    const double ds = 1.0 / c.tile;// before choppiness
                    for (size_t i = 0; i < cells; ++i)
                        dsp[i] = cfloat(float(cf.disp[2 * i] * ds), float(cf.disp[2 * i + 1] * ds));
                    fft2(*c.fft, dsp, false);
                    // Split Z = FFT(dx + i dz) into the spectra of dx and dz.
                    std::vector<cfloat> sdx(cells), sdz(cells);
                    for (int mz = 0; mz < n; ++mz)
                        for (int mx = 0; mx < n; ++mx) {
                            const size_t m = size_t(mz) * n + mx;
                            const size_t mm = size_t((n - mz) % n) * n + size_t((n - mx) % n);
                            const cfloat zc = std::conj(dsp[mm]);
                            sdx[m] = (dsp[m] + zc) * 0.5f;
                            sdz[m] = (dsp[m] - zc) * cfloat(0.f, -0.5f);
                        }
                    spectralGradient(c, sdx, c.gradDx);
                    spectralGradient(c, sdz, c.gradDz);
                    if (breakOn && c.index < 2) stencilGradients(c, sdx, sdz, meshEps, c.gradSx, c.gradSz);
                    // Crest sharpening test, free of any FFT sign convention:
                    // choppy displacement sharpens crests only if it converges
                    // there, i.e. corr(h, div D) < 0 (Tessendorf 2001, the
                    // Lagrangian x = q - k/|k| a sin(k q - w t) of a Gerstner
                    // wave). Accumulated by Parseval.
                    for (int mz = 0; mz < n; ++mz)
                        for (int mx = 0; mx < n; ++mx) {
                            const size_t m = size_t(mz) * n + mx;
                            const float kx = float(signedIndex(mx, n) * dk), kz = float(signedIndex(mz, n) * dk);
                            const cfloat dv = cfloat(0.f, 1.f) * (kx * sdx[m] + kz * sdz[m]);
                            divCorr[c.index][0] += double((std::conj(hs[m]) * dv).real());
                            divCorr[c.index][1] += double(std::norm(hs[m]));
                            divCorr[c.index][2] += double(std::norm(dv));
                        }
                }
            }

            if (doSnap) {
                ++linCount;
                // Point samples at uniform rest positions q: the Jacobian of
                // x = q + chop * D(q) summed over cascades, and the slope of the
                // displaced surface, grad_x h = M^-T grad_q h, weighted by the
                // world area J it covers.
                std::uniform_real_distribution<double> uq(0.0, tile0);
                const double wx = std::cos(theta), wz = std::sin(theta);
                for (int s = 0; s < kPointsPerSnap; ++s) {
                    const double qx = uq(rng), qz = uq(rng);
                    double hx = 0, hz = 0, axx = 0, axz = 0, azx = 0, azz = 0;
                    double a01[4] = {};
                    double s01[4] = {};// stenciled world-frame grad D, cascades 0+1
                    for (const auto& c: cas) {
                        if (c.n == 0 || c.gradH.empty()) continue;
                        const double cr = std::cos(c.rot), sr = std::sin(c.rot);
                        const double px = cr * qx - sr * qz, pz = sr * qx + cr * qz;
                        const cfloat gh = sampleBilinear(c.gradH, c.n, c.tile, px, pz);
                        const cfloat gx = sampleBilinear(c.gradDx, c.n, c.tile, px, pz);
                        const cfloat gz = sampleBilinear(c.gradDz, c.n, c.tile, px, pz);
                        // Domain to world: grad h_w = R^T grad h_d;
                        // dD_w/dw = R^T (dD_d/dp) R.
                        const double ghx = gh.real(), ghz = gh.imag();
                        hx += cr * ghx + sr * ghz;
                        hz += -sr * ghx + cr * ghz;
                        const double m00 = gx.real(), m01 = gx.imag(), m10 = gz.real(), m11 = gz.imag();
                        // T = R^T M R with R = [[cr, -sr], [sr, cr]].
                        const double r00 = m00 * cr + m01 * sr, r01 = -m00 * sr + m01 * cr;
                        const double r10 = m10 * cr + m11 * sr, r11 = -m10 * sr + m11 * cr;
                        const double t00 = cr * r00 + sr * r10, t01 = cr * r01 + sr * r11;
                        const double t10 = -sr * r00 + cr * r10, t11 = -sr * r01 + cr * r11;
                        axx += t00, axz += t01, azx += t10, azz += t11;
                        const double jc = (1 + chop * t00) * (1 + chop * t11) - chop * chop * t01 * t10;
                        if (jc < 0) ++jNegC[c.index];
                        if (c.index < 2) a01[0] += t00, a01[1] += t01, a01[2] += t10, a01[3] += t11;
                        if (breakOn && c.index < 2 && !c.gradSx.empty()) {
                            const cfloat sx = sampleBilinear(c.gradSx, c.n, c.tile, px, pz);
                            const cfloat sz = sampleBilinear(c.gradSz, c.n, c.tile, px, pz);
                            s01[0] += sx.real(), s01[1] += sx.imag(), s01[2] += sz.real(), s01[3] += sz.imag();
                        }
                    }
                    // The drawn choppiness: under the CWM the per-vertex
                    // limiter's (water_displace.comp), from cascades 0+1.
                    double lam = chop;
                    if (cwmOn) {
                        lam = cwmLimit(a01[0] + a01[3], a01[0] * a01[3] - a01[1] * a01[2], 1.0 + sharp);
                        if (lam < 1.0 + sharp) ++jLimited;
                    }
                    // With sharpness the fine cascade keeps min(lam, 1) (water_displace.comp).
                    const double lf = sharp > 0 ? std::min(lam, 1.0) : lam;
                    const double m00 = sharp > 0 ? 1 + lam * a01[0] + lf * (axx - a01[0]) : 1 + lam * axx;
                    const double m01 = sharp > 0 ? lam * a01[1] + lf * (axz - a01[1]) : lam * axz;
                    const double m10 = sharp > 0 ? lam * a01[2] + lf * (azx - a01[2]) : lam * azx;
                    const double m11 = sharp > 0 ? 1 + lam * a01[3] + lf * (azz - a01[3]) : 1 + lam * azz;
                    const double J = m00 * m11 - m01 * m10;
                    const double J01 = (1 + lam * a01[0]) * (1 + lam * a01[3]) - lam * lam * a01[1] * a01[2];
                    if (breakOn && J > 0) {
                        // water_displace.comp: eps = minus the most compressive
                        // eigenvalue of the symmetric part of grad D.
                        const double sxz = 0.5 * (s01[1] + s01[2]);
                        const double mid = 0.5 * (s01[0] + s01[3]);
                        const double rad = std::sqrt(0.25 * (s01[0] - s01[3]) * (s01[0] - s01[3]) + sxz * sxz);
                        const double eps = -(mid - rad);
                        breakW += J;
                        if (eps > double(DisplacedMesh::kBreakingSteepness)) breakArea += J;
                        const int b = std::clamp(int(eps / kEpsBin), 0, kEpsBins);
                        epsHist[b] += J;
                    }
                    ++jSamples;
                    if (J < 0) ++jNeg;
                    if (J01 < 0) ++jNeg01;
                    if (J > 0) {
                        // M^-T grad_q h.
                        const double sx = (m11 * hx - m10 * hz) / J;
                        const double sz = (-m01 * hx + m00 * hz) / J;
                        const double su = sx * wx + sz * wz, sc = -sx * wz + sz * wx;
                        geoSum[0] += J * su * su;
                        geoSum[1] += J * sc * sc;
                        geoSum[2] += J * su * sc;
                        geoW += J;
                    }
                }
            }
        }

        // ── Whitecap segment ─────────────────────────────────────────────────
        // At a game frame rate with the foam stepped every frame (main() sets
        // THREEPP_OCEAN_FOAM_INTERVAL=1): foam_world keeps the max of the new
        // deposit and the decayed old value, so at a few steps a second a
        // moving crest leaves gaps a 30-60 Hz game would fill. 20 s of spin-up
        // (ten of its 2 s e-folding times), then one read a second.
        double foamArea = 0, foamAreaFar = 0, foamMean = 0, foamRaw = 0, foamCovHalf = 0;
        int foamReads = 0;
        // Crest foam: the foam texture over the
        // water's rest coordinates. The shade draws min(F, 1) as the covered
        // fraction (its foam photograph is rank-equalised), so the mean of that
        // over the tile is the whitecap fraction, no mask to mirror.
        double crestCover = 0, crestSheet = 0, crestRaw = 0, crestMax = 0;
        int crestReads = 0;
        {
            const double fdt = 1.0 / cfg.foamHz;
            double tf = t0 + (nFrames - 1) * dt;
            const int spin = int(std::lround(20.0 * cfg.foamHz));
            for (int i = 0; i < spin; ++i) renderAt(tf += fdt);
            const FoamShade shade;
            std::mt19937 frng(4321);
            std::uniform_real_distribution<float> jit(0.f, 1.f);
            std::vector<float> foam;
            const int reads = std::max(1, int(std::lround(cfg.foamSeconds)));
            const int perRead = std::max(1, int(std::lround(cfg.foamHz)));
            for (int r = 0; r < reads; ++r) {
                for (int i = 0; i < perRead; ++i) renderAt(tf += fdt);
                uint32_t fres = 0;
                float ftile = 0.f;
                if (!renderer.readOceanFoam(*ocean, foam, fres, ftile) || foam.empty()) break;
                // One jittered point per accumulator texel.
                double area = 0, areaFar = 0, cov = 0, raw = 0, half = 0;
                const int res = int(fres);
                for (int iz = 0; iz < res; ++iz)
                    for (int ix = 0; ix < res; ++ix) {
                        const float px = (ix + jit(frng)) * ftile / res, pz = (iz + jit(frng)) * ftile / res;
                        const float c = FoamShade::coverage(foam, res, ftile, px, pz);
                        cov += c;
                        raw += foam[size_t(iz) * res + ix];
                        if (c > 0.5f) half += 1;
                        if (c <= 0.f) continue;
                        if (shade.mask(c, px, pz, float(tf), true) > 0.5f) area += 1;
                        if (shade.mask(c, px, pz, float(tf), false) > 0.5f) areaFar += 1;
                    }
                const double nf = double(res) * res;
                foamArea += area / nf, foamAreaFar += areaFar / nf, foamMean += cov / nf;
                foamRaw += raw / nf, foamCovHalf += half / nf;
                ++foamReads;
                std::vector<float> cf;
                uint32_t cres = 0;
                float ctile = 0.f;
                if (renderer.readOceanCrestFoam(*ocean, cf, cres, ctile) && !cf.empty()) {
                    double cover = 0, sheet = 0, sum = 0;
                    for (const float f : cf) {
                        cover += std::min(f, 1.f);
                        if (f >= 1.f) sheet += 1;
                        sum += f;
                        crestMax = std::max(crestMax, double(f));
                    }
                    const double cn = double(cf.size());
                    crestCover += cover / cn, crestSheet += sheet / cn, crestRaw += sum / cn;
                    ++crestReads;
                }
            }
        }

        std::vector<Metric> out;

        // ── 1. Hs and Tp ─────────────────────────────────────────────────────
        // Targets: Pierson and Moskowitz (1964) fully developed, Hs ~ 0.022-
        // 0.025 U10^2 (the range is the 19.5 m to 10 m reference height) and
        // Tp ~ 0.75 U10; JONSWAP fetch-limited (Hasselmann et al. 1973):
        // g Hs / U^2 = 1.6e-3 (g X / U^2)^0.5, fp U / g = 3.5 (g X / U^2)^-0.33,
        // capped at full development. Tolerance +-10 % (research_targets 1.1).
        double var = 0, m3 = 0;
        long long cnt = 0;
        for (const auto& s: eta) {
            const double mean = std::accumulate(s.begin(), s.end(), 0.0) / double(s.size());
            for (float v: s) {
                const double d = v - mean;
                var += d * d;
                m3 += d * d * d;
                ++cnt;
            }
        }
        var /= double(cnt);
        m3 /= double(cnt);
        const double hs = 4.0 * std::sqrt(var);
        const double skew = m3 / std::pow(var, 1.5);

        double hsLo = 0.022 * U * U, hsHi = 0.025 * U * U;
        double tpT = 0.75 * U;
        std::string hsSrc = "Pierson-Moskowitz 1964", tpSrc = hsSrc;
        if (cfg.fetch > 0.f) {
            const double xi = kG * cfg.fetch / (U * U);
            const double hsJ = 1.6e-3 * std::sqrt(xi) * U * U / kG;
            const double tpJ = 1.0 / (3.5 * std::pow(xi, -0.33) * kG / U);
            if (hsJ < hsLo) hsLo = hsHi = hsJ, hsSrc = "JONSWAP, Hasselmann 1973";
            if (tpJ < tpT) tpT = tpJ, tpSrc = "JONSWAP, Hasselmann 1973";
        }
        out.push_back({"Hs (4 std eta, gauges)", hs, 0.9 * hsLo, 1.1 * hsHi,
                       hsLo == hsHi ? fmt("%.2f m", hsLo) : fmt("%.2f", hsLo) + "-" + fmt("%.2f m", hsHi),
                       "+-10%", hsSrc});
        double restV = 0;
        for (double v: restVar) restV += v;
        restV /= std::max(1, linCount);
        out.push_back({"Hs (rest-domain fields)", 4.0 * std::sqrt(restV), 1, 0, "", "", "same as above, no chop inversion"});

        // Welch PSD of the gauge series: Hann segments of 2048 samples (or the
        // whole record), 50 % overlap, averaged over all gauges.
        const int L = std::min(2048, 1 << int(std::floor(std::log2(nFrames))));
        Fft fl(L);
        std::vector<double> psd(L / 2 + 1, 0.0);
        std::vector<float> win(L);
        double wsum2 = 0;
        for (int i = 0; i < L; ++i) {
            win[i] = float(0.5 - 0.5 * std::cos(2.0 * kPi * i / L));
            wsum2 += double(win[i]) * win[i];
        }
        int segs = 0;
        std::vector<cfloat> buf(L);
        for (const auto& s: eta) {
            for (int start = 0; start + L <= nFrames; start += L / 2) {
                double mean = 0;
                for (int i = 0; i < L; ++i) mean += s[start + i];
                mean /= L;
                for (int i = 0; i < L; ++i) buf[i] = cfloat(float((s[start + i] - mean) * win[i]), 0.f);
                fl.run(buf.data(), false);
                for (int j = 1; j <= L / 2; ++j) psd[j] += 2.0 * std::norm(buf[j]) / (cfg.hz * wsum2);
                ++segs;
            }
        }
        for (auto& p: psd) p /= std::max(1, segs);
        const double df = cfg.hz / L;
        // The tile holds discrete wavenumbers, so below about 0.4 Hz the
        // gauge PSD is a comb of lines. Like a buoy's 0.01 Hz bands, the peak
        // is taken on the PSD averaged over +-0.01 Hz; the strongest single
        // line is reported beside it.
        auto peakOf = [&](const std::vector<double>& sp) {
            int jp = 1;
            for (int j = 2; j < L / 2; ++j)
                if (sp[j] > sp[jp]) jp = j;
            double delta = 0;
            if (jp > 1 && jp < L / 2) {
                const double a = sp[jp - 1], b = sp[jp], c = sp[jp + 1];
                const double den = a - 2 * b + c;
                if (den < 0) delta = 0.5 * (a - c) / den;
            }
            return (jp + delta) * df;
        };
        std::vector<double> psdBand(psd.size(), 0.0);
        {
            const int hw = std::max(1, int(std::lround(0.01 / df)));
            for (int j = 1; j <= L / 2; ++j) {
                double acc = 0;
                int nn = 0;
                for (int i = std::max(1, j - hw); i <= std::min(L / 2, j + hw); ++i) acc += psd[i], ++nn;
                psdBand[j] = acc / nn;
            }
        }
        const double fp = peakOf(psdBand);
        // Tp from the tile's omnidirectional spectrum (research_targets 1.1,
        // the second method): every wavenumber bin of every cascade, its
        // height variance averaged over the snapshots, goes to the frequency
        // band of its deep-water f = sqrt(g k) / 2 pi, on 1/12-octave bands
        // (6 % in f), S(f) = variance / band width. Near the peak the tile
        // holds only a few bins, so the band maximum still jumps between
        // lattice radii; fp is taken as Young's (1995) weighted mean
        // fp = int f S^4 df / int S^4 df, the robust peak estimator for a
        // noisy spectrum, summed over the bands with each band's width
        // df_b (proportional to f on 1/12-octave bands; leaving df out
        // would bias fp low by about the variance of ln f under S^4). The
        // band maximum and the gauge peak are reported beside it; when an
        // estimator falls on the other side of a limit the row is MARGINAL.
        double tpK = 0, tpKBand = 0;
        {
            const double fLo = 0.02, step = std::pow(2.0, 1.0 / 12.0);
            const int nb = int(std::ceil(std::log(4.0 / fLo) / std::log(step)));
            std::vector<double> sb(nb, 0.0);
            for (const auto& c: cas) {
                if (c.n == 0 || c.eSum.empty()) continue;
                const double dk = 2.0 * kPi / c.tile;
                for (int mz = 0; mz < c.n; ++mz)
                    for (int mx = 0; mx < c.n; ++mx) {
                        const double k = dk * std::hypot(signedIndex(mx, c.n), signedIndex(mz, c.n));
                        if (k <= 0) continue;
                        const double fk = std::sqrt(kG * k) / (2.0 * kPi);
                        const int b = int(std::floor(std::log(fk / fLo) / std::log(step)));
                        if (b >= 0 && b < nb) sb[b] += c.eSum[size_t(mz) * c.n + mx];
                    }
            }
            int bp = 0;
            double num = 0, den = 0;
            for (int b = 0; b < nb; ++b) {
                sb[b] /= fLo * std::pow(step, b) * (step - 1.0);// variance per Hz
                if (sb[b] > sb[bp]) bp = b;
            }
            const double sMax = sb[bp] > 0 ? sb[bp] : 1.0;
            for (int b = 0; b < nb; ++b) {
                const double q = std::pow(sb[b] / sMax, 4.0);
                const double width = fLo * std::pow(step, b) * (step - 1.0);// df_b
                num += fLo * std::pow(step, b + 0.5) * q * width, den += q * width;
            }
            tpK = den > 0 ? den / num : 0.0;
            tpKBand = 1.0 / (fLo * std::pow(step, bp + 0.5));
        }
        {
            Metric tp{"Tp (tile k-spectrum, Young 1995)", tpK, 0.9 * tpT, 1.1 * tpT, fmt("%.2f s", tpT), "+-10%", tpSrc};
            tp.alt = {tpKBand, 1.0 / fp};
            if (tp.marginal()) tp.source += "; estimators disagree";
            out.push_back(tp);
        }
        out.push_back({"  Tp (tile k-spectrum, band maximum)", tpKBand, 1, 0, "", "", "1/12-octave bands"});
        out.push_back({"  Tp (gauge PSD peak, 0.01 Hz bands)", 1.0 / fp, 1, 0, fmt("%.2f s", tpT), "",
                       "unstable on the tile's comb of lines"});
        out.push_back({"  Tp of the strongest single line", 1.0 / peakOf(psd), 1, 0, "", "", "the tile's discrete modes"});
        if (!cfg.dump.empty()) {
            // For a look: frequency (Hz), PSD (m^2/Hz), PSD in 0.01 Hz bands.
            std::ofstream txt(cfg.dump + "_psd.txt");
            for (int j = 1; j <= L / 2; ++j) txt << j * df << " " << psd[j] << " " << psdBand[j] << "\n";
        }
        // Tail slope of the frequency spectrum: least squares of log S on
        // log f over quarter-octave band means in [1.5 fp, min(4 fp, 0.4 fs)]
        // (band means, so the comb of lines does not set the slope):
        // f^-4 (Toba 1973, JONSWAP tail) to f^-5 (Pierson-Moskowitz).
        {
            double sx = 0, sy = 0, sxx = 0, sxy = 0;
            int nn = 0;
            const double fTop = std::min(4.0 * fp, 0.4 * cfg.hz);
            for (double fa = 1.5 * fp; fa * std::pow(2.0, 0.25) <= fTop * 1.0001; fa *= std::pow(2.0, 0.25)) {
                const double fb = fa * std::pow(2.0, 0.25);
                double acc = 0;
                int cntB = 0;
                for (int j = 1; j <= L / 2; ++j)
                    if (j * df >= fa && j * df < fb) acc += psd[j], ++cntB;
                if (cntB == 0 || acc <= 0) continue;
                const double x = std::log(std::sqrt(fa * fb)), y = std::log(acc / cntB);
                sx += x, sy += y, sxx += x * x, sxy += x * y;
                ++nn;
            }
            const double slope = nn > 2 ? (nn * sxy - sx * sy) / (nn * sxx - sx * sx) : 0.0;
            // INFO only: the comb of lines and the holes between the
            // cascades decide this slope more than the spectrum does.
            out.push_back({"PSD tail slope (1.5-4 fp)", slope, 1, 0, "-4 to -5", "",
                           "PM 1964 (f^-5), Toba 1973 (f^-4)"});
        }

        // ── 2. Dispersion ────────────────────────────────────────────────────
        // Target: omega^2 = g k (deep water; the engine has no depth term)
        // within 5 % in omega (research_targets 1.1).
        //
        // Graded on the Eulerian grid: sampleHeight over a D x D square, so
        // the chop's resampling and the B-spline reconstruction are in the
        // data. Hann windows in x, z and t (the square is not periodic for
        // any cascade). A bin is graded when the 5 % band is at least 1.5
        // bins wide both ways, i.e. 10 % of k >= 1.5 dk (5 % in omega is 10 %
        // in k) and 5 % of omega >= 1.5 d omega, and it lies at or below half
        // the grid's Nyquist wavenumber.
        //
        // Graded: where the peak of the frequency spectrum sits in each |k|
        // ring of the grid (rings dk wide), against sqrt(g k) for the ring's
        // energy-weighted k, within 5 %; the row is the energy share of the
        // rings whose peak is inside, pass mark 95 % (my choice; the research
        // gives no fraction). Not the energy share inside the 5 % band: the
        // chop carries a short wave on the long waves' horizontal motion,
        // which keeps its peak on the curve but moves about 1 - exp(-(k
        // sigma_D)^2) of its power to sidebands (sigma_D the rms horizontal
        // displacement), as the full orbital motion does on a real sea
        // (Leckler et al. 2015, stereo (k, omega) spectra). That share falls
        // with Hs whatever the dispersion, so it is INFO.
        {
            const double dOm = 2.0 * kPi / (nDisp * dt);
            const double kLo = std::max(15.0 * dkE, std::pow(30.0 * dOm, 2.0) / kG), kHi = 0.5 * kNyqE;
            Fft ft(nDisp);
            std::vector<cfloat> ts(nDisp);
            double inBand = 0, graded = 0, gradedRaw = 0;
            // Energy by |k| ring (dkE) and |omega| bin: the graded rings, and
            // the dispersion diagram for --dump.
            const int nk = int(std::ceil(kHi / dkE)) + 1, nw = nDisp / 2 + 1;
            std::vector<double> diag(size_t(nk) * nw, 0.0);
            std::vector<double> ringE(nk, 0.0), ringOm(nk, 0.0);
            std::vector<char> ringGraded(nk, 0);
            for (size_t j = 0; j < eBins.size(); ++j) {
                const int m = eBins[j];
                const double k = dkE * std::hypot(signedIndex(m % D, D), signedIndex(m / D, D));
                const double om = std::sqrt(kG * k);
                const bool g = k >= kLo && k <= kHi;
                const size_t ring = size_t(std::lround(k / dkE));
                for (int i = 0; i < nDisp; ++i) {
                    const float w = float(0.5 - 0.5 * std::cos(2.0 * kPi * i / nDisp));
                    ts[i] = eSeries[j][i] * w;
                    if (g) gradedRaw += std::norm(eSeries[j][i]);
                }
                ft.run(ts.data(), false);
                double e = 0;
                for (int i = 0; i < nDisp; ++i) {
                    const double omi = std::abs(signedIndex(i, nDisp)) * dOm;
                    const double p = std::norm(ts[i]);
                    diag[ring * nw + std::abs(signedIndex(i, nDisp))] += p;
                    e += p;
                    if (!g) continue;
                    graded += p;
                    if (std::abs(omi - om) <= 0.05 * om) inBand += p;
                }
                if (g) ringE[ring] += e, ringOm[ring] += e * om, ringGraded[ring] = 1;
            }
            // Per graded ring: the peak omega (i >= 1, refined by a parabola
            // through its neighbours) against the ring's sqrt(g k).
            double eIn = 0, eAll = 0, offW = 0, offMax = 0;
            int ringsIn = 0, rings = 0;
            std::string ringDump;
            for (int a = 0; a < nk; ++a) {
                if (!ringGraded[a] || !(ringE[a] > 0)) continue;
                const double* row = &diag[size_t(a) * nw];
                int ip = 1;
                for (int i = 2; i < nw - 1; ++i)
                    if (row[i] > row[ip]) ip = i;
                double delta = 0;
                if (ip > 1 && ip < nw - 1) {
                    const double den = row[ip - 1] - 2 * row[ip] + row[ip + 1];
                    if (den < 0) delta = 0.5 * (row[ip - 1] - row[ip + 1]) / den;
                }
                const double omRef = ringOm[a] / ringE[a], omPeak = (ip + delta) * dOm;
                const double off = omPeak / omRef - 1.0;
                const bool in = std::abs(off) <= 0.05;
                ++rings, ringsIn += in ? 1 : 0;
                eAll += ringE[a], eIn += in ? ringE[a] : 0.0;
                offW += ringE[a] * std::abs(off), offMax = std::max(offMax, std::abs(off));
                char b[160];
                std::snprintf(b, sizeof(b), "%d %.5f %.5f %.5f %+.5f %.6g %d\n", a, kG > 0 ? omRef * omRef / kG : 0.0,
                              omRef, omPeak, off, ringE[a], in ? 1 : 0);
                ringDump += b;
            }
            const std::string band = fmt("lambda %.1f", 2.0 * kPi / kHi) + fmt("-%.1f m", 2.0 * kPi / kLo);
            out.push_back({"dispersion, Eulerian grid, peak in 5%", eAll > 0 ? eIn / eAll : 0.0, 0.95, 1.0,
                           "w^2 = g k", ">= 95% of energy", "linear deep-water theory; " + band});
            out.push_back({"  rings with the peak in 5% (share)", rings > 0 ? double(ringsIn) / rings : 0.0, 1, 0, "",
                           "", fmt("%.0f rings", double(rings))});
            out.push_back({"  peak offset, energy-weighted mean |dw/w|", eAll > 0 ? offW / eAll : 0.0, 1, 0, "<= 0.05",
                           "", ""});
            out.push_back({"  peak offset, worst ring |dw/w|", offMax, 1, 0, "", "", ""});
            out.push_back({"  energy share in the 5% band", graded > 0 ? inBand / graded : 0.0, 1, 0, "", "",
                           "falls with Hs: the chop's sidebands"});
            out.push_back({"  grid variance share in the graded band", eGridAll > 0 ? gradedRaw / eGridAll : 0.0, 1, 0, "",
                           "", fmt("grid %.0f", double(D)) + fmt("^2 over %.1f m", W)});
            if (!cfg.dump.empty()) {
                // For a look: one line per graded ring: index, k, sqrt(g k),
                // peak omega, offset, energy, inside.
                std::ofstream rtxt(cfg.dump + "_rings.txt");
                rtxt << ringDump;
            }
            if (!cfg.dump.empty()) {
                // For a look: one line per |k| bin, k (rad/m) then the energy
                // at omega = i d omega, i = 0 .. nDisp / 2.
                std::ofstream txt(cfg.dump + "_kw.txt");
                txt << dkE << " " << dOm << " " << kLo << " " << kHi << "\n";
                for (int a = 0; a < nk; ++a) {
                    txt << a * dkE;
                    for (int i = 0; i < nw; ++i) txt << " " << diag[size_t(a) * nw + i];
                    txt << "\n";
                }
            }
        }
        // The spectrum clock (INFO): the same test on the raw cascade fields,
        // which by construction are a fixed spectrum turned at sqrt(g k); it
        // checks the time stepping of the spectrum update and nothing after.
        {
            // (a) 3D FFT: spatial FFT per frame (exact: each cascade is periodic
            // over its tile), Hann window and FFT in time per recorded mode.
            // Only modes whose 5 % band spans at least the Hann main lobe
            // (+-2 bins) are graded; their energy share is reported.
            const double dOm = 2.0 * kPi / (nDisp * dt);
            double inBand = 0, tot = 0, resolvedE = 0, allE = 0;
            Fft ft(nDisp);
            std::vector<cfloat> ts(nDisp);
            for (auto& c: cas) {
                const int n = c.n;
                const double dk = 2.0 * kPi / c.tile;
                for (size_t j = 0; j < c.modes.size(); ++j) {
                    const int m = c.modes[j];
                    const double kx = signedIndex(m % n, n) * dk, kz = signedIndex(m / n, n) * dk;
                    const double om = std::sqrt(kG * std::hypot(kx, kz));
                    double e = 0;
                    for (int i = 0; i < nDisp; ++i) {
                        const float w = float(0.5 - 0.5 * std::cos(2.0 * kPi * i / nDisp));
                        ts[i] = c.series[j][i] * w;
                    }
                    ft.run(ts.data(), false);
                    double eIn = 0;
                    for (int i = 0; i < nDisp; ++i) {
                        const double omi = std::abs(signedIndex(i, nDisp)) * dOm;
                        const double p = std::norm(ts[i]);
                        e += p;
                        if (std::abs(omi - om) <= 0.05 * om) eIn += p;
                    }
                    allE += e;
                    if (0.05 * om >= 2.0 * dOm) {
                        resolvedE += e;
                        inBand += eIn;
                        tot += e;
                    }
                }
            }
            out.push_back({"  spectrum clock, 3D FFT, in 5% band", tot > 0 ? inBand / tot : 0.0, 1, 0, "w^2 = g k", "",
                           "raw cascade fields"});
            out.push_back({"  energy share of FFT-resolved modes", allE > 0 ? resolvedE / allE : 0.0, 1, 0, "", "",
                           fmt("record %.1f s", nDisp * dt)});

            // (b) Prony fit over every mode: cos(w dt) = C / (2 E).
            double pin = 0, ptot = 0;
            for (auto& c: cas) {
                const int n = c.n;
                const double dk = 2.0 * kPi / c.tile;
                for (size_t m = 0; m < c.e0.size(); ++m) {
                    const double e = c.e0[m];
                    if (e <= 0) continue;
                    const int mx = int(m % n), mz = int(m / n);
                    if ((mx == 0 && mz == 0) || mx == n / 2 || mz == n / 2) continue;
                    const double kx = signedIndex(mx, n) * dk, kz = signedIndex(mz, n) * dk;
                    const double om = std::sqrt(kG * std::hypot(kx, kz));
                    const double cosw = std::clamp(c.c1[m] / (2.0 * e), -1.0, 1.0);
                    const double omFit = std::acos(cosw) / dt;
                    ptot += e;
                    if (std::abs(omFit - om) <= 0.05 * om) pin += e;
                }
            }
            out.push_back({"  spectrum clock, fit to every mode, in 5%", ptot > 0 ? pin / ptot : 0.0, 1, 0, "w^2 = g k", "",
                           "raw cascade fields"});
        }

        // ── 3. Mean square slope ─────────────────────────────────────────────
        // Cox and Munk (1954), clean sea: mss_u = 3.16e-3 U, mss_c = 0.003 +
        // 1.92e-3 U, total 0.003 + 5.12e-3 U (+-0.004; graded +-20 %), the
        // upwind / crosswind ratio 1.1-1.6 (research_targets 1.2). U is at
        // 12.5 m in their fit; U10 is used as is.
        {
            const double wx = std::cos(theta), wz = std::sin(theta);
            double up = 0, cr = 0;
            for (int c = 0; c < 3; ++c) {
                const double a = lin[c][0] / std::max(1, linCount), b = lin[c][1] / std::max(1, linCount),
                             d = lin[c][2] / std::max(1, linCount);
                const double u = wx * wx * a + 2 * wx * wz * b + wz * wz * d;
                const double x = wz * wz * a - 2 * wx * wz * b + wx * wx * d;
                up += u, cr += x;
                out.push_back({fmt("  mss linear, cascade %.0f", c), u + x, 1, 0, "", "", "height gradient only"});
            }
            const double cmT = 0.003 + 5.12e-3 * U;
            const double cmU = 3.16e-3 * U, cmC = 0.003 + 1.92e-3 * U;
            out.push_back({"mss linear (grad h), total", up + cr, 0.8 * cmT, 1.2 * cmT, fmt("%.4f", cmT), "+-20%",
                           "Cox-Munk 1954"});
            out.push_back({"  mss linear upwind", up, 1, 0, fmt("%.4f", cmU), "", "Cox-Munk 1954"});
            out.push_back({"  mss linear crosswind", cr, 1, 0, fmt("%.4f", cmC), "", "Cox-Munk 1954"});
            out.push_back({"mss linear up/cross ratio", cr > 0 ? up / cr : 0.0, 1.1, 1.6, "1.1-1.6", "range",
                           "Cox-Munk 1954"});
            const double gu = geoW > 0 ? geoSum[0] / geoW : 0, gc = geoW > 0 ? geoSum[1] / geoW : 0;
            out.push_back({"mss displaced surface, total", gu + gc, 0.8 * cmT, 1.2 * cmT, fmt("%.4f", cmT), "+-20%",
                           "Cox-Munk 1954; chop-correct, area-weighted"});
            out.push_back({"mss displaced up/cross ratio", gc > 0 ? gu / gc : 0.0, 1.1, 1.6, "1.1-1.6", "range",
                           "Cox-Munk 1954"});

            // The shade's sub-grid slope. Every water pixel's GGX lobe has
            // effRough = sqrt(r^2 + slopeVarSq) (deferred_shade_50_water_
            // glass.glsl:308), r the material's base roughness (Ocean.cpp
            // makeOceanMaterial, 0.04), the glints floored at 0.03 (line 803).
            // The shade banks slope variance into roughness as r^2 = 2 mss
            // (deferred_shade_06_prologue_surface.glsl:197), so in its own
            // terms the base stands for mss = r^2 / 2. As a GGX lobe
            // (alpha = r^2, distGGX in deferred_shade_10_lighting_utils.glsl)
            // its slope width is alpha, Beckmann-equivalent mss = alpha^2
            // (Walter et al. 2007), far smaller; the larger is graded, so a
            // FAIL holds either way. The footprint floor (prologue lines
            // 213-214, roughness 0.45 smoothstep(0.25, 4, m per pixel))
            // depends on the view, not the sea: 0 for water nearer than
            // 0.25 m a pixel, r^2 / 2 = 0.10 beyond 4 m a pixel.
            double r = 0.04;
            if (auto ms = std::dynamic_pointer_cast<MeshStandardMaterial>(ocean->material())) r = ms->roughness;
            const double rg = std::max(r, 0.03);
            const double baseShade = 0.5 * rg * rg, baseGgx = std::pow(rg * rg, 2.0);
            out.push_back({"  shade base roughness r", rg, 1, 0, "", "", "Ocean.cpp; glint floor 0.03"});
            out.push_back({"  base mss, shade's banking (r^2 / 2)", baseShade, 1, 0, "", "", "prologue_surface.glsl:197"});
            out.push_back({"  base mss, GGX lobe (alpha^2 = r^4)", baseGgx, 1, 0, "", "", "Walter 2007"});
            out.push_back({"  footprint floor mss beyond 4 m/px", 0.5 * 0.45 * 0.45, 1, 0, "", "",
                           "prologue_surface.glsl:213; view-dependent"});
            out.push_back({"mss displaced + shade base, total", gu + gc + baseShade, 0.8 * cmT, 1.2 * cmT,
                           fmt("%.4f", cmT), "+-20%", "Cox-Munk 1954; near water, no footprint floor"});
            // The measured short-wave spectrum (THREEPP_OCEAN_SHORT_WAVE_SPECTRUM)
            // banks what the finest cascade cannot draw of it into the shade's
            // roughness (the chop moments' mean square slope): the renderer
            // reports it with the fields.
            double banked = 0;
            for (const auto& f: ocean->heightFields) banked += f.bankedMss;
            if (banked > 0) {
                out.push_back({"  mss banked short-wave tail", banked, 1, 0, "", "",
                               "Elfouhaily 1997; unresolved, to roughness"});
                out.push_back({"mss displaced + banked tail, total", gu + gc + banked, 0.8 * cmT, 1.2 * cmT,
                               fmt("%.4f", cmT), "+-20%", "Cox-Munk 1954; resolved + banked"});
            }
        }

        // ── 4. Crests ────────────────────────────────────────────────────────
        // Second-order (Stokes) crests are a + k a^2 / 2, so the height pdf is
        // positively skewed; a self-folding mesh (J < 0) above about 0.1 % of
        // the area at default choppiness fails (research_targets 1.3).
        {
            // Second-order (Stokes) narrow-band skewness lambda3 = 3 sigma k_m
            // (Tayfun 1980), lowered by spectral width to 3 mu (1 - nu + nu^2)
            // (Fedele and Tayfun 2009); graded 0.5-1.2 of 3 sigma k_m. k_m
            // from the gauge spectrum's mean frequency m1/m0, deep water.
            double m0 = 0, m1 = 0;
            for (int j = 1; j <= L / 2; ++j) m0 += psd[j] * df, m1 += j * df * psd[j] * df;
            const double om = 2.0 * kPi * (m0 > 0 ? m1 / m0 : 0.0);
            const double target = 3.0 * std::sqrt(m0) * om * om / kG;
            out.push_back({"height skewness (gauges)", skew, 0.5 * target, 1.2 * target,
                           fmt("3 sigma k_m = %.3f", target), "x0.5-1.2", "Tayfun 1980; Fedele-Tayfun 2009"});
        }
        const double jf = jSamples ? double(jNeg) / double(jSamples) : 0.0;
        out.push_back({"area with J < 0, all cascades", jf, 0.0, 0.001, "< 0.1%", "", "research_targets 1.3"});
        if (cwmOn)
            out.push_back({"  area where the CWM limiter binds", jSamples ? double(jLimited) / jSamples : 0.0, 1, 0,
                           "", "", "choppiness < 1 + sharpness there (water_displace.comp cwmLimit)"});
        out.push_back({"  area with J < 0, cascades 0+1 (foam's)", jSamples ? double(jNeg01) / jSamples : 0.0, 1, 0,
                       "", "", "foam_world.comp sees these"});
        for (int c = 0; c < 3; ++c)
            out.push_back({fmt("  area with J < 0, cascade %.0f alone", c),
                           jSamples ? double(jNegC[c]) / jSamples : 0.0, 1, 0, "", "", ""});

        {
            double cv = 0, vh = 0, vd = 0;
            for (int c = 0; c < 3; ++c) {
                cv += divCorr[c][0], vh += divCorr[c][1], vd += divCorr[c][2];
                if (divCorr[c][1] > 0 && divCorr[c][2] > 0)
                    out.push_back({fmt("  corr(h, div D), cascade %.0f", c),
                                   divCorr[c][0] / std::sqrt(divCorr[c][1] * divCorr[c][2]), 1, 0, "", "", ""});
            }
            out.push_back({"corr(h, div D), all cascades", vh > 0 && vd > 0 ? cv / std::sqrt(vh * vd) : 0.0, -1.0,
                           0.0, "< 0", "sign", "crests converge: Gerstner / Tessendorf 2001"});
        }
        // ── 5. Whitecap coverage ─────────────────────────────────────────────
        // Monahan and O'Muircheartaigh (1980): W = 3.84e-6 U10^3.41. Callaghan
        // et al. (2008): W = 3.18e-3 (U10 - 3.70)^3 % for 3.7-11.25 m/s and
        // 4.82e-4 (U10 + 1.98)^3 % for 9.25-23.09 m/s. Field scatter is a
        // factor of 3 to 10, so the honest target is between the two curves
        // and zero below 3.7 m/s (research_targets 1.4). Graded: the AREA
        // where the shade's foam mask is above 0.5, the white a camera sees,
        // as a field photo is thresholded. The mean coverage the shade is
        // handed counts every faint decayed texel, so it is INFO only.
        // Breaking crests (THREEPP_OCEAN_BREAKING_CRESTS): natural foam is
        // the shade's step on the crest steepness water_displace.comp writes
        // per vertex; its mask is > 0.5 exactly where the steepness exceeds
        // kBreakingSteepness. Mirrored here at uniform rest points of the
        // snapshots: the same steepness (cascades 0+1, choppiness 1, world-
        // frame grad D differenced at the mesh spacing), weighted by the drawn
        // area J. The accumulator then holds vessel foam only (none here).
        if (breakOn && breakW > 0) {
            const double wM = monahan(U), wC = callaghan(U);
            const double lo = U > 3.7 ? std::min(wM, wC) : 0.0;
            const double hi = U > 3.7 ? std::max(wM, wC) : 1e-4;
            out.push_back({"whitecap area (shade mask > 0.5)", breakArea / breakW, lo, hi,
                           fmt("%.2f", 100 * lo) + "-" + fmt("%.2f %%", 100 * hi), "between curves",
                           fmt("Monahan 1980, Callaghan 2008; crest steepness > %.3f", DisplacedMesh::kBreakingSteepness)});
            // Calibration: the threshold that would put the area on each curve.
            auto epsFor = [&](double w) {
                double acc = 0;
                for (int b = kEpsBins; b >= 0; --b) {
                    acc += epsHist[b];
                    if (acc >= w * breakW) return (b + 1) * kEpsBin;
                }
                return 0.0;
            };
            out.push_back({"  crest steepness giving Monahan's area", epsFor(wM), 1, 0, "", "", "calibration"});
            out.push_back({"  crest steepness giving Callaghan's area", epsFor(wC), 1, 0, "", "", "calibration"});
        }
        if (crestReads > 0) {
            // Rest-space fractions: a parcel's share of the tile at rest. In
            // the drawn sea fresh foam sits where the map is squeezed, so the
            // world-space fraction is a little smaller.
            const double wM = monahan(U), wC = callaghan(U);
            const double lo = U > 3.7 ? std::min(wM, wC) : 0.0;
            const double hi = U > 3.7 ? std::max(wM, wC) : 1e-4;
            out.push_back({"crest foam cover (mean of min(F, 1))", crestCover / crestReads, lo, hi,
                           fmt("%.2f", 100 * lo) + "-" + fmt("%.2f %%", 100 * hi), "between curves",
                           "Monahan 1980, Callaghan 2008; foam_crest.comp"});
            out.push_back({"  closed sheet (F >= 1)", crestSheet / crestReads, 1, 0, "", "",
                           "the breaking front and what it just left"});
            out.push_back({"  crest foam value mean (raw)", crestRaw / crestReads, 1, 0, "", "", ""});
            out.push_back({"  crest foam value max", crestMax, 1, 0, "", "", "the deposit cap when crests break hard"});
        }
        if (foamReads > 0) {
            const double wM = monahan(U), wC = callaghan(U);
            const double lo = U > 3.7 ? std::min(wM, wC) : 0.0;
            const double hi = U > 3.7 ? std::max(wM, wC) : 1e-4;
            if (!breakOn)
                out.push_back({"whitecap area (shade mask > 0.5)", foamArea / foamReads, lo, hi,
                               fmt("%.2f", 100 * lo) + "-" + fmt("%.2f %%", 100 * hi), "between curves",
                               "Monahan 1980, Callaghan 2008"});
            out.push_back({"  whitecap area, far (detail at its mean)", foamAreaFar / foamReads, 1, 0, "", "",
                           "the same mask, top mip"});
            out.push_back({"  coverage mean (shade's input)", foamMean / foamReads, 1, 0, "", "",
                           "clamp(1.3 B(foam), 0, 1)"});
            out.push_back({"  area with coverage > 0.5", foamCovHalf / foamReads, 1, 0, "", "", ""});
            out.push_back({"  foam value mean (raw)", foamRaw / foamReads, 1, 0, "", "", "foam_world.comp"});
            out.push_back({"  foam step (s)", 1.0 / cfg.foamHz, 0.0, kMaxFoamStep, "<= 0.25 s", "",
                           "foam every frame at --foam-hz"});
            const char* iv = std::getenv("THREEPP_OCEAN_FOAM_INTERVAL");
            out.push_back({"  foam interval in this segment (frames)", iv ? std::atof(iv) : -1.0, 1, 0, "", "",
                           "forced by this audit; the engine's own default is the 'engine defaults' case"});
        } else if (!breakOn) {
            out.push_back({"whitecap area (shade mask > 0.5)", -1, 1, 0, "", "", "no foam readback"});
        }

        out.push_back({"stale readback frames", double(staleFrames), 0, 0, "0", "", "harness self-check"});

        char tag[96];
        std::snprintf(tag, sizeof(tag), "U10=%.1f fetch=%.0f", U, double(cfg.fetch));
        return report(out, cfg, tag, chop);
    }

    // ── The engine's foam interval ───────────────────────────────────────────
    // The sea cases force THREEPP_OCEAN_FOAM_INTERVAL=1, so they grade foam
    // stepped every frame. The engine reads that variable once, with its own
    // default when it is unset (VulkanCoreGeometry.cpp, kFoamInterval). This
    // case measures what it runs: a U15 sea at 30 frames a second, the
    // accumulator read after every frame once foam has formed; a frame whose
    // read differs from the one before ran foam_world, and the interval is
    // frames over such frames. The sweep runs it with the variable unset.
    int runFoamDefault(const Config& cfg) {
        Gpu gpu;
        if (const int rc = makeGpu(gpu)) return rc;
        Canvas& canvas = *gpu.canvas;
        VulkanRenderer& renderer = *gpu.renderer;

        Scene scene;
        PerspectiveCamera camera(55.f, float(kW) / kH, 0.5f, 4000.f);
        camera.position.set(0.f, 40.f, 120.f);
        camera.lookAt(Vector3{0.f, 0.f, 0.f});
        Ocean::Options opt;
        opt.windSpeed = 15.f;
        opt.fetch = 0.f;
        auto ocean = Ocean::create(opt);
        scene.add(ocean);

        const char* env = std::getenv("THREEPP_OCEAN_FOAM_INTERVAL");
        std::printf("engine defaults: foam interval with THREEPP_OCEAN_FOAM_INTERVAL %s\n",
                    env ? (std::string("= ") + env).c_str() : "unset");
        std::fflush(stdout);
        const double fps = 30.0;
        double t = 10.0;
        auto renderAt = [&](double tt) {
            renderer.setSimTime(tt);
            canvas.animateOnce([&] { renderer.render(scene, camera); });
        };
        for (int i = 0; i < int(2 * fps); ++i) renderAt(t += 1.0 / fps);
        std::vector<float> prev, now;
        uint32_t res = 0;
        float tile = 0.f;
        std::vector<Metric> out;
        if (!renderer.readOceanFoam(*ocean, prev, res, tile) || prev.empty()) {
            out.push_back({"foam interval, engine default (frames)", -1, 1, 0, "", "", "no foam readback"});
        } else {
            const int frames = 24;
            int ran = 0, reads = 0;
            for (int i = 0; i < frames; ++i) {
                renderAt(t += 1.0 / fps);
                if (!renderer.readOceanFoam(*ocean, now, res, tile)) continue;
                ++reads;
                if (now != prev) ++ran;
                std::swap(prev, now);
            }
            if (reads != frames) ran = 0;// a missed read would hide a step
            const double interval = ran > 0 ? double(frames) / ran : -1.0;
            const std::string src = env ? std::string("THREEPP_OCEAN_FOAM_INTERVAL=") + env
                                        : std::string("VulkanCoreGeometry.cpp, variable unset");
            out.push_back({"foam interval, engine default (frames)", interval, 1, 0, "1 (every frame)", "", src});
            out.push_back({"  foam step at 30 fps, engine default (s)", interval / 30.0, 1, 0, "0.033 s", "", ""});
            out.push_back({"  foam step at 60 fps, engine default (s)", interval / 60.0, 1, 0, "0.017 s", "", ""});
            out.push_back({"  frames that ran foam_world, of 24", double(ran), 1, 0, "", "", "accumulator changed"});
        }
        return report(out, cfg, "engine defaults");
    }

    // ── 6. Kelvin wake ───────────────────────────────────────────────────────
    // A periodic, world-anchored image of one scalar of the wake: `along`
    // oscillates with the waves along the track and `energy` is its local
    // power. Today the ripples carry slopes only, so `along` is the slope
    // along the track and `energy` the squared slope magnitude; once they
    // carry height, pass h and h^2 and nothing below changes.
    struct WakeImage {
        int n = 0;
        double period = 0;
        std::vector<float> along, energy;
    };

    inline int wrapIndex(long long i, int n) {
        i %= n;
        return int(i < 0 ? i + n : i);
    }

    // Texel ix lies at world ix * period / n, modulo period.
    inline size_t texelOf(const WakeImage& w, double x, double z) {
        const double t = w.period / w.n;
        return size_t(wrapIndex(std::llround(z / t), w.n)) * w.n + size_t(wrapIndex(std::llround(x / t), w.n));
    }

    // The source has moved along +x and now sits at (xs, zs). Transverse
    // wavelength: 2 pi U^2 / g (+-5 %). Angle of the maximum-amplitude
    // ridge: 19.47 deg up to a hull Froude number of about 0.5, then
    // 1 / (2 sqrt(2 pi) Fr) rad (Rabaud and Moisy 2013), +-2 deg.
    // `ridge`, when given, receives the (distance behind, signed lateral
    // offset) of every maximum the angle fit used.
    // `transMax` bounds the transverse-wave record and `behindMax` the ridge
    // fit: a source that started D behind has transverse waves only within
    // D / 2 (they travel at half their phase speed) and a full cusp only
    // within 2 D / 3 (Kelvin's group-velocity circles of radius e / 4 about
    // 3 e / 4 behind, touched at 0.67 e along the track).
    std::vector<Metric> kelvinMetrics(const WakeImage& w, double U, double L, double xs, double zs, double transMax,
                                      double behindMax, std::vector<std::pair<double, double>>* ridge = nullptr) {
        std::vector<Metric> out;
        const double texel = w.period / w.n;
        const double lamT = 2.0 * kPi * U * U / kG;
        const double fr = U / std::sqrt(kG * L);

        // Transverse waves on the centreline, from one hull length behind
        // the centre to behindMax: the strongest wavenumber of a Hann-windowed,
        // zero-padded FFT, refined by a parabola through its neighbours.
        double lamMeas = 0;
        {
            std::vector<double> sline;
            for (double d = L; d <= transMax; d += texel) sline.push_back(w.along[texelOf(w, xs - d, zs)]);
            const int m = int(sline.size());
            int nfft = 1;
            while (nfft < 16 * m) nfft <<= 1;
            Fft f(nfft);
            std::vector<cfloat> b(nfft, cfloat(0.f, 0.f));
            double mean = 0;
            for (double v: sline) mean += v;
            mean /= std::max(1, m);
            for (int i = 0; i < m; ++i)
                b[i] = cfloat(float((sline[i] - mean) * (0.5 - 0.5 * std::cos(2.0 * kPi * i / std::max(1, m - 1)))), 0.f);
            f.run(b.data(), false);
            // Skip wavelengths longer than half the record.
            const int jMin = std::max(2, int(2.0 * nfft / std::max(1, m)));
            int jp = jMin;
            for (int j = jMin; j < nfft / 2; ++j)
                if (std::norm(b[j]) > std::norm(b[jp])) jp = j;
            double delta = 0;
            if (jp > jMin) {
                const double a = std::norm(b[jp - 1]), c0 = std::norm(b[jp]), c = std::norm(b[jp + 1]);
                const double den = a - 2 * c0 + c;
                if (den < 0) delta = 0.5 * (a - c) / den;
            }
            const double cyclesPerMetre = (jp + delta) / (nfft * texel);
            lamMeas = cyclesPerMetre > 0 ? 1.0 / cyclesPerMetre : 0.0;
        }
        out.push_back({"wake transverse wavelength (m)", lamMeas, 0.95 * lamT, 1.05 * lamT, fmt("%.2f m", lamT), "+-5%",
                       "2 pi U^2 / g, Kelvin"});

        // Ridge: average the power along the track over one transverse
        // wavelength (it oscillates with the crests), then for each distance
        // behind the source take the lateral offset of the maximum on each
        // side and fit a line. No averaging across the track: a box as wide
        // as a wavelength there merges the two ridges of a narrow wake into
        // one on the centreline.
        const int hw = std::max(1, int(std::lround(0.5 * (lamMeas > 0 ? lamMeas : lamT) / texel)));
        std::vector<double> sm(w.energy.size());
        for (int r = 0; r < w.n; ++r) {
            for (int c = 0; c < w.n; ++c) {
                double acc = 0;
                for (int k = -hw; k <= hw; ++k) acc += w.energy[size_t(r) * w.n + wrapIndex(c + k, w.n)];
                sm[size_t(r) * w.n + c] = acc;
            }
        }
        const double tanMax = std::tan(35.0 * kPi / 180.0);
        double angles[2] = {0, 0};
        int fitted = 0;
        for (int side = 0; side < 2; ++side) {
            const double sgn = side == 0 ? 1.0 : -1.0;
            double sx = 0, sy = 0, sxx = 0, sxy = 0;
            int nn = 0;
            for (double d = std::max(2.0 * L, 2.0 * lamT); d <= behindMax; d += texel) {
                double best = -1, bestZ = 0;
                const double zHi = std::min(0.4 * w.period, d * tanMax);
                for (double z = std::max(texel, 0.02 * d); z <= zHi; z += texel) {
                    const double v = sm[texelOf(w, xs - d, zs + sgn * z)];
                    if (v > best) best = v, bestZ = z;
                }
                if (best <= 0) continue;
                if (ridge) ridge->emplace_back(d, sgn * bestZ);
                sx += d, sy += bestZ, sxx += d * d, sxy += d * bestZ;
                ++nn;
            }
            if (nn > 2) {
                const double slope = (nn * sxy - sx * sy) / (nn * sxx - sx * sx);
                angles[side] = std::atan(slope) * 180.0 / kPi;
                ++fitted;
            }
        }
        // Each side is its own measurement: the track is off the patch's
        // mirror line, so the two sides sit on different texels.
        const double alphaT = std::min(19.47, 180.0 / kPi / (2.0 * std::sqrt(2.0 * kPi) * fr));
        const std::string src = fmt("Rabaud-Moisy 2013, Fr_L = %.2f", fr);
        out.push_back({"wake ridge angle, +z side (deg)", angles[0], alphaT - 2.0, alphaT + 2.0, fmt("%.2f deg", alphaT),
                       "+-2 deg", src});
        out.push_back({"wake ridge angle, -z side (deg)", angles[1], alphaT - 2.0, alphaT + 2.0, fmt("%.2f deg", alphaT),
                       "+-2 deg", src});
        out.push_back({"  ridge angle, mean of the sides", fitted == 2 ? 0.5 * (angles[0] + angles[1]) : 0.0, 1, 0, "",
                       "", ""});
        return out;
    }

    // A calm sea (waveScale 0, choppiness 0) and one pressure source, a hull
    // of `hullLength` x `hullBeam` bearing down with its weight, moving along
    // +x at `wakeSpeed` on the line z = trackZ, inside one wake patch whose
    // ripples' window trails it (the source at 80 % of the window, as
    // VesselWash places it). The patch's small sea is periodic over S, so the
    // run is kept to 0.8 S / U: no part of the wake has wrapped round into
    // the region measured.
    int runWake(const Config& cfg) {
        Gpu gpu;
        if (const int rc = makeGpu(gpu)) return rc;
        Canvas& canvas = *gpu.canvas;
        VulkanRenderer& renderer = *gpu.renderer;

        Scene scene;
        PerspectiveCamera camera(55.f, float(kW) / kH, 0.5f, 4000.f);
        camera.position.set(0.f, 120.f, 60.f);
        camera.lookAt(Vector3{0.f, 0.f, 0.f});

        const double U = cfg.wakeSpeed, L = cfg.hullLength, B = cfg.hullBeam, S = cfg.wakeWindow;
        const double zs = cfg.trackZ > -1e8 ? cfg.trackZ : 0.137 * S;
        Ocean::Options opt;
        opt.waveScale = 0.f;
        opt.choppiness = 0.f;
        auto ocean = Ocean::create(opt);
        ocean->wakeField.resolution = 256;
        ocean->wakeField.rippleResolution = uint32_t(cfg.rippleRes > 0 ? cfg.rippleRes : int(std::lround(2.0 * S)));
        scene.add(ocean);

        // Weight of a box hull of draft 0.4 B and block coefficient 0.5 in
        // sea water. The metrics are geometric, so only its sign matters.
        const double weight = cfg.wakeWeight > 0 ? cfg.wakeWeight : 1025.0 * kG * L * B * 0.4 * B * 0.5;
        // --wake-strips: the hull as usv_rig.Wash lays her weight out, one
        // line source a strip (vessel frame: x ahead, z to starboard).
        struct Strip { double x, z, share, r, half; };
        std::vector<Strip> strips;
        if (!cfg.wakeStrips.empty()) {
            std::ifstream in(cfg.wakeStrips);
            Strip st{};
            while (in >> st.x >> st.z >> st.share >> st.r >> st.half) strips.push_back(st);
            std::printf("wake: %zu strips from %s, weight %.0f N, draft %.2f m\n", strips.size(), cfg.wakeStrips.c_str(),
                        weight, double(cfg.wakeDraft));
        }
        const double hz = 20.0, dt = 1.0 / hz, t0 = 1.0;
        const double seconds = cfg.wakeSeconds > 0 ? cfg.wakeSeconds : 0.8 * S / U;
        const int frames = std::max(2, int(std::lround(seconds * hz)));
        std::printf("wake U=%.2f m/s hull %.1f x %.1f m Fr_L=%.2f window %.0f m, track z %.2f m, %d frames at %.0f Hz\n",
                    U, L, B, U / std::sqrt(kG * L), S, zs, frames, hz);
        std::fflush(stdout);

        double x = 0, xPrev = 0;
        for (int f = 0; f < frames; ++f) {
            xPrev = x;
            x = U * f * dt;
            ocean->clearWakeSources();
            if (strips.empty()) {
                DisplacedMesh::WakeSource src;
                src.x0 = float(xPrev), src.z0 = float(zs);
                src.x1 = float(x), src.z1 = float(zs);
                src.radius = float(0.5 * B);
                src.push = float(weight);
                src.ax = float(0.5 * L), src.az = 0.f;
                src.patch = 0;
                src.draft = cfg.wakeDraft;
                ocean->addWakeSource(src);
            }
            for (const auto& st: strips) {
                DisplacedMesh::WakeSource src;
                src.x0 = float(xPrev + st.x), src.z0 = float(zs + st.z);
                src.x1 = float(x + st.x), src.z1 = float(zs + st.z);
                src.radius = float(st.r);
                src.push = float(weight * st.share);
                src.ax = float(st.half), src.az = 0.f;
                src.patch = 0;
                src.draft = cfg.wakeDraft;
                ocean->addWakeSource(src);
            }
            auto& p = ocean->wakePatches[0];
            p.centerX = float(x - 0.3 * S);
            p.centerZ = float(zs);
            p.size = float(S);
            renderer.setSimTime(t0 + f * dt);
            canvas.animateOnce([&] { renderer.render(scene, camera); });
        }

        std::vector<float> rip;
        uint32_t n = 0;
        float period = 0.f;
        std::vector<Metric> out;
        if (!renderer.readWakeRipples(*ocean, 0, rip, n, period) || n == 0) {
            out.push_back({"wake ripples read back", 0, 1, 1, "1", "", "readWakeRipples"});
        } else {
            // rgba: slope x, slope z, their squared sum, and the height when
            // a branch writes it to alpha (0 on dev).
            const size_t texels = size_t(n) * n;
            double hMax = 0;
            for (size_t i = 0; i < texels; ++i) hMax = std::max(hMax, double(std::fabs(rip[4 * i + 3])));
            const bool height = cfg.wakeScalar == "height" || (cfg.wakeScalar == "auto" && hMax > 1e-6);
            WakeImage w;
            w.n = int(n);
            w.period = period;
            w.along.resize(texels);
            w.energy.resize(texels);
            double peak = 0;
            for (size_t i = 0; i < texels; ++i) {
                if (height) {
                    w.along[i] = rip[4 * i + 3];
                    w.energy[i] = rip[4 * i + 3] * rip[4 * i + 3];
                } else {
                    w.along[i] = rip[4 * i];// slope along +x, the track
                    w.energy[i] = rip[4 * i] * rip[4 * i] + rip[4 * i + 1] * rip[4 * i + 1];
                }
                peak = std::max(peak, double(w.energy[i]));
            }
            // Stay inside the window (0.8 S behind the source), off its
            // faded outer tenth, and inside what the run has built.
            const double D = U * (frames - 1) * dt;
            const double transMax = std::min(0.7 * period, 0.45 * D), behindMax = std::min(0.7 * period, 0.6 * D);
            std::vector<std::pair<double, double>> ridge;
            out = kelvinMetrics(w, U, L, x, zs, transMax, behindMax, &ridge);
            if (!cfg.dump.empty()) {
                // For a look: <dump>.f32 holds n*n floats of `along` then n*n
                // of `energy`; <dump>.txt holds n, period, the source x and z,
                // then one ridge maximum per line.
                std::ofstream raw(cfg.dump + ".f32", std::ios::binary);
                raw.write(reinterpret_cast<const char*>(w.along.data()), std::streamsize(w.along.size() * 4));
                raw.write(reinterpret_cast<const char*>(w.energy.data()), std::streamsize(w.energy.size() * 4));
                std::ofstream txt(cfg.dump + ".txt");
                txt << n << " " << period << " " << x << " " << zs << "\n";
                for (const auto& [d, z]: ridge) txt << d << " " << z << "\n";
            }
            out.push_back({height ? "  max height in the window (m)" : "  max slope in the window", std::sqrt(peak), 1, 0,
                           "", "", height ? "ripple height (alpha)" : "ripples are slope only"});
            out.push_back({"  wake scalar is height", height ? 1.0 : 0.0, 1, 0, "", "", "--wake-scalar"});
        }
        char tag[96];
        std::snprintf(tag, sizeof(tag), "wake U=%.2f L=%.1f B=%.1f", U, L, B);
        return report(out, cfg, tag);
    }


}// namespace

namespace {

    // Whitecap growth with the wind: n of W ~ U^n, least squares of log W on
    // log U over the sweep's winds, for one fetch. The band is the same fit
    // to Monahan's and Callaghan's curves at those winds, widened down to 3
    // (research_targets 1.4: "close to cubic growth"); field fits run 3 to
    // 3.5 (Monahan 3.41).
    std::vector<Row> growthRows(const std::vector<Row>& rows, const std::vector<float>& winds, float fetch,
                                const std::string& json) {
        auto fit = [&](const std::vector<double>& w) {
            double sx = 0, sy = 0, sxx = 0, sxy = 0;
            int n = 0;
            for (size_t i = 0; i < winds.size(); ++i) {
                if (!(w[i] > 0)) return std::nan("");
                const double x = std::log(double(winds[i])), y = std::log(w[i]);
                sx += x, sy += y, sxx += x * x, sxy += x * y, ++n;
            }
            return (n * sxy - sx * sy) / (n * sxx - sx * sx);
        };
        std::vector<double> meas, mo, ca;
        char want[64];
        for (float u: winds) {
            std::snprintf(want, sizeof(want), "U10=%.1f fetch=%.0f", double(u), double(fetch));
            double v = -1;
            for (const auto& r: rows)
                if (r.kase == want && r.metric == "whitecap area (shade mask > 0.5)") v = r.measured;
            meas.push_back(v);
            mo.push_back(monahan(u));
            ca.push_back(callaghan(u));
        }
        const double n = fit(meas), nM = fit(mo), nC = fit(ca);
        const double lo = std::min({3.0, nM, nC}), hi = std::max(nM, nC);
        const bool ok = n >= lo && n <= hi;
        char kase[64];
        std::snprintf(kase, sizeof(kase), "sweep fetch=%.0f", double(fetch));
        std::printf("\n%-42s %12.5g  %-14s %-16s %-6s %s\n", "whitecap growth exponent n (W ~ U^n)", n,
                    (fmt("%.2f", lo) + "-" + fmt("%.2f", hi)).c_str(), "range", ok ? "PASS" : "FAIL",
                    "Monahan 1980, Callaghan 2008 fitted at the same winds");
        std::fflush(stdout);
        if (!json.empty()) {
            std::ofstream js(json, std::ios::app);
            char b[512];
            std::snprintf(b, sizeof(b),
                          "{\"case\":\"%s\",\"metric\":\"whitecap growth exponent n\",\"measured\":%.6g,\"lo\":%.6g,"
                          "\"hi\":%.6g,\"target\":\"%.2f-%.2f\",\"tolerance\":\"range\",\"verdict\":\"%s\","
                          "\"source\":\"Monahan 1980, Callaghan 2008\",\"chop\":-1}",
                          kase, n, lo, hi, lo, hi, ok ? "PASS" : "FAIL");
            js << b << "\n";
        }
        return {Row{kase, "whitecap growth exponent n", ok ? "PASS" : "FAIL", n}};
    }

    void setEnv(const char* name, const char* value) {
#ifdef _WIN32
        _putenv_s(name, value);
#else
        ::setenv(name, value, 1);
#endif
    }

    void unsetEnv(const char* name) {
#ifdef _WIN32
        _putenv_s(name, "");// an empty value removes it
#else
        ::unsetenv(name);
#endif
    }

    // A child's exit code from std::system: cmd.exe returns it as is; POSIX
    // returns a wait status (the code is in its second byte).
    int exitCodeOf(int status) {
#ifdef _WIN32
        return status < 0 ? 1 : status;
#else
        return (status != -1 && WIFEXITED(status)) ? WEXITSTATUS(status) : 1;
#endif
    }

}// namespace

int main(int argc, char** argv) {
    Config cfg;
    bool single = false;
    for (int i = 1; i < argc; ++i) {
        auto val = [&](const char* flag) -> const char* {
            return (std::strcmp(argv[i], flag) == 0 && i + 1 < argc) ? argv[++i] : nullptr;
        };
        if (const char* s = val("--wind")) cfg.wind = float(std::atof(s)), single = true;
        else if (const char* s = val("--fetch")) cfg.fetch = float(std::atof(s)), single = true;
        else if (const char* s = val("--chop")) cfg.chop = float(std::atof(s));
        else if (const char* s = val("--theta")) cfg.theta = float(std::atof(s));
        else if (const char* s = val("--seconds")) cfg.seconds = std::atof(s);
        else if (const char* s = val("--hz")) cfg.hz = std::atof(s);
        else if (const char* s = val("--gauges")) cfg.gauges = std::atoi(s);
        else if (const char* s = val("--disp-grid")) cfg.dispGrid = std::atoi(s);
        else if (const char* s = val("--disp-span")) cfg.dispSpan = std::atof(s);
        else if (const char* s = val("--foam-hz")) cfg.foamHz = std::atof(s);
        else if (const char* s = val("--foam-seconds")) cfg.foamSeconds = std::atof(s);
        else if (const char* s = val("--json")) cfg.json = s;
        else if (const char* s = val("--compare")) cfg.compare = s;
        else if (const char* s = val("--against")) cfg.against = s;
        else if (const char* s = val("--wake")) cfg.wakeSpeed = float(std::atof(s)), single = true;
        else if (const char* s = val("--hull")) cfg.hullLength = float(std::atof(s));
        else if (const char* s = val("--beam")) cfg.hullBeam = float(std::atof(s));
        else if (const char* s = val("--window")) cfg.wakeWindow = float(std::atof(s));
        else if (const char* s = val("--ripple-res")) cfg.rippleRes = std::atoi(s);
        else if (const char* s = val("--track-z")) cfg.trackZ = std::atof(s);
        else if (const char* s = val("--wake-seconds")) cfg.wakeSeconds = std::atof(s);
        else if (const char* s = val("--wake-scalar")) cfg.wakeScalar = s;
        else if (const char* s = val("--dump")) cfg.dump = s;
        else if (const char* s = val("--wake-weight")) cfg.wakeWeight = std::atof(s);
        else if (const char* s = val("--wake-strips")) cfg.wakeStrips = s;
        else if (const char* s = val("--wake-draft")) cfg.wakeDraft = float(std::atof(s));
        else if (std::strcmp(argv[i], "--strict") == 0) cfg.strict = true;
        else if (std::strcmp(argv[i], "--foam-default") == 0) cfg.foamDefault = true, single = true;
        else {
            std::printf("unknown argument %s (see the usage block in ocean_metrics.cpp)\n", argv[i]);
            return 2;
        }
    }
    if (cfg.hz <= 0 || cfg.seconds <= 0 || cfg.wind <= 0 || cfg.foamSeconds <= 0) return 2;
    if (cfg.wakeScalar != "auto" && cfg.wakeScalar != "slope" && cfg.wakeScalar != "height") return 2;
    // The whitecap segment steps the foam every frame; a step above 0.25 s
    // would be clamped by foam_world's decay and read high.
    if (!(cfg.foamHz > 0) || 1.0 / cfg.foamHz > kMaxFoamStep) {
        std::printf("--foam-hz %.3g gives a foam step of more than %.2f s\n", cfg.foamHz, kMaxFoamStep);
        return 2;
    }

    // Two saved runs: no GPU.
    if (!cfg.compare.empty() && !cfg.against.empty()) {
        const auto base = loadRows(cfg.compare), now = loadRows(cfg.against);
        if (base.empty() || now.empty()) {
            std::printf("no metric rows in %s\n", base.empty() ? cfg.compare.c_str() : cfg.against.c_str());
            return 2;
        }
        printCompare(base, now);
        return 0;
    }

    if (single) {
        // The engine's own interval, under whatever the environment says.
        if (cfg.foamDefault) return runFoamDefault(cfg);
        // Foam every frame, read once by the renderer at its first frame.
        setEnv("THREEPP_OCEAN_FOAM_INTERVAL", "1");
        return cfg.wakeSpeed > 0.f ? runWake(cfg) : runCase(cfg);
    }
    // The sweep's 'engine defaults' case measures the interval the engine
    // ships, so the variable is removed for every child (the sea cases set
    // their own).
    unsetEnv("THREEPP_OCEAN_FOAM_INTERVAL");

    // The sweep: each case in its own process, so no renderer state carries
    // from one ocean to the next. Every other flag but --compare is passed
    // through; the sweep compares once, at its end. The children always
    // write JSON (to --json, or a file next to this one), so the growth
    // exponent and the comparison read them back.
    std::string json = cfg.json;
    if (json.empty()) {
        json = std::string(argv[0]) + ".sweep.jsonl";
        std::remove(json.c_str());
    }
    std::streamoff from = 0;
    {
        std::ifstream in(json, std::ios::binary | std::ios::ate);
        if (in) from = in.tellg();
    }
    std::string pass;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--compare") == 0 || std::strcmp(argv[i], "--json") == 0) {
            ++i;
            continue;
        }
        pass += std::string(" \"") + argv[i] + "\"";
    }
    pass += " \"--json\" \"" + json + "\"";
    const std::vector<float> winds = {7.f, 10.f, 15.f}, fetches = {0.f, 20000.f};
    std::vector<std::string> cases;
    for (float fetch: fetches)
        for (float u: winds) {
            char b[128];
            std::snprintf(b, sizeof(b), " --wind %.1f --fetch %.0f", u, fetch);
            cases.emplace_back(b);
        }
    // Wakes: a 30 m research vessel at 10 kn (Fr_L 0.30, the Kelvin regime)
    // and a 4 m USV at 15 kn (Fr_L 1.23, a narrowed wake).
    cases.emplace_back(" --wake 5.14 --hull 30 --beam 8");
    cases.emplace_back(" --wake 7.72 --hull 4 --beam 1.6");
    // The foam interval the engine ships (INFO).
    cases.emplace_back(" --foam-default");
    int worst = 0;
    for (const auto& c: cases) {
#ifdef _WIN32
        // cmd.exe strips the outer quotes of a /c line that starts with one.
        const std::string cmd = std::string("\"\"") + argv[0] + "\"" + c + pass + "\"";
#else
        const std::string cmd = std::string("\"") + argv[0] + "\"" + c + pass;
#endif
        std::fflush(stdout);
        const int rc = exitCodeOf(std::system(cmd.c_str()));
        if (rc == kSkipCode) return kSkipCode;
        worst = std::max(worst, rc);
    }
    auto rows = loadRows(json, from);
    std::vector<Row> extra;
    for (float fetch: fetches) {
        const auto g = growthRows(rows, winds, fetch, json);
        extra.insert(extra.end(), g.begin(), g.end());
        if (cfg.strict && g[0].verdict == "FAIL") worst = std::max(worst, 1);
    }
    rows.insert(rows.end(), extra.begin(), extra.end());
    for (const auto& r: rows)
        if (r.kase == "engine defaults" && r.metric == "foam interval, engine default (frames)" && r.measured > 1.0)
            std::printf("\nNOTE: the engine steps the whitecap foam every %.3g frames by default (VulkanCoreGeometry.cpp,\n"
                        "      kFoamInterval), a reduced update rate; the sea cases above grade it stepped every frame.\n",
                        r.measured);
    if (!cfg.compare.empty()) printCompare(loadRows(cfg.compare), rows);
    return worst;
}
