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
//                                         each case in its own child process
//   ocean_metrics --wind 10 --fetch 0     one case
//   ocean_metrics --json metrics.jsonl    also append JSON lines to this file
//
// Flags
//   --wind U      wind speed at 10 m (m/s)                 default 10
//   --fetch m     fetch (m), 0 = fully developed            default 0
//   --chop c      choppiness                                default Ocean's 0.55
//   --theta rad   wind direction                            default Ocean's 0.6
//   --seconds s   recorded sim time                         default 409.6
//   --hz f        sample rate of the pinned clock           default 10
//   --gauges n    n x n virtual wave gauges over the tile   default 16
//   --json path   append one JSON object per metric
//   --strict      exit 1 when any graded metric FAILs
// Exit codes: 0 ran, 1 a FAIL under --strict, 2 bad arguments, 42 no Vulkan
// device (the ctest skip code of the other Vulkan tests).
//
// How each metric is read
//   Wave gauges: DisplacedMesh::sampleHeight at fixed world points, so the
//     series is Eulerian (what a buoy or a lidar sees, chop included).
//   Spectra: the per-cascade spatial fields the renderer copies back for
//     sampleHeight (DisplacedMesh::heightFields), FFT'd here. Each cascade is
//     periodic over its own tile, so its spatial FFT has no leakage.
//   The data in heightFields lags the render by the renderer's frames in
//   flight; every series here is uniformly spaced, so the lag is a constant
//   time offset and changes nothing measured.
//
//   Whitecaps: VulkanRenderer::readOceanFoam, the foam accumulator itself,
//     not a thresholded image.
//
// Not measured here: the shade's sub-grid roughness (base roughness plus the
// view-dependent LEAN banking in deferred_shade_06_prologue_surface.glsl), the
// B-spline reconstruction of cascades 0 and 1 in water_displace.comp (spectra
// below are of the raw cascade fields), and the vessel wakes.

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
        bool graded() const { return lo <= hi; }
        bool pass() const { return measured >= lo && measured <= hi; }
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
        std::string json;
        bool strict = false;
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

    int runCase(const Config& cfg) {
        constexpr int W = 320, H = 180;
        std::unordered_map<std::string, Canvas::ParameterValue> canvasParams{
                {"vsync", false}, {"size", WindowSize{W, H}}, {"headless", true}};
        std::unique_ptr<Canvas> canvasPtr;
        std::unique_ptr<VulkanRenderer> rendererPtr;
        try {
            canvasPtr = std::make_unique<Canvas>("ocean_metrics", canvasParams);
            rendererPtr = std::make_unique<VulkanRenderer>(*canvasPtr);
        } catch (const std::exception& e) {
            std::printf("[skip] Vulkan/RT GPU unavailable: %s\n", e.what());
            return kSkipCode;
        }
        Canvas& canvas = *canvasPtr;
        VulkanRenderer& renderer = *rendererPtr;

        Scene scene;
        PerspectiveCamera camera(55.f, float(W) / H, 0.5f, 4000.f);
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
        const double chop = P.choppiness;
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

        std::printf("case U10=%.1f m/s fetch=%.0f m choppiness=%.2f theta=%.3f tiles=%.1f/%.2f/%.2f fft=%u/%u/%u "
                    "frames=%d at %.1f Hz\n",
                    U, double(cfg.fetch), chop, theta, P.tileSize0, P.tileSize1, P.tileSize2,
                    P.textureSize0, P.textureSize1, P.textureSize2, nFrames, cfg.hz);
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
        long long jSamples = 0, jNeg = 0, jNegC[3] = {}, jNeg01 = 0;
        double restVar[3] = {};
        double divCorr[3][3] = {};// per cascade: <h div D>, <h^2>, <(div D)^2>
        // Whitecaps: the foam accumulator read back after a spin-up of 20 s
        // (ten of foam_world's 2 s e-folding times).
        double foamCov = 0, foamRaw = 0, foamHalf = 0;
        int foamReads = 0;
        const int foamSpinUp = int(std::lround(20.0 * cfg.hz));
        std::vector<float> foam;
        int staleFrames = 0;
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

            if (doSnap && f >= foamSpinUp) {
                uint32_t fres = 0;
                float ftile = 0.f;
                if (renderer.readOceanFoam(*ocean, foam, fres, ftile) && !foam.empty()) {
                    // Coverage as the shade reads it: clamp(1.3 foam, 0, 1)
                    // (deferred_shade_50_water_glass.glsl).
                    double cov = 0, raw = 0, half = 0;
                    for (float v: foam) {
                        const double c = std::clamp(1.3 * double(v), 0.0, 1.0);
                        cov += c;
                        raw += v;
                        if (c > 0.5) half += 1;
                    }
                    const double nf = double(foam.size());
                    foamCov += cov / nf, foamRaw += raw / nf, foamHalf += half / nf;
                    ++foamReads;
                }
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
                    }
                    const double m00 = 1 + chop * axx, m01 = chop * axz;
                    const double m10 = chop * azx, m11 = 1 + chop * azz;
                    const double J = m00 * m11 - m01 * m10;
                    const double J01 = (1 + chop * a01[0]) * (1 + chop * a01[3]) - chop * chop * a01[1] * a01[2];
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
        const double tp = 1.0 / fp;
        out.push_back({"Tp (gauge PSD peak, 0.01 Hz bands)", tp, 0.9 * tpT, 1.1 * tpT, fmt("%.2f s", tpT), "+-10%",
                       tpSrc});
        out.push_back({"  Tp of the strongest single line", 1.0 / peakOf(psd), 1, 0, "", "", "the tile's discrete modes"});
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
            out.push_back({"PSD tail slope (1.5-4 fp)", slope, -5.0, -4.0, "-4 to -5", "range",
                           "PM 1964 (f^-5), Toba 1973 (f^-4)"});
        }

        // ── 2. Dispersion ────────────────────────────────────────────────────
        // Target: energy on omega^2 = g k (deep water; the engine has no depth
        // term) within 5 % in omega (research_targets 1.1). Pass mark 95 % of
        // the energy (my choice; the research gives no fraction).
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
            out.push_back({"dispersion 3D FFT, in 5% band", tot > 0 ? inBand / tot : 0.0, 0.95, 1.0,
                           "w^2 = g k", ">= 95% of energy", "linear deep-water theory"});
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
            out.push_back({"dispersion fit, all modes, in 5%", ptot > 0 ? pin / ptot : 0.0, 0.95, 1.0,
                           "w^2 = g k", ">= 95% of energy", "linear deep-water theory"});
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
            out.push_back({"  shade sub-grid roughness", 0.0, 1, 0, "", "", "not counted (view-dependent LEAN)"});
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
        // factor of 3 to 10, so the pass band is half the lower curve to twice
        // the upper one, and zero below 3.7 m/s (research_targets 1.4).
        if (foamReads > 0) {
            const double wM = 3.84e-6 * std::pow(U, 3.41);
            double wC = 0;
            if (U > 3.7) wC = U <= 11.25 ? 3.18e-5 * std::pow(U - 3.70, 3.0) : 4.82e-6 * std::pow(U + 1.98, 3.0);
            const double lo = U > 3.7 ? 0.5 * std::min(wM, wC) : 0.0;
            const double hi = U > 3.7 ? 2.0 * std::max(wM, wC) : 1e-4;
            out.push_back({"whitecap coverage (foam, shade's)", foamCov / foamReads, lo, hi,
                           fmt("%.2f", 100 * std::min(wM, wC)) + "-" + fmt("%.2f %%", 100 * std::max(wM, wC)),
                           "x0.5-2", "Monahan 1980, Callaghan 2008"});
            out.push_back({"  foam value mean (raw)", foamRaw / foamReads, 1, 0, "", "", "foam_world.comp"});
            out.push_back({"  area with coverage > 0.5", foamHalf / foamReads, 1, 0, "", "", ""});
        } else {
            out.push_back({"whitecap coverage (foam, shade's)", -1, 1, 0, "", "", "no foam readback"});
        }

        out.push_back({"stale readback frames", double(staleFrames), 0, 0, "0", "", "harness self-check"});

        // ── Print ────────────────────────────────────────────────────────────
        std::printf("\n%-42s %12s  %-14s %-16s %-6s %s\n", "metric", "measured", "target", "tolerance", "", "source");
        bool anyFail = false;
        for (const auto& m: out) {
            const char* v = m.graded() ? (m.pass() ? "PASS" : "FAIL") : "INFO";
            if (m.graded() && !m.pass()) anyFail = true;
            std::printf("%-42s %12.5g  %-14s %-16s %-6s %s\n", m.name.c_str(), m.measured, m.target.c_str(),
                        m.tolerance.c_str(), v, m.source.c_str());
        }
        std::fflush(stdout);
        if (!cfg.json.empty()) {
            std::ofstream js(cfg.json, std::ios::app);
            for (const auto& m: out) {
                std::string name = m.name;
                name.erase(0, name.find_first_not_of(' '));
                char b[1024];
                std::snprintf(b, sizeof(b),
                              "{\"wind\":%.2f,\"fetch\":%.0f,\"choppiness\":%.3f,\"metric\":\"%s\",\"measured\":%.6g,"
                              "\"lo\":%.6g,\"hi\":%.6g,\"target\":\"%s\",\"tolerance\":\"%s\",\"verdict\":\"%s\","
                              "\"source\":\"%s\"}",
                              U, double(cfg.fetch), chop, name.c_str(), m.measured, m.lo, m.hi, m.target.c_str(),
                              m.tolerance.c_str(), m.graded() ? (m.pass() ? "PASS" : "FAIL") : "INFO",
                              m.source.c_str());
                js << b << "\n";
            }
        }
        return (cfg.strict && anyFail) ? 1 : 0;
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
        else if (const char* s = val("--json")) cfg.json = s;
        else if (std::strcmp(argv[i], "--strict") == 0) cfg.strict = true;
        else {
            std::printf("unknown argument %s (see the usage block in ocean_metrics.cpp)\n", argv[i]);
            return 2;
        }
    }
    if (cfg.hz <= 0 || cfg.seconds <= 0 || cfg.wind <= 0) return 2;
    if (single) return runCase(cfg);

    // The sweep: each case in its own process, so no renderer state carries
    // from one ocean to the next. Every other flag is passed through.
    std::string pass;
    for (int i = 1; i < argc; ++i) pass += std::string(" \"") + argv[i] + "\"";
    int worst = 0;
    for (float fetch: {0.f, 20000.f}) {
        for (float u: {7.f, 10.f, 15.f}) {
            char b[128];
            std::snprintf(b, sizeof(b), " --wind %.1f --fetch %.0f", u, fetch);
#ifdef _WIN32
            // cmd.exe strips the outer quotes of a /c line that starts with one.
            const std::string cmd = std::string("\"\"") + argv[0] + "\"" + b + pass + "\"";
#else
            const std::string cmd = std::string("\"") + argv[0] + "\"" + b + pass;
#endif
            std::fflush(stdout);
            const int rc = std::system(cmd.c_str());
            if (rc == kSkipCode) return kSkipCode;
            worst = std::max(worst, rc);
        }
    }
    return worst;
}
