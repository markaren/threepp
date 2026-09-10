
#include "EnvSunExtract.hpp"

#include <algorithm>
#include <cmath>

namespace {

    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kTwoPi = 6.28318530717958647692f;
    constexpr float kDeg2Rad = kPi / 180.f;

    inline float texelLum(const float* p) {
        return 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2];
    }

    // Direction + solid angle of equirect texel (x, y). MUST match the shaders'
    // sampleEnvLod mapping (u = 0.5 + atan(z,x)/2π, v = 0.5 + asin(y)/π), or
    // the re-injected analytic sun points away from the disc it replaced.
    inline void texelDir(uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                         float out[3], float* dOmega) {
        const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(w);
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(h);
        const float phi = (u - 0.5f) * kTwoPi;
        const float lat = (v - 0.5f) * kPi;
        const float cl = std::cos(lat);
        out[0] = cl * std::cos(phi);
        out[1] = std::sin(lat);
        out[2] = cl * std::sin(phi);
        if (dOmega) *dOmega = cl * (kTwoPi / static_cast<float>(w)) * (kPi / static_cast<float>(h));
    }

}// namespace

namespace threepp {

    bool extractEnvSun(uint32_t w, uint32_t h, const float* pixels,
                       std::vector<float>& clamped, EnvSunExtract& out) {

        out = {};
        clamped.clear();

        // ── HDRI sun extraction (CPU, one-shot per env upload) ──────────────
        // Detect a dominant compact bright source and build a sun-clamped copy
        // to prefilter from. Detection: the peak texel must dominate the mean
        // luminance by 200× (bright clouds/sky bands sit at ~10-50×) and clear
        // an absolute HDR floor. The disc: every texel within 8° of the peak
        // whose luminance exceeds 5% of it. Fill: the mean of the 8°-16°
        // surround, so the replaced disc blends into the sky gradient (the
        // sub-threshold halo/glow around the sun stays in the env — desirable).
        if (!pixels || w <= 1 || h <= 1) return false;

        const size_t n = static_cast<size_t>(w) * h;
        float maxL = 0.f;
        size_t maxI = 0;
        double sumL = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const float L = texelLum(pixels + 4 * i);
            sumL += L;
            if (L > maxL) { maxL = L; maxI = i; }
        }
        const float meanL = static_cast<float>(sumL / static_cast<double>(n));
        if (maxL <= std::max(200.f * meanL, 32.f)) return false;

        float pd[3];
        texelDir(static_cast<uint32_t>(maxI % w), static_cast<uint32_t>(maxI / w), w, h, pd, nullptr);
        const float cosRegion = std::cos(8.f * kDeg2Rad);
        const float cosFill = std::cos(16.f * kDeg2Rad);
        const float thresh = 0.05f * maxL;
        double fill[3]{};
        size_t fillN = 0;
        std::vector<uint32_t> disc;
        for (uint32_t y = 0; y < h; ++y) {
            for (uint32_t x = 0; x < w; ++x) {
                float d[3], dO;
                texelDir(x, y, w, h, d, &dO);
                const float c = d[0] * pd[0] + d[1] * pd[1] + d[2] * pd[2];
                if (c <= cosFill) continue;
                const float* p = pixels + 4 * (static_cast<size_t>(y) * w + x);
                if (c > cosRegion && texelLum(p) > thresh) {
                    disc.push_back(y * w + x);
                } else {
                    for (int k = 0; k < 3; ++k) fill[k] += p[k];
                    ++fillN;
                }
            }
        }
        const float f[3] = {
                fillN ? static_cast<float>(fill[0] / static_cast<double>(fillN)) : 0.f,
                fillN ? static_cast<float>(fill[1] / static_cast<double>(fillN)) : 0.f,
                fillN ? static_cast<float>(fill[2] / static_cast<double>(fillN)) : 0.f};
        // Energy accounting: the disc texels are REPLACED by the fill, so
        // that much radiance stays in the env — the analytic light must
        // carry only the EXCESS Σ max(L−fill,0)·dΩ, or the sun is counted
        // ~twice and every lit surface brightens (washed-out pastels).
        double E[3]{}, dAcc[3]{}, omega = 0.0;
        for (const uint32_t idx : disc) {
            float d[3], dO;
            texelDir(idx % w, idx / w, w, h, d, &dO);
            const float* p = pixels + 4 * static_cast<size_t>(idx);
            const float ex[3] = {std::max(p[0] - f[0], 0.f),
                                 std::max(p[1] - f[1], 0.f),
                                 std::max(p[2] - f[2], 0.f)};
            const float exL = texelLum(ex);
            for (int k = 0; k < 3; ++k) {
                E[k] += static_cast<double>(ex[k]) * dO;
                dAcc[k] += static_cast<double>(d[k]) * exL * dO;
            }
            omega += dO;
        }
        const double dl = std::sqrt(dAcc[0] * dAcc[0] + dAcc[1] * dAcc[1] + dAcc[2] * dAcc[2]);
        if (omega <= 0.0 || disc.empty() || dl <= 1e-12 ||
            (E[0] + E[1] + E[2]) <= 1e-4) return false;

        clamped.assign(pixels, pixels + 4 * n);
        for (const uint32_t idx : disc) {
            clamped[4 * static_cast<size_t>(idx) + 0] = f[0];
            clamped[4 * static_cast<size_t>(idx) + 1] = f[1];
            clamped[4 * static_cast<size_t>(idx) + 2] = f[2];
        }
        // CIRCUMSOLAR-HALO CAP. The disc replacement above deliberately
        // leaves the sub-threshold halo in the env, but on a bright HDRI
        // that halo still sits hundreds of × above the mean sky — and the
        // Monte-Carlo GGX prefilter (finite samples/texel) renders such
        // texels into the mips as isolated HOT TEXELS. Every glossy env
        // lookup then bilinearly magnifies them into soft SQUARE blobs
        // scattered around the sun (the water sun-glitter "blocky
        // fireflies", present on every backend because they are baked
        // into the mip DATA). Soft-cap the prefilter SOURCE at 64× mean
        // luminance — bright clouds/sky bands sit at 10–50× (see the
        // detection comment above) and survive untouched — and fold the
        // removed energy into the analytic sun (it is circumsolar
        // radiance; direction-weighting preserves the aggregate
        // direction), so total illumination is conserved. Mip 0 keeps the
        // original pixels either way: backgrounds and true mirrors still
        // see the real sun and halo.
        const float capL = 64.f * meanL;
        for (size_t i = 0; i < n; ++i) {
            float* p = clamped.data() + 4 * i;
            const float L = texelLum(p);
            if (L <= capL) continue;
            float d[3], dO;
            texelDir(static_cast<uint32_t>(i % w), static_cast<uint32_t>(i / w), w, h, d, &dO);
            const float s = capL / L;
            const float ex[3] = {p[0] * (1.f - s), p[1] * (1.f - s), p[2] * (1.f - s)};
            const float exL = texelLum(ex);
            for (int k = 0; k < 3; ++k) {
                E[k] += static_cast<double>(ex[k]) * dO;
                dAcc[k] += static_cast<double>(d[k]) * exL * dO;
            }
            for (int k = 0; k < 3; ++k) p[k] *= s;
        }
        const double dl2 = std::sqrt(dAcc[0] * dAcc[0] + dAcc[1] * dAcc[1] + dAcc[2] * dAcc[2]);
        for (int k = 0; k < 3; ++k) {
            out.dir[k] = static_cast<float>(dAcc[k] / (dl2 > 1e-12 ? dl2 : 1.0));
            out.colorE[k] = static_cast<float>(E[k]);
        }
        out.angularRadiusDeg = std::clamp(
                std::sqrt(static_cast<float>(omega) / kPi) / kDeg2Rad, 0.05f, 10.f);
        out.found = true;
        return true;
    }

}// namespace threepp
