#ifndef THREEPP_WGPUPATHTRACERGGXLUT_HPP
#define THREEPP_WGPUPATHTRACERGGXLUT_HPP

// Private header — precomputes the single-scattering GGX white-furnace
// response E(cosθ_o, α) on the CPU. At runtime the compute shader samples
// this LUT to apply Turquin-2018-style multi-scattering energy compensation
// in the specular BRDF branch. Without the compensation, rough GGX loses
// ~30% of its energy at α=1 because VNDF-sampled reflections can fall below
// the surface (cos < 0) and are dropped.
//
// Axes of the LUT (values in [0,1]):
//   u = cos θ_o          (view angle, column index / (N-1) at cell center)
//   v = α = roughness²   (GGX α param; matches what the shader passes as
//                         `alpha` into sampleVNDF / ggxG1)
// Value stored is the Monte-Carlo estimate of E with F0 = 1 (conductor).
// Mirrors the exact estimator used in the shader spec branch:
//   estimator per sample = G1(wi) when cos_wi > 0, else 0

#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace threepp::wgpu_pt {

    /// Smith G1 for GGX (matches shader ggxG1 with a2 = alpha²).
    inline float ggxG1Cpu(float NdotX, float alpha) {
        const float a2 = alpha * alpha;
        const float denom = NdotX + std::sqrt(a2 + (1.f - a2) * NdotX * NdotX);
        return 2.f * NdotX / std::max(denom, 1e-6f);
    }

    /// VNDF sampling (Heitz 2018) — mirrors the shader's sampleVNDF exactly
    /// when wo.z >= 0. Returns reflected direction `wi` (in local frame where
    /// normal = +Z). Sets `ok = false` if the sample would land below the
    /// surface (cos_wi ≤ 0), which the shader drops via `break`.
    inline void sampleVndfCpu(const Vector3& wo, float alpha, float u1, float u2,
                               Vector3& wi, bool& ok) {
        // Stretch to isotropic configuration
        Vector3 woStr(alpha * wo.x, alpha * wo.y, wo.z);
        woStr.normalize();
        const float lensq = woStr.x * woStr.x + woStr.y * woStr.y;
        Vector3 T1;
        if (lensq > 1e-7f) {
            const float invLen = 1.f / std::sqrt(lensq);
            T1.set(-woStr.y * invLen, woStr.x * invLen, 0.f);
        } else {
            T1.set(1.f, 0.f, 0.f);
        }
        Vector3 T2;
        T2.crossVectors(woStr, T1);
        // Project disk sample
        const float r = std::sqrt(u1);
        const float phi = 6.2831853071796f * u2;
        const float t1s = r * std::cos(phi);
        const float s = 0.5f * (1.f + woStr.z);
        const float t2s_uniform = std::sqrt(std::max(0.f, 1.f - t1s * t1s));
        const float t2s_disk = r * std::sin(phi);
        const float t2s = (1.f - s) * t2s_uniform + s * t2s_disk;
        const float tz = std::sqrt(std::max(0.f, 1.f - t1s * t1s - t2s * t2s));
        const Vector3 nhLocal(T1.x * t1s + T2.x * t2s + woStr.x * tz,
                              T1.y * t1s + T2.y * t2s + woStr.y * tz,
                              T1.z * t1s + T2.z * t2s + woStr.z * tz);
        Vector3 hLocal(alpha * nhLocal.x, alpha * nhLocal.y,
                       std::max(1e-6f, nhLocal.z));
        hLocal.normalize();
        // Reflect wo about hLocal
        const float dot_wo_h = wo.dot(hLocal);
        wi.set(2.f * dot_wo_h * hLocal.x - wo.x,
               2.f * dot_wo_h * hLocal.y - wo.y,
               2.f * dot_wo_h * hLocal.z - wo.z);
        ok = (wi.z > 0.f);
    }

    /// Build a size N×N table of E(cos_o, α) for F0 = 1 (conductor).
    /// Column index = cos_o bin, row index = α bin. Returns N*N floats.
    /// Computation time: ~100ms at N=32, S=1024 on one thread.
    inline std::vector<float> buildGgxELut(int N = 32, int S = 1024) {
        std::vector<float> E(static_cast<std::size_t>(N) * N, 0.f);
        const int sqrtS = static_cast<int>(std::round(std::sqrt(static_cast<float>(S))));
        const int samples = sqrtS * sqrtS;
        for (int ai = 0; ai < N; ++ai) {
            // α ∈ (0, 1]. Avoid α=0 (degenerate mirror — VNDF is a delta).
            const float alpha = std::max(1e-3f, static_cast<float>(ai + 1) / static_cast<float>(N));
            for (int ci = 0; ci < N; ++ci) {
                const float cos_o = std::max(1e-3f, (static_cast<float>(ci) + 0.5f) / static_cast<float>(N));
                const float sin_o = std::sqrt(std::max(0.f, 1.f - cos_o * cos_o));
                const Vector3 wo(sin_o, 0.f, cos_o);
                double sum = 0.0;
                for (int i = 0; i < sqrtS; ++i) {
                    for (int j = 0; j < sqrtS; ++j) {
                        const float u1 = (static_cast<float>(i) + 0.5f) / static_cast<float>(sqrtS);
                        const float u2 = (static_cast<float>(j) + 0.5f) / static_cast<float>(sqrtS);
                        Vector3 wi;
                        bool ok = false;
                        sampleVndfCpu(wo, alpha, u1, u2, wi, ok);
                        if (ok) sum += static_cast<double>(ggxG1Cpu(wi.z, alpha));
                    }
                }
                E[static_cast<std::size_t>(ai) * N + ci] = static_cast<float>(sum / static_cast<double>(samples));
            }
        }
        return E;
    }

}// namespace threepp::wgpu_pt

#endif//THREEPP_WGPUPATHTRACERGGXLUT_HPP
