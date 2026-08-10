// CPU phase profiler for the scene-build / frame-prep path. Enabled by
// THREEPP_CPU_PHASE_PROFILE=1; zero-cost (one bool test per scope) when off.
// Prints per-phase averages to stderr every 300 frames.
//
// This is the instrument that located the per-instance CPU costs the entry-
// span rework removed (frustum cull, motion matrices, indirect build,
// emissive walk — see EntrySpan in VulkanSceneTypes.hpp). Kept behind the
// env var so a future "the renderer got slow at N instances" report can be
// broken down per phase without re-instrumenting; not part of the public
// surface (FrameTimings carries the coarse cpu* numbers).

#ifndef THREEPP_VULKAN_CPU_PHASE_PROF_HPP
#define THREEPP_VULKAN_CPU_PHASE_PROF_HPP

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace threepp::vulkan::cpuprof {

    struct Registry {
        bool on = false;
        unsigned frames = 0;
        std::map<std::string, double> acc;

        static Registry& get() {
            static Registry r = [] {
                Registry x;
                const char* e = std::getenv("THREEPP_CPU_PHASE_PROFILE");
                x.on = e && e[0] == '1';
                return x;
            }();
            return r;
        }

        void endFrame() {
            if (!on) return;
            if (++frames % 300 == 0) {
                std::fprintf(stderr, "[cpuprof] avg ms/frame over %u frames:\n", frames);
                for (auto& [k, v] : acc)
                    std::fprintf(stderr, "  %-34s %9.4f\n", k.c_str(), v / frames);
                std::fflush(stderr);
            }
        }
    };

    struct Scope {
        const char* name;
        std::chrono::high_resolution_clock::time_point t0;
        explicit Scope(const char* n) : name(n) {
            if (Registry::get().on) t0 = std::chrono::high_resolution_clock::now();
        }
        ~Scope() {
            auto& r = Registry::get();
            if (!r.on) return;
            r.acc[name] += std::chrono::duration<double, std::milli>(
                                   std::chrono::high_resolution_clock::now() - t0)
                                   .count();
        }
    };

}// namespace threepp::vulkan::cpuprof

#define THREEPP_CPUPROF_CAT2(a, b) a##b
#define THREEPP_CPUPROF_CAT(a, b) THREEPP_CPUPROF_CAT2(a, b)
#define THREEPP_CPUPROF(name) \
    ::threepp::vulkan::cpuprof::Scope THREEPP_CPUPROF_CAT(_cpuprof_, __COUNTER__)(name)

#endif// THREEPP_VULKAN_CPU_PHASE_PROF_HPP
