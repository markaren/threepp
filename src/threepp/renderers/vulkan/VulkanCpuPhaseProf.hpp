// CPU phase profiler for the scene-build / frame-prep path. Enabled by
// THREEPP_CPU_PHASE_PROFILE=1; zero-cost (one bool test per scope) when off.
// Prints per-phase averages to stderr every THREEPP_CPU_PHASE_WINDOW frames
// (default 300).
//
// This is the instrument that located the per-instance CPU costs the entry-
// span rework removed (frustum cull, motion matrices, indirect build,
// emissive walk — see EntrySpan in VulkanSceneTypes.hpp). Kept behind the
// env var so a future "the renderer got slow at N instances" report can be
// broken down per phase without re-instrumenting; not part of the public
// surface (FrameTimings carries the coarse cpu* numbers).
//
// WINDOWED, not cumulative. The accumulator is cleared after every print, so
// each block is the mean over THAT window alone. It used to divide a
// never-cleared accumulator by the total frame count, which made every print a
// running mean since process start: the first frame's pipeline-compile spike
// stayed baked into every later block (decaying only as 1/frames), and two
// prints could not be compared as before/after of anything. A population that
// ramps (grains pouring onto a belt) needs the LAST window to describe the
// LAST population, which a running mean cannot do.
//
// FLAT, not hierarchical: a nested Scope adds its milliseconds to its own key
// AND to every enclosing key, so a set of phases is only summable if it is
// mutually exclusive. Use Scope::stop() to end a phase early rather than
// nesting a second scope inside a function-scope one — that is what keeps
// frame.A_motionMats / frame.G_uploadMotion and frame.D_buildIndirect /
// frame.F_uploadDrawInfo siblings instead of parent/child.

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
        unsigned window = 300;// frames per printed block
        unsigned frames = 0;  // frames in the CURRENT window
        unsigned blocks = 0;  // windows printed so far
        std::map<std::string, double> acc;

        static Registry& get() {
            static Registry r = [] {
                Registry x;
                const char* e = std::getenv("THREEPP_CPU_PHASE_PROFILE");
                x.on = e && e[0] == '1';
                if (const char* w = std::getenv("THREEPP_CPU_PHASE_WINDOW")) {
                    const int n = std::atoi(w);
                    if (n > 0) x.window = static_cast<unsigned>(n);
                }
                return x;
            }();
            return r;
        }

        void endFrame() {
            if (!on) return;
            if (++frames < window) return;
            ++blocks;
            std::fprintf(stderr, "[cpuprof] block %u: avg ms/frame over %u frames\n",
                         blocks, frames);
            for (auto& [k, v] : acc)
                std::fprintf(stderr, "  %-34s %9.4f\n", k.c_str(), v / frames);
            std::fflush(stderr);
            // Windowed: start the next block from zero (see the header note).
            acc.clear();
            frames = 0;
        }
    };

    struct Scope {
        const char* name;
        std::chrono::high_resolution_clock::time_point t0;
        bool live = false;
        explicit Scope(const char* n) : name(n) {
            if (Registry::get().on) {
                live = true;
                t0 = std::chrono::high_resolution_clock::now();
            }
        }
        // End the phase before the enclosing block does. The point of this is
        // splitting a function-scope phase into siblings without hoisting every
        // local the second half reads: stop() where the first half ends, then
        // open a plain braced scope for the second. Idempotent, and the
        // destructor is a no-op afterwards — so an early `return` between the
        // stop() and the sibling scope still accounts correctly.
        void stop() {
            if (!live) return;
            live = false;
            Registry::get().acc[name] +=
                    std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t0)
                            .count();
        }
        ~Scope() { stop(); }
    };

}// namespace threepp::vulkan::cpuprof

#define THREEPP_CPUPROF_CAT2(a, b) a##b
#define THREEPP_CPUPROF_CAT(a, b) THREEPP_CPUPROF_CAT2(a, b)
#define THREEPP_CPUPROF(name) \
    ::threepp::vulkan::cpuprof::Scope THREEPP_CPUPROF_CAT(_cpuprof_, __COUNTER__)(name)
// Named variant, for the phases that must be stop()ped early to stay siblings
// of the phase that follows them in the same function.
#define THREEPP_CPUPROF_NAMED(var, name) \
    ::threepp::vulkan::cpuprof::Scope var(name)

#endif// THREEPP_VULKAN_CPU_PHASE_PROF_HPP
