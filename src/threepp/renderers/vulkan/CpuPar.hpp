// Fork-join over chunks of the frame-prep's per-entry passes.
//
// The frame's CPU passes walk every entry once each (shadow draw bucketing,
// the TLAS instance fill, ...): at tens of thousands of instances the walks
// are the frame's CPU cost, and each one's work is independent per span or
// per entry. A pass that opts in splits its items into chunks, runs each
// chunk into its own scratch and merges the chunks in order, so the bytes it
// produces do not depend on how many chunks ran or on which thread.
//
// The chunks run on the MSVC parallel STL (threepp::parallelForEach);
// THREEPP_VK_CPU_PAR=0 runs them one after the other on the frame thread,
// through the same chunking, so the two paths produce the same bytes. On a
// compiler whose parallel policy is serial (see utils/Parallel.hpp) the knob
// changes nothing. THREEPP_VK_CPU_PAR_CHUNKS=<n> caps the chunk count
// (default: the hardware threads, at most 16); THREEPP_VK_CPU_PAR_MIN=<items>
// is the smallest chunk worth a fork (default 1024 items): fewer items run
// as one. Measured 2026-10-29 on the pine forest drive (63-81k entries):
// the shadow draw walk 0.85 -> 0.50 ms; the reference still, 12k entries,
// never forks.

#ifndef THREEPP_VULKAN_CPU_PAR_HPP
#define THREEPP_VULKAN_CPU_PAR_HPP

#include "EnvKnob.hpp"
#include "threepp/utils/Parallel.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <numeric>
#include <thread>
#include <vector>

namespace threepp::vulkan::par {

    struct Config {
        bool     on       = true;
        unsigned chunks   = 1;   // most chunks a pass splits into
        unsigned minItems = 1024;// fewest items per chunk
    };

    inline const Config& config() {
        static const Config c = [] {
            Config x;
            x.on = env::flag("THREEPP_VK_CPU_PAR", true);
            const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
            x.chunks          = std::min(hw, 16u);
            if (const auto n = env::integer("THREEPP_VK_CPU_PAR_CHUNKS"); n && *n > 0)
                x.chunks = static_cast<unsigned>(std::min<long long>(*n, 64));
            if (const auto m = env::integer("THREEPP_VK_CPU_PAR_MIN"); m && *m > 0)
                x.minItems = static_cast<unsigned>(*m);
            if (!x.on) x.chunks = 1;
            return x;
        }();
        return c;
    }

    // The chunk count for `items` items: one when the knob is off or the
    // items are too few to share.
    inline unsigned chunksFor(size_t items) {
        const Config& c = config();
        if (!c.on || items < size_t(c.minItems) * 2u) return 1u;
        return static_cast<unsigned>(std::min<size_t>(c.chunks, items / c.minItems));
    }

    // Bounds of `chunks` consecutive pieces of [0, n) items such that each
    // piece carries about the same total weight (an item's weight is what it
    // costs to visit: a span's entry count). bounds gets chunks + 1 entries,
    // bounds[0] = 0 and bounds[chunks] = n; a piece may be empty when one
    // item outweighs the target.
    template<class Weight>
    void splitByWeight(size_t n, unsigned chunks, Weight weight, std::vector<uint32_t>& bounds) {
        bounds.clear();
        bounds.push_back(0u);
        if (chunks <= 1u || n == 0) {
            bounds.push_back(static_cast<uint32_t>(n));
            return;
        }
        uint64_t total = 0;
        for (size_t i = 0; i < n; ++i) total += weight(i);
        uint64_t acc = 0;
        size_t   i   = 0;
        for (unsigned c = 1; c < chunks; ++c) {
            // The c-th cut lands where the running weight passes c/chunks of
            // the total (integer arithmetic, no rounding drift across cuts).
            const uint64_t target = total * c / chunks;
            while (i < n && acc < target) acc += weight(i++);
            bounds.push_back(static_cast<uint32_t>(i));
        }
        bounds.push_back(static_cast<uint32_t>(n));
    }

    // Runs fn(chunk) for chunk in [0, chunks): in parallel when the knob is
    // on and there is more than one chunk, else in order on this thread.
    template<class Fn>
    void run(unsigned chunks, Fn&& fn) {
        if (chunks <= 1u || !config().on) {
            for (unsigned c = 0; c < chunks; ++c) fn(c);
            return;
        }
        thread_local std::vector<unsigned> ids;
        ids.resize(chunks);
        std::iota(ids.begin(), ids.end(), 0u);
        threepp::parallelForEach(ids.begin(), ids.end(), [&](unsigned c) { fn(c); });
    }

    // The frame prep's independent chains (the cull to indirect-build chain
    // and the shadow maps' prepare, per view) run at once;
    // THREEPP_VK_CPU_TASKS=0 runs them one after the other. Measured
    // 2026-10-29 on the pine forest drive: the secondary view's prep
    // 1.73 -> 1.15 ms, the render call 11.2 -> 9.5 ms with the chunks above.
    inline bool tasksOn() {
        static const bool on = env::flag("THREEPP_VK_CPU_TASKS", true);
        return on;
    }

    // Runs the callables at once when `on`, else one after the other in
    // argument order. An exception in a task is rethrown here once every
    // task has returned (a parallel algorithm would terminate instead).
    template<class... Fns>
    void tasks(bool on, Fns&&... fns) {
        if (!on) {
            (fns(), ...);
            return;
        }
        constexpr unsigned     n = sizeof...(Fns);
        std::function<void()>  fs[n] = {std::function<void()>(std::forward<Fns>(fns))...};
        std::exception_ptr     errs[n] = {};
        unsigned               ids[n];
        std::iota(ids, ids + n, 0u);
        threepp::parallelForEach(ids, ids + n, [&](unsigned i) {
            try {
                fs[i]();
            } catch (...) {
                errs[i] = std::current_exception();
            }
        });
        for (auto& e : errs)
            if (e) std::rethrow_exception(e);
    }

}// namespace threepp::vulkan::par

#endif// THREEPP_VULKAN_CPU_PAR_HPP
