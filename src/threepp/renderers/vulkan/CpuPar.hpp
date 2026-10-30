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
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <future>
#include <numeric>
#include <thread>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace threepp::vulkan::par {

    // What the work on this thread is FOR, carried by every fork below from
    // the thread that forks to the threads that run its pieces. The renderer
    // keeps the view a frame prep runs for here (VulkanRenderer::Impl::view),
    // so a view's prep can run on a pool thread, and the chunked walks it
    // forks in turn, while another view's prep runs beside it. Null on every
    // thread that runs no such work, and then nothing changes.
    inline void*& threadContext() {
        thread_local void* context = nullptr;
        return context;
    }

    // Sets threadContext() for a scope and puts back what was there.
    class ContextScope {
    public:
        explicit ContextScope(void* context): saved_(threadContext()) { threadContext() = context; }
        ~ContextScope() { threadContext() = saved_; }
        ContextScope(const ContextScope&)            = delete;
        ContextScope& operator=(const ContextScope&) = delete;

    private:
        void* saved_;
    };

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

    // Runs fn(chunk) for chunk in [0, chunks): in parallel when `parallel`
    // and there is more than one chunk, else in order on this thread.
    template<class Fn>
    void runIf(bool parallel, unsigned chunks, Fn&& fn) {
        if (chunks <= 1u || !parallel) {
            for (unsigned c = 0; c < chunks; ++c) fn(c);
            return;
        }
        thread_local std::vector<unsigned> ids;
        ids.resize(chunks);
        std::iota(ids.begin(), ids.end(), 0u);
        void* const context = threadContext();
        threepp::parallelForEach(ids.begin(), ids.end(), [&](unsigned c) {
            ContextScope scope(context);
            fn(c);
        });
    }

    // runIf under THREEPP_VK_CPU_PAR.
    template<class Fn>
    void run(unsigned chunks, Fn&& fn) {
        runIf(config().on, chunks, std::forward<Fn>(fn));
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
    // Waits for `done` to come true by spinning on this thread (bounded),
    // then returns whether it did. The waits here are a fraction of a
    // millisecond, and a thread that sleeps through one wakes on whichever
    // processor the scheduler has free: measured 2026-10-30 on a 5950X (two
    // core complexes, an L3 each), the frame thread then crossed to the other
    // complex on 4-7 % of frames instead of 1 %, and ran every later phase of
    // the frame, the next frame's scene sync included, against a cold L3
    // (+0.3-0.5 ms per frame). A machine with fewer than four hardware
    // threads does not spin: a job that has no processor to run on while
    // this thread spins would only wait longer.
    template<class Done>
    bool spinUntil(Done&& done) {
        static const bool spin = std::thread::hardware_concurrency() >= 4u;
        if (!spin) return done();
        const auto t0 = std::chrono::steady_clock::now();
        while (!done()) {
            for (int i = 0; i < 64; ++i) {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
                _mm_pause();
#else
                std::this_thread::yield();
#endif
            }
            if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(8)) return done();
        }
        return true;
    }

    template<class... Fns>
    void tasks(bool on, Fns&&... fns) {
        if (!on) {
            (fns(), ...);
            return;
        }
        constexpr unsigned     n = sizeof...(Fns);
        std::function<void()>  fs[n] = {std::function<void()>(std::forward<Fns>(fns))...};
        std::exception_ptr     errs[n] = {};
        void* const            context = threadContext();
#if defined(_MSC_VER)
        // The other callables are offered to the pool (std::async, the
        // Windows thread pool); this thread runs the first, then runs
        // whichever of the others no pool thread has claimed yet, then spins
        // until the rest have returned (spinUntil), so it never leaves its
        // processor for the join and never waits for a pool thread to appear
        // (the pool may be busy with the frame's uploads: on the entry-churn
        // test's add frame, waiting cost 1-4 ms). Before 2026-10-30 the fork
        // went through the parallel STL, whose caller sleeps for the tail of
        // the other task.
        std::atomic<bool>     claimed[n] = {};
        std::atomic<unsigned> remaining{n - 1u};
        std::future<void>     pend[n > 1u ? n - 1u : 1u];
        const auto            runOne = [&](unsigned i) {
            try {
                fs[i]();
            } catch (...) {
                errs[i] = std::current_exception();
            }
            remaining.fetch_sub(1u, std::memory_order_release);
        };
        for (unsigned i = 1; i < n; ++i)
            pend[i - 1u] = std::async(std::launch::async, [&, i] {
                if (claimed[i].exchange(true, std::memory_order_acq_rel)) return;
                ContextScope scope(context);
                runOne(i);
            });
        try {
            fs[0]();
        } catch (...) {
            errs[0] = std::current_exception();
        }
        for (unsigned i = 1; i < n; ++i)
            if (!claimed[i].exchange(true, std::memory_order_acq_rel)) runOne(i);
        spinUntil([&] { return remaining.load(std::memory_order_acquire) == 0u; });
        for (unsigned i = 1; i < n; ++i) pend[i - 1u].get();
#else
        // No parallel pool on this compiler (Parallel.hpp): in order, here.
        for (unsigned i = 0; i < n; ++i) {
            ContextScope scope(context);
            try {
                fs[i]();
            } catch (...) {
                errs[i] = std::current_exception();
            }
        }
#endif
        for (auto& e : errs)
            if (e) std::rethrow_exception(e);
    }

    // The secondary views' frame prep (camera UBOs, cull, draw records,
    // shadow maps) runs on the pool, every view at once and beside the
    // primary's graph build, instead of view after view between the
    // primary's build and theirs (VulkanCoreFrame, addSecondaryViewPasses).
    // THREEPP_VK_CPU_VIEWS=0 keeps the serial loop (on by default since
    // 2026-10-30, once join() stopped sleeping; see Group below).
    inline bool viewsOn() {
        static const bool on = env::flag("THREEPP_VK_CPU_VIEWS", true);
        return on;
    }

    // A pass whose recording is long and splits (the overlay depth prepass's
    // one draw per entry) records its pieces into secondary command buffers
    // on the pool, which the frame's command buffer then executes in order:
    // the same commands, recorded at once. THREEPP_VK_CPU_RECORD=0 records
    // them inline as before (default on since 2026-10-30: the pine forest
    // drive's prepass 0.72 -> 0.30 ms). Chunked as the walks above
    // (THREEPP_VK_CPU_PAR_CHUNKS caps the count; THREEPP_VK_CPU_PAR_MIN the
    // fewest items a piece takes), but not gated by THREEPP_VK_CPU_PAR.
    inline bool recordOn() {
        static const bool on = env::flag("THREEPP_VK_CPU_RECORD", true);
        return on;
    }
    inline unsigned recordChunksFor(size_t items) {
        if (!recordOn()) return 1u;
        const unsigned hw  = std::max(1u, std::thread::hardware_concurrency());
        unsigned       cap = std::min(hw, 16u);
        if (const auto n = env::integer("THREEPP_VK_CPU_PAR_CHUNKS"); n && *n > 0)
            cap = static_cast<unsigned>(std::min<long long>(*n, 64));
        const size_t minItems = config().minItems;
        if (items < minItems * 2u) return 1u;
        return static_cast<unsigned>(std::min<size_t>(cap, items / minItems));
    }

    // Jobs started together and joined later, while the starting thread
    // does its own work in between: on MSVC one std::async task that runs
    // them as a parallel loop; elsewhere (no parallel STL, see Parallel.hpp)
    // they run at start(), one after the other. Each job sets its own
    // threadContext(). An exception in a job is rethrown by join() once
    // every job has returned.
    //
    // join() spins (bounded) rather than sleeps while the jobs finish. The
    // wait is a fraction of a millisecond, and a thread that sleeps in it
    // wakes on whichever processor the scheduler has free: measured
    // 2026-10-30 on a 5950X (two core complexes, an L3 each), the frame
    // thread then crossed to the other complex on 4-7 % of frames instead
    // of 1 %, and ran every later phase of the frame, the next frame's
    // scene sync included, against a cold L3 (+0.3-0.5 ms per frame).
    class Group {
    public:
        Group() = default;
        Group(const Group&)            = delete;
        Group& operator=(const Group&) = delete;
        ~Group() {
            // Never leave a job running against state this frame tears down.
            if (pending_.valid()) pending_.wait();
        }

        void start(std::vector<std::function<void()>> jobs) {
            jobs_ = std::move(jobs);
            errs_.assign(jobs_.size(), nullptr);
            claimed_ = std::vector<std::atomic<bool>>(jobs_.size());
            remaining_.store(static_cast<unsigned>(jobs_.size()), std::memory_order_relaxed);
#if defined(_MSC_VER)
            pending_ = std::async(std::launch::async, [this] { runAll(); });
#else
            runAll();
#endif
        }

        // Runs here whatever job no pool thread has claimed yet (the pool
        // may be busy, or late to start the task), spins until the rest
        // have returned (spinUntil), then rethrows a job's exception.
        void join() {
            if (pending_.valid()) {
                for (unsigned i = 0; i < jobs_.size(); ++i) runIfUnclaimed(i);
                spinUntil([this] { return remaining_.load(std::memory_order_acquire) == 0u; });
                pending_.get();
            }
            for (auto& e : errs_)
                if (e) std::rethrow_exception(e);
        }

    private:
        // Each job runs exactly once, on whichever thread claims it first.
        void runIfUnclaimed(unsigned i) {
            if (claimed_[i].exchange(true, std::memory_order_acq_rel)) return;
            try {
                jobs_[i]();
            } catch (...) {
                errs_[i] = std::current_exception();
            }
            remaining_.fetch_sub(1u, std::memory_order_release);
        }

        void runAll() {
            std::vector<unsigned> ids(jobs_.size());
            std::iota(ids.begin(), ids.end(), 0u);
            threepp::parallelForEach(ids.begin(), ids.end(), [this](unsigned i) { runIfUnclaimed(i); });
        }

        std::vector<std::function<void()>> jobs_;
        std::vector<std::exception_ptr>     errs_;
        std::vector<std::atomic<bool>>      claimed_;
        std::future<void>                   pending_;
        std::atomic<unsigned>               remaining_{0};
    };

}// namespace threepp::vulkan::par

#endif// THREEPP_VULKAN_CPU_PAR_HPP
