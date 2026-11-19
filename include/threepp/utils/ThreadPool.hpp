// A small persistent thread pool for fork-join loops: parallelFor(begin, end, fn) calls fn(i) for every i in
// [begin, end) on the pool's threads and the calling thread, and returns when every call has returned.
//
// It exists for loops that fork many times a frame (a simulation's passes, a renderer's per-entry walks, each a few
// hundred microseconds): its workers stay awake for a moment after a loop (spinning, then sleeping on a condition
// variable), so the next fork costs a few microseconds where std::async costs tens. The items are handed out in chunks
// from an atomic counter, so uneven items balance themselves.
//
// Several loops can be open at once: one per caller, up to kSlots, from any threads, and a loop's fn may open a loop
// of its own (nested). The workers take chunks from whichever loops are open; each caller works its own loop and then
// waits only for the workers still inside one of its chunks. Past kSlots open loops a new one runs in order on its
// caller's thread.
//
// Contract, as threepp::parallelForEach's: each fn(i) writes only its own state, so the result never depends on the
// number of threads or on which thread ran which item. An exception from fn on the calling thread is passed on once
// the workers have let go of the loop; one on a worker terminates, so fn should not throw.
//
// - With no hardware threads to spare, or on a platform without threads (the web build without pthreads), every loop
//   runs in order on the calling thread.
// - ThreadPool::shared() is one pool per module; it is never destroyed (at exit the operating system ends its threads;
//   joining them from a static destructor can hang a DLL's unload). THREEPP_THREADS sets its total thread count (the
//   caller included; 1: no workers), THREEPP_THREADS_SPIN_US how long its workers spin after a loop.

#ifndef THREEPP_THREADPOOL_HPP
#define THREEPP_THREADPOOL_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace threepp {

    class ThreadPool {

    public:
        static constexpr int kSlots = 16;

        // `workers` threads besides the callers (default: one less than the hardware threads, at most 15); `spin`:
        // how long a worker spins for the next loop before it sleeps
        explicit ThreadPool(int workers = -1, std::chrono::microseconds spin = std::chrono::microseconds(200))
            : spin_(spin) {
            if (workers < 0) workers = std::min(15, static_cast<int>(std::thread::hardware_concurrency()) - 1);
#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)
            workers = 0;
#endif
            workers = std::max(workers, 0);
            threads_.reserve(workers);
            // every worker starts from the generation as it is now: one that comes up late must still see the loops
            // and the stop posted before it ran (reading the counter on its own first look, it would wait for good)
            const uint64_t start = generation_.load();
            for (int k = 0; k < workers; ++k) threads_.emplace_back([this, start] { workerLoop(start); });
        }

        ~ThreadPool() {
            stop_.store(true);
            wake();
            for (auto& t : threads_) t.join();
        }

        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;

        // the threads a loop can run on: the workers and the caller
        [[nodiscard]] int concurrency() const {
            return static_cast<int>(threads_.size()) + 1;
        }

        // fn(i) for every i in [begin, end); `grain`: the fewest items a thread takes at a time (0: about eight chunks
        // per thread)
        template<class Fn>
        void parallelFor(int begin, int end, Fn&& fn, int grain = 0) {
            if (end <= begin) return;
            const int n = end - begin;
            if (grain <= 0) grain = std::max(1, n / (8 * concurrency()));
            Slot* slot = threads_.empty() || n <= grain ? nullptr : claimSlot();
            if (!slot) {
                for (int i = begin; i < end; ++i) fn(i);
                return;
            }

            auto call = [&fn](int i0, int i1) {
                for (int i = i0; i < i1; ++i) fn(i);
            };
            Job job;
            job.ctx = &call;
            job.run = [](void* ctx, int i0, int i1) { (*static_cast<decltype(call)*>(ctx))(i0, i1); };
            job.next.store(begin, std::memory_order_relaxed);
            job.end = end;
            job.grain = grain;

            // closes the loop however this scope is left (an exception from fn included): no worker joins it after
            // this, the ones inside one of its chunks finish first, and only then is the slot free again
            struct Close {
                Slot& slot;
                ~Close() {
                    slot.job.store(closing());
                    while (slot.refs.load() != 0) pause();
                    slot.job.store(nullptr);
                }
            } close{*slot};

            slot->job.store(&job);
            wake();
            work(job);
        }

        // one pool per module, never destroyed (THREEPP_THREADS, THREEPP_THREADS_SPIN_US)
        static ThreadPool& shared() {
            static ThreadPool* pool = [] {
                int workers = -1;
                if (const char* t = std::getenv("THREEPP_THREADS"); t && std::atoi(t) > 0) workers = std::atoi(t) - 1;
                auto spin = std::chrono::microseconds(200);
                if (const char* s = std::getenv("THREEPP_THREADS_SPIN_US"); s && std::atoi(s) >= 0) spin = std::chrono::microseconds(std::atoi(s));
                return new ThreadPool(workers, spin);
            }();
            return *pool;
        }

    private:
        struct Job {
            void* ctx = nullptr;
            void (*run)(void*, int, int) = nullptr;
            std::atomic<int> next{0};
            int end = 0;
            int grain = 1;
        };

        // A loop's place: `job` is null (free), closing() (its caller is waiting for the workers inside it) or the
        // open job; `refs` counts the workers that have looked into it
        struct Slot {
            alignas(64) std::atomic<Job*> job{nullptr};
            std::atomic<int> refs{0};
        };

        std::vector<std::thread> threads_;
        std::array<Slot, kSlots> slots_;
        std::mutex sleep_;
        std::condition_variable wakeup_;
        std::atomic<uint64_t> generation_{0};
        std::atomic<int> sleeping_{0};
        std::atomic<bool> stop_{false};
        std::chrono::microseconds spin_;

        static Job* closing() {
            static Job marker;
            return &marker;
        }

        static void pause() {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
            _mm_pause();
#else
            std::this_thread::yield();
#endif
        }

        Slot* claimSlot() {
            for (auto& s : slots_) {
                Job* expected = nullptr;
                // a placeholder until the job is filled in: workers skip it as they skip a closing slot
                if (s.job.load(std::memory_order_relaxed) == nullptr && s.job.compare_exchange_strong(expected, closing())) return &s;
            }
            return nullptr;
        }

        void wake() {
            generation_.fetch_add(1);
            if (sleeping_.load() > 0) {
                // under the lock, so a worker between its last look and its sleep cannot miss the wake-up
                std::lock_guard lock(sleep_);
            }
            wakeup_.notify_all();
        }

        static bool work(Job& job) {
            bool did = false;
            while (true) {
                const int i0 = job.next.fetch_add(job.grain, std::memory_order_relaxed);
                if (i0 >= job.end) return did;
                job.run(job.ctx, i0, std::min(i0 + job.grain, job.end));
                did = true;
            }
        }

        // one pass over the slots, joining every open loop that still has items; whether any item was run
        bool scan() {
            bool did = false;
            for (auto& s : slots_) {
                if (s.job.load(std::memory_order_relaxed) == nullptr) continue;
                // the count goes up before the job is read, and a caller marks its loop closing before it reads the
                // count, so a job is never touched after its caller has let it go
                s.refs.fetch_add(1);
                Job* job = s.job.load();
                if (job != nullptr && job != closing()) did |= work(*job);
                s.refs.fetch_sub(1);
            }
            return did;
        }

        void workerLoop(uint64_t seen) {
            while (true) {
                // wait for a new generation: spin a little, then sleep
                const auto t0 = std::chrono::steady_clock::now();
                int spins = 0;
                // (the stop is looked at too: a worker whose last scan began after the destructor's bump has that
                // generation as `seen` already)
                while (generation_.load() == seen && !stop_.load()) {
                    pause();
                    if ((++spins & 255) == 0 && std::chrono::steady_clock::now() - t0 > spin_) {
                        std::unique_lock lock(sleep_);
                        sleeping_.fetch_add(1);
                        wakeup_.wait(lock, [&] { return generation_.load() != seen || stop_.load(); });
                        sleeping_.fetch_sub(1);
                    }
                }
                if (stop_.load()) return;
                // take the generation before looking, so a loop opened during the scan is not slept through
                do {
                    seen = generation_.load();
                } while (scan());
            }
        }
    };

}// namespace threepp

#endif//THREEPP_THREADPOOL_HPP
