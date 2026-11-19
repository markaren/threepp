// A small persistent thread pool for fork-join loops: parallelFor(begin, end, fn) calls fn(i) for every i in
// [begin, end) on the pool's threads and the calling thread, and returns when every call has returned.
//
// It exists for loops that fork many times a frame (a simulation's passes, each a few hundred microseconds): its
// workers stay awake for a moment after a loop (spinning, then sleeping on a condition variable), so the next fork
// costs a few microseconds where the parallel STL or std::async cost tens. The items are handed out in chunks from an
// atomic counter, so uneven items balance themselves.
//
// Contract, as threepp::parallelForEach's: each fn(i) writes only its own state, so the result never depends on the
// number of threads or on which thread ran which item. An exception from fn on the calling thread is passed on once
// the workers have let go of the loop; one on a worker terminates, so fn should not throw.
//
// - A parallelFor called from inside a running one (from fn), or from a second thread while another thread's loop is
//   running, runs its items in order on the calling thread: no deadlock, no waiting.
// - With no hardware threads to spare, or on a platform without threads (the web build without pthreads), every loop
//   runs in order on the calling thread.
// - ThreadPool::shared() is one pool per module for everyone to use; it is never destroyed (at exit the operating
//   system ends its threads; joining them from a static destructor can hang a DLL's unload).

#ifndef THREEPP_THREADPOOL_HPP
#define THREEPP_THREADPOOL_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
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
        // `workers` threads besides the caller's (default: one less than the hardware threads, at most 15: past
        // sixteen threads a frame's loops gain little and every extra thread spins after each one)
        explicit ThreadPool(int workers = -1) {
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
            generation_.fetch_add(1);
            {
                std::lock_guard lock(sleep_);
            }
            wake_.notify_all();
            for (auto& t : threads_) t.join();
        }

        ThreadPool(const ThreadPool&) = delete;
        ThreadPool& operator=(const ThreadPool&) = delete;

        // the threads a loop runs on: the workers and the caller
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
            if (threads_.empty() || n <= grain || insideLoop() || !submit_.try_lock()) {
                for (int i = begin; i < end; ++i) fn(i);
                return;
            }
            std::lock_guard lock(submit_, std::adopt_lock);

            auto call = [&fn](int i0, int i1) {
                for (int i = i0; i < i1; ++i) fn(i);
            };
            Job job;
            job.ctx = &call;
            job.run = [](void* ctx, int i0, int i1) { (*static_cast<decltype(call)*>(ctx))(i0, i1); };
            job.next.store(begin, std::memory_order_relaxed);
            job.end = end;
            job.grain = grain;

            // closes the job however this scope is left (an exception from fn included): no worker picks it up after
            // this, and the ones that did have finished their chunks before the job (on this stack) goes
            struct Close {
                ThreadPool& pool;
                ~Close() {
                    pool.job_.store(nullptr);
                    while (pool.inflight_.load() != 0) pause();
                    insideLoop() = false;
                }
            } close{*this};

            job_.store(&job);
            generation_.fetch_add(1);
            if (sleeping_.load() > 0) {
                // under the lock, so a worker between its last look and its sleep cannot miss the wake-up
                std::lock_guard sleepLock(sleep_);
            }
            wake_.notify_all();

            insideLoop() = true;
            work(job);
        }

        // one pool per module, never destroyed
        static ThreadPool& shared() {
            static ThreadPool* pool = new ThreadPool();
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

        std::vector<std::thread> threads_;
        std::mutex submit_;
        std::mutex sleep_;
        std::condition_variable wake_;
        std::atomic<Job*> job_{nullptr};
        std::atomic<uint64_t> generation_{0};
        std::atomic<int> inflight_{0};
        std::atomic<int> sleeping_{0};
        std::atomic<bool> stop_{false};

        // how long a worker spins for the next loop before it sleeps
        static constexpr auto spinFor = std::chrono::microseconds(200);

        static bool& insideLoop() {
            thread_local bool inside = false;
            return inside;
        }

        static void pause() {
#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386__)
            _mm_pause();
#else
            std::this_thread::yield();
#endif
        }

        static void work(Job& job) {
            while (true) {
                const int i0 = job.next.fetch_add(job.grain, std::memory_order_relaxed);
                if (i0 >= job.end) return;
                job.run(job.ctx, i0, std::min(i0 + job.grain, job.end));
            }
        }

        void workerLoop(uint64_t seen) {
            insideLoop() = true;
            while (true) {
                // wait for a new generation: spin a little, then sleep
                const auto t0 = std::chrono::steady_clock::now();
                int spins = 0;
                while (generation_.load() == seen) {
                    pause();
                    if ((++spins & 255) == 0 && std::chrono::steady_clock::now() - t0 > spinFor) {
                        std::unique_lock lock(sleep_);
                        sleeping_.fetch_add(1);
                        wake_.wait(lock, [&] { return generation_.load() != seen; });
                        sleeping_.fetch_sub(1);
                    }
                }
                seen = generation_.load();
                if (stop_.load()) return;
                // join the loop if it is still open: the count goes up before the job is read, and the caller closes
                // the job before it reads the count, so a job is never touched after its caller has let it go
                inflight_.fetch_add(1);
                if (Job* job = job_.load()) work(*job);
                inflight_.fetch_sub(1);
            }
        }
    };

}// namespace threepp

#endif//THREEPP_THREADPOOL_HPP
