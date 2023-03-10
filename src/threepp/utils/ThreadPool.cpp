
#include "threepp/utils/ThreadPool.hpp"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

using namespace threepp::utils;

struct ThreadPool::Impl {

    explicit Impl(unsigned int threadCount)
        : done_(false) {

        threadCount = std::min(std::thread::hardware_concurrency(), std::max(1u, threadCount));
        try {
            for (unsigned i = 0; i < threadCount; ++i) {
                threads_.emplace_back(&Impl::worker_thread, this);
            }
        } catch (...) {
            done_ = true;
            throw;
        }
    }

    void submit(std::function<void()> f) {

        std::unique_lock<std::mutex> lck(m_);
        workQueue_.emplace(std::move(f));
        lck.unlock();
        cvWorker_.notify_one();
    }

    ~Impl() noexcept {

        std::unique_lock<std::mutex> lck(m_);
        done_ = true;
        lck.unlock();
        cvWorker_.notify_all();

        for (auto& thread : threads_) {
            thread.join();
        }
    }

private:
    bool done_;
    std::mutex m_;

    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> workQueue_;

    std::condition_variable cvWorker_;

    void worker_thread() {

        while (true) {

            std::unique_lock<std::mutex> lck(m_);

            // If no work is available, block the thread here
            cvWorker_.wait(lck, [this]() { return done_ || !workQueue_.empty(); });
            if (!workQueue_.empty()) {

                auto task = std::move(workQueue_.front());
                workQueue_.pop();

                lck.unlock();

                // Run work function outside mutex lock context
                task();

            } else if (done_) {

                break;
            }
        }
    }
};

ThreadPool::ThreadPool(unsigned int threadCount)
    : pimpl_(std::make_unique<Impl>(threadCount)) {}

void ThreadPool::submit(const std::function<void()>& f) {

    pimpl_->submit(f);
}

ThreadPool::~ThreadPool() = default;
