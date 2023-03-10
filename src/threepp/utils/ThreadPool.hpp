
#ifndef THREEPP_THREADPOOL_HPP
#define THREEPP_THREADPOOL_HPP

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

namespace threepp::utils {

    class ThreadPool {

    public:
        explicit ThreadPool(unsigned int threadCount)
            : done_(false), pendingTasks_(0) {

            threadCount = std::min(std::thread::hardware_concurrency(), std::max(1u, threadCount));
            try {
                for (unsigned i = 0; i < threadCount; ++i) {
                    threads_.emplace_back(&ThreadPool::worker_thread, this);
                }
            } catch (...) {
                done_ = true;
                throw;
            }
        }

        ThreadPool(const ThreadPool&) = delete;
        ThreadPool(const ThreadPool&&) = delete;

        void submit(std::function<void()> f) {

            std::unique_lock<std::mutex> lck(m_);
            workQueue_.emplace(std::move(f));
            lck.unlock();
            cvWorker_.notify_one();
        }

        ~ThreadPool() noexcept {

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

        unsigned int pendingTasks_;

        void worker_thread() {

            while (true) {

                std::unique_lock<std::mutex> lck(m_);

                // If no work is available, block the thread here
                cvWorker_.wait(lck, [this]() { return done_ || !workQueue_.empty(); });
                if (!workQueue_.empty()) {

                    ++pendingTasks_;

                    auto task = std::move(workQueue_.front());
                    workQueue_.pop();

                    lck.unlock();

                    // Run work function outside mutex lock context
                    task();

                    lck.lock();
                    --pendingTasks_;
                    lck.unlock();

                } else if (done_) {

                    break;
                }
            }
        }
    };


}// namespace threepp::utils

#endif//THREEPP_THREADPOOL_HPP
