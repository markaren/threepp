
#ifndef THREEPP_THREADPOOL_HPP
#define THREEPP_THREADPOOL_HPP

#include <functional>
#include <memory>

namespace threepp::utils {

    class ThreadPool {

    public:
        explicit ThreadPool(unsigned int threadCount = 1);

        ThreadPool(const ThreadPool&) = delete;
        ThreadPool(const ThreadPool&&) = delete;
        ThreadPool operator=(const ThreadPool&) = delete;

        void submit(const std::function<void()>& f);

        void wait();

        ~ThreadPool();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };


}// namespace threepp::utils

#endif//THREEPP_THREADPOOL_HPP
