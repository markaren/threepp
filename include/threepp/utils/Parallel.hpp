// Portable parallel std::for_each.
//
// Runs on threepp::ThreadPool::shared() (utils/ThreadPool.hpp), a pool of persistent threads that stays awake for a
// moment between loops, on every platform that has threads; loops may run at once from several threads and inside
// each other. Where there are no threads (the web build without pthreads) the call runs in order.
//
// Contract: each invocation of fn must write disjoint state, so the serial path is behaviourally identical to the
// parallel one: only the threading differs, never the result. The iterators must be random access.

#ifndef THREEPP_UTILS_PARALLEL_HPP
#define THREEPP_UTILS_PARALLEL_HPP

#include "threepp/utils/ThreadPool.hpp"

#include <iterator>
#include <utility>

namespace threepp {

    template<class It, class Fn>
    void parallelForEach(It first, It last, Fn fn) {
        const auto n = static_cast<int>(std::distance(first, last));
        ThreadPool::shared().parallelFor(0, n, [&](int i) { fn(first[i]); }, 1);
    }

}// namespace threepp

#endif// THREEPP_UTILS_PARALLEL_HPP
