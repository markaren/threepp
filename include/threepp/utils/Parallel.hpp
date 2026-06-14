// Portable parallel std::for_each.
//
// Uses the C++17 Parallel STL (std::execution::par) on MSVC, which ships native
// support with no extra dependency. Everywhere else the call runs serially:
// Apple's libc++ does not implement the parallel execution policies at all, and
// libstdc++'s policies require Intel TBB to be linked. Gating on _MSC_VER (rather
// than the __cpp_lib_parallel_algorithm feature-test macro) is deliberate —
// libstdc++ advertises that macro even when TBB is absent, which would select the
// parallel path and then fail to link.
//
// Contract: each invocation of fn must write disjoint state, so the serial
// fallback is behaviourally identical to the parallel path — only the threading
// differs, never the result.

#ifndef THREEPP_UTILS_PARALLEL_HPP
#define THREEPP_UTILS_PARALLEL_HPP

#include <algorithm>
#include <utility>
#ifdef _MSC_VER
#include <execution>
#endif

namespace threepp {

    template<class It, class Fn>
    void parallelForEach(It first, It last, Fn fn) {
#ifdef _MSC_VER
        std::for_each(std::execution::par, first, last, std::move(fn));
#else
        std::for_each(first, last, std::move(fn));
#endif
    }

}// namespace threepp

#endif// THREEPP_UTILS_PARALLEL_HPP
