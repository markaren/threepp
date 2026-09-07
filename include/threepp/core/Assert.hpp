#ifndef THREEPP_ASSERT_HPP
#define THREEPP_ASSERT_HPP

#include <cstdio>
#include <cstdlib>

// Debug-only invariant checks.
//
// threepp's hot paths are all unchecked `operator[]` on std::vector, so an
// off-by-one in an itemSize/count calculation silently corrupts neighbouring
// attribute data instead of failing where the mistake was made. THREEPP_ASSERT
// compiles to nothing in release builds, so it costs nothing to place these
// checks directly on BufferAttribute's element accessors.
//
// Enabled by default whenever NDEBUG is not defined. Override explicitly with
// -DTHREEPP_ENABLE_ASSERTS=1 to keep the checks in an optimised build (useful
// when chasing a loader that only misbehaves under -O2), or =0 to strip them.

#ifndef THREEPP_ENABLE_ASSERTS
#ifdef NDEBUG
#define THREEPP_ENABLE_ASSERTS 0
#else
#define THREEPP_ENABLE_ASSERTS 1
#endif
#endif

#if THREEPP_ENABLE_ASSERTS

namespace threepp::detail {

    [[noreturn]] inline void assertionFailed(const char* expr, const char* file, int line, const char* msg) {

        std::fprintf(stderr,
                     "[threepp] assertion failed: %s\n"
                     "  at %s:%d\n"
                     "  %s\n",
                     expr, file, line, msg ? msg : "(no message)");
        std::fflush(stderr);
        std::abort();
    }

}// namespace threepp::detail

#define THREEPP_ASSERT_MSG(expr, msg)                                                    \
    (static_cast<bool>(expr)                                                             \
             ? void(0)                                                                   \
             : ::threepp::detail::assertionFailed(#expr, __FILE__, __LINE__, (msg)))

#define THREEPP_ASSERT(expr) THREEPP_ASSERT_MSG(expr, nullptr)

#else

// The expression must not be evaluated when the checks are compiled out.
#define THREEPP_ASSERT_MSG(expr, msg) (void(0))
#define THREEPP_ASSERT(expr) (void(0))

#endif

#endif//THREEPP_ASSERT_HPP
