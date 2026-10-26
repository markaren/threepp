// Environment knobs of the Vulkan renderer: diagnostics, A/B hatches and
// startup overrides read from THREEPP_* variables. One reading for all of them,
// so a value means the same thing whichever knob it is given to:
//
//   flag     unset or empty: the default. Otherwise on, unless the value
//            starts with '0': "1", "yes" and "on" turn a knob on, "0" off.
//   integer  unset, empty or not a number: nullopt, and the caller's default.
//   number   the same, for a floating-point value.
//
// Nothing here caches. A knob read once per process holds the result in a
// function-local static at its site; the others are read again per call, and
// tests set variables between renderer instances.
//
// Knobs that take a string (a directory, a name list, an enum spelling) read
// std::getenv directly: there is nothing to parse.

#ifndef THREEPP_VULKAN_ENVKNOB_HPP
#define THREEPP_VULKAN_ENVKNOB_HPP

#include <cstdlib>
#include <optional>

namespace threepp::vulkan::env {

    inline bool flag(const char* name, bool dflt = false) {
        const char* e = std::getenv(name);
        if (!e || !*e) return dflt;
        return *e != '0';
    }

    // `base` as std::strtoll's: 0 also takes 0x-prefixed hex (bit masks).
    inline std::optional<long long> integer(const char* name, int base = 10) {
        const char* e = std::getenv(name);
        if (!e || !*e) return std::nullopt;
        char* end = nullptr;
        const long long v = std::strtoll(e, &end, base);
        if (end == e) return std::nullopt;
        return v;
    }

    inline std::optional<double> number(const char* name) {
        const char* e = std::getenv(name);
        if (!e || !*e) return std::nullopt;
        char* end = nullptr;
        const double v = std::strtod(e, &end);
        if (end == e) return std::nullopt;
        return v;
    }

}// namespace threepp::vulkan::env

#endif// THREEPP_VULKAN_ENVKNOB_HPP
