
#ifndef THREEPP_LOG_HPP
#define THREEPP_LOG_HPP

// threepp's console messages, one facade for all of them.
//
//   static constinit threepp::log::Category kLog{"core.geometry"};
//   THREEPP_LOG(kLog, Warn) << "index " << i << " is out of range";
//   THREEPP_LOGF(kLog, Debug, "fit %u entries", n);
//   THREEPP_LOG_ONCE(kLog, Warn) << "reported the first time this line runs";
//
// A statement is one record. The macros test the category's threshold first,
// so a disabled message formats nothing and evaluates none of its operands.
// The default sink writes one line per record to stderr:
//
//   [core.geometry] warning: index 7 is out of range
//
// Thresholds default to Info. THREEPP_LOG=vulkan.probe:debug,core:warn sets
// them at startup (a name covers itself and the names below it, "*" covers
// all, a bare level applies to "*"); THREEPP_LOG_FILE=<path> also writes every
// record, with its frame number and the seconds since the first record, to
// that file. THREEPP_LOG wins over setLevel(); among setLevel() calls the
// later one wins for the categories it covers.

#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#if defined(_MSC_VER)
#include <sal.h>
#define THREEPP_LOG_PRINTF_FMT _Printf_format_string_
#define THREEPP_LOG_PRINTF_ATTR(fmtIndex, argIndex)
#elif defined(__GNUC__) || defined(__clang__)
#define THREEPP_LOG_PRINTF_FMT
#define THREEPP_LOG_PRINTF_ATTR(fmtIndex, argIndex) __attribute__((format(printf, fmtIndex, argIndex)))
#else
#define THREEPP_LOG_PRINTF_FMT
#define THREEPP_LOG_PRINTF_ATTR(fmtIndex, argIndex)
#endif

namespace threepp::log {

    enum class Level { Trace, Debug, Info, Warn, Error, Off };

    // A named source of messages. Declare one per feature with static storage
    // (`static constinit` or `inline constinit`): the constructor is constexpr,
    // so there is no static initialisation order to get wrong. The threshold is
    // looked up the first time the category is asked, and kept current by
    // setLevel() after that.
    class Category {
    public:
        constexpr explicit Category(const char* name): name_(name) {}

        Category(const Category&) = delete;
        Category& operator=(const Category&) = delete;

        [[nodiscard]] const char* name() const { return name_; }

        // One relaxed atomic load once resolved.
        [[nodiscard]] bool enabled(Level l) const {
            int t = threshold_.load(std::memory_order_relaxed);
            if (t == kUnresolved) t = resolve();
            return l != Level::Off && static_cast<int>(l) >= t;
        }

    private:
        static constexpr int kUnresolved = -1;

        int resolve() const;
        friend struct Registry;

        const char* name_;
        mutable std::atomic<int> threshold_{kUnresolved};
    };

    struct Record {
        Level level;
        std::string_view category;
        std::string_view message;
        uint64_t frame;
    };

    // Replaces the default stderr sink; an empty function restores it. The
    // sink is called under the facade's lock, one record at a time, and must
    // not log itself.
    void setSink(std::function<void(const Record&)> sink);

    // Threshold for every category named `categoryPrefix` or below it
    // ("vulkan" covers "vulkan.overlay", not "vulkanx"; "*" covers all).
    void setLevel(std::string_view categoryPrefix, Level l);

    // Applies a THREEPP_LOG style spec ("vulkan.probe:debug,core:warn") as a
    // series of setLevel() calls. Returns false if any entry was not
    // understood (the others still apply).
    bool setLevels(std::string_view spec);

    // The frame number records carry; the renderer calls it at frame begin.
    void setFrame(uint64_t frame);
    [[nodiscard]] uint64_t frame();

    // Lower-case level word: "trace", "debug", "info", "warning", "error", "off".
    [[nodiscard]] std::string_view levelName(Level l);

    // Hands one finished record to the sinks. The macros call it.
    void write(const Category& cat, Level level, std::string_view message);

    // printf-style record; THREEPP_LOGF calls it after the threshold test.
    void writef(const Category& cat, Level level, THREEPP_LOG_PRINTF_FMT const char* fmt, ...)
            THREEPP_LOG_PRINTF_ATTR(3, 4);

    // One record built from several statements: stream into it, and it is
    // written when flush() is called or it goes out of scope. A Line for a
    // disabled level ignores what is streamed into it and writes nothing.
    class Line {
    public:
        Line(const Category& cat, Level level)
            : cat_(cat), level_(level), enabled_(cat.enabled(level)) {}

        Line(const Line&) = delete;
        Line& operator=(const Line&) = delete;

        ~Line() { flush(); }

        [[nodiscard]] bool enabled() const { return enabled_; }
        explicit operator bool() const { return enabled_; }

        // True while nothing was streamed since the last flush.
        [[nodiscard]] bool empty() const { return !pending_; }

        template<class T>
        Line& operator<<(const T& v) {
            if (enabled_) {
                if (!os_) os_.emplace();
                *os_ << v;
                pending_ = true;
            }
            return *this;
        }

        // Stream manipulators (std::setprecision is a value, these are functions).
        Line& operator<<(std::ostream& (*manip)(std::ostream&)) {
            if (enabled_) {
                if (!os_) os_.emplace();
                manip(*os_);
            }
            return *this;
        }
        Line& operator<<(std::ios_base& (*manip)(std::ios_base&)) {
            if (enabled_) {
                if (!os_) os_.emplace();
                manip(*os_);
            }
            return *this;
        }

        // Writes what was streamed as one record (nothing if nothing was) and
        // starts a new one with the same category, level and stream flags.
        void flush() {
            if (enabled_ && pending_) {
                write(cat_, level_, os_->str());
                os_->str({});
                pending_ = false;
            }
        }

    private:
        const Category& cat_;
        Level level_;
        bool enabled_;
        bool pending_ = false;
        std::optional<std::ostringstream> os_;
    };

}// namespace threepp::log

#define THREEPP_LOG(cat, lvl)                                    \
    if (!(cat).enabled(::threepp::log::Level::lvl)) {            \
    } else                                                       \
        ::threepp::log::Line((cat), ::threepp::log::Level::lvl)

#define THREEPP_LOGF(cat, lvl, ...)                              \
    do {                                                         \
        if ((cat).enabled(::threepp::log::Level::lvl))           \
            ::threepp::log::writef((cat), ::threepp::log::Level::lvl, __VA_ARGS__); \
    } while (false)

// The first time this line runs with its level enabled; never again.
#define THREEPP_LOG_ONCE(cat, lvl)                                                    \
    if (static std::atomic<bool> threeppLogOnce_{false};                              \
        !(cat).enabled(::threepp::log::Level::lvl) || threeppLogOnce_.exchange(true)) { \
    } else                                                                            \
        ::threepp::log::Line((cat), ::threepp::log::Level::lvl)

#endif//THREEPP_LOG_HPP
