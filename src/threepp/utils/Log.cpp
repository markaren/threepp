
#include "threepp/utils/Log.hpp"
#include "threepp/utils/StringUtils.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

namespace threepp::log {

    namespace {

        struct Rule {
            std::string prefix;// "" covers every category
            Level level;
        };

        // "vulkan" covers "vulkan" and "vulkan.x", not "vulkanx".
        bool covers(std::string_view prefix, std::string_view name) {
            if (prefix.empty()) return true;
            if (name.size() < prefix.size() || name.compare(0, prefix.size(), prefix) != 0) return false;
            return name.size() == prefix.size() || name[prefix.size()] == '.';
        }

        std::string_view trim(std::string_view s) {
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
            while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
            return s;
        }

        std::optional<Level> parseLevel(std::string_view s) {
            const auto l = utils::toLower(std::string(trim(s)));
            if (l == "trace") return Level::Trace;
            if (l == "debug") return Level::Debug;
            if (l == "info") return Level::Info;
            if (l == "warn" || l == "warning") return Level::Warn;
            if (l == "error") return Level::Error;
            if (l == "off" || l == "none") return Level::Off;
            return std::nullopt;
        }

        std::string normalisePrefix(std::string_view p) {
            p = trim(p);
            if (p == "*") return {};
            // "vulkan.*" means the same as "vulkan".
            if (p.size() >= 2 && p.substr(p.size() - 2) == ".*") p.remove_suffix(2);
            return utils::toLower(std::string(p));
        }

        // Appends rule {prefix, l}, dropping the earlier rules it shadows.
        void addRule(std::vector<Rule>& rules, std::string prefix, Level l) {
            rules.erase(std::remove_if(rules.begin(), rules.end(),
                                       [&](const Rule& r) { return covers(prefix, r.prefix); }),
                        rules.end());
            rules.push_back({std::move(prefix), l});
        }

        // Parses a spec into rules; false if an entry was not understood.
        bool parseSpec(std::string_view spec, std::vector<Rule>& rules) {
            bool ok = true;
            while (!spec.empty()) {
                const auto comma = spec.find(',');
                auto entry = trim(spec.substr(0, comma));
                spec = comma == std::string_view::npos ? std::string_view{} : spec.substr(comma + 1);
                if (entry.empty()) continue;
                const auto colon = entry.rfind(':');
                std::optional<Level> level;
                std::string prefix;
                if (colon == std::string_view::npos) {
                    level = parseLevel(entry);// a bare level applies to all
                } else {
                    level = parseLevel(entry.substr(colon + 1));
                    prefix = normalisePrefix(entry.substr(0, colon));
                }
                if (!level) {
                    ok = false;
                    continue;
                }
                addRule(rules, std::move(prefix), *level);
            }
            return ok;
        }

        Level levelFor(const std::vector<Rule>& rules, std::string_view name, Level fallback) {
            Level l = fallback;
            for (const auto& r : rules) {
                if (covers(r.prefix, name)) l = r.level;// the later rule wins
            }
            return l;
        }

    }// namespace

    struct Registry {
        std::mutex mutex;
        std::vector<const Category*> categories;
        std::vector<Rule> apiRules;
        std::vector<Rule> envRules;
        bool envBad = false;
        std::string envSpec;

        std::function<void(const Record&)> sink;
        std::unique_ptr<std::FILE, int (*)(std::FILE*)> file{nullptr, &std::fclose};
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

        Registry() {
            if (const char* s = std::getenv("THREEPP_LOG")) {
                envSpec = s;
                envBad = !parseSpec(envSpec, envRules);
            }
            if (const char* path = std::getenv("THREEPP_LOG_FILE"); path && *path) {
#if defined(_MSC_VER)
                std::FILE* f = nullptr;
                if (fopen_s(&f, path, "w") != 0) f = nullptr;
#else
                std::FILE* f = std::fopen(path, "w");
#endif
                file.reset(f);
            }
        }

        int thresholdFor(std::string_view name) const {
            Level l = levelFor(apiRules, name, Level::Info);
            l = levelFor(envRules, name, l);
            return static_cast<int>(l);
        }

        // Caller holds the mutex.
        void refreshAll() {
            for (const auto* c : categories) {
                c->threshold_.store(thresholdFor(c->name_), std::memory_order_relaxed);
            }
        }

        int resolve(const Category& c) {
            std::lock_guard lock(mutex);
            int t = c.threshold_.load(std::memory_order_relaxed);
            if (t != Category::kUnresolved) return t;
            categories.push_back(&c);
            t = thresholdFor(c.name_);
            c.threshold_.store(t, std::memory_order_relaxed);
            return t;
        }
    };

    namespace {

        std::atomic<uint64_t> gFrame{0};

        Registry& registry() {
            // Built on first use, never during static initialisation of a
            // Category, and never destroyed: destructors and atexit handlers
            // that run late in process exit (a renderer torn down by a static,
            // the strict-validation verdict) can still log.
            static Registry* r = new Registry;
            return *r;
        }

        // Caller holds the registry mutex.
        void emit(Registry& r, const Record& rec) {
            const auto levelWord = rec.level == Level::Warn    ? std::string_view("warning: ")
                                   : rec.level == Level::Error ? std::string_view("error: ")
                                                               : std::string_view{};
            if (r.sink) {
                r.sink(rec);
            } else {
                std::string line;
                line.reserve(rec.category.size() + levelWord.size() + rec.message.size() + 4);
                line.append("[").append(rec.category).append("] ").append(levelWord).append(rec.message).append("\n");
                std::fwrite(line.data(), 1, line.size(), stderr);
                std::fflush(stderr);
            }
            if (r.file) {
                const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - r.start).count();
                char head[64];
                std::snprintf(head, sizeof(head), "f%llu %.3fs ", static_cast<unsigned long long>(rec.frame), secs);
                std::string line(head);
                line.append("[").append(rec.category).append("] ").append(levelWord).append(rec.message).append("\n");
                std::fwrite(line.data(), 1, line.size(), r.file.get());
                std::fflush(r.file.get());
            }
        }

        constinit Category kLogCategory{"core.log"};

    }// namespace

    int Category::resolve() const {
        auto& r = registry();
        const int t = r.resolve(*this);
        // Say once that THREEPP_LOG was not understood, through the facade itself.
        static std::atomic<bool> reported{false};
        if (r.envBad && this != &kLogCategory && !reported.exchange(true)) {
            write(kLogCategory, Level::Warn,
                  "THREEPP_LOG=\"" + r.envSpec + "\" has an entry that is not <category>:<level>; that entry is ignored "
                  "(levels: trace, debug, info, warn, error, off)");
        }
        return t;
    }

    void setSink(std::function<void(const Record&)> sink) {
        auto& r = registry();
        std::lock_guard lock(r.mutex);
        r.sink = std::move(sink);
    }

    void setLevel(std::string_view categoryPrefix, Level l) {
        auto& r = registry();
        std::lock_guard lock(r.mutex);
        addRule(r.apiRules, normalisePrefix(categoryPrefix), l);
        r.refreshAll();
    }

    bool setLevels(std::string_view spec) {
        auto& r = registry();
        std::lock_guard lock(r.mutex);
        const bool ok = parseSpec(spec, r.apiRules);
        r.refreshAll();
        return ok;
    }

    void setFrame(uint64_t f) {
        gFrame.store(f, std::memory_order_relaxed);
    }

    uint64_t frame() {
        return gFrame.load(std::memory_order_relaxed);
    }

    std::string_view levelName(Level l) {
        switch (l) {
            case Level::Trace: return "trace";
            case Level::Debug: return "debug";
            case Level::Info: return "info";
            case Level::Warn: return "warning";
            case Level::Error: return "error";
            default: return "off";
        }
    }

    void write(const Category& cat, Level level, std::string_view message) {
        if (level == Level::Off) return;
        // The sink adds the newline; a message that brings its own loses it.
        while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) message.remove_suffix(1);
        auto& r = registry();
        const Record rec{level, cat.name(), message, gFrame.load(std::memory_order_relaxed)};
        std::lock_guard lock(r.mutex);
        emit(r, rec);
    }

    void writef(const Category& cat, Level level, const char* fmt, ...) {
        char stackBuf[512];
        va_list args;
        va_start(args, fmt);
        va_list copy;
        va_copy(copy, args);
        const int n = std::vsnprintf(stackBuf, sizeof(stackBuf), fmt, args);
        va_end(args);
        if (n < 0) {
            va_end(copy);
            return;
        }
        if (static_cast<size_t>(n) < sizeof(stackBuf)) {
            va_end(copy);
            write(cat, level, std::string_view(stackBuf, static_cast<size_t>(n)));
            return;
        }
        std::string big(static_cast<size_t>(n) + 1, '\0');
        std::vsnprintf(big.data(), big.size(), fmt, copy);
        va_end(copy);
        big.resize(static_cast<size_t>(n));
        write(cat, level, big);
    }

}// namespace threepp::log
