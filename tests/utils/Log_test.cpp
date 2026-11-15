
#include <catch2/catch_test_macros.hpp>

#include "threepp/utils/Log.hpp"

#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#define THREEPP_DUP _dup
#define THREEPP_DUP2 _dup2
#define THREEPP_CLOSE _close
#define THREEPP_FILENO _fileno
#else
#include <unistd.h>
#define THREEPP_DUP dup
#define THREEPP_DUP2 dup2
#define THREEPP_CLOSE close
#define THREEPP_FILENO fileno
#endif

using namespace threepp;

namespace {

    constinit log::Category kVk{"vulkan"};
    constinit log::Category kVkOverlay{"vulkan.overlay"};
    constinit log::Category kVkOverlayHud{"vulkan.overlay.hud"};
    constinit log::Category kVulkanx{"vulkanx"};
    constinit log::Category kCore{"core.geometry"};

    struct Captured {
        log::Level level;
        std::string category;
        std::string message;
        uint64_t frame;
    };

    // Captures records for the life of the object, and puts every threshold
    // back to the default afterwards.
    struct Capture {
        std::mutex m;
        std::vector<Captured> records;

        Capture() {
            log::setLevel("*", log::Level::Info);
            log::setSink([this](const log::Record& r) {
                std::lock_guard lock(m);
                records.push_back({r.level, std::string(r.category), std::string(r.message), r.frame});
            });
        }
        ~Capture() {
            log::setSink({});
            log::setLevel("*", log::Level::Info);
            log::setFrame(0);
        }
    };

    // Points fd 2 at a file for its lifetime; finish() restores stderr and
    // returns what was written, with "\r\n" folded to "\n".
    class StderrToFile {
    public:
        explicit StderrToFile(std::string path): path_(std::move(path)) {
            std::fflush(stderr);
            saved_ = THREEPP_DUP(2);
            FILE* f = std::fopen(path_.c_str(), "wb");
            REQUIRE(f != nullptr);
            THREEPP_DUP2(THREEPP_FILENO(f), 2);
            std::fclose(f);
        }
        std::string finish() {
            if (saved_ < 0) return text_;
            std::fflush(stderr);
            THREEPP_DUP2(saved_, 2);
            THREEPP_CLOSE(saved_);
            saved_ = -1;
            if (FILE* f = std::fopen(path_.c_str(), "rb")) {
                char buf[4096];
                size_t n;
                while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) text_.append(buf, n);
                std::fclose(f);
            }
            std::remove(path_.c_str());
            std::string clean;
            for (char c : text_)
                if (c != '\r') clean += c;
            text_ = clean;
            return text_;
        }
        ~StderrToFile() { finish(); }

    private:
        std::string path_;
        std::string text_;
        int saved_ = -1;
    };

    int sideEffects = 0;
    int bump() {
        ++sideEffects;
        return sideEffects;
    }

}// namespace

TEST_CASE("Log: Info is the default threshold") {
    Capture cap;
    CHECK_FALSE(kVk.enabled(log::Level::Debug));
    CHECK(kVk.enabled(log::Level::Info));
    CHECK(kVk.enabled(log::Level::Error));
    CHECK_FALSE(kVk.enabled(log::Level::Off));
}

TEST_CASE("Log: a prefix covers its own name and the names below it") {
    Capture cap;
    log::setLevel("vulkan", log::Level::Debug);
    CHECK(kVk.enabled(log::Level::Debug));
    CHECK(kVkOverlay.enabled(log::Level::Debug));
    CHECK(kVkOverlayHud.enabled(log::Level::Debug));
    CHECK_FALSE(kVulkanx.enabled(log::Level::Debug));
    CHECK_FALSE(kCore.enabled(log::Level::Debug));

    // The later call wins for what it covers.
    log::setLevel("vulkan.overlay", log::Level::Error);
    CHECK(kVk.enabled(log::Level::Debug));
    CHECK_FALSE(kVkOverlay.enabled(log::Level::Warn));
    CHECK_FALSE(kVkOverlayHud.enabled(log::Level::Warn));
    CHECK(kVkOverlayHud.enabled(log::Level::Error));

    log::setLevel("*", log::Level::Off);
    CHECK_FALSE(kVk.enabled(log::Level::Error));
    CHECK_FALSE(kCore.enabled(log::Level::Error));
}

TEST_CASE("Log: THREEPP_LOG style specs") {
    Capture cap;
    CHECK(log::setLevels("vulkan.overlay:debug, core:warn"));
    CHECK(kVkOverlay.enabled(log::Level::Debug));
    CHECK_FALSE(kVk.enabled(log::Level::Debug));
    CHECK_FALSE(kCore.enabled(log::Level::Info));
    CHECK(kCore.enabled(log::Level::Warn));

    CHECK(log::setLevels("*:trace"));
    CHECK(kCore.enabled(log::Level::Trace));
    CHECK(kVk.enabled(log::Level::Trace));

    CHECK(log::setLevels("error"));// a bare level means "*"
    CHECK_FALSE(kVkOverlay.enabled(log::Level::Warn));
    CHECK(kVkOverlay.enabled(log::Level::Error));

    CHECK(log::setLevels("VULKAN.*:Warning"));
    CHECK(kVkOverlay.enabled(log::Level::Warn));
    CHECK_FALSE(kVkOverlay.enabled(log::Level::Info));

    // A bad entry is reported, the good ones still apply.
    CHECK_FALSE(log::setLevels("core:loud,vulkan:off"));
    CHECK_FALSE(kVk.enabled(log::Level::Error));
}

TEST_CASE("Log: a custom sink sees category, level, message and frame") {
    Capture cap;
    log::setFrame(42);
    THREEPP_LOG(kVkOverlay, Warn) << "HUD sprite count " << 300 << " exceeds " << 256;
    THREEPP_LOGF(kCore, Error, "fit %u entries", 7u);
    THREEPP_LOG(kCore, Debug) << "not shown";

    REQUIRE(cap.records.size() == 2);
    CHECK(cap.records[0].level == log::Level::Warn);
    CHECK(cap.records[0].category == "vulkan.overlay");
    CHECK(cap.records[0].message == "HUD sprite count 300 exceeds 256");
    CHECK(cap.records[0].frame == 42);
    CHECK(cap.records[1].level == log::Level::Error);
    CHECK(cap.records[1].category == "core.geometry");
    CHECK(cap.records[1].message == "fit 7 entries");
}

TEST_CASE("Log: once fires the first time its line runs") {
    Capture cap;
    for (int i = 0; i < 5; ++i) {
        THREEPP_LOG_ONCE(kVk, Warn) << "first " << i;
    }
    REQUIRE(cap.records.size() == 1);
    CHECK(cap.records[0].message == "first 0");
}

TEST_CASE("Log: a disabled message does not evaluate its operands") {
    Capture cap;
    sideEffects = 0;
    THREEPP_LOG(kVk, Debug) << bump();
    THREEPP_LOGF(kVk, Trace, "%d", bump());
    THREEPP_LOG_ONCE(kVk, Debug) << bump();
    CHECK(sideEffects == 0);
    CHECK(cap.records.empty());

    THREEPP_LOG(kVk, Info) << bump();
    CHECK(sideEffects == 1);
}

TEST_CASE("Log: a Line is one record built from several statements") {
    Capture cap;
    {
        log::Line line(kVk, log::Level::Info);
        line << "items:";
        for (int i = 0; i < 3; ++i) line << ' ' << i;
    }
    {
        log::Line off(kVk, log::Level::Debug);
        CHECK_FALSE(off.enabled());
        off << "dropped";
    }
    {
        log::Line rows(kVk, log::Level::Info);
        rows << "row 1";
        rows.flush();
        rows << "row 2";
    }
    REQUIRE(cap.records.size() == 3);
    CHECK(cap.records[0].message == "items: 0 1 2");
    CHECK(cap.records[1].message == "row 1");
    CHECK(cap.records[2].message == "row 2");
}

TEST_CASE("Log: the default sink's line format") {
    log::setLevel("*", log::Level::Info);
    log::setSink({});
    std::string text;
    {
        StderrToFile redirect("Log_test_stderr_format.txt");
        THREEPP_LOG(kVkOverlay, Warn) << "HUD sprite count 300 exceeds 256; the rest are not drawn";
        THREEPP_LOG(kCore, Error) << "bad index\n";// a trailing newline is the sink's job
        THREEPP_LOG(kCore, Info) << "plain";
        text = redirect.finish();
    }
    CHECK(text == "[vulkan.overlay] warning: HUD sprite count 300 exceeds 256; the rest are not drawn\n"
                  "[core.geometry] error: bad index\n"
                  "[core.geometry] plain\n");
}

TEST_CASE("Log: two threads never interleave within a line") {
    log::setLevel("*", log::Level::Info);
    log::setSink({});
    constexpr int kPerThread = 2000;
    auto worker = [](char c) {
        for (int i = 0; i < kPerThread; ++i) {
            THREEPP_LOG(kVk, Info) << std::string(40, c) << ' ' << i << ' ' << std::string(40, c);
        }
    };
    std::string text;
    {
        StderrToFile redirect("Log_test_stderr_threads.txt");
        std::thread a(worker, 'a');
        std::thread b(worker, 'b');
        a.join();
        b.join();
        text = redirect.finish();
    }
    int lines = 0;
    size_t pos = 0;
    while (pos < text.size()) {
        auto end = text.find('\n', pos);
        REQUIRE(end != std::string::npos);
        const std::string line = text.substr(pos, end - pos);
        pos = end + 1;
        ++lines;
        REQUIRE(line.rfind("[vulkan] ", 0) == 0);
        const std::string body = line.substr(9);
        const char c = body.front();
        CHECK((c == 'a' || c == 'b'));
        CHECK(body.find(c == 'a' ? 'b' : 'a') == std::string::npos);
        CHECK(body.size() >= 83);
    }
    CHECK(lines == 2 * kPerThread);
}
