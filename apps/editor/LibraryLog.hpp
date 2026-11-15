// The library's log messages, for the Console tab.
//
// threepp::log hands every record to one sink, under its own lock and on
// whichever thread logged. LibraryLog installs a sink that keeps writing the
// line to stderr, as the default sink does, and also queues it; the editor
// drains the queue once per frame on the UI thread. It is a member declared
// before the canvas and the renderer, so the lines they print while they are
// constructed (the GPU, a fallback) are in the queue too, and the default sink
// is back before they are destroyed.

#ifndef THREEPP_EDITOR_LIBRARYLOG_HPP
#define THREEPP_EDITOR_LIBRARYLOG_HPP

#include "threepp/utils/Log.hpp"

#include <cstdio>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace threepp::editor {

    class LibraryLog {
    public:
        LibraryLog() {
            log::setSink([this](const log::Record& r) {
                const char* word = r.level == log::Level::Warn    ? "warning: "
                                   : r.level == log::Level::Error ? "error: "
                                                                  : "";
                std::string body;
                body.reserve(r.category.size() + r.message.size() + 4);
                body.append("[").append(r.category).append("] ").append(r.message);

                // stderr, in the default sink's format.
                std::string line;
                line.append("[").append(r.category).append("] ").append(word).append(r.message).append("\n");
                std::fwrite(line.data(), 1, line.size(), stderr);
                std::fflush(stderr);

                // The console colours a line by how it starts, so the level
                // word leads there: "warning: [vulkan.overlay] ...".
                std::lock_guard lock(mutex_);
                if (pending_.size() < kMaxPending) pending_.push_back(word + body);
            });
        }

        ~LibraryLog() { log::setSink({}); }

        LibraryLog(const LibraryLog&) = delete;
        LibraryLog& operator=(const LibraryLog&) = delete;

        // The lines queued since the last call, in the order they were logged.
        [[nodiscard]] std::vector<std::string> take() {
            std::lock_guard lock(mutex_);
            return std::exchange(pending_, {});
        }

    private:
        // A burst between two frames is capped; the rest stays on stderr only.
        static constexpr std::size_t kMaxPending = 512;

        std::mutex mutex_;
        std::vector<std::string> pending_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_LIBRARYLOG_HPP
