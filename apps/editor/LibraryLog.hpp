// The library's log messages, for the Console tab.
//
// threepp::log hands every record to one sink, under its own lock and on
// whichever thread logged. LibraryLog installs a sink that keeps writing the
// line to stderr, as the default sink does, and also queues it; the editor
// drains the queue once per frame on the UI thread, and right after each of
// its own messages. It is a member declared before the canvas and the
// renderer, so the lines they print while they are constructed (the GPU, a
// fallback) are in the queue too, and the default sink is back before they are
// destroyed.
//
// The editor's own messages are records in kLogEditor ("editor"), and the play
// sessions' in the editor.* categories of EditorLog.hpp. The console shows
// those without the category, as the editor wrote them; the library's lines
// keep it.

#ifndef THREEPP_EDITOR_LIBRARYLOG_HPP
#define THREEPP_EDITOR_LIBRARYLOG_HPP

#include "threepp/utils/Log.hpp"

#include <cstdio>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace threepp::editor {

    // The editor app's own messages: the console lines of EditorApp::log and
    // the command line's errors.
    inline constinit log::Category kLogEditor{"editor"};

    class LibraryLog {
    public:
        LibraryLog() {
            log::setSink([this](const log::Record& r) {
                const char* word = r.level == log::Level::Warn    ? "warning: "
                                   : r.level == log::Level::Error ? "error: "
                                                                  : "";
                // stderr, in the default sink's format.
                std::string line;
                line.append("[").append(r.category).append("] ").append(word).append(r.message).append("\n");
                std::fwrite(line.data(), 1, line.size(), stderr);
                std::fflush(stderr);

                // The console colours a line by how it starts, so the level
                // word leads there: "warning: [vulkan.overlay] ...".
                std::string body = word;
                if (!isEditorCategory(r.category)) body.append("[").append(r.category).append("] ");
                body.append(r.message);

                std::lock_guard lock(mutex_);
                if (pending_.size() < kMaxPending) pending_.push_back(std::move(body));
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
        // "editor" and the names below it.
        static bool isEditorCategory(std::string_view category) {
            constexpr std::string_view editor = "editor";
            return category.substr(0, editor.size()) == editor &&
                   (category.size() == editor.size() || category[editor.size()] == '.');
        }

        // A burst between two frames is capped; the rest stays on stderr only.
        static constexpr std::size_t kMaxPending = 512;

        std::mutex mutex_;
        std::vector<std::string> pending_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_LIBRARYLOG_HPP
