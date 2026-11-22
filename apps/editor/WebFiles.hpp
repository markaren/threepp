// The editor's files on the web. A page has no disk of its own: the editor
// reads and writes Emscripten's in-memory file system, and these move files
// between it and the visitor's computer. Upload is the browser's own file
// picker, writing into a directory of that file system; download hands a file
// written there to the browser as a save. Web builds only.

#ifndef THREEPP_EDITOR_WEBFILES_HPP
#define THREEPP_EDITOR_WEBFILES_HPP

#ifdef __EMSCRIPTEN__

#include <filesystem>
#include <vector>

namespace threepp::editor::web {

    // Opens the browser's file picker. The files chosen arrive in `directory`
    // a little later (reading them is asynchronous); takeUploaded() hands them
    // over once they are there.
    void requestUpload(const std::filesystem::path& directory);

    // The files written by uploads since the last call, in the order chosen.
    [[nodiscard]] std::vector<std::filesystem::path> takeUploaded();

    // Offers `file` to the visitor as a download under its own name.
    void offerDownload(const std::filesystem::path& file);

}// namespace threepp::editor::web

#endif

#endif//THREEPP_EDITOR_WEBFILES_HPP
