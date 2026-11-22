
#ifdef __EMSCRIPTEN__

#include "WebFiles.hpp"

#include <emscripten.h>

#include <cstdlib>
#include <sstream>
#include <string>

EM_JS_DEPS(threepp_editor_webfiles, "$UTF8ToString,$stringToNewUTF8");

// JS strings in double quotes only: EM_JS bodies go through the C preprocessor,
// which mangles single-quoted strings and regex literals.
//
// The picker is a hidden <input type=file> made on demand. Each chosen file is
// read whole and written into the directory, then its path is queued for the
// next takeUploaded(); a failed read is reported in the console and skipped.
EM_JS(void, threepp_editor_request_upload, (const char* directory), {
    const dir = UTF8ToString(directory);
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.style.display = "none";
    input.addEventListener("change", async () => {
        Module.threeppEditorUploads = Module.threeppEditorUploads || [];
        for (const file of Array.from(input.files)) {
            try {
                const data = new Uint8Array(await file.arrayBuffer());
                const path = (dir.endsWith("/") ? dir : dir + "/") + file.name;
                FS.writeFile(path, data);
                Module.threeppEditorUploads.push(path);
            } catch (e) {
                console.error("upload of " + file.name + " failed: " + e);
            }
        }
        input.remove();
    });
    document.body.appendChild(input);
    input.click();
});

// Newline-joined paths, or null when nothing has arrived. Caller frees.
EM_JS(char*, threepp_editor_take_uploads, (), {
    const queue = Module.threeppEditorUploads;
    if (!queue || queue.length === 0) return 0;
    Module.threeppEditorUploads = [];
    return stringToNewUTF8(queue.join(String.fromCharCode(10)));
});

EM_JS(void, threepp_editor_offer_download, (const char* path), {
    const p = UTF8ToString(path);
    let data;
    try {
        data = FS.readFile(p);
    } catch (e) {
        console.error("download of " + p + " failed: " + e);
        return;
    }
    const blob = new Blob([data], {type: "application/octet-stream"});
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = p.substring(p.lastIndexOf("/") + 1);
    a.style.display = "none";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
});

namespace threepp::editor::web {

    void requestUpload(const std::filesystem::path& directory) {

        threepp_editor_request_upload(directory.generic_string().c_str());
    }

    std::vector<std::filesystem::path> takeUploaded() {

        std::vector<std::filesystem::path> paths;
        char* joined = threepp_editor_take_uploads();
        if (!joined) return paths;

        std::istringstream lines(joined);
        std::free(joined);
        std::string line;
        while (std::getline(lines, line)) {
            if (!line.empty()) paths.emplace_back(line);
        }
        return paths;
    }

    void offerDownload(const std::filesystem::path& file) {

        threepp_editor_offer_download(file.generic_string().c_str());
    }

}// namespace threepp::editor::web

#endif
