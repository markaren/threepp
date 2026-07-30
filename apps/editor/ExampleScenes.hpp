// The scenes the editor ships with, compiled into the binary.
//
// File ▸ Open Example lists these. Each one is a complete three.js "Object"
// JSON document — the same format Save writes — embedded as source rather than
// found on disk, for the reason ViewportMarkers.cpp embeds its SVG: the editor
// does not depend on locating asset files at runtime. Opening one yields an
// UNTITLED document, so the first Save asks where to put it and nothing can
// overwrite a shipped example in place.
//
// The documents themselves are produced by apps/editor/tools/HoverArenaAuthor.cpp
// and written into ExampleScenes.cpp by it; that file is generated and says so.
// This header is separate so the headless test can load the same string the
// menu does, without linking the app.

#ifndef THREEPP_EDITOR_EXAMPLESCENES_HPP
#define THREEPP_EDITOR_EXAMPLESCENES_HPP

#include <string>
#include <string_view>
#include <vector>

namespace threepp::editor::examples {

    struct Example {
        // Stable identity: the menu label may be reworded, this may not. Also
        // what `threepp_editor --example=<slug>` takes.
        std::string_view slug;
        std::string_view label;
        // One line for the console, saying what the scene is for.
        std::string_view summary;
    };

    // Every shipped example, in menu order.
    [[nodiscard]] const std::vector<Example>& all();

    // nullptr when nothing ships under that slug.
    [[nodiscard]] const Example* find(std::string_view slug);

    // The document, assembled from the chunks compiled into the binary. Empty
    // for an unknown slug — which is the same answer as "no document", and the
    // caller has to check either way.
    [[nodiscard]] std::string json(std::string_view slug);

}// namespace threepp::editor::examples

#endif//THREEPP_EDITOR_EXAMPLESCENES_HPP
