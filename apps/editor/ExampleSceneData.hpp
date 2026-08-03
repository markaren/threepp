// The documents behind the shipped examples, one generated translation unit
// each.
//
// Internal to the examples library — ExampleScenes.hpp is what everything else
// includes. The split exists because each example is written out by its OWN
// author program (apps/editor/tools/*Author.cpp), and a generated file that
// also carried the REGISTRY would mean regenerating one example silently
// dropped the others from the menu. So the generated files carry nothing but
// their own bytes, and the registry that names them is hand-written beside
// them in ExampleScenes.cpp.

#ifndef THREEPP_EDITOR_EXAMPLESCENEDATA_HPP
#define THREEPP_EDITOR_EXAMPLESCENEDATA_HPP

#include <string>

namespace threepp::editor::examples::detail {

    // Each assembles its document from the chunks compiled into the binary
    // (MSVC refuses a single string literal over 65535 bytes), so each call
    // builds a fresh string — they are called once per Open, not per frame.

    // ExampleSceneHoverArena.cpp, from tools/HoverArenaAuthor.cpp.
    [[nodiscard]] std::string hoverArenaJson();

    // ExampleSceneTimberYard.cpp, from tools/TimberYardAuthor.cpp.
    [[nodiscard]] std::string timberYardJson();

}// namespace threepp::editor::examples::detail

#endif//THREEPP_EDITOR_EXAMPLESCENEDATA_HPP
