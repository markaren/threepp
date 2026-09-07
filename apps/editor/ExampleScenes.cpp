// The scenes the editor ships, and the order File ▸ Open Example lists them in.
//
// HAND-WRITTEN, unlike the documents themselves: each example's JSON lives in
// its own generated translation unit (see ExampleSceneData.hpp), written by the
// program that authors that scene. Keeping the registry out of those files is
// what lets a second example exist without the first one's author program
// knowing about it — the earlier arrangement, where one generated file carried
// both the bytes and the list, meant regenerating one example dropped the other
// from the menu.
//
// Adding an example is: write tools/<Name>Author.cpp, add its target and its
// generated .cpp to apps/editor/CMakeLists.txt, declare its accessor in
// ExampleSceneData.hpp, and add the two lines below.

#include "ExampleScenes.hpp"

#include "ExampleSceneData.hpp"

#include <string>
#include <vector>

namespace threepp::editor::examples {

    const std::vector<Example>& all() {

        // Menu order. Hover Arena first because it is the one that teaches the
        // editor's runtime; Timber Yard second because it assumes it.
        static const std::vector<Example> examples{
                {"hover-arena", "Hover Arena",
                 "a physics drone, five trigger rings and a scoreboard - fly it with W/S/A/D"},
                {"timber-yard", "Timber Yard",
                 "a jointed gate, a conveyor and a breakable stop bar, run by coroutine scripts"},
        };
        return examples;
    }

    const Example* find(std::string_view slug) {

        for (const auto& example : all()) {
            if (example.slug == slug) return &example;
        }
        return nullptr;
    }

    std::string json(std::string_view slug) {

        if (slug == "hover-arena") return detail::hoverArenaJson();
        if (slug == "timber-yard") return detail::timberYardJson();
        // An unknown slug has no document, which is the same answer as "no
        // document" — the caller has to check either way.
        return {};
    }

}// namespace threepp::editor::examples
