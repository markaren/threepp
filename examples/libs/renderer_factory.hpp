// renderer_factory.hpp — the examples' "which backend?" helper.
//
// Not library API, and deliberately not in the threepp namespace: choosing a
// renderer by PROMPTING ON STDIN is a demo convenience. It is how an example
// lets you compare GL against Vulkan without a rebuild, and it is the one thing
// an application must never do — printing a menu and blocking on std::cin
// stalls a windowed program behind a prompt nobody sees and hangs every piped
// or scripted run. The editor and the player therefore name their backend in
// code (see EditorApp.cpp / PlayerApp.cpp), and the prompt lives here, beside
// the only code that wants it.
//
// Header-only and in examples/libs, which is on every example target's include
// path (examples/AddExample.cmake), so any demo can just
// `#include "renderer_factory.hpp"`.
#pragma once

#include "threepp/canvas/Canvas.hpp"
#include "threepp/renderers/GLRenderer.hpp"
#include "threepp/renderers/Renderer.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

inline std::unique_ptr<threepp::Renderer> createRenderer(
        threepp::Canvas& canvas,
        std::optional<threepp::GraphicsAPI> api = std::nullopt) {

    using threepp::GraphicsAPI;

    GraphicsAPI chosen = api.value_or(GraphicsAPI::OpenGL);

    if (!api.has_value()) {
#ifdef __EMSCRIPTEN__
        // The browser has no interactive console: reading std::cin triggers a
        // blocking window.prompt() popup (or hangs the demo). Skip the menu and
        // keep the default backend. Pass an explicit GraphicsAPI to override.
        chosen = GraphicsAPI::OpenGL;
#else
        std::cout << "Select renderer:\n  [1] OpenGL (default)";
#ifdef THREEPP_WITH_VULKAN
        std::cout << "\n  [2] Vulkan Deferred renderer";
#endif
        std::cout << "\n  [0] Abort and Exit\n> ";
        std::string line;
        if (std::getline(std::cin, line)) {
            // piped stdin on Windows delivers "2\r" — strip trailing whitespace
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
                line.pop_back();
            }
#ifdef THREEPP_WITH_VULKAN
            if (line == "2") chosen = GraphicsAPI::Vulkan;
#endif
        }
        if (line == "0") {
            std::cout << "Aborting.\n";
            std::exit(0);
        }
#endif// __EMSCRIPTEN__
    }

    if (chosen == GraphicsAPI::Vulkan) {
#ifdef THREEPP_WITH_VULKAN
        std::cout << "Using Vulkan renderer\n";
        return std::make_unique<threepp::VulkanRenderer>(canvas);
#else
        throw std::runtime_error("Vulkan renderer not available (build with -DTHREEPP_WITH_VULKAN=ON)");
#endif
    }

    std::cout << "Using OpenGL renderer\n";
    return std::make_unique<threepp::GLRenderer>(canvas);
}
