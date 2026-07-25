
#include "threepp/renderers/RendererFactory.hpp"

#include "threepp/canvas/Canvas.hpp"
#include "threepp/renderers/GLRenderer.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <iostream>
#include <stdexcept>

namespace threepp {

    std::unique_ptr<Renderer> createRenderer(Canvas& canvas, std::optional<GraphicsAPI> api) {

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
            return std::make_unique<VulkanRenderer>(canvas);
#else
            throw std::runtime_error("Vulkan renderer not available (build with -DTHREEPP_WITH_VULKAN=ON)");
#endif
        }

        std::cout << "Using OpenGL renderer\n";
        return std::make_unique<GLRenderer>(canvas);
    }

}// namespace threepp
