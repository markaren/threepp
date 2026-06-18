
#include "threepp/renderers/RendererFactory.hpp"

#include "threepp/canvas/Canvas.hpp"
#include "threepp/renderers/GLRenderer.hpp"

#ifdef THREEPP_WITH_WGPU
#include "threepp/renderers/CrossRenderer.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#endif

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <iostream>
#include <stdexcept>

namespace threepp {

    std::unique_ptr<Renderer> createRenderer(Canvas& canvas, std::optional<GraphicsAPI> api) {

        GraphicsAPI chosen = api.value_or(GraphicsAPI::OpenGL);

        if (!api.has_value()) {
            std::cout << "Select renderer:\n  [1] OpenGL (default)";
#ifdef THREEPP_WITH_WGPU
            std::cout << "\n  [2] WebGPU\n  [3] Cross (GL left, WGPU right)";
#endif
#ifdef THREEPP_WITH_VULKAN
            std::cout << "\n  [4] Vulkan Path-tracer";
#endif
            std::cout << "\n  [0] Abort and Exit\n> ";
            std::string line;
            if (std::getline(std::cin, line)) {
                // piped stdin on Windows delivers "2\r" — strip trailing whitespace
                while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
                    line.pop_back();
                }
#ifdef THREEPP_WITH_WGPU
                if (line == "2") chosen = GraphicsAPI::WebGPU;
                else if (line == "3") chosen = GraphicsAPI::Cross;
#endif
#ifdef THREEPP_WITH_VULKAN
                if (line == "4") chosen = GraphicsAPI::Vulkan;
#endif
            }
            if (line == "0") {
                std::cout << "Aborting.\n";
                std::exit(0);
            }
        }

        if (chosen == GraphicsAPI::WebGPU) {
#ifdef THREEPP_WITH_WGPU
            std::cout << "Using WebGPU renderer\n";
            return std::make_unique<WgpuRenderer>(canvas);
#else
            throw std::runtime_error("WebGPU renderer not available (build with -DTHREEPP_WITH_WGPU=ON)");
#endif
        }

        if (chosen == GraphicsAPI::Cross) {
#ifdef THREEPP_WITH_WGPU
            std::cout << "Using Cross renderer\n";
            return std::make_unique<CrossRenderer>(canvas);
#else
            throw std::runtime_error("Cross renderer not available (build with -DTHREEPP_WITH_WGPU=ON)");
#endif
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
