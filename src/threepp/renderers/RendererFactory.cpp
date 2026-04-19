
#include "threepp/renderers/RendererFactory.hpp"

#include "threepp/canvas/Canvas.hpp"
#include "threepp/renderers/GLRenderer.hpp"

#ifdef THREEPP_WITH_WGPU
#include "threepp/renderers/WgpuRenderer.hpp"
#endif

#include <iostream>
#include <stdexcept>

namespace threepp {

    std::unique_ptr<Renderer> createRenderer(Canvas& canvas, std::optional<GraphicsAPI> api) {

        GraphicsAPI chosen = api.value_or(GraphicsAPI::OpenGL);

#ifdef THREEPP_WITH_WGPU
        if (!api.has_value()) {
            std::cout << "Select renderer:\n  [1] OpenGL (default)\n  [2] WebGPU\n> ";
            std::string line;
            if (std::getline(std::cin, line) && line == "2") {
                chosen = GraphicsAPI::WebGPU;
            }
        }
#endif

        if (chosen == GraphicsAPI::WebGPU) {
#ifdef THREEPP_WITH_WGPU
            std::cout << "Using WebGPU renderer\n";
            auto renderer = std::make_unique<WgpuRenderer>(canvas);
            return renderer;
#else
            throw std::runtime_error("WebGPU/Wgpu renderer not available (build with -DTHREEPP_WITH_WGPU=ON)");
#endif
        }

        std::cout << "Using OpenGL renderer\n";
        auto renderer = std::make_unique<GLRenderer>(canvas);
        return renderer;
    }

}// namespace threepp
