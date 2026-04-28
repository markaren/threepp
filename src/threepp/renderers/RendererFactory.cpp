
#include "threepp/renderers/RendererFactory.hpp"

#include "threepp/canvas/Canvas.hpp"
#include "threepp/renderers/GLRenderer.hpp"

#ifdef THREEPP_WITH_WGPU
#include "threepp/renderers/CrossRenderer.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#endif

#include <iostream>
#include <stdexcept>

namespace threepp {

    std::unique_ptr<Renderer> createRenderer(Canvas& canvas, std::optional<GraphicsAPI> api) {

        GraphicsAPI chosen = api.value_or(GraphicsAPI::OpenGL);

#ifdef THREEPP_WITH_WGPU
        if (!api.has_value()) {
            std::cout << "Select renderer:\n  [1] OpenGL (default)\n  [2] WebGPU\n  [3] Cross (WGPU rendering, GL display)\n> ";
            std::string line;
            if (std::getline(std::cin, line)) {
                if (line == "2") chosen = GraphicsAPI::WebGPU;
                else if (line == "3") chosen = GraphicsAPI::Cross;
            }
        }
#endif

        if (chosen == GraphicsAPI::WebGPU) {
#ifdef THREEPP_WITH_WGPU
            std::cout << "Using WebGPU renderer\n";
            return std::make_unique<WgpuRenderer>(canvas);
#else
            throw std::runtime_error("WebGPU/Wgpu renderer not available (build with -DTHREEPP_WITH_WGPU=ON)");
#endif
        }

        if (chosen == GraphicsAPI::Cross) {
#ifdef THREEPP_WITH_WGPU
            std::cout << "Using Cross renderer (Split screen GL/WGPU rendering)\n";
            return std::make_unique<CrossRenderer>(canvas);
#else
            throw std::runtime_error("Cross renderer requires WebGPU support (build with -DTHREEPP_WITH_WGPU=ON)");
#endif
        }

        std::cout << "Using OpenGL renderer\n";
        return std::make_unique<GLRenderer>(canvas);
    }

}// namespace threepp
