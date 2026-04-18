
#include "threepp/renderers/RendererFactory.hpp"

#include "threepp/canvas/Canvas.hpp"
#include "threepp/renderers/GLRenderer.hpp"

#ifdef THREEPP_WITH_WGPU
#include "threepp/renderers/WgpuRenderer.hpp"
#endif

#include <stdexcept>

namespace threepp {

    std::unique_ptr<Renderer> createRenderer(Canvas& canvas) {
        if (canvas.graphicsApi() == GraphicsAPI::WebGPU) {
#ifdef THREEPP_WITH_WGPU
            return std::make_unique<WgpuRenderer>(canvas);
#else
            throw std::runtime_error("WebGPU/Wgpu renderer not available (build with -DTHREEPP_WITH_WGPU=ON)");
#endif
        }
        return std::make_unique<GLRenderer>(canvas);
    }

}// namespace threepp
