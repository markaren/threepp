
#include "threepp/renderers/RendererFactory.hpp"

#include "threepp/canvas/Canvas.hpp"
#include "threepp/renderers/GLRenderer.hpp"

#ifdef THREEPP_WITH_DAWN
#include "threepp/renderers/DawnRenderer.hpp"
#endif

#include <stdexcept>

namespace threepp {

    std::unique_ptr<Renderer> createRenderer(Canvas& canvas) {
        if (canvas.graphicsApi() == GraphicsAPI::WebGPU) {
#ifdef THREEPP_WITH_DAWN
            return std::make_unique<DawnRenderer>(canvas);
#else
            throw std::runtime_error("WebGPU/Dawn renderer not available (build with -DTHREEPP_WITH_DAWN=ON)");
#endif
        }
        return std::make_unique<GLRenderer>(canvas.size());
    }

}// namespace threepp
