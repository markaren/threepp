
#ifndef THREEPP_RENDERERFACTORY_HPP
#define THREEPP_RENDERERFACTORY_HPP

#include "threepp/renderers/Renderer.hpp"

#include <memory>
#include <optional>

namespace threepp {

    class Canvas;

    std::unique_ptr<Renderer> createRenderer(Canvas& canvas, std::optional<GraphicsAPI> api = std::nullopt);

}// namespace threepp

#endif//THREEPP_RENDERERFACTORY_HPP
