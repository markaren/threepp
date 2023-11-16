
#include "threepp/renderers/gl/GLBackground.hpp"
#include "threepp/renderers/gl/GLCubeMaps.hpp"
#include "threepp/renderers/gl/GLState.hpp"

#include "threepp/renderers/GLRenderer.hpp"

using namespace threepp;
using namespace threepp::gl;

GLBackground::GLBackground(GLRenderer& renderer, GLCubeMaps& cubemaps, GLState& state, bool premultipliedAlpha)
    : renderer(renderer), cubemaps(cubemaps), state(state), premultipliedAlpha(premultipliedAlpha) {}

void GLBackground::render(Object3D* scene) {

    bool forceClear = false;
    bool isScene = scene->is<Scene>();

    auto& _background = scene->as<Scene>()->background;

    if (_background && _background.isTexture()) {
    }

    //    std::optional<Color> background = isScene ? scene->as<Scene>()->background : std::nullopt;

    if (!_background) {

        setClear(clearColor, clearAlpha);

    } else if (_background && _background.isColor()) {

        setClear(_background.color(), 1);
        forceClear = true;
    }

    if (renderer.autoClear || forceClear) {

        renderer.clear(renderer.autoClearColor, renderer.autoClearDepth, renderer.autoClearStencil);
    }

    if (_background && _background.isTexture()) {

        auto tex = _background.texture();
        if (auto cubeTexture = std::dynamic_pointer_cast<CubeTexture>(tex)) {
        }
    }
}

void GLBackground::setClearColor(const Color& color, float alpha) {

    clearColor.copy(color);
    clearAlpha = alpha;
    setClear(clearColor, clearAlpha);
}

const Color& GLBackground::getClearColor() const {

    return clearColor;
}

float GLBackground::getClearAlpha() const {

    return clearAlpha;
}

void GLBackground::setClearAlpha(float alpha) {

    clearAlpha = alpha;
    setClear(clearColor, clearAlpha);
}

void GLBackground::setClear(const Color& color, float alpha) {

    state.colorBuffer.setClear(color.r, color.g, color.b, alpha, premultipliedAlpha);
}
