
#include "threepp/renderers/gl/GLBackground.hpp"
#include "threepp/renderers/gl/GLState.hpp"

#include "threepp/renderers/GLRenderer.hpp"

using namespace threepp;
using namespace threepp::gl;

GLBackground::GLBackground(GLState& state, bool premultipliedAlpha): state(state), premultipliedAlpha(premultipliedAlpha) {}

void GLBackground::render(GLRenderer& renderer, Scene* scene) {

    bool forceClear = false;
    auto& background = scene->background;

    if (!background) {

        setClear(clearColor, clearAlpha);

    } else {

        setClear(*background, 1);
        forceClear = true;
    }

    if (renderer.autoClear || forceClear) {

        renderer.clear(renderer.autoClearColor, renderer.autoClearDepth, renderer.autoClearStencil);
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
