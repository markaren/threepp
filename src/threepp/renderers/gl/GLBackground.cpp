
#include "threepp/renderers/gl/GLBackground.hpp"

#include "threepp/renderers/GLRenderer.hpp"

#include "threepp/utils/InstanceOf.hpp"

using namespace threepp::gl;

GLBackground::GLBackground(GLState &state, bool premultipliedAlpha) : state(state), premultipliedAlpha(premultipliedAlpha) {}

void GLBackground::render(threepp::GLRenderer &renderer, threepp::Object3D *scene) {

    bool forceClear = false;
    bool isScene = instanceof <Scene>(scene);
    std::optional<Color> background = isScene ? dynamic_cast<Scene *>(scene)->background : std::nullopt;

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

void GLBackground::setClearColor(const threepp::Color &color, float alpha) {

    clearColor.copy(color);
    clearAlpha = alpha;
    setClear(clearColor, clearAlpha);
}

const threepp::Color &GLBackground::getClearColor() const {

    return clearColor;
}

float GLBackground::getClearAlpha() const {

    return clearAlpha;
}

void GLBackground::setClearAlpha(float alpha) {

    clearAlpha = alpha;
    setClear(clearColor, clearAlpha);
}

void GLBackground::setClear(const threepp::Color &color, float alpha) {

    state.colorBuffer.setClear(color.r, color.g, color.b, alpha, premultipliedAlpha);
}
