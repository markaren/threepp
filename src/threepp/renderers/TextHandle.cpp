
#include "threepp/renderers/TextHandle.hpp"

#define GLT_IMPLEMENTATION
#define GLT_MANUAL_VIEWPORT
#include <glad/glad.h>
#include <gltext.h>

using namespace threepp;

struct TextHandle::Impl {

    GLTtext *text = gltCreateText();

    ~Impl() {
        gltDeleteText(text);
    }
};

TextHandle::TextHandle(const std::string &str)
    : pimpl_(std::make_unique<Impl>()) {
    setText(str);
}

bool threepp::TextHandle::init() {
    return gltInit();
}

void TextHandle::setText(const std::string &str) {
    gltSetText(pimpl_->text, str.c_str());
}

void threepp::TextHandle::render(bool blendingEnabled) {
    if (!blendingEnabled) {
        glEnable(GL_BLEND);
    }
    gltColor(color.r, color.g, color.b, alpha);
    gltDrawText2DAligned(pimpl_->text, static_cast<float>(x), static_cast<float>(y), scale, horizontalAlignment, verticalAlignment);
    if (!blendingEnabled) {
        glDisable(GL_BLEND);
    }
}

void threepp::TextHandle::setViewport(int width, int height) {
    gltViewport(width, height);
}

void threepp::TextHandle::terminate() {
    gltTerminate();
}

void threepp::TextHandle::beginDraw() {
    gltBeginDraw();
}

void threepp::TextHandle::endDraw() {
    gltEndDraw();
}

TextHandle::~TextHandle() = default;
