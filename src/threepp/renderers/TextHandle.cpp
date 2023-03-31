
#include "threepp/renderers/TextHandle.hpp"

#define GLT_IMPLEMENTATION
#define GLT_MANUAL_VIEWPORT
#include <glad/glad.h>
#include <gltext.h>

using namespace threepp;

struct TextHandle::Impl {

    GLTtext* text = gltCreateText();

    ~Impl() {
        gltDeleteText(text);
    }
};

TextHandle::TextHandle(const std::string& str)
    : pimpl_(std::make_unique<Impl>()) {
    setText(str);
}

bool TextHandle::init() {
    return gltInit();
}

void TextHandle::setText(const std::string& str) {
    gltSetText(pimpl_->text, str.c_str());
}

void TextHandle::render() {
    gltColor(color.r, color.g, color.b, alpha);
    gltDrawText2DAligned(pimpl_->text, static_cast<float>(x), static_cast<float>(y), scale, horizontalAlignment, verticalAlignment);
}

void TextHandle::setViewport(int width, int height) {
    if (width > 0 && height > 0) {
        gltViewport(width, height);
    }
}

void TextHandle::terminate() {
    gltTerminate();
}

void TextHandle::beginDraw(bool blendingEnabled) {
    if (!blendingEnabled) {
        glEnable(GL_BLEND);
    }
    gltBeginDraw();
}

void TextHandle::endDraw(bool blendingEnabled) {
    gltEndDraw();
    if (!blendingEnabled) {
        glDisable(GL_BLEND);
    }
}

TextHandle::~TextHandle() = default;
