
#include "threepp/renderers/TextHandle.hpp"

#define GLT_IMPLEMENTATION
#define GLT_MANUAL_VIEWPORT
#include <glad/glad.h>
#include <gltext.h>

using namespace threepp;

namespace {

    inline int getValue(TextHandle::HorizontalAlignment alignment) {
        switch (alignment) {
            case TextHandle::HorizontalAlignment::LEFT:
                return 0;
            case TextHandle::HorizontalAlignment::CENTER:
                return 1;
            case TextHandle::HorizontalAlignment::RIGHT:
                return 2;
            default:
                return 0;
        }
    }

    inline int getValue(TextHandle::VerticalAlignment alignment) {
        switch (alignment) {
            case TextHandle::VerticalAlignment::TOP:
                return 0;
            case TextHandle::VerticalAlignment::CENTER:
                return 1;
            case TextHandle::VerticalAlignment::BOTTOM:
                return 2;
            default:
                return 0;
        }
    }

}// namespace

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
    gltDrawText2DAligned(pimpl_->text, static_cast<float>(x), static_cast<float>(y), scale, getValue(horizontalAlignment), getValue(verticalAlignment));
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
    glDisable(GL_DEPTH_TEST);
    gltBeginDraw();
}

void TextHandle::endDraw(bool blendingEnabled) {
    gltEndDraw();
    if (!blendingEnabled) {
        glDisable(GL_BLEND);
    }
    glEnable(GL_DEPTH_TEST);
}

TextHandle::~TextHandle() = default;
