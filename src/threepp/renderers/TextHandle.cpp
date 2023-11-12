
#include "threepp/renderers/TextRenderer.hpp"

#define GLT_IMPLEMENTATION
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

void TextHandle::setText(const std::string& str) {
    gltSetText(pimpl_->text, str.c_str());
}

void TextHandle::render() {
    gltColor(color.r, color.g, color.b, alpha);
    gltDrawText2DAligned(pimpl_->text, static_cast<float>(x), static_cast<float>(y), scale, getValue(horizontalAlignment), getValue(verticalAlignment));
}

int threepp::TextHandle::getTextSize() const {

    return static_cast<int>(gltGetTextHeight(pimpl_->text, scale));
}

TextHandle::~TextHandle() = default;

TextRenderer::TextRenderer() {
    gltInit();
}

TextRenderer::~TextRenderer() {
    gltTerminate();
}

void TextRenderer::setViewport(int width, int height) {
    if (width > 0 && height > 0) {
        gltViewport(width, height);
    }
}

void TextRenderer::render() {

    if (textHandles_.empty()) return;

    glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
    gltBeginDraw();

    auto it = textHandles_.begin();
    while (it != textHandles_.end()) {

        if ((*it)->invalidate_) {
            it = textHandles_.erase(it);
        } else {
            (*it)->render();
            ++it;
        }
    }

    gltEndDraw();
    glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
}
