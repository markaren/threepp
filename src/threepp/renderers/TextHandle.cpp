
#include "threepp/renderers/TextRenderer.hpp"


#ifndef EMSCRIPTEN
#define GLT_IMPLEMENTATION
#include <glad/glad.h>
#include <gltext.h>
#endif


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

#ifndef EMSCRIPTEN
struct TextHandle::Impl {

    GLTtext* text = gltCreateText();

    ~Impl() {
        gltDeleteText(text);
    }
};
#else
struct TextHandle::Impl {
};
#endif

TextHandle::TextHandle(const std::string& str)
    : pimpl_(std::make_unique<Impl>()) {
    setText(str);
}

void TextHandle::setText(const std::string& str) {
#ifndef EMSCRIPTEN
    gltSetText(pimpl_->text, str.c_str());
#endif
}

void TextHandle::render() {
#ifndef EMSCRIPTEN
    gltColor(color.r, color.g, color.b, alpha);
    gltDrawText2DAligned(pimpl_->text, static_cast<float>(x), static_cast<float>(y), scale, getValue(horizontalAlignment), getValue(verticalAlignment));
#endif
}

int threepp::TextHandle::getTextSize() const {
#ifndef EMSCRIPTEN
    return static_cast<int>(gltGetTextHeight(pimpl_->text, scale));
#else
    return 0;
#endif
}

TextHandle::~TextHandle() = default;

TextRenderer::TextRenderer() {
#ifndef EMSCRIPTEN
    gltInit();
#endif
}

TextRenderer::~TextRenderer() {
#ifndef EMSCRIPTEN
    gltTerminate();
#endif
}

void TextRenderer::render() {

    if (textHandles_.empty()) return;

#ifndef EMSCRIPTEN
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
#endif
}

void TextRenderer::clear() {

    textHandles_.clear();
}

TextHandle* TextRenderer::createHandle(const std::string& text) {

    auto handle = std::unique_ptr<TextHandle>(new TextHandle(text));
    textHandles_.emplace_back(std::move(handle));

    return textHandles_.back().get();
}
