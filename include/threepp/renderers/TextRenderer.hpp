
#ifndef THREEPP_TEXTRENDERER_HPP
#define THREEPP_TEXTRENDERER_HPP

#include "threepp/math/Color.hpp"

#include <memory>
#include <string>
#include <vector>

namespace threepp {

    class TextHandle {

    public:
        enum class HorizontalAlignment {
            LEFT,
            CENTER,
            RIGHT
        };

        enum class VerticalAlignment {
            TOP,
            CENTER,
            BOTTOM
        };

        int x = 0;
        int y = 0;
        float scale = 1;

        Color color;
        float alpha = 1;
        HorizontalAlignment horizontalAlignment{HorizontalAlignment::LEFT};
        VerticalAlignment verticalAlignment{VerticalAlignment::TOP};

        [[nodiscard]] int getTextSize() const;

        void setText(const std::string& str);

        void setPosition(int xPos, int yPos) {
            x = xPos;
            y = yPos;
        }

        void invalidate() {
            invalidate_ = true;
        }

        ~TextHandle();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        bool invalidate_ = false;

        explicit TextHandle(const std::string& str);

        void render();

        friend class TextRenderer;
    };


    class TextRenderer {

    public:
        TextRenderer();

        TextRenderer(TextRenderer&&) = delete;
        TextRenderer(const TextRenderer&) = delete;
        TextRenderer& operator=(const TextRenderer&) = delete;

        TextHandle* createHandle(const std::string& text = "");

        void render();

        void clear();

        ~TextRenderer();

    private:
        std::vector<std::unique_ptr<TextHandle>> textHandles_;
    };


}// namespace threepp

#endif//THREEPP_TEXTRENDERER_HPP
