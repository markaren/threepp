
#ifndef THREEPP_TEXTHANDLE_HPP
#define THREEPP_TEXTHANDLE_HPP

#include "threepp/math/Color.hpp"

#include <memory>
#include <string>

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

        explicit TextHandle(const std::string& str);

        void setText(const std::string& str);

        void setPosition(int xPos, int yPos) {
            x = xPos;
            y = yPos;
        }

        void invalidate() {
            invalidate_ = true;
        }

        [[nodiscard]] bool isValid() const {
            return !invalidate_;
        }

        ~TextHandle();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        bool invalidate_ = false;

        void render();

        static void setViewport(int width, int height);

        static bool init();

        static void beginDraw(bool blendingEnabled);
        static void endDraw(bool blendingEnabled);

        static void terminate();

        friend class GLRenderer;
    };

}// namespace threepp

#endif//THREEPP_TEXTHANDLE_HPP
