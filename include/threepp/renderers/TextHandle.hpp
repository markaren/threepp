
#ifndef THREEPP_TEXTHANDLE_HPP
#define THREEPP_TEXTHANDLE_HPP

#include "threepp/math/Color.hpp"

#include <string>
#include <memory>

namespace threepp {

    class TextHandle {

    public:
        int x = 0;
        int y = 0;
        float scale = 1;

        Color color;
        float alpha = 1;

        explicit TextHandle(const std::string& str);

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

        void render();

        friend class GLRenderer;
    };

}

#endif//THREEPP_TEXTHANDLE_HPP
