
#ifndef THREEPP_CANVAS_HPP
#define THREEPP_CANVAS_HPP

#include <functional>
#include <memory>

#include "threepp/core/Clock.hpp"

namespace threepp {

    class Canvas {

    public:
        struct Parameters {

            int width_;
            int height_;

            std::string title_;

            Parameters(): width_(640), height_(480) {}

            Parameters &title(std::string value) {

                this->title_ = std::move(value);

                return *this;
            }

            Parameters &width(int value) {

                this->width_ = value;

                return *this;
            }

            Parameters &height(int value) {

                this->height_ = value;

                return *this;
            }
        };

        explicit Canvas(const Parameters &params = Parameters());

        [[nodiscard]] int getWidth() const;

        [[nodiscard]] int getHeight() const;

        [[nodiscard]] int getAspect() const;

        void setSize(int width, int height);

        void animate(const std::function<void(float)> &f) const;

        ~Canvas();

    private:
        class Impl;

        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_CANVAS_HPP
