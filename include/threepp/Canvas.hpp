
#ifndef THREEPP_CANVAS_HPP
#define THREEPP_CANVAS_HPP

#include <functional>
#include <memory>

#include "threepp/core/Clock.hpp"

namespace threepp {

    class Canvas {

    public:
        struct Parameters {

            int width_ = 640;
            int height_ = 480;

            std::string title_;

            Parameters() = default;

            Parameters &title(std::string value) {

                this->title_ = std::move(value);

                return *this;
            }

            Parameters &width(unsigned int value) {

                this->width_ = value;

                return *this;
            }

            Parameters &height(unsigned int value) {

                this->height_ = value;

                return *this;
            }
        };

        explicit Canvas(const Parameters &params);

        Canvas(const Canvas &) = delete;

        [[nodiscard]] int getWidth() const;

        [[nodiscard]] int getHeight() const;

        void animate(const std::function<void(float)> &f) const;

        ~Canvas();

    private:
        struct Impl;

        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_CANVAS_HPP
