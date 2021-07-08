
#ifndef THREEPP_CANVAS_HPP
#define THREEPP_CANVAS_HPP

#include <functional>
#include <memory>

#include "threepp/core/Clock.hpp"

namespace threepp {

    struct WindowSize {
        int width;
        int height;

        [[nodiscard]] float getAspect() const {
            return (float) width / (float) height;
        }
    };

    class Canvas {

    public:
        struct Parameters;

        explicit Canvas(const Parameters &params = Parameters());

        [[nodiscard]] WindowSize getSize() const;

        [[nodiscard]] float getAspect() const;

        void setSize(WindowSize size);

        void onWindowResize(std::function<void(WindowSize)> f);

        void animate(const std::function<void(float)> &f) const;

        ~Canvas();

    private:
        class Impl;
        std::unique_ptr<Impl> pimpl_;

    public:
        struct Parameters {

            Parameters()
                : size_{640, 480},
                  antialiasing_{0} {}

            Parameters&title(std::string value) {

                this->title_ = std::move(value);

                return *this;
            }

            Parameters &size(WindowSize size) {

                this->size_ = size;

                return *this;
            }

            Parameters &size(int width, int height) {

                return this->size({width, height});
            }

            Parameters &antialising(int antialiasing) {

                this->antialiasing_ = antialiasing;

                return *this;
            }

        private:
            WindowSize size_;
            int antialiasing_;
            std::string title_ = "untitled";

            friend class Canvas::Impl;
        };
    };

}// namespace threepp

#endif//THREEPP_CANVAS_HPP
