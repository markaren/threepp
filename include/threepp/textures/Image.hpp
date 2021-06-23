
#ifndef THREEPP_IMAGE_HPP
#define THREEPP_IMAGE_HPP

#include <utility>

namespace threepp {

    class Image {

    public:

        Image(unsigned int width, unsigned int height, std::string data = std::string())
            : width_(width), height_(height), data_(std::move(data)){};

        [[nodiscard]] unsigned int width() const {
            return width_;
        }

        [[nodiscard]] unsigned int height() const {
            return height_;
        }

        [[nodiscard]] const std::string &getData() const {
            return data_;
        }

    private:
        unsigned int width_;
        unsigned int height_;

        std::string data_;

    };

}

#endif//THREEPP_IMAGE_HPP
