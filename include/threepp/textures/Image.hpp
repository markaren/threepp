
#ifndef THREEPP_IMAGE_HPP
#define THREEPP_IMAGE_HPP

#include <utility>

namespace threepp {

    class Image {

    public:

        Image(unsigned int width, unsigned int height, const unsigned char *data = nullptr)
            : width_(width), height_(height), data_(data){};

        [[nodiscard]] unsigned int width() const {

            return width_;
        }

        [[nodiscard]] unsigned int height() const {

            return height_;
        }

        [[nodiscard]] const unsigned char *getData() const {

            return data_;
        }

        ~Image() {

            delete data_;
        }

    private:
        unsigned int width_;
        unsigned int height_;

        const unsigned char* data_;

    };

}

#endif//THREEPP_IMAGE_HPP
