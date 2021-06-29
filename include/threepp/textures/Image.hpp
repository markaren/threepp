
#ifndef THREEPP_IMAGE_HPP
#define THREEPP_IMAGE_HPP

#include <vector>

namespace threepp {

    class Image {

    public:

        unsigned int width;
        unsigned int height;
        unsigned int depth = 0;

        Image(unsigned int width, unsigned int height, std::vector<unsigned char> data = {})
            : width(width), height(height), data_(std::move(data)){};

        Image(unsigned int width, unsigned int height, unsigned int depth)
                : width(width), height(height), depth(depth){};

        [[nodiscard]] const std::vector<unsigned char> &getData() const {

            return data_;
        }

    private:
        std::vector<unsigned char> data_;

    };

}

#endif//THREEPP_IMAGE_HPP
