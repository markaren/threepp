
#ifndef THREEPP_IMAGE_HPP
#define THREEPP_IMAGE_HPP

#include <vector>

namespace threepp {

    class Image {

    public:

        unsigned int width;
        unsigned int height;
        unsigned int depth = 0;

        Image(unsigned int width, unsigned int height, std::shared_ptr<unsigned char> data)
            : width(width), height(height), data_(data){};

        Image(unsigned int width, unsigned int height, unsigned int depth)
                : width(width), height(height), depth(depth){};

        [[nodiscard]] const unsigned char *getData() const {

            return data_.get();
        }

    private:
        std::shared_ptr<unsigned char> data_{};

    };

}

#endif//THREEPP_IMAGE_HPP
