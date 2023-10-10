
#ifndef THREEPP_IMAGE_HPP
#define THREEPP_IMAGE_HPP

#include <memory>
#include <utility>
#include <vector>

namespace threepp {

    class Image {

    public:
        unsigned int width;
        unsigned int height;
        unsigned int depth;

        Image(const std::vector<unsigned char>& data, unsigned int width, unsigned int height, bool flipped = true)
            : data_(data), width(width), height(height), depth(0), flipped_(flipped){};

        Image(const std::vector<unsigned char>& data, unsigned int width, unsigned int height, unsigned int depth, bool flipped = true)
            : data_(data), width(width), height(height), depth(depth), flipped_(flipped){};

        [[nodiscard]] bool flipped() const {

            return flipped_;
        }

        [[nodiscard]] std::vector<unsigned char>& data() {

            return data_;
        }

    private:
        bool flipped_;
        std::vector<unsigned char> data_;
    };

}// namespace threepp

#endif//THREEPP_IMAGE_HPP
