
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
        unsigned int depth = 0;

        Image(unsigned int width, unsigned int height, std::shared_ptr<unsigned char> data, bool flipped = true)
            : width(width), height(height), data_(std::move(data)), flipped_(flipped){};

        Image(unsigned int width, unsigned int height, unsigned int depth)
            : width(width), height(height), depth(depth), flipped_(false){};

        [[nodiscard]] bool flipped() const {

            return flipped_;
        }

        [[nodiscard]] unsigned char *getData() {

            return data_.get();
        }

        [[nodiscard]] const unsigned char *getData() const {

            return data_.get();
        }

    private:
        bool flipped_;
        std::shared_ptr<unsigned char> data_{};
    };

}// namespace threepp

#endif//THREEPP_IMAGE_HPP
