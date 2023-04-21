
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

        Image(std::shared_ptr<unsigned char> data, unsigned int width, unsigned int height, bool flipped = true)
            : data_(std::move(data)), width(width), height(height), depth(0), flipped_(flipped){};

        Image(std::shared_ptr<unsigned char> data, unsigned int width, unsigned int height, unsigned int depth, bool flipped = true)
            : data_(std::move(data)), width(width), height(height), depth(depth), flipped_(flipped){};

        [[nodiscard]] bool flipped() const {

            return flipped_;
        }

        [[nodiscard]] unsigned char* getData() {

            return data_.get();
        }

        [[nodiscard]] const unsigned char* getData() const {

            return data_.get();
        }

    private:
        bool flipped_;
        std::shared_ptr<unsigned char> data_{};
    };

}// namespace threepp

#endif//THREEPP_IMAGE_HPP
