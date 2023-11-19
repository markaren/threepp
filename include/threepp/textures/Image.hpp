
#ifndef THREEPP_IMAGE_HPP
#define THREEPP_IMAGE_HPP

#include <memory>
#include <utility>
#include <variant>
#include <vector>

namespace threepp {

    typedef std::variant<std::vector<unsigned char>, std::vector<float>> ImageData;

    class Image {

    public:
        unsigned int width;
        unsigned int height;
        unsigned int depth;

        Image(ImageData data, unsigned int width, unsigned int height, bool flipped = true)
            : data_(std::move(data)), width(width), height(height), depth(0), flipped_(flipped){};

        Image(ImageData data, unsigned int width, unsigned int height, unsigned int depth, bool flipped = true)
            : data_(std::move(data)), width(width), height(height), depth(depth), flipped_(flipped){};

        [[nodiscard]] bool flipped() const {

            return flipped_;
        }

        void setData(ImageData data) {

            data_ = std::move(data);
        }

        template<class T = unsigned char>
        [[nodiscard]] std::vector<T>& data() {

            return std::get<std::vector<T>>(data_);
        }

    private:
        bool flipped_;
        ImageData data_;
    };

}// namespace threepp

#endif//THREEPP_IMAGE_HPP
