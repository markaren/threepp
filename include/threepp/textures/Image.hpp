
#ifndef THREEPP_IMAGE_HPP
#define THREEPP_IMAGE_HPP

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

        Image(ImageData data, unsigned int width, unsigned int height)
            : width(width), height(height), depth(0), data_(std::move(data)){};

        Image(ImageData data, unsigned int width, unsigned int height, unsigned int depth)
            : width(width), height(height), depth(depth), data_(std::move(data)){};

        void setData(ImageData data) {

            data_ = std::move(data);
        }

        template<class T = unsigned char>
        [[nodiscard]] std::vector<T>& data() {

            return std::get<std::vector<T>>(data_);
        }

    private:
        ImageData data_;
    };

}// namespace threepp

#endif//THREEPP_IMAGE_HPP
