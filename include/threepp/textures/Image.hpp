
#ifndef THREEPP_IMAGE_HPP
#define THREEPP_IMAGE_HPP

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace threepp {

    using ImageData = std::variant<std::vector<unsigned char>, std::vector<float>>;

    class Image {

    public:
        struct CompressedFormat {
            uint32_t format;
        };

        std::optional<unsigned int> compressedFormat;

        Image(ImageData data, unsigned int width, unsigned int height)
            : width_(width), height_(height), depth_(0), data_(std::move(data)) {}

        Image(ImageData data, unsigned int width, unsigned int height, unsigned int depth)
            : width_(width), height_(height), depth_(depth), data_(std::move(data)) {}

        Image(std::vector<unsigned char> data, unsigned int width, unsigned int height,
              CompressedFormat glCompressedFormat)
            : width_(width), height_(height), depth_(0),
              compressedFormat(glCompressedFormat.format),
              data_(std::move(data)) {}

        [[nodiscard]] unsigned int width() const noexcept { return width_; }
        [[nodiscard]] unsigned int height() const noexcept { return height_; }
        [[nodiscard]] unsigned int depth() const noexcept { return depth_; }

        [[nodiscard]] bool isFloat() const noexcept {
            return std::holds_alternative<std::vector<float>>(data_);
        }

        [[nodiscard]] int channels() const noexcept {
            const size_t slices = depth_ > 0 ? depth_ : 1;
            const size_t pixels = static_cast<size_t>(width_) * height_ * slices;
            if (pixels == 0) return 0;
            return std::visit([pixels](const auto& v) -> int {
                return static_cast<int>(v.size() / pixels);
            }, data_);
        }

        void setData(ImageData data) {
            data_ = std::move(data);
        }

        template<class T = unsigned char>
        [[nodiscard]] std::vector<T>& data() {
            return std::get<std::vector<T>>(data_);
        }

        template<class T = unsigned char>
        [[nodiscard]] const std::vector<T>& data() const {
            return std::get<std::vector<T>>(data_);
        }

    private:
        unsigned int width_;
        unsigned int height_;
        unsigned int depth_;
        ImageData data_;
    };

}// namespace threepp

#endif//THREEPP_IMAGE_HPP
