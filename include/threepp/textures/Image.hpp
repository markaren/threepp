
#ifndef THREEPP_IMAGE_HPP
#define THREEPP_IMAGE_HPP

#include <optional>
#include <utility>
#include <variant>
#include <vector>
#include <cstdint>

namespace threepp {

    typedef std::variant<std::vector<unsigned char>, std::vector<float>> ImageData;

    class Image {

    public:
        unsigned int width;
        unsigned int height;
        unsigned int depth;

        struct CompressedFormat {
            uint32_t format;
        };

        // When set, the data_ buffer holds a compressed block payload and this
        // value is the GL compressed internal format token (e.g.
        // GL_COMPRESSED_RGBA_S3TC_DXT5_EXT). Used by GLTextures to call
        // glCompressedTexImage2D instead of glTexImage2D.
        std::optional<unsigned int> compressedFormat;

        Image(ImageData data, unsigned int width, unsigned int height)
            : width(width), height(height), depth(0), data_(std::move(data)){};

        Image(ImageData data, unsigned int width, unsigned int height, unsigned int depth)
            : width(width), height(height), depth(depth), data_(std::move(data)){};

        // Constructor for compressed block data.
        Image(std::vector<unsigned char> data, unsigned int width, unsigned int height,
              CompressedFormat glCompressedFormat)
            : width(width), height(height), depth(0),
              compressedFormat(glCompressedFormat.format),
              data_(std::move(data)){};

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
