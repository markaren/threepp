
#ifndef THREEPP_IMAGE_HPP
#define THREEPP_IMAGE_HPP

#include <memory>
#include <utility>
#include <vector>

namespace threepp {

    class Image {

    public:
        enum class Format {
            RGB,
            RGBA
        };

        unsigned int width;
        unsigned int height;
        unsigned int depth;

        Image(const std::shared_ptr<unsigned char>& data, unsigned int width, unsigned int height, Format format, bool flipped = true)
            : Image(data, width, height, 0, format, flipped){};

        Image(const std::shared_ptr<unsigned char>& data, unsigned int width, unsigned int height, unsigned int depth, Format format, bool flipped = true)
            : data_(data), width(width), height(height), depth(depth), format_(format), flipped_(flipped){};

        [[nodiscard]] bool flipped() const {

            return flipped_;
        }

        [[nodiscard]] int numChannels() const {
            return getChannels(format_);
        }

        [[nodiscard]] unsigned char* getData() {

            return data_.get();
        }

        [[nodiscard]] const unsigned char* getData() const {

            return data_.get();
        }

        static int getChannels(const Format& format) {
            switch (format) {
                case Format::RGB:
                    return 3;
                case Format::RGBA:
                    return 4;
                default:
                    return -1;
            }
        }

    private:
        bool flipped_;
        Format format_;
        std::shared_ptr<unsigned char> data_{};
    };

}// namespace threepp

#endif//THREEPP_IMAGE_HPP
