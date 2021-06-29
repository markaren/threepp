
#ifndef THREEPP_IMAGE_HPP
#define THREEPP_IMAGE_HPP

namespace threepp {

    class Image {

    public:

        unsigned int width;
        unsigned int height;
        unsigned int depth = 0;

        Image(unsigned int width, unsigned int height, const unsigned char *data = nullptr)
            : width(width), height(height), data_(data){};

        Image(unsigned int width, unsigned int height, unsigned int depth)
                : width(width), height(height), depth(depth), data_(nullptr){};

        [[nodiscard]] const unsigned char *getData() const {

            return data_;
        }

        ~Image() {

            delete data_;
        }

    private:
        const unsigned char* data_;

    };

}

#endif//THREEPP_IMAGE_HPP
