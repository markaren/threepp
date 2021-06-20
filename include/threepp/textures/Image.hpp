
#ifndef THREEPP_IMAGE_HPP
#define THREEPP_IMAGE_HPP

namespace threepp {

    class Image {

    public:

        Image(unsigned int width, unsigned int height): width_(width), height_(height){};

        [[nodiscard]] unsigned int width() const {
            return width_;
        }

        [[nodiscard]] unsigned int height() const {
            return height_;
        }

    private:
        unsigned int width_;
        unsigned int height_;

    };

}

#endif//THREEPP_IMAGE_HPP
