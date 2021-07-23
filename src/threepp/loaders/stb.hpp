
#ifndef THREEPP_STB_HPP
#define THREEPP_STB_HPP

#include <string>

namespace threepp {

    struct ImageStruct {

        /*! The width, in pixels, of this image.
        */
        int width;
        /*! The height, in pixels, of this image.
         */
        int height;
        /*! The pixel data of this image, arranged left-to-right, top-to-bottom.
         */
        unsigned char *pixels;
    };

    ImageStruct stb_load(const std::string &path, int channels);

}// namespace

#endif//THREEPP_STB_HPP
