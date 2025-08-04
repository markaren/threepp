
#ifndef THREEPP_DATATEXTURE_HPP
#define THREEPP_DATATEXTURE_HPP

#include "threepp/textures/Texture.hpp"

namespace threepp {

    class DataTexture: public Texture {

    public:
        void setData(const ImageData& data) {

            image().setData(data);
        }

        template<class T = unsigned char>
        static std::shared_ptr<DataTexture> create(
                int channels,
                unsigned int width, unsigned int height) {

            return std::shared_ptr<DataTexture>(new DataTexture(std::vector<T>(width * height * channels), width, height));
        }

        static std::shared_ptr<DataTexture> create(
                const ImageData& data,
                unsigned int width, unsigned int height) {

            return std::shared_ptr<DataTexture>(new DataTexture(data, width, height));
        }

    private:
        explicit DataTexture(ImageData data, unsigned int width, unsigned int height)
            : Texture({Image(std::move(data), width, height)}) {
            this->magFilter = Filter::Nearest;
            this->minFilter = Filter::Nearest;

            this->generateMipmaps = false;
            this->unpackAlignment = 1;

            this->needsUpdate();
        }
    };

}// namespace threepp

#endif//THREEPP_DATATEXTURE_HPP
