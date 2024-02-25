
#ifndef THREEPP_DATATEXTURE_HPP
#define THREEPP_DATATEXTURE_HPP

#include "threepp/textures/Texture.hpp"

namespace threepp {

    class DataTexture: public Texture {

    public:
        void setData(const ImageData& data) {

            image.front().setData(data);
        }

        static std::shared_ptr<DataTexture> create(
                const ImageData& data,
                unsigned int width = 1, unsigned int height = 1) {
            return std::shared_ptr<DataTexture>(new DataTexture(data, width, height));
        }

    private:
        explicit DataTexture(const ImageData& data, unsigned int width, unsigned int height)
            : Texture({}) {

            this->image.emplace_back(data, width, height);

            this->magFilter = Filter::Nearest;
            this->minFilter = Filter::Nearest;

            this->generateMipmaps = false;
            this->unpackAlignment = 1;

            this->needsUpdate();
        }
    };


}// namespace threepp

#endif//THREEPP_DATATEXTURE_HPP
