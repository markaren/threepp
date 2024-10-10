
#ifndef THREEPP_DATATEXTURE_HPP
#define THREEPP_DATATEXTURE_HPP

#include "threepp/textures/Texture.hpp"

namespace threepp {

    template<class T = unsigned char>
    class DataTexture: public Texture {

    public:
        void setData(const std::vector<T>& data) {

            image().setData(data);
        }

        static std::shared_ptr<DataTexture> create(
                int channels,
                unsigned int width, unsigned int height) {

            return std::shared_ptr<DataTexture>(new DataTexture(width*height*channels, width, height));
        }

        static std::shared_ptr<DataTexture> create(
                const std::vector<T>& data,
                unsigned int width, unsigned int height) {

            return std::shared_ptr<DataTexture>(new DataTexture(data, width, height));
        }

    private:
        explicit DataTexture(std::vector<T> data, unsigned int width, unsigned int height)
            : Texture({Image(std::move(data), width, height)}) {
            this->magFilter = Filter::Nearest;
            this->minFilter = Filter::Nearest;

            this->generateMipmaps = false;
            this->unpackAlignment = 1;

            this->needsUpdate();
        }

        explicit DataTexture(size_t size, unsigned int width, unsigned int height)
            : DataTexture(std::vector<T>(size), width, height) {}
    };

    extern template class DataTexture<unsigned char>;
    extern template class DataTexture<float>;


}// namespace threepp

#endif//THREEPP_DATATEXTURE_HPP
