
#ifndef THREEPP_DATATEXTURE_HPP
#define THREEPP_DATATEXTURE_HPP

#include "threepp/textures/Texture.hpp"

namespace threepp {

    class DataTexture: public Texture {

    public:
        static std::shared_ptr<DataTexture> create(const std::vector<unsigned char>& data, unsigned int width = 1, unsigned int height = 1);

    private:
        explicit DataTexture(const std::vector<unsigned char>& data, unsigned int width, unsigned int height);
    };

}// namespace threepp

#endif//THREEPP_DATATEXTURE_HPP
