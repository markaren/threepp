//

#ifndef THREEPP_DATATEXTURE3D_HPP
#define THREEPP_DATATEXTURE3D_HPP

#include "threepp/textures/Texture.hpp"

namespace threepp {

    class DataTexture3D: public Texture {

    public:
        int wrapR = ClampToEdgeWrapping;

        static std::shared_ptr<DataTexture3D> create(const std::vector<unsigned char>& data, unsigned int width = 1, unsigned int height = 1, unsigned int depth = 1);

    private:
        DataTexture3D(const std::vector<unsigned char>& data, unsigned int width, unsigned int height, unsigned int depth);
    };

}// namespace threepp

#endif//THREEPP_DATATEXTURE3D_HPP
