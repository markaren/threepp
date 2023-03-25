//

#ifndef THREEPP_DATATEXTURE3D_HPP
#define THREEPP_DATATEXTURE3D_HPP

#include "threepp/textures/Texture.hpp"

namespace threepp {

    class DataTexture3D: public Texture {

    public:
        bool flipY = false;
        int wrapR = ClampToEdgeWrapping;

        DataTexture3D(std::vector<unsigned char> data, unsigned int width, unsigned int height, unsigned int depth);
    };

}// namespace threepp

#endif//THREEPP_DATATEXTURE3D_HPP
