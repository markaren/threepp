
#ifndef THREEPP_CUBETEXTURE_HPP
#define THREEPP_CUBETEXTURE_HPP

#include "threepp/textures/Texture.hpp"

namespace threepp {

    class CubeTexture: public Texture {

    public:
        std::shared_ptr<CubeTexture> create(const std::array<Image, 6>& images) {
            return std::shared_ptr<CubeTexture>(new CubeTexture(images));
        }

        const std::array<Image, 6>& getImages() const {

            return images_;
        }

    private:
        std::array<Image, 6> images_;

        explicit CubeTexture(const std::array<Image, 6>& images): images_(images) {}
    };

}// namespace threepp

#endif//THREEPP_CUBETEXTURE_HPP
