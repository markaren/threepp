
#ifndef THREEPP_CUBETEXTURE_HPP
#define THREEPP_CUBETEXTURE_HPP

#include "threepp/textures/Texture.hpp"

namespace threepp {

    class CubeTexture: public Texture {

    public:
        bool _needsFlipEnvMap = true;

        static std::shared_ptr<CubeTexture> create(const std::vector<Image>& images = {}) {

            return std::shared_ptr<CubeTexture>(new CubeTexture(images));
        }

    private:

        explicit CubeTexture(const std::vector<Image>& images): Texture(images) {

            this->mapping = Mapping::CubeReflection;
            this->format = Format::RGB;
        }
    };

}// namespace threepp

#endif//THREEPP_CUBETEXTURE_HPP
