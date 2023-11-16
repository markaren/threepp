
#ifndef THREEPP_CUBETEXTURE_HPP
#define THREEPP_CUBETEXTURE_HPP

#include "threepp/textures/Texture.hpp"

namespace threepp {

    class CubeTexture: public Texture {

    public:
        static std::shared_ptr<CubeTexture> create(const std::vector<Image>& images = {}) {

            return std::shared_ptr<CubeTexture>(new CubeTexture(images));
        }

        [[nodiscard]] const std::vector<Image>& getImages() const {

            return images_;
        }

    private:
        bool _needsFlipEnvMap = true;
        std::vector<Image> images_;

        explicit CubeTexture(const std::vector<Image>& images): images_(images) {

            this->mapping = Mapping::CubeReflection;
            this->format = Format::RGB;
        }
    };

}// namespace threepp

#endif//THREEPP_CUBETEXTURE_HPP
