// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLProperties.js

#ifndef THREEPP_GLPROPERTIES_HPP
#define THREEPP_GLPROPERTIES_HPP

#include <unordered_map>

namespace threepp::gl {

    struct GLTextureProperties {

        struct Properties {

            unsigned int version;

            bool glInit;

            int maxMipLevel;

            GLuint glTexture;


        };

        Properties &get(const std::string &key) {

            return properties_.at(key);
        }

        void remove(const std::string &key) {

            properties_.erase(key);
        }

    private:
        std::unordered_map<std::string, Properties> properties_;

    };

    struct GLProperties {

        GLTextureProperties textureProperties;

    };

}

#endif//THREEPP_GLPROPERTIES_HPP
