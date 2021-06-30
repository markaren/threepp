// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLProperties.js

#ifndef THREEPP_GLPROPERTIES_HPP
#define THREEPP_GLPROPERTIES_HPP

#include "threepp/scenes/Scene.hpp"

#include <glad/glad.h>

#include <optional>
#include <unordered_map>

namespace threepp::gl {

    struct TextureProperties {

        unsigned int version;

        bool glInit;

        int maxMipLevel;

        GLuint glTexture;
    };

    struct MaterialProperties {

        std::shared_ptr<GLProgram> program;
        std::vector<std::shared_ptr<GLProgram>> programs;

        std::optional<Texture> environment;

        std::optional<FogVariant> fog;

        std::vector<float> clippingState;

        std::optional<Texture> envMap;

        int outputEncoding;
        bool instancing;
        std::optional<int> numClippingPlanes;
        int numIntersection;
        bool vertexAlphas;

        unsigned int version;

        bool needsLights;
        unsigned int lightsStateVersion;
        std::shared_ptr<GLProgram> currentProgram;
        std::unordered_map<std::string, Uniform> &uniforms;
        bool receiveShadow;
    };

    template<class T>
    struct GLTypeProperties {

        T &get(const std::string &key) {

            return properties_.at(key);
        }

        void remove(const std::string &key) {

            properties_.erase(key);
        }

        void dispose() {

            properties_.clear();
        }

    private:
        std::unordered_map<std::string, T> properties_;
    };

    struct GLProperties {

        GLTypeProperties<TextureProperties> textureProperties;
        GLTypeProperties<MaterialProperties> materialProperties;

        void dispose() {

            textureProperties.dispose();
            materialProperties.dispose();
        }
    };

}// namespace threepp::gl

#endif//THREEPP_GLPROPERTIES_HPP
