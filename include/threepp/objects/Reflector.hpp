// https://github.com/mrdoob/three.js/blob/r129/examples/js/objects/Reflector.js

#ifndef THREEPP_REFLECTOR_HPP
#define THREEPP_REFLECTOR_HPP

#include <utility>

#include "threepp/objects/Mesh.hpp"
#include "threepp/core/Shader.hpp"

namespace threepp {

    class Reflector: public Mesh {

    public:

        struct Options {

            std::optional<unsigned int> color;
            std::optional<unsigned int> textureWidth;
            std::optional<unsigned int> textureHeight;
            std::optional<float> clipBias;
            std::optional<Shader> shader;

        };

        static std::shared_ptr<Reflector> create(const std::shared_ptr<BufferGeometry> &geometry, Options options = Options()) {

            return std::shared_ptr<Reflector>(new Reflector(geometry, std::move(options)));
        }

    protected:
        Reflector(const std::shared_ptr<BufferGeometry> &geometry, Options options);

    };

}

#endif//THREEPP_REFLECTOR_HPP
