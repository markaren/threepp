// https://github.com/mrdoob/three.js/blob/r129/examples/js/objects/Reflector.js

#ifndef THREEPP_REFLECTOR_HPP
#define THREEPP_REFLECTOR_HPP

#include "threepp/core/Shader.hpp"
#include "threepp/objects/Mesh.hpp"

namespace threepp {

    class Reflector: public Mesh {

    public:
        struct Options {

            std::optional<Color> color;
            std::optional<unsigned int> textureWidth;
            std::optional<unsigned int> textureHeight;
            std::optional<float> clipBias;
            std::optional<Shader> shader;
        };

        Reflector(const std::shared_ptr<BufferGeometry>& geometry, Options options);

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<Reflector> create(const std::shared_ptr<BufferGeometry>& geometry, Options options = Options());

        ~Reflector() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_REFLECTOR_HPP
