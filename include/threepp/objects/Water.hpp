// https://github.com/mrdoob/three.js/blob/r129/examples/js/objects/Water.js

#ifndef THREEPP_WATER_HPP
#define THREEPP_WATER_HPP

#include <utility>

#include "threepp/objects/Mesh.hpp"

namespace threepp {

    class Water: public Mesh {

    public:
        struct Options {

            std::optional<unsigned int> textureWidth;
            std::optional<unsigned int> textureHeight;
            std::optional<float> clipBias;
            std::optional<float> alpha;
            std::optional<float> time;
            std::shared_ptr<Texture> waterNormals;
            std::optional<Vector3> sunDirection;
            std::optional<Color> sunColor;
            std::optional<Color> waterColor;
            std::optional<Vector3> eye;
            std::optional<float> distortionScale;
            std::optional<int> side;
            std::optional<bool> fog;
        };

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<Water> create(const std::shared_ptr<BufferGeometry>& geometry, Options options = Options());

        ~Water() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        Water(const std::shared_ptr<BufferGeometry>& geometry, Options options);
    };

}// namespace threepp

#endif//THREEPP_WATER_HPP
