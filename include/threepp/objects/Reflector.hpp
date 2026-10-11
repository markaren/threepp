// https://github.com/mrdoob/three.js/blob/dev/examples/jsm/objects/Reflector.js

#ifndef THREEPP_REFLECTOR_HPP
#define THREEPP_REFLECTOR_HPP

#include "threepp/core/Shader.hpp"
#include "threepp/objects/Mesh.hpp"

namespace threepp {

    class Camera;

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

        [[nodiscard]] const std::string& type() const override;

        // The camera the mirror image is drawn with when `camera` looks at the
        // mirror: a clone of it, made the first time that camera is seen and kept
        // for its lifetime (three.js' getReflectionCamera). Before each mirror
        // render it is placed as the reflection of `camera` and takes over its
        // projection and far plane; everything else - its layers above all - is
        // the clone's own. So the mirror shows what `camera` could see at the
        // time of the first render, and an object the mirror must not show goes
        // on a layer this camera is then told not to look at.
        [[nodiscard]] Camera& getReflectionCamera(Camera& camera);

        static std::shared_ptr<Reflector> create(const std::shared_ptr<BufferGeometry>& geometry, Options options = Options());

        ~Reflector() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_REFLECTOR_HPP
