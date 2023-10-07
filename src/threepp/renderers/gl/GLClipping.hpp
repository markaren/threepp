// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLClipping.js

#ifndef THREEPP_GLCLIPPING_HPP
#define THREEPP_GLCLIPPING_HPP

#include "threepp/math/Plane.hpp"

#include "threepp/core/Uniform.hpp"

#include <optional>

namespace threepp {

    class Camera;
    class Material;

    namespace gl {

        class GLProperties;

        struct GLClipping {

            std::optional<std::vector<float>> globalState;

            size_t numGlobalPlanes = 0;
            bool localClippingEnabled = false;
            bool renderingShadows = false;

            Plane plane;
            Matrix3 viewNormalMatrix;

            Uniform uniform;

            size_t numPlanes = 0;
            size_t numIntersection = 0;

            explicit GLClipping(GLProperties& properties);

            bool init(const std::vector<Plane>& planes, bool enableLocalClipping, Camera* camera);

            void beginShadows();

            void endShadows();

            void setState(Material* material, Camera* camera, bool useCache);

            void resetGlobalState();

            void projectPlanes();

            std::vector<float> projectPlanes(const std::vector<Plane>& planes, Camera* camera, int dstOffset, bool skipTransform = false);

            ~GLClipping();

        private:
            struct Impl;
            std::unique_ptr<Impl> pimpl_;
        };

    }// namespace gl

}// namespace threepp

#endif//THREEPP_GLCLIPPING_HPP
