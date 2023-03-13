// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLClipping.js

#ifndef THREEPP_GLCLIPPING_HPP
#define THREEPP_GLCLIPPING_HPP

#include "GLProperties.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/math/Plane.hpp"

#include "threepp/core/Uniform.hpp"

namespace threepp::gl {

    struct GLClipping {

        std::optional<std::vector<float>> globalState;

        int numGlobalPlanes = 0;
        bool localClippingEnabled = false;
        bool renderingShadows = false;

        Plane plane;
        Matrix3 viewNormalMatrix;

        Uniform uniform;

        int numPlanes = 0;
        int numIntersection = 0;

        explicit GLClipping(GLProperties& properties);

        bool init(const std::vector<Plane>& planes, bool enableLocalClipping, Camera* camera);

        void beginShadows();

        void endShadows();

        void setState(Material* material, Camera* camera, bool useCache);

        void resetGlobalState();

        void projectPlanes();

        std::vector<float> projectPlanes(
                const std::vector<Plane>& planes, Camera* camera,
                int dstOffset, bool skipTransform = false);

    private:
        GLProperties& properties;
    };

}// namespace threepp::gl

#endif//THREEPP_GLCLIPPING_HPP
