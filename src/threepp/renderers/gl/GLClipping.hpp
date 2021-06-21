// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLClipping.js

#ifndef THREEPP_GLCLIPPING_HPP
#define THREEPP_GLCLIPPING_HPP

#include "threepp/cameras/Camera.hpp"
#include "threepp/math/Plane.hpp"

#include "threepp/core/Uniform.hpp"

namespace threepp::gl {

    struct GLClipping {

        std::optional<std::vector<float>> globalState;

        size_t numGlobalPlanes = 0;
        bool localClippingEnabled = false;
        bool renderingShadows = false;

        Plane plane;
        Matrix3 viewNormalMatrix;

        Uniform uniform;

        size_t numPlanes = 0;
        unsigned int numIntersection = 0;

        GLClipping() {
            uniform.needsUpdate = false;
        }

        bool init(const std::vector<Plane> &planes, bool enableLocalClipping, Camera &camera) {

            bool enabled =
                    !planes.empty() ||
                    enableLocalClipping ||
                    // enable state of previous frame - the clipping code has to
                    // run another frame in order to reset the state:
                    numGlobalPlanes != 0 || localClippingEnabled;

            localClippingEnabled = enableLocalClipping;

            globalState = projectPlanes(planes, camera, 0);
            numGlobalPlanes = planes.size();

            return enabled;
        }

        void beginShadows() {

            renderingShadows = true;
            projectPlanes();
        }

        void resetGlobalState() {

            if (!uniform.hasValue() || uniform.value<std::vector<float>>() != globalState) {

                uniform.setValue(globalState);
                uniform.needsUpdate = numGlobalPlanes > 0;
            }

            numPlanes = numGlobalPlanes;
            numIntersection = 0;
        }

        void projectPlanes() {

            numPlanes = 0;
            numIntersection = 0;
        }

        std::optional<std::vector<float>> projectPlanes(const std::vector<Plane> &planes, const Camera &camera, int dstOffset, bool skipTransform = false) {

            const auto nPlanes = planes.size();
            std::vector<float> dstArray;

            dstArray = uniform.value<std::vector<float>>();

            if (!skipTransform || !dstArray.empty()) {

                const auto flatSize = dstOffset + nPlanes * 4;
                const auto viewMatrix = camera.matrixWorldInverse;

                viewNormalMatrix.getNormalMatrix(viewMatrix);

                if (dstArray.size() < flatSize) {

                    dstArray = std::vector<float>(flatSize);
                }

                for (int i = 0, i4 = dstOffset; i != nPlanes; ++i, i4 += 4) {

                    plane.copy(planes[i]).applyMatrix4(viewMatrix, viewNormalMatrix);

                    plane.normal.toArray(dstArray, i4);
                    dstArray[i4 + 3] = plane.constant;
                }
            }

            uniform.setValue(dstArray);
            uniform.needsUpdate = true;


            numPlanes = nPlanes;
            numIntersection = 0;

            return dstArray;
        }
    };

}// namespace threepp::gl

#endif//THREEPP_GLCLIPPING_HPP
