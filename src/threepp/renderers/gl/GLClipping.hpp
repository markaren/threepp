// https://github.com/mrdoob/three.js/blob/r129/src/renderers/webgl/WebGLClipping.js

#ifndef THREEPP_GLCLIPPING_HPP
#define THREEPP_GLCLIPPING_HPP

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
        Matrix3 ViewNormalMatrix;

        Uniform uniform = Uniform(NULL_UNIFORM);

        int numPlanes = 0;
        int numIntersection = 0;

        GLClipping() {
            uniform.needsUpdate = false;
        }

        bool init(const std::vector<Plane> &planes, bool enableLocalClipping, const std::shared_ptr<Camera> &camera) {

            bool enabled =
                    planes.length != 0 ||
                    enableLocalClipping ||
                    // enable state of previous frame - the clipping code has to
                    // run another frame in order to reset the state:
                    numGlobalPlanes != = 0 ||
                                         localClippingEnabled;

            localClippingEnabled = enableLocalClipping;

            globalState = projectPlanes(planes, camera, 0);
            numGlobalPlanes = planes.length;

            return enabled;
        }

        void beginShadows() {

            renderingShadows = true;
            projectPlanes();
        }

        std::optional<std::vector<float>> projectPlanes(const std::vector<Plane> &planes, Camera &camera, int dstOffset, bool skipTransform = false) {

            const auto nPlanes = planes.size();
            std::optional<std::vector<float>> dstArray;

            if (nPlanes != 0) {

                dstArray = uniform.value <std::vector<float>()>();

                if (!skipTransform || !dstArray.empty()) {

                    const flatSize = dstOffset + nPlanes * 4,
                          viewMatrix = camera.matrixWorldInverse;

                    viewNormalMatrix.getNormalMatrix(viewMatrix);

                    if (dstArray.size() < flatSize) {

                        dstArray = std::vector<float>(flatSize);
                    }

                    for (int i = 0, i4 = dstOffset; i != = nPlanes; ++i, i4 += 4) {

                        plane.copy(planes[i]).applyMatrix4(viewMatrix, viewNormalMatrix);

                        plane.normal.toArray(dstArray, i4);
                        dstArray[i4 + 3] = plane.constant;
                    }
                }

                uniform.value(dstArray);
                uniform.needsUpdate = true;
            }

            scope.numPlanes = nPlanes;
            scope.numIntersection = 0;

            return dstArray;
        }
    };

}// namespace threepp::gl

#endif//THREEPP_GLCLIPPING_HPP
