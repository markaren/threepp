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

        explicit GLClipping(GLProperties &properties) : properties(properties) {

            uniform.needsUpdate = false;
        }

        bool init(const std::vector<Plane> &planes, bool enableLocalClipping, Camera* camera) {

            bool enabled =
                    !planes.empty() ||
                    enableLocalClipping ||
                    // enable state of previous frame - the clipping code has to
                    // run another frame in order to reset the state:
                    numGlobalPlanes != 0 || localClippingEnabled;

            localClippingEnabled = enableLocalClipping;

            globalState = projectPlanes(planes, camera, 0);
            numGlobalPlanes = (int) planes.size();

            return enabled;
        }

        void beginShadows() {

            renderingShadows = true;
            projectPlanes();
        }

        void endShadows() {

            renderingShadows = false;
            resetGlobalState();
        }

        void setState(Material* material, Camera* camera, bool useCache) {

            auto &planes = material->clippingPlanes;
            auto clipIntersection = material->clipIntersection;
            auto clipShadows = material->clipShadows;

            auto &materialProperties = properties.materialProperties.get(material->uuid);

            if (!localClippingEnabled || planes.empty() || renderingShadows && !clipShadows) {

                // there's no local clipping

                if (renderingShadows) {

                    // there's no global clipping

                    projectPlanes();

                } else {

                    resetGlobalState();
                }

            } else {

                const auto nGlobal = renderingShadows ? 0 : numGlobalPlanes,
                           lGlobal = nGlobal * 4;

                auto &dstArray = materialProperties.clippingState;

                uniform.setValue(dstArray);// ensure unique state

                dstArray = projectPlanes(planes, camera, lGlobal, useCache);

                for (int i = 0; i != lGlobal; ++i) {

                    dstArray[i] = globalState.value()[i];
                }

                materialProperties.clippingState = dstArray;
                this->numIntersection = clipIntersection ? this->numPlanes : 0;
                this->numPlanes += nGlobal;
            }
        }

        void resetGlobalState() {

            if (!uniform.hasValue() || uniform.value<std::vector<float>>() != globalState) {

                uniform.setValue(*globalState);
                uniform.needsUpdate = numGlobalPlanes > 0;
            }

            numPlanes = numGlobalPlanes;
            numIntersection = 0;
        }

        void projectPlanes() {

            numPlanes = 0;
            numIntersection = 0;
        }

        std::vector<float> projectPlanes(
                const std::vector<Plane> &planes, Camera* camera,
                int dstOffset, bool skipTransform = false) {

            int nPlanes = (int) planes.size();
            std::vector<float> dstArray;

            if (nPlanes != 0) {

                if (uniform.hasValue()) {
                    dstArray = uniform.value<std::vector<float>>();
                }

                if (!skipTransform || dstArray.empty()) {

                    const auto flatSize = dstOffset + nPlanes * 4;
                    const auto &viewMatrix = camera->matrixWorldInverse;

                    viewNormalMatrix.getNormalMatrix(viewMatrix);

                    if (dstArray.size() < flatSize) {

                        dstArray.resize(flatSize);
                    }

                    for (int i = 0, i4 = dstOffset; i != nPlanes; ++i, i4 += 4) {

                        plane.copy(planes[i]).applyMatrix4(viewMatrix, viewNormalMatrix);

                        plane.normal.toArray(dstArray, i4);
                        dstArray[i4 + 3] = plane.constant;
                    }
                }

                uniform.setValue(dstArray);
                uniform.needsUpdate = true;
            }

            numPlanes = nPlanes;
            numIntersection = 0;

            return dstArray;
        }

    private:
        GLProperties &properties;
    };

}// namespace threepp::gl

#endif//THREEPP_GLCLIPPING_HPP
