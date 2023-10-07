
#include "threepp/renderers/gl/GLClipping.hpp"

#include "threepp/renderers/gl/GLProperties.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/materials/Material.hpp"

using namespace threepp;
using namespace threepp::gl;


struct GLClipping::Impl {

    GLClipping& scope;
    GLProperties& properties;

    Impl(GLClipping& scope, GLProperties& properties): scope(scope), properties(properties) {}


    bool init(const std::vector<Plane>& planes, bool enableLocalClipping, Camera* camera) {

        bool enabled =
                !planes.empty() ||
                enableLocalClipping ||
                // enable state of previous frame - the clipping code has to
                // run another frame in order to reset the state:
                scope.numGlobalPlanes != 0 || scope.localClippingEnabled;

        scope.localClippingEnabled = enableLocalClipping;

        scope.globalState = scope.projectPlanes(planes, camera, 0);
        scope.numGlobalPlanes = planes.size();

        return enabled;
    }

    void beginShadows() {

        scope.renderingShadows = true;
        projectPlanes();
    }

    void endShadows() {

        scope.renderingShadows = false;
        resetGlobalState();
    }

    void setState(Material* material, Camera* camera, bool useCache) {

        auto& planes = material->clippingPlanes;
        auto clipIntersection = material->clipIntersection;
        auto clipShadows = material->clipShadows;

        auto materialProperties = properties.materialProperties.get(material->uuid());

        if (!scope.localClippingEnabled || planes.empty() || scope.renderingShadows && !clipShadows) {

            // there's no local clipping

            if (scope.renderingShadows) {

                // there's no global clipping

                projectPlanes();

            } else {

                resetGlobalState();
            }

        } else {

            const auto nGlobal = scope.renderingShadows ? 0 : scope.numGlobalPlanes,
                       lGlobal = nGlobal * 4;

            auto& dstArray = materialProperties->clippingState;

            scope.uniform.setValue(dstArray);// ensure unique state

            dstArray = projectPlanes(planes, camera, lGlobal, useCache);

            for (unsigned i = 0; i != lGlobal; ++i) {

                dstArray[i] = scope.globalState.value()[i];
            }

            materialProperties->clippingState = dstArray;
            scope.numIntersection = clipIntersection ? scope.numPlanes : 0;
            scope.numPlanes += nGlobal;
        }
    }

    void resetGlobalState() {

        if (!scope.uniform.hasValue() || scope.uniform.value<std::vector<float>>() != scope.globalState) {

            scope.uniform.setValue(*scope.globalState);
            scope.uniform.needsUpdate = scope.numGlobalPlanes > 0;
        }

        scope.numPlanes = scope.numGlobalPlanes;
        scope.numIntersection = 0;
    }

    void projectPlanes() {

        scope.numPlanes = 0;
        scope.numIntersection = 0;
    }

    std::vector<float> projectPlanes(const std::vector<Plane>& planes, Camera* camera, size_t dstOffset, bool skipTransform) {

        const auto nPlanes = planes.size();
        std::vector<float> dstArray;

        if (nPlanes != 0) {

            if (scope.uniform.hasValue()) {
                dstArray = scope.uniform.value<std::vector<float>>();
            }

            if (!skipTransform || dstArray.empty()) {

                const auto flatSize = dstOffset + nPlanes * 4;
                const auto& viewMatrix = camera->matrixWorldInverse;

                scope.viewNormalMatrix.getNormalMatrix(viewMatrix);

                if (dstArray.size() < flatSize) {

                    dstArray.resize(flatSize);
                }

                for (unsigned i = 0, i4 = dstOffset; i != nPlanes; ++i, i4 += 4) {

                    scope.plane.copy(planes[i]).applyMatrix4(viewMatrix, scope.viewNormalMatrix);

                    scope.plane.normal.toArray(dstArray, i4);
                    dstArray[i4 + 3] = scope.plane.constant;
                }
            }

            scope.uniform.setValue(dstArray);
            scope.uniform.needsUpdate = true;
        }

        scope.numPlanes = nPlanes;
        scope.numIntersection = 0;

        return dstArray;
    }
};

GLClipping::GLClipping(GLProperties& properties)
    : pimpl_(std::make_unique<Impl>(*this, properties)) {

    uniform.needsUpdate = false;
}

bool GLClipping::init(const std::vector<Plane>& planes, bool enableLocalClipping, Camera* camera) {

    return pimpl_->init(planes, enableLocalClipping, camera);
}

void GLClipping::beginShadows() {

    pimpl_->beginShadows();
}

void GLClipping::endShadows() {

    pimpl_->endShadows();
}

void GLClipping::setState(Material* material, Camera* camera, bool useCache) {

    pimpl_->setState(material, camera, useCache);
}

void GLClipping::resetGlobalState() {

    pimpl_->resetGlobalState();
}

void GLClipping::projectPlanes() {

    numPlanes = 0;
    numIntersection = 0;
}

std::vector<float> GLClipping::projectPlanes(const std::vector<Plane>& planes, Camera* camera, int dstOffset, bool skipTransform) {

    return pimpl_->projectPlanes(planes, camera, dstOffset, skipTransform);
}

gl::GLClipping::~GLClipping() = default;
