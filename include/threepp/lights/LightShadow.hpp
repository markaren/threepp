// https://github.com/mrdoob/three.js/blob/r129/src/lights/LightShadow.js

#ifndef THREEPP_LIGHTSHADOW_HPP
#define THREEPP_LIGHTSHADOW_HPP

#include <utility>

#include "threepp/cameras/Camera.hpp"

#include "threepp/math/Frustum.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector4.hpp"

namespace threepp {

    class LightShadow {

    public:
        float bias = 0;
        float normalBias = 0;
        float radius = 1;

        Vector2 mapSize = Vector2(512, 512);

        Matrix4 matrix;

        bool autoUpdate = true;
        bool needsUpdate = false;

        LightShadow(const LightShadow &) = delete;

        [[nodiscard]] int getViewportCount() const {

            return this->_viewportCount;
        }

        Frustum &getFrustum() {

            return this->_frustum;
        }

        void updateMatrices(const Light &light, const Object3D &target) {

            const auto &shadowCamera = this->camera_;
            auto &shadowMatrix = this->matrix;

            _lightPositionWorld.setFromMatrixPosition(light.matrixWorld);
            shadowCamera->position.copy(_lightPositionWorld);

            _lookTarget.setFromMatrixPosition(target.matrixWorld);
            shadowCamera->lookAt(_lookTarget);
            shadowCamera->updateMatrixWorld();

            _projScreenMatrix.multiplyMatrices(shadowCamera->projectionMatrix, shadowCamera->matrixWorldInverse);
            this->_frustum.setFromProjectionMatrix(_projScreenMatrix);

            shadowMatrix.set(
                    0.5, 0.0, 0.0, 0.5,
                    0.0, 0.5, 0.0, 0.5,
                    0.0, 0.0, 0.5, 0.5,
                    0.0, 0.0, 0.0, 1.0);

            shadowMatrix.multiply(shadowCamera->projectionMatrix);
            shadowMatrix.multiply(shadowCamera->matrixWorldInverse);
        }

        Vector4 &getViewport(int viewportIndex) {

            return this->_viewports[viewportIndex];
        }

        Vector2 &getFrameExtents() {

            return this->_frameExtents;
        }

        void dispose() {

            //            if ( this.map ) {
            //
            //                this.map.dispose();
            //
            //            }
            //
            //            if ( this.mapPass ) {
            //
            //                this.mapPass.dispose();
            //
            //            }
        }


    protected:
        explicit LightShadow(std::shared_ptr<Camera> camera) : camera_(std::move(camera)) {}


    private:
        std::shared_ptr<Camera> camera_;

        Frustum _frustum;
        Vector2 _frameExtents;

        int _viewportCount = 1;

        std::vector<Vector4> _viewports{Vector4(0, 0, 1, 1)};

        static Matrix4 _projScreenMatrix;
        static Vector3 _lightPositionWorld;
        static Vector3 _lookTarget;
    };

}// namespace threepp

#endif//THREEPP_LIGHTSHADOW_HPP
