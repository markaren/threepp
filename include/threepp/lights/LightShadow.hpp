// https://github.com/mrdoob/three.js/blob/r129/src/lights/LightShadow.js

#ifndef THREEPP_LIGHTSHADOW_HPP
#define THREEPP_LIGHTSHADOW_HPP

#include "threepp/lights/Light.hpp"
#include "threepp/cameras/Camera.hpp"

#include "threepp/math/Frustum.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector4.hpp"

#include "threepp/textures/Texture.hpp"

namespace threepp {

    class LightShadow {

    public:
        std::shared_ptr<Camera> camera;

        float bias = 0;
        float normalBias = 0;
        float radius = 1;

        Vector2 mapSize = Vector2(512, 512);

        std::optional<Texture> map;
        Matrix4 matrix;

        bool autoUpdate = true;
        bool needsUpdate = false;

        [[nodiscard]] int getViewportCount() const {

            return this->_viewportCount;
        }

        Frustum &getFrustum() {

            return this->_frustum;
        }

        void updateMatrices(const Light &light, const Object3D &target) {

            const auto &shadowCamera = this->camera;
            auto &shadowMatrix = this->matrix;

            Vector3 _lightPositionWorld{};
            _lightPositionWorld.setFromMatrixPosition(light.matrixWorld);
            shadowCamera->position.copy(_lightPositionWorld);

            Vector3 _lookTarget{};
            _lookTarget.setFromMatrixPosition(target.matrixWorld);
            shadowCamera->lookAt(_lookTarget);
            shadowCamera->updateMatrixWorld();

            Matrix4 _projScreenMatrix{};
            _projScreenMatrix.multiplyMatrices(shadowCamera->projectionMatrix, shadowCamera->matrixWorldInverse);
            this->_frustum.setFromProjectionMatrix(_projScreenMatrix);

            shadowMatrix.set(
                    0.5f, 0.0f, 0.0f, 0.5f,
                    0.0f, 0.5f, 0.0f, 0.5f,
                    0.0f, 0.0f, 0.5f, 0.5f,
                    0.0f, 0.0f, 0.0f, 1.0f);

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

            if (this->map) {

                this->map->dispose();
            }
            // TODO
            //            if (this.mapPass) {
            //
            //                this.mapPass.dispose();
            //            }
        }

        virtual ~LightShadow() = default;

    protected:
        explicit LightShadow(std::shared_ptr<Camera> camera)
            : camera(std::move(camera)) {}

    protected:

        Frustum _frustum;
        Vector2 _frameExtents;

        int _viewportCount = 1;

        std::vector<Vector4> _viewports{Vector4(0, 0, 1, 1)};

    };

}// namespace threepp

#endif//THREEPP_LIGHTSHADOW_HPP
