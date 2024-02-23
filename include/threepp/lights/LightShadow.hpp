// https://github.com/mrdoob/three.js/blob/r129/src/lights/LightShadow.js

#ifndef THREEPP_LIGHTSHADOW_HPP
#define THREEPP_LIGHTSHADOW_HPP

#include "threepp/cameras/Camera.hpp"
#include "threepp/lights/Light.hpp"

#include "threepp/math/Frustum.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector4.hpp"

namespace threepp {

    class GLRenderTarget;

    class LightShadow {

    public:
        std::shared_ptr<Camera> camera;

        float bias = 0;
        float normalBias = 0;
        float radius = 1;

        Vector2 mapSize{1024, 1024};

        std::unique_ptr<GLRenderTarget> map;
        std::unique_ptr<GLRenderTarget> mapPass;

        Matrix4 matrix;

        bool autoUpdate = true;
        bool needsUpdate = false;

        LightShadow(LightShadow&&) = delete;
        LightShadow(const LightShadow&) = delete;
        LightShadow& operator=(const LightShadow&) = delete;

        [[nodiscard]] size_t getViewportCount() const;

        [[nodiscard]] const Frustum& getFrustum() const;

        virtual void updateMatrices(Light* light);

        Vector4& getViewport(size_t viewportIndex);

        Vector2& getFrameExtents();

        void dispose();

        virtual ~LightShadow();

    protected:
        Frustum _frustum;
        Vector2 _frameExtents{1, 1};

        Vector3 _lightPositionWorld;
        Vector3 _lookTarget;
        Matrix4 _projScreenMatrix;

        std::vector<Vector4> _viewports{Vector4(0, 0, 1, 1)};

        explicit LightShadow(std::shared_ptr<Camera> camera);
    };

}// namespace threepp

#endif//THREEPP_LIGHTSHADOW_HPP
