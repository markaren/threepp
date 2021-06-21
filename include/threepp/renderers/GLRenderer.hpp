// https://github.com/mrdoob/three.js/blob/r129/src/renderers/WebGLRenderer.js

#ifndef THREEPP_GLRENDERER_HPP
#define THREEPP_GLRENDERER_HPP

#include "threepp/cameras/Camera.hpp"
#include "threepp/math/Plane.hpp"
#include "threepp/math/Vector4.hpp"

#include "threepp/Canvas.hpp"
#include "threepp/constants.hpp"

#include "threepp/renderers/gl/GLCapabilities.hpp"
#include "threepp/renderers/gl/GLState.hpp"
#include "threepp/renderers/gl/GLInfo.hpp"

#include <memory>
#include <threepp/math/Frustum.hpp>
#include <vector>

namespace threepp {

    class GLRenderer {

    public:

        struct Parameters {

            bool alpha;
            bool depth;
            bool stencil;
            bool antialias;
            bool premultipliedAlpha;
            bool preserveDrawingBuffer;

        };

        // clearing

        bool autoClear = true;
        bool autoClearColor = true;
        bool autoClearDepth = true;
        bool autoClearStencil = true;

        // scene graph

        bool sortObjects = true;

        // user-defined clipping

        std::vector<Plane> clippingPlanes;
        bool localClippingEnabled = false;

        // physically based shading

        float gammaFactor = 2.0f;	// for backwards compatibility
        int outputEncoding = LinearEncoding;

        // physical lights

        bool physicallyCorrectLights = false;

        // tone mapping

        int toneMapping = NoToneMapping;
        float toneMappingExposure = 1.0f;

        explicit GLRenderer(const Canvas &canvas, const Parameters &parameters);

        void initGLContext();



    private:

        int _currentActiveCubeFace = 0;
        int _currentActiveMipmapLevel = 0;
        int _currentMaterialId = - 1;

        std::shared_ptr<Camera> _currentCamera = nullptr;
        Vector4 _currentViewport;
        Vector4 _currentScissor;
        std::optional<bool> _currentScissorTest;

        //

        int _width;
        int _height;

        int _pixelRatio = 1;

        Vector4 _viewPort;
        Vector4 _scissor;

        // frustum

        Frustum _frustum;

        // clipping

        bool _clippingEnabled = false;
        bool _localClippingEnabled = false;

        // camera matrices cache

        Matrix4 _projScreenMatrix;

        Vector3 _vector3;



        [[nodiscard]] int getTargetPixelRatio() const {
            return _pixelRatio;
        }

    };

}

#endif//THREEPP_GLRENDERER_HPP
