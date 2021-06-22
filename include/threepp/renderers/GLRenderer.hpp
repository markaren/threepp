// https://github.com/mrdoob/three.js/blob/r129/src/renderers/WebGLRenderer.js

#ifndef THREEPP_GLRENDERER_HPP
#define THREEPP_GLRENDERER_HPP

#include "threepp/cameras/Camera.hpp"
#include "threepp/math/Plane.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector4.hpp"

#include "threepp/Canvas.hpp"
#include "threepp/constants.hpp"

#include "threepp/renderers/gl/GLCapabilities.hpp"
#include "threepp/renderers/gl/GLInfo.hpp"
#include "threepp/renderers/gl/GLState.hpp"

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

        float gammaFactor = 2.0f;// for backwards compatibility
        int outputEncoding = LinearEncoding;

        // physical lights

        bool physicallyCorrectLights = false;

        // tone mapping

        int toneMapping = NoToneMapping;
        float toneMappingExposure = 1.0f;

        explicit GLRenderer(Canvas &canvas, const Parameters &parameters = Parameters());

        void initGLContext();

        [[nodiscard]] int getTargetPixelRatio() const;

        void getSize(Vector2 &target) const;

        void setSize(int width, int height);

        void getDrawingBufferSize(Vector2 &target) const;

        void setDrawingBufferSize(int width, int height, int pixelRatio);

        void getCurrentViewport(Vector4 &target) const;

        void getViewport(Vector4 &target) const;

        void setViewport(const Vector4 &v);

        void setViewport(int x, int y, int width, int height);

        void getScissor(Vector4 &target);

        void setScissor(const Vector4 &v);

        void setScissor(int x, int y, int width, int height);

        [[nodiscard]] bool getScissorTest() const;

        void setScissorTest(bool boolean);


    private:
        Canvas &canvas_;

        int _currentActiveCubeFace = 0;
        int _currentActiveMipmapLevel = 0;
        int _currentMaterialId = -1;

        std::shared_ptr<Camera> _currentCamera = nullptr;
        Vector4 _currentViewport;
        Vector4 _currentScissor;
        std::optional<bool> _currentScissorTest;

        //

        int _width;
        int _height;

        int _pixelRatio = 1;
        //std::function<bool(int, int)> _opaqueSort;

        Vector4 _viewport;
        Vector4 _scissor;
        bool _scissorTest = false;

        // frustum

        Frustum _frustum;

        // clipping

        bool _clippingEnabled = false;
        bool _localClippingEnabled = false;

        // camera matrices cache

        Matrix4 _projScreenMatrix;

        Vector3 _vector3;

        gl::GLCapabilities capabilities;
        gl::GLState state;
        gl::GLInfo info;
    };

}// namespace threepp

#endif//THREEPP_GLRENDERER_HPP
