// https://github.com/mrdoob/three.js/blob/r129/src/renderers/WebGLRenderer.js

#ifndef THREEPP_GLRENDERER_HPP
#define THREEPP_GLRENDERER_HPP

#include "threepp/cameras/Camera.hpp"
#include "threepp/math/Plane.hpp"

#include "threepp/constants.hpp"

#include <memory>
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

        explicit GLRenderer(const Parameters &parameters);

        void initGLContext();

    private:

        int _currentActiveCubeFace = 0;
        int _currentActiveMipmapLevel = 0;
        int _currentMaterialId = - 1;

        std::shared_ptr<Camera> _currentCamera = nullptr;


    };

}

#endif//THREEPP_GLRENDERER_HPP
