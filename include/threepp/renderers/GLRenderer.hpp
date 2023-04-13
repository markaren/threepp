// https://github.com/mrdoob/three.js/blob/r129/src/renderers/WebGLRenderer.js

#ifndef THREEPP_GLRENDERER_HPP
#define THREEPP_GLRENDERER_HPP

#include "threepp/math/Color.hpp"
#include "threepp/math/Plane.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector4.hpp"

#include "threepp/Canvas.hpp"
#include "threepp/constants.hpp"
#include "threepp/core/misc.hpp"

#include "threepp/renderers/gl/GLInfo.hpp"
#include "threepp/renderers/gl/GLShadowMap.hpp"
#include "threepp/renderers/gl/GLState.hpp"

#include "TextHandle.hpp"

#include <memory>
#include <vector>

namespace threepp {

    class Camera;
    class Scene;
    class BufferGeometry;
    class Object3D;
    class Material;
    class Texture;
    class GLRenderTarget;

    class GLRenderer {

    public:
        struct Parameters {

            bool alpha;
            bool depth;
            bool premultipliedAlpha;
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

        bool checkShaderErrors = false;

        explicit GLRenderer(Canvas& canvas, const Parameters& parameters = {});

        const gl::GLInfo& info();

        gl::GLShadowMap& shadowMap();

        [[nodiscard]] const gl::GLShadowMap& shadowMap() const;

        gl::GLState& state();

        [[nodiscard]] int getTargetPixelRatio() const;

        void setPixelRatio(int value);

        [[nodiscard]] WindowSize getSize() const;

        void setSize(WindowSize size);

        void getDrawingBufferSize(Vector2& target) const;

        void setDrawingBufferSize(int width, int height, int pixelRatio);

        void getCurrentViewport(Vector4& target) const;

        void getViewport(Vector4& target) const;

        void setViewport(const Vector4& v);

        void setViewport(int x, int y, int width, int height);

        void getScissor(Vector4& target);

        void setScissor(const Vector4& v);

        void setScissor(int x, int y, int width, int height);

        [[nodiscard]] bool getScissorTest() const;

        void setScissorTest(bool boolean);

        // Clearing

        void getClearColor(Color& target) const;

        void setClearColor(const Color& color, float alpha = 1);

        [[nodiscard]] float getClearAlpha() const;

        void setClearAlpha(float clearAlpha);

        void clear(bool color = true, bool depth = true, bool stencil = true);

        void clearColor();
        void clearDepth();
        void clearStencil();

        void dispose();

        void render(Scene* scene, Camera* camera);

        void render(const std::shared_ptr<Scene>& scene, const std::shared_ptr<Camera>& camera);

        void renderBufferDirect(Camera* camera, Scene* scene, BufferGeometry* geometry, Material* material, Object3D* object, std::optional<GeometryGroup> group);

        [[nodiscard]] int getActiveCubeFace() const;

        [[nodiscard]] int getActiveMipmapLevel() const;

        std::shared_ptr<GLRenderTarget>& getRenderTarget();

        void setRenderTarget(const std::shared_ptr<GLRenderTarget>& renderTarget, int activeCubeFace = 0, int activeMipmapLevel = 0);

        void copyFramebufferToTexture(const Vector2& position, Texture& texture, int level = 0);

        void enableTextRendering();

        TextHandle& textHandle(const std::string& str = "");

        [[nodiscard]] const gl::GLInfo& info() const;

        ~GLRenderer();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_GLRENDERER_HPP
