// https://github.com/mrdoob/three.js/blob/r129/src/renderers/WebGLRenderer.js

#ifndef THREEPP_GLRENDERER_HPP
#define THREEPP_GLRENDERER_HPP

#include "threepp/renderers/Renderer.hpp"

#include "threepp/renderers/gl/GLInfo.hpp"
#include "threepp/renderers/gl/GLShadowMap.hpp"
#include "threepp/renderers/gl/GLState.hpp"

namespace threepp {

    class Camera;
    class Scene;
    class BufferGeometry;
    class Object3D;
    class Material;
    class Texture;
    class RenderTarget;
    class BufferAttribute;

    class GLRenderer : public Renderer {

    public:
        struct Parameters {

            // bool alpha;
            // bool depth;
            bool premultipliedAlpha;
        };

        explicit GLRenderer(std::pair<int, int> size = {}, const Parameters& parameters = {});

        GLRenderer(GLRenderer&&) = delete;
        GLRenderer(const GLRenderer&) = delete;
        GLRenderer& operator=(const GLRenderer&) = delete;
        GLRenderer& operator=(GLRenderer&&) = delete;

        // --- GL-specific accessors (not on Renderer base) ---

        const gl::GLInfo& info();

        gl::GLShadowMap& shadowMap();

        [[nodiscard]] const gl::GLShadowMap& shadowMap() const;

        gl::GLState& state();

        [[nodiscard]] std::optional<unsigned int> getGlTextureId(Texture& texture) const;

        [[nodiscard]] std::optional<unsigned int> getGlBufferId(BufferAttribute& bufferAttribute) const;

        // --- Renderer interface overrides ---

        [[nodiscard]] float getTargetPixelRatio() const override;

        void setPixelRatio(float value) override;

        [[nodiscard]] WindowSize size() const override;

        void setSize(const std::pair<int, int>& size) override;

        void setViewport(const Vector4& v) override;

        void setViewport(int x, int y, int width, int height) override;

        void setScissor(const Vector4& v) override;

        void setScissor(int x, int y, int width, int height) override;

        void setScissorTest(bool boolean) override;

        void setClearColor(const Color& color, float alpha = 1) override;

        void clear(bool color = true, bool depth = true, bool stencil = true) override;

        void render(Object3D& scene, Camera& camera) override;

        RenderTarget* getRenderTarget() override;

        void setRenderTarget(RenderTarget* renderTarget, int activeCubeFace = 0, int activeMipmapLevel = 0) override;

        [[nodiscard]] std::vector<unsigned char> readRGBPixels() override;

        void dispose() override;

        // --- Additional GLRenderer-specific methods ---

        void getDrawingBufferSize(Vector2& target) const;

        void setDrawingBufferSize(const std::pair<int, int>& size, int pixelRatio);

        void getCurrentViewport(Vector4& target) const;

        void getViewport(Vector4& target) const;

        void setViewport(const std::pair<int, int>& pos, const std::pair<int ,int>& size);

        void getScissor(Vector4& target);

        void setScissor(const std::pair<int, int>& pos, const std::pair<int, int>& size);

        [[nodiscard]] bool getScissorTest() const;

        void getClearColor(Color& target) const;

        [[nodiscard]] float getClearAlpha() const;

        void setClearAlpha(float clearAlpha);

        void clearColor();
        void clearDepth();
        void clearStencil();

        void renderBufferDirect(Camera* camera, Scene* scene, BufferGeometry* geometry, Material* material, Object3D* object, std::optional<GeometryGroup> group);

        [[nodiscard]] int getActiveCubeFace() const;

        [[nodiscard]] int getActiveMipmapLevel() const;

        void copyFramebufferToTexture(const Vector2& position, Texture& texture, int level = 0);

        void readPixels(const Vector2& position, const std::pair<int, int>& size, Format format, unsigned char* data);

        // Experimental threepp function
        void copyTextureToImage(Texture& texture);

        void resetState();

        [[nodiscard]] const gl::GLInfo& info() const;

        void writeFramebuffer(const std::filesystem::path& filename);

        ~GLRenderer() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_GLRENDERER_HPP
