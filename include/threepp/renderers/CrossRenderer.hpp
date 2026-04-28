
#ifndef THREEPP_CROSSRENDERER_HPP
#define THREEPP_CROSSRENDERER_HPP

#include "threepp/renderers/Renderer.hpp"

#include <memory>

namespace threepp {

    class Canvas;
    class GLRenderer;
    class WgpuRenderer;

    // Side-by-side comparison renderer.
    //
    // The visible canvas is initialised as an OpenGL window. render() draws
    // the scene twice: the OpenGL backend fills the left half of the window,
    // and a headless WebGPU backend renders to an off-screen target which is
    // read back and presented as a fullscreen-quad blit on the right half.
    //
    // Mirrors the pattern proven in examples/wgpu/water_side_by_side.cpp.
    // Note: the camera's aspect should match a single half of the window
    // (width/2 over height), since both halves share the same camera.
    class CrossRenderer: public Renderer {

    public:
        explicit CrossRenderer(Canvas& canvas);

        void render(Object3D& scene, Camera& camera) override;

        [[nodiscard]] WindowSize size() const override;
        void setSize(const std::pair<int, int>& size) override;

        [[nodiscard]] float getTargetPixelRatio() const override;
        void setPixelRatio(float value) override;

        void setViewport(const Vector4& v) override;
        void setViewport(int x, int y, int width, int height) override;

        void setScissor(const Vector4& v) override;
        void setScissor(int x, int y, int width, int height) override;
        void setScissorTest(bool boolean) override;

        ShadowMapConfig& shadowMap() override;
        [[nodiscard]] const ShadowMapConfig& shadowMap() const override;

        void setClearColor(const Color& color, float alpha = 1) override;
        void getClearColor(Color& target) const override;
        void setClearAlpha(float alpha) override;
        [[nodiscard]] float getClearAlpha() const override;

        void clear(bool color = true, bool depth = true, bool stencil = true) override;

        RenderTarget* getRenderTarget() override;
        void setRenderTarget(RenderTarget* renderTarget, int activeCubeFace = 0, int activeMipmapLevel = 0) override;

        [[nodiscard]] std::vector<unsigned char> readRGBPixels() override;

        [[nodiscard]] bool renderTargetFlipY() const override;

        void dispose() override;

        // Inner-renderer accessors for advanced integrations (e.g. attaching
        // ImGui to the GL context, driving the path tracer with the WGPU
        // device, or applying backend-specific tweaks per side).
        GLRenderer& glRenderer();
        WgpuRenderer& wgpuRenderer();

        ~CrossRenderer() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_CROSSRENDERER_HPP
