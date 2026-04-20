#ifndef THREEPP_EFFECTCOMPOSER_HPP
#define THREEPP_EFFECTCOMPOSER_HPP

#include <filesystem>
#include <memory>
#include <vector>

namespace threepp {

    class WgpuRenderer;
    class ShaderPass;
    class Object3D;
    class Camera;

    /// Post-processing effect chain for WgpuRenderer.
    /// Renders a scene, then applies a chain of ShaderPass effects via ping-pong textures.
    class EffectComposer {

    public:
        explicit EffectComposer(WgpuRenderer& renderer);
        ~EffectComposer();

        EffectComposer(const EffectComposer&) = delete;
        EffectComposer& operator=(const EffectComposer&) = delete;

        void addPass(std::shared_ptr<ShaderPass> pass);

        void render(Object3D& scene, Camera& camera);

        std::vector<unsigned char> readRGBPixels();

        /// Write the final composed image to disk. Supports .png, .jpg/.jpeg, .bmp.
        void writeFramebuffer(const std::filesystem::path& filename);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_EFFECTCOMPOSER_HPP
