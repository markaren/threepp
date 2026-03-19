#ifndef THREEPP_EFFECTCOMPOSER_HPP
#define THREEPP_EFFECTCOMPOSER_HPP

#include <memory>
#include <vector>

namespace threepp {

    class DawnRenderer;
    class ShaderPass;
    class Object3D;
    class Camera;

    /// Post-processing effect chain for DawnRenderer.
    /// Renders a scene, then applies a chain of ShaderPass effects via ping-pong textures.
    class EffectComposer {

    public:
        explicit EffectComposer(DawnRenderer& renderer);
        ~EffectComposer();

        EffectComposer(const EffectComposer&) = delete;
        EffectComposer& operator=(const EffectComposer&) = delete;

        void addPass(std::shared_ptr<ShaderPass> pass);

        void render(Object3D& scene, Camera& camera);

        std::vector<unsigned char> readRGBPixels();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_EFFECTCOMPOSER_HPP
