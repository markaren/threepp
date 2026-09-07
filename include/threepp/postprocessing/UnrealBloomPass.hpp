// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/postprocessing/UnrealBloomPass.js

#ifndef THREEPP_POSTPROCESSING_UNREALBLOOMPASS_HPP
#define THREEPP_POSTPROCESSING_UNREALBLOOMPASS_HPP

#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/postprocessing/Pass.hpp"

#include <array>
#include <vector>

namespace threepp {

    class MeshBasicMaterial;
    class RenderTarget;
    class ShaderMaterial;

    // Bloom the way Unreal does it: threshold the image, then blur what is left
    // over a chain of five progressively halved mips and add them back
    // weighted. The wide, soft falloff comes from the small mips — blurring the
    // full-resolution image far enough to look like light bleeding would cost
    // far more than blurring a 32-pixel-wide one.
    class UnrealBloomPass: public Pass {

    public:
        // Multiplies the composited bloom before it is added back.
        float strength;

        // Shifts weight between the tight mips and the wide ones, 0..1. Higher
        // spreads the glow further out.
        float radius;

        // Luminance a pixel must exceed to bloom at all. Around 1.0 with a
        // tone-mapped scene, lower for one that never reaches white.
        float threshold;

        // Per-mip tint, five entries, white by default. A warm large mip over
        // neutral small ones is what gives a lens its character.
        std::array<Vector3, 5> tintColors{};

        // `resolution` is the size the internal chain is built for — the
        // composer overrides it through setSize as soon as the pass is added.
        explicit UnrealBloomPass(const Vector2& resolution = Vector2(256, 256),
                                 float strength = 1.f,
                                 float radius = 0.f,
                                 float threshold = 0.f);

        void setSize(unsigned int width, unsigned int height) override;

        void render(GLRenderer& renderer,
                    RenderTarget* writeBuffer,
                    RenderTarget* readBuffer,
                    float deltaTime,
                    bool maskActive) override;

        ~UnrealBloomPass() override;

    private:
        static constexpr int nMips = 5;

        std::vector<std::unique_ptr<RenderTarget>> renderTargetsHorizontal_;
        std::vector<std::unique_ptr<RenderTarget>> renderTargetsVertical_;
        std::unique_ptr<RenderTarget> renderTargetBright_;

        std::shared_ptr<ShaderMaterial> highPassMaterial_;
        std::vector<std::shared_ptr<ShaderMaterial>> blurMaterials_;
        std::shared_ptr<ShaderMaterial> compositeMaterial_;
        std::shared_ptr<ShaderMaterial> blendMaterial_;
        std::shared_ptr<MeshBasicMaterial> basicMaterial_;

        std::unique_ptr<FullScreenQuad> fsQuad_;
    };

}// namespace threepp

#endif//THREEPP_POSTPROCESSING_UNREALBLOOMPASS_HPP
