// https://github.com/mrdoob/three.js/blob/r186/examples/jsm/postprocessing/GTAOPass.js

#ifndef THREEPP_POSTPROCESSING_GTAOPASS_HPP
#define THREEPP_POSTPROCESSING_GTAOPASS_HPP

#include "threepp/math/Box3.hpp"
#include "threepp/postprocessing/Pass.hpp"

#include <optional>
#include <vector>

namespace threepp {

    class Camera;
    class DepthTexture;
    class MeshNormalMaterial;
    class Object3D;
    class RenderTarget;
    class Scene;
    class ShaderMaterial;
    class Texture;

    // Knobs of the AO estimate itself. Unset fields are left as they are, so
    // updateGtaoMaterial({.radius = 0.5f}) touches only the radius. Defaults
    // noted are the pass's starting values (three.js r186's).
    struct GTAOParameters {
        // World-space reach of the horizon search (0.25). The size of the
        // creases that darken; too large and whole objects shade each other.
        std::optional<float> radius;
        // >1 packs the samples towards the centre (1).
        std::optional<float> distanceExponent;
        // View-depth difference beyond which a sample no longer occludes (1).
        // What stops a foreground silhouette from darkening what is behind it.
        std::optional<float> thickness;
        // 0..1, how much less the far samples count (1).
        std::optional<float> distanceFallOff;
        // Exponent on the AO factor (1); >1 darkens.
        std::optional<float> scale;
        // Depth taps per pixel (16), split over 3 slice directions (5 from
        // 30 up). The main cost knob; recompiles the shader.
        std::optional<int> samples;
        // Measure `radius` in screen pixels (x100) instead of world units
        // (false). Recompiles the shader.
        std::optional<bool> screenSpaceRadius;
    };

    // Knobs of the Poisson denoise that follows the AO.
    struct PoissonDenoiseParameters {
        // Edge stops: larger lets taps across luminance / plane distance /
        // normal differences count (10, 2, 3).
        std::optional<float> lumaPhi;
        std::optional<float> depthPhi;
        std::optional<float> normalPhi;
        // Pixel radius of the disk (8).
        std::optional<float> radius;
        // Radius distribution of the taps (2 on the pass; the shader starts
        // with 1 until the pattern is regenerated, as in three.js).
        std::optional<float> radiusExponent;
        // Turns of the tap spiral (2).
        std::optional<float> rings;
        // Taps per pixel (16). Cost knob, together with GTAO's samples.
        std::optional<int> samples;
    };

    // Ground-truth ambient occlusion, as a post pass: darkens creases, corners
    // and contact points where the environment's light cannot reach.
    //
    // Per frame it renders the scene once more with a MeshNormalMaterial
    // override into a normal target that also carries a depth texture (points
    // and lines are hidden for it), estimates AO from the two, denoises the
    // result, and multiplies it into the image. The multiply happens in the
    // composer's linear light, before its output transform.
    //
    // `output` selects what reaches the next pass - the lit image with AO
    // (Default), or one of the intermediate buffers for inspection.
    class GTAOPass: public Pass {

    public:
        enum class Output {
            Off = -1,// the pass does nothing and leaves the image as it is
            Default = 0,
            Diffuse = 1,
            Depth = 2,
            Normal = 3,
            AO = 4,
            Denoise = 5
        };

        Output output = Output::Default;

        // 0 = no AO, 1 = full AO.
        float blendIntensity = 1.f;

        GTAOPass(Scene& scene, Camera& camera,
                 unsigned int width = 512, unsigned int height = 512,
                 const std::optional<GTAOParameters>& aoParameters = std::nullopt,
                 const std::optional<PoissonDenoiseParameters>& pdParameters = std::nullopt);

        void updateGtaoMaterial(const GTAOParameters& parameters);

        void updatePdMaterial(const PoissonDenoiseParameters& parameters);

        // Fade AO out beyond `box` (grown by the radius) - keeps a small scene
        // from shading a distant backdrop. nullopt removes the clip box.
        void setSceneClipBox(const std::optional<Box3>& box);

        // The denoised AO, 1 = unoccluded.
        [[nodiscard]] Texture* gtaoMap() const;

        void setSize(unsigned int width, unsigned int height) override;

        void render(GLRenderer& renderer,
                    RenderTarget* writeBuffer,
                    RenderTarget* readBuffer,
                    float deltaTime,
                    bool maskActive) override;

        ~GTAOPass() override;

    private:
        Scene* scene_;
        Camera* camera_;

        unsigned int width_;
        unsigned int height_;

        float pdRings_ = 2.f;
        float pdRadiusExponent_ = 2.f;
        int pdSamples_ = 16;

        std::vector<Object3D*> visibilityCache_;

        std::shared_ptr<Texture> gtaoNoiseTexture_;
        std::shared_ptr<Texture> pdNoiseTexture_;

        std::shared_ptr<DepthTexture> depthTexture_;
        std::unique_ptr<RenderTarget> normalRenderTarget_;
        std::unique_ptr<RenderTarget> gtaoRenderTarget_;
        std::unique_ptr<RenderTarget> pdRenderTarget_;

        std::shared_ptr<ShaderMaterial> gtaoMaterial_;
        std::shared_ptr<MeshNormalMaterial> normalMaterial_;
        std::shared_ptr<ShaderMaterial> pdMaterial_;
        std::shared_ptr<ShaderMaterial> depthRenderMaterial_;
        std::shared_ptr<ShaderMaterial> copyMaterial_;
        std::shared_ptr<ShaderMaterial> blendMaterial_;

        std::unique_ptr<FullScreenQuad> fsQuad_;

        void renderPass(GLRenderer& renderer, const std::shared_ptr<ShaderMaterial>& material,
                        RenderTarget* target, std::optional<unsigned int> clearColor = std::nullopt, float clearAlpha = 0.f);
        void renderOverride(GLRenderer& renderer, RenderTarget* target, unsigned int clearColor, float clearAlpha);
        void overrideVisibility();
        void restoreVisibility();
    };

}// namespace threepp

#endif//THREEPP_POSTPROCESSING_GTAOPASS_HPP
