// Vulkan deferred renderer. Built against the Vulkan SDK and requires
// VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure (ray-queried
// AO/GI) and VK_KHR_ray_query.
//
// Shades a clean, analytic, noise-free base from the raster material G-buffer
// (direct analytic lights + split-sum specular IBL + approximate diffuse IBL)
// plus ray-queried ambient occlusion / global illumination — interactive and
// noise-free, the default for synthetic-perception work. The reference path
// tracer lives in its sibling VulkanPathTracer; both share VulkanRendererCore.
//
// Co-exists with GLRenderer / WgpuRenderer; selected by the application when a
// Canvas is created with GraphicsAPI::Vulkan.

#ifndef THREEPP_VULKANRENDERER_HPP
#define THREEPP_VULKANRENDERER_HPP

#include "threepp/canvas/Canvas.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/renderers/VulkanRendererCore.hpp"

#include <memory>

namespace threepp {

    class VulkanRenderer : public VulkanRendererCore {

    public:
        explicit VulkanRenderer(Canvas& canvas);

        ~VulkanRenderer() override;

        VulkanRenderer(const VulkanRenderer&) = delete;
        VulkanRenderer& operator=(const VulkanRenderer&) = delete;

        // ── Deferred shading knobs ────────────────────────────────────────────

        // Ray-traced env ambient-occlusion / GI. ON by default — soft RT AO/GI
        // (costs occlusion rays; paired with setDenoise, also on by default, to keep
        // the 1-spp gather noise-free). Turn OFF to drop the per-pixel ray cost, or
        // if occlusion-testing the IBL makes a bright HDRI look like it casts shadows.
        void setDeferredAO(bool enabled);
        [[nodiscard]] bool deferredAO() const;

        // World-space irradiance probe grid (DDGI-lite) — multi-bounce diffuse
        // GI for the deferred gather. A 32×16×32 grid of SH-L1 probes is fitted
        // to the scene AABB and refreshed round-robin (2048 probes × 64 rays
        // per frame, ~sub-ms); the stochastic GI gather then adds each hit's
        // probe irradiance, supplying the bounce-2..∞ + through-the-opening
        // sky light a 1-bounce gather cannot.
        // It ALSO switches the deferred ambient model from cosmetic to
        // MEASURED: scene ambient is gated by the gather's real sky visibility,
        // the rough split-sum env specular gets probe-derived specular
        // occlusion, and reflected hits take probe irradiance instead of the
        // env+ambient fill. Enclosed interiors therefore stop being "lit with
        // no light" — they go honestly dark and receive only what bounces in
        // through actual openings (e.g. the Sponza ground-floor corridors);
        // pair with setAutoExposure for interior scenes.
        // ON by default (≈ neutral outdoors; ~0.3 ms probe update). Requires
        // setDeferredAO(true) + setDenoise(true) (the probe term rides the
        // denoised GI channel); the cache converges over a few dozen frames
        // after scene load. setProbeGI(false) restores the legacy cosmetic
        // ambient (ungated env/ambient fill, no multi-bounce).
        void setProbeGI(bool enabled);
        [[nodiscard]] bool probeGI() const;

        // Volumetric SPOT-light beams: ray-marched single scattering through a
        // uniform thin haze. `density` is the scattering coefficient σ (1/m; 0 =
        // off, no cost); `anisotropy` is the Henyey-Greenstein g.
        void setDeferredVolumetrics(float density, float anisotropy = 0.55f);

        // Volumetric DIRECTIONAL-light fog: ray-marches the camera→surface air
        // column, tracing an RT shadow ray toward each sun per step so trees and
        // terrain carve real light shafts, and brightens the haze toward the sun
        // via the Henyey-Greenstein phase (driven by setFogAnisotropy). Only
        // contributes when scene.fog is set; opt-in because of the per-step
        // shadow-ray cost. This is what gives an outdoor (sun-lit) deferred scene
        // genuine volume rather than flat distance-faded haze.
        void setVolumetricFog(bool enabled);
        [[nodiscard]] bool volumetricFog() const;

        // Procedural star field on SKY pixels — hash-based points in direction
        // space, pixel-crisp at any resolution/FOV. 0 disables; ~1.0 = night sky.
        void setDeferredStarfield(float intensity);

        // HDRI sun extraction. The environment map's dominant compact bright
        // source (the sun) is removed from the PMREM's glossy / rough mips at
        // upload — a ~10⁴:1 disc cannot be Monte-Carlo prefiltered smoothly and
        // shows up as bright blocky "spec blobs" in reflections — and its exact
        // energy can be re-injected as an analytic directional light: a sharp
        // correct sun highlight (the ONLY sun reflection), soft RT shadows
        // (setSunAngularRadius), GI bounce and volumetric shafts. The sky
        // background and true mirror lookups (env mip 0) keep the visible disc.
        //
        // ONE-SUN POLICY — a scene must not end up with the sun twice. Scenes
        // authored for raster renderers add an explicit DirectionalLight as the
        // sun stand-in (raster can't shadow from an env map); an extractor that
        // ALSO injects the env's sun would then light and shadow the scene with
        // two suns (the reported double-shadow). So:
        //   Auto (default) — extract (clamp the glossy mips) always; INJECT the
        //     analytic sun only while the scene has NO visible DirectionalLight.
        //     If the artist provided a sun light, theirs owns direct sun +
        //     shadow (the raster/three.js convention) and the env supplies
        //     sky/ambience only. Exactly one sun in every renderer.
        //   Always — extract AND inject regardless of scene lights (a scene
        //     that genuinely wants the env sun PLUS extra directional lights).
        //   Off — no extraction at all: raw env in every mip (legacy; the HDRI
        //     sun prefilters into blocky spec blobs), nothing injected.
        // Auto↔Always applies next frame; Off toggles force an env re-upload.
        enum class EnvSunPolicy { Auto, Always, Off };
        void setEnvSunPolicy(EnvSunPolicy policy);
        [[nodiscard]] EnvSunPolicy envSunPolicy() const;

        // Back-compat shim: true → Auto, false → Off.
        void setEnvSunExtraction(bool enabled);
        [[nodiscard]] bool envSunExtraction() const;

        // The measured env sun (valid while envSunFound()): unit direction
        // TOWARD the sun and the disc's integrated energy Σ L·dΩ (linear RGB
        // irradiance). Use to ALIGN an explicit DirectionalLight with the HDRI
        // (e.g. so a raster renderer's shadow direction matches the sky).
        [[nodiscard]] bool envSunFound() const;
        [[nodiscard]] Vector3 envSunDirection() const;
        [[nodiscard]] Vector3 envSunColor() const;

        // ── Automatic exposure (eye adaptation) ──────────────────────────────
        // When enabled the renderer samples the log2-luma histogram of the
        // rendered frame each tick and drives toneMappingExposure toward the
        // value that maps the scene's weighted-average luminance to 18% gray,
        // using an asymmetric EMA (fast constriction, slow dilation).
        // toneMappingExposure is IGNORED while auto-exposure is active.
        void setAutoExposure(bool enabled);
        [[nodiscard]] bool autoExposure() const;

        // EV per second for brightness adaptation (default 2.0).
        // Dilation (scene-darkens) is applied at 0.5× this speed.
        void setAutoExposureSpeed(float evPerSecond);

        // Exposure clamp in EV relative to 1.0 (default -3 to +3 EV).
        // E.g. setAutoExposureRange(-2, 4) limits to 0.25× .. 16× exposure.
        void setAutoExposureRange(float minEV, float maxEV);

        // ── Raster G-buffer MSAA (edge/silhouette anti-flicker) ─────────────
        // Rasterizes the material G-buffer at `samples` (1, 2, or 4) per
        // pixel instead of 1, then resolves each pixel to the majority-
        // covering surface (dominant-sample pick, not a box/average blend —
        // averaging normals/ids/depth across a silhouette produces nonsense).
        // Targets the source of the 1-spp jittered-coverage edge flicker
        // (leaf canopies, low-poly rock fields shimmering on a STATIC
        // camera): sample coverage is exact and far more temporally stable
        // than a single jittered point sample. Default 1 = today's path,
        // byte-identical output, zero extra cost. Reallocates render-extent
        // resources (render pass + pipelines + MS images) — same
        // vkDeviceWaitIdle / deferred-apply-mid-frame contract as
        // setRenderScale; safe to call from inside the user's animate
        // lambda. VRAM cost is real (roughly samples× the G-buffer's raster
        // attachments); 2 is the recommended quality step, 4 for maximum
        // stability.
        void setGbufferMsaa(uint32_t samples);
        [[nodiscard]] uint32_t gbufferMsaa() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;

        [[nodiscard]] CoreImpl* coreImpl() const override;
        void disposeImpl() override;
    };

}// namespace threepp

#endif//THREEPP_VULKANRENDERER_HPP
