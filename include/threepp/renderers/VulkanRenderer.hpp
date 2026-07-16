// Vulkan deferred renderer. Built against the Vulkan SDK and requires
// VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure (ray-queried
// AO/GI) and VK_KHR_ray_query.
//
// Shades a clean, analytic, noise-free base from the raster material G-buffer
// (direct analytic lights + split-sum specular IBL + approximate diffuse IBL)
// plus ray-queried ambient occlusion / global illumination — interactive and
// noise-free, the default for synthetic-perception work. Its shared
// infrastructure lives in the VulkanRendererCore base.
//
// Co-exists with GLRenderer / WgpuRenderer; selected by the application when a
// Canvas is created with GraphicsAPI::Vulkan.

#ifndef THREEPP_VULKANRENDERER_HPP
#define THREEPP_VULKANRENDERER_HPP

#include "threepp/canvas/Canvas.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/renderers/VulkanRendererCore.hpp"

#include <memory>
#include <optional>
#include <utility>

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
        // {density, anisotropy} as last set.
        [[nodiscard]] std::pair<float, float> deferredVolumetrics() const;

        // DEPRECATED (Phase 2 fog unification) — a no-op. The directional sun
        // shafts + aerial glow are now ALWAYS on when the fog medium is present:
        // set scene.fog (or setHeightFog) and the volumetrics follow automatically
        // — the froxels own the near field [0, 512 m] and the per-pixel march owns
        // the far tail [512 m, ∞]. Kept only so existing callers compile.
        void setVolumetricFog(bool enabled);
        [[nodiscard]] bool volumetricFog() const;

        // ── Volumetric clouds (Nubis/HZD-style far-field cloud layer) ────────
        // A raymarched, wind-driven, procedurally-shaped cloud deck occupying
        // the world-space shell [bottomY, topY], composited over the sky (and,
        // depth-aware, in front of terrain). Density is analytic Perlin-Worley
        // noise (no baked assets), remapped by coverage, shaped by a height
        // gradient and eroded by detail — the classic Decima recipe. Lit by the
        // scene's claimed sun (one-sun policy) with a Beer light-march + powder
        // term + dual-lobe Henyey-Greenstein phase, and by env ambient. nullopt
        // = off (default), and off is free (image-identical to no clouds).
        struct CloudSettings {
            float coverage    = 0.45f;         // 0 = clear sky, 1 = overcast
            float density     = 1.0f;          // density multiplier
            float bottomY     = 600.0f;        // shell base (world Y, m)
            float topY        = 1400.0f;       // shell top (world Y, m)
            Vector3 wind{8.0f, 0.0f, 2.0f};    // m/s xz drift (y ignored)
            float evolveSpeed = 1.0f;          // shape churn rate
        };
        void setClouds(const std::optional<CloudSettings>& settings);
        [[nodiscard]] std::optional<CloudSettings> clouds() const;

        // ── Fog medium PROFILE control (advanced) ────────────────────────────
        // Phase 2 fog unification: there is ONE air-fog medium. `scene.fog` is
        // the primary knob — present, it supplies the medium DENSITY (FogExp2
        // density / linear-Fog span) and COLOUR, and the volumetrics run
        // automatically. setHeightFog is the ADVANCED control of that same
        // medium's PROFILE: an exponential height falloff (baseY / falloff) ×
        // wind-scrolled 3D noise, evaluated inside the 0.25–512 m view froxels.
        //
        // PRECEDENCE:
        //   • scene.fog absent  → setHeightFog's `density` CREATES the medium
        //     (back-compat: existing ground-mist users keep working unchanged).
        //   • scene.fog present → scene.fog's density WINS; setHeightFog's
        //     `density` field is IGNORED and only its profile (baseY / falloff /
        //     noiseAmount) shapes the medium. scene.fog alone uses a near-uniform
        //     default profile (baseY 0, huge falloff ≈ homogeneous).
        //
        // The froxels run HETEROGENEOUS whenever a medium exists: per-slice
        // density, a froxel sun in-scatter term (1 RT shadow ray + a short
        // self-shadow march) for [0, 512 m], and the per-pixel march for the far
        // tail. NOTE the froxel medium is the height-fog profile ONLY: the
        // setClouds layer is integrated by the far cloud march over the WHOLE ray,
        // so the two volumes split by phenomenon — no cloud/froxel hand-off. The
        // underwater murk is a SEPARATE medium (setUnderwaterMurk). nullopt = the
        // default near-uniform profile (no explicit height falloff).
        struct HeightFogSettings {
            float density     = 0.02f;// σ_t at baseY
            float baseY       = 0.0f;
            float falloff     = 80.0f;// exponential height scale (m)
            float noiseAmount = 0.6f; // 0 = smooth analytic, 1 = fully noise-modulated
        };
        void setHeightFog(const std::optional<HeightFogSettings>& settings);
        [[nodiscard]] std::optional<HeightFogSettings> heightFog() const;

        // Procedural star field on SKY pixels — hash-based points in direction
        // space, pixel-crisp at any resolution/FOV. 0 disables; ~1.0 = night sky.
        void setDeferredStarfield(float intensity);
        [[nodiscard]] float deferredStarfield() const;

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
        [[nodiscard]] float autoExposureSpeed() const;

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
