// How a document asks to be RENDERED, stored on the scene root.
//
// The editor writes this into `scene.userData["render"]`, so the look a user
// dialled in through the Renderer Settings panel — exposure, fog, bloom, GI,
// render scale — travels with the scene the same way its geometry does. No
// sidecar file, no schema extension: the entry rides through
// ObjectExporter/ObjectLoader like every other userData string.
//
// Encoding follows PhysicsConfig: one flat `key=value;key=value` string, since
// userData round-trips scalars only. Unknown keys are ignored on read, which is
// what lets a document written by a newer editor still load here.
//
// BASELINE, not absolute state. Every entry point takes a `base` config that
// supplies whatever the text does not mention, and encode() writes only the
// fields that DIFFER from it. The editor passes the renderer's state as it was
// at startup, which buys two things: a saved document stays a short, readable
// diff of the editor's defaults rather than a forty-key dump, and a document
// that says nothing about (say) fog opens with the editor's fog rather than
// whatever the previously open document left behind.
//
// The struct's own defaults are a freshly constructed VulkanRenderer, so
// RenderConfig{} is a meaningful base for a caller with no renderer to capture.
//
// Deliberately NOT covered: the diagnostic controls (G-buffer debug view,
// overlay layer, frame timings) and the sensor-simulation ones (lens
// distortion, sensor noise). Those describe an INSTRUMENT pointed at the scene,
// not how the scene should look, and they belong to the sensor authoring that
// SensorConfig already carries per object.

#ifndef THREEPP_EDITOR_RENDERCONFIG_HPP
#define THREEPP_EDITOR_RENDERCONFIG_HPP

#include "threepp/constants.hpp"

#include <optional>
#include <string>

namespace threepp {

    class Object3D;
    class Renderer;

}

namespace threepp::editor {

    struct RenderConfig {

        // Which reconstruction runs at the end of the frame. FSR and DLSS are
        // build- and hardware-dependent; a document asking for one the running
        // build does not have falls back to the built-in TAA rather than failing.
        enum class Upscaler {
            Taa,
            Fsr,
            Dlss
        };

        // Mirrors VulkanRenderer::EnvSunPolicy. Restated here so the header
        // stays free of the Vulkan backend (and of THREEPP_WITH_VULKAN, which
        // would make this struct's layout depend on a build option).
        enum class EnvSun {
            Auto,
            Always,
            Off
        };

        // --- every backend ---------------------------------------------------
        ToneMapping toneMapping = ToneMapping::None;
        float exposure = 1.f;
        bool shadows = false;
        ShadowMap shadowType = ShadowMap::PFC;

        // --- Vulkan: resolution and anti-aliasing ----------------------------
        float renderScale = 1.f;
        Upscaler upscaler = Upscaler::Taa;
        int gbufferMsaa = 1;
        float textureAnisotropy = 0.f;// 0 = auto (picked by render mode)

        // --- Vulkan: display -------------------------------------------------
        bool autoExposure = false;
        float autoExposureSpeed = 2.f;
        float whiteBalanceK = 6500.f;
        float whiteBalanceTint = 0.f;
        float sharpen = 0.5f;

        // --- Vulkan: lighting and GI -----------------------------------------
        bool ao = true;
        bool probeGI = true;
        bool denoise = true;
        bool restirDI = true;
        float sunAngularRadius = 0.f;
        EnvSun envSun = EnvSun::Auto;
        float fireflyClamp = 30.f;

        // --- Vulkan: camera --------------------------------------------------
        bool physicalCamera = false;
        bool physicalLightUnits = false;
        float aperture = 16.f;
        float shutterSeconds = 1.f / 125.f;
        float iso = 100.f;
        float exposureCompensation = 0.f;
        bool depthOfField = false;
        float focusDistance = 10.f;
        float motionBlur = 0.f;

        // --- Vulkan: effects -------------------------------------------------
        float bloom = 0.f;
        float bloomThreshold = 1.f;
        // heightFog false = no explicit renderer-side medium: scene.fog drives
        // the density with a near-uniform profile. The profile fields are kept
        // either way, so switching the fog off and on again does not lose them.
        bool heightFog = false;
        float fogDensity = 0.02f;
        float fogBaseY = 0.f;
        float fogFalloff = 80.f;
        float fogNoise = 0.6f;
        float fogAnisotropy = 0.f;
        float beamDensity = 0.f;
        float beamAnisotropy = 0.55f;
        bool clouds = false;
        float cloudCoverage = 0.45f;
        float cloudDensity = 1.f;
        float cloudBottomY = 600.f;
        float cloudTopY = 1400.f;
        float cloudWindX = 8.f;
        float cloudWindZ = 2.f;
        float cloudEvolve = 1.f;

        // --- Vulkan: performance ---------------------------------------------
        bool occlusionCulling = false;
        bool autoLod = true;
        float autoLodError = 0.75f;
        bool normalMapToksvig = true;

        static constexpr const char* userDataKey = "render";

        friend bool operator==(const RenderConfig&, const RenderConfig&) = default;

        // Everything the renderer is right now. A backend without a given knob
        // reports the struct default for it, so a config captured under OpenGL
        // and applied under Vulkan is the Vulkan defaults plus the tone map.
        [[nodiscard]] static RenderConfig capture(const Renderer& renderer);

        // Push onto the renderer. Only the fields that actually differ from its
        // current state are written — render scale and G-buffer MSAA reallocate
        // behind a vkDeviceWaitIdle, and an unchanged value must not pay for it.
        // Knobs the backend does not have are skipped.
        void apply(Renderer& renderer) const;

        // Only the fields that differ from `base`; empty when nothing does.
        [[nodiscard]] std::string encode(const RenderConfig& base = {}) const;
        // `base` supplies every field the text omits.
        [[nodiscard]] static RenderConfig decode(const std::string& text,
                                                 const RenderConfig& base = {});

        // nullopt when the object carries no render entry (or an unreadable one).
        [[nodiscard]] static std::optional<RenderConfig> read(const Object3D& object,
                                                              const RenderConfig& base = {});

        // Writes the entry; a config identical to `base` removes it, so a
        // document that just uses the editor's defaults saves no render block.
        void write(Object3D& object, const RenderConfig& base = {}) const;

        static void erase(Object3D& object);
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_RENDERCONFIG_HPP
