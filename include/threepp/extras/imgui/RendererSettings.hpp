// Reusable ImGui panel for the generic runtime settings of a threepp
// Renderer — the knobs every example otherwise hand-rolls (exposure, tone
// map, upscaler, MSAA, denoiser, bloom, ...). The panel adapts to the
// renderer it is given: the full deferred surface for VulkanRenderer,
// shadow-map + tone-map controls for GLRenderer, tone-map only otherwise.
//
// Compact by default — the whole panel folds into a single collapsed
// "Renderer settings" row, so it can ride along in any example's window
// without eating space:
//
//   RendererSettings settings(renderer);
//   ImguiFunctionalContext ui(canvas, renderer, [&] {
//       ImGui::Begin("My demo");
//       ...app-specific widgets...
//       settings.drawCollapsed();   // one row until expanded
//       ImGui::End();
//   });
//
// Turn-key — for the common example shape (one window + a few app widgets +
// ImGui input capture), RendererSettingsUi owns the ImGui context, the
// window and the IOCapture wiring; `extra` widgets appear above the
// collapsed settings row:
//
//   RendererSettingsUi ui(canvas, renderer, [&] {
//       if (ImGui::Button("Toggle lights")) ...;
//   });
//   canvas.animate([&] {
//       ...
//       renderer.render(scene, camera);
//       ui.render();
//   });
//
// State model: every widget reads the renderer's current state through its
// getters each frame and only writes on user edits — the panel never
// overrides what an example configured at startup, and it stays in sync
// with programmatic setter calls at runtime.

#ifndef THREEPP_IMGUI_RENDERER_SETTINGS_HPP
#define THREEPP_IMGUI_RENDERER_SETTINGS_HPP

#include <imgui.h>

#include <threepp/renderers/GLRenderer.hpp>
#include <threepp/renderers/Renderer.hpp>

#ifdef THREEPP_WITH_VULKAN
#include <threepp/renderers/VulkanRenderer.hpp>
#endif

#include "ImguiContext.hpp"

#include <threepp/canvas/Canvas.hpp>
#include <threepp/input/IOCapture.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>

class RendererSettings {

public:
    explicit RendererSettings(threepp::Renderer& renderer)
        : renderer_(&renderer) {
#ifdef THREEPP_WITH_VULKAN
        vk_ = dynamic_cast<threepp::VulkanRenderer*>(&renderer);
#endif
        gl_ = dynamic_cast<threepp::GLRenderer*>(&renderer);
    }

    // One collapsed row that expands into the full panel — the compact,
    // embed-anywhere form.
    void drawCollapsed(const char* label = "Renderer settings") {
        if (section(label, false)) {
            drawSections();
            ImGui::TreePop();
        }
    }

    // FPS line + the panel sections, expanded at top level — embed inside
    // your own Begin()/End() when the settings ARE the window's content.
    void drawContent() {
        drawFps();
        ImGui::Separator();
        drawSections();
    }

    // Standalone window. `extra` (optional) is drawn above the settings —
    // the hook for an example's app-specific widgets.
    void draw(const char* title = "Renderer",
              const std::function<void()>& extra = nullptr) {
        ImGui::SetNextWindowPos({0, 0}, ImGuiCond_FirstUseEver);
        ImGui::Begin(title, nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        drawFps();
        if (extra) {
            ImGui::Separator();
            extra();
        }
        ImGui::Separator();
        drawCollapsed();
        ImGui::End();
    }

private:
    threepp::Renderer* renderer_;
    threepp::GLRenderer* gl_ = nullptr;

    // ---- shared widgets --------------------------------------------------

    static void drawFps() {
        const ImGuiIO& io = ImGui::GetIO();
        ImGui::Text("%.1f FPS (%.2f ms)", io.Framerate,
                    io.Framerate > 0.f ? 1000.f / io.Framerate : 0.f);
    }

    // Framed tree node — reads like a CollapsingHeader but nests/indents
    // properly, so the panel works at any depth. The node pushes its label
    // as an ID scope, keeping same-named widgets in different sections
    // apart. Caller must TreePop() when this returns true.
    static bool section(const char* label, bool defaultOpen) {
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth;
        if (defaultOpen) flags |= ImGuiTreeNodeFlags_DefaultOpen;
        return ImGui::TreeNodeEx(label, flags);
    }

    void drawSections() {
#ifdef THREEPP_WITH_VULKAN
        if (vk_) {
            drawVulkan();
            return;
        }
#endif
        if (gl_) {
            drawGL();
            return;
        }
        drawToneMapAndExposure(false, false);
    }

    void drawToneMapAndExposure(bool allowAgX, bool exposureDriven) {
        using threepp::ToneMapping;
        struct Entry {
            const char* name;
            ToneMapping value;
        };
        static const Entry entries[] = {
                {"None", ToneMapping::None},
                {"Linear", ToneMapping::Linear},
                {"Reinhard", ToneMapping::Reinhard},
                {"Cineon", ToneMapping::Cineon},
                {"ACES Filmic", ToneMapping::ACESFilmic},
                {"Neutral (PBR)", ToneMapping::Neutral},
                {"AgX", ToneMapping::AgX},
        };
        const int count = allowAgX ? 7 : 6;

        const char* preview = "Custom";
        for (int i = 0; i < count; ++i) {
            if (entries[i].value == renderer_->toneMapping) preview = entries[i].name;
        }
        if (ImGui::BeginCombo("Tone map", preview)) {
            for (int i = 0; i < count; ++i) {
                const bool selected = entries[i].value == renderer_->toneMapping;
                if (ImGui::Selectable(entries[i].name, selected)) {
                    renderer_->toneMapping = entries[i].value;
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (exposureDriven) ImGui::BeginDisabled();
        ImGui::SliderFloat("Exposure", &renderer_->toneMappingExposure,
                           0.05f, 4.f, "%.2f", ImGuiSliderFlags_Logarithmic);
        if (exposureDriven) {
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextDisabled("(auto)");
        }
    }

    // ---- OpenGL ----------------------------------------------------------

    void drawGL() {
        drawToneMapAndExposure(false, false);

        if (section("Shadows", false)) {
            auto& sm = gl_->shadowMap();
            if (ImGui::Checkbox("Enabled", &sm.enabled)) sm.needsUpdate = true;
            static const char* types[] = {"Basic", "PCF", "PCF soft", "VSM"};
            int type = static_cast<int>(sm.type);
            if (ImGui::Combo("Type", &type, types, IM_ARRAYSIZE(types))) {
                sm.type = static_cast<threepp::ShadowMap>(type);
                sm.needsUpdate = true;
            }
            ImGui::Checkbox("Auto update", &sm.autoUpdate);
            ImGui::TreePop();
        }
    }

#ifdef THREEPP_WITH_VULKAN

    threepp::VulkanRenderer* vk_ = nullptr;

    // Remembered across an off→on toggle of the optional-struct settings.
    threepp::VulkanRenderer::CloudSettings lastClouds_{};
    threepp::VulkanRenderer::HeightFogSettings lastHeightFog_{};
    // Live slider value while a waitIdle-heavy setting is being dragged;
    // applied on release (see renderScale below).
    float renderScaleEdit_ = 1.f;
    bool renderScaleActive_ = false;

    void drawVulkan() {
        if (section("Display", true)) {
            drawVulkanDisplay();
            ImGui::TreePop();
        }
        if (section("Resolution & AA", true)) {
            drawVulkanResolution();
            ImGui::TreePop();
        }
        if (section("Lighting & GI", false)) {
            drawVulkanLighting();
            ImGui::TreePop();
        }
        if (section("Camera", false)) {
            drawVulkanCamera();
            ImGui::TreePop();
        }
        if (section("Effects", false)) {
            drawVulkanEffects();
            ImGui::TreePop();
        }
        if (section("Performance", false)) {
            drawVulkanPerformance();
            ImGui::TreePop();
        }
        if (section("Debug", false)) {
            drawVulkanDebug();
            ImGui::TreePop();
        }
    }

    void drawVulkanDisplay() {
        const bool exposureDriven = vk_->autoExposure() || vk_->physicalCamera();
        drawToneMapAndExposure(true, exposureDriven);

        bool autoExp = vk_->autoExposure();
        if (ImGui::Checkbox("Auto exposure", &autoExp)) vk_->setAutoExposure(autoExp);
        if (autoExp) {
            float speed = vk_->autoExposureSpeed();
            if (ImGui::SliderFloat("Adaptation (EV/s)", &speed, 0.25f, 8.f,
                                   "%.2f", ImGuiSliderFlags_Logarithmic)) {
                vk_->setAutoExposureSpeed(speed);
            }
        }

        // LUT rebake on the CPU — only call the setter on actual edits.
        auto [wbK, wbTint] = vk_->whiteBalance();
        bool wb = false;
        wb |= ImGui::SliderFloat("White balance (K)", &wbK, 1667.f, 25000.f,
                                 "%.0f", ImGuiSliderFlags_Logarithmic);
        wb |= ImGui::SliderFloat("Tint", &wbTint, -1.f, 1.f, "%.2f");
        if (wb) vk_->setWhiteBalance(wbK, wbTint);

        float sharpen = vk_->sharpenStrength();
        if (ImGui::SliderFloat("Sharpen (RCAS)", &sharpen, 0.f, 1.f, "%.2f")) {
            vk_->setSharpenStrength(sharpen);
        }
    }

    void drawVulkanResolution() {
        // setRenderScale waitIdles + reallocates — apply on slider release,
        // not per drag tick.
        if (!renderScaleActive_) renderScaleEdit_ = vk_->renderScale();
        if (ImGui::SliderFloat("Render scale", &renderScaleEdit_, 0.25f, 1.f, "%.2f")) {
            renderScaleActive_ = true;
        }
        if (renderScaleActive_ && ImGui::IsItemDeactivated()) {
            if (ImGui::IsItemDeactivatedAfterEdit()) vk_->setRenderScale(renderScaleEdit_);
            renderScaleActive_ = false;
        }

        const bool fsrAvail = vk_->fsrAvailable();
        const bool dlssAvail = vk_->dlssAvailable();
        if (fsrAvail || dlssAvail) {
            const char* names[3];
            int count = 0;
            const int taaIdx = count;
            names[count++] = "TAA (built-in)";
            const int fsrIdx = fsrAvail ? count : -1;
            if (fsrAvail) names[count++] = "FSR 3.1";
            const int dlssIdx = dlssAvail ? count : -1;
            if (dlssAvail) names[count++] = "DLSS";

            int current = vk_->dlss() ? dlssIdx : (vk_->fsr() ? fsrIdx : taaIdx);
            if (current < 0) current = taaIdx;
            if (ImGui::Combo("Upscaler", &current, names, count)) {
                vk_->setDlss(current == dlssIdx);
                vk_->setFsr(current == fsrIdx);
            }
        }

        static const char* msaaNames[] = {"1x (off)", "2x", "4x"};
        const uint32_t msaa = vk_->gbufferMsaa();
        int msaaIdx = msaa >= 4 ? 2 : (msaa == 2 ? 1 : 0);
        if (ImGui::Combo("G-buffer MSAA", &msaaIdx, msaaNames, IM_ARRAYSIZE(msaaNames))) {
            vk_->setGbufferMsaa(msaaIdx == 2 ? 4u : (msaaIdx == 1 ? 2u : 1u));
        }

        // AUTO = 16x when the raster is unjittered (MSAA / event camera),
        // isotropic under TAA/DLSS/FSR jitter (grazing-angle sharpening
        // defeats temporal reconstruction).
        static const char* anisoNames[] = {"Auto (by mode)", "1x (iso)", "2x", "4x", "8x", "16x"};
        static const float anisoValues[] = {0.f, 1.f, 2.f, 4.f, 8.f, 16.f};
        const float aniso = vk_->textureAnisotropy();
        int anisoIdx = 0;
        for (int i = 1; i < IM_ARRAYSIZE(anisoValues); ++i) {
            if (aniso >= anisoValues[i]) anisoIdx = i;
        }
        if (ImGui::Combo("Texture aniso", &anisoIdx, anisoNames, IM_ARRAYSIZE(anisoNames))) {
            vk_->setTextureAnisotropy(anisoValues[anisoIdx]);
        }
    }

    void drawVulkanLighting() {
        bool ao = vk_->deferredAO();
        if (ImGui::Checkbox("RT AO / GI", &ao)) vk_->setDeferredAO(ao);

        bool probes = vk_->probeGI();
        if (ImGui::Checkbox("Probe GI (multi-bounce)", &probes)) vk_->setProbeGI(probes);

        bool denoise = vk_->denoise();
        if (ImGui::Checkbox("Denoiser", &denoise)) vk_->setDenoise(denoise);

        bool restir = vk_->restirDIEnabled();
        if (ImGui::Checkbox("ReSTIR DI", &restir)) vk_->setRestirDIEnabled(restir);

        float sunSoft = vk_->sunAngularRadius();
        if (ImGui::SliderFloat("Sun softness (deg)", &sunSoft, 0.f, 3.f, "%.2f")) {
            vk_->setSunAngularRadius(sunSoft);
        }

        using Policy = threepp::VulkanRenderer::EnvSunPolicy;
        static const char* policyNames[] = {"Auto", "Always", "Off"};
        int policy = static_cast<int>(vk_->envSunPolicy());
        if (ImGui::Combo("Env sun extraction", &policy, policyNames, IM_ARRAYSIZE(policyNames))) {
            vk_->setEnvSunPolicy(static_cast<Policy>(policy));
        }

        float clamp = vk_->fireflyClamp();
        if (clamp > 1e6f) clamp = 0.f;// disabled sentinel
        if (ImGui::SliderFloat("Firefly clamp", &clamp, 0.f, 64.f, "%.0f (0 = off)")) {
            vk_->setFireflyClamp(clamp);
        }
    }

    void drawVulkanCamera() {
        bool physCam = vk_->physicalCamera();
        if (ImGui::Checkbox("Physical camera (EV100)", &physCam)) vk_->setPhysicalCamera(physCam);

        bool physUnits = vk_->physicalLightUnits();
        if (ImGui::Checkbox("Physical light units", &physUnits)) vk_->setPhysicalLightUnits(physUnits);

        // Aperture drives DoF bokeh even while physical exposure is off.
        auto ce = vk_->cameraExposure();
        int shutterDen = static_cast<int>(std::lround(1.f / std::max(ce.shutterSeconds, 1.f / 8000.f)));
        shutterDen = std::clamp(shutterDen, 1, 8000);
        bool exp = false;
        exp |= ImGui::SliderFloat("Aperture", &ce.aperture, 1.f, 22.f,
                                  "f/%.1f", ImGuiSliderFlags_Logarithmic);
        exp |= ImGui::SliderInt("Shutter", &shutterDen, 1, 8000,
                                "1/%d s", ImGuiSliderFlags_Logarithmic);
        exp |= ImGui::SliderFloat("ISO", &ce.iso, 25.f, 25600.f,
                                  "%.0f", ImGuiSliderFlags_Logarithmic);
        if (exp) vk_->setCameraExposure(ce.aperture, 1.f / static_cast<float>(shutterDen), ce.iso);

        float evComp = vk_->exposureCompensation();
        if (ImGui::SliderFloat("EV compensation", &evComp, -5.f, 5.f, "%.1f")) {
            vk_->setExposureCompensation(evComp);
        }

        bool dof = vk_->depthOfField();
        if (ImGui::Checkbox("Depth of field", &dof)) vk_->setDepthOfField(dof);
        if (dof) {
            float focus = vk_->focusDistance();
            if (ImGui::SliderFloat("Focus distance (m)", &focus, 0.1f, 500.f,
                                   "%.1f", ImGuiSliderFlags_Logarithmic)) {
                vk_->setFocusDistance(focus);
            }
        }

        float mb = vk_->motionBlur();
        if (ImGui::SliderFloat("Motion blur", &mb, 0.f, 1.f, "%.2f shutter")) {
            vk_->setMotionBlur(mb);
        }
    }

    void drawVulkanEffects() {
        float bloom = vk_->bloomIntensity();
        if (ImGui::SliderFloat("Bloom", &bloom, 0.f, 1.f, "%.3f",
                               ImGuiSliderFlags_Logarithmic)) {
            vk_->setBloomIntensity(bloom);
        }
        if (bloom > 0.f) {
            float thresh = vk_->bloomThreshold();
            if (ImGui::SliderFloat("Bloom threshold", &thresh, 0.f, 8.f, "%.2f")) {
                vk_->setBloomThreshold(thresh);
            }
        }

        // ── Fog (one unified volumetric medium) ─────────────────────────────
        // ONE primary knob: Fog density. This drives the RENDERER-side medium
        // (setHeightFog) — the froxel volumetrics (god rays, aerial perspective)
        // run automatically whenever a medium is present, so there is no separate
        // "volumetric fog" toggle any more. NOTE: if the APP sets scene.fog, that
        // takes precedence (it supplies density + colour); this control then only
        // shapes the profile. The panel holds the renderer, not the scene, so it
        // cannot read/write scene.fog directly — drive scene.fog from the app.
        {
            auto hf = vk_->heightFog();
            float fogDensity = hf ? hf->density : 0.f;
            if (ImGui::SliderFloat("Fog density", &fogDensity, 0.f, 0.2f, "%.4f",
                                   ImGuiSliderFlags_Logarithmic)) {
                if (fogDensity > 1e-6f) {
                    threepp::VulkanRenderer::HeightFogSettings s = hf.value_or(lastHeightFog_);
                    s.density = fogDensity;
                    vk_->setHeightFog(s);
                } else {
                    if (hf) lastHeightFog_ = *hf;
                    vk_->setHeightFog(std::nullopt);
                }
            }
            if (ImGui::TreeNode("Advanced fog")) {
                float fogG = vk_->getFogAnisotropy();
                if (ImGui::SliderFloat("Anisotropy", &fogG, -0.95f, 0.95f, "%.2f")) {
                    vk_->setFogAnisotropy(fogG);
                }
                // Profile (only meaningful once a renderer-side medium exists —
                // a near-uniform default otherwise). Editing these switches the
                // medium from the uniform default to a layered height fog.
                if (auto m = vk_->heightFog()) {
                    auto s = *m;
                    bool ch = false;
                    ch |= ImGui::SliderFloat("Base Y (m)", &s.baseY, -100.f, 500.f, "%.0f");
                    ch |= ImGui::SliderFloat("Falloff (m)", &s.falloff, 1.f, 2000.f, "%.0f",
                                             ImGuiSliderFlags_Logarithmic);
                    ch |= ImGui::SliderFloat("Noise", &s.noiseAmount, 0.f, 1.f, "%.2f");
                    if (ch) vk_->setHeightFog(s);
                } else {
                    ImGui::TextDisabled("(uniform profile — raise Fog density to shape it)");
                }
                ImGui::TreePop();
            }
        }

        auto [beamDensity, beamAniso] = vk_->deferredVolumetrics();
        bool beams = false;
        beams |= ImGui::SliderFloat("Spot beams (1/m)", &beamDensity, 0.f, 0.2f, "%.3f",
                                    ImGuiSliderFlags_Logarithmic);
        beams |= ImGui::SliderFloat("Beam anisotropy", &beamAniso, -0.9f, 0.9f, "%.2f");
        if (beams) vk_->setDeferredVolumetrics(beamDensity, beamAniso);

        float stars = vk_->deferredStarfield();
        if (ImGui::SliderFloat("Starfield", &stars, 0.f, 2.f, "%.2f")) {
            vk_->setDeferredStarfield(stars);
        }

        bool cloudsOn = vk_->clouds().has_value();
        if (ImGui::Checkbox("Volumetric clouds", &cloudsOn)) {
            if (cloudsOn) {
                vk_->setClouds(lastClouds_);
            } else {
                if (auto cur = vk_->clouds()) lastClouds_ = *cur;
                vk_->setClouds(std::nullopt);
            }
        }
        if (auto clouds = vk_->clouds()) {
            ImGui::PushID("clouds");
            bool ch = false;
            ch |= ImGui::SliderFloat("Coverage", &clouds->coverage, 0.f, 1.f, "%.2f");
            ch |= ImGui::SliderFloat("Density", &clouds->density, 0.f, 4.f, "%.2f");
            ch |= ImGui::SliderFloat("Bottom (m)", &clouds->bottomY, 0.f, 4000.f, "%.0f");
            ch |= ImGui::SliderFloat("Top (m)", &clouds->topY, 200.f, 8000.f, "%.0f");
            if (ch) {
                clouds->topY = std::max(clouds->topY, clouds->bottomY + 50.f);
                vk_->setClouds(*clouds);
            }
            ImGui::PopID();
        }

        // (Height fog is now folded into the unified "Fog density" + "Advanced
        // fog" controls at the top of this section — no separate toggle.)
    }

    void drawVulkanPerformance() {
        bool occ = vk_->occlusionCulling();
        if (ImGui::Checkbox("Occlusion culling", &occ)) vk_->setOcclusionCulling(occ);

        bool lod = vk_->autoLod();
        if (ImGui::Checkbox("Auto mesh LOD", &lod)) vk_->setAutoLod(lod);
        if (lod) {
            // Screen-space error budget: 0.75 px = visually lossless default;
            // raising it is the perf lever (log slider — the useful range is
            // sub-pixel to a few pixels).
            float lodErr = vk_->autoLodError();
            if (ImGui::SliderFloat("LOD error (px)", &lodErr, 0.25f, 4.f, "%.2f",
                                   ImGuiSliderFlags_Logarithmic)) {
                vk_->setAutoLodError(lodErr);
            }
            const auto s = vk_->autoLodStats();
            ImGui::TextDisabled("LOD entries 0..4: %u/%u/%u/%u/%u  chains %u (+%u queued)",
                                s.entriesPerLevel[0], s.entriesPerLevel[1], s.entriesPerLevel[2],
                                s.entriesPerLevel[3], s.entriesPerLevel[4],
                                s.chainsReady, s.chainsQueued);
        }

        if (ImGui::TreeNode("Frame timings (ms)")) {
            const auto t = vk_->lastFrameTimings();
            ImGui::Text("shade %.2f  denoise %.2f  taa %.2f", t.pathTraceMs, t.denoiseMs, t.taaMs);
            ImGui::Text("raster %.2f  resolve %.2f  edges %.2f", t.rasterGbufMs, t.gbufResolveMs, t.shadeBMs);
            ImGui::Text("overlay %.2f  dof %.2f", t.overlayMs, t.dofMs);
            ImGui::Text("cpu: scene %.2f  record %.2f  frame %.2f",
                        t.cpuEnsureSceneMs, t.cpuRecordMs, t.cpuFrameMs);
            ImGui::TreePop();
        }
    }

    void drawVulkanDebug() {
        static const char* views[] = {"Off", "Normals", "Motion", "Instance ids", "Albedo"};
        int view = vk_->hybridDebugView();
        if (view < 0 || view >= IM_ARRAYSIZE(views)) view = 0;
        if (ImGui::Combo("G-buffer view", &view, views, IM_ARRAYSIZE(views))) {
            vk_->setHybridDebugView(view);
        }

        bool toksvig = vk_->normalMapToksvig();
        if (ImGui::Checkbox("Normal-map spec AA (Toksvig)", &toksvig)) {
            vk_->setNormalMapToksvig(toksvig);
        }

        int overlay = vk_->overlayLayer();
        if (ImGui::InputInt("Overlay layer (-1 off)", &overlay)) {
            vk_->setOverlayLayer(overlay < -1 ? -1 : overlay);
        }
    }

#endif// THREEPP_WITH_VULKAN
};

// Turn-key bundle: ImGui context/backends + one window + ImGui input capture
// on the canvas. `extra` widgets (optional) appear between the FPS line and
// the collapsed renderer-settings row.
class RendererSettingsUi: public ImguiContext {

public:
    RendererSettingsUi(threepp::Canvas& canvas, threepp::Renderer& renderer,
                       std::function<void()> extra = nullptr,
                       std::string title = "Renderer")
        : ImguiContext(canvas, renderer),
          canvas_(&canvas),
          settings_(renderer),
          extra_(std::move(extra)),
          title_(std::move(title)) {

        ioCapture_.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
        ioCapture_.preventScrollEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
        ioCapture_.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
        canvas.setIOCapture(&ioCapture_);
    }

    [[nodiscard]] RendererSettings& settings() {
        return settings_;
    }

    ~RendererSettingsUi() override {
        canvas_->setIOCapture(nullptr);
    }

protected:
    void onRender() override {
        settings_.draw(title_.c_str(), extra_);
    }

private:
    threepp::Canvas* canvas_;
    RendererSettings settings_;
    std::function<void()> extra_;
    std::string title_;
    threepp::IOCapture ioCapture_;
};

#endif// THREEPP_IMGUI_RENDERER_SETTINGS_HPP
