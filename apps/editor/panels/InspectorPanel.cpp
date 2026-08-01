
#include "../EditorApp.hpp"
#include "../EditorTheme.hpp"
#include "../ImportFormats.hpp"
#include "../PanelLayout.hpp"

#include "threepp/extras/editor/AnimationConfig.hpp"
#include "threepp/extras/editor/ArticulationConfig.hpp"
#include "threepp/extras/editor/ConveyorConfig.hpp"
#include "threepp/extras/editor/GeneratorConfig.hpp"
#include "threepp/extras/editor/MaterialTextureSlots.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"

#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/lights/Light.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

using namespace threepp;
using namespace threepp::editor;

namespace {

    bool section(const char* label, bool defaultOpen = true) {

        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth;
        if (defaultOpen) flags |= ImGuiTreeNodeFlags_DefaultOpen;
        return ImGui::TreeNodeEx(label, flags);
    }

    // threepp keeps colors in the renderer's LINEAR working space; a color
    // picker must show what the user typed in, so both directions go through
    // the sRGB hex accessors rather than touching r/g/b.
    void toSrgbFloats(const Color& color, float out[3]) {

        const unsigned int hex = color.getHex();
        out[0] = static_cast<float>((hex >> 16) & 0xff) / 255.f;
        out[1] = static_cast<float>((hex >> 8) & 0xff) / 255.f;
        out[2] = static_cast<float>(hex & 0xff) / 255.f;
    }

    Color fromSrgbFloats(const float in[3]) {

        const auto channel = [](float v) {
            return static_cast<unsigned int>(std::lround(std::clamp(v, 0.f, 1.f) * 255.f));
        };
        Color color;
        color.setHex((channel(in[0]) << 16) | (channel(in[1]) << 8) | channel(in[2]));
        return color;
    }

    // --- texture thumbnails --------------------------------------------------
    // Drawn as a mosaic of filled rects sampled from the texture's CPU-side
    // pixels, rather than by uploading a GPU texture for ImGui. The renderer
    // owns the GPU-side handle of a threepp Texture and does not expose it, and
    // this way the preview looks identical on every backend for the cost of a
    // few hundred rects.
    constexpr int kThumbCells = 12;

    struct Thumbnail {
        ImU32 cells[kThumbCells * kThumbCells]{};
        bool valid = false;
    };

    ImU32 samplePixel(const Image& image, int x, int y) {

        const int channels = image.channels();
        if (channels < 1) return IM_COL32(90, 90, 90, 255);
        const std::size_t index = (static_cast<std::size_t>(y) * image.width() + x) * channels;

        float rgb[3]{0, 0, 0};
        if (image.isFloat()) {
            const auto& data = image.data<float>();
            if (index + channels > data.size()) return IM_COL32(90, 90, 90, 255);
            for (int c = 0; c < 3; ++c) {
                const float v = data[index + std::min(c, channels - 1)];
                // Reinhard + gamma: an HDR environment map is otherwise a white
                // square.
                rgb[c] = std::pow(v / (1.f + v), 1.f / 2.2f);
            }
        } else {
            const auto& data = image.data<unsigned char>();
            if (index + channels > data.size()) return IM_COL32(90, 90, 90, 255);
            for (int c = 0; c < 3; ++c) {
                rgb[c] = static_cast<float>(data[index + std::min(c, channels - 1)]) / 255.f;
            }
        }

        const auto byte = [](float v) {
            return static_cast<int>(std::lround(std::clamp(v, 0.f, 1.f) * 255.f));
        };
        return IM_COL32(byte(rgb[0]), byte(rgb[1]), byte(rgb[2]), 255);
    }

    // Keyed by uuid, but stamped with the texture's version: a uuid alone says
    // "the same texture", not "the same pixels", and an image edited in place
    // keeps its uuid while everything the mosaic sampled changes.
    struct CachedThumbnail {
        unsigned int version = 0;
        Thumbnail thumb;
    };

    std::unordered_map<std::string, CachedThumbnail>& thumbnailCache() {

        static std::unordered_map<std::string, CachedThumbnail> cache;
        return cache;
    }

    const Thumbnail& thumbnailFor(const std::shared_ptr<Texture>& texture) {

        auto& cache = thumbnailCache();
        static Thumbnail empty;

        if (!texture) return empty;

        const auto version = texture->version();
        auto it = cache.find(texture->uuid());
        if (it != cache.end() && it->second.version == version) return it->second.thumb;

        // A long session opening one texture after another would otherwise grow
        // this without bound. Dropping the lot costs one re-sample per visible
        // row on the next frame — the inspector shows six at most.
        if (cache.size() > 256) cache.clear();

        Thumbnail thumb;
        if (!texture->images().empty()) {
            const auto& image = texture->image();
            if (image.width() > 0 && image.height() > 0 && !image.compressedFormat) {
                for (int y = 0; y < kThumbCells; ++y) {
                    for (int x = 0; x < kThumbCells; ++x) {
                        const int sx = static_cast<int>(
                                (static_cast<float>(x) + 0.5f) / kThumbCells * static_cast<float>(image.width()));
                        const int sy = static_cast<int>(
                                (static_cast<float>(y) + 0.5f) / kThumbCells * static_cast<float>(image.height()));
                        thumb.cells[y * kThumbCells + x] = samplePixel(
                                image,
                                std::min<int>(sx, static_cast<int>(image.width()) - 1),
                                std::min<int>(sy, static_cast<int>(image.height()) - 1));
                    }
                }
                thumb.valid = true;
            }
        }

        // insert_or_assign, not emplace: a stale entry for this uuid is exactly
        // the case that got us here.
        return cache.insert_or_assign(texture->uuid(), CachedThumbnail{version, thumb})
                .first->second.thumb;
    }

    void drawThumbnail(const std::shared_ptr<Texture>& texture, float size) {

        auto* drawList = ImGui::GetWindowDrawList();
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        ImGui::Dummy({size, size});

        const ImU32 border = IM_COL32(70, 70, 70, 255);
        if (!texture) {
            drawList->AddRectFilled(origin, {origin.x + size, origin.y + size}, IM_COL32(30, 30, 30, 255), 3.f);
            drawList->AddRect(origin, {origin.x + size, origin.y + size}, border, 3.f);
            return;
        }

        const auto& thumb = thumbnailFor(texture);
        if (!thumb.valid) {
            // A texture whose pixels the CPU no longer holds (or a compressed
            // upload) still deserves a marker.
            drawList->AddRectFilled(origin, {origin.x + size, origin.y + size}, IM_COL32(55, 60, 70, 255), 3.f);
            drawList->AddRect(origin, {origin.x + size, origin.y + size}, border, 3.f);
            return;
        }

        const float cell = size / kThumbCells;
        for (int y = 0; y < kThumbCells; ++y) {
            for (int x = 0; x < kThumbCells; ++x) {
                const ImVec2 a{origin.x + x * cell, origin.y + y * cell};
                const ImVec2 b{a.x + cell + 0.5f, a.y + cell + 0.5f};
                drawList->AddRectFilled(a, b, thumb.cells[y * kThumbCells + x]);
            }
        }
        drawList->AddRect(origin, {origin.x + size, origin.y + size}, border, 3.f);
    }

}// namespace


void EditorApp::clearThumbnailCache() {

    thumbnailCache().clear();
}

void EditorApp::drawInspector() {

    const auto* viewport = ImGui::GetMainViewport();
    const float s = contentScale_;

    const float width = inspectorPx();
    const float top = menuHeight_ + toolbarHeight_;
    const float bottom = statusHeight_ + bottomBandPx();
    const float height = std::max(viewport->Size.y - top - bottom, 40.f * s);

    ImGui::SetNextWindowPos({viewport->Pos.x + viewport->Size.x - width, viewport->Pos.y + top});
    ImGui::SetNextWindowSize({width, height});

    if (ImGui::Begin("Inspector", nullptr, layout::panelFlags)) {

        auto* selected = selection_.get();
        if (!selected) {
            ImGui::TextColored(theme::muted(), "Nothing selected.");
            ImGui::TextWrapped("Click an object in the viewport or the hierarchy.");
            ImGui::End();
            return;
        }

        ImGui::TextColored(theme::accent(), "%s", selected->type().c_str());
        ImGui::SameLine();
        ImGui::TextColored(theme::muted(), "id %u", selected->id);
        // How many objects this one node is standing in for belongs with the
        // type and the id: it is what the node IS. The Instancing section below
        // carries the rest, but it sits under Material and Geometry, and the
        // count is the fact you want without hunting for it.
        if (auto* instanced = selected->as<InstancedMesh>()) {
            ImGui::SameLine();
            ImGui::TextColored(theme::muted(), "· %d instances",
                               static_cast<int>(instanced->count()));
        }

        // A running simulation owns the transforms; editing them from here
        // would be silently overwritten on the next step.
        const bool locked = isPlaying();
        if (locked) {
            ImGui::TextColored(theme::warning(), "Read-only while playing");
        }
        ImGui::Separator();

        if (locked) ImGui::BeginDisabled();

        drawObjectSection(*selected);
        drawTransformSection(*selected);
        drawMaterialSection(*selected);
        drawGeometrySection(*selected);
        drawGeneratorSection(*selected);
        drawInstancingSection(*selected);
        drawLightSection(*selected);
        drawCameraSection(*selected);
        drawAnimationSection(*selected);
        drawJointsSection(*selected);
        drawSplineSection(*selected);
        drawConveyorSection(*selected);
        drawScriptSection(*selected);
        drawPhysicsSection(*selected);
        drawSensorSection(*selected);

        if (locked) ImGui::EndDisabled();
    }
    ImGui::End();

    // Dragging right narrows a right-hand panel, hence the -1.
    drawSplitter("##inspectorSplit",
                 viewport->Pos.x + viewport->Size.x - width - layout::splitterThickness * s,
                 viewport->Pos.y + top, height, settings_.inspectorWidth, -1.f);
}


// ------------------------------------------------------------------- object

void EditorApp::drawObjectSection(Object3D& object) {

    if (!section("Object")) return;

    auto* target = &object;

    // Name — committed on Enter or focus loss, not per keystroke.
    char nameBuffer[128];
    const auto n = std::min(object.name.size(), sizeof(nameBuffer) - 1);
    std::memcpy(nameBuffer, object.name.data(), n);
    nameBuffer[n] = '\0';
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputText("##name", nameBuffer, sizeof(nameBuffer))) {
        object.name = nameBuffer;
    }
    if (ImGui::IsItemActivated()) nameBeforeEdit_ = object.name;
    if (ImGui::IsItemDeactivatedAfterEdit() && nameBeforeEdit_ != object.name) {
        const std::string before = nameBeforeEdit_;
        const std::string after = object.name;
        object.name = before;// let the command make the change, so undo is exact
        commands_.execute(makeProperty<std::string>(
                "Rename", {}, [target](const std::string& value) { target->name = value; },
                before, after));
        document_.setDirty(true);
    }

    const auto boolField = [&](const char* label, bool Object3D::*member, const char* action) {
        bool value = object.*member;
        if (ImGui::Checkbox(label, &value)) {
            commands_.execute(makeProperty<bool>(
                    action, {},
                    [target, member](const bool& v) { target->*member = v; },
                    !value, value));
            document_.setDirty(true);
        }
    };

    boolField("Visible", &Object3D::visible, "Toggle Visible");
    ImGui::SameLine();
    boolField("Frustum cull", &Object3D::frustumCulled, "Toggle Frustum Culling");
    boolField("Cast shadow", &Object3D::castShadow, "Toggle Cast Shadow");
    ImGui::SameLine();
    boolField("Receive shadow", &Object3D::receiveShadow, "Toggle Receive Shadow");

    int renderOrder = object.renderOrder;
    ImGui::SetNextItemWidth(-90 * contentScale_);
    const bool changed = ImGui::DragInt("Render order", &renderOrder, 0.1f);
    if (ImGui::IsItemActivated()) commands_.beginTransaction();
    if (changed) {
        commands_.execute(makeProperty<int>(
                "Render Order", "renderOrder:" + object.uuid,
                [target](const int& v) { target->renderOrder = v; },
                object.renderOrder, renderOrder));
        document_.setDirty(true);
    }
    if (ImGui::IsItemDeactivated()) commands_.endTransaction();

    ImGui::TreePop();
}


// ----------------------------------------------------------------- transform

void EditorApp::drawTransformSection(Object3D& object) {

    if (object.is<Scene>()) return;
    if (!section("Transform")) return;

    const auto before = SetTransformCommand::read(object);
    const float speed = 0.01f;

    const auto commit = [&](const SetTransformCommand::Trs& after, const char* label) {
        commands_.execute(std::make_unique<SetTransformCommand>(object, before, after, label));
        document_.setDirty(true);
    };

    ImGui::PushItemWidth(-70 * contentScale_);

    {
        float position[3]{object.position.x, object.position.y, object.position.z};
        const bool changed = ImGui::DragFloat3("Position", position, speed, 0, 0, "%.3f");
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = before;
            after.position.set(position[0], position[1], position[2]);
            commit(after, "Move");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    {
        // Degrees in the UI, radians in the model — three.js convention for
        // Euler angles, and the only readable choice for a numeric field.
        float degrees[3]{
                math::radToDeg(object.rotation.x),
                math::radToDeg(object.rotation.y),
                math::radToDeg(object.rotation.z)};
        const bool changed = ImGui::DragFloat3("Rotation", degrees, 0.5f, 0, 0, "%.2f deg");
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            Euler euler(math::degToRad(degrees[0]), math::degToRad(degrees[1]),
                        math::degToRad(degrees[2]), object.rotation.getOrder());
            auto after = before;
            after.quaternion.setFromEuler(euler);
            commit(after, "Rotate");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    {
        float scale[3]{object.scale.x, object.scale.y, object.scale.z};
        const bool changed = ImGui::DragFloat3("Scale", scale, speed, 0, 0, "%.3f");
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = before;
            // A zero scale produces a singular matrix — decompose() then hands
            // back garbage and the object can never be scaled up again.
            after.scale.set(std::abs(scale[0]) < 1e-4f ? 1e-4f : scale[0],
                            std::abs(scale[1]) < 1e-4f ? 1e-4f : scale[1],
                            std::abs(scale[2]) < 1e-4f ? 1e-4f : scale[2]);
            commit(after, "Scale");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    ImGui::PopItemWidth();

    if (ImGui::SmallButton("Reset transform")) {
        SetTransformCommand::Trs identity;
        commit(identity, "Reset Transform");
    }

    ImGui::TreePop();
}


// ------------------------------------------------------------------ material

void EditorApp::drawTextureSlot(const Object3D& owner, Material& material, const char* label,
                                const std::shared_ptr<Texture>& current,
                                const std::function<void(const std::shared_ptr<Texture>&)>& setter,
                                bool srgb) {

    ImGui::PushID(label);

    // The whole row is grouped so its screen rect can be recorded below: a file
    // dropped from the OS carries no ImGui payload, so the only way to know
    // which slot the user aimed at is to hit-test the cursor against the rows
    // this frame drew.
    ImGui::BeginGroup();

    const float thumb = 34 * contentScale_;
    drawThumbnail(current, thumb);
    ImGui::SameLine();

    ImGui::BeginGroup();
    ImGui::TextUnformatted(label);
    if (ImGui::SmallButton("Load...")) {
        // The modal resolves in a later frame, so remember which slot asked —
        // by uuid and slot name, never by pointer. Everything this row holds
        // (material, setter, texture) belongs to the current frame; the dialog
        // outlives it, and a delete or a play/stop in between would leave the
        // pointer dangling. assignTextureToSlot() re-resolves both.
        pendingTextureSlot_ = {owner.uuid, label};
        pendingDialog_ = PendingDialog::Texture;
        fileBrowser_.open("Load Texture", FileBrowser::Mode::Open, settings_.textureDir,
                          {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif"});
    }
    ImGui::SameLine();
    if (current) {
        if (ImGui::SmallButton("Clear")) {
            commands_.execute(std::make_unique<SetMaterialMapCommand>(
                    material, label, setter, current, nullptr));
            document_.setDirty(true);
        }
    } else {
        ImGui::TextColored(theme::muted(), "none");
    }
    ImGui::EndGroup();

    ImGui::EndGroup();

    // Recorded every frame the row is visible, and consumed at the end of the
    // same frame — so the raw material pointer and the setter's captured
    // material cannot outlive what they point at.
    const auto rowMin = ImGui::GetItemRectMin();
    const auto rowMax = ImGui::GetItemRectMax();
    frameTextureSlots_.push_back({{&material, setter, current, label, srgb},
                                  rowMin.x, rowMin.y, rowMax.x, rowMax.y});

    ImGui::PopID();
}

void EditorApp::drawMaterialSection(Object3D& object) {

    auto material = object.material();
    if (!material) return;
    if (!section("Material")) return;

    auto* raw = material.get();
    ImGui::TextColored(theme::muted(), "%s", raw->type().c_str());

    const auto touched = [&] {
        document_.setDirty(true);
    };

    // Vulkan refreshes an entry's MaterialDesc only when Material::version()
    // moves — the scene diff never memcmps the live floats. GL re-reads them
    // every frame, so an edit to Color or Roughness looked applied there and
    // looked ignored under Vulkan until some unrelated selection change (a
    // click in the viewport swaps the outline object) forced a rebuild. Every
    // setter goes through this, not just the call sites: undo and redo replay
    // the setter alone and owe the renderer the same bump.
    const auto sync = [raw](auto setter) {
        return [raw, setter = std::move(setter)](const auto& value) {
            setter(value);
            raw->needsUpdate();
        };
    };

    ImGui::PushItemWidth(-100 * contentScale_);

    if (auto* withColor = dynamic_cast<MaterialWithColor*>(raw)) {
        float rgb[3];
        toSrgbFloats(withColor->color, rgb);
        const bool changed = ImGui::ColorEdit3("Color", rgb);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            commands_.execute(makeProperty<Color>(
                    "Color", "color:" + raw->uuid(),
                    sync([withColor](const Color& v) { withColor->color = v; }),
                    withColor->color, fromSrgbFloats(rgb)));
            touched();
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    if (auto* withEmissive = dynamic_cast<MaterialWithEmissive*>(raw)) {
        float rgb[3];
        toSrgbFloats(withEmissive->emissive, rgb);
        const bool changed = ImGui::ColorEdit3("Emissive", rgb);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            commands_.execute(makeProperty<Color>(
                    "Emissive", "emissive:" + raw->uuid(),
                    sync([withEmissive](const Color& v) { withEmissive->emissive = v; }),
                    withEmissive->emissive, fromSrgbFloats(rgb)));
            touched();
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();

        float intensity = withEmissive->emissiveIntensity;
        const bool ch = ImGui::DragFloat("Emissive intensity", &intensity, 0.01f, 0.f, 100.f);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (ch) {
            commands_.execute(makeProperty<float>(
                    "Emissive Intensity", "emissiveIntensity:" + raw->uuid(),
                    sync([withEmissive](const float& v) { withEmissive->emissiveIntensity = v; }),
                    withEmissive->emissiveIntensity, intensity));
            touched();
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    const auto floatField = [&](const char* label, const std::string& key, float* value,
                                float speed, float min, float max,
                                const std::function<void(const float&)>& setter) {
        float edited = *value;
        const bool changed = ImGui::DragFloat(label, &edited, speed, min, max);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            commands_.execute(makeProperty<float>(label, key + raw->uuid(), sync(setter), *value, edited));
            touched();
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    if (auto* withRoughness = dynamic_cast<MaterialWithRoughness*>(raw)) {
        floatField("Roughness", "roughness:", &withRoughness->roughness, 0.005f, 0.f, 1.f,
                   [withRoughness](const float& v) { withRoughness->roughness = v; });
    }
    if (auto* withMetalness = dynamic_cast<MaterialWithMetalness*>(raw)) {
        floatField("Metalness", "metalness:", &withMetalness->metalness, 0.005f, 0.f, 1.f,
                   [withMetalness](const float& v) { withMetalness->metalness = v; });
    }

    floatField("Opacity", "opacity:", &raw->opacity, 0.005f, 0.f, 1.f,
               [raw](const float& v) { raw->opacity = v; });

    {
        bool transparent = raw->transparent;
        if (ImGui::Checkbox("Transparent", &transparent)) {
            commands_.execute(makeProperty<bool>(
                    "Transparent", {},
                    sync([raw](const bool& v) { raw->transparent = v; }),
                    !transparent, transparent));
            touched();
        }
    }

    {
        static const char* sides[] = {"Front", "Back", "Double"};
        int side = static_cast<int>(raw->side);
        if (ImGui::Combo("Side", &side, sides, IM_ARRAYSIZE(sides))) {
            commands_.execute(makeProperty<int>(
                    "Side", {},
                    sync([raw](const int& v) { raw->side = static_cast<Side>(v); }),
                    static_cast<int>(raw->side), side));
            touched();
        }
    }

    {
        // Which faces go into the shadow map, which is not the same question as
        // which faces are shaded. Unset ("Auto") means the renderer's rule: a
        // single-sided material casts from its BACK faces, and that is exactly
        // what stops it shadowing itself. A double-sided one has no far side to
        // move to, so under Auto it self-shadows into a moire. Pinning it to
        // Back cures that — at the cost of thin geometry (a plane, a leaf) no
        // longer casting at all, which is why it is a choice and not a default.
        static const char* shadowSides[] = {"Auto", "Front", "Back", "Double"};
        const auto encode = [](const std::optional<Side>& s) {
            return s ? static_cast<int>(*s) + 1 : 0;
        };
        int shadowSide = encode(raw->shadowSide);
        if (ImGui::Combo("Shadow side", &shadowSide, shadowSides, IM_ARRAYSIZE(shadowSides))) {
            commands_.execute(makeProperty<int>(
                    "Shadow side", {},
                    sync([raw](const int& v) {
                        if (v == 0) {
                            raw->shadowSide.reset();
                        } else {
                            raw->shadowSide = static_cast<Side>(v - 1);
                        }
                    }),
                    encode(raw->shadowSide), shadowSide));
            touched();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Which faces are drawn into the shadow map.\n"
                              "Auto: back faces for a single-sided material, both for Double.\n"
                              "A Double-sided material self-shadows under Auto (moire banding);\n"
                              "set Back to clear it. Thin surfaces then stop casting a shadow.");
        }
    }

    if (auto* withWireframe = dynamic_cast<MaterialWithWireframe*>(raw)) {
        bool wireframe = withWireframe->wireframe;
        if (ImGui::Checkbox("Wireframe", &wireframe)) {
            commands_.execute(makeProperty<bool>(
                    "Wireframe", {},
                    sync([withWireframe](const bool& v) { withWireframe->wireframe = v; }),
                    !wireframe, wireframe));
            touched();
        }
    }

    if (auto* withFlat = dynamic_cast<MaterialWithFlatShading*>(raw)) {
        ImGui::SameLine();
        bool flat = withFlat->flatShading;
        if (ImGui::Checkbox("Flat shading", &flat)) {
            commands_.execute(makeProperty<bool>(
                    "Flat Shading", {},
                    sync([withFlat](const bool& v) { withFlat->flatShading = v; }),
                    !flat, flat));
            touched();
        }
    }

    ImGui::PopItemWidth();

    // Collapsed until asked for — the section is long and most objects never
    // need it. A drop aimed at one of these rows implies it is already open;
    // a drop anywhere else is resolved by file name instead, so the feature
    // does not depend on this being expanded.
    if (openTextureSectionOnce_) {
        ImGui::SetNextItemOpen(true);
        openTextureSectionOnce_ = false;
    }
    if (ImGui::TreeNodeEx("Textures", ImGuiTreeNodeFlags_SpanAvailWidth)) {

        // Same list the file-drop handler resolves against — see
        // MaterialTextureSlots.hpp for why it is not spelled out twice.
        for (const auto& slot : textureSlotsOf(*raw)) {
            drawTextureSlot(object, *raw, slot.name.c_str(), slot.current, slot.set, slot.srgb);
        }

        ImGui::TreePop();
    }

    ImGui::TreePop();
}


// ------------------------------------------------------------------ geometry

void EditorApp::drawGeometrySection(Object3D& object) {

    auto geometry = object.geometry();
    if (!geometry) return;
    if (!section("Geometry", false)) return;

    ImGui::TextColored(theme::muted(), "%s", geometry->type().c_str());

    int vertices = 0;
    if (const auto* position = geometry->getAttribute<float>("position")) {
        vertices = position->count();
    }
    const auto* index = geometry->getIndex();
    const int indices = index ? index->count() : 0;

    ImGui::Text("Vertices  %d", vertices);
    ImGui::Text("Indices   %d", indices);
    ImGui::Text("Triangles %d", (indices ? indices : vertices) / 3);

    ImGui::TreePop();
}


// --------------------------------------------------------------- generator

void EditorApp::drawGeneratorSection(Object3D& object) {

    // Offered on the SCENE by default — a generated scene's rule belongs to the
    // scene — and on a Group, for a scene wanting several independently
    // re-runnable ones. Not on a mesh: generated content goes in a child, and a
    // mesh with a generated child is a confusing thing to have built by accident.
    const bool isScene = &object == &document_.scene();
    // Not on generated output: a generator there would be wiped by the next run of
    // the generator above it, which is a trap rather than a feature.
    const bool isGeneratedOutput =
            object.userData.find(GeneratorConfig::generatedKey) != object.userData.end();
    const bool eligible = !isGeneratedOutput && (isScene || object.as<Group>() != nullptr);
    const auto stored = GeneratorConfig::read(object);
    if (!eligible && !stored) return;
    if (!section("Generator", stored.has_value())) return;

    auto* target = &object;
    const auto config = stored.value_or(GeneratorConfig{});
    // Same shape every other config section uses: one undoable property write per
    // edit, coalesced by a per-object merge key so typing is not 400 undo steps.
    const auto commit = [&](GeneratorConfig after, const std::string& label) {
        commands_.execute(makeProperty<GeneratorConfig>(
                label, "generator:" + object.uuid,
                [target](const GeneratorConfig& value) { value.write(*target); },
                config, std::move(after)));
        document_.setDirty(true);
    };

    if (!stored) {
        ImGui::TextColored(theme::muted(), "Build this %s with a Python script.",
                           isScene ? "scene" : "group");
        if (ImGui::Button("Add generator script")) {
            GeneratorConfig fresh;
            fresh.source = generatorTemplate();
            commit(fresh, "Add Generator");
        }
        ImGui::TreePop();
        return;
    }

    // The source itself. A modest box: this is for reading and small edits, and
    // anything longer wants the Script Editor's room (below).
    ImGui::TextColored(theme::muted(), "%d lines",
                       1 + static_cast<int>(std::count(config.source.begin(),
                                                       config.source.end(), '\n')));
    std::string buffer = config.source;
    buffer.resize(std::max<std::size_t>(buffer.size() + 4096, 8192));
    const float height = ImGui::GetTextLineHeight() * 12.f;
    if (ImGui::InputTextMultiline("##generatorSource", buffer.data(), buffer.size(),
                                  {-1.f, height})) {
        auto edited = config;
        edited.source = buffer.c_str();
        commit(edited, "Edit Generator");
    }

    const bool playing = isPlaying();
    if (playing) ImGui::BeginDisabled();
    if (ImGui::Button("Regenerate")) {
        // Deferred by one frame: regenerate replaces the node this panel may be
        // drawing from, and the selection re-resolve that follows must not run
        // inside the ImGui tree that is reading it.
        pendingRegenerate_ = object.uuid;
    }
#ifdef THREEPP_EDITOR_WITH_PYTHON
    // The room the box above does not have, with completion for `import threepp`:
    // the same scratch-file round trip an inline behaviour script gets, and a save
    // re-runs the generator so the loop is edit-save-look.
    ImGui::SameLine();
    const bool editing = externalEditActive(object);
    if (editing) ImGui::BeginDisabled();
    if (ImGui::Button("Edit in VS Code")) {
        startExternalEdit(object, ExternalEditKind::Generator);
    }
    if (editing) ImGui::EndDisabled();
    if (editing) {
        ImGui::SameLine();
        if (ImGui::Button("Stop editing")) stopExternalEdit("closed from the inspector");
    }
#endif
    if (playing) ImGui::EndDisabled();
    ImGui::SameLine();
    // Deferred with the same one-frame hop as Regenerate: it removes the output
    // node, which this panel and the hierarchy are mid-read of.
    if (ImGui::Button("Clear")) pendingGeneratorClear_ = object.uuid;
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Removes the script and the objects it generated.");
    }

    if (auto* generated = GeneratorConfig::generatedChild(object)) {
        ImGui::TextColored(theme::muted(), "Output: %d object(s)",
                           static_cast<int>(generated->children.size()));
    } else {
        ImGui::TextColored(theme::muted(), "No output yet - press Regenerate.");
    }
    // The rule that makes the script the source of truth, said where it bites.
    ImGui::TextColored(theme::muted(), "Regenerating replaces the output; edits to it are lost.");
#ifndef THREEPP_EDITOR_WITH_PYTHON
    ImGui::TextColored(theme::warning(), "Built without Python - saved, but cannot run.");
#endif

    ImGui::TreePop();
}


// --------------------------------------------------------------- instancing

void EditorApp::drawInstancingSection(Object3D& object) {

    auto* instanced = object.as<InstancedMesh>();
    if (!instanced) return;
    if (!section("Instancing")) return;

    // The count is what makes one draw call stand in for N objects, so it leads.
    // It is read-only here: the instance transforms are the payload, and there is
    // no authoring verb for them yet (picking one is all this pass offers).
    ImGui::Text("Instances %d", static_cast<int>(instanced->count()));

    if (selectedInstance_) {
        ImGui::TextColored(theme::accent(), "Picked instance %d", *selectedInstance_);
        const auto size = instanceBox_.getSize();
        ImGui::TextColored(theme::muted(), "Bounds %.2f x %.2f x %.2f", size.x, size.y, size.z);
    } else {
        ImGui::TextColored(theme::muted(), "Click an instance in the viewport to single it out.");
    }
    ImGui::TextColored(theme::muted(), "The gizmo moves all %d together.",
                       static_cast<int>(instanced->count()));

    ImGui::TreePop();
}


// -------------------------------------------------------------------- lights

void EditorApp::drawLightSection(Object3D& object) {

    auto* light = object.as<Light>();
    if (!light) return;
    if (!section("Light")) return;

    ImGui::PushItemWidth(-100 * contentScale_);

    {
        float rgb[3];
        toSrgbFloats(light->color, rgb);
        const bool changed = ImGui::ColorEdit3("Color", rgb);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            commands_.execute(makeProperty<Color>(
                    "Light Color", "lightColor:" + object.uuid,
                    [light](const Color& v) { light->color = v; },
                    light->color, fromSrgbFloats(rgb)));
            document_.setDirty(true);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    const auto floatField = [&](const char* label, const std::string& key, float* value,
                                float speed, float min, float max,
                                const std::function<void(const float&)>& setter) {
        float edited = *value;
        const bool changed = ImGui::DragFloat(label, &edited, speed, min, max);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            commands_.execute(makeProperty<float>(label, key + object.uuid, setter, *value, edited));
            document_.setDirty(true);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    floatField("Intensity", "intensity:", &light->intensity, 0.02f, 0.f, 1000.f,
               [light](const float& v) { light->intensity = v; });

    if (auto* point = object.as<PointLight>()) {
        floatField("Distance", "distance:", &point->distance, 0.05f, 0.f, 1000.f,
                   [point](const float& v) { point->distance = v; });
        floatField("Decay", "decay:", &point->decay, 0.01f, 0.f, 4.f,
                   [point](const float& v) { point->decay = v; });
    }

    if (auto* spot = object.as<SpotLight>()) {
        floatField("Distance", "distance:", &spot->distance, 0.05f, 0.f, 1000.f,
                   [spot](const float& v) { spot->distance = v; });
        float angleDegrees = math::radToDeg(spot->angle);
        const bool changed = ImGui::DragFloat("Angle", &angleDegrees, 0.2f, 1.f, 89.f, "%.1f deg");
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            commands_.execute(makeProperty<float>(
                    "Angle", "angle:" + object.uuid,
                    [spot](const float& v) { spot->angle = v; },
                    spot->angle, math::degToRad(angleDegrees)));
            document_.setDirty(true);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();

        floatField("Penumbra", "penumbra:", &spot->penumbra, 0.005f, 0.f, 1.f,
                   [spot](const float& v) { spot->penumbra = v; });
        floatField("Decay", "decay:", &spot->decay, 0.01f, 0.f, 4.f,
                   [spot](const float& v) { spot->decay = v; });
    }

    if (auto* hemisphere = object.as<HemisphereLight>()) {
        float rgb[3];
        toSrgbFloats(hemisphere->groundColor, rgb);
        if (ImGui::ColorEdit3("Ground color", rgb)) {
            commands_.execute(makeProperty<Color>(
                    "Ground Color", "groundColor:" + object.uuid,
                    [hemisphere](const Color& v) { hemisphere->groundColor = v; },
                    hemisphere->groundColor, fromSrgbFloats(rgb)));
            document_.setDirty(true);
        }
    }

    ImGui::PopItemWidth();

    auto* target = &object;
    bool castShadow = object.castShadow;
    if (ImGui::Checkbox("Cast shadow##light", &castShadow)) {
        commands_.execute(makeProperty<bool>(
                "Toggle Cast Shadow", {},
                [target](const bool& v) { target->castShadow = v; },
                !castShadow, castShadow));
        document_.setDirty(true);
    }

    ImGui::TreePop();
}


// -------------------------------------------------------------------- camera

void EditorApp::drawCameraSection(Object3D& object) {

    auto* camera = object.as<PerspectiveCamera>();
    if (!camera) return;
    if (!section("Camera")) return;

    ImGui::PushItemWidth(-100 * contentScale_);

    const auto floatField = [&](const char* label, const std::string& key, float* value,
                                float speed, float min, float max,
                                const std::function<void(const float&)>& setter) {
        float edited = *value;
        const bool changed = ImGui::DragFloat(label, &edited, speed, min, max);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            commands_.execute(makeProperty<float>(label, key + object.uuid, setter, *value, edited));
            document_.setDirty(true);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    floatField("FOV", "fov:", &camera->fov, 0.2f, 1.f, 179.f,
               [camera](const float& v) {
                   camera->fov = v;
                   camera->updateProjectionMatrix();
               });
    floatField("Near", "near:", &camera->nearPlane, 0.005f, 0.001f, 1000.f,
               [camera](const float& v) {
                   camera->nearPlane = v;
                   camera->updateProjectionMatrix();
               });
    floatField("Far", "far:", &camera->farPlane, 1.f, 0.01f, 100000.f,
               [camera](const float& v) {
                   camera->farPlane = v;
                   camera->updateProjectionMatrix();
               });

    ImGui::PopItemWidth();
    ImGui::TreePop();
}


// ----------------------------------------------------------------- animation

void EditorApp::drawAnimationSection(Object3D& object) {

    // Clips live on the import root (that is where every loader puts them),
    // so the section appears there — which is also what the picker selects.
    if (object.animations.empty()) return;
    if (!section("Animation", false)) return;

    auto* target = &object;
    auto config = AnimationConfig::read(object).value_or(AnimationConfig{});
    const auto before = config;

    const auto commit = [&](AnimationConfig after, const char* label) {
        commands_.execute(makeProperty<AnimationConfig>(
                label, "animation:" + object.uuid,
                [target](const AnimationConfig& value) { value.write(*target); },
                before, after));
        document_.setDirty(true);
    };

    const auto& clips = object.animations;

    ImGui::PushItemWidth(-100 * contentScale_);

    {
        int index = 0;
        for (int i = 0; i < static_cast<int>(clips.size()); ++i) {
            if (clips[i]->name() == config.clip) index = i;
        }
        char current[160];
        std::snprintf(current, sizeof(current), "%s (%.2fs)",
                      clips[index]->name().c_str(), clips[index]->getDuration());
        if (ImGui::BeginCombo("Clip", current)) {
            for (int i = 0; i < static_cast<int>(clips.size()); ++i) {
                char label[160];
                std::snprintf(label, sizeof(label), "%s (%.2fs)",
                              clips[i]->name().c_str(), clips[i]->getDuration());
                if (ImGui::Selectable(label, i == index)) {
                    auto after = config;
                    after.clip = clips[i]->name();
                    commit(after, "Animation Clip");
                }
            }
            ImGui::EndCombo();
        }
    }

    bool autoplay = config.autoplay;
    if (ImGui::Checkbox("Play on Start", &autoplay)) {
        auto after = config;
        after.autoplay = autoplay;
        commit(after, autoplay ? "Enable Autoplay" : "Disable Autoplay");
    }

    bool loop = config.loop;
    if (ImGui::Checkbox("Loop", &loop)) {
        auto after = config;
        after.loop = loop;
        commit(after, "Animation Loop");
    }

    {
        float speed = config.speed;
        const bool changed = ImGui::DragFloat("Speed", &speed, 0.01f, 0.05f, 5.f, "%.2fx");
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.speed = speed;
            commit(after, "Animation Speed");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    ImGui::PopItemWidth();
    ImGui::Spacing();

    const bool previewing = isPreviewing(object);
    if (ImGui::Button(previewing ? "Stop Preview" : "Preview", {110 * contentScale_, 0})) {
        if (previewing) {
            stopAnimationPreview();
        } else {
            startAnimationPreview(object, config.clip, config.loop, config.speed);
        }
    }
    // Re-query rather than reusing the flag above: the button just ran, and
    // either branch changes what it described. isPreviewing() is also what
    // makes the dereference safe — it is false whenever there is no preview.
    if (isPreviewing(object)) {
        ImGui::SameLine();
        ImGui::TextColored(theme::playing(), "playing %s", animPreview_->clip.c_str());
    }

    ImGui::TextColored(theme::muted(), "Stored in userData[\"animation\"]");

    ImGui::TreePop();
}


// ------------------------------------------------------------- articulation

void EditorApp::drawArticulationBlock(Object3D& object, Robot& robot) {

    auto* target = &object;
    auto config = ArticulationConfig::read(object).value_or(ArticulationConfig{});
    const auto before = config;

    const auto commit = [&](ArticulationConfig after, const char* label) {
        commands_.execute(makeProperty<ArticulationConfig>(
                label, "articulation:" + object.uuid,
                [target](const ArticulationConfig& value) { value.write(*target); },
                before, after));
        document_.setDirty(true);
    };

    ImGui::Spacing();

    bool simulate = config.enabled;
    if (ImGui::Checkbox("Simulate", &simulate)) {
        auto after = config;
        after.enabled = simulate;
        commit(after, simulate ? "Simulate Robot" : "Stop Simulating Robot");
        config.enabled = simulate;// so the widgets below reflect the toggle this frame
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Play as a PhysX reduced-coordinate articulation.\n"
                          "Colliders are primitive/bbox approximations, not the visual meshes.");
    }

    if (!config.enabled) return;

    ImGui::PushItemWidth(-110 * contentScale_);

    bool fixedBase = config.fixedBase;
    if (ImGui::Checkbox("Fixed Base", &fixedBase)) {
        auto after = config;
        after.fixedBase = fixedBase;
        commit(after, "Articulation Fixed Base");
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("On: the base is bolted to the world (an arm).\n"
                          "Off: the base floats free (a quadruped, a drone).");
    }

    const auto floatField = [&](const char* label, float value, float speed, float min, float max,
                                void (*assign)(ArticulationConfig&, float), const char* action,
                                ImGuiSliderFlags flags = 0) {
        float edited = value;
        const bool changed = ImGui::DragFloat(label, &edited, speed, min, max, "%.3f", flags);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            assign(after, edited);
            commit(after, action);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    floatField(
            "Stiffness", config.stiffness, 1.f, 0.f, 100000.f,
            [](ArticulationConfig& c, float v) { c.stiffness = v; }, "Articulation Stiffness",
            ImGuiSliderFlags_Logarithmic);
    floatField(
            "Damping", config.damping, 0.5f, 0.f, 10000.f,
            [](ArticulationConfig& c, float v) { c.damping = v; }, "Articulation Damping",
            ImGuiSliderFlags_Logarithmic);
    floatField(
            "Max Force", config.maxForce, 100.f, 0.f, 1e7f,
            [](ArticulationConfig& c, float v) { c.maxForce = v; }, "Articulation Max Force",
            ImGuiSliderFlags_Logarithmic);

    bool selfCollision = config.selfCollision;
    if (ImGui::Checkbox("Self Collision", &selfCollision)) {
        auto after = config;
        after.selfCollision = selfCollision;
        commit(after, "Articulation Self Collision");
    }

    {
        int edited = config.iterations;
        const bool changed = ImGui::DragInt("Iterations", &edited, 0.2f, 1, 255);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.iterations = edited;
            commit(after, "Articulation Iterations");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    floatField(
            "Density", config.density, 5.f, 1.f, 100000.f,
            [](ArticulationConfig& c, float v) { c.density = v; }, "Articulation Density");

    ImGui::PopItemWidth();

#ifndef THREEPP_EDITOR_WITH_PHYSX
    ImGui::TextColored(theme::muted(), "Built without PhysX - authored and saved, not simulated.");
#endif
}


// -------------------------------------------------------------------- joints

void EditorApp::drawJointsSection(Object3D& object) {

    auto* robot = object.as<Robot>();
    if (!robot) return;
    if (!section("Robot")) return;

    // Collision hulls are wireframe stand-ins drawn on top of the visual
    // meshes; useful when checking a URDF, noise the rest of the time.
    {
        auto config = RobotConfig::read(object).value_or(RobotConfig{});
        bool show = config.showColliders;
        if (ImGui::Checkbox("Show Colliders", &show)) {
            auto* target = robot;
            commands_.execute(makeProperty<bool>(
                    show ? "Show Colliders" : "Hide Colliders", "colliders:" + object.uuid,
                    [target](const bool& value) {
                        target->showColliders(value);
                        auto updated = RobotConfig::read(*target).value_or(RobotConfig{});
                        updated.showColliders = value;
                        if (!updated.urdf.empty()) updated.write(*target);
                    },
                    config.showColliders, show));
            document_.setDirty(true);
        }
    }

    // Whether Play simulates this robot as a PhysX articulation (joints become
    // real DOFs, gravity acts, and the joint sensors have something to read) or
    // leaves it a kinematic prop. Same undoable command pattern as the Physics
    // section, and PhysX-free authoring — the block draws without the SDK.
    drawArticulationBlock(object, *robot);

    if (robot->numDOF() == 0) {
        ImGui::TextColored(theme::muted(), "No articulated joints.");
        ImGui::TreePop();
        return;
    }

    ImGui::Spacing();

    const auto info = robot->getArticulatedJointInfo();

    ImGui::PushItemWidth(-110 * contentScale_);

    for (std::size_t i = 0; i < robot->numDOF(); ++i) {

        const bool revolute = info[i].type == Robot::JointType::Revolute;
        // Revolute joints read in degrees, prismatic in metres. The document
        // always stores radians; only this widget converts.
        const auto range = robot->getJointRange(i, revolute);

        // An unlimited joint (a continuous revolute, say) reports an infinite
        // range, which a slider cannot use — fall back to a turn either way.
        const float fallbackMin = revolute ? -360.f : -1.f;
        const float fallbackMax = revolute ? 360.f : 1.f;
        const float min = std::isfinite(range.min) ? range.min : fallbackMin;
        const float max = std::isfinite(range.max) ? range.max : fallbackMax;

        float shown = robot->getJointValue(i, revolute);

        const auto label = info[i].name.empty() ? "joint " + std::to_string(i + 1) : info[i].name;
        const bool changed = ImGui::SliderFloat(label.c_str(), &shown, min, max,
                                                revolute ? "%.1f deg" : "%.3f m");
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto* target = robot;
            const float before = robot->getJointValue(i);
            const float after = revolute ? math::degToRad(shown) : shown;
            commands_.execute(makeProperty<float>(
                    "Joint " + label, "joint:" + object.uuid + ":" + std::to_string(i),
                    [this, target, i](const float& value) { setJointValue(*target, i, value); },
                    before, after));
            document_.setDirty(true);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    ImGui::PopItemWidth();
    ImGui::Spacing();

    if (ImGui::Button("Home", {110 * contentScale_, 0})) {
        // One undo entry for the whole pose, not one per joint.
        commands_.beginTransaction();
        for (std::size_t i = 0; i < robot->numDOF(); ++i) {
            auto* target = robot;
            const float before = robot->getJointValue(i);
            if (before == 0.f) continue;
            commands_.execute(makeProperty<float>(
                    "Home Joints", "home:" + object.uuid + ":" + std::to_string(i),
                    [this, target, i](const float& value) { setJointValue(*target, i, value); },
                    before, 0.f));
        }
        commands_.endTransaction();
        document_.setDirty(true);
    }
    ImGui::SameLine();
    ImGui::TextColored(theme::muted(), "%zu DOF", robot->numDOF());

    if (const auto config = RobotConfig::read(object)) {
        ImGui::TextColored(theme::muted(), "%s",
                           std::filesystem::path(config->urdf).filename().string().c_str());
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", config->urdf.c_str());
    }

    ImGui::TreePop();
}


// -------------------------------------------------------------------- script

void EditorApp::drawScriptSection(Object3D& object) {

    if (!section("Script", false)) return;

    auto* target = &object;
    const auto config = ScriptConfig::read(object).value_or(ScriptConfig{});

    const auto commit = [&](ScriptConfig after, const std::string& label) {
        commands_.execute(makeProperty<ScriptConfig>(
                label, "script:" + object.uuid,
                [target](const ScriptConfig& value) { value.write(*target); },
                config, std::move(after)));
        document_.setDirty(true);
    };

    // --- where the code is ---------------------------------------------------
    // Two forms, never both: a .py file, or source stored in this scene and
    // edited in the Script Editor tab.
    const float buttonWidth = 90 * contentScale_;

    if (config.isInline()) {
        if (ImGui::Button("Edit...", {buttonWidth, 0})) openScriptEditor(object);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Open the Script Editor on this source.");
    } else {
        if (ImGui::Button(config.path.empty() ? "Attach..." : "Change...", {buttonWidth, 0})) {
            // The dialog resolves a frame or more later; remember the object by
            // uuid, since a play/stop in between replaces the whole graph.
            scriptTargetUuid_ = object.uuid;
            pendingDialog_ = PendingDialog::Script;
            fileBrowser_.open("Attach Script", FileBrowser::Mode::Open,
                              settings_.scriptDir.empty() ? assetDir_ : std::filesystem::path(settings_.scriptDir),
                              formats::scripts());
        }
    }

    if (config.empty()) {
        ImGui::SameLine();
        if (ImGui::Button("New Inline Script", {buttonWidth * 1.9f, 0})) {
            setInlineScript(object, inlineScriptTemplate(), "New Inline Script");
            openScriptEditor(object);
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Write the script into this scene instead of a file,\n"
                              "and open it in the Script Editor.");
        }
        ImGui::TextColored(theme::muted(), "No script attached.");
        ImGui::TextWrapped("A class with any of start(obj) / update(dt) / stop() — "
                           "in a .py file named after it, or inline in this scene.");
        ImGui::TreePop();
        return;
    }

    ImGui::SameLine();
    if (ImGui::Button("Clear", {buttonWidth * 0.7f, 0})) {
        commit(ScriptConfig{}, "Clear Script");
    }

    // --- the same script, in a real code editor ------------------------------
    // A file is simply handed over (every Play recompiles it, so it is already
    // hot). Inline source has no file, so one is exported and watched — see
    // apps/editor/ExternalScriptEdit.cpp.
    if (externalEditActive(object)) {
        if (ImGui::Button("Stop external edit", {buttonWidth * 1.9f, 0})) {
            stopExternalEdit("stopped from the inspector");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Stop watching the exported file and delete it.\n"
                              "Everything saved so far is already in the scene.");
        }
    } else if (ImGui::Button("Edit in VS Code", {buttonWidth * 1.9f, 0})) {
        if (config.isInline()) {
            startExternalEdit(object);
        } else {
            openScriptFileExternally(config.path);
        }
    }
    if (!externalEditActive(object) && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
        ImGui::SetTooltip(config.isInline()
                                  ? "Export the source to a file, open it in VS Code, and take\n"
                                    "every save back into the scene while the session is live.\n"
                                    "A .vscode/settings.json is written beside it so Pylance\n"
                                    "completes `import threepp`."
                                  : "Open the script's folder in VS Code, with a\n"
                                    ".vscode/settings.json that completes `import threepp`.\n"
                                    "Press Play to run whatever you saved.");
    }

    if (config.isInline()) {
        // No file name to show, so say what it is instead. The first line of
        // the source on hover is usually the template's header comment or the
        // class, which is enough to tell two objects' scripts apart.
        ImGui::TextColored(theme::accent(), "(inline script)");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", config.source.substr(0, config.source.find('\n')).c_str());
        }
    } else {
        // Filename with the full path on hover, same as the Robot section: the
        // inspector is too narrow for a real path and too useful to truncate.
        ImGui::TextColored(theme::accent(), "%s",
                           std::filesystem::path(config.path).filename().string().c_str());
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", config.path.c_str());
    }

    // --- what went wrong last time ------------------------------------------
#ifdef THREEPP_EDITOR_WITH_PYTHON
    if (scripts_) {
        const auto error = scripts_->errorFor(object.uuid);
        if (!error.empty()) {
            ImGui::Spacing();
            ImGui::PushStyleColor(ImGuiCol_Text, theme::danger());
            // The traceback's first line is the summary; the rest is on hover,
            // because a section is not a console.
            const auto firstLine = error.substr(0, error.find('\n'));
            ImGui::TextWrapped("%s", firstLine.c_str());
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", error.c_str());
        }
    }
#endif

    ImGui::Spacing();

    std::error_code ec;
    // Inline source is always "there"; only a file can have moved away.
    const bool onDisk = config.isInline() || std::filesystem::exists(config.path, ec);
    if (!onDisk) {
        ImGui::TextColored(theme::danger(), "File not found.");
    }

    // --- parameters ----------------------------------------------------------
    // Rendered from the class when Python can tell us the types, and read-only
    // from the document otherwise, so the values are never silently dropped.
    const auto drawStored = [&] {
        if (config.fields.empty()) {
            ImGui::TextColored(theme::muted(), "No stored parameters.");
            return;
        }
        for (const auto& field : config.fields) {
            ImGui::TextColored(theme::muted(), "%s", field.name.c_str());
            ImGui::SameLine(140 * contentScale_);
            ImGui::TextUnformatted(field.value.c_str());
        }
    };

    bool drewFields = false;

#ifdef THREEPP_EDITOR_WITH_PYTHON
    if (onDisk) {
        const auto& inspection =
                config.isInline()
                        ? inspectScriptSource(object.uuid,
                                              object.name.empty() ? object.type() : object.name,
                                              config.source)
                        : inspectScript(config.path);
        if (!inspection.error.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Text, theme::danger());
            ImGui::TextWrapped("%s", inspection.error.substr(0, inspection.error.find('\n')).c_str());
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", inspection.error.c_str());
            ImGui::Spacing();
        } else {
            ImGui::TextColored(theme::muted(), "class %s", inspection.className.c_str());

            // Values the class no longer exposes are pruned on the next edit,
            // so a renamed attribute does not linger in the document forever.
            std::vector<std::string> live;
            live.reserve(inspection.fields.size());
            for (const auto& field : inspection.fields) live.push_back(field.name);

            const auto edited = [&](const std::string& name, const std::string& value) {
                auto after = config;
                after.retainFields(live);
                after.setField(name, value);
                return after;
            };

            ImGui::PushItemWidth(-120 * contentScale_);
            for (const auto& field : inspection.fields) {

                // The document's override if there is one, the class attribute
                // otherwise — which is exactly what the script would see.
                const auto stored = config.field(field.name).value_or(field.defaultValue);
                const auto label = field.name;

                switch (field.type) {
                    case ScriptField::Type::Bool: {
                        bool value = ScriptConfig::toBool(stored);
                        if (ImGui::Checkbox(label.c_str(), &value)) {
                            commit(edited(field.name, ScriptConfig::toText(value)), "Script " + label);
                        }
                        break;
                    }
                    case ScriptField::Type::Int: {
                        int value = ScriptConfig::toInt(stored);
                        const bool changed = ImGui::DragInt(label.c_str(), &value, 0.1f);
                        if (ImGui::IsItemActivated()) commands_.beginTransaction();
                        if (changed) {
                            commit(edited(field.name, ScriptConfig::toText(value)), "Script " + label);
                        }
                        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
                        break;
                    }
                    case ScriptField::Type::Float: {
                        float value = ScriptConfig::toFloat(stored);
                        const bool changed = ImGui::DragFloat(label.c_str(), &value, 0.01f);
                        if (ImGui::IsItemActivated()) commands_.beginTransaction();
                        if (changed) {
                            commit(edited(field.name, ScriptConfig::toText(value)), "Script " + label);
                        }
                        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
                        break;
                    }
                    case ScriptField::Type::String: {
                        char buffer[256];
                        const auto n = std::min(stored.size(), sizeof(buffer) - 1);
                        std::memcpy(buffer, stored.data(), n);
                        buffer[n] = '\0';
                        // Committed on Enter or focus loss, not per keystroke —
                        // otherwise every letter is an undo entry.
                        ImGui::InputText(label.c_str(), buffer, sizeof(buffer));
                        if (ImGui::IsItemDeactivatedAfterEdit() && stored != buffer) {
                            commit(edited(field.name, ScriptConfig::sanitized(buffer)), "Script " + label);
                        }
                        break;
                    }
                }
            }
            ImGui::PopItemWidth();

            if (inspection.fields.empty()) {
                ImGui::TextColored(theme::muted(), "No exposed parameters.");
            }
            drewFields = true;
        }
    }
#else
    ImGui::TextColored(theme::muted(), "Built without Python scripting.");
    ImGui::TextWrapped("Scripts are still authored and saved; a build with "
                       "Python will run them.");
    ImGui::Spacing();
#endif

    if (!drewFields) drawStored();

    ImGui::TextColored(theme::muted(), "Stored in userData[\"%s\"]",
                       config.isInline() ? ScriptConfig::sourceKey : ScriptConfig::pathKey);

    ImGui::TreePop();
}


// ------------------------------------------------------------------- physics

void EditorApp::drawPhysicsSection(Object3D& object) {

    if (object.is<Scene>()) return;
    if (!section("Physics", false)) return;

#ifndef THREEPP_EDITOR_WITH_PHYSX
    ImGui::TextColored(theme::muted(), "Built without PhysX.");
    ImGui::TextWrapped("Settings are still authored and saved; a build with the "
                       "PhysX SDK will simulate them.");
    ImGui::Spacing();
#endif

    auto* target = &object;
    auto config = PhysicsConfig::read(object).value_or(PhysicsConfig{});
    const auto before = config;

    const auto commit = [&](PhysicsConfig after, const char* label) {
        commands_.execute(makeProperty<PhysicsConfig>(
                label, "physics:" + object.uuid,
                [target](const PhysicsConfig& value) { value.write(*target); },
                before, after));
        document_.setDirty(true);
    };

    bool enabled = config.enabled;
    if (ImGui::Checkbox("Enabled", &enabled)) {
        auto after = config;
        after.enabled = enabled;
        // A body that has never been configured starts as a dynamic box, which
        // is what "make this fall" means to almost everyone.
        commit(after, enabled ? "Enable Physics" : "Disable Physics");
    }

    if (!config.enabled) {
        ImGui::TreePop();
        return;
    }

    ImGui::PushItemWidth(-100 * contentScale_);

    {
        static const char* bodies[] = {"Static", "Dynamic", "Kinematic", "Soft"};
        int body = static_cast<int>(config.body);
        if (ImGui::Combo("Body", &body, bodies, IM_ARRAYSIZE(bodies))) {
            auto after = config;
            after.body = static_cast<PhysicsConfig::Body>(body);
            commit(after, "Physics Body Type");
        }
    }

    const bool soft = config.body == PhysicsConfig::Body::Soft;

    // A soft body's collider is always a tetrahedral volume cooked from the
    // mesh, so the shape picker has nothing to offer it.
    if (!soft) {
        static const char* shapes[] = {"Auto", "Box", "Sphere", "Capsule",
                                       "Convex", "TriMesh", "Convex Pieces"};
        int shape = static_cast<int>(config.shape);
        if (ImGui::Combo("Shape", &shape, shapes, IM_ARRAYSIZE(shapes))) {
            auto after = config;
            after.shape = static_cast<PhysicsConfig::Shape>(shape);
            commit(after, "Physics Shape");
        }
        if (config.shape == PhysicsConfig::Shape::TriMesh &&
            config.body != PhysicsConfig::Body::Static) {
            ImGui::TextColored(theme::warning(), "TriMesh is static-only");
        }

        // Beside the shape, because that is what it changes: a trigger's shape
        // is cooked as an overlap volume instead of a collider. Hidden for a
        // soft body (whose collider is the cooked tet volume — see below), and
        // the key still round-trips, so the tick survives a trip through Soft.
        bool trigger = config.trigger;
        if (ImGui::Checkbox("Trigger", &trigger)) {
            auto after = config;
            after.trigger = trigger;
            commit(after, trigger ? "Make Trigger Volume" : "Clear Trigger Volume");
        }
    }

    const auto floatField = [&](const char* label, float value, float speed, float min, float max,
                                void (*assign)(PhysicsConfig&, float), const char* action,
                                ImGuiSliderFlags flags = 0) {
        float edited = value;
        const bool changed = ImGui::DragFloat(label, &edited, speed, min, max, "%.3f", flags);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            assign(after, edited);
            commit(after, action);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    const auto intField = [&](const char* label, int value, float speed, int min, int max,
                              void (*assign)(PhysicsConfig&, int), const char* action) {
        int edited = value;
        const bool changed = ImGui::DragInt(label, &edited, speed, min, max);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            assign(after, edited);
            commit(after, action);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    if (config.body == PhysicsConfig::Body::Dynamic || soft) {
        floatField(
                "Mass (kg)", config.mass, 0.05f, 0.001f, 10000.f,
                [](PhysicsConfig& c, float v) { c.mass = v; }, "Physics Mass");
    }
    floatField(
            "Friction", config.friction, 0.005f, 0.f, 2.f,
            [](PhysicsConfig& c, float v) { c.friction = v; }, "Physics Friction");
    if (!soft) {
        floatField(
                "Restitution", config.restitution, 0.005f, 0.f, 1.f,
                [](PhysicsConfig& c, float v) { c.restitution = v; }, "Physics Restitution");
    }

    // Convex-pieces (V-HACD) parameters, shown only while that shape is picked —
    // the same reveal-on-selection the soft-body section uses.
    if (!soft && config.shape == PhysicsConfig::Shape::Pieces) {
        ImGui::Spacing();
        intField(
                "Max Hulls", config.hulls, 0.2f, 1, 128,
                [](PhysicsConfig& c, int v) { c.hulls = v; }, "Convex Pieces Hulls");
        intField(
                "Verts / Hull", config.hullVerts, 0.2f, 8, 64,
                [](PhysicsConfig& c, int v) { c.hullVerts = v; }, "Convex Pieces Hull Verts");
        intField(
                "Voxel Res", config.voxels, 500.f, 10000, 1000000,
                [](PhysicsConfig& c, int v) { c.voxels = v; }, "Convex Pieces Resolution");
    }

    if (soft) {
        ImGui::Spacing();
        // Stiffness spans four decades between jelly and hard rubber, so the
        // drag is logarithmic — a linear one is unusable at the soft end.
        floatField(
                "Stiffness (Pa)", config.youngsModulus, 0.01f, 1e3f, 1e9f,
                [](PhysicsConfig& c, float v) { c.youngsModulus = v; }, "Soft Body Stiffness",
                ImGuiSliderFlags_Logarithmic);
        floatField(
                "Poisson Ratio", config.poissonsRatio, 0.002f, 0.f, 0.49f,
                [](PhysicsConfig& c, float v) { c.poissonsRatio = v; }, "Soft Body Poisson Ratio");
        intField(
                "Resolution", config.voxelResolution, 0.1f, 2, 64,
                [](PhysicsConfig& c, int v) { c.voxelResolution = v; }, "Soft Body Resolution");
        intField(
                "Iterations", config.solverIterations, 0.2f, 1, 255,
                [](PhysicsConfig& c, int v) { c.solverIterations = v; }, "Soft Body Iterations");

        bool selfCollision = config.selfCollision;
        if (ImGui::Checkbox("Self Collision", &selfCollision)) {
            auto after = config;
            after.selfCollision = selfCollision;
            commit(after, "Soft Body Self Collision");
        }
    }

    ImGui::PopItemWidth();

    if (soft) {
        ImGui::TextWrapped("Deformable volume: the mesh itself bends. Cooked from a closed "
                           "triangle surface, simulated on the GPU (needs CUDA).");
    }
    if (!soft && config.shape == PhysicsConfig::Shape::Pieces) {
        ImGui::TextWrapped("V-HACD splits the mesh into convex hulls so a concave shape "
                           "collides like itself (a mug holds water). Cooked once per "
                           "geometry on Play; a dense mesh can take a moment.");
    }
    if (!soft && config.trigger) {
        ImGui::TextWrapped("Overlap volume: bodies pass straight through it, and a script "
                           "on either side gets on_trigger_enter / on_trigger_exit. It "
                           "generates no contacts at all, so on_collision_* never fires "
                           "for it.");
        // PhysX has no triangle-mesh trigger; the play session substitutes a
        // convex hull and logs one line. Flagged here for the shape that says
        // so outright — whether *Auto* resolves to one depends on the geometry,
        // which only the cook knows, so that case is left to the console.
        if (config.shape == PhysicsConfig::Shape::TriMesh) {
            ImGui::TextColored(theme::warning(),
                               "PhysX has no triangle-mesh trigger - a convex hull is used");
        }
    }

    ImGui::TextColored(theme::muted(), "Stored in userData[\"physics\"]");

    ImGui::TreePop();
}


// ------------------------------------------------------------------- spline

void EditorApp::drawSplineSection(Object3D& object) {

    // Two forms of the same section, because a spline and one of its control
    // points are both ordinary scene nodes: only the userData entry and the
    // parent link tell them apart.
    if (auto* spline = SplineConfig::splineOf(object)) {

        if (!section("Spline Point")) return;

        // Point index, NOT child index: a spline that generates geometry keeps
        // that mesh among the same children without it being a point.
        const auto index = SplineConfig::pointIndexOf(*spline, object);
        const auto count = SplineConfig::controlPoints(*spline).size();
        ImGui::TextColored(theme::muted(), "Point %zu of %zu in \"%s\"",
                           index + 1, count, spline->name.c_str());

        // Uuid rather than the pointer: the deferred edit runs later in the
        // frame, and an undo (or a stop) in between replaces the graph.
        const auto uuid = spline->uuid;
        const auto insert = [this, uuid](std::size_t slot, const char* label) {
            deferred_ = [this, uuid, slot, label] {
                if (auto* live = findByUuid(document_.scene(), uuid)) {
                    addSplinePoint(*live, slot, label);
                }
            };
        };

        if (ImGui::Button("Insert Before")) insert(index, "Insert Spline Point");
        ImGui::SameLine();
        if (ImGui::Button("Insert After")) insert(index + 1, "Insert Spline Point");
        ImGui::SameLine();
        if (ImGui::Button("Select Spline")) {
            deferred_ = [this, uuid] {
                if (auto* live = findByUuid(document_.scene(), uuid)) selectObject(live);
            };
        }

        ImGui::TextColored(theme::muted(), "Delete removes it from the curve.");

        ImGui::TreePop();
        return;
    }

    if (!SplineConfig::isSpline(object)) return;
    if (!section("Spline")) return;

    auto* target = &object;
    auto config = SplineConfig::read(object).value_or(SplineConfig{});
    const auto before = config;

    const auto commit = [&](SplineConfig after, const char* label) {
        commands_.execute(makeProperty<SplineConfig>(
                label, "spline:" + object.uuid,
                [target](const SplineConfig& value) { value.write(*target); },
                before, after));
        document_.setDirty(true);
    };

    ImGui::PushItemWidth(-100 * contentScale_);

    {
        static const char* types[] = {"Centripetal", "Chordal", "CatmullRom"};
        int type = static_cast<int>(config.type);
        if (ImGui::Combo("Type", &type, types, IM_ARRAYSIZE(types))) {
            auto after = config;
            after.type = static_cast<SplineConfig::Type>(type);
            commit(after, "Spline Type");
        }
    }

    {
        float tension = config.tension;
        const bool changed = ImGui::DragFloat("Tension", &tension, 0.005f, 0.f, 1.f);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.tension = tension;
            commit(after, "Spline Tension");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
        // Shown regardless, because that is three.js's own semantics: the value
        // is stored and does nothing until the type is CatmullRom.
        if (config.type != SplineConfig::Type::CatmullRom) {
            ImGui::TextColored(theme::muted(), "Tension applies to CatmullRom only");
        }
    }

    {
        bool closed = config.closed;
        if (ImGui::Checkbox("Closed", &closed)) {
            auto after = config;
            after.closed = closed;
            commit(after, closed ? "Close Spline" : "Open Spline");
        }
    }

    {
        int samples = config.samples;
        const bool changed = ImGui::DragInt("Samples/Segment", &samples, 0.25f, 1,
                                            SplineConfig::maxSamples);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.samples = std::clamp(samples, 1, SplineConfig::maxSamples);
            commit(after, "Spline Samples");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    // --- generated geometry ------------------------------------------------
    // Only the config is edited here. The mesh itself is derived state that the
    // per-frame sync adds, rebuilds and removes to follow it — see
    // SplineOverlay.cpp.
    ImGui::Spacing();
    {
        static const char* kinds[] = {"None", "Tube"};
        int mesh = static_cast<int>(config.mesh);
        if (ImGui::Combo("Mesh", &mesh, kinds, IM_ARRAYSIZE(kinds))) {
            auto after = config;
            after.mesh = static_cast<SplineConfig::MeshKind>(mesh);
            commit(after, "Spline Mesh");
        }
    }

    const auto floatField = [&](const char* label, float value, float step, float lo, float hi,
                                void (*apply)(SplineConfig&, float), const char* undoLabel) {
        const bool changed = ImGui::DragFloat(label, &value, step, lo, hi);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            apply(after, std::clamp(value, lo, hi));
            commit(after, undoLabel);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    if (config.mesh == SplineConfig::MeshKind::Tube) {
        floatField(
                "Radius", config.radius, 0.005f, 0.001f, 100.f,
                [](SplineConfig& c, float v) { c.radius = v; }, "Tube Radius");

        int segments = config.radialSegments;
        const bool changed = ImGui::DragInt("Radial Segments", &segments, 0.1f, 3, 64);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.radialSegments = std::clamp(segments, 3, 64);
            commit(after, "Tube Radial Segments");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    ImGui::PopItemWidth();
    ImGui::Spacing();

    if (config.mesh != SplineConfig::MeshKind::None) {
        ImGui::TextColored(theme::muted(),
                           "The generated mesh is a real child - saved, and physics-configurable.");
    }

    const auto count = SplineConfig::controlPoints(object).size();
    if (auto curve = config.curve(object)) {
        ImGui::Text("%zu control points, length %.2f", count, curve->getLength());
    } else {
        ImGui::TextColored(theme::warning(), "%zu control points - two are needed for a curve", count);
    }

    const auto uuid = object.uuid;
    if (ImGui::Button("Add Point")) {
        deferred_ = [this, uuid] {
            if (auto* live = findByUuid(document_.scene(), uuid)) {
                addSplinePoint(*live, AddObjectCommand::atEnd, "Add Spline Point");
            }
        };
    }

    ImGui::TextColored(theme::muted(), "Children are the control points, in order.");
    ImGui::TextColored(theme::muted(), "Stored in userData[\"spline\"]");

    ImGui::TreePop();
}


// ----------------------------------------------------------------- conveyor

void EditorApp::drawConveyorSection(Object3D& object) {

    // Four forms of the same section, like the spline's two: a conveyor, one
    // of its waypoints, an attached wall and one of the wall's points are all
    // ordinary scene nodes — userData entries and parent links tell them apart.
    if (auto* wall = ConveyorWallConfig::wallOf(object)) {

        if (!section("Wall Point")) return;

        const auto points = ConveyorWallConfig::pointNodes(*wall);
        std::size_t index = 0;
        for (std::size_t i = 0; i < points.size(); ++i) {
            if (points[i] == &object) index = i;
        }
        ImGui::TextColored(theme::muted(), "Point %zu of %zu in \"%s\"",
                           index + 1, points.size(), wall->name.c_str());

        const auto uuid = wall->uuid;
        if (ImGui::Button("Select Wall")) {
            deferred_ = [this, uuid] {
                if (auto* live = findByUuid(document_.scene(), uuid)) selectObject(live);
            };
        }

        ImGui::TextColored(theme::muted(), "Drag it in plan - the base finds the deck.");
        ImGui::TextColored(theme::muted(), "Delete removes it from the wall.");

        ImGui::TreePop();
        return;
    }

    if (ConveyorWallConfig::isWall(object)) {

        if (!section("Conveyor Wall")) return;

        auto* target = &object;
        const auto before = ConveyorWallConfig::read(object).value_or(ConveyorWallConfig{});
        const auto commitWall = [&](ConveyorWallConfig after, const char* label) {
            commands_.execute(makeProperty<ConveyorWallConfig>(
                    label, "conveyorWall:" + object.uuid,
                    [target](const ConveyorWallConfig& value) { value.write(*target); },
                    before, after));
            document_.setDirty(true);
        };

        ImGui::PushItemWidth(-110 * contentScale_);
        {
            float height = before.height;
            const bool changed = ImGui::DragFloat("Height", &height, 0.005f, 0.02f, 3.f);
            if (ImGui::IsItemActivated()) commands_.beginTransaction();
            if (changed) {
                auto after = before;
                after.height = std::clamp(height, 0.02f, 3.f);
                commitWall(after, "Wall Height");
            }
            if (ImGui::IsItemDeactivated()) commands_.endTransaction();
        }
        ImGui::PopItemWidth();

        const auto uuid = object.uuid;
        if (ImGui::Button("Add Point")) {
            deferred_ = [this, uuid] {
                if (auto* live = findByUuid(document_.scene(), uuid)) {
                    addConveyorWallPoint(*live, "Add Wall Point");
                }
            };
        }
        ImGui::SameLine();
        if (object.parent) {
            const auto parentUuid = object.parent->uuid;
            if (ImGui::Button("Select Conveyor")) {
                deferred_ = [this, parentUuid] {
                    if (auto* live = findByUuid(document_.scene(), parentUuid)) selectObject(live);
                };
            }
        }

        ImGui::TextColored(theme::muted(), "Children are the wall's points, in order.");
        ImGui::TextColored(theme::muted(),
                           "The wall FOLLOWS the belt between its points, base on the deck.");
        ImGui::TextColored(theme::muted(),
                           "Drag an end along the belt for length; drag a point toward");
        ImGui::TextColored(theme::muted(), "the middle to sweep that stretch into a diverter.");
        ImGui::TextColored(theme::muted(), "Stored in userData[\"conveyorWall\"]");

        ImGui::TreePop();
        return;
    }

    if (auto* conveyor = ConveyorConfig::conveyorOf(object)) {

        if (!section("Conveyor Waypoint")) return;

        const auto index = ConveyorConfig::pointIndexOf(*conveyor, object);
        const auto count = ConveyorConfig::waypointNodes(*conveyor).size();
        ImGui::TextColored(theme::muted(), "Waypoint %zu of %zu in \"%s\"",
                           index + 1, count, conveyor->name.c_str());

        auto* node = &object;
        const auto wpBefore = ConveyorWaypointConfig::read(object);
        const auto commitWp = [&](ConveyorWaypointConfig after, const char* label) {
            commands_.execute(makeProperty<ConveyorWaypointConfig>(
                    label, "conveyorWp:" + object.uuid,
                    [node](const ConveyorWaypointConfig& value) { value.write(*node); },
                    wpBefore, after));
            document_.setDirty(true);
        };

        const auto parentConfig = ConveyorConfig::read(*conveyor).value_or(ConveyorConfig{});

        // Rounded corner, on interior waypoints only — the ends have no corner
        // to round. The arc is inserted TANGENT to both neighbouring segments
        // (centre and tangent points derived, drawn as an overlay helper while
        // this waypoint is selected), and the radius clamps itself to what the
        // segments allow — a bend cannot kink, whatever gets dragged where.
        if (index > 0 && index + 1 < count) {
            ImGui::PushItemWidth(-110 * contentScale_);
            float radius = wpBefore.cornerRadius;
            const bool changed = ImGui::DragFloat("Corner Radius", &radius, 0.01f, 0.f, 50.f);
            if (ImGui::IsItemActivated()) commands_.beginTransaction();
            if (changed) {
                auto after = wpBefore;
                after.cornerRadius = std::max(radius, 0.f);
                commitWp(after, "Corner Radius");
            }
            if (ImGui::IsItemDeactivated()) commands_.endTransaction();
            ImGui::PopItemWidth();
            ImGui::TextColored(theme::muted(),
                               wpBefore.cornerRadius > 1e-4f
                                       ? "Tangent bend; shrinks itself to fit the segments."
                                       : "0 = sharp corner. Raise it to round the bend.");
        }

        // Surface of the segment LEAVING this waypoint. Hidden on the last
        // waypoint (no segment leaves it) and on a separator (a wall has no
        // surface).
        if (!parentConfig.separator && index + 1 < count) {
            int kind = static_cast<int>(wpBefore.segKind);
            ImGui::Text("Segment to next:");
            bool changed = ImGui::RadioButton("Flat", &kind, 0);
            ImGui::SameLine();
            changed |= ImGui::RadioButton("Rollers", &kind, 1);
            ImGui::SameLine();
            changed |= ImGui::RadioButton("Cleats", &kind, 2);
            if (changed) {
                auto after = wpBefore;
                after.segKind = static_cast<conveyor::SegKind>(kind);
                commitWp(after, "Segment Surface");
            }
        }

        // Uuid rather than the pointer: the deferred edit runs later in the
        // frame, and an undo (or a stop) in between replaces the graph.
        const auto uuid = conveyor->uuid;
        const auto insert = [this, uuid](std::size_t slot, const char* label) {
            deferred_ = [this, uuid, slot, label] {
                if (auto* live = findByUuid(document_.scene(), uuid)) {
                    addConveyorPoint(*live, slot, label);
                }
            };
        };

        if (ImGui::Button("Insert Before")) insert(index, "Insert Waypoint");
        ImGui::SameLine();
        if (ImGui::Button("Insert After")) insert(index + 1, "Insert Waypoint");
        ImGui::SameLine();
        if (ImGui::Button("Select Conveyor")) {
            deferred_ = [this, uuid] {
                if (auto* live = findByUuid(document_.scene(), uuid)) selectObject(live);
            };
        }

        ImGui::TextColored(theme::muted(), "Delete removes it from the path.");

        ImGui::TreePop();
        return;
    }

    if (!ConveyorConfig::isConveyor(object)) return;
    if (!section("Conveyor")) return;

    auto* target = &object;
    auto config = ConveyorConfig::read(object).value_or(ConveyorConfig{});
    const auto before = config;

    const auto commit = [&](ConveyorConfig after, const char* label) {
        commands_.execute(makeProperty<ConveyorConfig>(
                label, "conveyor:" + object.uuid,
                [target](const ConveyorConfig& value) { value.write(*target); },
                before, after));
        document_.setDirty(true);
    };

    ImGui::PushItemWidth(-110 * contentScale_);

    const auto floatField = [&](const char* label, float value, float step, float lo, float hi,
                                void (*apply)(ConveyorConfig&, float), const char* undoLabel) {
        const bool changed = ImGui::DragFloat(label, &value, step, lo, hi);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            apply(after, std::clamp(value, lo, hi));
            commit(after, undoLabel);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    {
        bool separator = config.separator;
        if (ImGui::Checkbox("Separator (wall, no belt)", &separator)) {
            auto after = config;
            after.separator = separator;
            commit(after, separator ? "Make Separator" : "Make Belt");
            config = ConveyorConfig::read(object).value_or(config);
        }
    }

    if (config.separator) {
        floatField(
                "Wall Height", config.wallHeight, 0.01f, 0.05f, 5.f,
                [](ConveyorConfig& c, float v) { c.wallHeight = v; }, "Wall Height");
    } else {
        floatField(
                "Belt Width", config.width, 0.01f, 0.05f, 5.f,
                [](ConveyorConfig& c, float v) { c.width = v; }, "Belt Width");
        floatField(
                "Belt Speed (m/s)", config.speed, 0.01f, 0.f, 10.f,
                [](ConveyorConfig& c, float v) { c.speed = v; }, "Belt Speed");

        bool reverse = config.reverse;
        if (ImGui::Checkbox("Reverse Flow", &reverse)) {
            auto after = config;
            after.reverse = reverse;
            commit(after, "Reverse Flow");
        }
        ImGui::SameLine();
        bool frame = config.frame;
        if (ImGui::Checkbox("Frame", &frame)) {
            auto after = config;
            after.frame = frame;
            commit(after, frame ? "Add Frame" : "Remove Frame");
        }
    }

    {
        bool smooth = config.smooth;
        if (ImGui::Checkbox("Smooth (spline)", &smooth)) {
            auto after = config;
            after.smooth = smooth;
            commit(after, "Conveyor Smoothing");
        }
    }

    {
        int samples = config.samples;
        const bool changed = ImGui::DragInt("Samples/Segment", &samples, 0.25f, 2,
                                            ConveyorConfig::maxSamples);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.samples = std::clamp(samples, 2, ConveyorConfig::maxSamples);
            commit(after, "Conveyor Samples");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    // Tuning for whichever segments opt into rollers / cleats (chosen per
    // waypoint). Shown only when at least one segment uses them.
    if (!config.separator) {
        bool anyRollers = false, anyCleats = false;
        const auto nodes = ConveyorConfig::waypointNodes(object);
        for (std::size_t i = 0; i + 1 < nodes.size(); ++i) {
            const auto wp = ConveyorWaypointConfig::read(*nodes[i]);
            if (wp.segKind == conveyor::SegKind::Rollers) anyRollers = true;
            else if (wp.segKind == conveyor::SegKind::Cleats) anyCleats = true;
        }
        if (anyRollers) {
            floatField(
                    "Roller Radius", config.rollerRadius, 0.002f, 0.01f, 0.5f,
                    [](ConveyorConfig& c, float v) { c.rollerRadius = v; }, "Roller Radius");
        }
        if (anyCleats) {
            floatField(
                    "Cleat Height", config.cleatHeight, 0.005f, 0.02f, 1.f,
                    [](ConveyorConfig& c, float v) { c.cleatHeight = v; }, "Cleat Height");
            floatField(
                    "Cleat Spacing", config.cleatSpacing, 0.01f, 0.1f, 5.f,
                    [](ConveyorConfig& c, float v) { c.cleatSpacing = v; }, "Cleat Spacing");
        }
    }

    ImGui::PopItemWidth();
    ImGui::Spacing();

    const auto count = ConveyorConfig::waypointNodes(object).size();
    if (count >= 2) {
        const auto spec = config.spec(object);
        const auto pts = conveyor::resamplePath(spec.waypoints, spec.smooth, spec.samples);
        float length = 0.f;
        for (std::size_t i = 0; i + 1 < pts.size(); ++i) length += pts[i].distanceTo(pts[i + 1]);
        ImGui::Text("%zu waypoints, path length %.2f", count, length);
    } else {
        ImGui::TextColored(theme::warning(), "%zu waypoints - two are needed for a path", count);
    }

    const auto uuid = object.uuid;
    if (ImGui::Button("Add Waypoint")) {
        deferred_ = [this, uuid] {
            if (auto* live = findByUuid(document_.scene(), uuid)) {
                addConveyorPoint(*live, AddObjectCommand::atEnd, "Add Waypoint");
            }
        };
    }
    if (!config.separator) {
        ImGui::SameLine();
        if (ImGui::Button("Add Wall")) {
            deferred_ = [this, uuid] {
                if (auto* live = findByUuid(document_.scene(), uuid)) {
                    addConveyorWall(*live, "Add Wall");
                }
            };
        }
    }

    ImGui::TextColored(theme::muted(), "Children are the waypoints, in order.");
    ImGui::TextColored(theme::muted(), "Pick a waypoint for rounded corners and per-segment surfaces.");
    ImGui::TextColored(theme::muted(), "Play drives the belt; bodies and soft bodies convey.");
    ImGui::TextColored(theme::muted(), "Stored in userData[\"conveyor\"]");

    ImGui::TreePop();
}


// ------------------------------------------------------------------- sensor

void EditorApp::drawSensorSection(Object3D& object) {

    if (object.is<Scene>()) return;
    if (!section("Sensor", false)) return;

    auto* target = &object;
    auto config = SensorConfig::read(object).value_or(SensorConfig{});
    const auto before = config;

    const auto commit = [&](SensorConfig after, const char* label) {
        commands_.execute(makeProperty<SensorConfig>(
                label, "sensor:" + object.uuid,
                [target](const SensorConfig& value) { value.write(*target); },
                before, after));
        document_.setDirty(true);
    };

    // Add/Remove is a userData edit, not a graph edit, so it goes straight
    // through the property command like the physics Enabled box — nothing here
    // can invalidate `object`, which is why this one does not need the deferred
    // re-resolve the spline section's Insert buttons do.
    bool enabled = config.enabled;
    if (ImGui::Checkbox("Enabled", &enabled)) {
        auto after = config;
        after.enabled = enabled;
        commit(after, enabled ? "Add Sensor" : "Remove Sensor");
        // Re-read: the click already changed the document, and drawing the rest
        // of this frame from the pre-click value would show the wrong fields for
        // one frame (and, on Remove, fields for an entry that is gone).
        config = SensorConfig::read(object).value_or(SensorConfig{});
    }

    if (!config.enabled) {
        ImGui::TextColored(theme::muted(),
                           "Measures in this object's world frame. The transform gizmo aims it.");
        ImGui::TreePop();
        return;
    }

    ImGui::PushItemWidth(-110 * contentScale_);

    {
        static const char* types[] = {"IMU", "Depth Camera", "LIDAR",
                                      "Joint Encoder", "Contact", "Force/Torque"};
        int type = static_cast<int>(config.type);
        if (ImGui::Combo("Type", &type, types, IM_ARRAYSIZE(types))) {
            auto after = config;
            after.type = static_cast<SensorConfig::Type>(type);
            // Only the type changes. Every other key is written regardless of
            // type (see SensorConfig), so the settings of the type being left
            // behind are still there when the user comes back to it.
            commit(after, "Sensor Type");
            config = SensorConfig::read(object).value_or(config);
        }
    }

    const auto floatField = [&](const char* label, float value, float speed, float min, float max,
                                void (*assign)(SensorConfig&, float), const char* action,
                                const char* format = "%.4f") {
        float edited = value;
        const bool changed = ImGui::DragFloat(label, &edited, speed, min, max, format);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            assign(after, std::clamp(edited, min, max));
            commit(after, action);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    const auto intField = [&](const char* label, int value, float speed, int min, int max,
                              void (*assign)(SensorConfig&, int), const char* action) {
        int edited = value;
        const bool changed = ImGui::DragInt(label, &edited, speed, min, max);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            assign(after, std::clamp(edited, min, max));
            commit(after, action);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    // The joint an Encoder / Force-Torque sensor measures. Walk up to the Robot
    // this sensor is authored on and offer its articulated joints by name; a
    // sensor authored off a robot (or on a robot with Simulate off) is a mistake
    // worth naming here, where the picker lives, rather than as silence at Play.
    // `offerAll` adds the encoder's "All joints" entry — one authored sensor,
    // one live encoder per DOF at Play. A load cell sits in one joint, so the
    // Force/Torque picker does not offer it.
    const auto jointPicker = [&](bool offerAll) {
        Robot* robot = nullptr;
        for (Object3D* o = &object; o != nullptr; o = o->parent) {
            if (auto* r = o->as<Robot>()) {
                robot = r;
                break;
            }
        }
        if (!robot) {
            ImGui::TextColored(theme::warning(),
                               "Author this sensor on a robot or one of its links.");
            return;
        }
        if (!ArticulationConfig::read(*robot)) {
            ImGui::TextColored(theme::warning(),
                               "Turn Simulate on in the robot's Robot section to read a joint.");
        }
        const auto info = robot->getArticulatedJointInfo();
        if (info.empty()) {
            ImGui::TextColored(theme::muted(), "The robot has no articulated joints.");
            return;
        }
        // The combo shows "(choose)" when nothing is picked yet, then the joint
        // names in the robot's own order. Selecting one is one undoable edit.
        std::vector<const char*> names;
        names.reserve(info.size() + 2);
        names.push_back("(choose)");
        if (offerAll) names.push_back("All joints");
        const int base = offerAll ? 2 : 1;
        int current = 0;
        if (offerAll && config.joint == SensorConfig::allJoints) current = 1;
        for (std::size_t i = 0; i < info.size(); ++i) {
            names.push_back(info[i].name.c_str());
            if (info[i].name == config.joint) current = static_cast<int>(i) + base;
        }
        if (ImGui::Combo("Joint", &current, names.data(), static_cast<int>(names.size()))) {
            auto after = config;
            if (current == 0) {
                after.joint.clear();
            } else if (offerAll && current == 1) {
                after.joint = SensorConfig::allJoints;
            } else {
                after.joint = info[static_cast<std::size_t>(current - base)].name;
            }
            commit(after, "Sensor Joint");
            config = SensorConfig::read(object).value_or(config);
        }
        if (config.joint.empty()) {
            ImGui::TextColored(theme::muted(), "Pick which joint this sensor reads.");
        } else if (config.joint == SensorConfig::allJoints) {
            if (offerAll) {
                ImGui::TextColored(theme::muted(),
                                   "One live encoder per articulated DOF at Play.");
            } else {
                // A leftover from flipping the type combo away from Encoder —
                // the settings of the other type are still your settings, but
                // this one has no meaning here.
                ImGui::TextColored(theme::warning(),
                                   "All joints is an encoder thing - pick the one this "
                                   "load cell sits in.");
            }
        }
    };

    floatField(
            "Rate (Hz)", config.rateHz, 0.25f, 0.f, 2000.f,
            [](SensorConfig& c, float v) { c.rateHz = v; }, "Sensor Rate", "%.1f");
    if (config.rateHz <= 0.f) {
        ImGui::TextColored(theme::muted(),
                           SensorConfig::isVision(config.type)
                                   ? "0 = scan every frame (expensive)"
                                   : "0 = sample every physics substep");
    }
    intField(
            "Seed", config.seed, 1.f, 0, 1000000,
            [](SensorConfig& c, int v) { c.seed = v; }, "Sensor Seed");

    ImGui::Spacing();

    switch (config.type) {

        case SensorConfig::Type::Imu: {
            // Continuous-time densities, the way a spec sheet quotes them, so the
            // authored numbers are rate-independent.
            floatField(
                    "Gyro Density", config.gyroNoiseDensity, 0.0002f, 0.f, 1.f,
                    [](SensorConfig& c, float v) { c.gyroNoiseDensity = v; },
                    "IMU Gyro Noise", "%.5f");
            floatField(
                    "Gyro Bias Walk", config.gyroRandomWalk, 1e-5f, 0.f, 0.1f,
                    [](SensorConfig& c, float v) { c.gyroRandomWalk = v; },
                    "IMU Gyro Bias Walk", "%.6f");
            floatField(
                    "Accel Density", config.accelNoiseDensity, 0.002f, 0.f, 10.f,
                    [](SensorConfig& c, float v) { c.accelNoiseDensity = v; },
                    "IMU Accel Noise", "%.5f");
            floatField(
                    "Accel Bias Walk", config.accelRandomWalk, 0.0002f, 0.f, 1.f,
                    [](SensorConfig& c, float v) { c.accelRandomWalk = v; },
                    "IMU Accel Bias Walk", "%.6f");

            if (ImGui::SmallButton("Perfect")) {
                auto after = config;
                after.gyroNoiseDensity = 0.f;
                after.gyroRandomWalk = 0.f;
                after.accelNoiseDensity = 0.f;
                after.accelRandomWalk = 0.f;
                commit(after, "Perfect IMU");
                config = SensorConfig::read(object).value_or(config);
            }
            ImGui::SameLine();
            ImGui::TextColored(theme::muted(), "zero noise = ground truth");
            break;
        }

        case SensorConfig::Type::Depth: {
            floatField(
                    "FOV (deg)", config.fovY, 0.25f, 1.f, 179.f,
                    [](SensorConfig& c, float v) { c.fovY = v; }, "Depth FOV", "%.1f");
            intField(
                    "Width", config.width, 1.f, 8, SensorConfig::maxImageSize,
                    [](SensorConfig& c, int v) { c.width = v; }, "Depth Width");
            intField(
                    "Height", config.height, 1.f, 8, SensorConfig::maxImageSize,
                    [](SensorConfig& c, int v) { c.height = v; }, "Depth Height");
            break;
        }

        case SensorConfig::Type::Lidar: {
            static const char* beams[] = {"Dense Grid", "VLP-16", "HDL-32E", "OS1-64", "OS0-128"};
            int pattern = static_cast<int>(config.beams);
            if (ImGui::Combo("Beams", &pattern, beams, IM_ARRAYSIZE(beams))) {
                auto after = config;
                after.beams = static_cast<SensorConfig::Beams>(pattern);
                commit(after, "LIDAR Beam Pattern");
                config = SensorConfig::read(object).value_or(config);
            }
            intField(
                    "Face Size", config.faceSize, 2.f, 16, SensorConfig::maxFaceSize,
                    [](SensorConfig& c, int v) { c.faceSize = v; }, "LIDAR Face Size");
            ImGui::TextColored(theme::muted(),
                               "Six 90-degree depth passes per scan - face size is each "
                               "one's resolution.");
            break;
        }

        case SensorConfig::Type::Encoder: {
            jointPicker(true);
            floatField(
                    "Resolution", config.encoderResolution, 1e-5f, 0.f, 1.f,
                    [](SensorConfig& c, float v) { c.encoderResolution = v; },
                    "Encoder Resolution", "%.6f");
            ImGui::TextColored(theme::muted(), "rad (or m) per tick; 0 = ideal");
            break;
        }

        case SensorConfig::Type::Contact: {
            floatField(
                    "Force Threshold", config.contactForceThreshold, 0.05f, 0.f, 10000.f,
                    [](SensorConfig& c, float v) { c.contactForceThreshold = v; },
                    "Contact Force Threshold", "%.2f");
            break;
        }

        case SensorConfig::Type::ForceTorque: {
            jointPicker(false);
            ImGui::TextColored(theme::muted(),
                               "Reads the 6-axis wrench through the joint, in its child frame.");
            break;
        }
    }

    // Shared by both ranging sensors: the frustum they see through and the
    // per-return noise. Not a density — one laser pulse's uncertainty does not
    // depend on how long ago the previous scan was.
    if (SensorConfig::isVision(config.type)) {
        ImGui::Spacing();
        floatField(
                "Near (m)", config.nearPlane, 0.005f, 0.001f, 100.f,
                [](SensorConfig& c, float v) { c.nearPlane = v; }, "Sensor Near", "%.3f");
        floatField(
                "Far (m)", config.farPlane, 0.25f, 0.01f, 10000.f,
                [](SensorConfig& c, float v) { c.farPlane = v; }, "Sensor Far", "%.2f");
        if (config.farPlane <= config.nearPlane) {
            ImGui::TextColored(theme::warning(), "Far must be beyond Near");
        }
        floatField(
                "Range Sigma (m)", config.rangeStddev, 0.001f, 0.f, 5.f,
                [](SensorConfig& c, float v) { c.rangeStddev = v; }, "Range Noise", "%.4f");
        floatField(
                "Sigma per m", config.rangeStddevPerMetre, 0.0002f, 0.f, 1.f,
                [](SensorConfig& c, float v) { c.rangeStddevPerMetre = v; },
                "Range Noise per Metre", "%.5f");
        floatField(
                "Range Bias (m)", config.rangeBias, 0.001f, -5.f, 5.f,
                [](SensorConfig& c, float v) { c.rangeBias = v; }, "Range Bias", "%.4f");
    }

    ImGui::PopItemWidth();
    ImGui::Spacing();

    // The mistake that authors cleanly and measures nothing: a proprioceptive
    // sensor with no rigid body under it. Cheap to answer here, and Play would
    // otherwise be the first place it shows up.
    if (config.type == SensorConfig::Type::Imu || config.type == SensorConfig::Type::Contact) {
        bool onBody = false;
        for (const Object3D* node = &object; node && !onBody; node = node->parent) {
            const auto physics = PhysicsConfig::read(*node);
            onBody = physics && physics->enabled &&
                     physics->body != PhysicsConfig::Body::Static &&
                     physics->body != PhysicsConfig::Body::Soft;
        }
        if (!onBody) {
            ImGui::TextColored(theme::warning(),
                               "No dynamic or kinematic body here or above - this will not "
                               "measure. Add Physics to this object or a parent.");
        }
    }
    ImGui::TextColored(theme::muted(),
                       "Rebuilt from this config on every Play, so the seed replays exactly.");
    ImGui::TextColored(theme::muted(), "Stored in userData[\"sensor\"]");

    ImGui::TreePop();
}
