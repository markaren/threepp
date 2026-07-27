
#include "../EditorApp.hpp"
#include "../EditorTheme.hpp"
#include "../PanelLayout.hpp"

#include "threepp/extras/editor/AnimationConfig.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"

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

    const Thumbnail& thumbnailFor(const std::shared_ptr<Texture>& texture) {

        static std::unordered_map<std::string, Thumbnail> cache;
        static Thumbnail empty;

        if (!texture) return empty;

        auto it = cache.find(texture->uuid());
        if (it != cache.end()) return it->second;

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

        return cache.emplace(texture->uuid(), thumb).first->second;
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


void EditorApp::drawInspector() {

    const auto* viewport = ImGui::GetMainViewport();
    const float s = contentScale_;

    const float width = layout::inspectorWidth * s;
    const float top = menuHeight_ + toolbarHeight_;
    const float bottom = statusHeight_ + (bottomPanelOpen_ ? layout::bottomHeight * s
                                                           : ImGui::GetFrameHeight() + 6 * s);

    ImGui::SetNextWindowPos({viewport->Pos.x + viewport->Size.x - width, viewport->Pos.y + top});
    ImGui::SetNextWindowSize({width, std::max(viewport->Size.y - top - bottom, 40.f * s)});

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
        drawLightSection(*selected);
        drawCameraSection(*selected);
        drawAnimationSection(*selected);
        drawJointsSection(*selected);
        drawPhysicsSection(*selected);

        if (locked) ImGui::EndDisabled();
    }
    ImGui::End();
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

void EditorApp::drawTextureSlot(Material& material, const char* label,
                                const std::shared_ptr<Texture>& current,
                                const std::function<void(const std::shared_ptr<Texture>&)>& setter,
                                bool srgb) {

    ImGui::PushID(label);

    const float thumb = 34 * contentScale_;
    drawThumbnail(current, thumb);
    ImGui::SameLine();

    ImGui::BeginGroup();
    ImGui::TextUnformatted(label);
    if (ImGui::SmallButton("Load...")) {
        // The modal resolves in a later frame, so remember which slot asked.
        textureSlotTarget_ = {&material, setter, current, label, srgb};
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

    ImGui::PushItemWidth(-100 * contentScale_);

    if (auto* withColor = dynamic_cast<MaterialWithColor*>(raw)) {
        float rgb[3];
        toSrgbFloats(withColor->color, rgb);
        const bool changed = ImGui::ColorEdit3("Color", rgb);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            commands_.execute(makeProperty<Color>(
                    "Color", "color:" + raw->uuid(),
                    [withColor](const Color& v) { withColor->color = v; },
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
                    [withEmissive](const Color& v) { withEmissive->emissive = v; },
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
                    [withEmissive](const float& v) { withEmissive->emissiveIntensity = v; },
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
            commands_.execute(makeProperty<float>(label, key + raw->uuid(), setter, *value, edited));
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
                    [raw](const bool& v) {
                        raw->transparent = v;
                        raw->needsUpdate();
                    },
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
                    [raw](const int& v) {
                        raw->side = static_cast<Side>(v);
                        raw->needsUpdate();
                    },
                    static_cast<int>(raw->side), side));
            touched();
        }
    }

    if (auto* withWireframe = dynamic_cast<MaterialWithWireframe*>(raw)) {
        bool wireframe = withWireframe->wireframe;
        if (ImGui::Checkbox("Wireframe", &wireframe)) {
            commands_.execute(makeProperty<bool>(
                    "Wireframe", {},
                    [withWireframe](const bool& v) { withWireframe->wireframe = v; },
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
                    [withFlat](const bool& v) {
                        withFlat->flatShading = v;
                        withFlat->needsUpdate();
                    },
                    !flat, flat));
            touched();
        }
    }

    ImGui::PopItemWidth();

    if (ImGui::TreeNodeEx("Textures", ImGuiTreeNodeFlags_SpanAvailWidth)) {

        if (auto* m = dynamic_cast<MaterialWithMap*>(raw)) {
            drawTextureSlot(*raw, "map", m->map,
                            [m](const std::shared_ptr<Texture>& t) { m->map = t; }, true);
        }
        if (auto* m = dynamic_cast<MaterialWithNormalMap*>(raw)) {
            drawTextureSlot(*raw, "normalMap", m->normalMap,
                            [m](const std::shared_ptr<Texture>& t) { m->normalMap = t; }, false);
        }
        if (auto* m = dynamic_cast<MaterialWithRoughness*>(raw)) {
            drawTextureSlot(*raw, "roughnessMap", m->roughnessMap,
                            [m](const std::shared_ptr<Texture>& t) { m->roughnessMap = t; }, false);
        }
        if (auto* m = dynamic_cast<MaterialWithMetalness*>(raw)) {
            drawTextureSlot(*raw, "metalnessMap", m->metalnessMap,
                            [m](const std::shared_ptr<Texture>& t) { m->metalnessMap = t; }, false);
        }
        if (auto* m = dynamic_cast<MaterialWithAoMap*>(raw)) {
            drawTextureSlot(*raw, "aoMap", m->aoMap,
                            [m](const std::shared_ptr<Texture>& t) { m->aoMap = t; }, false);
        }
        if (auto* m = dynamic_cast<MaterialWithEmissive*>(raw)) {
            drawTextureSlot(*raw, "emissiveMap", m->emissiveMap,
                            [m](const std::shared_ptr<Texture>& t) { m->emissiveMap = t; }, true);
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


// -------------------------------------------------------------------- joints

void EditorApp::drawJointsSection(Object3D& object) {

    auto* robot = object.as<Robot>();
    if (!robot || robot->numDOF() == 0) return;
    if (!section("Joints")) return;

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
        static const char* bodies[] = {"Static", "Dynamic", "Kinematic"};
        int body = static_cast<int>(config.body);
        if (ImGui::Combo("Body", &body, bodies, IM_ARRAYSIZE(bodies))) {
            auto after = config;
            after.body = static_cast<PhysicsConfig::Body>(body);
            commit(after, "Physics Body Type");
        }
    }

    {
        static const char* shapes[] = {"Auto", "Box", "Sphere", "Capsule", "Convex", "TriMesh"};
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
    }

    const auto floatField = [&](const char* label, float value, float speed, float min, float max,
                                void (*assign)(PhysicsConfig&, float), const char* action) {
        float edited = value;
        const bool changed = ImGui::DragFloat(label, &edited, speed, min, max);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            assign(after, edited);
            commit(after, action);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    if (config.body == PhysicsConfig::Body::Dynamic) {
        floatField(
                "Mass (kg)", config.mass, 0.05f, 0.001f, 10000.f,
                [](PhysicsConfig& c, float v) { c.mass = v; }, "Physics Mass");
    }
    floatField(
            "Friction", config.friction, 0.005f, 0.f, 2.f,
            [](PhysicsConfig& c, float v) { c.friction = v; }, "Physics Friction");
    floatField(
            "Restitution", config.restitution, 0.005f, 0.f, 1.f,
            [](PhysicsConfig& c, float v) { c.restitution = v; }, "Physics Restitution");

    ImGui::PopItemWidth();

    ImGui::TextColored(theme::muted(), "Stored in userData[\"physics\"]");

    ImGui::TreePop();
}
