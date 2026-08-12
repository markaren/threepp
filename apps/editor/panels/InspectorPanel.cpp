
#include "../EditorApp.hpp"
#include "../EditorTheme.hpp"
#include "../ImportFormats.hpp"
#include "../PanelLayout.hpp"

#include "threepp/extras/editor/AnimationConfig.hpp"
#include "threepp/extras/editor/ArticulationConfig.hpp"
#include "threepp/extras/editor/ConveyorConfig.hpp"
#include "threepp/extras/editor/GeneratorConfig.hpp"
#include "threepp/extras/editor/GranularConfig.hpp"
#include "threepp/extras/editor/JointConfig.hpp"
#include "threepp/extras/editor/MaterialTextureSlots.hpp"
#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/ParticleFieldConfig.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"
#include "threepp/extras/editor/SoundConfig.hpp"
#include "threepp/extras/editor/SplatImportConfig.hpp"
#include "threepp/extras/editor/SplatSurfaceCache.hpp"
#include "threepp/extras/editor/SplatSurfaceConfig.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"
#include "threepp/extras/editor/TextConfig.hpp"
#include "threepp/extras/editor/TreeConfig.hpp"
#include "threepp/extras/editor/VehicleConfig.hpp"

#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/objects/SplatCloud.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Clock.hpp"
#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/lights/Light.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/renderers/RenderTarget.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/extras/editor/ShadowFit.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <random>
#include <unordered_map>
#include <unordered_set>

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
    const float top = menuHeight_;
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
        drawSplatSection(*selected);
        drawInstancingSection(*selected);
        drawLightSection(*selected);
        drawCameraSection(*selected);
        drawAnimationSection(*selected);
        drawJointsSection(*selected);
        drawSplineSection(*selected);
        drawConveyorSection(*selected);
        drawParticleFieldSection(*selected);
        drawGranularSection(*selected);
        drawSoundSection(*selected);
        drawTextSection(*selected);
        drawTreeSection(*selected);
        drawScriptSection(*selected);
        drawPhysicsSection(*selected);
        drawVehicleSection(*selected);
        drawJointAuthoringSection(*selected);
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

EditorApp::UvTransform EditorApp::uvTransformOf(const Texture& texture) {

    return {texture.repeat, texture.offset, texture.center, texture.rotation};
}

EditorApp::TextureSampling EditorApp::samplingOf(const Texture& texture) {

    return {texture.wrapS, texture.wrapT, texture.minFilter, texture.magFilter,
            texture.anisotropy, texture.colorSpace};
}

void EditorApp::applyUvTransform(Material& material,
                                 const std::vector<std::shared_ptr<Texture>>& textures,
                                 const UvTransform& after, const char* label) {

    if (textures.empty()) return;

    std::vector<UvTransform> before;
    before.reserve(textures.size());
    for (const auto& texture : textures) before.push_back(uvTransformOf(*texture));

    auto* raw = &material;
    commands_.execute(makeProperty<std::vector<UvTransform>>(
            label, "uv:" + material.uuid() + ":" + label,
            [textures, raw](const std::vector<UvTransform>& values) {
                for (std::size_t i = 0; i < textures.size() && i < values.size(); ++i) {
                    auto& texture = *textures[i];
                    texture.repeat = values[i].repeat;
                    texture.offset = values[i].offset;
                    texture.center = values[i].center;
                    texture.rotation = values[i].rotation;
                }
                // No texture version bump: matrixAutoUpdate has GL re-derive the
                // matrix every frame from these four fields. Vulkan instead
                // copies it into the entry's MaterialDesc, which only refreshes
                // when the material's version moves - hence the bump.
                raw->needsUpdate();
            },
            std::move(before), std::vector<UvTransform>(textures.size(), after)));
    document_.setDirty(true);
}

void EditorApp::applyTextureSampling(Material& material, const std::shared_ptr<Texture>& texture,
                                     const TextureSampling& after, const char* label) {

    auto* raw = &material;
    commands_.execute(makeProperty<TextureSampling>(
            label, "sampling:" + texture->uuid(),
            [texture, raw](const TextureSampling& value) {
                texture->wrapS = value.wrapS;
                texture->wrapT = value.wrapT;
                texture->minFilter = value.minFilter;
                texture->magFilter = value.magFilter;
                texture->anisotropy = value.anisotropy;
                texture->colorSpace = value.colorSpace;
                // Wrap, filtering and anisotropy are sampler state that GL only
                // re-applies while uploading, and that Vulkan reads while
                // rebuilding the image - both gated on the TEXTURE's version.
                // The colour space additionally decides a shader define in GL
                // and the image format in Vulkan, which is a material rebuild.
                texture->needsUpdate();
                raw->needsUpdate();
            },
            samplingOf(*texture), after));
    document_.setDirty(true);
}

void EditorApp::drawUvTransformBlock(Material& material,
                                     const std::vector<std::shared_ptr<Texture>>& textures) {

    if (textures.empty()) return;

    const UvTransform current = uvTransformOf(*textures.front());

    // A loaded document can arrive with its maps transformed differently. The
    // block shows the first one and says so; the next edit unifies them, which
    // is the only honest thing one set of fields can do.
    const auto mixed = [&textures](auto read) {
        for (const auto& texture : textures) {
            if (read(*texture) != read(*textures.front())) return true;
        }
        return false;
    };

    ImGui::PushID("uv");
    ImGui::PushItemWidth(-100 * contentScale_);

    {
        // The ### keeps the widget's identity while the visible label picks up
        // "(mixed)", so a drag is not interrupted by the maps agreeing.
        const char* title = mixed([](const Texture& t) { return t.repeat; })
                                    ? "Tiling (mixed)###tiling"
                                    : "Tiling###tiling";
        float repeat[2]{current.repeat.x, current.repeat.y};
        const bool changed = ImGui::DragFloat2(title, repeat, 0.01f);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = current;
            after.repeat.set(repeat[0], repeat[1]);
            applyUvTransform(material, textures, after, "Tiling");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("How many times the image repeats across the surface.\n"
                              "Above 1 it only tiles when Wrap is Repeat - see the\n"
                              "\"...\" button on a slot.\n"
                              "Applies to every map on this material: the OpenGL\n"
                              "backend shares one UV transform between them.");
        }
    }

    {
        const char* title = mixed([](const Texture& t) { return t.offset; })
                                    ? "Offset (mixed)###offset"
                                    : "Offset###offset";
        float offset[2]{current.offset.x, current.offset.y};
        const bool changed = ImGui::DragFloat2(title, offset, 0.005f);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = current;
            after.offset.set(offset[0], offset[1]);
            applyUvTransform(material, textures, after, "Offset");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    {
        const char* title = mixed([](const Texture& t) { return t.rotation; })
                                    ? "Rotation (mixed)###rotation"
                                    : "Rotation###rotation";
        float degrees = math::radToDeg(current.rotation);
        const bool changed = ImGui::DragFloat(title, &degrees, 0.5f, -360.f, 360.f, "%.1f deg");
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = current;
            after.rotation = math::degToRad(degrees);
            // Turning a map about its (0,0) corner swings it off the surface,
            // which reads as a bug rather than as a rotation. So the first
            // non-zero angle also moves an UNTOUCHED pivot to the middle of the
            // tile, the same default three.js and Unity present. A pivot the
            // user already moved is left where they put it.
            if (after.rotation != 0.f && current.center.x == 0.f && current.center.y == 0.f) {
                after.center.set(0.5f, 0.5f);
            }
            applyUvTransform(material, textures, after, "Rotation");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Turns the maps about the middle of a tile.");
        }
    }

    ImGui::PopItemWidth();
    ImGui::PopID();
    ImGui::Separator();
}

void EditorApp::drawTextureSettingsPopup(Material& material, const std::shared_ptr<Texture>& texture) {

    if (!ImGui::BeginPopup("settings")) return;

    if (!texture->images().empty()) {
        const auto& image = texture->image();
        ImGui::TextColored(theme::muted(), "%u x %u", image.width(), image.height());
    }
    if (!texture->sourceFile.empty()) {
        ImGui::TextColored(theme::muted(), "%s", texture->sourceFile.filename().string().c_str());
    }

    ImGui::PushItemWidth(120 * contentScale_);

    {
        // One control for both axes. A loaded asset can have them differ, and
        // that is shown rather than quietly unified - but picking anything
        // writes both, which is what an authoring tool is for.
        static const char* wraps[] = {"Repeat", "Clamp", "Mirror", "Mixed"};
        static const TextureWrapping modes[] = {TextureWrapping::Repeat,
                                                TextureWrapping::ClampToEdge,
                                                TextureWrapping::MirroredRepeat};
        const bool mixed = texture->wrapS != texture->wrapT;
        int wrap = 3;
        if (!mixed) {
            for (int i = 0; i < 3; ++i) {
                if (modes[i] == texture->wrapS) wrap = i;
            }
        }
        if (ImGui::Combo("Wrap", &wrap, wraps, mixed ? 4 : 3) && wrap < 3) {
            auto after = samplingOf(*texture);
            after.wrapS = after.wrapT = modes[wrap];
            applyTextureSampling(material, texture, after, "Texture Wrap");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("What is sampled outside the image: Repeat tiles it,\n"
                              "Clamp stretches its edge pixels, Mirror alternates.\n"
                              "The Vulkan backend draws Mirror as Repeat.");
        }
    }

    {
        static const char* filters[] = {"Smooth", "Pixelated"};
        int filter = texture->magFilter == Filter::Nearest ? 1 : 0;
        if (ImGui::Combo("Filtering", &filter, filters, IM_ARRAYSIZE(filters))) {
            auto after = samplingOf(*texture);
            after.magFilter = filter ? Filter::Nearest : Filter::Linear;
            // NearestMipmapLinear, not NearestMipmapNearest: hard TEXELS are the
            // whole point, but a hard jump between mip levels is a visible band
            // crossing the surface, and blending the two mips costs none of the
            // crispness that was asked for.
            after.minFilter = filter ? Filter::NearestMipmapLinear : Filter::LinearMipmapLinear;
            applyTextureSampling(material, texture, after, "Texture Filtering");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Pixelated keeps hard texel edges (pixel art).\n"
                              "OpenGL only - the Vulkan backend samples every\n"
                              "material texture the same way.");
        }
    }

    {
        static const char* levels[] = {"1x", "2x", "4x", "8x", "16x"};
        static const int values[] = {1, 2, 4, 8, 16};
        int level = 0;
        for (int i = 0; i < IM_ARRAYSIZE(levels); ++i) {
            if (texture->anisotropy >= values[i]) level = i;
        }
        if (ImGui::Combo("Anisotropy", &level, levels, IM_ARRAYSIZE(levels))) {
            auto after = samplingOf(*texture);
            after.anisotropy = values[level];
            applyTextureSampling(material, texture, after, "Texture Anisotropy");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Sharpens a map seen edge-on, like a floor running\n"
                              "away to the horizon. OpenGL only.");
        }
    }

    ImGui::Separator();

    {
        // Last, and rarely the right thing to touch: the loaders already tag a
        // slot the way it is meant to be read. Getting it wrong leaves the
        // shading quietly off rather than visibly broken, which is exactly why
        // being able to see and correct it is worth a row.
        static const char* spaces[] = {"sRGB", "Linear"};
        int space = texture->colorSpace == ColorSpace::sRGB ? 0 : 1;
        if (ImGui::Combo("Color space", &space, spaces, IM_ARRAYSIZE(spaces))) {
            auto after = samplingOf(*texture);
            after.colorSpace = space == 0 ? ColorSpace::sRGB : ColorSpace::NoColorSpace;
            applyTextureSampling(material, texture, after, "Texture Color Space");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("sRGB for images of colour (base colour, emissive).\n"
                              "Linear for data (normals, roughness, metalness,\n"
                              "occlusion) - decoding those as colour is wrong.");
        }
    }

    ImGui::PopItemWidth();
    ImGui::EndPopup();
}

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
                          formats::images());
    }
    ImGui::SameLine();
    if (current) {
        if (ImGui::SmallButton("Clear")) {
            commands_.execute(std::make_unique<SetMaterialMapCommand>(
                    material, label, setter, current, nullptr));
            document_.setDirty(true);
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("...")) ImGui::OpenPopup("settings");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Wrapping, filtering, colour space");
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

    // After the rect is taken, and outside both groups: a popup is its own
    // window, and the row's drop target must be the row rather than whatever
    // the settings panel happens to cover this frame.
    if (current) drawTextureSettingsPopup(material, current);

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
        const auto slots = textureSlotsOf(*raw);

        // One entry per DISTINCT texture: the same image assigned to two slots
        // must not be written twice by the transform block.
        std::vector<std::shared_ptr<Texture>> assigned;
        for (const auto& slot : slots) {
            if (!slot.current) continue;
            if (std::find(assigned.begin(), assigned.end(), slot.current) == assigned.end()) {
                assigned.push_back(slot.current);
            }
        }
        drawUvTransformBlock(*raw, assigned);

        for (const auto& slot : slots) {
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
    // Not on an object that already carries an authoring rule of its own. A
    // tree is not a generic container: it HAS a generator, a specific one, and
    // a second generic one on top would put a `generated` child beside Trunk
    // and Leaves meaning something else entirely. The generic pattern is for
    // things that have no pattern.
    //
    // Splines and conveyors are Groups with exactly the same claim and are
    // still offered one, which is a wrinkle older than this predicate; left
    // alone deliberately rather than missed.
    const bool ownAuthoring = TreeConfig::isTree(object);
    const bool eligible =
            !isGeneratedOutput && !ownAuthoring && (isScene || object.as<Group>() != nullptr);
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


// -------------------------------------------------------------------- splats

void EditorApp::drawSplatSection(Object3D& object) {

    auto* cloud = object.as<SplatCloud>();
    if (!cloud) return;
    if (!section("Splats")) return;

    const auto& data = cloud->data();

    ImGui::Text("Splats     %zu", cloud->splatCount());
    // The SH degree is what a scan's view-dependent colour costs and how much
    // of the cloud's memory is spherical harmonics: degree 3 is 16 of the 19
    // texels a splat occupies. Worth stating, and there is nowhere else it
    // shows.
    ImGui::Text("SH degree  %d", data.shDegree);
    if (!data.extras.empty()) {
        // Per-splat properties the loader kept but nothing renders — semantic
        // labels, confidences, learned descriptors. Named because their
        // presence is the whole reason SplatData has an escape hatch.
        std::string names;
        for (const auto& [name, values] : data.extras) {
            if (!names.empty()) names += ", ";
            names += name;
        }
        ImGui::TextWrapped("Extras     %s", names.c_str());
    }

    if (const auto config = editor::SplatImportConfig::read(object)) {

        ImGui::Separator();
        // Read-only, and selectable rather than an InputText: the path is not
        // an editable property of the cloud (changing it would not reload
        // anything), but it is very often what you want to copy out.
        ImGui::TextColored(theme::muted(), "Source");
        ImGui::TextWrapped("%s", config->source.c_str());

        if (config->culled) {
            ImGui::TextColored(theme::muted(), "Culled %zu outlier splats on import.",
                               config->removed);
        }
        if (config->flippedX) {
            ImGui::TextColored(theme::muted(), "Flipped 180 degrees about X for +Y-up.");
        }
        // Only for a source that HAS levels: -1 means the file declared none,
        // and printing "level -1" would invent a distinction the scan does not
        // make.
        if (config->lod >= 0) {
            ImGui::TextColored(theme::muted(), "Read detail level %d.", config->lod);
        }
    }

    // --- surface bake -------------------------------------------------------
    // The one authoring verb a scan has: fuse it into a triangle surface that
    // PhysX collides with and the sensors return from (plans/splat-surface-bake.md).
    // The mesh is never saved — the bake is deterministic in the cloud, this
    // config and the node's transform, so the config IS the mesh (see
    // SplatSurfaceConfig).
    {
        using Config = editor::SplatSurfaceConfig;

        auto* target = &object;
        const auto config = Config::read(object).value_or(Config{});

        const auto commit = [&](Config after, std::string label) {
            commands_.execute(makeProperty<Config>(
                    std::move(label), "splatSurface:" + object.uuid,
                    [target](const Config& value) { value.write(*target); },
                    config, std::move(after)));
            document_.setDirty(true);
        };

        ImGui::SeparatorText("Surface");

        bool enabled = config.enabled;
        if (ImGui::Checkbox("Collide and sense", &enabled)) {
            auto after = config;
            after.enabled = enabled;
            commit(std::move(after), enabled ? "Enable Splat Surface" : "Disable Splat Surface");
        }
        ImGui::TextColored(theme::muted(),
                           "Play bakes a triangle surface from the scan: a PhysX static "
                           "collider, and a target the lidar and depth sensors see.");

        if (config.enabled) {
            ImGui::PushItemWidth(-130 * contentScale_);
            {
                float voxel = config.voxelSize;
                const bool changed = ImGui::DragFloat("Voxel (m)", &voxel, 0.002f, 0.f, 0.5f, "%.3f");
                if (ImGui::IsItemActivated()) commands_.beginTransaction();
                if (changed) {
                    auto after = config;
                    after.voxelSize = std::clamp(voxel, 0.f, 0.5f);
                    commit(std::move(after), "Splat Surface Voxel");
                }
                if (ImGui::IsItemDeactivated()) commands_.endTransaction();
            }
            {
                int island = config.minComponentVoxels;
                const bool changed = ImGui::DragInt("Island cells", &island, 8.f, 0, 100000);
                if (ImGui::IsItemActivated()) commands_.beginTransaction();
                if (changed) {
                    auto after = config;
                    after.minComponentVoxels = std::max(island, 0);
                    commit(std::move(after), "Splat Surface Islands");
                }
                if (ImGui::IsItemDeactivated()) commands_.endTransaction();
            }
            {
                int poses = config.poseCount;
                const bool changed = ImGui::DragInt("Poses", &poses, 0.5f, 0, 256);
                if (ImGui::IsItemActivated()) commands_.beginTransaction();
                if (changed) {
                    auto after = config;
                    after.poseCount = std::clamp(poses, 0, 256);
                    commit(std::move(after), "Splat Surface Poses");
                }
                if (ImGui::IsItemDeactivated()) commands_.endTransaction();
            }
            ImGui::PopItemWidth();
            ImGui::TextColored(theme::muted(), "Voxel 0 sizes itself from the scan; poses 0 uses 26.");

            const bool canBake = editor::SplatSurfaceCache::available(renderer_.get());
            if (!canBake) {
                // The depth AOV the bake reads is a Vulkan G-buffer attachment.
                ImGui::TextColored(theme::warning(),
                                   "The bake needs the Vulkan backend - restart with --vulkan.");
            }
            ImGui::BeginDisabled(!canBake);
            if (ImGui::Button("Bake now")) bakeSplatSurface(object);
            ImGui::EndDisabled();
            ImGui::SameLine();
            // Warm means Play is instant; cold means Play pays the bake, with
            // the viewport flashing through the pose loop while it does.
            const bool warm = splatSurfaces_ && splatSurfaces_->find(*cloud, config) != nullptr;
            ImGui::TextColored(warm ? theme::accent() : theme::muted(),
                               warm ? "cached" : "not baked yet");

            // A view state, not a document property: no command, no dirty flag,
            // nothing saved (SplatSurfaceOverlay.cpp). Turning it on bakes
            // through the SAME memo the button and Play use, so a warm scan is
            // instant and a cold one pays the pose loop exactly once.
            bool preview = splatSurfacePreview_;
            ImGui::BeginDisabled(!canBake);
            if (ImGui::Checkbox("Show surface", &preview)) {
                splatSurfacePreview_ = preview;
                if (preview && !warm) bakeSplatSurface(object);
            }
            ImGui::EndDisabled();
            ImGui::TextColored(theme::muted(),
                               "Editor-only wireframe over the scan. Hidden while playing, and "
                               "gone until you re-bake once the scan moves or a knob changes.");

            if (splatBakeNode_ == object.uuid && !splatBakeStats_.empty()) {
                ImGui::TextWrapped("%s", splatBakeStats_.c_str());
            }
        }
    }

    ImGui::Separator();
    // The limitation, on the object it applies to. The console and the status
    // bar say it at Play and at Save; this is where you find out why without
    // having pressed anything.
    ImGui::TextColored(theme::warning(), "Not serialized yet");
    ImGui::TextWrapped("This cloud is not written to the scene file and does not survive Play. "
                       "Re-import the .ply after Stop.");

    ImGui::TreePop();
}


// Bake the cloud's surface into the app's memo, right now.
//
// The editor has no busy modal to hide behind, and this is honest about that:
// the bake blocks the frame it was pressed on (~0.4 s on a synthetic cloud) and
// the viewport visibly flashes through the pose loop, because bakeSurface drives
// the PRIMARY view. The button is inside the inspector's play lock, so it cannot
// run while a session is holding the graph.
void EditorApp::bakeSplatSurface(Object3D& object) {

#ifdef THREEPP_WITH_VULKAN
    auto* cloud = object.as<SplatCloud>();
    if (!cloud || !splatSurfaces_) return;

    const auto config = editor::SplatSurfaceConfig::read(object)
                                .value_or(editor::SplatSurfaceConfig{});

    Clock clock;
    clock.start();
    std::string problem;
    const auto* mesh = splatSurfaces_->bake(renderer_.get(), *cloud, config, &problem);
    const float seconds = clock.getElapsedTime();

    splatBakeNode_ = object.uuid;
    if (!mesh) {
        splatBakeStats_ = "Bake failed: " + problem;
        log("splat surface: \"" + object.name + "\" - " + problem);
        flashStatus("Surface bake failed");
        return;
    }

    // What it produced AND what it dropped, which is the contract every bounded
    // thing in this repo keeps.
    char line[256];
    std::snprintf(line, sizeof(line),
                  "%zu triangles, %.3f m voxels, %d poses, %.2f s "
                  "(dropped %llu fringe, %llu outlier of %llu samples)",
                  mesh->triangleCount(), static_cast<double>(mesh->stats.voxelSize),
                  mesh->stats.poses, static_cast<double>(seconds),
                  static_cast<unsigned long long>(mesh->stats.skippedFringe),
                  static_cast<unsigned long long>(mesh->stats.skippedOutlier),
                  static_cast<unsigned long long>(mesh->stats.depthSamples));
    splatBakeStats_ = line;
    log("splat surface: \"" + object.name + "\" baked - " + splatBakeStats_);
    flashStatus("Surface baked");
#else
    (void) object;
    log("splat surface: this build has no Vulkan backend to bake with");
#endif
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

    drawLightShadowSection(object);

    ImGui::TreePop();
}

// The shadow camera. Until this existed the panel offered no way to reach it at
// all: a scene whose content ran past the default Ortho(-5,5,5,-5) lost its
// shadows out there, one whose content sat well inside it got a fraction of the
// depth range, and either way the only recourse was code — which is what
// HoverArenaAuthor had to take, setting that scene's camera by hand.
//
// So the fields are the feature. "Fit to scene" is a convenience that fills
// them in once; it is not a mode, because no single rule serves every scene.
void EditorApp::drawLightShadowSection(Object3D& object) {

    auto* directional = object.as<DirectionalLight>();
    if (!directional || !directional->shadow) return;
    if (!object.castShadow) return;

    auto* camera = dynamic_cast<OrthographicCamera*>(directional->shadow->camera.get());
    if (!camera) return;

    if (!ImGui::TreeNodeEx("Shadow", ImGuiTreeNodeFlags_DefaultOpen)) return;

    // A button, not a mode. No fit is right for every scene — a large ground
    // plane fits to an extent that spreads the map too thin for anything
    // standing on it — so this proposes a starting point and the fields below
    // own it from there. Undoable like any other edit.
    if (ImGui::Button("Fit to scene")) {
        const Box3 bounds = editor::ShadowFit::shadowBounds(document_.scene());
        if (!bounds.isEmpty()) {
            const Vector3 before{camera->right, camera->nearPlane, camera->farPlane};
            editor::ShadowFit::fit(*directional, bounds);
            const Vector3 after{camera->right, camera->nearPlane, camera->farPlane};

            if (!before.equals(after)) {
                commands_.execute(makeProperty<Vector3>(
                        "Fit Shadow Camera", "shadowfit:" + object.uuid,
                        [camera](const Vector3& v) {
                            camera->left = -v.x;
                            camera->right = v.x;
                            camera->top = v.x;
                            camera->bottom = -v.x;
                            camera->nearPlane = v.y;
                            camera->farPlane = v.z;
                            camera->updateProjectionMatrix();
                        },
                        before, after));
                document_.setDirty(true);
            }
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Size the camera to everything that casts or receives a shadow.\n"
                          "A starting point - a large ground plane fits wider than you\n"
                          "want, and the extent below is then yours to set. Nothing\n"
                          "outside these bounds casts or receives at all.");
    }

    ImGui::PushItemWidth(140.f);

    const auto cameraField = [&](const char* label, const std::string& key, float* value,
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

    // One extent rather than four edges: a square shadow camera is what keeps
    // the fit rotation-stable, and four fields invite a lopsided one that
    // swims as the light turns.
    float extent = camera->right;
    const bool extentChanged = ImGui::DragFloat("Extent", &extent, 0.25f, 0.1f, 5000.f);
    if (ImGui::IsItemActivated()) commands_.beginTransaction();
    if (extentChanged) {
        commands_.execute(makeProperty<float>(
                "Shadow Extent", "shadowextent:" + object.uuid,
                [camera](const float& v) {
                    camera->left = -v;
                    camera->right = v;
                    camera->top = v;
                    camera->bottom = -v;
                    camera->updateProjectionMatrix();
                },
                camera->right, extent));
        document_.setDirty(true);
    }
    if (ImGui::IsItemDeactivated()) commands_.endTransaction();

    cameraField("Near", "shadownear:", &camera->nearPlane, 0.05f, 0.01f, 1000.f,
                [camera](const float& v) { camera->nearPlane = v; camera->updateProjectionMatrix(); });
    cameraField("Far", "shadowfar:", &camera->farPlane, 0.5f, 0.1f, 10000.f,
                [camera](const float& v) { camera->farPlane = v; camera->updateProjectionMatrix(); });

    // Neither of these is something a fit could decide for you.
    auto* shadow = directional->shadow.get();
    static const char* sizes[] = {"512", "1024", "2048", "4096"};
    const int current = static_cast<int>(std::lround(std::log2(std::max(512.f, shadow->mapSize.x)))) - 9;
    int picked = std::clamp(current, 0, 3);
    if (ImGui::Combo("Map size", &picked, sizes, IM_ARRAYSIZE(sizes))) {
        const float chosen = static_cast<float>(512 << picked);
        commands_.execute(makeProperty<float>(
                "Shadow Map Size", "shadowmapsize:" + object.uuid,
                [shadow](const float& v) {
                    shadow->mapSize.set(v, v);
                    // The targets were built for the old size; drop them so the
                    // next shadow render allocates at the new one.
                    shadow->dispose();
                    shadow->map.reset();
                    shadow->mapPass.reset();
                },
                shadow->mapSize.x, chosen));
        document_.setDirty(true);
    }

    cameraField("Bias", "shadowbias:", &shadow->bias, 0.0001f, -0.01f, 0.01f,
                [shadow](const float& v) { shadow->bias = v; });

    ImGui::PopItemWidth();
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


// ------------------------------------------------------------ joint authoring

// The section for a JOINT NODE (see JointConfig): its transform is the joint
// frame, its parent chain is body A, and the other body is picked here by
// name. Not to be confused with drawJointsSection above, which slides a
// robot's URDF joints.
void EditorApp::drawJointAuthoringSection(Object3D& object) {

    if (!JointConfig::isJoint(object)) return;
    if (!section("Joint")) return;

    auto* target = &object;
    auto config = JointConfig::read(object).value_or(JointConfig{});
    const auto before = config;

    const auto commit = [&](const JointConfig& after, const char* label, const char* field) {
        commands_.execute(makeProperty<JointConfig>(
                label, "joint:" + object.uuid + ":" + field,
                [target](const JointConfig& value) { value.write(*target); },
                before, after));
        document_.setDirty(true);
    };

    // Which body this node hangs off — the walk the play session does — so
    // the model (parent chain = body A) is visible rather than remembered.
    {
        const char* bodyA = nullptr;
        for (const auto* o = object.parent; o != nullptr; o = o->parent) {
            const auto physics = PhysicsConfig::read(*o);
            const bool robot = RobotConfig::read(*o).has_value() &&
                               ArticulationConfig::read(*o).has_value();
            if ((physics && physics->enabled) || robot) {
                bodyA = o->name.empty() ? "(unnamed)" : o->name.c_str();
                break;
            }
        }
        ImGui::TextColored(theme::muted(), "Connects %s to %s",
                           bodyA ? bodyA : "the world",
                           config.body.empty() ? "the world" : config.body.c_str());
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Body A is the nearest ancestor with a rigid body.\n"
                              "This node's transform is the joint frame:\n"
                              "anchor at its origin, axis along its local X.");
        }
    }

    ImGui::PushItemWidth(-110 * contentScale_);

    if (ImGui::BeginCombo("Type", JointConfig::label(config.type))) {
        for (const auto type : JointConfig::types) {
            if (ImGui::Selectable(JointConfig::label(type), type == config.type)) {
                auto after = config;
                after.type = type;
                commit(after, "Joint Type", "type");
                config.type = type;// so the widgets below reflect it this frame
            }
        }
        ImGui::EndCombo();
    }

    // The other body, from the bodies that exist to be picked: everything the
    // play session could resolve (an enabled rigid body, or a simulated
    // robot). A combo rather than a text field because the reference IS a
    // scene object — typing a name that resolves is the error path, choosing
    // one that exists is the feature.
    {
        const char* shown = config.body.empty() ? "(world)" : config.body.c_str();
        if (ImGui::BeginCombo("Body B", shown)) {
            if (ImGui::Selectable("(world)", config.body.empty())) {
                auto after = config;
                after.body.clear();
                commit(after, "Joint Body", "body");
            }
            document_.scene().traverse([&](Object3D& candidate) {
                if (&candidate == &object) return;
                // Ancestors are body A's side of this joint already.
                for (const auto* o = object.parent; o != nullptr; o = o->parent) {
                    if (o == &candidate) return;
                }
                const auto physics = PhysicsConfig::read(candidate);
                const bool robot = RobotConfig::read(candidate).has_value() &&
                                   ArticulationConfig::read(candidate).has_value();
                if (!(physics && physics->enabled) && !robot) return;
                if (candidate.name.empty()) return;// no name, no reference
                if (ImGui::Selectable(candidate.name.c_str(), candidate.name == config.body)) {
                    auto after = config;
                    after.body = candidate.name;
                    commit(after, "Joint Body", "body");
                }
            });
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("The other side of the joint, by name.\n"
                              "(world) pins body A to a fixed point in space.");
        }
    }

    const auto floatField = [&](const char* label, float value, float speed, float min, float max,
                                const char* format, void (*assign)(JointConfig&, float),
                                const char* action, const char* field,
                                ImGuiSliderFlags flags = 0) {
        float edited = value;
        const bool changed = ImGui::DragFloat(label, &edited, speed, min, max, format, flags);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            assign(after, edited);
            commit(after, action, field);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    // --- motion range. Angles are shown in degrees and stored in radians,
    // the same split the Robot section's sliders make.
    if (config.type == JointConfig::Type::Distance) {
        floatField(
                "Min Distance", config.lower, 0.01f, 0.f, 1000.f, "%.2f m",
                [](JointConfig& c, float v) { c.lower = v; }, "Joint Min Distance", "lower");
        floatField(
                "Max Distance", config.upper, 0.01f, 0.f, 1000.f, "%.2f m",
                [](JointConfig& c, float v) { c.upper = v; }, "Joint Max Distance", "upper");
    } else if (config.type != JointConfig::Type::Fixed) {
        bool limited = config.limited;
        if (ImGui::Checkbox("Limited", &limited)) {
            auto after = config;
            after.limited = limited;
            commit(after, "Joint Limited", "limited");
            config.limited = limited;
        }
        if (config.limited) {
            switch (config.type) {
                case JointConfig::Type::Revolute:
                    floatField(
                            "Lower", math::radToDeg(config.lower), 0.5f, -360.f, 360.f, "%.1f deg",
                            [](JointConfig& c, float v) { c.lower = math::degToRad(v); },
                            "Joint Lower Limit", "lower");
                    floatField(
                            "Upper", math::radToDeg(config.upper), 0.5f, -360.f, 360.f, "%.1f deg",
                            [](JointConfig& c, float v) { c.upper = math::degToRad(v); },
                            "Joint Upper Limit", "upper");
                    break;
                case JointConfig::Type::Prismatic:
                    floatField(
                            "Lower", config.lower, 0.01f, -100.f, 100.f, "%.3f m",
                            [](JointConfig& c, float v) { c.lower = v; },
                            "Joint Lower Limit", "lower");
                    floatField(
                            "Upper", config.upper, 0.01f, -100.f, 100.f, "%.3f m",
                            [](JointConfig& c, float v) { c.upper = v; },
                            "Joint Upper Limit", "upper");
                    break;
                default:// Spherical: the swing cone
                    floatField(
                            "Cone Y", math::radToDeg(config.coneY), 0.5f, 1.f, 179.f, "%.1f deg",
                            [](JointConfig& c, float v) { c.coneY = math::degToRad(v); },
                            "Joint Cone Y", "coney");
                    floatField(
                            "Cone Z", math::radToDeg(config.coneZ), 0.5f, 1.f, 179.f, "%.1f deg",
                            [](JointConfig& c, float v) { c.coneZ = math::degToRad(v); },
                            "Joint Cone Z", "conez");
                    break;
            }
        }
    }

    // --- drive. "Driven" is not a stored flag — it IS stiffness/damping > 0,
    // which is exactly the condition under which the play session builds a
    // drive. The checkbox seeds a WORKING drive (the articulation defaults)
    // rather than leaving the sliders at zero: these are logarithmic drags,
    // and a log drag anchored at 0 compresses so hard near the bottom that it
    // can never escape it — the fields read 0 or 0.001 forever. Seeded, they
    // drag normally; dragged back to zero, the joint is passive again and the
    // checkbox reads accordingly.
    if (config.type != JointConfig::Type::Fixed) {
        ImGui::Spacing();
        const bool distance = config.type == JointConfig::Type::Distance;
        bool driven = config.stiffness > 0.f || config.damping > 0.f;
        if (ImGui::Checkbox(distance ? "Spring" : "Driven", &driven)) {
            auto after = config;
            after.stiffness = driven ? 500.f : 0.f;
            after.damping = driven ? 50.f : 0.f;
            commit(after, driven ? (distance ? "Joint Spring" : "Joint Driven") : "Joint Passive",
                   "driven");
            config = after;// so the fields below appear/vanish this frame
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip(distance
                                      ? "Softens the tether: stiffness/damping act as its spring.\n"
                                        "Off, the max distance is a hard stop."
                                      : "PD drive: stiffness pulls toward Target, damping toward\n"
                                        "Velocity. Off, the joint is passive.");
        }
        if (driven) {
            floatField(
                    "Stiffness", config.stiffness, 1.f, 0.f, 100000.f, "%.3f",
                    [](JointConfig& c, float v) { c.stiffness = v; }, "Joint Stiffness", "stiffness",
                    ImGuiSliderFlags_Logarithmic);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Target acts through this. Ctrl+click to type a value.");
            }
            floatField(
                    "Damping", config.damping, 0.5f, 0.f, 10000.f, "%.3f",
                    [](JointConfig& c, float v) { c.damping = v; }, "Joint Damping", "damping",
                    ImGuiSliderFlags_Logarithmic);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Velocity acts through this. Ctrl+click to type a value.");
            }
            if (!distance) {
                floatField(
                        "Max Force", config.maxForce, 100.f, 0.f, 1e7f, "%.3f",
                        [](JointConfig& c, float v) { c.maxForce = v; }, "Joint Max Force",
                        "maxforce", ImGuiSliderFlags_Logarithmic);
            }
            if (config.type == JointConfig::Type::Revolute) {
                floatField(
                        "Target", math::radToDeg(config.target), 0.5f, -360.f, 360.f, "%.1f deg",
                        [](JointConfig& c, float v) { c.target = math::degToRad(v); },
                        "Joint Target", "target");
                floatField(
                        "Velocity", math::radToDeg(config.velocity), 1.f, -3600.f, 3600.f, "%.1f deg/s",
                        [](JointConfig& c, float v) { c.velocity = math::degToRad(v); },
                        "Joint Velocity", "velocity");
            } else if (config.type == JointConfig::Type::Prismatic) {
                floatField(
                        "Target", config.target, 0.01f, -100.f, 100.f, "%.3f m",
                        [](JointConfig& c, float v) { c.target = v; }, "Joint Target", "target");
                floatField(
                        "Velocity", config.velocity, 0.01f, -100.f, 100.f, "%.3f m/s",
                        [](JointConfig& c, float v) { c.velocity = v; }, "Joint Velocity", "velocity");
            } else if (config.type == JointConfig::Type::Spherical) {
                ImGui::TextColored(theme::muted(), "Springs back to the frame's rest orientation.");
            }
        }
    }

    // --- failure. Same derived-checkbox pattern as the drive, for the same
    // log-drag-from-zero reason; the seed is sturdy but within reach of a hard
    // crash on a metre-scale prop.
    {
        ImGui::Spacing();
        bool breakable = config.breakForce > 0.f || config.breakTorque > 0.f;
        if (ImGui::Checkbox("Breakable", &breakable)) {
            auto after = config;
            after.breakForce = breakable ? 1e4f : 0.f;
            after.breakTorque = breakable ? 1e4f : 0.f;
            commit(after, breakable ? "Joint Breakable" : "Joint Unbreakable", "breakable");
            config = after;
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("The joint separates for good past these solver loads.\n"
                              "Off = unbreakable.");
        }
        if (breakable) {
            floatField(
                    "Break Force", config.breakForce, 10.f, 0.f, 1e9f, "%.3f",
                    [](JointConfig& c, float v) { c.breakForce = v; }, "Joint Break Force",
                    "breakforce", ImGuiSliderFlags_Logarithmic);
            floatField(
                    "Break Torque", config.breakTorque, 10.f, 0.f, 1e9f, "%.3f",
                    [](JointConfig& c, float v) { c.breakTorque = v; }, "Joint Break Torque",
                    "breaktorque", ImGuiSliderFlags_Logarithmic);
        }
    }

    bool collide = config.collide;
    if (ImGui::Checkbox("Collide", &collide)) {
        auto after = config;
        after.collide = collide;
        commit(after, "Joint Collide", "collide");
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Whether the two jointed bodies still collide with\n"
                          "each other. Off keeps contacts at the anchor from\n"
                          "fighting the constraint.");
    }

    ImGui::PopItemWidth();

#ifndef THREEPP_EDITOR_WITH_PHYSX
    ImGui::TextColored(theme::muted(), "Built without PhysX - authored and saved, not simulated.");
#endif

    ImGui::TreePop();
}


// ------------------------------------------------------------------- vehicle

void EditorApp::drawVehicleSection(Object3D& object) {

    const bool authored = VehicleConfig::isVehicle(object);

    // The section invites: any node with enough descendant meshes to pick
    // four wheels from gets the checkbox. A bare mesh or an empty group is
    // not a car, and stays uncluttered.
    if (!authored) {
        int meshes = 0;
        object.traverseType<Mesh>([&](Mesh& mesh) {
            if (&mesh != &object) ++meshes;
        });
        if (meshes < 4) return;
    }

    // Open for an authored vehicle — the section IS what the node is about —
    // collapsed while it is only the invitation on a plausible model.
    if (!section("Vehicle", authored)) return;

    auto* target = &object;

    // Presence is the identity (VehicleConfig's rule), so the checkbox moves
    // the whole authoring as one optional value — one undo step either way.
    bool simulate = authored;
    if (ImGui::Checkbox("Simulate as Vehicle", &simulate)) {
        const auto was = VehicleConfig::read(object);
        commands_.execute(makeProperty<std::optional<VehicleConfig>>(
                simulate ? "Add Vehicle" : "Remove Vehicle",
                "vehicle:" + object.uuid + ":presence",
                [target](const std::optional<VehicleConfig>& value) {
                    if (value) {
                        value->write(*target);
                    } else {
                        VehicleConfig::erase(*target);
                    }
                },
                was, simulate ? std::optional<VehicleConfig>(VehicleConfig{}) : std::nullopt));
        document_.setDirty(true);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Play builds a drivable PhysX vehicle from this model.\n"
                          "Pick the four wheel meshes below; everything geometric\n"
                          "is derived from them. Drive with W/S/A/D, SPACE brakes.");
    }

    if (!simulate) {
        ImGui::TextColored(theme::muted(), "Not simulated as a vehicle.");
        ImGui::TreePop();
        return;
    }

    auto config = VehicleConfig::read(object).value_or(VehicleConfig{});
    const auto before = config;

    const auto commit = [&](const VehicleConfig& after, const char* label, const char* field) {
        commands_.execute(makeProperty<VehicleConfig>(
                label, "vehicle:" + object.uuid + ":" + field,
                [target](const VehicleConfig& value) { value.write(*target); },
                before, after));
        document_.setDirty(true);
    };

    ImGui::PushItemWidth(-110 * contentScale_);

    // --- the four picks. Combos over every named descendant with mesh
    // geometry beneath it — groups included, because an imported wheel is
    // often an ASSEMBLY (rim + tire + caliper) whose group is the node that
    // measures right. The reference is a scene object, so choosing one that
    // exists is the feature (the Joint section's Body B rule). A repeated
    // name is listed (and stored) with an ordinal — "Wheel", "Wheel#2" — the
    // N-th node of that name in document order, so four wheels all called
    // "Wheel" are still four distinct picks. Wheel order is the runtime's:
    // FR, FL, RR, RL.
    static constexpr const char* wheelFields[4] = {"wheelfr", "wheelfl", "wheelrr", "wheelrl"};
    std::vector<std::string> wheelCandidates;
    {
        std::unordered_set<const Object3D*> withMesh;
        object.traverseType<Mesh>([&](Mesh& mesh) {
            for (Object3D* node = &mesh; node && node != &object; node = node->parent) {
                withMesh.insert(node);
            }
        });
        std::unordered_map<std::string, int> seen;
        object.traverse([&](Object3D& node) {
            if (&node == &object || node.name.empty()) return;
            if (!withMesh.count(&node)) return;
            const int n = ++seen[node.name];
            wheelCandidates.push_back(n == 1 ? node.name
                                             : node.name + "#" + std::to_string(n));
        });
    }
    // An imported car is hundreds of meshes, so each combo opens on a SEARCH
    // field: case-insensitive substring over the names, keyboard already in
    // the box, and Enter commits a lone match — "type fl, Enter" is the
    // intended gesture. One buffer serves all four combos, since only one
    // popup is ever open; it resets on every open.
    static char wheelSearch[64] = "";
    const auto matchesSearch = [](const std::string& name, const char* needle) {
        if (!*needle) return true;
        const auto lower = [](std::string text) {
            for (auto& ch : text) {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }
            return text;
        };
        return lower(name).find(lower(needle)) != std::string::npos;
    };
    for (int i = 0; i < 4; ++i) {
        const char* shown = config.wheels[i].empty() ? "(pick a wheel)" : config.wheels[i].c_str();
        if (ImGui::BeginCombo(VehicleConfig::wheelLabels[i], shown, ImGuiComboFlags_HeightLarge)) {
            if (ImGui::IsWindowAppearing()) {
                wheelSearch[0] = '\0';
                ImGui::SetKeyboardFocusHere();
            }
            ImGui::SetNextItemWidth(-1);
            const bool entered = ImGui::InputTextWithHint(
                    "##wheelSearch", "Search...", wheelSearch, sizeof(wheelSearch),
                    ImGuiInputTextFlags_EnterReturnsTrue);

            const auto pick = [&](const std::string& name) {
                auto after = config;
                after.wheels[i] = name;
                commit(after, "Vehicle Wheel", wheelFields[i]);
            };

            // "(none)" only on an empty search: once something is typed, the
            // list is candidates, and clearing the slot is not one.
            if (!wheelSearch[0] && ImGui::Selectable("(none)", config.wheels[i].empty())) {
                auto after = config;
                after.wheels[i].clear();
                commit(after, "Vehicle Wheel", wheelFields[i]);
            }
            const std::string* lone = nullptr;
            int matched = 0;
            for (const auto& name : wheelCandidates) {
                if (!matchesSearch(name, wheelSearch)) continue;
                ++matched;
                lone = matched == 1 ? &name : nullptr;
                if (ImGui::Selectable(name.c_str(), name == config.wheels[i])) {
                    pick(name);
                }
            }
            if (matched == 0) {
                ImGui::TextColored(theme::muted(), "No node name matches.");
            }
            if (entered && lone) {
                pick(*lone);
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndCombo();
        }
    }

    // --- drive train. Automatic all the way: the engine drive keeps its
    // autobox, and no manual-shift control exists anywhere in the editor.
    if (ImGui::BeginCombo("Drive", VehicleConfig::label(config.drive))) {
        for (const auto drive : VehicleConfig::drives) {
            if (ImGui::Selectable(VehicleConfig::label(drive), drive == config.drive)) {
                auto after = config;
                after.drive = drive;
                commit(after, "Vehicle Drive", "drive");
                config.drive = drive;// so the widgets below reflect it this frame
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Direct: throttle torque straight to the wheels - simple and robust.\n"
                          "Engine: engine/clutch/gearbox chain with an automatic gearbox.");
    }
    if (ImGui::BeginCombo("Driven Wheels", VehicleConfig::label(config.driven))) {
        for (const auto driven : VehicleConfig::drivens) {
            if (ImGui::Selectable(VehicleConfig::label(driven), driven == config.driven)) {
                auto after = config;
                after.driven = driven;
                commit(after, "Vehicle Driven Wheels", "driven");
            }
        }
        ImGui::EndCombo();
    }

    const auto floatField = [&](const char* label, float value, float speed, float min, float max,
                                const char* format, void (*assign)(VehicleConfig&, float),
                                const char* action, const char* field) {
        float edited = value;
        const bool changed = ImGui::DragFloat(label, &edited, speed, min, max, format);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            assign(after, edited);
            commit(after, action, field);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    // --- geometry. Derived from the picks while Auto is on, and shown, so
    // the numbers Play will use are never a mystery. Unticking seeds every
    // field with the derived values so nothing jumps — the Joint section's
    // Driven-checkbox pattern.
    ImGui::Spacing();
    const auto geo = config.derived(object);

    bool autoGeo = config.autoGeometry;
    if (ImGui::Checkbox("Auto Geometry", &autoGeo)) {
        auto after = config;
        after.autoGeometry = autoGeo;
        if (!autoGeo && geo.valid) {
            after.chassisWidth = geo.chassisWidth;
            after.chassisHeight = geo.chassisHeight;
            after.chassisLength = geo.chassisLength;
            after.wheelRadius = geo.wheelRadius;
            after.wheelWidth = geo.wheelWidth;
            after.trackWidth = geo.trackWidth;
            after.wheelbase = geo.wheelbase;
            after.suspensionY = geo.suspensionY;
        }
        commit(after, autoGeo ? "Vehicle Auto Geometry" : "Vehicle Manual Geometry", "auto");
        config = after;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Read chassis dimensions, wheel radius, track and wheelbase\n"
                          "off the picked wheel meshes. Untick to type them instead\n"
                          "(seeded with the derived values).");
    }
    if (config.autoGeometry) {
        if (geo.valid) {
            ImGui::TextColored(theme::muted(), "Chassis %.2f x %.2f x %.2f m",
                               static_cast<double>(geo.chassisWidth),
                               static_cast<double>(geo.chassisHeight),
                               static_cast<double>(geo.chassisLength));
            ImGui::TextColored(theme::muted(), "Wheel r %.2f m, track %.2f m, wheelbase %.2f m",
                               static_cast<double>(geo.wheelRadius),
                               static_cast<double>(geo.trackWidth),
                               static_cast<double>(geo.wheelbase));
        } else {
            ImGui::TextColored(theme::warning(), "%s", geo.problem.c_str());
        }
    } else {
        floatField(
                "Chassis Width", config.chassisWidth, 0.01f, 0.1f, 10.f, "%.2f m",
                [](VehicleConfig& c, float v) { c.chassisWidth = v; },
                "Vehicle Chassis Width", "chassiswidth");
        floatField(
                "Chassis Height", config.chassisHeight, 0.01f, 0.1f, 10.f, "%.2f m",
                [](VehicleConfig& c, float v) { c.chassisHeight = v; },
                "Vehicle Chassis Height", "chassisheight");
        floatField(
                "Chassis Length", config.chassisLength, 0.01f, 0.1f, 30.f, "%.2f m",
                [](VehicleConfig& c, float v) { c.chassisLength = v; },
                "Vehicle Chassis Length", "chassislength");
        floatField(
                "Wheel Radius", config.wheelRadius, 0.005f, 0.01f, 3.f, "%.3f m",
                [](VehicleConfig& c, float v) { c.wheelRadius = v; },
                "Vehicle Wheel Radius", "wheelradius");
        floatField(
                "Wheel Width", config.wheelWidth, 0.005f, 0.02f, 2.f, "%.3f m",
                [](VehicleConfig& c, float v) { c.wheelWidth = v; },
                "Vehicle Wheel Width", "wheelwidth");
        floatField(
                "Track Width", config.trackWidth, 0.01f, 0.1f, 10.f, "%.2f m",
                [](VehicleConfig& c, float v) { c.trackWidth = v; },
                "Vehicle Track Width", "track");
        floatField(
                "Wheelbase", config.wheelbase, 0.01f, 0.1f, 20.f, "%.2f m",
                [](VehicleConfig& c, float v) { c.wheelbase = v; },
                "Vehicle Wheelbase", "wheelbase");
        floatField(
                "Suspension Y", config.suspensionY, 0.005f, -5.f, 5.f, "%.3f m",
                [](VehicleConfig& c, float v) { c.suspensionY = v; },
                "Vehicle Suspension Y", "suspensiony");
    }

    // --- the always-authored scalars.
    ImGui::Spacing();
    floatField(
            "Mass", config.mass, 5.f, 1.f, 50000.f, "%.0f kg",
            [](VehicleConfig& c, float v) { c.mass = v; }, "Vehicle Mass", "mass");
    if (config.drive == VehicleConfig::Drive::Direct) {
        floatField(
                "Throttle Torque", config.throttleTorque, 10.f, 0.f, 20000.f, "%.0f N*m",
                [](VehicleConfig& c, float v) { c.throttleTorque = v; },
                "Vehicle Throttle Torque", "throttle");
    }
    floatField(
            "Tire Friction", config.tireFriction, 0.01f, 0.1f, 4.f, "%.2f",
            [](VehicleConfig& c, float v) { c.tireFriction = v; },
            "Vehicle Tire Friction", "friction");
    floatField(
            "Max Steer", math::radToDeg(config.maxSteerAngle), 0.5f, 5.f, 85.f, "%.1f deg",
            [](VehicleConfig& c, float v) { c.maxSteerAngle = math::degToRad(v); },
            "Vehicle Max Steer", "steer");
    floatField(
            "Brake Torque", config.maxBrakeTorque, 10.f, 0.f, 50000.f, "%.0f N*m",
            [](VehicleConfig& c, float v) { c.maxBrakeTorque = v; },
            "Vehicle Brake Torque", "brake");
    floatField(
            "Susp. Travel", config.suspensionTravel, 0.005f, 0.02f, 2.f, "%.3f m",
            [](VehicleConfig& c, float v) { c.suspensionTravel = v; },
            "Vehicle Suspension Travel", "travel");
    floatField(
            "Susp. Stiffness", config.suspensionStiffness, 100.f, 100.f, 500000.f, "%.0f N/m",
            [](VehicleConfig& c, float v) { c.suspensionStiffness = v; },
            "Vehicle Suspension Stiffness", "stiffness");
    floatField(
            "Susp. Damping", config.suspensionDamping, 10.f, 0.f, 100000.f, "%.0f Ns/m",
            [](VehicleConfig& c, float v) { c.suspensionDamping = v; },
            "Vehicle Suspension Damping", "damping");

    ImGui::PopItemWidth();

#ifndef THREEPP_EDITOR_WITH_PHYSX
    ImGui::TextColored(theme::muted(), "Built without PhysX - authored and saved, not simulated.");
#endif

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


// --------------------------------------------------------------------- text

void EditorApp::drawSoundSection(Object3D& object) {

    const auto current = SoundConfig::read(object);
    if (!current) return;
    if (!section("Sound")) return;

    auto* target = &object;
    const SoundConfig config = *current;
    const auto before = SoundAuthoring::read(object);

    // Both userData entries move as one value — see SoundAuthoring — so
    // "Remove sound" is a single undo step rather than two.
    const auto commit = [&](SoundConfig after, const char* label) {
        SoundAuthoring value{after, before.file};
        commands_.execute(makeProperty<SoundAuthoring>(
                label, "sound:" + object.uuid,
                [target](const SoundAuthoring& v) { v.write(*target); },
                before, std::move(value)));
        document_.setDirty(true);
    };

    // --- the file -----------------------------------------------------------
    const float buttonWidth = 90 * contentScale_;

    if (ImGui::Button(before.file.empty() ? "Load..." : "Change...", {buttonWidth, 0})) {
        // The dialog resolves a frame or more later; remember the object by
        // uuid, since a play/stop in between replaces the whole graph.
        soundTargetUuid_ = object.uuid;
        pendingDialog_ = PendingDialog::Sound;
        fileBrowser_.open("Load Sound", FileBrowser::Mode::Open,
                          settings_.soundDir.empty() ? assetDir_ : std::filesystem::path(settings_.soundDir),
                          formats::sounds());
    }
    ImGui::SameLine();
    if (before.file.empty()) {
        ImGui::TextColored(theme::warning(), "no file - nothing will play");
    } else {
        ImGui::TextUnformatted(std::filesystem::path(before.file).filename().string().c_str());
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", before.file.c_str());
    }

    // --- audition -----------------------------------------------------------
    // Edit mode only. The global rejectWhilePlaying gate does not cover this —
    // it is not a document mutation — so the button is simply not offered while
    // the scene's own sounds are playing.
#ifdef THREEPP_WITH_AUDIO
    if (!isPlaying()) {
        const bool auditioning = isAuditioning(object);
        const bool canPlay = !before.file.empty() && !auditionUnavailable_;
        if (!canPlay && !auditioning) ImGui::BeginDisabled();
        if (ImGui::Button(auditioning ? "Stop" : "Audition", {buttonWidth, 0})) {
            if (auditioning) {
                stopAudition();
            } else {
                startAudition(object);
            }
        }
        if (!canPlay && !auditioning) ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip(auditionUnavailable_
                                      ? "No audio device on this machine."
                                      : "Hear the file at the authored volume and rate,\n"
                                        "without pressing Play. Plays FLAT - the distance\n"
                                        "falloff is only applied during Play.");
        }
        if (auditioning) {
            ImGui::SameLine();
            ImGui::TextColored(theme::playing(), "auditioning");
        }
    }
#else
    ImGui::TextColored(theme::muted(), "Built without audio - authoring only.");
#endif

    ImGui::Spacing();
    ImGui::PushItemWidth(-140 * contentScale_);

    // --- playback -----------------------------------------------------------
    {
        bool positional = config.positional;
        if (ImGui::Checkbox("Positional", &positional)) {
            auto after = config;
            after.positional = positional;
            commit(after, positional ? "Positional Sound" : "Ambient Sound");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("On: the sound comes FROM this object and gets quieter with\n"
                              "distance. Off: it plays at a constant level wherever the\n"
                              "listener stands - music, room tone.");
        }
    }

    {
        bool autoplay = config.autoplay;
        if (ImGui::Checkbox("Play on Start", &autoplay)) {
            auto after = config;
            after.autoplay = autoplay;
            commit(after, autoplay ? "Enable Autoplay" : "Disable Autoplay");
        }
        ImGui::SameLine();
        bool loop = config.loop;
        if (ImGui::Checkbox("Loop", &loop)) {
            auto after = config;
            after.loop = loop;
            commit(after, "Sound Loop");
        }
    }

    {
        float volume = config.volume;
        const bool changed = ImGui::SliderFloat("Volume", &volume, 0.f, 1.f, "%.2f");
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.volume = std::clamp(volume, 0.f, 1.f);
            commit(after, "Sound Volume");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    {
        float rate = config.rate;
        const bool changed = ImGui::SliderFloat("Playback rate", &rate, 0.25f, 4.f, "%.2fx");
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.rate = std::clamp(rate, 0.25f, 4.f);
            commit(after, "Sound Rate");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Speed and pitch together, like a tape machine.");
        }
    }

    // --- distance falloff ---------------------------------------------------
    // Hidden, never dropped: switching Positional off and on again must not
    // reset a tuned curve (SoundConfig writes these entries regardless).
    if (config.positional) {

        ImGui::Spacing();

        // Under the None model there is no ramp — no start, no steepness. Grey
        // the two fields that only shape the ramp, so they cannot promise a
        // falloff the curve refuses; max distance stays live because the
        // session's audibility gate honours it under every model.
        const bool noFalloff = config.model == SoundConfig::DistanceModel::None;

        {
            if (noFalloff) ImGui::BeginDisabled();
            float minDistance = config.minDistance;
            const bool changed = ImGui::DragFloat("Min distance", &minDistance, 0.05f, 0.01f, 10000.f, "%.2f m");
            if (ImGui::IsItemActivated()) commands_.beginTransaction();
            if (changed) {
                auto after = config;
                after.minDistance = std::max(minDistance, 0.01f);
                commit(after, "Sound Min Distance");
            }
            if (ImGui::IsItemDeactivated()) commands_.endTransaction();
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Inside this radius the sound plays at full volume.");
            }
            if (noFalloff) ImGui::EndDisabled();
        }

        {
            float maxDistance = config.maxDistance;
            const bool changed = ImGui::DragFloat("Max distance", &maxDistance, 0.5f, 0.01f, 100000.f, "%.2f m");
            if (ImGui::IsItemActivated()) commands_.beginTransaction();
            if (changed) {
                auto after = config;
                after.maxDistance = std::max(maxDistance, after.minDistance);
                commit(after, "Sound Max Distance");
            }
            if (ImGui::IsItemDeactivated()) commands_.endTransaction();
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("The sound is silent past this distance, whatever the model\n"
                                  "- the last stretch eases out so crossing the ring never\n"
                                  "clicks. The default 10000 m is effectively no limit.");
            }
        }

        {
            if (noFalloff) ImGui::BeginDisabled();
            float rolloff = config.rolloff;
            const bool changed = ImGui::DragFloat("Rolloff", &rolloff, 0.01f, 0.f, 20.f, "%.2f");
            if (ImGui::IsItemActivated()) commands_.beginTransaction();
            if (changed) {
                auto after = config;
                after.rolloff = std::clamp(rolloff, 0.f, 20.f);
                commit(after, "Sound Rolloff");
            }
            if (ImGui::IsItemDeactivated()) commands_.endTransaction();
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("How fast the volume drops once you are past the min\n"
                                  "distance. Higher = steeper; 0 = no drop at all.");
            }
            if (noFalloff) ImGui::EndDisabled();
        }

        if (ImGui::BeginCombo("Distance model", SoundConfig::label(config.model))) {
            for (const auto model : SoundConfig::models) {
                if (ImGui::Selectable(SoundConfig::label(model), model == config.model)) {
                    auto after = config;
                    after.model = model;
                    commit(after, "Sound Distance Model");
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("The shape of the falloff curve between min and max.\n"
                              "Inverse is how real sound behaves; None keeps the volume\n"
                              "constant and is the honest way to say \"do not attenuate\".");
        }

        ImGui::TextColored(theme::muted(), "Rings show min and max.");
    }

    ImGui::PopItemWidth();
    ImGui::Spacing();

    if (ImGui::SmallButton("Remove sound")) {
        if (isAuditioning(object)) stopAudition();
        commands_.execute(makeProperty<SoundAuthoring>(
                "Remove Sound", "sound:" + object.uuid,
                [target](const SoundAuthoring& v) { v.write(*target); },
                before, SoundAuthoring{}));
        document_.setDirty(true);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Drops the sound entry and the file reference. The object stays.");
    }

    ImGui::TextColored(theme::muted(), "Stored in userData[\"sound\"] / [\"soundFile\"]");

    ImGui::TreePop();
}


// ---------------------------------------------------------------------- text

void EditorApp::drawTextSection(Object3D& object) {

    if (!TextConfig::isText(object)) return;
    if (!section("Text")) return;

    auto* target = &object;
    const auto config = TextConfig::read(object).value_or(TextConfig{});

    // Same shape every other config section uses — one undoable property write
    // per edit, merge-keyed so typing is not one undo step per keystroke. The
    // setter goes through apply(), so execute, undo and redo all rebuild the
    // geometry the entries describe.
    const auto commit = [&](TextConfig after, const char* label) {
        commands_.execute(makeProperty<TextConfig>(
                label, "text:" + object.uuid,
                [target](const TextConfig& value) { value.apply(*target); },
                config, std::move(after)));
        document_.setDirty(true);
    };

    // The content. Modest height: a label is a line or three, and the box
    // grows nothing by being tall.
    std::string buffer = config.text;
    buffer.resize(std::max<std::size_t>(buffer.size() + 512, 1024));
    if (ImGui::InputTextMultiline("##textContent", buffer.data(), buffer.size(),
                                  {-1.f, ImGui::GetTextLineHeight() * 3.5f})) {
        auto after = config;
        after.text = buffer.c_str();
        commit(std::move(after), "Edit Text");
    }
    if (config.text.empty()) {
        ImGui::TextColored(theme::muted(), "Empty text draws nothing.");
    }

    ImGui::PushItemWidth(-100 * contentScale_);

    {
        float size = config.size;
        const bool changed = ImGui::DragFloat("Size", &size, 0.01f, 0.01f, 100.f);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.size = std::clamp(size, 0.01f, 100.f);
            commit(std::move(after), "Text Size");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    {
        float depth = config.depth;
        const bool changed = ImGui::DragFloat("Depth", &depth, 0.005f, 0.f, 100.f);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.depth = std::clamp(depth, 0.f, 100.f);
            commit(std::move(after), "Text Depth");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
        if (config.depth <= 0.f) {
            ImGui::TextColored(theme::muted(), "Depth 0 is flat - a sign, not solid type.");
        }
    }

    {
        static const char* aligns[] = {"Left", "Center", "Right"};
        int align = static_cast<int>(config.align);
        if (ImGui::Combo("Align", &align, aligns, IM_ARRAYSIZE(aligns))) {
            auto after = config;
            after.align = static_cast<TextConfig::Align>(align);
            commit(std::move(after), "Text Align");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Where the origin sits on the block:\n"
                              "what the gizmo grabs, and what Position means.");
        }
    }

    {
        int segments = config.curveSegments;
        const bool changed = ImGui::DragInt("Curve Segments", &segments, 0.1f, 1,
                                            TextConfig::maxCurveSegments);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.curveSegments = std::clamp(segments, 1, TextConfig::maxCurveSegments);
            commit(std::move(after), "Text Curve Segments");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    ImGui::PopItemWidth();

    ImGui::TextColored(theme::muted(), "Stored in userData[\"text\"]");

    ImGui::TreePop();
}


// --------------------------------------------------------------------- tree

void EditorApp::drawTreeSection(Object3D& object) {

    if (!TreeConfig::isTree(object)) return;
    if (!section("Tree")) return;

    using vegetation::BarkStyle;
    using vegetation::BranchingMode;
    using vegetation::CrownShape;
    using vegetation::LeafShape;
    using vegetation::LeafStyle;
    using vegetation::TreeParams;

    auto* target = &object;
    const auto config = TreeConfig::read(object).value_or(TreeConfig{});

    // Only the config is edited here. The meshes are derived state that
    // syncTreeOverlays regrows to follow it — see TreeOverlay.cpp. That is also
    // what keeps undo cheap: the command carries forty floats, not a canopy.
    const auto commit = [&](TreeConfig after, std::string label) {
        commands_.execute(makeProperty<TreeConfig>(
                std::move(label), "tree:" + object.uuid,
                [target](const TreeConfig& value) { value.write(*target); },
                config, std::move(after)));
        document_.setDirty(true);
    };

    // The three widget shapes below are written once and driven by
    // pointer-to-member: the generator has forty-odd knobs, and forty
    // hand-inlined copies of the transaction dance is how one of them ends up
    // silently missing its beginTransaction.
    const auto sliderFloat = [&](const char* label, float TreeParams::* field,
                                 float min, float max, const char* format = "%.2f") {
        float value = config.params.*field;
        const bool changed = ImGui::SliderFloat(label, &value, min, max, format);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.params.*field = std::clamp(value, min, max);
            commit(std::move(after), std::string("Tree ") + label);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    const auto sliderInt = [&](const char* label, int TreeParams::* field, int min, int max) {
        int value = config.params.*field;
        const bool changed = ImGui::SliderInt(label, &value, min, max);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.params.*field = std::clamp(value, min, max);
            commit(std::move(after), std::string("Tree ") + label);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    const auto colorEdit = [&](const char* label, std::array<float, 3> TreeParams::* field) {
        std::array<float, 3> value = config.params.*field;
        const bool changed = ImGui::ColorEdit3(label, value.data());
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.params.*field = value;
            commit(std::move(after), std::string("Tree ") + label);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    // Combos are not draggable, so they need no transaction — one click, one
    // undo step.
    const auto combo = [&](const char* label, int current, const char* items,
                           const std::function<void(TreeConfig&, int)>& apply) {
        int value = current;
        if (ImGui::Combo(label, &value, items)) {
            auto after = config;
            apply(after, value);
            commit(std::move(after), std::string("Tree ") + label);
        }
    };

    // --- species ----------------------------------------------------------
    // Buttons rather than a combo showing the current species: applying a
    // preset RESETS every field but the seed, so after one edit the params no
    // longer name a preset and a combo would have to lie about which.
    ImGui::TextUnformatted("Preset");
    for (int preset = 0; preset < TreeConfig::presetCount; ++preset) {
        ImGui::SameLine();
        if (ImGui::Button(TreeConfig::presetLabel(preset))) {
            auto after = config;
            vegetation::applyPreset(preset, after.params);
            commit(std::move(after), std::string("Tree ") + TreeConfig::presetLabel(preset));
        }
        // Inside the loop: IsItemHovered() reads the item just submitted, so
        // hanging one tooltip off the end would label only the last button.
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s. Replaces every parameter but the seed.",
                              TreeConfig::presetLabel(preset));
        }
    }

    ImGui::PushItemWidth(-110 * contentScale_);

    // --- the individual ----------------------------------------------------
    {
        int seed = static_cast<int>(config.params.seed);
        if (ImGui::InputInt("Seed", &seed)) {
            auto after = config;
            after.params.seed = static_cast<unsigned int>(std::max(seed, 0));
            commit(std::move(after), "Tree Seed");
        }
        ImGui::SameLine();
        if (ImGui::Button("Randomize")) {
            auto after = config;
            after.params.seed = std::random_device{}();
            commit(std::move(after), "Tree Seed");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Another tree of the same species.");
        }
    }

    ImGui::SeparatorText("Trunk");
    sliderFloat("Height", &TreeParams::trunkHeight, 0.5f, 12.f, "%.1f");
    sliderFloat("Radius", &TreeParams::trunkRadius, 0.02f, 0.5f, "%.3f");
    sliderFloat("Lean", &TreeParams::trunkLean, 0.f, 0.25f, "%.3f");
    sliderFloat("Bend", &TreeParams::trunkBend, 0.f, 3.f);
    sliderFloat("Twist", &TreeParams::trunkTwist, -1.5f, 1.5f);

    ImGui::SeparatorText("Crown");
    combo("Shape", static_cast<int>(config.params.crownShape),
          "Sphere\0Ellipsoid\0Cone\0Hemisphere\0Cylinder\0",
          [](TreeConfig& c, int v) { c.params.crownShape = static_cast<CrownShape>(v); });
    sliderFloat("Radius X", &TreeParams::crownRadiusX, 0.5f, 10.f, "%.1f");
    sliderFloat("Radius Z", &TreeParams::crownRadiusZ, 0.5f, 10.f, "%.1f");
    sliderFloat("Crown Height", &TreeParams::crownHeight, 1.f, 15.f, "%.1f");

    ImGui::SeparatorText("Branching");
    combo("Mode", static_cast<int>(config.params.branchingMode), "Colonise\0Whorl\0",
          [](TreeConfig& c, int v) { c.params.branchingMode = static_cast<BranchingMode>(v); });

    // The two skeleton builders take disjoint parameter sets, so showing both
    // at once would be forty knobs of which half do nothing.
    if (config.params.branchingMode == BranchingMode::Colonise) {
        sliderInt("Attractors", &TreeParams::attractorCount, 100, 3000);
        sliderFloat("Influence Dist", &TreeParams::influenceDistance, 1.f, 10.f, "%.1f");
        sliderFloat("Kill Dist", &TreeParams::killDistance, 0.2f, 3.f);
        sliderFloat("Segment Length", &TreeParams::segmentLength, 0.1f, 1.5f);
        sliderInt("Max Iterations", &TreeParams::maxIterations, 50, 500);
        sliderFloat("Randomness", &TreeParams::randomness, 0.f, 0.3f, "%.3f");
        sliderFloat("Tropism", &TreeParams::tropism, -0.2f, 0.1f, "%.3f");
    } else {
        sliderFloat("Whorl Spacing", &TreeParams::whorlSpacing, 0.3f, 1.5f);
        sliderInt("Branches / Whorl", &TreeParams::branchesPerWhorl, 2, 10);
        sliderFloat("Whorl Jitter", &TreeParams::whorlJitter, 0.f, 1.f);
        sliderFloat("Branch Droop", &TreeParams::branchDroop, 0.f, 1.f);
        sliderFloat("Tip Upturn", &TreeParams::branchTipUpturn, 0.f, 1.f);
        sliderFloat("Crown Profile", &TreeParams::crownProfileExponent, 0.5f, 3.f);
        sliderFloat("Side Twigs", &TreeParams::sideTwigDensity, 0.f, 1.f);
        sliderFloat("Segment Length", &TreeParams::segmentLength, 0.1f, 1.5f);
    }

    sliderFloat("Radius Exponent", &TreeParams::radiusExponent, 1.5f, 4.f, "%.1f");
    sliderFloat("Min Branch Radius", &TreeParams::minBranchRadius, 0.001f, 0.02f, "%.4f");
    sliderInt("Radial Segments", &TreeParams::radialSegments, 3, 12);

    ImGui::SeparatorText("Bark");
    combo("Bark Style", static_cast<int>(config.params.barkStyle), "Furrowed\0Plated\0Papery\0",
          [](TreeConfig& c, int v) { c.params.barkStyle = static_cast<BarkStyle>(v); });
    sliderFloat("Bark Bump", &TreeParams::barkBumpAmp, 0.f, 0.3f, "%.3f");
    sliderInt("Bark Lobes", &TreeParams::barkBumpLobes, 2, 12);
    sliderFloat("Root Flare", &TreeParams::rootFlareAsym, 0.f, 1.f);

    ImGui::SeparatorText("Leaves");
    combo("Leaf Style", static_cast<int>(config.params.leafStyle),
          "Quad\0Cluster\0CrossQuad\0Blob\0Frond\0",
          [](TreeConfig& c, int v) { c.params.leafStyle = static_cast<LeafStyle>(v); });
    // Frond draws the needle atlas, which has no blade outline to pick.
    if (config.params.leafStyle != LeafStyle::Frond) {
        combo("Leaf Shape", static_cast<int>(config.params.leafShape),
              "Ovate\0Lobed\0Serrate\0Lanceolate\0",
              [](TreeConfig& c, int v) { c.params.leafShape = static_cast<LeafShape>(v); });
    }
    sliderFloat("Leaf Size", &TreeParams::leafSize, 0.05f, 1.f);
    sliderFloat("Leaf Density", &TreeParams::leafDensity, 0.f, 1.f);
    sliderFloat("Clumping", &TreeParams::leafClumping, 0.f, 0.9f);
    sliderInt("Per Cluster", &TreeParams::leavesPerCluster, 1, 10);
    sliderFloat("Leaf Spread", &TreeParams::leafSpread, 0.f, 1.f);
    sliderFloat("Foliage AO", &TreeParams::foliageOcclusion, 0.f, 1.f);

    ImGui::SeparatorText("Colors");
    colorEdit("Bark Color", &TreeParams::barkColor);
    colorEdit("Leaf Color", &TreeParams::leafColor);

    ImGui::PopItemWidth();

    // What the config actually grew, since a canopy is hard to count by eye
    // and the parameters that drive the cost are not the obvious ones.
    {
        const auto triangles = [](const Object3D* node) {
            const auto* mesh = node ? node->as<Mesh>() : nullptr;
            const auto geometry = mesh ? mesh->geometry() : nullptr;
            if (!geometry) return 0;
            if (const auto* index = geometry->getIndex()) return static_cast<int>(index->count() / 3);
            const auto* position = geometry->getAttribute<float>("position");
            return position ? static_cast<int>(position->count() / 3) : 0;
        };
        ImGui::TextColored(theme::muted(), "%d + %d triangles (trunk + leaves)",
                           triangles(TreeConfig::derivedPart(object, TreeConfig::Part::Trunk)),
                           triangles(TreeConfig::derivedPart(object, TreeConfig::Part::Leaves)));
    }
    ImGui::TextColored(theme::muted(), "Bark and leaf textures redraw when you let go.");
    ImGui::TextColored(theme::muted(), "Stored in userData[\"tree\"]");

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

        // The grow loop lives HERE, on the point being dragged: Insert After,
        // drag the new point where the wall should reach next, repeat.
        const auto uuid = wall->uuid;
        const auto insert = [this, uuid](std::size_t slot, const char* label) {
            deferred_ = [this, uuid, slot, label] {
                if (auto* live = findByUuid(document_.scene(), uuid)) {
                    addConveyorWallPoint(*live, slot, label);
                }
            };
        };
        if (ImGui::Button("Insert Before")) insert(index, "Insert Wall Point");
        ImGui::SameLine();
        if (ImGui::Button("Insert After")) insert(index + 1, "Insert Wall Point");
        ImGui::SameLine();
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
                    addConveyorWallPoint(*live, AddObjectCommand::atEnd, "Add Wall Point");
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
                           "Slide the whole wall along the belt with the gizmo, then grow");
        ImGui::TextColored(theme::muted(),
                           "it point by point (Insert After on a point). A point pulled");
        ImGui::TextColored(theme::muted(), "toward the middle sweeps that stretch into a diverter.");
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


// ---------------------------------------------------------------- particles

void EditorApp::drawParticleFieldSection(Object3D& object) {

    if (!ParticleFieldConfig::isParticleField(object)) return;
    if (!section("Particle Field")) return;

    using Config = ParticleFieldConfig;

    auto* target = &object;
    const auto config = Config::read(object).value_or(Config{});

    // Only the config is edited here. The preview FIELD is derived state
    // syncParticleOverlays follows — which is also what keeps undo cheap and
    // what makes a structural edit (capacity, radius, proxy, resolution) an
    // ordinary property write rather than a special case.
    const auto commit = [&](Config after, std::string label) {
        commands_.execute(makeProperty<Config>(
                std::move(label), "particles:" + object.uuid,
                [target](const Config& value) { value.write(*target); },
                config, std::move(after)));
        document_.setDirty(true);
    };

    // The widget shapes, written once and driven by pointer-to-member: this
    // config has seventy knobs, and seventy hand-inlined copies of the
    // transaction dance is how one of them ends up missing its
    // beginTransaction (the tree section's argument, at twice the scale).
    const auto dragFloat = [&](const char* label, float Config::* field, float step,
                               float lo, float hi, const char* format = "%.3f") {
        float value = config.*field;
        const bool changed = ImGui::DragFloat(label, &value, step, lo, hi, format);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.*field = std::clamp(value, lo, hi);
            commit(std::move(after), std::string("Particles ") + label);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    const auto dragVec3 = [&](const char* label, Vector3 Config::* field, float step,
                              float lo, float hi) {
        const Vector3& current = config.*field;
        float value[3]{current.x, current.y, current.z};
        const bool changed = ImGui::DragFloat3(label, value, step, lo, hi, "%.3f");
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            (after.*field).set(std::clamp(value[0], lo, hi), std::clamp(value[1], lo, hi),
                               std::clamp(value[2], lo, hi));
            commit(std::move(after), std::string("Particles ") + label);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    const auto colorField = [&](const char* label, Color Config::* field) {
        float value[3];
        toSrgbFloats(config.*field, value);
        const bool changed = ImGui::ColorEdit3(label, value);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.*field = fromSrgbFloats(value);
            commit(std::move(after), std::string("Particles ") + label);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    // Checkboxes are one click, one undo step — no transaction.
    const auto toggle = [&](const char* label, bool Config::* field) {
        bool value = config.*field;
        if (ImGui::Checkbox(label, &value)) {
            auto after = config;
            after.*field = value;
            commit(std::move(after), std::string("Particles ") + label);
        }
        return config.*field;
    };

    // --- presets ----------------------------------------------------------
    // Buttons rather than a combo showing the current one: a preset replaces
    // every field, so after one edit the config no longer names a preset and a
    // combo would have to lie about which.
    struct Preset {
        const char* label;
        Config (*make)();
    };
    static constexpr Preset presets[]{{"Snow", &Config::snow},
                                      {"Rain", &Config::rain},
                                      {"Embers", &Config::embers},
                                      {"Motes", &Config::motes}};
    ImGui::TextUnformatted("Preset");
    for (const auto& preset : presets) {
        ImGui::SameLine();
        if (ImGui::Button(preset.label)) {
            commit(preset.make(), std::string("Particles ") + preset.label);
        }
        // Inside the loop: IsItemHovered() reads the item just submitted.
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s. Four points in one parameter space - this replaces "
                              "every field.",
                              preset.label);
        }
    }

    if (!particlePreviewAvailable()) {
        ImGui::TextColored(theme::muted(),
                           "Particles render on the Vulkan backend only - preview disabled (--vulkan)");
    }

    ImGui::PushItemWidth(-130 * contentScale_);

    // --- structural -------------------------------------------------------
    // A field is created once at its final capacity and never resized, and a
    // density volume's resolution is latched at first enable, so these four
    // destroy and rebuild the preview instead of being pushed into it. Edits
    // still just work; the rebuild absorbs them.
    ImGui::SeparatorText("Rebuild");
    {
        int capacity = config.capacity;
        const bool changed = ImGui::DragInt("Capacity", &capacity, 250.f, 1, 5000000);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.capacity = std::max(capacity, 1);
            commit(std::move(after), "Particles Capacity");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }
    dragFloat("Radius", &Config::radius, 0.001f, 0.0001f, 1.f, "%.4f");
    {
        int proxy = static_cast<int>(config.proxy);
        if (ImGui::Combo("Proxy", &proxy, "None\0Sphere\0Flake\0")) {
            auto after = config;
            after.proxy = static_cast<Config::Proxy>(proxy);
            commit(std::move(after), "Particles Proxy");
        }
    }
    {
        int resolution = config.densityResolution;
        const bool changed = ImGui::DragInt("Density Res", &resolution, 1.f, 8, 256);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.densityResolution = std::clamp(resolution, 8, 256);
            commit(std::move(after), "Particles Density Res");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }

    // --- emitter ----------------------------------------------------------
    ImGui::SeparatorText("Emitter");
    dragVec3("Velocity", &Config::velocity, 0.05f, -100.f, 100.f);
    dragFloat("Speed Spread", &Config::speedSpread, 0.01f, 0.f, 10.f);
    dragVec3("Accel", &Config::accel, 0.01f, -50.f, 50.f);
    dragVec3("Wind", &Config::wind, 0.01f, -50.f, 50.f);
    dragVec3("Spawn Extent", &Config::spawnHalfExtent, 0.05f, 0.001f, 500.f);
    ImGui::TextColored(theme::muted(),
                       "A THIN slab swept over velocity x lifetime is the steady cloud.");
    dragFloat("Lifetime", &Config::lifetime, 0.05f, 0.001f, 600.f, "%.2f");
    dragFloat("Life Jitter", &Config::lifetimeJitter, 0.01f, 0.f, 1.f);
    dragFloat("Duty Cycle", &Config::dutyCycle, 0.01f, 0.001f, 1.f);
    dragFloat("Size", &Config::size, 0.001f, 0.f, 10.f, "%.4f");
    dragFloat("Size Jitter", &Config::sizeJitter, 0.01f, 0.f, 1.f);
    dragFloat("Drift Amplitude", &Config::driftAmplitude, 0.01f, 0.f, 20.f);
    dragFloat("Drift Frequency", &Config::driftFrequency, 0.01f, 0.f, 20.f);
    dragFloat("Drift Growth", &Config::driftGrowth, 0.01f, 0.f, 1.f);
    dragFloat("Drift Scale", &Config::driftScale, 0.1f, 0.f, 200.f, "%.2f");
    {
        int seed = config.seed;
        if (ImGui::InputInt("Seed", &seed)) {
            auto after = config;
            after.seed = std::max(seed, 0);
            commit(std::move(after), "Particles Seed");
        }
    }
    if (toggle("Follow Camera", &Config::follow)) {
        dragFloat("Follow Snap", &Config::followSnap, 0.05f, 0.f, 100.f, "%.2f");
        ImGui::TextColored(theme::muted(),
                           "Snap an INTEGER number of density voxels, or the haze swims.");
    }

    // --- surface landing --------------------------------------------------
    ImGui::SeparatorText("Surface Landing");
    if (toggle("Land on Surfaces", &Config::surface)) {
        dragFloat("Rest Seconds", &Config::surfaceRest, 0.05f, 0.f, 60.f, "%.2f");
        dragFloat("Rest Jitter", &Config::surfaceRestJitter, 0.01f, 0.f, 1.f);
        dragFloat("Fade Seconds", &Config::surfaceFade, 0.05f, 0.f, 60.f, "%.2f");
        dragFloat("Bias", &Config::surfaceBias, 0.001f, 0.f, 1.f, "%.4f");
        dragFloat("Splash Seconds", &Config::surfaceSplash, 0.01f, 0.f, 10.f);
        if (config.surfaceSplash > 0.f) {
            dragFloat("Splash Grow", &Config::surfaceSplashGrow, 0.1f, 1.f, 64.f, "%.2f");
        }
        {
            int resolution = config.surfaceResolution;
            const bool changed = ImGui::DragInt("Bake Res", &resolution, 1.f, 16, 1024);
            if (ImGui::IsItemActivated()) commands_.beginTransaction();
            if (changed) {
                auto after = config;
                after.surfaceResolution = std::clamp(resolution, 16, 1024);
                commit(std::move(after), "Particles Bake Res");
            }
            if (ImGui::IsItemDeactivated()) commands_.endTransaction();
        }
        ImGui::TextColored(theme::muted(), "Particles rest and FADE; they do not pile up.");
    }

    // --- representations --------------------------------------------------
    ImGui::SeparatorText("Mesh Proxy");
    if (toggle("Draw Proxies", &Config::mesh)) {
        if (config.proxy == Config::Proxy::None) {
            ImGui::TextColored(theme::warning(), "Proxy is None - nothing to draw.");
        }
        dragFloat("LOD Far", &Config::meshLodFar, 0.1f, 0.f, 500.f, "%.2f");
        dragFloat("LOD Fade", &Config::meshLodFade, 0.1f, 0.f, 500.f, "%.2f");
        dragFloat("Near Cull", &Config::meshNearCull, 0.05f, 0.f, 50.f, "%.2f");
        if (config.billboard && config.meshLodFar > 0.f) {
            ImGui::TextColored(theme::muted(),
                               "Sprites fade in over the same band the proxies shrink out over.");
        }
    }

    ImGui::SeparatorText("Billboard");
    if (toggle("Draw Sprites", &Config::billboard)) {
        colorField("Hot Color", &Config::colorHot);
        colorField("Cool Color", &Config::colorCool);
        dragFloat("Intensity", &Config::billboardIntensity, 0.01f, 0.f, 100.f, "%.3f");
        dragFloat("Sprite Size", &Config::billboardSize, 0.01f, 0.0001f, 50.f);
        dragFloat("Softness", &Config::billboardSoftness, 0.01f, 0.f, 1.f);
        dragFloat("Fade Power", &Config::billboardFade, 0.01f, 0.f, 8.f);
        dragFloat("Bright Jitter", &Config::billboardJitter, 0.01f, 0.f, 1.f);
        dragFloat("Size Taper", &Config::billboardTaper, 0.01f, 0.f, 1.f);
        dragFloat("Stretch (s)", &Config::billboardStretch, 0.001f, 0.f, 1.f, "%.4f");
        if (config.billboardStretch > 0.f) {
            dragFloat("Stretch Max", &Config::billboardStretchMax, 0.5f, 1.f, 200.f, "%.1f");
        }
        dragFloat("Near Fade", &Config::billboardNearFade, 0.01f, 0.f, 20.f);
        dragFloat("Glow", &Config::billboardGlow, 0.1f, 0.f, 64.f, "%.2f");
        if (config.billboardGlow > 0.f) {
            dragFloat("Glow Threshold", &Config::billboardGlowThreshold, 0.01f, 0.f, 20.f);
        }
        ImGui::TextColored(theme::muted(), "Additive and unlit - an ember IS the light source.");
    }

    ImGui::SeparatorText("Density Volume");
    if (toggle("Scatter into a Volume", &Config::density)) {
        dragFloat("Sigma / Particle", &Config::sigma, 0.001f, 0.0001f, 100.f, "%.4f");
        colorField("Albedo", &Config::albedo);
        dragFloat("Anisotropy", &Config::anisotropy, 0.01f, -0.95f, 0.95f);
        dragVec3("Volume Extent", &Config::densityHalfExtent, 0.1f, 0.001f, 500.f);

        // Warning only, and deliberately: the renderer REPORTS an overflow
        // rather than dropping a volume, so refusing the fifth here would be
        // stricter than the thing it is warning about.
        const int used = particleDensityCount();
        if (used > Config::maxDensityFields) {
            ImGui::TextColored(theme::warning(), "%d of %d density volumes in scene",
                               used, Config::maxDensityFields);
        } else {
            ImGui::TextColored(theme::muted(), "%d of %d density volumes in scene",
                               used, Config::maxDensityFields);
        }

        // Fire, and only fire, needs these — a dust or snow field has no
        // emission path at all while emissiveIntensity is 0.
        if (ImGui::TreeNodeEx("Advanced", ImGuiTreeNodeFlags_SpanAvailWidth)) {
            dragFloat("Emissive", &Config::emissiveIntensity, 0.5f, 0.f, 500.f, "%.1f");
            if (config.emissiveIntensity > 0.f) {
                dragFloat("Temp Bottom (K)", &Config::tempBottom, 10.f, 300.f, 6000.f, "%.0f");
                dragFloat("Temp Top (K)", &Config::tempTop, 10.f, 300.f, 6000.f, "%.0f");
                dragFloat("Temp Falloff", &Config::tempFalloff, 0.05f, 0.05f, 8.f, "%.2f");
            }
            ImGui::TreePop();
        }
    }

    ImGui::PopItemWidth();

    ImGui::TextColored(theme::muted(), "The node's transform IS the emitter frame.");
    ImGui::TextColored(theme::muted(), "Stored in userData[\"particles\"]");

    ImGui::TreePop();
}

void EditorApp::drawGranularSection(Object3D& object) {

    if (!GranularConfig::isGranular(object)) return;
    if (!section("Granular Particles")) return;

    using Config = GranularConfig;

    auto* target = &object;
    const auto config = Config::read(object).value_or(Config{});

    const auto commit = [&](Config after, std::string label) {
        commands_.execute(makeProperty<Config>(
                std::move(label), "granular:" + object.uuid,
                [target](const Config& value) { value.write(*target); },
                config, std::move(after)));
        document_.setDirty(true);
    };

    const auto dragFloat = [&](const char* label, float Config::* field, float step,
                               float lo, float hi, const char* format = "%.3f") {
        float value = config.*field;
        const bool changed = ImGui::DragFloat(label, &value, step, lo, hi, format);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.*field = std::clamp(value, lo, hi);
            commit(std::move(after), std::string("Granular ") + label);
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    };

    ImGui::PushItemWidth(-130 * contentScale_);

    ImGui::SeparatorText("Grains");
    // Grain diameter: the render radius is half of it and everything else
    // derives from it, which is why it is the first knob and not a detail.
    dragFloat("Spacing", &Config::spacing, 0.002f, 0.002f, 1.f, "%.4f");
    {
        int capacity = config.capacity;
        const bool changed = ImGui::DragInt("Capacity", &capacity, 250.f, 1, 4000000);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.capacity = std::max(capacity, 1);
            commit(std::move(after), "Granular Capacity");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }
    {
        int iterations = config.iterations;
        const bool changed = ImGui::DragInt("Iterations", &iterations, 0.25f, 1, 32);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.iterations = std::clamp(iterations, 1, 32);
            commit(std::move(after), "Granular Iterations");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }
    dragFloat("Max Velocity", &Config::maxVelocity, 0.1f, 0.f, 200.f, "%.2f");
    ImGui::TextColored(theme::muted(), "Max Velocity 0 derives a clamp from the spacing.");

    ImGui::SeparatorText("Material");
    // The repose angle of a heap IS its internal friction; cohesion is the
    // other half of how steep a pile stands.
    dragFloat("Friction", &Config::friction, 0.01f, 0.f, 2.f);
    dragFloat("Damping", &Config::damping, 0.01f, 0.f, 10.f);
    dragFloat("Adhesion", &Config::adhesion, 0.01f, 0.f, 10.f);
    dragFloat("Cohesion", &Config::cohesion, 0.01f, 0.f, 10.f);
    dragFloat("Viscosity", &Config::viscosity, 0.01f, 0.f, 10.f);
    dragFloat("Gravity Scale", &Config::gravityScale, 0.01f, -4.f, 4.f);

    ImGui::SeparatorText("Chute");
    dragFloat("Mouth X", &Config::emitExtentX, 0.01f, 0.001f, 20.f);
    dragFloat("Mouth Z", &Config::emitExtentZ, 0.01f, 0.001f, 20.f);
    dragFloat("Rate (/s)", &Config::rate, 25.f, 0.f, 500000.f, "%.0f");
    {
        const Vector3& current = config.emitVelocity;
        float value[3]{current.x, current.y, current.z};
        const bool changed = ImGui::DragFloat3("Pour Velocity", value, 0.05f, -50.f, 50.f, "%.3f");
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.emitVelocity.set(value[0], value[1], value[2]);
            commit(std::move(after), "Granular Pour Velocity");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }
    dragFloat("Mass", &Config::mass, 0.01f, 0.f, 100.f);
    dragFloat("Pour For (s)", &Config::emitFor, 0.1f, 0.f, 600.f, "%.2f");
    dragFloat("Lattice Jitter", &Config::jitter, 0.01f, 0.f, 1.f);
    ImGui::TextColored(theme::muted(), "Mass 0 = 1 kg; Pour For 0 = until capacity.");

    ImGui::SeparatorText("Visual");
    {
        int visual = static_cast<int>(config.visual);
        if (ImGui::Combo("Draw As", &visual, "Auto\0Instanced\0Field\0")) {
            auto after = config;
            after.visual = static_cast<Config::Visual>(visual);
            commit(std::move(after), "Granular Visual");
        }
    }
    {
        float value[3];
        toSrgbFloats(config.color, value);
        const bool changed = ImGui::ColorEdit3("Grain Color", value);
        if (ImGui::IsItemActivated()) commands_.beginTransaction();
        if (changed) {
            auto after = config;
            after.color = fromSrgbFloats(value);
            commit(std::move(after), "Granular Grain Color");
        }
        if (ImGui::IsItemDeactivated()) commands_.endTransaction();
    }
    dragFloat("Roughness", &Config::roughness, 0.01f, 0.f, 1.f);

    ImGui::PopItemWidth();

    ImGui::TextColored(theme::muted(), "The node's transform IS the chute frame; -Y pours down.");
    ImGui::TextColored(theme::muted(), "Grains are a PhysX PBD sim - they exist only while playing.");
    ImGui::TextColored(theme::muted(), "Stored in userData[\"granular\"]");

    ImGui::TreePop();
}


// ------------------------------------------------------------------- sensor

void EditorApp::drawSensorSection(Object3D& object) {

    if (object.is<Scene>()) return;
    if (!section("Sensor", false)) return;

    auto* target = &object;
    // The host gates what can live here: a Camera hosts the pinhole sensors —
    // its frustum IS their pose and optics — and everything else hosts the
    // rest. Authoring-side only: SensorPlaySession builds whatever it finds,
    // so a scene written before the gate still plays.
    auto* hostCamera = object.as<PerspectiveCamera>();
    auto config = SensorConfig::read(object).value_or(SensorConfig{});
    // The camera is the frustum's source of truth and the string follows it —
    // passively, outside the command stack. The undoable edit is the camera's
    // (the Camera section above); this derived copy heals whenever the two are
    // on screen together, so undoing a FOV change heals right back too. Play
    // reads the camera directly either way (see SensorPlaySession::build).
    if (hostCamera && config.enabled && SensorConfig::isPinhole(config.type) && !isPlaying() &&
        (config.fovY != hostCamera->fov ||
         config.nearPlane != hostCamera->nearPlane ||
         config.farPlane != hostCamera->farPlane)) {
        config.fovY = hostCamera->fov;
        config.nearPlane = hostCamera->nearPlane;
        config.farPlane = hostCamera->farPlane;
        config.write(object);
        document_.setDirty(true);
    }
    const auto before = config;

    const auto commit = [&](SensorConfig after, const char* label) {
        // On a camera host the camera's frustum is the truth (the play session
        // reads it directly); stamp it into the flat string on every write so
        // userData never drifts from what Play will build.
        if (hostCamera && SensorConfig::isPinhole(after.type)) {
            after.fovY = hostCamera->fov;
            after.nearPlane = hostCamera->nearPlane;
            after.farPlane = hostCamera->farPlane;
        }
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
        // A camera's default sensor is itself: the colour camera. The generic
        // default (IMU) belongs to the link-shaped hosts.
        if (enabled && hostCamera && !SensorConfig::isPinhole(after.type)) {
            after.type = SensorConfig::Type::Camera;
        }
        commit(after, enabled ? "Add Sensor" : "Remove Sensor");
        // Re-read: the click already changed the document, and drawing the rest
        // of this frame from the pre-click value would show the wrong fields for
        // one frame (and, on Remove, fields for an entry that is gone).
        config = SensorConfig::read(object).value_or(SensorConfig{});
    }

    if (!config.enabled) {
        if (hostCamera) {
            ImGui::TextColored(theme::muted(),
                               "Records this camera's frustum. Aim the camera; the dock "
                               "shows exactly what it will capture.");
        } else {
            ImGui::TextColored(theme::muted(),
                               "Measures in this object's world frame. The transform gizmo aims it.");
        }
        ImGui::TreePop();
        return;
    }

    ImGui::PushItemWidth(-110 * contentScale_);

    {
        // Only what the host can carry — plus whatever is ALREADY here, so the
        // combo never lies about a legacy rig (a depth eye authored on a plain
        // object before the pinholes moved onto cameras).
        static constexpr SensorConfig::Type all[] = {
                SensorConfig::Type::Imu, SensorConfig::Type::Depth,
                SensorConfig::Type::Lidar, SensorConfig::Type::Encoder,
                SensorConfig::Type::Contact, SensorConfig::Type::ForceTorque,
                SensorConfig::Type::Camera};
        std::vector<SensorConfig::Type> offered;
        std::vector<const char*> names;
        int current = 0;
        for (const auto candidate : all) {
            const bool belongs = SensorConfig::isPinhole(candidate) == (hostCamera != nullptr);
            if (!belongs && candidate != config.type) continue;
            if (candidate == config.type) current = static_cast<int>(offered.size());
            offered.push_back(candidate);
            names.push_back(SensorConfig::label(candidate));
        }
        if (ImGui::Combo("Type", &current, names.data(), static_cast<int>(names.size()))) {
            auto after = config;
            after.type = offered[static_cast<std::size_t>(current)];
            // Only the type changes. Every other key is written regardless of
            // type (see SensorConfig), so the settings of the type being left
            // behind are still there when the user comes back to it.
            commit(after, "Sensor Type");
            config = SensorConfig::read(object).value_or(config);
        }
    }

    // The legacy shape: a pinhole authored on a plain object, from before these
    // moved onto cameras. It still plays — the gate is authoring-side — but
    // everything a camera host gives (the frustum helper, the dock preview,
    // fov/near/far in one place) is one click away.
    if (!hostCamera && SensorConfig::isPinhole(config.type)) {
        ImGui::TextColored(theme::warning(),
                           "%s sensors live on a Camera object now.",
                           SensorConfig::label(config.type));
        const auto uuid = object.uuid;
        if (ImGui::SmallButton("Move To Camera Child")) {
            // Deferred like every graph edit launched from a panel: the add
            // reshapes the hierarchy this frame is still drawing.
            deferred_ = [this, uuid] {
                if (auto* live = findByUuid(document_.scene(), uuid)) {
                    moveSensorToCameraChild(*live);
                }
            };
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
        // On a JOINT NODE the node itself is the reference: the play session
        // reads that joint and consults no name, so a picker here would be a
        // control wired to nothing.
        if (JointConfig::isJoint(object)) {
            ImGui::TextColored(theme::muted(), "Reads this node's own joint.");
            return;
        }
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
            // On a camera host the frustum is the camera's; only the image
            // dimensions are the sensor's own.
            if (!hostCamera) {
                floatField(
                        "FOV (deg)", config.fovY, 0.25f, 1.f, 179.f,
                        [](SensorConfig& c, float v) { c.fovY = v; }, "Depth FOV", "%.1f");
            }
            intField(
                    "Width", config.width, 1.f, 8, SensorConfig::maxImageSize,
                    [](SensorConfig& c, int v) { c.width = v; }, "Depth Width");
            intField(
                    "Height", config.height, 1.f, 8, SensorConfig::maxImageSize,
                    [](SensorConfig& c, int v) { c.height = v; }, "Depth Height");
            break;
        }

        case SensorConfig::Type::Camera: {
            if (!hostCamera) {
                floatField(
                        "FOV (deg)", config.fovY, 0.25f, 1.f, 179.f,
                        [](SensorConfig& c, float v) { c.fovY = v; }, "Camera FOV", "%.1f");
            }
            intField(
                    "Width", config.width, 1.f, 8, SensorConfig::maxImageSize,
                    [](SensorConfig& c, int v) { c.width = v; }, "Camera Width");
            intField(
                    "Height", config.height, 1.f, 8, SensorConfig::maxImageSize,
                    [](SensorConfig& c, int v) { c.height = v; }, "Camera Height");
            ImGui::TextColored(theme::muted(),
                               "Looks down this object's -Z. Read it from a script with "
                               "editor.camera_from_object(obj).image");
            ImGui::TextColored(theme::muted(),
                               "Record writes one PNG per frame plus an index CSV.");
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
        if (hostCamera && SensorConfig::isPinhole(config.type)) {
            // One source of truth: the Camera section above is where the
            // frustum is edited, so these numbers are shown, not editable —
            // two drag fields for the same plane would fight.
            ImGui::TextColored(theme::muted(),
                               "Frustum from this camera: FOV %.1f deg, near %.3f, far %.2f.",
                               hostCamera->fov, hostCamera->nearPlane, hostCamera->farPlane);
            ImGui::TextColored(theme::muted(), "Edit them in the Camera section.");
        } else {
            floatField(
                    "Near (m)", config.nearPlane, 0.005f, 0.001f, 100.f,
                    [](SensorConfig& c, float v) { c.nearPlane = v; }, "Sensor Near", "%.3f");
            floatField(
                    "Far (m)", config.farPlane, 0.25f, 0.01f, 10000.f,
                    [](SensorConfig& c, float v) { c.farPlane = v; }, "Sensor Far", "%.2f");
            if (config.farPlane <= config.nearPlane) {
                ImGui::TextColored(theme::warning(), "Far must be beyond Near");
            }
        }
    }

    // Range noise is a RANGING sensor's, not every vision sensor's: a sigma in
    // metres has nothing to say about a colour pixel. The Camera shares the
    // frustum above and stops there.
    if (SensorConfig::isRanging(config.type)) {
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

namespace {

    // "Move To Camera Child" in ONE undo step. Three commands — add the
    // camera, write its config, clear the host's — would be three entries,
    // because the stack's transactions coalesce by merge key only (see
    // CommandStack::push), and a half-undone move is the worst of both
    // models: a scene with two of the sensor, or none. Same shape as
    // EditorApp.cpp's RegenerateCommand, and for the same reason.
    class MoveSensorToCameraCommand: public Command {

    public:
        MoveSensorToCameraCommand(Object3D& host, std::shared_ptr<Object3D> camera,
                                  SensorConfig config)
            : host_(&host), camera_(std::move(camera)), config_(config),
              hostUuid_(host.uuid), cameraUuid_(camera_->uuid) {}

        void redo() override {

            if (!host_ || !camera_) return;
            host_->add(camera_);
            config_.write(*camera_);
            SensorConfig::erase(*host_);
        }

        void undo() override {

            if (camera_) camera_->removeFromParent();
            if (host_) config_.write(*host_);
        }

        [[nodiscard]] std::string name() const override { return "Move Sensor To Camera"; }

        bool mergeWith(const Command&) override { return false; }

        // The camera is what the history keeps alive; whether it is currently
        // in the scene or held detached is the stack's to weigh.
        void retainedRoots(std::vector<Object3D*>& out) const override {

            if (camera_) out.push_back(camera_.get());
        }

        [[nodiscard]] bool rebind(Object3D& root) override {

            auto* host = findByUuid(root, hostUuid_);
            if (!host || !camera_) return false;
            host_ = host;
            // Same contract as AddObjectCommand: after a scene replace the
            // camera usually came back as a fresh instance behind the same
            // uuid, and undo must remove THAT one. Absent from the new graph
            // (the move currently undone), the retained instance stays the
            // target — the shared_ptr keeps it alive.
            if (auto* live = findByUuid(root, cameraUuid_)) {
                auto owned = live->weak_from_this().lock();
                if (!owned) return false;
                camera_ = std::move(owned);
            }
            return true;
        }

    private:
        Object3D* host_;
        std::shared_ptr<Object3D> camera_;
        SensorConfig config_;
        std::string hostUuid_;
        std::string cameraUuid_;
    };

}// namespace

void EditorApp::moveSensorToCameraChild(Object3D& host) {

    // The same gate addObject holds for every add; asked here because the
    // bespoke command below does not route through it.
    if (rejectWhilePlaying("Move Sensor To Camera")) return;

    const auto config = SensorConfig::read(host);
    if (!config || !config->enabled || !SensorConfig::isPinhole(config->type)) return;

    auto camera = ObjectFactory::createCamera(document_.scene());
    // At the host's origin, axes aligned: the host's transform is what aimed
    // this sensor until now, and identity here keeps that aim exactly.
    camera->position.set(0, 0, 0);
    // The authored numbers survive the move: the camera is stamped FROM the
    // config, and only from here on does the relationship invert.
    camera->fov = std::clamp(config->fovY, 1.f, 179.f);
    camera->nearPlane = std::max(config->nearPlane, 1e-3f);
    camera->farPlane = std::max(config->farPlane, camera->nearPlane + 1e-3f);
    camera->updateProjectionMatrix();

    auto* added = camera.get();
    commands_.execute(std::make_unique<MoveSensorToCameraCommand>(host, std::move(camera), *config));
    document_.setDirty(true);
    // What addObject would have done for the node it adds — and selecting a
    // camera is also what aims the dock at it, so the preview the legacy hint
    // promised is on screen when the click lands.
    selectObject(added);
    scrollTo_ = added;
}
