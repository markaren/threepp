
#include "../ConfigFields.hpp"
#include "../EditorApp.hpp"
#include "../EditorTheme.hpp"
#include "../ImportFormats.hpp"
#include "../PanelLayout.hpp"

#include "threepp/extras/editor/AcousticSurfaceConfig.hpp"
#include "threepp/extras/editor/AnimationConfig.hpp"
#include "threepp/extras/editor/ArticulationConfig.hpp"
#include "threepp/extras/editor/CharacterConfig.hpp"
#include "threepp/extras/editor/ConveyorConfig.hpp"
#include "threepp/extras/editor/FlockConfig.hpp"
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
#include "threepp/extras/editor/TerrainConfig.hpp"
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
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <random>
#include <unordered_map>
#include <variant>
#include <unordered_set>

using namespace threepp;
using namespace threepp::editor;

namespace {

    bool section(const char* label, bool defaultOpen = true) {

        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth;
        if (defaultOpen) flags |= ImGuiTreeNodeFlags_DefaultOpen;
        return ImGui::TreeNodeEx(label, flags);
    }

    // toSrgbFloats / fromSrgbFloats live in ConfigFields.hpp — the color field
    // there needs them too, and one copy of a color-space conversion is the
    // only safe number of copies.

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
        drawAcousticsSection(*selected);
        drawTextSection(*selected);
        drawTreeSection(*selected);
        drawTerrainSection(*selected);
        drawFlockSection(*selected);
        drawScriptSection(*selected);
        drawPhysicsSection(*selected);
        drawVehicleSection(*selected);
        drawCharacterSection(*selected);
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
    committed(commands_, changed, [&] {
        commands_.execute(makeProperty<int>(
                "Render Order", "renderOrder:" + object.uuid,
                [target](const int& v) { target->renderOrder = v; },
                object.renderOrder, renderOrder));
        document_.setDirty(true);
    });

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
        committed(commands_, changed, [&] {
            auto after = before;
            after.position.set(position[0], position[1], position[2]);
            commit(after, "Move");
        });
    }

    {
        // Degrees in the UI, radians in the model — three.js convention for
        // Euler angles, and the only readable choice for a numeric field.
        float degrees[3]{
                math::radToDeg(object.rotation.x),
                math::radToDeg(object.rotation.y),
                math::radToDeg(object.rotation.z)};
        const bool changed = ImGui::DragFloat3("Rotation", degrees, 0.5f, 0, 0, "%.2f deg");
        committed(commands_, changed, [&] {
            Euler euler(math::degToRad(degrees[0]), math::degToRad(degrees[1]),
                        math::degToRad(degrees[2]), object.rotation.getOrder());
            auto after = before;
            after.quaternion.setFromEuler(euler);
            commit(after, "Rotate");
        });
    }

    {
        float scale[3]{object.scale.x, object.scale.y, object.scale.z};
        const bool changed = ImGui::DragFloat3("Scale", scale, speed, 0, 0, "%.3f");
        committed(commands_, changed, [&] {
            auto after = before;
            // A zero scale produces a singular matrix — decompose() then hands
            // back garbage and the object can never be scaled up again.
            after.scale.set(std::abs(scale[0]) < 1e-4f ? 1e-4f : scale[0],
                            std::abs(scale[1]) < 1e-4f ? 1e-4f : scale[1],
                            std::abs(scale[2]) < 1e-4f ? 1e-4f : scale[2]);
            commit(after, "Scale");
        });
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
        committed(commands_, changed, [&] {
            auto after = current;
            after.repeat.set(repeat[0], repeat[1]);
            applyUvTransform(material, textures, after, "Tiling");
        });
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
        committed(commands_, changed, [&] {
            auto after = current;
            after.offset.set(offset[0], offset[1]);
            applyUvTransform(material, textures, after, "Offset");
        });
    }

    {
        const char* title = mixed([](const Texture& t) { return t.rotation; })
                                    ? "Rotation (mixed)###rotation"
                                    : "Rotation###rotation";
        float degrees = math::radToDeg(current.rotation);
        const bool changed = ImGui::DragFloat(title, &degrees, 0.5f, -360.f, 360.f, "%.1f deg");
        committed(commands_, changed, [&] {
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
        });
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
        committed(commands_, changed, [&] {
            commands_.execute(makeProperty<Color>(
                    "Color", "color:" + raw->uuid(),
                    sync([withColor](const Color& v) { withColor->color = v; }),
                    withColor->color, fromSrgbFloats(rgb)));
            touched();
        });
    }

    if (auto* withEmissive = dynamic_cast<MaterialWithEmissive*>(raw)) {
        float rgb[3];
        toSrgbFloats(withEmissive->emissive, rgb);
        const bool changed = ImGui::ColorEdit3("Emissive", rgb);
        committed(commands_, changed, [&] {
            commands_.execute(makeProperty<Color>(
                    "Emissive", "emissive:" + raw->uuid(),
                    sync([withEmissive](const Color& v) { withEmissive->emissive = v; }),
                    withEmissive->emissive, fromSrgbFloats(rgb)));
            touched();
        });

        float intensity = withEmissive->emissiveIntensity;
        const bool ch = ImGui::DragFloat("Emissive intensity", &intensity, 0.01f, 0.f, 100.f);
        committed(commands_, ch, [&] {
            commands_.execute(makeProperty<float>(
                    "Emissive Intensity", "emissiveIntensity:" + raw->uuid(),
                    sync([withEmissive](const float& v) { withEmissive->emissiveIntensity = v; }),
                    withEmissive->emissiveIntensity, intensity));
            touched();
        });
    }

    const auto floatField = [&](const char* label, const std::string& key, float* value,
                                float speed, float min, float max,
                                const std::function<void(const float&)>& setter) {
        dragProperty(commands_, label, key + raw->uuid(), value, speed, min, max,
                     sync(setter), touched);
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

    // --- point rendering ----------------------------------------------------
    // The one look control a cloud has. 0 is the Gaussians, 1 is every splat
    // as an opaque disc of Point Size pixels — the point cloud view — and the
    // slider between is a continuous dissolve. Both backends honour it; the
    // depth sort, the occlusion by meshes and the overlay stamp are the same
    // either way. Saved with the cloud's threeppSplat block, like a look.
    {
        float mix = cloud->pointMix();
        if (ImGui::SliderFloat("Point Mix", &mix, 0.f, 1.f, "%.2f")) cloud->setPointMix(mix);
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("0 = Gaussian splats, 1 = one opaque disc per splat (point cloud).\n"
                              "In between blends footprint and opacity.");
        }
        float size = cloud->pointSize();
        if (ImGui::SliderFloat("Point Size", &size, 1.f, 12.f, "%.1f px")) cloud->setPointSize(size);
    }

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

        if (config->pointCloud) {
            ImGui::TextColored(theme::muted(),
                               "Point cloud: one Gaussian per point, sized from the spacing.");
        }
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
            // Which builder. A loaded point cloud is a set of surface samples
            // already, so Auto meshes it straight from the points on the CPU;
            // a Gaussian scan goes through the rendered depth-fusion bake,
            // which is what carves its fog splats away.
            {
                static const char* methods = "Auto\0Depth fusion\0Points (direct)\0";
                int method = config.method;
                if (ImGui::Combo("Method", &method, methods)) {
                    auto after = config;
                    after.method = std::clamp(method, 0, 2);
                    commit(std::move(after), "Splat Surface Method");
                }
            }
            const bool pointRoute = editor::SplatSurfaceCache::usesPointRoute(*cloud, config);
            ImGui::TextColored(theme::muted(),
                               pointRoute ? "Direct: a union-of-balls field over the points, marching "
                                            "cubes, no renderer. The surface sits half a voxel "
                                            "outside the samples."
                                          : "Depth fusion: renders the scan from a set of poses and "
                                            "fuses the depth maps. Needs Vulkan; carves floaters.");

            // What the surface is to PhysX. A static scan keeps its exact
            // triangles; a moving one is split into convex hulls at Play.
            {
                static const char* bodies = "Static\0Dynamic\0Kinematic\0";
                int body = config.body;
                if (ImGui::Combo("Body", &body, bodies)) {
                    auto after = config;
                    after.body = std::clamp(body, 0, 2);
                    commit(std::move(after), "Splat Surface Body");
                }
            }
            if (config.body != editor::SplatSurfaceConfig::Static) {
                if (config.body == editor::SplatSurfaceConfig::Dynamic) {
                    float mass = config.mass;
                    const bool changed = ImGui::DragFloat("Mass (kg)", &mass, 0.1f, 0.001f, 100000.f, "%.3f");
                    committed(commands_, changed, [&] {
                        auto after = config;
                        after.mass = std::max(mass, 0.001f);
                        commit(std::move(after), "Splat Surface Mass");
                    });
                }
                int hulls = config.hulls;
                const bool changed = ImGui::DragInt("Hulls", &hulls, 0.25f, 1, 256);
                committed(commands_, changed, [&] {
                    auto after = config;
                    after.hulls = std::clamp(hulls, 1, 256);
                    commit(std::move(after), "Splat Surface Hulls");
                });
                ImGui::TextColored(theme::muted(),
                                   config.body == editor::SplatSurfaceConfig::Dynamic
                                           ? "Play splits the surface into convex hulls (V-HACD) and "
                                             "simulates them as one body; the scan moves with it."
                                           : "Play splits the surface into convex hulls (V-HACD); the "
                                             "scan's transform drives them and they push dynamics.");
            }
            {
                float voxel = config.voxelSize;
                const bool changed = ImGui::DragFloat("Voxel (m)", &voxel, 0.002f, 0.f, 0.5f, "%.3f");
                committed(commands_, changed, [&] {
                    auto after = config;
                    after.voxelSize = std::clamp(voxel, 0.f, 0.5f);
                    commit(std::move(after), "Splat Surface Voxel");
                });
            }
            {
                int island = config.minComponentVoxels;
                const bool changed = ImGui::DragInt("Island cells", &island, 8.f, 0, 100000);
                committed(commands_, changed, [&] {
                    auto after = config;
                    after.minComponentVoxels = std::max(island, 0);
                    commit(std::move(after), "Splat Surface Islands");
                });
            }
            if (!pointRoute) {
                int poses = config.poseCount;
                const bool changed = ImGui::DragInt("Poses", &poses, 0.5f, 0, 256);
                committed(commands_, changed, [&] {
                    auto after = config;
                    after.poseCount = std::clamp(poses, 0, 256);
                    commit(std::move(after), "Splat Surface Poses");
                });
            }
            ImGui::PopItemWidth();
            ImGui::TextColored(theme::muted(),
                               pointRoute ? "Voxel 0 is twice the points' median spacing."
                                          : "Voxel 0 sizes itself from the scan; poses 0 uses 26.");

            if (!pointRoute) {
                bool interior = config.interior;
                if (ImGui::Checkbox("Interior", &interior)) {
                    auto after = config;
                    after.interior = interior;
                    commit(std::move(after), interior ? "Splat Surface Interior" : "Splat Surface Orbit");
                }
                ImGui::TextColored(theme::muted(),
                                   "Tick for a scan of the INSIDE of a room: the cameras stand in the "
                                   "scan and look out. Orbited from outside, a room bakes the OUTSIDE "
                                   "of its walls and nothing can walk in it.");
            }

            const bool canBake = editor::SplatSurfaceCache::availableFor(renderer_.get(), *cloud, config);
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
    // How it is saved, on the object it applies to: the document keeps a path
    // and the import ops, never the splats, so a scan moved on disk has to be
    // re-imported.
    ImGui::TextColored(theme::muted(), "Saved by reference");
    ImGui::TextWrapped("The scene file stores the scan's path and the import settings and reloads "
                       "it on open; a .tpz carries a copy of the file. Play keeps the live cloud.");

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
    if (mesh->stats.poses == 0) {
        // The direct point route: nothing was rendered, so the numbers that
        // describe it are the voxels the points occupied and the islands it
        // dropped.
        std::snprintf(line, sizeof(line),
                      "%zu triangles, %.3f m voxels, from %llu occupied voxels, %.2f s "
                      "(dropped %u of %u islands)",
                      mesh->triangleCount(), static_cast<double>(mesh->stats.voxelSize),
                      static_cast<unsigned long long>(mesh->stats.observedVoxels),
                      static_cast<double>(seconds), mesh->stats.culledComponents,
                      mesh->stats.components);
    } else {
        std::snprintf(line, sizeof(line),
                      "%zu triangles, %.3f m voxels, %d poses, %.2f s "
                      "(dropped %llu fringe, %llu outlier of %llu samples)",
                      mesh->triangleCount(), static_cast<double>(mesh->stats.voxelSize),
                      mesh->stats.poses, static_cast<double>(seconds),
                      static_cast<unsigned long long>(mesh->stats.skippedFringe),
                      static_cast<unsigned long long>(mesh->stats.skippedOutlier),
                      static_cast<unsigned long long>(mesh->stats.depthSamples));
    }
    splatBakeStats_ = line;
    log("splat surface: \"" + object.name + "\" baked - " + splatBakeStats_);
    flashStatus("Surface baked");
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
        committed(commands_, changed, [&] {
            commands_.execute(makeProperty<Color>(
                    "Light Color", "lightColor:" + object.uuid,
                    [light](const Color& v) { light->color = v; },
                    light->color, fromSrgbFloats(rgb)));
            document_.setDirty(true);
        });
    }

    const auto floatField = [&](const char* label, const std::string& key, float* value,
                                float speed, float min, float max,
                                const std::function<void(const float&)>& setter) {
        dragProperty(commands_, label, key + object.uuid, value, speed, min, max, setter,
                     [this] { document_.setDirty(true); });
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
        committed(commands_, changed, [&] {
            commands_.execute(makeProperty<float>(
                    "Angle", "angle:" + object.uuid,
                    [spot](const float& v) { spot->angle = v; },
                    spot->angle, math::degToRad(angleDegrees)));
            document_.setDirty(true);
        });

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
        committed(commands_, changed, [&] {
            commands_.execute(makeProperty<float>(label, key + object.uuid, setter, *value, edited));
            document_.setDirty(true);
        });
    };

    // One extent rather than four edges: a square shadow camera is what keeps
    // the fit rotation-stable, and four fields invite a lopsided one that
    // swims as the light turns.
    float extent = camera->right;
    const bool extentChanged = ImGui::DragFloat("Extent", &extent, 0.25f, 0.1f, 5000.f);
    committed(commands_, extentChanged, [&] {
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
    });

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
        dragProperty(commands_, label, key + object.uuid, value, speed, min, max, setter,
                     [this] { document_.setDirty(true); });
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
        committed(commands_, changed, [&] {
            auto after = config;
            after.speed = speed;
            commit(after, "Animation Speed");
        });
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

    using Config = ArticulationConfig;

    ConfigFields<Config> fields(commands_, object, Config::read(object).value_or(Config{}),
                                "articulation:" + object.uuid, "Articulation",
                                [this] { document_.setDirty(true); });

    ImGui::Spacing();

    fields.check("Simulate", &Config::enabled, "Simulate Robot", "Stop Simulating Robot");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Play as a PhysX reduced-coordinate articulation.\n"
                          "Colliders are primitive/bbox approximations, not the visual meshes.");
    }

    if (!fields->enabled) return;

    ImGui::PushItemWidth(-110 * contentScale_);

    fields.check("Fixed Base", &Config::fixedBase);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("On: the base is bolted to the world (an arm).\n"
                          "Off: the base floats free (a quadruped, a drone).");
    }

    // Stiffness, damping and force span decades, so they drag logarithmically.
    fields.dragFloat("Stiffness", &Config::stiffness, 1.f, 0.f, 100000.f, "%.3f", nullptr,
                     ImGuiSliderFlags_Logarithmic);
    fields.dragFloat("Damping", &Config::damping, 0.5f, 0.f, 10000.f, "%.3f", nullptr,
                     ImGuiSliderFlags_Logarithmic);
    fields.dragFloat("Max Force", &Config::maxForce, 100.f, 0.f, 1e7f, "%.3f", nullptr,
                     ImGuiSliderFlags_Logarithmic);
    fields.check("Self Collision", &Config::selfCollision);
    fields.dragInt("Iterations", &Config::iterations, 0.2f, 1, 255);
    fields.dragFloat("Density", &Config::density, 5.f, 1.f, 100000.f);

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
        committed(commands_, changed, [&] {
            auto* target = robot;
            const float before = robot->getJointValue(i);
            const float after = revolute ? math::degToRad(shown) : shown;
            commands_.execute(makeProperty<float>(
                    "Joint " + label, "joint:" + object.uuid + ":" + std::to_string(i),
                    [this, target, i](const float& value) { setJointValue(*target, i, value); },
                    before, after));
            document_.setDirty(true);
        });
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
        committed(commands_, changed, [&] {
            auto after = config;
            assign(after, edited);
            commit(after, action, field);
        });
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
        committed(commands_, changed, [&] {
            auto after = config;
            assign(after, edited);
            commit(after, action, field);
        });
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


// ----------------------------------------------------------------- character

void EditorApp::drawCharacterSection(Object3D& object) {

    const bool authored = CharacterConfig::isCharacter(object);

    // The section invites: a node with a SkinnedMesh under it is a rigged
    // character and gets the checkbox. A box or a static prop is not, and
    // stays uncluttered. (Clips alone are not enough — a swinging door has
    // those too.)
    if (!authored) {
        bool skinned = false;
        object.traverse([&](Object3D& node) {
            if (!skinned && node.as<SkinnedMesh>()) skinned = true;
        });
        if (!skinned) return;
    }

    if (!section("Character", authored)) return;

    auto* target = &object;

    // Presence is the identity (CharacterConfig's rule), so the checkbox moves
    // the whole authoring as one optional value — one undo step either way.
    bool simulate = authored;
    if (ImGui::Checkbox("Simulate as Character", &simulate)) {
        const auto was = CharacterConfig::read(object);
        commands_.execute(makeProperty<std::optional<CharacterConfig>>(
                simulate ? "Add Character" : "Remove Character",
                "character:" + object.uuid + ":presence",
                [target](const std::optional<CharacterConfig>& value) {
                    if (value) {
                        value->write(*target);
                    } else {
                        CharacterConfig::erase(*target);
                    }
                },
                was, simulate ? std::optional<CharacterConfig>(CharacterConfig{}) : std::nullopt));
        document_.setDirty(true);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Play gives this model a PhysX capsule controller and drives\n"
                          "its clips. W/S walk and backpedal, A/D strafe, SHIFT runs,\n"
                          "SPACE jumps; the camera follows and the character faces it.\n"
                          "Which clip is which is read off the clips' own root motion.");
    }

    if (!simulate) {
        ImGui::TextColored(theme::muted(), "Not simulated as a character.");
        ImGui::TreePop();
        return;
    }

    auto config = CharacterConfig::read(object).value_or(CharacterConfig{});
    const auto before = config;

    const auto commit = [&](const CharacterConfig& after, const char* label, const char* field) {
        commands_.execute(makeProperty<CharacterConfig>(
                label, "character:" + object.uuid + ":" + field,
                [target](const CharacterConfig& value) { value.write(*target); },
                before, after));
        document_.setDirty(true);
    };

    ImGui::PushItemWidth(-140 * contentScale_);

    const auto geo = config.derived(object);

    // --- how the body relates to the view ---------------------------------
    if (ImGui::BeginCombo("Facing", CharacterConfig::label(config.facing))) {
        for (const auto facing : CharacterConfig::facings) {
            if (ImGui::Selectable(CharacterConfig::label(facing), facing == config.facing)) {
                auto after = config;
                after.facing = facing;
                commit(after, "Character Facing", "facing");
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Camera: the body turns to face the view, so A/D genuinely\n"
                          "strafe and S backpedals - what a locomotion pack's side and\n"
                          "backward clips are for.\n"
                          "Movement: the body turns towards where it is going, so every\n"
                          "direction plays the forward gait.");
    }

    const auto floatField = [&](const char* label, float value, float speed, float min, float max,
                                const char* format, void (*assign)(CharacterConfig&, float),
                                const char* action, const char* field) {
        float edited = value;
        const bool changed = ImGui::DragFloat(label, &edited, speed, min, max, format);
        committed(commands_, changed, [&] {
            auto after = config;
            assign(after, edited);
            commit(after, action, field);
        });
    };

    // --- the capsule. Measured off the model while Auto is on, and shown, so
    // the numbers Play will use are never a mystery.
    ImGui::Spacing();
    bool autoGeo = config.autoGeometry;
    if (ImGui::Checkbox("Auto Capsule", &autoGeo)) {
        auto after = config;
        after.autoGeometry = autoGeo;
        if (!autoGeo && geo.valid) {
            after.height = geo.height;
            after.radius = geo.radius;
        }
        commit(after, autoGeo ? "Character Auto Capsule" : "Character Manual Capsule", "autogeom");
        config = after;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Measure the standing height and body radius off the model.\n"
                          "Untick to type them instead (seeded with the derived values).");
    }
    if (config.autoGeometry) {
        if (geo.valid) {
            ImGui::TextColored(theme::muted(), "Capsule %.2f m tall, r %.2f m",
                               static_cast<double>(geo.height), static_cast<double>(geo.radius));
        } else {
            ImGui::TextColored(theme::warning(), "%s", geo.problem.c_str());
        }
    } else {
        floatField(
                "Height", config.height, 0.01f, 0.2f, 5.f, "%.2f m",
                [](CharacterConfig& c, float v) { c.height = v; },
                "Character Height", "height");
        floatField(
                "Radius", config.radius, 0.005f, 0.05f, 2.f, "%.3f m",
                [](CharacterConfig& c, float v) { c.radius = v; },
                "Character Radius", "radius");
    }

    // --- the gait speeds, read off the clips ------------------------------
    ImGui::Spacing();
    bool autoSpeeds = config.autoSpeeds;
    if (ImGui::Checkbox("Auto Speeds", &autoSpeeds)) {
        auto after = config;
        after.autoSpeeds = autoSpeeds;
        if (!autoSpeeds) {
            if (geo.slot(Gait::Walk).speed > 0.f) after.walkSpeed = geo.slot(Gait::Walk).speed;
            if (geo.slot(Gait::Run).speed > 0.f) after.runSpeed = geo.slot(Gait::Run).speed;
        }
        commit(after, autoSpeeds ? "Character Auto Speeds" : "Character Manual Speeds", "autospeeds");
        config = after;
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Travel at the speed the walk and run clips were AUTHORED at.\n"
                          "That is what stops the feet sliding; type your own only if\n"
                          "you want the character faster than its animation.");
    }
    if (config.autoSpeeds) {
        ImGui::TextColored(theme::muted(), "Walk %.2f m/s, run %.2f m/s",
                           static_cast<double>(geo.slot(Gait::Walk).speed),
                           static_cast<double>(geo.slot(Gait::Run).speed));
    } else {
        floatField(
                "Walk Speed", config.walkSpeed, 0.05f, 0.1f, 20.f, "%.2f m/s",
                [](CharacterConfig& c, float v) { c.walkSpeed = v; },
                "Character Walk Speed", "walkspeed");
        floatField(
                "Run Speed", config.runSpeed, 0.05f, 0.1f, 30.f, "%.2f m/s",
                [](CharacterConfig& c, float v) { c.runSpeed = v; },
                "Character Run Speed", "runspeed");
    }

    // --- the always-authored scalars.
    ImGui::Spacing();
    floatField(
            "Mass", config.mass, 1.f, 1.f, 500.f, "%.0f kg",
            [](CharacterConfig& c, float v) { c.mass = v; }, "Character Mass", "mass");
    floatField(
            "Jump Height", config.jumpHeight, 0.01f, 0.f, 5.f, "%.2f m",
            [](CharacterConfig& c, float v) { c.jumpHeight = v; },
            "Character Jump Height", "jump");
    floatField(
            "Gravity", config.gravity, 0.1f, 0.1f, 60.f, "%.1f m/s2",
            [](CharacterConfig& c, float v) { c.gravity = v; },
            "Character Gravity", "gravity");
    floatField(
            "Step Offset", config.stepOffset, 0.005f, 0.f, 1.f, "%.3f m",
            [](CharacterConfig& c, float v) { c.stepOffset = v; },
            "Character Step Offset", "step");
    floatField(
            "Slope Limit", math::radToDeg(config.slopeLimit), 0.5f, 5.f, 85.f, "%.1f deg",
            [](CharacterConfig& c, float v) { c.slopeLimit = math::degToRad(v); },
            "Character Slope Limit", "slope");
    floatField(
            "Turn Rate", config.turnRate, 0.5f, 0.5f, 60.f, "%.1f /s",
            [](CharacterConfig& c, float v) { c.turnRate = v; },
            "Character Turn Rate", "turnrate");
    floatField(
            "Acceleration", config.accel, 0.5f, 0.5f, 60.f, "%.1f /s",
            [](CharacterConfig& c, float v) { c.accel = v; },
            "Character Acceleration", "accel");
    floatField(
            "Blend Time", config.blendTime, 0.005f, 0.f, 1.f, "%.3f s",
            [](CharacterConfig& c, float v) { c.blendTime = v; },
            "Character Blend Time", "blend");

    // --- which clip is which -----------------------------------------------
    // Auto-matched from each clip's own root motion (see CharacterConfig), so
    // this list is a READOUT first and an override second: it shows what the
    // matcher decided and at what speed, and a combo is there for the clip it
    // got wrong.
    ImGui::Spacing();
    if (ImGui::TreeNodeEx("Clips", ImGuiTreeNodeFlags_SpanAvailWidth)) {
        static constexpr const char* clipFields[kGaitCount] = {
                "clipidle", "clipwalk", "cliprun", "clipwalkback", "cliprunback",
                "clipstrafeleft", "clipstrafeleftfast", "clipstraferight",
                "clipstraferightfast", "clipjump"};

        for (std::size_t i = 0; i < kGaitCount; ++i) {
            const auto& slot = geo.gaits[i];
            const std::string matched = slot.clip ? slot.clip->name() : std::string("(none)");
            const std::string shown =
                    config.clips[i].empty() ? matched + "  (auto)" : config.clips[i];
            if (ImGui::BeginCombo(CharacterConfig::gaitLabels[i], shown.c_str(),
                                  ImGuiComboFlags_HeightLarge)) {
                if (ImGui::Selectable("(auto)", config.clips[i].empty())) {
                    auto after = config;
                    after.clips[i].clear();
                    commit(after, "Character Clip", clipFields[i]);
                }
                for (const auto& clip : object.animations) {
                    if (!clip) continue;
                    const auto name = clip->name();
                    if (ImGui::Selectable(name.c_str(), name == config.clips[i])) {
                        auto after = config;
                        after.clips[i] = name;
                        commit(after, "Character Clip", clipFields[i]);
                    }
                }
                ImGui::EndCombo();
            }
            if (slot.clip && slot.speed > 0.f) {
                ImGui::SameLine();
                ImGui::TextColored(theme::muted(), "%.2f m/s", static_cast<double>(slot.speed));
            }
        }
        ImGui::TreePop();
    }
    if (!geo.problem.empty()) {
        ImGui::TextColored(theme::warning(), "%s", geo.problem.c_str());
    }

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
                        committed(commands_, changed, [&] {
                            commit(edited(field.name, ScriptConfig::toText(value)), "Script " + label);
                        });
                        break;
                    }
                    case ScriptField::Type::Float: {
                        float value = ScriptConfig::toFloat(stored);
                        const bool changed = ImGui::DragFloat(label.c_str(), &value, 0.01f);
                        committed(commands_, changed, [&] {
                            commit(edited(field.name, ScriptConfig::toText(value)), "Script " + label);
                        });
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

    using Config = PhysicsConfig;

    ConfigFields<Config> fields(commands_, object, Config::read(object).value_or(Config{}),
                                "physics:" + object.uuid, "Physics",
                                [this] { document_.setDirty(true); });
    const auto& config = fields.value();

    // A body that has never been configured starts as a dynamic box, which is
    // what "make this fall" means to almost everyone.
    fields.check("Enabled", &Config::enabled, "Enable Physics", "Disable Physics");

    if (!config.enabled) {
        ImGui::TreePop();
        return;
    }

    ImGui::PushItemWidth(-100 * contentScale_);

    {
        static const char* bodies[] = {"Static", "Dynamic", "Kinematic", "Soft"};
        fields.combo("Body", &Config::body, bodies, IM_ARRAYSIZE(bodies), "Physics Body Type");
    }

    const bool soft = config.body == Config::Body::Soft;

    // A soft body's collider is always a tetrahedral volume cooked from the
    // mesh, so the shape picker has nothing to offer it.
    if (!soft) {
        static const char* shapes[] = {"Auto", "Box", "Sphere", "Capsule",
                                       "Convex", "TriMesh", "Convex Pieces"};
        fields.combo("Shape", &Config::shape, shapes, IM_ARRAYSIZE(shapes), "Physics Shape");
        if (config.shape == Config::Shape::TriMesh && config.body != Config::Body::Static) {
            ImGui::TextColored(theme::warning(), "TriMesh is static-only");
        }

        // Beside the shape, because that is what it changes: a trigger's shape
        // is cooked as an overlap volume instead of a collider. Hidden for a
        // soft body (whose collider is the cooked tet volume — see below), and
        // the key still round-trips, so the tick survives a trip through Soft.
        fields.check("Trigger", &Config::trigger, "Make Trigger Volume", "Clear Trigger Volume");
    }

    if (config.body == Config::Body::Dynamic || soft) {
        fields.dragFloat("Mass (kg)", &Config::mass, 0.05f, 0.001f, 10000.f, "%.3f", "Physics Mass");
    }
    fields.dragFloat("Friction", &Config::friction, 0.005f, 0.f, 2.f);
    if (!soft) {
        fields.dragFloat("Restitution", &Config::restitution, 0.005f, 0.f, 1.f);
    }

    // Convex-pieces (V-HACD) parameters, shown only while that shape is picked —
    // the same reveal-on-selection the soft-body section uses.
    if (!soft && config.shape == Config::Shape::Pieces) {
        ImGui::Spacing();
        fields.dragInt("Max Hulls", &Config::hulls, 0.2f, 1, 128, "Convex Pieces Hulls");
        fields.dragInt("Verts / Hull", &Config::hullVerts, 0.2f, 8, 64, "Convex Pieces Hull Verts");
        fields.dragInt("Voxel Res", &Config::voxels, 500.f, 10000, 1000000,
                       "Convex Pieces Resolution");
    }

    if (soft) {
        ImGui::Spacing();
        // Stiffness spans four decades between jelly and hard rubber, so the
        // drag is logarithmic — a linear one is unusable at the soft end.
        fields.dragFloat("Stiffness (Pa)", &Config::youngsModulus, 0.01f, 1e3f, 1e9f, "%.3f",
                         "Soft Body Stiffness", ImGuiSliderFlags_Logarithmic);
        fields.dragFloat("Poisson Ratio", &Config::poissonsRatio, 0.002f, 0.f, 0.49f, "%.3f",
                         "Soft Body Poisson Ratio");
        fields.dragInt("Resolution", &Config::voxelResolution, 0.1f, 2, 64, "Soft Body Resolution");
        fields.dragInt("Iterations", &Config::solverIterations, 0.2f, 1, 255,
                       "Soft Body Iterations");
        fields.check("Self Collision", &Config::selfCollision, "Soft Body Self Collision");
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

    using Config = SplineConfig;

    ConfigFields<Config> fields(commands_, object, Config::read(object).value_or(Config{}),
                                "spline:" + object.uuid, "Spline",
                                [this] { document_.setDirty(true); });
    const auto& config = fields.value();

    ImGui::PushItemWidth(-100 * contentScale_);

    {
        static const char* types[] = {"Centripetal", "Chordal", "CatmullRom"};
        fields.combo("Type", &Config::type, types, IM_ARRAYSIZE(types));
    }

    fields.dragFloat("Tension", &Config::tension, 0.005f, 0.f, 1.f);
    // Shown regardless, because that is three.js's own semantics: the value is
    // stored and does nothing until the type is CatmullRom.
    if (config.type != Config::Type::CatmullRom) {
        ImGui::TextColored(theme::muted(), "Tension applies to CatmullRom only");
    }

    fields.check("Closed", &Config::closed, "Close Spline", "Open Spline");
    fields.dragInt("Samples/Segment", &Config::samples, 0.25f, 1, Config::maxSamples,
                   "Spline Samples");

    // --- generated geometry ------------------------------------------------
    // Only the config is edited here. The mesh itself is derived state that the
    // per-frame sync adds, rebuilds and removes to follow it — see
    // SplineOverlay.cpp.
    ImGui::Spacing();
    {
        static const char* kinds[] = {"None", "Tube"};
        fields.combo("Mesh", &Config::mesh, kinds, IM_ARRAYSIZE(kinds));
    }

    if (config.mesh == Config::MeshKind::Tube) {
        fields.dragFloat("Radius", &Config::radius, 0.005f, 0.001f, 100.f, "%.3f", "Tube Radius");
        fields.dragInt("Radial Segments", &Config::radialSegments, 0.1f, 3, 64,
                       "Tube Radial Segments");
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
        committed(commands_, changed, [&] {
            auto after = config;
            after.volume = std::clamp(volume, 0.f, 1.f);
            commit(after, "Sound Volume");
        });
    }

    {
        float rate = config.rate;
        const bool changed = ImGui::SliderFloat("Playback rate", &rate, 0.25f, 4.f, "%.2fx");
        committed(commands_, changed, [&] {
            auto after = config;
            after.rate = std::clamp(rate, 0.25f, 4.f);
            commit(after, "Sound Rate");
        });
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
            committed(commands_, changed, [&] {
                auto after = config;
                after.minDistance = std::max(minDistance, 0.01f);
                commit(after, "Sound Min Distance");
            });
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Inside this radius the sound plays at full volume.");
            }
            if (noFalloff) ImGui::EndDisabled();
        }

        {
            float maxDistance = config.maxDistance;
            const bool changed = ImGui::DragFloat("Max distance", &maxDistance, 0.5f, 0.01f, 100000.f, "%.2f m");
            committed(commands_, changed, [&] {
                auto after = config;
                after.maxDistance = std::max(maxDistance, after.minDistance);
                commit(after, "Sound Max Distance");
            });
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
            committed(commands_, changed, [&] {
                auto after = config;
                after.rolloff = std::clamp(rolloff, 0.f, 20.f);
                commit(after, "Sound Rolloff");
            });
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


// ----------------------------------------------------------------- acoustics

void EditorApp::drawAcousticsSection(Object3D& object) {

    // Meshes only: an acoustic surface is a BVH over triangles, so there is
    // nothing to author on a group, a light or a camera.
    if (!object.as<Mesh>()) return;
    if (!section("Acoustics", false)) return;

    auto* target = &object;
    const auto config = AcousticSurfaceConfig::read(object).value_or(AcousticSurfaceConfig{});

    const auto commit = [&](AcousticSurfaceConfig after, const char* label) {
        commands_.execute(makeProperty<AcousticSurfaceConfig>(
                label, "acoustics:" + object.uuid,
                [target](const AcousticSurfaceConfig& value) { value.write(*target); },
                config, after));
        document_.setDirty(true);
    };

    bool enabled = config.enabled;
    if (ImGui::Checkbox("Acoustic surface", &enabled)) {
        auto after = config;
        after.enabled = enabled;
        commit(after, enabled ? "Enable Acoustic Surface" : "Disable Acoustic Surface");
    }
    ImGui::TextColored(theme::muted(),
                       "Play traces rays from the listener to every positional sound: "
                       "flagged meshes muffle what is behind them and set the reverb.");

    if (!config.enabled) {
        ImGui::TreePop();
        return;
    }

    ImGui::PushItemWidth(-130 * contentScale_);
    {
        float transmission = config.transmission;
        const bool changed = ImGui::DragFloat("Transmission", &transmission, 0.005f, 0.f, 1.f, "%.2f");
        committed(commands_, changed, [&] {
            auto after = config;
            after.transmission = std::clamp(transmission, 0.f, 1.f);
            commit(after, "Acoustic Transmission");
        });
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("How much sound gets THROUGH. 0 is concrete, 0.6 a curtain,\n"
                          "1 is a surface the sound does not notice.");
    }
    {
        float absorption = config.absorption;
        const bool changed = ImGui::DragFloat("Absorption", &absorption, 0.005f, 0.f, 1.f, "%.2f");
        committed(commands_, changed, [&] {
            auto after = config;
            after.absorption = std::clamp(absorption, 0.f, 1.f);
            commit(after, "Acoustic Absorption");
        });
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("How much a REFLECTION loses here, which is what sets the\n"
                          "reverb tail. 0.05 is bare concrete, 0.6 soft furnishing.");
    }
    ImGui::PopItemWidth();

    ImGui::TextColored(theme::muted(), "Stored in userData[\"acousticSurface\"]");

    ImGui::TreePop();
}


// ---------------------------------------------------------------------- text

void EditorApp::drawTextSection(Object3D& object) {

    if (!TextConfig::isText(object)) return;
    if (!section("Text")) return;

    using Config = TextConfig;

    // The writer goes through apply() rather than write(), so execute, undo and
    // redo all rebuild the geometry the entries describe.
    ConfigFields<Config> fields(commands_, object, Config::read(object).value_or(Config{}),
                                "text:" + object.uuid, "Text",
                                [this] { document_.setDirty(true); });
    fields.writeWith([](const Config& value, Object3D& target) { value.apply(target); });
    const auto& config = fields.value();

    // The content. Modest height: a label is a line or three, and the box
    // grows nothing by being tall.
    std::string buffer = config.text;
    buffer.resize(std::max<std::size_t>(buffer.size() + 512, 1024));
    if (ImGui::InputTextMultiline("##textContent", buffer.data(), buffer.size(),
                                  {-1.f, ImGui::GetTextLineHeight() * 3.5f})) {
        auto after = config;
        after.text = buffer.c_str();
        fields.commit(std::move(after), "Edit Text");
    }
    if (config.text.empty()) {
        ImGui::TextColored(theme::muted(), "Empty text draws nothing.");
    }

    ImGui::PushItemWidth(-100 * contentScale_);

    fields.dragFloat("Size", &Config::size, 0.01f, 0.01f, 100.f);
    fields.dragFloat("Depth", &Config::depth, 0.005f, 0.f, 100.f);
    if (config.depth <= 0.f) {
        ImGui::TextColored(theme::muted(), "Depth 0 is flat - a sign, not solid type.");
    }

    {
        static const char* aligns[] = {"Left", "Center", "Right"};
        fields.combo("Align", &Config::align, aligns, IM_ARRAYSIZE(aligns));
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Where the origin sits on the block:\n"
                              "what the gizmo grabs, and what Position means.");
        }
    }

    fields.dragInt("Curve Segments", &Config::curveSegments, 0.1f, 1, Config::maxCurveSegments);

    ImGui::PopItemWidth();

    ImGui::TextColored(theme::muted(), "Stored in userData[\"text\"]");

    ImGui::TreePop();
}


// ------------------------------------------------------------------- terrain

namespace {

    // A terrain parameter edit, as ONE undo entry that restores the exact
    // heights it replaced.
    //
    // Recompute-based undo would be wrong here, not merely slow: the sculpt
    // layer is recovered by subtracting a re-baked base, and re-baking on every
    // undo/redo cycle walks the heights by float rounding until the mound the
    // user carved has crept. So both generations are held as geometry
    // SNAPSHOTS, exactly as EditorApp.cpp's RegenerateCommand holds both
    // generations of a generator's output, and for the same reason. The albedo
    // rides along: it is baked from the same heights.
    class TerrainEditCommand: public Command {

    public:
        struct Snapshot {
            std::shared_ptr<BufferGeometry> geometry;
            std::vector<unsigned char> albedo;
            int dim = 0;
            TerrainConfig config;
        };

        TerrainEditCommand(Object3D& target, Snapshot before, Snapshot after, std::string label)
            : target_(&target), before_(std::move(before)), after_(std::move(after)),
              label_(std::move(label)), uuid_(target.uuid) {}

        void redo() override { restore(after_); }
        void undo() override { restore(before_); }

        [[nodiscard]] std::string name() const override { return label_; }

        // Two regenerations are two distinct landscapes; collapsing them would
        // drop the middle one's geometry on the floor. Slider DRAGS still
        // coalesce, because the inspector only commits on release.
        bool mergeWith(const Command&) override { return false; }

        [[nodiscard]] bool rebind(Object3D& root) override {

            auto* found = findByUuid(root, uuid_);
            if (!found) return false;
            target_ = found;
            return true;
        }

    private:
        void restore(const Snapshot& snapshot) {

            if (!target_ || !snapshot.geometry) return;
            if (auto* mesh = target_->as<Mesh>()) mesh->setGeometry(snapshot.geometry);
            TerrainConfig::applyAlbedo(*target_, snapshot.albedo, snapshot.dim);
            snapshot.config.write(*target_);
        }

        Object3D* target_;
        Snapshot before_;
        Snapshot after_;
        std::string label_;
        std::string uuid_;
    };

    TerrainEditCommand::Snapshot snapshotTerrain(const Object3D& object, const TerrainConfig& config) {

        TerrainEditCommand::Snapshot out;
        out.config = config;
        if (const auto* mesh = object.as<Mesh>()) out.geometry = mesh->geometry();
        if (const auto texture = TerrainConfig::albedoTexture(object)) {
            const auto& image = texture->image();
            if (!image.isFloat() && !image.isHalfFloat()) {
                out.albedo = image.data<unsigned char>();
                out.dim = static_cast<int>(image.width());
            }
        }
        return out;
    }

}// namespace

void EditorApp::commitTerrain(Object3D& object, const TerrainConfig& before,
                              const TerrainConfig& after, const std::string& label) {

    if (rejectWhilePlaying(label.c_str())) return;

    auto snapBefore = snapshotTerrain(object, before);
    // rebuild() does the delta recovery: the sculpt layer is the difference
    // between the mesh in front of us and what `before` would have produced,
    // and it is carried onto the new bake.
    TerrainConfig::rebuild(object, before, after);
    auto snapAfter = snapshotTerrain(object, after);

    commands_.execute(std::make_unique<TerrainEditCommand>(
            object, std::move(snapBefore), std::move(snapAfter), label));
    document_.setDirty(true);
}

void EditorApp::drawTerrainSection(Object3D& object) {

    if (!TerrainConfig::isTerrain(object)) return;
    if (!section("Terrain")) return;

    using terrain::ErosionType;
    using terrain::Falloff;
    using terrain::NoiseType;
    using terrain::TerrainParams;

    auto* target = &object;
    const auto config = TerrainConfig::read(object).value_or(TerrainConfig::makeDefault());

    const auto commit = [&](TerrainConfig after, const std::string& label) {
        // A slider commit re-bakes RAW. Erosion is a ~1 s pass and must never
        // ride on a drag release — it lives behind Generate, and the flag drops
        // to 0 until the user asks for it again.
        after.eroded = false;
        commitTerrain(*target, config, after, label);
    };

    // Pointer-to-member widgets, TreeConfig's reason verbatim: the generator has
    // forty knobs and forty hand-inlined transaction dances is how one of them
    // ends up silently missing its beginTransaction.
    const auto sliderFloat = [&](const char* label, float TerrainParams::* field,
                                 float min, float max, const char* format = "%.2f") {
        float value = config.params.*field;
        const bool changed = ImGui::SliderFloat(label, &value, min, max, format);
        committed(commands_, changed, [&] {
            auto after = config;
            after.params.*field = std::clamp(value, min, max);
            commit(std::move(after), std::string("Terrain ") + label);
        });
    };

    const auto sliderInt = [&](const char* label, int TerrainParams::* field, int min, int max) {
        int value = config.params.*field;
        const bool changed = ImGui::SliderInt(label, &value, min, max);
        committed(commands_, changed, [&] {
            auto after = config;
            after.params.*field = std::clamp(value, min, max);
            commit(std::move(after), std::string("Terrain ") + label);
        });
    };

    const auto colorEdit = [&](const char* label, std::array<float, 3> TerrainParams::* field) {
        std::array<float, 3> value = config.params.*field;
        const bool changed = ImGui::ColorEdit3(label, value.data());
        committed(commands_, changed, [&] {
            auto after = config;
            after.params.*field = value;
            commit(std::move(after), std::string("Terrain ") + label);
        });
    };

    ImGui::PushItemWidth(-110 * contentScale_);

    // --- seed / presets ---------------------------------------------------
    {
        int preset = 0;
        if (ImGui::Combo("Preset", &preset, "Alpine\0Rolling Hills\0Desert Mesa\0Volcanic\0")) {
            auto after = config;
            after.applyPreset(preset);
            commit(std::move(after), std::string("Terrain ") + TerrainConfig::presetLabel(preset));
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Landscape character only - this terrain keeps its own\n"
                              "size, resolution and seed.");
        }
    }
    {
        int seed = static_cast<int>(config.params.seed);
        if (ImGui::InputInt("Seed", &seed)) {
            auto after = config;
            after.params.seed = static_cast<unsigned int>(std::max(seed, 0));
            commit(std::move(after), "Terrain Seed");
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Reroll")) {
            auto after = config;
            after.params.seed = std::random_device{}();
            commit(std::move(after), "Terrain Reroll");
        }
    }

    // --- shape -------------------------------------------------------------
    sliderFloat("World Size", &TerrainParams::worldSize, 10.f, 2000.f, "%.0f m");
    sliderInt("Resolution", &TerrainParams::resolution, 16, 512);
    sliderFloat("Amplitude", &TerrainParams::amplitude, 0.5f, 600.f, "%.1f m");
    {
        static const char* kinds = "fBm\0Ridged\0Hybrid\0";
        int value = static_cast<int>(config.params.noiseType);
        if (ImGui::Combo("Noise", &value, kinds)) {
            auto after = config;
            after.params.noiseType = static_cast<NoiseType>(std::clamp(value, 0, 2));
            commit(std::move(after), "Terrain Noise");
        }
    }
    sliderFloat("Feature Scale", &TerrainParams::featureScale, 5.f, 800.f, "%.0f m");
    sliderInt("Octaves", &TerrainParams::octaves, 1, 10);
    sliderFloat("Lacunarity", &TerrainParams::lacunarity, 1.2f, 3.5f);
    sliderFloat("Gain", &TerrainParams::gain, 0.2f, 0.8f);
    sliderFloat("Warp", &TerrainParams::warp, 0.f, 1.f);
    sliderFloat("Ridge Sharpness", &TerrainParams::ridgeSharpness, 0.f, 1.f);
    sliderFloat("Height Exponent", &TerrainParams::heightExponent, 0.5f, 3.f);
    sliderInt("Terraces", &TerrainParams::terraces, 0, 24);
    {
        static const char* kinds = "None\0Radial\0";
        int value = static_cast<int>(config.params.falloff);
        if (ImGui::Combo("Falloff", &value, kinds)) {
            auto after = config;
            after.params.falloff = static_cast<Falloff>(std::clamp(value, 0, 1));
            commit(std::move(after), "Terrain Falloff");
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Radial fades the patch into a plain and tucks its rim\n"
                              "under the surrounding ground instead of leaving a lip.");
        }
    }
    if (config.params.falloff == Falloff::Radial) {
        sliderFloat("Falloff Start", &TerrainParams::falloffStart, 0.05f, 0.95f);
    }

    // --- erosion -----------------------------------------------------------
    ImGui::SeparatorText("Erosion");
    {
        static const char* kinds = "None\0Hydraulic\0Thermal\0Both\0";
        int value = static_cast<int>(config.params.erosion);
        if (ImGui::Combo("Type", &value, kinds)) {
            auto after = config;
            after.params.erosion = static_cast<ErosionType>(std::clamp(value, 0, 3));
            commit(std::move(after), "Terrain Erosion Type");
        }
    }
    sliderInt("Droplets", &TerrainParams::droplets, 0, 250000);
    sliderInt("Talus Sweeps", &TerrainParams::thermalIterations, 0, 80);
    if (ImGui::Button("Generate (erode)", {-1.f, 0.f})) {
        auto after = config;
        after.eroded = after.params.erosion != ErosionType::None;
        commitTerrain(object, config, after, "Terrain Erode");
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Runs the erosion pass - about a second at this resolution.\n"
                          "Slider edits re-bake the RAW field, never this.");
    }
    ImGui::TextColored(theme::muted(), config.eroded ? "Baked: eroded" : "Baked: raw noise");

    // --- splat -------------------------------------------------------------
    ImGui::SeparatorText("Surface");
    sliderFloat("Snow Line", &TerrainParams::snowLine, 0.f, 1.2f);
    sliderFloat("Snow Slope Max", &TerrainParams::snowSlopeMax, 0.f, 1.f);
    sliderFloat("Grass Slope Max", &TerrainParams::slopeGrassMax, 0.f, 1.f);
    sliderFloat("Rock Slope Min", &TerrainParams::slopeRockMin, 0.f, 1.f);
    colorEdit("Grass", &TerrainParams::grassColor);
    colorEdit("Scree", &TerrainParams::screeColor);
    colorEdit("Rock", &TerrainParams::rockColor);
    colorEdit("Snow", &TerrainParams::snowColor);

    // --- sculpt ------------------------------------------------------------
    // Only while the tool is armed: brush settings on a terrain nobody is
    // painting are three widgets asking to be mistaken for parameters.
    if (sculptTool_) {
        ImGui::SeparatorText("Brush");
        {
            static const char* kinds = "Raise / Lower\0Smooth\0Flatten\0";
            int value = static_cast<int>(brush_.kind);
            if (ImGui::Combo("Brush", &value, kinds)) {
                brush_.kind = static_cast<TerrainBrush::Kind>(
                        std::clamp(value, 0, TerrainBrush::kindCount - 1));
            }
        }
        ImGui::SliderFloat("Radius", &brush_.radius, 0.5f,
                           std::max(config.params.worldSize * 0.25f, 5.f), "%.1f m");
        ImGui::SliderFloat("Strength", &brush_.strength, 0.5f, 40.f, "%.1f");
        if (brush_.kind == TerrainBrush::Kind::Raise) {
            ImGui::Checkbox("Lower", &brush_.invert);
            ImGui::SameLine();
            ImGui::TextColored(theme::muted(), "(Shift inverts while dragging)");
        }
        ImGui::TextColored(theme::muted(),
                           "One drag is one undo entry. Flatten levels to the\n"
                           "height under the press.");
    }

    ImGui::PopItemWidth();

    ImGui::TextColored(theme::muted(),
                       "Geometry is baked into the document - sculpted edits survive\n"
                       "parameter changes and opening a scene never regenerates.");
    ImGui::TextColored(theme::muted(), "Stored in userData[\"terrain\"]");

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
        committed(commands_, changed, [&] {
            auto after = config;
            after.params.*field = std::clamp(value, min, max);
            commit(std::move(after), std::string("Tree ") + label);
        });
    };

    const auto sliderInt = [&](const char* label, int TreeParams::* field, int min, int max) {
        int value = config.params.*field;
        const bool changed = ImGui::SliderInt(label, &value, min, max);
        committed(commands_, changed, [&] {
            auto after = config;
            after.params.*field = std::clamp(value, min, max);
            commit(std::move(after), std::string("Tree ") + label);
        });
    };

    const auto colorEdit = [&](const char* label, std::array<float, 3> TreeParams::* field) {
        std::array<float, 3> value = config.params.*field;
        const bool changed = ImGui::ColorEdit3(label, value.data());
        committed(commands_, changed, [&] {
            auto after = config;
            after.params.*field = value;
            commit(std::move(after), std::string("Tree ") + label);
        });
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
            committed(commands_, changed, [&] {
                auto after = before;
                after.height = std::clamp(height, 0.02f, 3.f);
                commitWall(after, "Wall Height");
            });
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
            committed(commands_, changed, [&] {
                auto after = wpBefore;
                after.cornerRadius = std::max(radius, 0.f);
                commitWp(after, "Corner Radius");
            });
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

    using Config = ConveyorConfig;

    ConfigFields<Config> fields(commands_, object, Config::read(object).value_or(Config{}),
                                "conveyor:" + object.uuid, "Conveyor",
                                [this] { document_.setDirty(true); });
    const Config& config = fields.value();

    ImGui::PushItemWidth(-110 * contentScale_);

    fields.check("Separator (wall, no belt)", &Config::separator, "Make Separator", "Make Belt");

    if (config.separator) {
        fields.dragFloat("Wall Height", &Config::wallHeight, 0.01f, 0.05f, 5.f, "%.3f", "Wall Height");
    } else {
        fields.dragFloat("Belt Width", &Config::width, 0.01f, 0.05f, 5.f, "%.3f", "Belt Width");
        fields.dragFloat("Belt Speed (m/s)", &Config::speed, 0.01f, 0.f, 10.f, "%.3f", "Belt Speed");

        fields.check("Reverse Flow", &Config::reverse, "Reverse Flow");
        ImGui::SameLine();
        fields.check("Frame", &Config::frame, "Add Frame", "Remove Frame");
    }

    fields.check("Smooth (spline)", &Config::smooth, "Conveyor Smoothing");
    fields.dragInt("Samples/Segment", &Config::samples, 0.25f, 2, Config::maxSamples,
                   "Conveyor Samples");

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
            fields.dragFloat("Roller Radius", &Config::rollerRadius, 0.002f, 0.01f, 0.5f, "%.3f", "Roller Radius");
        }
        if (anyCleats) {
            fields.dragFloat("Cleat Height", &Config::cleatHeight, 0.005f, 0.02f, 1.f, "%.3f", "Cleat Height");
            fields.dragFloat("Cleat Spacing", &Config::cleatSpacing, 0.01f, 0.1f, 5.f, "%.3f", "Cleat Spacing");
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

    // Only the config is edited here. The preview FIELD is derived state
    // syncParticleOverlays follows — which is also what keeps undo cheap and
    // what makes a structural edit (capacity, radius, proxy, resolution) an
    // ordinary property write rather than a special case.
    //
    // Seventy knobs go through ConfigFields rather than seventy hand-inlined
    // copies of the transaction dance — which is how one of them ends up
    // silently missing its beginTransaction.
    ConfigFields<Config> fields(commands_, object, Config::read(object).value_or(Config{}),
                                "particles:" + object.uuid, "Particles",
                                [this] { document_.setDirty(true); });
    const auto& config = fields.value();

    // Reads the toggle back so a revealed block draws the frame it is turned on.
    const auto toggle = [&](const char* label, bool Config::* field) {
        fields.check(label, field);
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
            fields.commit(preset.make(), std::string("Particles ") + preset.label);
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
        committed(commands_, changed, [&] {
            auto after = config;
            after.capacity = std::max(capacity, 1);
            fields.commit(std::move(after), "Particles Capacity");
        });
    }
    fields.dragFloat("Radius", &Config::radius, 0.001f, 0.0001f, 1.f, "%.4f");
    {
        int proxy = static_cast<int>(config.proxy);
        if (ImGui::Combo("Proxy", &proxy, "None\0Sphere\0Flake\0")) {
            auto after = config;
            after.proxy = static_cast<Config::Proxy>(proxy);
            fields.commit(std::move(after), "Particles Proxy");
        }
    }
    {
        int resolution = config.densityResolution;
        const bool changed = ImGui::DragInt("Density Res", &resolution, 1.f, 8, 256);
        committed(commands_, changed, [&] {
            auto after = config;
            after.densityResolution = std::clamp(resolution, 8, 256);
            fields.commit(std::move(after), "Particles Density Res");
        });
    }

    // --- emitter ----------------------------------------------------------
    ImGui::SeparatorText("Emitter");
    fields.dragVector3("Velocity", &Config::velocity, 0.05f, -100.f, 100.f);
    fields.dragFloat("Speed Spread", &Config::speedSpread, 0.01f, 0.f, 10.f);
    fields.dragVector3("Accel", &Config::accel, 0.01f, -50.f, 50.f);
    fields.dragVector3("Wind", &Config::wind, 0.01f, -50.f, 50.f);
    fields.dragVector3("Spawn Extent", &Config::spawnHalfExtent, 0.05f, 0.001f, 500.f);
    ImGui::TextColored(theme::muted(),
                       "A THIN slab swept over velocity x lifetime is the steady cloud.");
    fields.dragFloat("Lifetime", &Config::lifetime, 0.05f, 0.001f, 600.f, "%.2f");
    fields.dragFloat("Life Jitter", &Config::lifetimeJitter, 0.01f, 0.f, 1.f);
    fields.dragFloat("Duty Cycle", &Config::dutyCycle, 0.01f, 0.001f, 1.f);
    fields.dragFloat("Size", &Config::size, 0.001f, 0.f, 10.f, "%.4f");
    fields.dragFloat("Size Jitter", &Config::sizeJitter, 0.01f, 0.f, 1.f);
    fields.dragFloat("Drift Amplitude", &Config::driftAmplitude, 0.01f, 0.f, 20.f);
    fields.dragFloat("Drift Frequency", &Config::driftFrequency, 0.01f, 0.f, 20.f);
    fields.dragFloat("Drift Growth", &Config::driftGrowth, 0.01f, 0.f, 1.f);
    fields.dragFloat("Drift Scale", &Config::driftScale, 0.1f, 0.f, 200.f, "%.2f");
    {
        int seed = config.seed;
        if (ImGui::InputInt("Seed", &seed)) {
            auto after = config;
            after.seed = std::max(seed, 0);
            fields.commit(std::move(after), "Particles Seed");
        }
    }
    if (toggle("Follow Camera", &Config::follow)) {
        fields.dragFloat("Follow Snap", &Config::followSnap, 0.05f, 0.f, 100.f, "%.2f");
        ImGui::TextColored(theme::muted(),
                           "Snap an INTEGER number of density voxels, or the haze swims.");
    }

    // --- surface landing --------------------------------------------------
    ImGui::SeparatorText("Surface Landing");
    if (toggle("Land on Surfaces", &Config::surface)) {
        fields.dragFloat("Rest Seconds", &Config::surfaceRest, 0.05f, 0.f, 60.f, "%.2f");
        fields.dragFloat("Rest Jitter", &Config::surfaceRestJitter, 0.01f, 0.f, 1.f);
        fields.dragFloat("Fade Seconds", &Config::surfaceFade, 0.05f, 0.f, 60.f, "%.2f");
        fields.dragFloat("Bias", &Config::surfaceBias, 0.001f, 0.f, 1.f, "%.4f");
        fields.dragFloat("Splash Seconds", &Config::surfaceSplash, 0.01f, 0.f, 10.f);
        if (config.surfaceSplash > 0.f) {
            fields.dragFloat("Splash Grow", &Config::surfaceSplashGrow, 0.1f, 1.f, 64.f, "%.2f");
        }
        {
            int resolution = config.surfaceResolution;
            const bool changed = ImGui::DragInt("Bake Res", &resolution, 1.f, 16, 1024);
            committed(commands_, changed, [&] {
                auto after = config;
                after.surfaceResolution = std::clamp(resolution, 16, 1024);
                fields.commit(std::move(after), "Particles Bake Res");
            });
        }
        ImGui::TextColored(theme::muted(), "Particles rest and FADE; they do not pile up.");
    }

    // --- representations --------------------------------------------------
    ImGui::SeparatorText("Mesh Proxy");
    if (toggle("Draw Proxies", &Config::mesh)) {
        if (config.proxy == Config::Proxy::None) {
            ImGui::TextColored(theme::warning(), "Proxy is None - nothing to draw.");
        }
        fields.dragFloat("LOD Far", &Config::meshLodFar, 0.1f, 0.f, 500.f, "%.2f");
        fields.dragFloat("LOD Fade", &Config::meshLodFade, 0.1f, 0.f, 500.f, "%.2f");
        fields.dragFloat("Near Cull", &Config::meshNearCull, 0.05f, 0.f, 50.f, "%.2f");
        if (config.billboard && config.meshLodFar > 0.f) {
            ImGui::TextColored(theme::muted(),
                               "Sprites fade in over the same band the proxies shrink out over.");
        }
    }

    ImGui::SeparatorText("Billboard");
    if (toggle("Draw Sprites", &Config::billboard)) {
        fields.color("Hot Color", &Config::colorHot);
        fields.color("Cool Color", &Config::colorCool);
        fields.dragFloat("Intensity", &Config::billboardIntensity, 0.01f, 0.f, 100.f, "%.3f");
        fields.dragFloat("Sprite Size", &Config::billboardSize, 0.01f, 0.0001f, 50.f);
        fields.dragFloat("Softness", &Config::billboardSoftness, 0.01f, 0.f, 1.f);
        fields.dragFloat("Fade Power", &Config::billboardFade, 0.01f, 0.f, 8.f);
        fields.dragFloat("Bright Jitter", &Config::billboardJitter, 0.01f, 0.f, 1.f);
        fields.dragFloat("Size Taper", &Config::billboardTaper, 0.01f, 0.f, 1.f);
        fields.dragFloat("Stretch (s)", &Config::billboardStretch, 0.001f, 0.f, 1.f, "%.4f");
        if (config.billboardStretch > 0.f) {
            fields.dragFloat("Stretch Max", &Config::billboardStretchMax, 0.5f, 1.f, 200.f, "%.1f");
        }
        fields.dragFloat("Near Fade", &Config::billboardNearFade, 0.01f, 0.f, 20.f);
        fields.dragFloat("Glow", &Config::billboardGlow, 0.1f, 0.f, 64.f, "%.2f");
        if (config.billboardGlow > 0.f) {
            fields.dragFloat("Glow Threshold", &Config::billboardGlowThreshold, 0.01f, 0.f, 20.f);
        }
        ImGui::TextColored(theme::muted(), "Additive and unlit - an ember IS the light source.");
    }

    ImGui::SeparatorText("Density Volume");
    if (toggle("Scatter into a Volume", &Config::density)) {
        fields.dragFloat("Sigma / Particle", &Config::sigma, 0.001f, 0.0001f, 100.f, "%.4f");
        fields.color("Albedo", &Config::albedo);
        fields.dragFloat("Anisotropy", &Config::anisotropy, 0.01f, -0.95f, 0.95f);
        fields.dragVector3("Volume Extent", &Config::densityHalfExtent, 0.1f, 0.001f, 500.f);

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
            fields.dragFloat("Emissive", &Config::emissiveIntensity, 0.5f, 0.f, 500.f, "%.1f");
            if (config.emissiveIntensity > 0.f) {
                fields.dragFloat("Temp Bottom (K)", &Config::tempBottom, 10.f, 300.f, 6000.f, "%.0f");
                fields.dragFloat("Temp Top (K)", &Config::tempTop, 10.f, 300.f, 6000.f, "%.0f");
                fields.dragFloat("Temp Falloff", &Config::tempFalloff, 0.05f, 0.05f, 8.f, "%.2f");
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

    ConfigFields<Config> fields(commands_, object, Config::read(object).value_or(Config{}),
                                "granular:" + object.uuid, "Granular",
                                [this] { document_.setDirty(true); });

    ImGui::PushItemWidth(-130 * contentScale_);

    ImGui::SeparatorText("Grains");
    // Grain diameter: the render radius is half of it and everything else
    // derives from it, which is why it is the first knob and not a detail.
    fields.dragFloat("Spacing", &Config::spacing, 0.002f, 0.002f, 1.f, "%.4f");
    fields.dragInt("Capacity", &Config::capacity, 250.f, 1, 4000000);
    fields.dragInt("Iterations", &Config::iterations, 0.25f, 1, 32);
    fields.dragFloat("Max Velocity", &Config::maxVelocity, 0.1f, 0.f, 200.f, "%.2f");
    ImGui::TextColored(theme::muted(), "Max Velocity 0 derives a clamp from the spacing.");

    ImGui::SeparatorText("Material");
    // The repose angle of a heap IS its internal friction; cohesion is the
    // other half of how steep a pile stands.
    fields.dragFloat("Friction", &Config::friction, 0.01f, 0.f, 2.f);
    fields.dragFloat("Damping", &Config::damping, 0.01f, 0.f, 10.f);
    fields.dragFloat("Adhesion", &Config::adhesion, 0.01f, 0.f, 10.f);
    fields.dragFloat("Cohesion", &Config::cohesion, 0.01f, 0.f, 10.f);
    fields.dragFloat("Viscosity", &Config::viscosity, 0.01f, 0.f, 10.f);
    fields.dragFloat("Gravity Scale", &Config::gravityScale, 0.01f, -4.f, 4.f);

    ImGui::SeparatorText("Chute");
    fields.dragFloat("Mouth X", &Config::emitExtentX, 0.01f, 0.001f, 20.f);
    fields.dragFloat("Mouth Z", &Config::emitExtentZ, 0.01f, 0.001f, 20.f);
    fields.dragFloat("Rate (/s)", &Config::rate, 25.f, 0.f, 500000.f, "%.0f");
    fields.dragVector3("Pour Velocity", &Config::emitVelocity, 0.05f, -50.f, 50.f, "%.3f",
                       "Granular Pour Velocity");
    fields.dragFloat("Mass", &Config::mass, 0.01f, 0.f, 100.f);
    fields.dragFloat("Pour For (s)", &Config::emitFor, 0.1f, 0.f, 600.f, "%.2f");
    fields.dragFloat("Lattice Jitter", &Config::jitter, 0.01f, 0.f, 1.f);
    ImGui::TextColored(theme::muted(), "Mass 0 = 1 kg; Pour For 0 = until capacity.");

    ImGui::SeparatorText("Visual");
    fields.combo("Draw As", &Config::visual, "Auto\0Instanced\0Field\0", 3, "Granular Visual");
    fields.color("Grain Color", &Config::color, "Granular Grain Color");
    fields.dragFloat("Roughness", &Config::roughness, 0.01f, 0.f, 1.f);

    ImGui::PopItemWidth();

    ImGui::TextColored(theme::muted(), "The node's transform IS the chute frame; -Y pours down.");
    ImGui::TextColored(theme::muted(), "Grains are a PhysX PBD sim - they exist only while playing.");
    ImGui::TextColored(theme::muted(), "Stored in userData[\"granular\"]");

    ImGui::TreePop();
}


// -------------------------------------------------------------------- flock

void EditorApp::drawFlockSection(Object3D& object) {

    if (!FlockConfig::isFlock(object)) return;
    if (!section("Flock")) return;

    using Config = FlockConfig;

    ConfigFields<Config> fields(commands_, object, Config::read(object).value_or(Config{}),
                                "flock:" + object.uuid, "Flock",
                                [this] { document_.setDirty(true); });

    ImGui::PushItemWidth(-130 * contentScale_);

    ImGui::SeparatorText("Population");
    fields.dragInt("Seed", &Config::seed, 1.f, 0, 999999);
    // 18 reads as "a place where birds live"; 200 reads as "a bird
    // simulation" — the config header carries the full warning.
    fields.dragInt("Birds", &Config::birdCount, 0.25f, 0, 256);
    fields.dragFloat("Body Mass (kg)", &Config::massKg, 0.001f, 0.01f, 1.f, "%.3f");

    ImGui::SeparatorText("Territory");
    fields.dragFloat("Roam Radius", &Config::roamRadius, 0.25f, 4.f, 500.f);
    fields.dragFloat("Cruise Altitude", &Config::cruiseAltitude, 0.1f, 1.f, 200.f);
    // The band the helper draws: each bird prefers ground + altitude ×
    // (1 ± spread). 0 flies a plane — a formation, not a flock.
    fields.dragFloat("Altitude Spread", &Config::altitudeSpread, 0.005f, 0.f, 1.f);
    fields.dragFloat("Cruise Speed", &Config::cruiseSpeed, 0.05f, 1.f, 40.f);

    ImGui::SeparatorText("Perching");
    fields.check("Perching", &Config::perching, "Enable Flock Perching",
                 "Disable Flock Perching");
    fields.dragFloat("Max Perched", &Config::maxPerchedFraction, 0.01f, 0.f, 1.f);

    ImGui::SeparatorText("Look");
    fields.check("Cast Shadow", &Config::castShadow, "Enable Flock Shadows",
                 "Disable Flock Shadows");
    fields.dragFloat("Wind X", &Config::windX, 0.01f, -10.f, 10.f);
    fields.dragFloat("Wind Z", &Config::windZ, 0.01f, -10.f, 10.f);

    ImGui::PopItemWidth();

    ImGui::TextColored(theme::muted(), "The node's position is the territory's home.");
    ImGui::TextColored(theme::muted(), "Birds fly while playing; perches bake from the scene at Play.");
    ImGui::TextColored(theme::muted(), "Stored in userData[\"flock\"]");

    ImGui::TreePop();
}


// ------------------------------------------------------------------- sensor

void EditorApp::drawSensorSection(Object3D& object) {

    if (object.is<Scene>()) return;
    if (!section("Sensor", false)) return;

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
    using Config = SensorConfig;

    // On a camera host the camera's frustum is the truth (the play session
    // reads it directly); the normalizer stamps it into the flat string on
    // every write, so userData never drifts from what Play will build.
    ConfigFields<Config> fields(commands_, object, config, "sensor:" + object.uuid, "Sensor",
                                [this] { document_.setDirty(true); });
    fields.normalizeWith([hostCamera](Config& after) {
        if (!hostCamera || !Config::isPinhole(after.type)) return;
        after.fovY = hostCamera->fov;
        after.nearPlane = hostCamera->nearPlane;
        after.farPlane = hostCamera->farPlane;
    });
    // ConfigFields keeps this current across every commit below, which is why
    // none of them re-reads: drawing the rest of a frame from the pre-click
    // value would show the wrong fields for one frame (and, on Remove, fields
    // for an entry that is gone).
    const Config& live = fields.value();

    // Add/Remove is a userData edit, not a graph edit, so it goes straight
    // through the property command like the physics Enabled box — nothing here
    // can invalidate `object`, which is why this one does not need the deferred
    // re-resolve the spline section's Insert buttons do.
    bool enabled = live.enabled;
    if (ImGui::Checkbox("Enabled", &enabled)) {
        auto after = live;
        after.enabled = enabled;
        // A camera's default sensor is itself: the colour camera. The generic
        // default (IMU) belongs to the link-shaped hosts.
        if (enabled && hostCamera && !Config::isPinhole(after.type)) {
            after.type = Config::Type::Camera;
        }
        fields.commit(std::move(after), enabled ? "Add Sensor" : "Remove Sensor");
    }

    if (!live.enabled) {
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
            if (!belongs && candidate != live.type) continue;
            if (candidate == live.type) current = static_cast<int>(offered.size());
            offered.push_back(candidate);
            names.push_back(SensorConfig::label(candidate));
        }
        if (ImGui::Combo("Type", &current, names.data(), static_cast<int>(names.size()))) {
            auto after = live;
            after.type = offered[static_cast<std::size_t>(current)];
            // Only the type changes. Every other key is written regardless of
            // type (see SensorConfig), so the settings of the type being left
            // behind are still there when the user comes back to it.
            fields.commit(after, "Sensor Type");
        }
    }

    // The legacy shape: a pinhole authored on a plain object, from before these
    // moved onto cameras. It still plays — the gate is authoring-side — but
    // everything a camera host gives (the frustum helper, the dock preview,
    // fov/near/far in one place) is one click away.
    if (!hostCamera && SensorConfig::isPinhole(live.type)) {
        ImGui::TextColored(theme::warning(),
                           "%s sensors live on a Camera object now.",
                           SensorConfig::label(live.type));
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
        if (offerAll && live.joint == SensorConfig::allJoints) current = 1;
        for (std::size_t i = 0; i < info.size(); ++i) {
            names.push_back(info[i].name.c_str());
            if (info[i].name == live.joint) current = static_cast<int>(i) + base;
        }
        if (ImGui::Combo("Joint", &current, names.data(), static_cast<int>(names.size()))) {
            auto after = live;
            if (current == 0) {
                after.joint.clear();
            } else if (offerAll && current == 1) {
                after.joint = SensorConfig::allJoints;
            } else {
                after.joint = info[static_cast<std::size_t>(current - base)].name;
            }
            fields.commit(after, "Sensor Joint");
        }
        if (live.joint.empty()) {
            ImGui::TextColored(theme::muted(), "Pick which joint this sensor reads.");
        } else if (live.joint == SensorConfig::allJoints) {
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

    fields.dragFloat("Rate (Hz)", &Config::rateHz, 0.25f, 0.f, 2000.f, "%.1f", "Sensor Rate");
    if (live.rateHz <= 0.f) {
        ImGui::TextColored(theme::muted(),
                           SensorConfig::isVision(live.type)
                                   ? "0 = scan every frame (expensive)"
                                   : "0 = sample every physics substep");
    }
    fields.dragInt("Seed", &Config::seed, 1.f, 0, 1000000, "Sensor Seed");

    ImGui::Spacing();

    switch (live.type) {

        case SensorConfig::Type::Imu: {
            // Continuous-time densities, the way a spec sheet quotes them, so the
            // authored numbers are rate-independent.
            fields.dragFloat("Gyro Density", &Config::gyroNoiseDensity, 0.0002f, 0.f, 1.f, "%.5f", "IMU Gyro Noise");
            fields.dragFloat("Gyro Bias Walk", &Config::gyroRandomWalk, 1e-5f, 0.f, 0.1f, "%.6f", "IMU Gyro Bias Walk");
            fields.dragFloat("Accel Density", &Config::accelNoiseDensity, 0.002f, 0.f, 10.f, "%.5f", "IMU Accel Noise");
            fields.dragFloat("Accel Bias Walk", &Config::accelRandomWalk, 0.0002f, 0.f, 1.f, "%.6f", "IMU Accel Bias Walk");

            if (ImGui::SmallButton("Perfect")) {
                auto after = live;
                after.gyroNoiseDensity = 0.f;
                after.gyroRandomWalk = 0.f;
                after.accelNoiseDensity = 0.f;
                after.accelRandomWalk = 0.f;
                fields.commit(after, "Perfect IMU");
            }
            ImGui::SameLine();
            ImGui::TextColored(theme::muted(), "zero noise = ground truth");
            break;
        }

        case SensorConfig::Type::Depth: {
            // On a camera host the frustum is the camera's; only the image
            // dimensions are the sensor's own.
            if (!hostCamera) {
                fields.dragFloat("FOV (deg)", &Config::fovY, 0.25f, 1.f, 179.f, "%.1f", "Depth FOV");
            }
            fields.dragInt("Width", &Config::width, 1.f, 8, SensorConfig::maxImageSize, "Depth Width");
            fields.dragInt("Height", &Config::height, 1.f, 8, SensorConfig::maxImageSize, "Depth Height");
            break;
        }

        case SensorConfig::Type::Camera: {
            if (!hostCamera) {
                fields.dragFloat("FOV (deg)", &Config::fovY, 0.25f, 1.f, 179.f, "%.1f", "Camera FOV");
            }
            fields.dragInt("Width", &Config::width, 1.f, 8, SensorConfig::maxImageSize, "Camera Width");
            fields.dragInt("Height", &Config::height, 1.f, 8, SensorConfig::maxImageSize, "Camera Height");
            ImGui::TextColored(theme::muted(),
                               "Looks down this object's -Z. Read it from a script with "
                               "editor.camera_from_object(obj).image");
            ImGui::TextColored(theme::muted(),
                               "Record writes one PNG per frame plus an index CSV.");
            break;
        }

        case SensorConfig::Type::Lidar: {
            static const char* beams[] = {"Dense Grid", "VLP-16", "HDL-32E", "OS1-64", "OS0-128"};
            fields.combo("Beams", &Config::beams, beams, IM_ARRAYSIZE(beams),
                         "LIDAR Beam Pattern");
            fields.dragInt("Face Size", &Config::faceSize, 2.f, 16, SensorConfig::maxFaceSize, "LIDAR Face Size");
            ImGui::TextColored(theme::muted(),
                               "Six 90-degree depth passes per scan - face size is each "
                               "one's resolution.");
            break;
        }

        case SensorConfig::Type::Encoder: {
            jointPicker(true);
            fields.dragFloat("Resolution", &Config::encoderResolution, 1e-5f, 0.f, 1.f, "%.6f", "Encoder Resolution");
            ImGui::TextColored(theme::muted(), "rad (or m) per tick; 0 = ideal");
            break;
        }

        case SensorConfig::Type::Contact: {
            fields.dragFloat("Force Threshold", &Config::contactForceThreshold, 0.05f, 0.f, 10000.f, "%.2f", "Contact Force Threshold");
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
    if (SensorConfig::isVision(live.type)) {
        ImGui::Spacing();
        if (hostCamera && SensorConfig::isPinhole(live.type)) {
            // One source of truth: the Camera section above is where the
            // frustum is edited, so these numbers are shown, not editable —
            // two drag fields for the same plane would fight.
            ImGui::TextColored(theme::muted(),
                               "Frustum from this camera: FOV %.1f deg, near %.3f, far %.2f.",
                               hostCamera->fov, hostCamera->nearPlane, hostCamera->farPlane);
            ImGui::TextColored(theme::muted(), "Edit them in the Camera section.");
        } else {
            fields.dragFloat("Near (m)", &Config::nearPlane, 0.005f, 0.001f, 100.f, "%.3f", "Sensor Near");
            fields.dragFloat("Far (m)", &Config::farPlane, 0.25f, 0.01f, 10000.f, "%.2f", "Sensor Far");
            if (live.farPlane <= live.nearPlane) {
                ImGui::TextColored(theme::warning(), "Far must be beyond Near");
            }
        }
    }

    // Range noise is a RANGING sensor's, not every vision sensor's: a sigma in
    // metres has nothing to say about a colour pixel. The Camera shares the
    // frustum above and stops there.
    if (SensorConfig::isRanging(live.type)) {
        fields.dragFloat("Range Sigma (m)", &Config::rangeStddev, 0.001f, 0.f, 5.f, "%.4f", "Range Noise");
        fields.dragFloat("Sigma per m", &Config::rangeStddevPerMetre, 0.0002f, 0.f, 1.f, "%.5f", "Range Noise per Metre");
        fields.dragFloat("Range Bias (m)", &Config::rangeBias, 0.001f, -5.f, 5.f, "%.4f", "Range Bias");
    }

    ImGui::PopItemWidth();
    ImGui::Spacing();

    // The mistake that authors cleanly and measures nothing: a proprioceptive
    // sensor with no rigid body under it. Cheap to answer here, and Play would
    // otherwise be the first place it shows up.
    if (live.type == SensorConfig::Type::Imu || live.type == SensorConfig::Type::Contact) {
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
