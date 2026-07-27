
#include "../EditorApp.hpp"
#include "../EditorTheme.hpp"
#include "../PanelLayout.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/lights/Light.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <cstring>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // A one-glyph type hint in front of the name — enough to tell a light from
    // a mesh from a group while scanning, without an icon font.
    const char* kindTag(const Object3D& object) {

        if (object.is<Light>()) return "L";
        if (object.is<Camera>()) return "C";
        if (object.is<Mesh>()) return "M";
        if (!object.children.empty()) return "G";
        return "-";
    }

    std::string displayName(const Object3D& object) {

        return object.name.empty() ? "(" + object.type() + ")" : object.name;
    }

}// namespace


void EditorApp::drawAddMenu(Object3D& parent) {

    Object3D* target = &parent;

    if (ImGui::BeginMenu("Mesh")) {
        for (const auto type : ObjectFactory::primitives) {
            if (ImGui::MenuItem(ObjectFactory::label(type))) {
                deferred_ = [this, type, target] {
                    addObject(ObjectFactory::createPrimitive(type, document_.scene()), *target,
                              std::string("Add ") + ObjectFactory::label(type));
                };
            }
        }
        ImGui::EndMenu();
    }

    if (ImGui::BeginMenu("Light")) {
        for (const auto kind : ObjectFactory::lights) {
            if (ImGui::MenuItem(ObjectFactory::label(kind))) {
                deferred_ = [this, kind, target] {
                    addObject(ObjectFactory::createLight(kind, document_.scene()), *target,
                              std::string("Add ") + ObjectFactory::label(kind));
                };
            }
        }
        ImGui::EndMenu();
    }

    if (ImGui::MenuItem("Group")) {
        deferred_ = [this, target] {
            addObject(ObjectFactory::createGroup(document_.scene()), *target, "Add Group");
        };
    }
    if (ImGui::MenuItem("Camera")) {
        deferred_ = [this, target] {
            addObject(ObjectFactory::createCamera(document_.scene()), *target, "Add Camera");
        };
    }
    if (ImGui::MenuItem("Spline")) {
        deferred_ = [this, target] {
            addObject(ObjectFactory::createSpline(document_.scene()), *target, "Add Spline");
        };
    }
}

void EditorApp::drawHierarchyNode(Object3D& object) {

    if (document_.isEditorOnly(object)) return;

    ImGui::PushID(&object);

    const bool selected = selection_.get() == &object;
    const bool leaf = object.children.empty();

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow |
                               ImGuiTreeNodeFlags_OpenOnDoubleClick |
                               ImGuiTreeNodeFlags_SpanAvailWidth |
                               ImGuiTreeNodeFlags_DefaultOpen;
    if (leaf) flags |= ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
    if (selected) flags |= ImGuiTreeNodeFlags_Selected;

    if (scrollTo_ == &object) {
        ImGui::SetScrollHereY(0.5f);
        scrollTo_ = nullptr;
    }

    bool open = false;

    if (renaming_ == &object) {
        // Inline rename: the input replaces the whole row (children included)
        // until the edit is committed or abandoned. One frame of a collapsed
        // subtree is a fair price for a rename UI with no special cases.
        ImGui::SetNextItemWidth(-1);
        char buffer[128];
        const auto n = std::min(renameBuffer_.size(), sizeof(buffer) - 1);
        std::memcpy(buffer, renameBuffer_.data(), n);
        buffer[n] = '\0';
        ImGui::SetKeyboardFocusHere();
        const bool committed = ImGui::InputText(
                "##rename", buffer, sizeof(buffer),
                ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
        renameBuffer_ = buffer;
        if (committed) {
            const std::string before = object.name;
            const std::string after = renameBuffer_;
            if (before != after) {
                auto* target = &object;
                commands_.execute(makeProperty<std::string>(
                        "Rename", {},
                        [target](const std::string& value) { target->name = value; },
                        before, after));
                document_.setDirty(true);
            }
            renaming_ = nullptr;
        } else if (ImGui::IsItemDeactivated()) {
            renaming_ = nullptr;
        }
        ImGui::PopID();
        return;
    }

    {
        if (!object.visible) ImGui::PushStyleColor(ImGuiCol_Text, theme::muted());
        open = ImGui::TreeNodeEx("##node", flags, "%s  %s", kindTag(object),
                                 displayName(object).c_str());
        if (!object.visible) ImGui::PopStyleColor();

        if (ImGui::IsItemClicked(ImGuiMouseButton_Left) && !ImGui::IsItemToggledOpen()) {
            selectObject(&object);
        }
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            renaming_ = &object;
            renameBuffer_ = object.name;
        }

        // --- drag & drop reparenting ----------------------------------------
        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceNoHoldToOpenOthers)) {
            Object3D* payload = &object;
            ImGui::SetDragDropPayload("THREEPP_EDITOR_OBJECT", &payload, sizeof(payload));
            ImGui::TextUnformatted(displayName(object).c_str());
            ImGui::EndDragDropSource();
        }
        if (ImGui::BeginDragDropTarget()) {
            if (const auto* payload = ImGui::AcceptDragDropPayload("THREEPP_EDITOR_OBJECT")) {
                auto* dragged = *static_cast<Object3D* const*>(payload->Data);
                auto* target = &object;
                if (dragged && dragged != target) {
                    deferred_ = [this, dragged, target] { reparent(*dragged, *target); };
                }
            }
            ImGui::EndDragDropTarget();
        }

        // --- context menu ----------------------------------------------------
        if (ImGui::BeginPopupContextItem("##ctx")) {
            selectObject(&object);
            if (ImGui::BeginMenu("Add")) {
                drawAddMenu(object);
                ImGui::EndMenu();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Rename")) {
                renaming_ = &object;
                renameBuffer_ = object.name;
            }
            if (ImGui::MenuItem("Duplicate", "Ctrl+D")) {
                deferred_ = [this] { duplicateSelected(); };
            }
            if (ImGui::MenuItem("Focus", "F")) focusSelected();
            ImGui::Separator();
            if (ImGui::MenuItem("Delete", "Del")) {
                deferred_ = [this] { deleteSelected(); };
            }
            ImGui::EndPopup();
        }
    }

    if (open && !leaf) {
        // Copy the child pointers: a context-menu action may edit the vector.
        const auto children = object.children;
        for (auto* child : children) {
            if (child) drawHierarchyNode(*child);
        }
        ImGui::TreePop();
    }

    ImGui::PopID();
}

void EditorApp::drawHierarchy() {

    const auto* viewport = ImGui::GetMainViewport();
    const float s = contentScale_;

    const float width = hierarchyPx();
    const float top = menuHeight_ + toolbarHeight_;
    const float bottom = statusHeight_ + (bottomPanelOpen_ ? layout::bottomHeight * s
                                                           : ImGui::GetFrameHeight() + 6 * s);
    const float height = std::max(viewport->Size.y - top - bottom, 40.f * s);

    ImGui::SetNextWindowPos({viewport->Pos.x, viewport->Pos.y + top});
    // A window shrunk below the chrome would otherwise ask for a negative size.
    ImGui::SetNextWindowSize({width, height});

    // A deep tree indents past the panel edge; without this the far end of the
    // hierarchy is simply unreachable, since the panel cannot be widened past
    // the screen either.
    if (ImGui::Begin("Hierarchy", nullptr, layout::panelFlags | ImGuiWindowFlags_HorizontalScrollbar)) {

        auto& scene = document_.scene();

        // Scene root row: the drop target for "move to top level" and the
        // anchor for Add.
        const bool sceneSelected = selection_.get() == &scene;
        if (ImGui::Selectable(scene.name.empty() ? "Scene" : scene.name.c_str(), sceneSelected)) {
            selectObject(&scene);
        }
        if (ImGui::BeginDragDropTarget()) {
            if (const auto* payload = ImGui::AcceptDragDropPayload("THREEPP_EDITOR_OBJECT")) {
                auto* dragged = *static_cast<Object3D* const*>(payload->Data);
                if (dragged) {
                    deferred_ = [this, dragged] { reparent(*dragged, document_.scene()); };
                }
            }
            ImGui::EndDragDropTarget();
        }
        if (ImGui::BeginPopupContextItem("##sceneCtx")) {
            if (ImGui::BeginMenu("Add")) {
                drawAddMenu(scene);
                ImGui::EndMenu();
            }
            ImGui::EndPopup();
        }

        ImGui::Separator();

        ImGui::Indent(6 * s);
        const auto children = scene.children;
        for (auto* child : children) {
            if (child) drawHierarchyNode(*child);
        }
        ImGui::Unindent(6 * s);

        // Clicking empty space below the tree deselects — the same gesture as
        // clicking empty space in the viewport.
        ImGui::InvisibleButton("##blank", {std::max(ImGui::GetContentRegionAvail().x, 1.f),
                                           std::max(ImGui::GetContentRegionAvail().y, 1.f)});
        if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) selectObject(nullptr);
    }
    ImGui::End();

    drawSplitter("##hierarchySplit", viewport->Pos.x + width, viewport->Pos.y + top,
                 height, settings_.hierarchyWidth, 1.f);
}
