// The viewport chrome: what used to be a toolbar, as viewport furniture.
//
// The tool palette — the transform tools Select, Move, Rotate and Scale, the
// space toggle and Snap — stacked in the top-left corner; the transport —
// Play, Pause, Stop — as a pill in the top-centre; the viewpoint picker under
// the view gizmo top-right. All but the picker are drawn with the view
// gizmo's brush: ImGui's background draw list, vector icons, no textures, no
// second render pass, the same picture on either backend. The picker stays a
// real ImGui window, because a combo is not worth reinventing in a draw list.
//
// The icons are sprites in the old sense — a dozen draw-list primitives each —
// because a bitmap would need an atlas, a loader and a DPI ladder, and a
// pointer, four arrows and a circular arrow do not. Every shape is drawn from
// the cell's centre at content scale, so the palette is crisp at any DPI.
//
// Interaction follows the view gizmo exactly (hover suppresses picking and
// orbit through toolPaletteHovered_, a press belongs to where it started, a
// click is a release that never became a drag), so the two corners feel like
// one instrument.

#include "EditorApp.hpp"
#include "EditorTheme.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

#include <algorithm>
#include <array>
#include <cmath>

using namespace threepp;
using namespace threepp::editor;

namespace {

    enum class Tool { Select, Move, Rotate, Scale, Sculpt, Space, Snap };

    constexpr std::array<Tool, 7> kTools{Tool::Select, Tool::Move, Tool::Rotate,
                                         Tool::Scale, Tool::Sculpt, Tool::Space, Tool::Snap};

    // --- the sprites ------------------------------------------------------
    // Each draws into a cell centred on `c`, `h` the icon's half-extent,
    // `s` the content scale for stroke widths.

    void drawSelectIcon(ImDrawList* draw, const ImVec2& c, float h, float s, ImU32 col) {

        // The classic pointer. Head and tail as separate primitives, because
        // at a 16 px icon the concave-polygon staircase melts into a blob -
        // a triangle and a stroke survive the rasterizer.
        const float u = h / 8.f;
        const ImVec2 tip{c.x - 2.6f * u, c.y - 7.2f * u};
        const ImVec2 heel{c.x - 2.6f * u, c.y + 3.6f * u};
        const ImVec2 shoulder{c.x + 3.4f * u, c.y + 0.4f * u};
        draw->AddTriangleFilled(tip, heel, shoulder, col);
        draw->AddLine({c.x + 0.3f * u, c.y + 1.3f * u},
                      {c.x + 2.9f * u, c.y + 6.6f * u}, col, 2.2f * s);
    }

    void drawMoveIcon(ImDrawList* draw, const ImVec2& c, float h, float s, ImU32 col) {

        const float head = 3.6f * s;
        const float arm = h;
        const float shaft = arm - head + 0.5f * s;
        draw->AddLine({c.x - shaft, c.y}, {c.x + shaft, c.y}, col, 1.8f * s);
        draw->AddLine({c.x, c.y - shaft}, {c.x, c.y + shaft}, col, 1.8f * s);
        const float w = 2.6f * s;
        draw->AddTriangleFilled({c.x + arm, c.y}, {c.x + arm - head, c.y - w},
                                {c.x + arm - head, c.y + w}, col);
        draw->AddTriangleFilled({c.x - arm, c.y}, {c.x - arm + head, c.y + w},
                                {c.x - arm + head, c.y - w}, col);
        draw->AddTriangleFilled({c.x, c.y + arm}, {c.x - w, c.y + arm - head},
                                {c.x + w, c.y + arm - head}, col);
        draw->AddTriangleFilled({c.x, c.y - arm}, {c.x + w, c.y - arm + head},
                                {c.x - w, c.y - arm + head}, col);
    }

    void drawRotateIcon(ImDrawList* draw, const ImVec2& c, float h, float s, ImU32 col) {

        // An open ring with an arrowhead on its leading end. The gap sits in
        // the upper right, the head at its clockwise edge, pointing into it.
        const float r = h - 1.5f * s;
        const float a0 = -0.55f;// the gap's far edge, just above +x
        const float a1 = 4.65f; // near the top - the arc runs a0..a1, ~300 deg
        draw->PathArcTo(c, r, a0, a1, 32);
        draw->PathStroke(col, ImDrawFlags_None, 2.f * s);

        // Head at the a1 end, pointing along increasing angle (clockwise on
        // screen - y runs down).
        const ImVec2 end{c.x + r * std::cos(a1), c.y + r * std::sin(a1)};
        const ImVec2 tangent{-std::sin(a1), std::cos(a1)};
        const ImVec2 normal{std::cos(a1), std::sin(a1)};
        const float head = 4.4f * s, half = 2.6f * s;
        draw->AddTriangleFilled({end.x + tangent.x * head, end.y + tangent.y * head},
                                {end.x + normal.x * half, end.y + normal.y * half},
                                {end.x - normal.x * half, end.y - normal.y * half}, col);
    }

    void drawScaleIcon(ImDrawList* draw, const ImVec2& c, float h, float s, ImU32 col) {

        const float u = h / 8.f;
        // The object, outlined, in the lower left - outlined, not filled, so
        // it stays an object next to the arrow instead of merging with it,
        // and square-cornered: at five pixels, any rounding reads as a circle.
        draw->AddRect({c.x - 7.5f * u, c.y + 1.5f * u}, {c.x - 1.5f * u, c.y + 7.5f * u}, col,
                      0.f, 0, 1.6f * s);
        // ...growing out of its corner along the diagonal...
        draw->AddLine({c.x - 1.5f * u, c.y + 1.5f * u}, {c.x + 4.4f * u, c.y - 4.4f * u}, col,
                      1.8f * s);
        // ...towards the corner it is scaling into.
        const ImVec2 tip{c.x + 7.f * u, c.y - 7.f * u};
        draw->AddTriangleFilled(tip, {tip.x - 4.6f * u, tip.y + 1.2f * u},
                                {tip.x - 1.2f * u, tip.y + 4.6f * u}, col);
    }

    void drawSculptIcon(ImDrawList* draw, const ImVec2& c, float h, float s, ImU32 col) {

        // A mound with a brush coming down onto it. Both halves are convex
        // primitives and the mound is a filled triangle rather than an arc:
        // at 16 px a shallow curve rasterizes to a stepped smear, and a
        // concave polygon melts (the select pointer's lesson).
        const float u = h / 8.f;
        // Ground line, with the mound standing on it.
        draw->AddLine({c.x - 7.5f * u, c.y + 5.6f * u}, {c.x + 7.5f * u, c.y + 5.6f * u},
                      col, 1.6f * s);
        draw->AddTriangleFilled({c.x - 6.6f * u, c.y + 5.6f * u},
                                {c.x + 1.2f * u, c.y + 5.6f * u},
                                {c.x - 2.7f * u, c.y - 0.6f * u}, col);
        // The brush: a straight shaft and a wedge tip, angled in from upper
        // right. Square-ended - a rounded cap at this size reads as a dot.
        draw->AddLine({c.x + 7.2f * u, c.y - 7.2f * u}, {c.x + 2.4f * u, c.y - 1.6f * u},
                      col, 2.0f * s);
        draw->AddTriangleFilled({c.x + 0.6f * u, c.y + 1.2f * u},
                                {c.x + 1.4f * u, c.y - 3.2f * u},
                                {c.x + 4.4f * u, c.y - 0.6f * u}, col);
    }

    void drawWorldIcon(ImDrawList* draw, const ImVec2& c, float h, float s, ImU32 col) {

        // A globe: rim, equator, one meridian.
        const float r = h - 1.5f * s;
        draw->AddCircle(c, r, col, 0, 1.5f * s);
        draw->AddLine({c.x - r + 1.f * s, c.y}, {c.x + r - 1.f * s, c.y}, col, 1.2f * s);
        draw->AddEllipse(c, {r * 0.45f, r - 0.5f * s}, col, 0.f, 0, 1.2f * s);
    }

    void drawLocalIcon(ImDrawList* draw, const ImVec2& c, float h, float s, ImU32 col) {

        // A cube seen from its corner: the object's own axes. Hexagonal
        // silhouette, three edges meeting in the middle.
        const float r = h - 1.f * s;
        std::array<ImVec2, 6> hex;
        for (int i = 0; i < 6; ++i) {
            const float a = (30.f + 60.f * static_cast<float>(i)) * 3.14159265f / 180.f;
            hex[static_cast<std::size_t>(i)] = {c.x + r * std::cos(a), c.y + r * std::sin(a)};
        }
        draw->AddPolyline(hex.data(), 6, col, ImDrawFlags_Closed, 1.5f * s);
        // Vertices at 90 (bottom), 210 (upper... in y-down terms: 270 is up.
        // The three edges from the centre go to every OTHER vertex.
        for (int i = 1; i < 6; i += 2) {
            draw->AddLine(c, hex[static_cast<std::size_t>(i)], col, 1.2f * s);
        }
    }

    void drawSnapIcon(ImDrawList* draw, const ImVec2& c, float h, float s, ImU32 col, ImU32 tip) {

        // A horseshoe magnet, poles down.
        const float u = h / 8.f;
        const float r = 4.6f * u;
        const float legTop = c.y + 1.2f * u;
        const float legBottom = c.y + 6.5f * u;
        draw->PathLineTo({c.x - r, legBottom});
        draw->PathLineTo({c.x - r, legTop});
        draw->PathArcTo({c.x, legTop}, r, 3.14159265f, 2.f * 3.14159265f, 20);
        draw->PathLineTo({c.x + r, legBottom});
        draw->PathStroke(col, ImDrawFlags_None, 3.f * s);
        // The poles, a shade brighter - the business end.
        draw->AddLine({c.x - r, legBottom - 1.8f * u}, {c.x - r, legBottom}, tip, 3.f * s);
        draw->AddLine({c.x + r, legBottom - 1.8f * u}, {c.x + r, legBottom}, tip, 3.f * s);
    }

}// namespace

void EditorApp::drawToolPalette() {

    toolPaletteHovered_ = false;

    const float s = contentScale_;
    const auto* viewport = ImGui::GetMainViewport();

    const float cell = 34.f * s;
    const float pad = 5.f * s;
    const float margin = 14.f * s;
    const float sep = 9.f * s;// the band between the modes and the toggles

    const float width = cell + pad * 2.f;
    const float height = pad * 2.f + cell * static_cast<float>(kTools.size()) + sep;

    // Top-left of the viewport: under the menu bar, clear of the hierarchy -
    // the mirror of the view gizmo's corner.
    const ImVec2 origin(viewport->Pos.x + hierarchyPx() + margin,
                        viewport->Pos.y + menuHeight_ + margin);

    // A window small enough that the panels have eaten the viewport has no
    // corner to draw in. The bottom edge counts the bottom panel: a palette
    // over another panel is a palette that steals its clicks.
    if (origin.x + width > viewport->Pos.x + viewport->Size.x - inspectorPx() - margin) return;
    const float floorY = viewport->Pos.y + viewport->Size.y - statusHeight_ -
                         (bottomPanelOpen_ ? settings_.bottomPanelHeight : 0.f) - margin;
    if (origin.y + height > floorY) return;

    const ImGuiIO& io = ImGui::GetIO();
    const auto onPanel = [&](const ImVec2& p) {
        return p.x >= origin.x && p.x <= origin.x + width &&
               p.y >= origin.y && p.y <= origin.y + height;
    };
    const bool overPanel = onPanel(io.MousePos);
    // An ImGui window over this corner owns the mouse; the palette sits under
    // every window and yields exactly as the viewport does.
    const bool interactive = overPanel && !io.WantCaptureMouse && !fileBrowser_.isOpen();

    // Press ownership, ImGui's own rule: a drag belongs to whatever it
    // STARTED on (see the view gizmo, whose logic this is).
    bool ownsPress = !ImGui::IsAnyMouseDown();
    if (!ownsPress) {
        for (int button = 0; button < 3 && !ownsPress; ++button) {
            if (ImGui::IsMouseDown(button)) ownsPress = onPanel(io.MouseClickedPos[button]);
        }
    }
    toolPaletteHovered_ = interactive && ownsPress;

    const auto drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.f);
    const bool released = interactive && ownsPress &&
                          ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !gizmo_->isDragging() &&
                          std::abs(drag.x) < 3.f * s && std::abs(drag.y) < 3.f * s;

    auto* draw = ImGui::GetBackgroundDrawList();

    // Backdrop, the view gizmo's: readable over whatever the scene puts here,
    // a touch stronger while the pointer is on it.
    draw->AddRectFilled(origin, {origin.x + width, origin.y + height},
                        IM_COL32(12, 14, 17, interactive ? 165 : 105), 9.f * s);

    const ImU32 accent = ImGui::ColorConvertFloat4ToU32(theme::accent());
    const bool gizmoAvailable = !isPlaying();

    float y = origin.y + pad;
    for (const Tool tool : kTools) {

        if (tool == Tool::Space) {
            // The band: modes above, preferences below.
            const float mid = y + sep * 0.5f;
            draw->AddLine({origin.x + pad + 3.f * s, mid},
                          {origin.x + width - pad - 3.f * s, mid},
                          IM_COL32(255, 255, 255, 26), 1.f * s);
            y += sep;
        }

        const ImVec2 r0{origin.x + pad, y};
        const ImVec2 r1{origin.x + pad + cell, y + cell};
        const ImVec2 centre{(r0.x + r1.x) * 0.5f, (r0.y + r1.y) * 0.5f};
        y += cell;

        // The gizmo is parked for the duration of Play (see startPlay), so
        // the whole palette sleeps with it - including Snap, which can affect
        // nothing while there is no gizmo to snap. (Shift stays the runtime
        // hold-to-snap override either way.)
        // Sculpt is the one tool with a subject requirement: it edits a height
        // lattice, and there is no lattice under a selected box. Greyed rather
        // than hidden, so the tool is discoverable before you have a terrain.
        const bool enabled = gizmoAvailable &&
                             (tool != Tool::Sculpt || sculptTarget() != nullptr);

        bool active = false;
        switch (tool) {
            case Tool::Select: active = gizmoMode_ == "select" && !sculptTool_; break;
            case Tool::Move: active = gizmoMode_ == "translate" && !sculptTool_; break;
            case Tool::Rotate: active = gizmoMode_ == "rotate" && !sculptTool_; break;
            case Tool::Scale: active = gizmoMode_ == "scale" && !sculptTool_; break;
            case Tool::Sculpt: active = sculptTool_; break;
            case Tool::Snap: active = snapEnabled_; break;
            // Both spaces are a state, not a mode - the icon says which.
            case Tool::Space: break;
        }

        const bool hot = enabled && interactive && ownsPress &&
                         io.MousePos.x >= r0.x && io.MousePos.x <= r1.x &&
                         io.MousePos.y >= r0.y && io.MousePos.y <= r1.y;

        if (active) {
            // Parked with the gizmo (Play): the active cell stays marked but
            // stops shouting - a full-strength accent under a grey icon would
            // claim a liveness the tools do not have.
            ImVec4 fill = theme::accent();
            if (!enabled) fill.w *= 0.32f;
            draw->AddRectFilled(r0, r1, ImGui::ColorConvertFloat4ToU32(fill), 6.f * s);
        } else if (hot) {
            draw->AddRectFilled(r0, r1, IM_COL32(255, 255, 255, 22), 6.f * s);
        }

        // Hover lifts towards white rather than brightening - the same rule
        // the view gizmo's balls follow, for the same reason.
        const ImU32 icon = !enabled  ? IM_COL32(116, 121, 128, 255)
                           : active  ? IM_COL32(250, 252, 255, 255)
                           : hot     ? IM_COL32(255, 255, 255, 255)
                                     : IM_COL32(208, 213, 219, 255);

        const float h = 8.f * s;
        switch (tool) {
            case Tool::Select: drawSelectIcon(draw, centre, h, s, icon); break;
            case Tool::Move: drawMoveIcon(draw, centre, h, s, icon); break;
            case Tool::Rotate: drawRotateIcon(draw, centre, h, s, icon); break;
            case Tool::Scale: drawScaleIcon(draw, centre, h, s, icon); break;
            case Tool::Sculpt: drawSculptIcon(draw, centre, h, s, icon); break;
            case Tool::Space:
                if (gizmoWorldSpace_) {
                    drawWorldIcon(draw, centre, h, s, icon);
                } else {
                    drawLocalIcon(draw, centre, h, s, icon);
                }
                break;
            case Tool::Snap:
                // Asleep, the poles grey out with the body: an accent tip on a
                // parked tool would claim a liveness it does not have.
                drawSnapIcon(draw, centre, h, s, icon,
                             !enabled ? icon
                             : active ? IM_COL32(255, 255, 255, 255)
                                      : accent);
                break;
        }

        if (!hot) continue;

        switch (tool) {
            case Tool::Select: ImGui::SetTooltip("Select - click picks, no gizmo"); break;
            case Tool::Move: ImGui::SetTooltip("Move (W)"); break;
            case Tool::Rotate: ImGui::SetTooltip("Rotate (E)"); break;
            case Tool::Scale: ImGui::SetTooltip("Scale (R)"); break;
            case Tool::Sculpt:
                ImGui::SetTooltip("Sculpt the selected terrain's heights.\n"
                                  "Drag to paint; hold Shift to invert raise/lower.\n"
                                  "Brush type, radius and strength are in the inspector.");
                break;
            case Tool::Space:
                ImGui::SetTooltip(gizmoWorldSpace_
                                          ? "World space - the gizmo on the world axes (Q)"
                                          : "Local space - the gizmo on the object's axes (Q)");
                break;
            case Tool::Snap:
                ImGui::SetTooltip("Snap translate/rotate/scale to a grid.\n"
                                  "Hold Shift for the same effect.");
                break;
        }

        if (!released) continue;

        switch (tool) {
            // Picking a transform tool leaves Sculpt: a gizmo and a brush both
            // wanting the drag is one of them silently losing.
            case Tool::Select: sculptTool_ = false; gizmoMode_ = "select"; applyGizmoMode(); break;
            case Tool::Move: sculptTool_ = false; gizmoMode_ = "translate"; applyGizmoMode(); break;
            case Tool::Rotate: sculptTool_ = false; gizmoMode_ = "rotate"; applyGizmoMode(); break;
            case Tool::Scale: sculptTool_ = false; gizmoMode_ = "scale"; applyGizmoMode(); break;
            case Tool::Sculpt:
                sculptTool_ = !sculptTool_;
                // The gizmo would sit on the terrain's origin, in the way of the
                // ground being brushed; Select is the mode with no handles.
                if (sculptTool_) {
                    gizmoMode_ = "select";
                    applyGizmoMode();
                }
                break;
            case Tool::Space: gizmoWorldSpace_ = !gizmoWorldSpace_; applyGizmoMode(); break;
            case Tool::Snap: snapEnabled_ = !snapEnabled_; break;
        }
    }
}

// ------------------------------------------------------------- the transport

void EditorApp::drawTransportBar() {

    const float s = contentScale_;
    const auto* viewport = ImGui::GetMainViewport();

    const float cell = 34.f * s;
    const float pad = 5.f * s;

    const float width = cell * 3.f + pad * 2.f;
    const float height = cell + pad * 2.f;// 44*s - the play banner stands on this

    // Top-centre, floating over the 3D view: where the eye already goes when
    // something starts moving.
    const ImVec2 origin(viewport->Pos.x + (viewport->Size.x - width) * 0.5f,
                        viewport->Pos.y + menuHeight_ + 10.f * s);

    // A window narrow enough that the pill would sit on the palette or the
    // panels has no centre to float in.
    if (origin.x < viewport->Pos.x + hierarchyPx() + 70.f * s) return;
    if (origin.x + width > viewport->Pos.x + viewport->Size.x - inspectorPx() - 70.f * s) return;

    const ImGuiIO& io = ImGui::GetIO();
    const auto onPill = [&](const ImVec2& p) {
        return p.x >= origin.x && p.x <= origin.x + width &&
               p.y >= origin.y && p.y <= origin.y + height;
    };
    const bool interactive = onPill(io.MousePos) && !io.WantCaptureMouse && !fileBrowser_.isOpen();

    bool ownsPress = !ImGui::IsAnyMouseDown();
    if (!ownsPress) {
        for (int button = 0; button < 3 && !ownsPress; ++button) {
            if (ImGui::IsMouseDown(button)) ownsPress = onPill(io.MouseClickedPos[button]);
        }
    }
    // OR, not assign: the palette reset the flag at the top of this frame and
    // both are the same piece of furniture as far as the pick gate cares.
    toolPaletteHovered_ = toolPaletteHovered_ || (interactive && ownsPress);

    const auto drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.f);
    const bool released = interactive && ownsPress &&
                          ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !gizmo_->isDragging() &&
                          std::abs(drag.x) < 3.f * s && std::abs(drag.y) < 3.f * s;

    auto* draw = ImGui::GetBackgroundDrawList();
    draw->AddRectFilled(origin, {origin.x + width, origin.y + height},
                        IM_COL32(12, 14, 17, interactive ? 165 : 105), 9.f * s);

    const bool playing = isPlaying();
    const bool paused = playing && play_.paused();

    for (int index = 0; index < 3; ++index) {

        const ImVec2 r0{origin.x + pad + cell * static_cast<float>(index), origin.y + pad};
        const ImVec2 r1{r0.x + cell, r0.y + cell};
        const ImVec2 centre{(r0.x + r1.x) * 0.5f, (r0.y + r1.y) * 0.5f};

        const bool enabled = index == 0 || playing;
        const bool hot = enabled && interactive && ownsPress &&
                         io.MousePos.x >= r0.x && io.MousePos.x <= r1.x &&
                         io.MousePos.y >= r0.y && io.MousePos.y <= r1.y;

        // The running state on the buttons themselves, in the play banner's
        // colours: green under Play while the clock runs, amber under Pause
        // while it holds.
        if (index == 0 && playing && !paused) {
            draw->AddRectFilled(r0, r1, ImGui::ColorConvertFloat4ToU32(theme::playing()), 6.f * s);
        } else if (index == 1 && paused) {
            draw->AddRectFilled(r0, r1, ImGui::ColorConvertFloat4ToU32(theme::warning()), 6.f * s);
        } else if (hot) {
            draw->AddRectFilled(r0, r1, IM_COL32(255, 255, 255, 22), 6.f * s);
        }

        const ImU32 icon = !enabled ? IM_COL32(116, 121, 128, 255)
                           : hot    ? IM_COL32(255, 255, 255, 255)
                                    : IM_COL32(224, 228, 233, 255);

        const float u = s;
        if (index == 0) {
            draw->AddTriangleFilled({centre.x + 5.5f * u, centre.y},
                                    {centre.x - 3.5f * u, centre.y - 5.5f * u},
                                    {centre.x - 3.5f * u, centre.y + 5.5f * u}, icon);
        } else if (index == 1) {
            draw->AddRectFilled({centre.x - 4.5f * u, centre.y - 5.f * u},
                                {centre.x - 1.5f * u, centre.y + 5.f * u}, icon, 1.f * s);
            draw->AddRectFilled({centre.x + 1.5f * u, centre.y - 5.f * u},
                                {centre.x + 4.5f * u, centre.y + 5.f * u}, icon, 1.f * s);
        } else {
            draw->AddRectFilled({centre.x - 4.5f * u, centre.y - 4.5f * u},
                                {centre.x + 4.5f * u, centre.y + 4.5f * u}, icon, 1.f * s);
        }

        if (!hot) continue;
        if (index == 0) ImGui::SetTooltip(playing ? "Restart play" : "Play");
        if (index == 1) ImGui::SetTooltip(paused ? "Resume" : "Pause");
        if (index == 2) ImGui::SetTooltip("Stop");

        if (!released) continue;
        if (index == 0) startPlay();
        if (index == 1) togglePause();
        if (index == 2) stopPlay();
    }
}

// ------------------------------------------------------- the viewpoint picker

void EditorApp::drawViewpointPicker() {

    const float s = contentScale_;
    const auto* viewport = ImGui::GetMainViewport();

    const float width = 130.f * s;
    // Right-aligned under the view gizmo's disc (margin + 2*radius tall, see
    // drawViewGizmo) - the picker says in words what the gizmo says in balls,
    // and answers to the same corner.
    const ImVec2 pos(viewport->Pos.x + viewport->Size.x - inspectorPx() - 14.f * s - width,
                     viewport->Pos.y + menuHeight_ + 14.f * s + 80.f * s + 10.f * s);
    if (pos.x < viewport->Pos.x + hierarchyPx() + 14.f * s) return;

    ImGui::SetNextWindowPos(pos);
    if (ImGui::Begin("##viewpoint", nullptr,
                     ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBackground |
                             ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_AlwaysAutoResize)) {

        // ASCII only: the ImGui default font has no glyph for a middle dot and
        // draws a box in its place.
        const std::string label =
                std::string(viewPresetLabel(viewPreset())) + (orthographic() ? " / Ortho" : " / Persp");

        ImGui::SetNextItemWidth(width);
        if (ImGui::BeginCombo("##viewpointCombo", label.c_str())) {

            static constexpr ViewPreset presets[] = {
                    ViewPreset::Front, ViewPreset::Back,
                    ViewPreset::Right, ViewPreset::Left,
                    ViewPreset::Top, ViewPreset::Bottom};

            for (auto preset : presets) {
                if (ImGui::Selectable(viewPresetLabel(preset),
                                      viewPreset() == preset && orthographic())) {
                    setOrthographic(true);
                    setViewPreset(preset);
                }
            }
            ImGui::Separator();
            bool ortho = orthographic();
            if (ImGui::Checkbox("Orthographic", &ortho)) setOrthographic(ortho);

            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("Axis views: Num1/3/7 (Ctrl for the opposite side).\n"
                              "Num5 toggles orthographic. Alt+digit works without a numpad.");
        }
    }
    ImGui::End();
}
