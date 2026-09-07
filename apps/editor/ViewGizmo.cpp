// The camera-orientation gizmo, after three.js editor's: a ball at each end
// of the three world axes in the viewport's top-right corner, drawn where the
// axes actually point from the current camera. Clicking a ball swings the
// view onto that axis (orthographic, like the numpad views); clicking the
// axis the camera already stands on swings to the far end.
//
// Drawn entirely with ImGui's background draw list. No second scene, no
// second render pass, the same picture on either backend: the 3D is nothing
// but the camera quaternion - each world axis rotated into camera space,
// x/-y as the screen offset, z as the stacking order.

#include "EditorApp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

#include <algorithm>
#include <array>
#include <cmath>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // three.js editor's axis palette, so the gizmo reads the same here.
    constexpr float kAxisColor[3][3] = {
            {1.00f, 0.21f, 0.33f},// X
            {0.54f, 0.86f, 0.00f},// Y
            {0.17f, 0.56f, 1.00f},// Z
    };

    // `fade` darkens a ball on the far side; `lift` pulls a hovered one
    // towards white, because scaling an already-saturated primary up does not
    // read as "brighter".
    ImU32 axisColor(int axis, float fade, float lift, float alpha = 1.f) {
        const auto channel = [&](float value) {
            const float faded = value * fade;
            return faded + (1.f - faded) * lift;
        };
        return ImGui::ColorConvertFloat4ToU32(
                {channel(kAxisColor[axis][0]), channel(kAxisColor[axis][1]),
                 channel(kAxisColor[axis][2]), alpha});
    }

    EditorApp::ViewPreset presetFor(int axis, bool positive) {
        using VP = EditorApp::ViewPreset;
        switch (axis) {
            case 0: return positive ? VP::Right : VP::Left;
            case 1: return positive ? VP::Top : VP::Bottom;
            default: return positive ? VP::Front : VP::Back;
        }
    }

    EditorApp::ViewPreset opposite(EditorApp::ViewPreset preset) {
        using VP = EditorApp::ViewPreset;
        switch (preset) {
            case VP::Front: return VP::Back;
            case VP::Back: return VP::Front;
            case VP::Left: return VP::Right;
            case VP::Right: return VP::Left;
            case VP::Top: return VP::Bottom;
            case VP::Bottom: return VP::Top;
            default: return preset;
        }
    }

}// namespace

void EditorApp::drawViewGizmo() {

    viewGizmoHovered_ = false;

    const float s = contentScale_;
    const auto* viewport = ImGui::GetMainViewport();

    const float radius = 40.f * s;
    const float margin = 14.f * s;
    // Top-right of the viewport: under the menu bar, clear of the inspector.
    const ImVec2 center(viewport->Pos.x + viewport->Size.x - inspectorPx() - margin - radius,
                        viewport->Pos.y + menuHeight_ + margin + radius);

    // A window small enough that the panels have eaten the viewport has no
    // corner to draw in.
    if (center.x - radius < viewport->Pos.x + hierarchyPx() + margin) return;

    // Each world axis end, rotated into camera space: x/-y is the screen
    // offset, z says which balls are in front (+1 points at the viewer).
    Quaternion inverse;
    inverse.copy(viewCamera().quaternion).invert();

    struct Ball {
        int axis = 0;
        bool positive = false;
        ImVec2 pos;
        float depth = 0.f;
        float r = 0.f;
    };
    std::array<Ball, 6> balls;

    const float arm = radius - 11.5f * s;
    std::size_t count = 0;
    for (int axis = 0; axis < 3; ++axis) {
        for (const bool positive : {true, false}) {
            Vector3 dir(axis == 0 ? 1.f : 0.f, axis == 1 ? 1.f : 0.f, axis == 2 ? 1.f : 0.f);
            if (!positive) dir.negate();
            dir.applyQuaternion(inverse);
            // A touch of perspective: the near balls a little larger.
            const float size = (positive ? 9.f : 7.f) * s * (0.9f + 0.1f * (dir.z * 0.5f + 0.5f));
            balls[count++] = {axis, positive,
                             {center.x + dir.x * arm, center.y - dir.y * arm},
                             dir.z, size};
        }
    }
    std::sort(balls.begin(), balls.end(),
              [](const Ball& a, const Ball& b) { return a.depth < b.depth; });

    const ImGuiIO& io = ImGui::GetIO();
    const auto onDisc = [&](const ImVec2& p) {
        const float px = p.x - center.x;
        const float py = p.y - center.y;
        return px * px + py * py <= radius * radius;
    };
    const bool overDisc = onDisc(io.MousePos);
    // An ImGui window over this corner (a dialog, a menu) owns the mouse; the
    // gizmo sits under every window and yields exactly as the viewport does.
    const bool interactive = overDisc && !io.WantCaptureMouse && !fileBrowser_.isOpen();

    // Press ownership, ImGui's own rule: a drag belongs to whatever it STARTED
    // on. viewGizmoHovered_ suppresses the canvas mouse events (ioCapture_),
    // and suppressing the release of a drag that merely ENDS here would leave
    // the orbit stuck mid-rotate, never having heard the button go up.
    bool ownsPress = !ImGui::IsAnyMouseDown();
    if (!ownsPress) {
        for (int button = 0; button < 3 && !ownsPress; ++button) {
            if (ImGui::IsMouseDown(button)) ownsPress = onDisc(io.MouseClickedPos[button]);
        }
    }

    viewGizmoHovered_ = interactive && ownsPress;

    // A drag in the viewport mid-flight wins - the tween is a convenience,
    // not a mode the user has to wait out.
    if (viewTween_.active && !io.WantCaptureMouse && !overDisc && ImGui::IsAnyMouseDown()) {
        viewTween_.active = false;
    }

    // The frontmost ball under the pointer: the list is sorted back-to-front,
    // so the last hit is the one on top. A drag that started elsewhere is not
    // pointing, so it neither highlights nor clicks.
    const Ball* hot = nullptr;
    if (interactive && ownsPress) {
        for (const auto& ball : balls) {
            const float bx = io.MousePos.x - ball.pos.x;
            const float by = io.MousePos.y - ball.pos.y;
            const float reach = ball.r + 2.f * s;
            if (bx * bx + by * by <= reach * reach) hot = &ball;
        }
    }

    auto* draw = ImGui::GetBackgroundDrawList();

    // Backdrop, so the balls stay readable over whatever the scene puts in
    // this corner; a touch stronger while the pointer is on it.
    draw->AddCircleFilled(center, radius, IM_COL32(12, 14, 17, interactive ? 150 : 90));

    for (const auto& ball : balls) {
        // Balls on the far side fade back rather than shrink out of reach.
        const float fade = 0.55f + 0.45f * (ball.depth * 0.5f + 0.5f);
        const bool isHot = hot == &ball;
        const ImU32 color = axisColor(ball.axis, fade, isHot ? 0.35f : 0.f);

        if (ball.positive) {
            draw->AddLine(center, ball.pos, color, 2.2f * s);
            draw->AddCircleFilled(ball.pos, ball.r, color);
            const char label[2] = {static_cast<char>('X' + ball.axis), '\0'};
            const ImVec2 half = ImGui::CalcTextSize(label);
            draw->AddText({ball.pos.x - half.x * 0.5f, ball.pos.y - half.y * 0.5f},
                          IM_COL32(20, 23, 26, 255), label);
        } else {
            // The far end: hollow, a faint fill under a coloured rim.
            draw->AddCircleFilled(ball.pos, ball.r,
                                  axisColor(ball.axis, fade, 0.f, isHot ? 0.9f : 0.25f));
            draw->AddCircle(ball.pos, ball.r, color, 0, 1.6f * s);
        }
    }

    if (hot) {
        ImGui::SetTooltip("%s", viewPresetLabel(presetFor(hot->axis, hot->positive)));

        // A click, not the end of a drag that happened to land here - the
        // same test the pick gate applies.
        const auto drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.f);
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && !gizmo_->isDragging() &&
            std::abs(drag.x) < 3.f * s && std::abs(drag.y) < 3.f * s) {
            // An axis view is an orthographic view - the numpad's policy, and
            // the gizmo is the same request by mouse.
            setOrthographic(true);
            startViewTween(presetFor(hot->axis, hot->positive));
        }
    }
}

void EditorApp::startViewTween(ViewPreset preset) {

    Vector3 from;
    from.subVectors(viewCamera().position, orbit_->target);
    if (from.length() < 1e-6f) {
        setViewPreset(preset);
        return;
    }
    from.normalize();

    // Asking for the axis the camera already stands on means the far end -
    // the same flip Ctrl gives the numpad keys.
    if (from.dot(viewPresetDirection(preset)) > 0.9999f) preset = opposite(preset);

    viewTween_ = {true, 0.f, from, preset};
}

void EditorApp::updateViewTween(float dt) {

    if (!viewTween_.active) return;

    constexpr float duration = 0.35f;
    viewTween_.t += dt / duration;

    if (viewTween_.t >= 1.f) {
        viewTween_.active = false;
        // Landing goes through setViewPreset so the flight cannot drift from
        // the label: the exact direction, the ortho clear-distance push and
        // the grid placement are all its.
        setViewPreset(viewTween_.preset);
        return;
    }

    // Smoothstep: ease out of the standing view and into the axis one.
    const float t = viewTween_.t;
    const float k = t * t * (3.f - 2.f * t);

    Quaternion full;
    full.setFromUnitVectors(viewTween_.from, viewPresetDirection(viewTween_.preset));
    Quaternion part;
    part.slerp(full, k);
    Vector3 dir = viewTween_.from;
    dir.applyQuaternion(part);

    // The view turns, it does not dolly: distance and target stay the orbit's
    // own, which is also what lets a follow chase or a wheel zoom compose
    // with the flight.
    Camera& camera = viewCamera();
    const Vector3 target = orbit_->target;
    const float distance = std::max(camera.position.distanceTo(target), 0.01f);
    camera.position.copy(target).addScaledVector(dir, distance);
    camera.lookAt(target);
}
