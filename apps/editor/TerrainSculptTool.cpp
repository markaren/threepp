// The Sculpt tool in the viewport: where the pointer lands on a terrain, which
// brush is armed, and who owns the drag.
//
// The kernels live in extras/editor/TerrainSculpt.hpp and know nothing about
// mice. What is here is the three things that cannot be headless:
//
//   PRESS OWNERSHIP. With Sculpt armed and the pointer over the SELECTED
//   terrain, a left press belongs to the stroke and OrbitControls must not see
//   it — otherwise the first drag spins the camera and paints a smear across
//   the world. A miss falls straight through to normal navigation, because a
//   tool that hijacks every press is a tool you cannot look around with. The
//   answer is computed during the frame and consulted by ioCapture on the next
//   event, which is exactly how the tool palette and the view gizmo hold their
//   presses.
//
//   THE RING CURSOR. A brush with no cursor is a guess. The ring is a polyline
//   conformed to the surface (sampled height + a small lift), updated IN PLACE
//   — setXYZ and needsUpdate, never a fresh attribute: a replaced attribute
//   whose allocation the recycler handed back at the same address reads as
//   already-uploaded and the ring freezes where it was born.
//
//   STROKE BOUNDARIES. Press snapshots the whole Y column (a megabyte at 512²,
//   and bounded). Release diffs it to a tight rect, pushes ONE undo entry, and
//   only THEN recomputes bounds and re-bakes the splat — both are whole-mesh
//   work and neither belongs on a mouse-move.

#include "EditorApp.hpp"
#include "EditorTheme.hpp"

#include "threepp/extras/editor/TerrainConfig.hpp"
#include "threepp/extras/editor/TerrainSculpt.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"

#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;
using namespace threepp::editor;

namespace {

    constexpr int kRingSegments = 48;
    // Above the surface by a fraction of the brush, so the ring reads as a
    // cursor lying ON the ground rather than z-fighting into it.
    constexpr float kRingLift = 0.03f;

    // One undo entry per stroke. Holds the rect that moved and both sides of
    // it, so undo replays no arithmetic — a sculpt that re-derived its own
    // heights would drift a little further on every cycle, and the whole point
    // of the delta model is that the mesh is exact.
    class TerrainStrokeCommand: public Command {

    public:
        TerrainStrokeCommand(Object3D& target, TerrainSculpt::Patch patch, int dim,
                             std::string label)
            : target_(&target), patch_(std::move(patch)), dim_(dim),
              label_(std::move(label)), uuid_(target.uuid) {}

        void redo() override { restore(false); }
        void undo() override { restore(true); }

        [[nodiscard]] std::string name() const override { return label_; }

        // Two strokes are two strokes. Press-to-release is the entry; merging
        // would make an undo swallow work the user watched happen separately.
        bool mergeWith(const Command&) override { return false; }

        [[nodiscard]] bool rebind(Object3D& root) override {

            auto* found = findByUuid(root, uuid_);
            if (!found) return false;
            target_ = found;
            return true;
        }

    private:
        void restore(bool useBefore) {

            auto* mesh = target_ ? target_->as<Mesh>() : nullptr;
            if (!mesh || !mesh->geometry()) return;
            const auto lattice = TerrainLattice::of(*mesh->geometry(), dim_);
            if (!lattice.valid()) return;

            auto heights = TerrainConfig::heightsOf(*mesh->geometry());
            TerrainSculpt::applyPatch(heights, dim_, patch_, useBefore);

            TerrainSculpt::Rect rect;
            rect.add(patch_.x0, patch_.z0);
            rect.add(patch_.x0 + patch_.w - 1, patch_.z0 + patch_.h - 1);
            TerrainSculpt::refresh(*mesh->geometry(), heights, lattice, rect);
            mesh->geometry()->computeBoundingBox();
            mesh->geometry()->computeBoundingSphere();

            if (const auto config = TerrainConfig::read(*target_)) {
                TerrainConfig::applyAlbedo(*target_,
                                           config->bakeAlbedo(config->fieldOf(*mesh->geometry(), heights)),
                                           dim_);
            }
        }

        Object3D* target_;
        TerrainSculpt::Patch patch_;
        int dim_;
        std::string label_;
        std::string uuid_;
    };

}// namespace


bool EditorApp::sculptArmed() const {

    if (!sculptTool_ || isPlaying()) return false;
    const auto* selected = selection_.get();
    return selected && TerrainConfig::isTerrain(*selected);
}

Object3D* EditorApp::sculptTarget() const {

    auto* selected = selection_.get();
    if (!selected || !TerrainConfig::isTerrain(*selected)) return nullptr;
    return selected;
}

// The stroke owns the pointer while it runs, and the frame BEFORE it starts:
// ioCapture asks this before the press reaches OrbitControls, and the hover was
// computed at essentially the same mouse position one frame ago.
bool EditorApp::sculptOwnsMouse() const {

    return sculptStroke_.active || (sculptArmed() && sculptHover_);
}

void EditorApp::updateSculpt() {

    // The tool follows the selection: moving off a terrain drops back to Select
    // rather than leaving a brush armed over nothing.
    if (sculptTool_ && !sculptTarget()) {
        sculptTool_ = false;
        if (sculptStroke_.active) endSculptStroke();
    }

    auto* target = sculptArmed() ? sculptTarget() : nullptr;
    auto* mesh = target ? target->as<Mesh>() : nullptr;
    if (!mesh || !mesh->geometry()) {
        sculptHover_ = false;
        if (sculptStroke_.active) endSculptStroke();
        syncBrushRing();
        return;
    }

    const auto config = TerrainConfig::read(*target).value_or(TerrainConfig::makeDefault());
    const int dim = config.dim();
    const auto lattice = TerrainLattice::of(*mesh->geometry(), dim);
    if (!lattice.valid()) {
        sculptHover_ = false;
        syncBrushRing();
        return;
    }

    const ImGuiIO& io = ImGui::GetIO();
    const auto* viewport = ImGui::GetMainViewport();
    const bool pointerFree = !io.WantCaptureMouse && !fileBrowser_.isOpen() &&
                             !viewGizmoHovered_ && !toolPaletteHovered_;

    // --- where is the pointer on the terrain? -------------------------------
    bool hit = false;
    Vector3 local;
    if (pointerFree && viewport->Size.x > 0.f && viewport->Size.y > 0.f) {
        const Vector2 ndc{
                ((io.MousePos.x - viewport->Pos.x) / viewport->Size.x) * 2.f - 1.f,
                -(((io.MousePos.y - viewport->Pos.y) / viewport->Size.y) * 2.f - 1.f)};
        raycaster_.setFromCamera(ndc, viewCamera());

        // Into the mesh's LOCAL space, where the lattice lives. A terrain that
        // has been moved, turned or scaled is still a lattice there.
        target->updateMatrixWorld();
        Matrix4 inverse;
        inverse.copy(*target->matrixWorld).invert();
        Vector3 origin = raycaster_.ray.origin;
        Vector3 through = raycaster_.ray.origin + raycaster_.ray.direction * 1000.f;
        origin.applyMatrix4(inverse);
        through.applyMatrix4(inverse);
        Vector3 direction = through - origin;

        const auto heights = sculptStroke_.active && sculptStroke_.dim == dim
                                     ? sculptStroke_.heights
                                     : TerrainConfig::heightsOf(*mesh->geometry());
        hit = TerrainSculpt::raycast(heights, lattice, origin, direction,
                                     direction.length() * 1.5f, local);
    }

    sculptHover_ = hit;
    if (hit) sculptHoverLocal_ = local;

    // --- stroke boundaries ---------------------------------------------------
    const bool pressed = pointerFree && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
                         !gizmo_->isDragging();

    if (pressed && hit && !sculptStroke_.active) {
        if (!rejectWhilePlaying("Sculpt")) {
            sculptStroke_.active = true;
            sculptStroke_.uuid = target->uuid;
            sculptStroke_.dim = dim;
            // The WHOLE Y column, once. A megabyte at 512², and it is what
            // makes the release diff a diff rather than a guess.
            sculptStroke_.before = TerrainConfig::heightsOf(*mesh->geometry());
            sculptStroke_.heights = sculptStroke_.before;
            sculptStroke_.rect = {};
            // Flatten levels to the height under the PRESS, so the plateau is
            // the one the user pointed at and does not chase the cursor.
            sculptStroke_.flattenTarget = local.y;
            commands_.beginTransaction();
        }
    }

    if (sculptStroke_.active) {
        if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            endSculptStroke();
        } else if (hit) {
            auto brush = brush_;
            // Shift inverts raise into lower for the duration of the stroke —
            // the same hold-to-modify idiom Snap uses.
            brush.invert = brush.invert != io.KeyShift;
            const float dt = std::clamp(io.DeltaTime, 0.f, 0.05f);
            const auto touched = TerrainSculpt::apply(sculptStroke_.heights, lattice, brush,
                                                      local.x, local.z, dt,
                                                      sculptStroke_.flattenTarget);
            if (!touched.empty()) {
                sculptStroke_.rect.add(touched.x0, touched.z0);
                sculptStroke_.rect.add(touched.x1, touched.z1);
                // Region-scoped: positions and analytic normals over the rect
                // and one ring, nothing else. Bounds and the splat wait for the
                // release (see the header note).
                TerrainSculpt::refresh(*mesh->geometry(), sculptStroke_.heights, lattice, touched);
            }
        }
    }

    syncBrushRing();
}

void EditorApp::endSculptStroke() {

    if (!sculptStroke_.active) return;
    sculptStroke_.active = false;

    auto* target = findByUuid(document_.scene(), sculptStroke_.uuid);
    auto* mesh = target ? target->as<Mesh>() : nullptr;
    if (!mesh || !mesh->geometry()) {
        commands_.endTransaction();
        return;
    }

    const auto after = TerrainConfig::heightsOf(*mesh->geometry());
    auto patch = TerrainSculpt::diff(sculptStroke_.before, after, sculptStroke_.dim);
    if (patch.empty()) {
        // A press that never moved the ground is not an undo entry.
        commands_.endTransaction();
        return;
    }

    // Whole-mesh work, once, here: the bounds the selection outline and the
    // raycast read, and the splat that has to follow the surface the stroke
    // left behind.
    mesh->geometry()->computeBoundingBox();
    mesh->geometry()->computeBoundingSphere();
    if (const auto config = TerrainConfig::read(*target)) {
        TerrainConfig::applyAlbedo(*target,
                                   config->bakeAlbedo(config->fieldOf(*mesh->geometry(), after)),
                                   sculptStroke_.dim);
    }

    commands_.push(std::make_unique<TerrainStrokeCommand>(
            *target, std::move(patch), sculptStroke_.dim,
            std::string("Sculpt ") + TerrainBrush::label(brush_.kind)));
    commands_.endTransaction();
    document_.setDirty(true);

    sculptStroke_.before.clear();
    sculptStroke_.before.shrink_to_fit();
    sculptStroke_.heights.clear();
    sculptStroke_.heights.shrink_to_fit();
}

void EditorApp::syncBrushRing() {

    const bool wanted = sculptArmed() && (sculptHover_ || sculptStroke_.active);

    if (!brushRing_) {
        if (!wanted) return;
        auto material = LineBasicMaterial::create(
                LineBasicMaterial::Params().color(Color(0xffffff)).toneMapped(false));
        material->depthTest = false;
        std::vector<float> positions(static_cast<size_t>(kRingSegments + 1) * 3, 0.f);
        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        brushRing_ = Line::create(geometry, material);
        brushRing_->renderOrder = 10000;
        // Written in place every frame, so a cached bound would be wrong the
        // moment the pointer moves.
        brushRing_->frustumCulled = false;
        // The ring is authored in the terrain's space; the overlay is not under
        // it, so the terrain's world matrix is adopted outright.
        brushRing_->matrixAutoUpdate = false;
        overlay_->add(brushRing_);
    }

    brushRing_->visible = wanted;
    if (!wanted) return;

    auto* target = sculptTarget();
    auto* mesh = target ? target->as<Mesh>() : nullptr;
    if (!mesh || !mesh->geometry()) {
        brushRing_->visible = false;
        return;
    }

    const auto config = TerrainConfig::read(*target).value_or(TerrainConfig::makeDefault());
    const auto lattice = TerrainLattice::of(*mesh->geometry(), config.dim());
    if (!lattice.valid()) {
        brushRing_->visible = false;
        return;
    }
    const auto& heights = sculptStroke_.active && sculptStroke_.dim == config.dim()
                                  ? sculptStroke_.heights
                                  : TerrainConfig::heightsOf(*mesh->geometry());

    auto* position = brushRing_->geometry()->getAttribute<float>("position");
    if (!position) return;

    const float radius = std::max(brush_.radius, 1e-3f);
    const float lift = std::max(radius * kRingLift, lattice.cellSize() * 0.25f);
    for (int i = 0; i <= kRingSegments; ++i) {
        const float a = math::TWO_PI * static_cast<float>(i) / static_cast<float>(kRingSegments);
        const float x = sculptHoverLocal_.x + radius * std::cos(a);
        const float z = sculptHoverLocal_.z + radius * std::sin(a);
        float y = sculptHoverLocal_.y;
        // Conformed to the surface, not a flat disc floating over a slope.
        // Off the patch the ring keeps the hover height rather than diving.
        TerrainSculpt::sample(heights, lattice, x, z, y);
        // setXYZ + needsUpdate, NEVER a replaced attribute (see the header).
        position->setXYZ(i, x, y + lift, z);
    }
    position->needsUpdate();

    target->updateMatrixWorld();
    brushRing_->matrix->copy(*target->matrixWorld);
    brushRing_->matrixWorldNeedsUpdate = true;
}
