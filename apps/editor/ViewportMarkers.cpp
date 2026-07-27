// Viewport markers: billboarded icons for scene objects that have nothing to
// draw.
//
// A Camera or a Light is invisible in a rendered scene, which leaves it
// unclickable and easy to lose. Every editor solves this the same way — a flat
// icon that always faces the viewer at a constant screen size — and so does
// this one. The artwork is SVG parsed at startup by threepp's own SVGLoader,
// embedded as source rather than shipped as files so the editor does not depend
// on its working directory.
//
// Markers live under the editor overlay, so they are never saved, never picked
// as themselves (clicking one selects its owner) and never appear in the
// through-the-lens camera preview, which hides the overlay wholesale.

#include "EditorApp.hpp"
#include "EditorTheme.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/geometries/ShapeGeometry.hpp"
#include "threepp/helpers/CameraHelper.hpp"
#include "threepp/lights/Light.hpp"
#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <imgui.h>// theme colours are ImVec4

#include <algorithm>
#include <cmath>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // A video camera: body, plus the lens wedge that gives it a facing.
    constexpr const char* kCameraSvg = R"(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
<path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/>
</svg>)";

    // A bulb, for every light type — the inspector says which kind it is.
    constexpr const char* kLightSvg = R"(<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
<path d="M9 21c0 .55.45 1 1 1h4c.55 0 1-.45 1-1v-1H9v1zm3-19C8.14 2 5 5.14 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.86-3.14-7-7-7z"/>
</svg>)";

    // On-screen height at 100% DPI, multiplied by the monitor content scale.
    constexpr float kMarkerPixels = 26.f;

    // Drawn after scene geometry so an icon is never buried inside the object
    // it stands for.
    constexpr int kMarkerRenderOrder = 4000;

    struct BuiltMarker {
        std::shared_ptr<Object3D> node;
        std::vector<std::shared_ptr<MeshBasicMaterial>> materials;
    };

    // Parses `source` into a node normalised to a 1x1 box centred on its own
    // origin, so the caller only ever sets position/rotation/scale. Returns an
    // empty result if the SVG yields no fillable shape.
    BuiltMarker buildMarker(const char* source) {

        SVGLoader loader;
        std::vector<SVGLoader::SVGData> paths;
        try {
            paths = loader.parse(source);
        } catch (const std::exception&) {
            return {};
        }

        BuiltMarker built;
        auto art = Group::create();

        for (const auto& entry : paths) {
            if (entry.style.fill && *entry.style.fill == "none") continue;

            const auto shapes = SVGLoader::createShapes(entry);
            if (shapes.empty()) continue;

            auto material = MeshBasicMaterial::create();
            material->side = Side::Double;// the y-flip below reverses winding
            material->transparent = true; // draws in the pass after opaque
            material->depthTest = false;  // an icon is UI, not geometry
            material->depthWrite = false;
            material->toneMapped = false;

            auto mesh = Mesh::create(ShapeGeometry::create(shapes), material);
            mesh->renderOrder = kMarkerRenderOrder;
            art->add(mesh);
            built.materials.push_back(material);
        }

        if (art->children.empty()) return {};

        art->updateMatrixWorld(true);
        Box3 box;
        box.setFromObject(*art);
        if (box.isEmpty()) return {};

        const auto center = box.getCenter();
        const auto size = box.getSize();
        const float extent = std::max(size.x, size.y);
        if (!(extent > 0.f)) return {};

        // SVG space is y-down; the negative y scale flips it upright. The
        // offset is expressed in already-scaled units because a node's local
        // matrix applies scale before translation.
        art->scale.set(1.f / extent, -1.f / extent, 1.f / extent);
        art->position.set(-center.x / extent, center.y / extent, 0.f);

        built.node = Group::create();
        built.node->add(art);
        built.node->renderOrder = kMarkerRenderOrder;
        return built;
    }

}// namespace


void EditorApp::syncViewportMarkers() {

    if (!markers_) return;

    // --- who needs a marker this frame ------------------------------------
    static thread_local std::vector<Object3D*> owners;
    owners.clear();
    document_.scene().traverse([this](Object3D& object) {
        if (document_.isEditorOnly(object)) return;
        if (object.as<Camera>() || object.as<Light>()) owners.push_back(&object);
    });

    // --- retire markers whose owner is gone -------------------------------
    for (auto it = viewportMarkers_.begin(); it != viewportMarkers_.end();) {
        if (std::find(owners.begin(), owners.end(), it->owner) == owners.end()) {
            it->node->removeFromParent();
            it = viewportMarkers_.erase(it);
        } else {
            ++it;
        }
    }

    // --- create markers for new owners ------------------------------------
    for (auto* owner : owners) {
        const bool known = std::any_of(
                viewportMarkers_.begin(), viewportMarkers_.end(),
                [owner](const ViewportMarker& m) { return m.owner == owner; });
        if (known) continue;

        const bool isCamera = owner->as<Camera>() != nullptr;
        auto built = buildMarker(isCamera ? kCameraSvg : kLightSvg);
        // A marker that failed to parse is simply skipped; the object stays
        // selectable from the hierarchy, and the selftest asserts it parses.
        if (!built.node) continue;

        markers_->add(built.node);
        viewportMarkers_.push_back(
                ViewportMarker{owner, built.node, std::move(built.materials)});
    }

    if (viewportMarkers_.empty()) return;

    // --- place, face and size them ----------------------------------------
    // Constant screen size: at distance d a pixel spans
    // 2*d*tan(fov/2)/heightInPixels world units.
    const auto height = static_cast<float>(renderer_->size().height());
    const float halfFovTan = std::tan(math::degToRad(camera_.fov) * 0.5f);
    const float pixels = kMarkerPixels * contentScale_;
    const auto* selected = selection_.get();

    Vector3 world;
    for (auto& marker : viewportMarkers_) {

        marker.owner->getWorldPosition(world);
        marker.node->position.copy(world);
        // Billboard: adopt the viewer's orientation outright, so the icon is
        // never edge-on regardless of how the owner is rotated.
        marker.node->quaternion.copy(camera_.quaternion);

        const float distance = camera_.position.distanceTo(world);
        const float scale = pixels * (2.f * distance * halfFovTan / std::max(1.f, height));
        marker.node->scale.set(scale, scale, scale);

        const bool isSelected = selected == marker.owner;
        const auto& tint = isSelected ? theme::accent() : theme::markerIdle();
        for (const auto& material : marker.materials) {
            material->color.setRGB(tint.x, tint.y, tint.z);
            material->opacity = isSelected ? 1.f : 0.75f;
        }
    }
}

Object3D* EditorApp::markerOwnerOf(Object3D* hit) const {

    for (Object3D* node = hit; node; node = node->parent) {
        for (const auto& marker : viewportMarkers_) {
            if (marker.node.get() == node) return marker.owner;
        }
    }
    return nullptr;
}

void EditorApp::clearViewportMarkers() {

    for (auto& marker : viewportMarkers_) marker.node->removeFromParent();
    viewportMarkers_.clear();
}

void EditorApp::syncCameraHelper() {

    auto* selected = selection_.get();
    auto* camera = selected ? selected->as<Camera>() : nullptr;

    if (!camera) {
        if (cameraHelper_) {
            cameraHelper_->removeFromParent();
            cameraHelper_.reset();
            cameraHelperFor_ = nullptr;
        }
        return;
    }

    if (cameraHelperFor_ != camera) {
        if (cameraHelper_) cameraHelper_->removeFromParent();
        // The helper aliases the camera's world matrix and holds a reference to
        // it, so it must not outlive the selection that owns it.
        cameraHelper_ = CameraHelper::create(*camera);
        cameraHelperFor_ = camera;
        overlay_->add(cameraHelper_);
    }

    // Cheap, and the frustum has to track fov/near/far edits live.
    cameraHelper_->update();
}
