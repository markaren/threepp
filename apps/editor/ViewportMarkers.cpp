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
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/lights/Light.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
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

    // One icon per kind of object, so a scene reads at a glance: the light
    // types are told apart by shape, not by clicking each one.
    //
    // The artwork is original: plain geometry on a 24x24 viewBox, no traced
    // icon set, which keeps the editor's assets as first-party as its code.
    enum class Icon {
        Camera,
        DirectionalLight,
        PointLight,
        SpotLight,
        AmbientLight,
        HemisphereLight,
        kCount
    };

    const char* iconSource(Icon icon) {

        switch (icon) {

            // Body plus the lens wedge that gives the camera a facing.
            case Icon::Camera:
                return R"(<svg viewBox="0 0 24 24">
<path d="M3 7 L16 7 L16 17 L3 17 Z"/>
<path d="M17 10.5 L21.5 7 L21.5 17 L17 13.5 Z"/>
</svg>)";

            // A sun: parallel light from far away.
            case Icon::DirectionalLight:
                return R"(<svg viewBox="0 0 24 24">
<path d="M12 8 C14.21 8 16 9.79 16 12 C16 14.21 14.21 16 12 16 C9.79 16 8 14.21 8 12 C8 9.79 9.79 8 12 8 Z"/>
<path d="M11.25 1.5 L12.75 1.5 L12.75 5 L11.25 5 Z"/>
<path d="M11.25 19 L12.75 19 L12.75 22.5 L11.25 22.5 Z"/>
<path d="M1.5 11.25 L5 11.25 L5 12.75 L1.5 12.75 Z"/>
<path d="M19 11.25 L22.5 11.25 L22.5 12.75 L19 12.75 Z"/>
<path d="M17.13 7.93 L19.63 5.43 L18.57 4.37 L16.07 6.87 Z"/>
<path d="M6.87 7.93 L4.37 5.43 L5.43 4.37 L7.93 6.87 Z"/>
<path d="M17.13 16.07 L19.63 18.57 L18.57 19.63 L16.07 17.13 Z"/>
<path d="M6.87 16.07 L4.37 18.57 L5.43 19.63 L7.93 17.13 Z"/>
</svg>)";

            // A bulb: light radiating from a point.
            case Icon::PointLight:
                return R"(<svg viewBox="0 0 24 24">
<path d="M12 3 C8.69 3 6 5.69 6 9 C6 11.22 7.21 13.16 9 14.2 L9 16.5 L15 16.5 L15 14.2 C16.79 13.16 18 11.22 18 9 C18 5.69 15.31 3 12 3 Z"/>
<path d="M9.5 18 L14.5 18 L14.5 19.5 L9.5 19.5 Z"/>
<path d="M10.5 21 L13.5 21 L13.5 22.2 L10.5 22.2 Z"/>
</svg>)";

            // A housing with a widening beam: a cone with a direction.
            case Icon::SpotLight:
                return R"(<svg viewBox="0 0 24 24">
<path d="M9 2 L15 2 L15 6 L9 6 Z"/>
<path d="M9.8 7 L14.2 7 L18.5 21 L5.5 21 Z"/>
</svg>)";

            // A core ringed by light: illumination from every direction.
            case Icon::AmbientLight:
                return R"(<svg viewBox="0 0 24 24">
<path d="M12 7.5 C14.49 7.5 16.5 9.51 16.5 12 C16.5 14.49 14.49 16.5 12 16.5 C9.51 16.5 7.5 14.49 7.5 12 C7.5 9.51 9.51 7.5 12 7.5 Z"/>
<path d="M19.7 11.2 L21.3 11.2 L21.3 12.8 L19.7 12.8 Z"/>
<path d="M17.21 17.21 L18.81 17.21 L18.81 18.81 L17.21 18.81 Z"/>
<path d="M11.2 19.7 L12.8 19.7 L12.8 21.3 L11.2 21.3 Z"/>
<path d="M5.19 17.21 L6.79 17.21 L6.79 18.81 L5.19 18.81 Z"/>
<path d="M2.7 11.2 L4.3 11.2 L4.3 12.8 L2.7 12.8 Z"/>
<path d="M5.19 5.19 L6.79 5.19 L6.79 6.79 L5.19 6.79 Z"/>
<path d="M11.2 2.7 L12.8 2.7 L12.8 4.3 L11.2 4.3 Z"/>
<path d="M17.21 5.19 L18.81 5.19 L18.81 6.79 L17.21 6.79 Z"/>
</svg>)";

            // Sky dome over ground: two colours, one from above, one below.
            case Icon::HemisphereLight:
            default:
                return R"(<svg viewBox="0 0 24 24">
<path d="M4 13 C4 8.58 7.58 5 12 5 C16.42 5 20 8.58 20 13 Z"/>
<path d="M3 15.5 L21 15.5 L21 18.5 L3 18.5 Z"/>
</svg>)";
        }
    }

    Icon iconFor(Object3D& object) {

        if (object.as<Camera>()) return Icon::Camera;
        if (object.as<DirectionalLight>()) return Icon::DirectionalLight;
        if (object.as<SpotLight>()) return Icon::SpotLight;
        if (object.as<PointLight>()) return Icon::PointLight;
        if (object.as<HemisphereLight>()) return Icon::HemisphereLight;
        // AmbientLight, and any light type added later, until it earns a shape.
        return Icon::AmbientLight;
    }

    // On-screen height at 100% DPI, multiplied by the monitor content scale.
    constexpr float kMarkerPixels = 26.f;

    // Drawn after scene geometry so an icon is never buried inside the object
    // it stands for.
    constexpr int kMarkerRenderOrder = 4000;

    struct BuiltMarker {
        std::shared_ptr<Object3D> node;
        std::vector<std::shared_ptr<MeshBasicMaterial>> materials;
    };

    // The triangulated icon, normalised to a 1x1 box centred on its origin so
    // a marker only ever sets position/rotation/scale.
    struct IconArt {
        std::vector<std::shared_ptr<BufferGeometry>> geometries;
        Vector3 scale{1.f, 1.f, 1.f};
        Vector3 offset;
        bool attempted = false;
        bool valid = false;
    };

    // Parsed and triangulated once per icon, then shared by every marker of
    // that kind — geometry is immutable here, and only the materials differ
    // (they carry the per-marker selection tint).
    const IconArt& iconArt(Icon icon) {

        static IconArt cache[static_cast<int>(Icon::kCount)];
        IconArt& art = cache[static_cast<int>(icon)];
        if (art.attempted) return art;
        art.attempted = true;

        SVGLoader loader;
        std::vector<SVGLoader::SVGData> paths;
        try {
            paths = loader.parse(iconSource(icon));
        } catch (const std::exception&) {
            return art;
        }

        // Meshed once here purely to measure the union bounds; the geometries
        // outlive this scratch group.
        auto scratch = Group::create();
        for (const auto& entry : paths) {
            if (entry.style.fill && *entry.style.fill == "none") continue;

            const auto shapes = SVGLoader::createShapes(entry);
            if (shapes.empty()) continue;

            auto geometry = ShapeGeometry::create(shapes);
            art.geometries.push_back(geometry);
            scratch->add(Mesh::create(geometry, MeshBasicMaterial::create()));
        }
        if (art.geometries.empty()) return art;

        scratch->updateMatrixWorld(true);
        Box3 box;
        box.setFromObject(*scratch);
        if (box.isEmpty()) return art;

        const auto center = box.getCenter();
        const auto size = box.getSize();
        const float extent = std::max(size.x, size.y);
        if (!(extent > 0.f)) return art;

        // SVG space is y-down; the negative y scale flips it upright. The
        // offset is expressed in already-scaled units because a node's local
        // matrix applies scale before translation.
        art.scale.set(1.f / extent, -1.f / extent, 1.f / extent);
        art.offset.set(-center.x / extent, center.y / extent, 0.f);
        art.valid = true;
        return art;
    }

    // Returns an empty result if the icon yielded no fillable shape.
    BuiltMarker buildMarker(Icon icon) {

        const auto& art = iconArt(icon);
        if (!art.valid) return {};

        BuiltMarker built;
        auto node = Group::create();

        for (const auto& geometry : art.geometries) {
            auto material = MeshBasicMaterial::create();
            material->side = Side::Double;// the y-flip reverses winding
            material->transparent = true; // draws in the pass after opaque
            material->depthTest = false;  // an icon is UI, not geometry
            material->depthWrite = false;
            material->toneMapped = false;

            auto mesh = Mesh::create(geometry, material);
            mesh->renderOrder = kMarkerRenderOrder;
            node->add(mesh);
            built.materials.push_back(material);
        }

        node->scale.copy(art.scale);
        node->position.copy(art.offset);

        built.node = Group::create();
        built.node->add(node);
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

        auto built = buildMarker(iconFor(*owner));
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
