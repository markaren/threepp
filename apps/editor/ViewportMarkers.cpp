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

#include "threepp/extras/editor/ConveyorConfig.hpp"
#include "threepp/extras/editor/GranularConfig.hpp"
#include "threepp/extras/editor/JointConfig.hpp"
#include "threepp/extras/editor/ParticleFieldConfig.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"
#include "threepp/extras/editor/SoundConfig.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"
#include "threepp/extras/editor/VehicleConfig.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/geometries/ShapeGeometry.hpp"
#include "threepp/helpers/CameraHelper.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/lights/Light.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/loaders/SVGLoader.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <imgui.h>// theme colours are ImVec4

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

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
        SplinePoint,
        ConveyorPoint,
        WallPoint,
        Sensor,
        Sound,
        Joint,
        Vehicle,
        Particles,
        Granular,
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
                return R"(<svg viewBox="0 0 24 24">
<path d="M4 13 C4 8.58 7.58 5 12 5 C16.42 5 20 8.58 20 13 Z"/>
<path d="M3 15.5 L21 15.5 L21 18.5 L3 18.5 Z"/>
</svg>)";

            // A domed puck on a mount, emitting. Reads as instrumentation from
            // across the viewport without needing to say which KIND — the
            // inspector and the Sensors tab both name that, and one glyph that
            // means "there is a sensor here" is what a marker is for.
            case Icon::Sensor:
                return R"(<svg viewBox="0 0 24 24">
<path d="M12 5.5 C14.76 5.5 17 7.74 17 10.5 L7 10.5 C7 7.74 9.24 5.5 12 5.5 Z"/>
<path d="M7 10.5 L17 10.5 L17 19.5 L7 19.5 Z"/>
<path d="M5.5 19.5 L18.5 19.5 L18.5 21.8 L5.5 21.8 Z"/>
<path d="M11.25 1.2 L12.75 1.2 L12.75 4 L11.25 4 Z"/>
<path d="M6 2.6 L7.3 1.85 L8.9 4.6 L7.6 5.35 Z"/>
<path d="M18 2.6 L16.7 1.85 L15.1 4.6 L16.4 5.35 Z"/>
</svg>)";

            // A driver with a horn and two radiating arcs. Says "something is
            // heard from here" — which is the whole of what a sound node is,
            // the file and the falloff being the inspector's business.
            case Icon::Sound:
                return R"(<svg viewBox="0 0 24 24">
<path d="M3.5 9.5 L7.5 9.5 L13 4.5 L13 19.5 L7.5 14.5 L3.5 14.5 Z"/>
<path d="M15.74 8.95 A4.1 4.1 0 0 1 15.74 15.05 L15.01 14.23 A3 3 0 0 0 15.01 9.77 Z"/>
<path d="M17.61 7.85 A6.2 6.2 0 0 1 17.61 16.15 L16.79 15.41 A5.1 5.1 0 0 0 16.79 8.59 Z"/>
</svg>)";

            // An open hinge: two leaves meeting at a knuckle ring. Says "two
            // bodies are constrained here" — which type, and to what, is the
            // inspector's business.
            case Icon::Joint:
                return R"(<svg viewBox="0 0 24 24">
<path d="M2.5 10.6 L8 10.6 L8 13.4 L2.5 13.4 Z"/>
<path d="M14.4 9.7 L18.9 4.6 L21.1 6.5 L16.6 11.6 Z"/>
<path d="M12 8 C14.21 8 16 9.79 16 12 C16 14.21 14.21 16 12 16 C9.79 16 8 14.21 8 12 C8 9.79 9.79 8 12 8 Z M12 10.2 C11.01 10.2 10.2 11.01 10.2 12 C10.2 12.99 11.01 13.8 12 13.8 C12.99 13.8 13.8 12.99 13.8 12 C13.8 11.01 12.99 10.2 12 10.2 Z"/>
</svg>)";

            // A car in profile: cabin over body, two ringed wheels. Says
            // "this model drives" — the picks and the tuning are the
            // inspector's business.
            case Icon::Vehicle:
                return R"(<svg viewBox="0 0 24 24">
<path d="M2.5 10.5 L6 10 L8.5 6.5 L15.5 6.5 L18.5 10 L21.5 11 L21.5 15 L2.5 15 Z"/>
<path d="M7 13.9 C8.44 13.9 9.6 15.06 9.6 16.5 C9.6 17.94 8.44 19.1 7 19.1 C5.56 19.1 4.4 17.94 4.4 16.5 C4.4 15.06 5.56 13.9 7 13.9 Z M7 15.3 C6.34 15.3 5.8 15.84 5.8 16.5 C5.8 17.16 6.34 17.7 7 17.7 C7.66 17.7 8.2 17.16 8.2 16.5 C8.2 15.84 7.66 15.3 7 15.3 Z"/>
<path d="M17 13.9 C18.44 13.9 19.6 15.06 19.6 16.5 C19.6 17.94 18.44 19.1 17 19.1 C15.56 19.1 14.4 17.94 14.4 16.5 C14.4 15.06 15.56 13.9 17 13.9 Z M17 15.3 C16.34 15.3 15.8 15.84 15.8 16.5 C15.8 17.16 16.34 17.7 17 17.7 C17.66 17.7 18.2 17.16 18.2 16.5 C18.2 15.84 17.66 15.3 17 15.3 Z"/>
</svg>)";

            // A six-armed flake with a hub, plus two falling specks. A particle
            // field draws nothing at its own origin (its Mesh geometry is a
            // zero-area placeholder), so this icon is the ONLY thing standing
            // where the emitter is — and on OpenGL it is the only thing at all.
            case Icon::Particles:
                return R"(<svg viewBox="0 0 24 24">
<path d="M8.6 3.6 L10.9 5.9 L10.9 3 L13.1 3 L13.1 5.9 L15.4 3.6 L16.95 5.15 L13.1 9 L13.1 10.9 L15 10.9 L18.85 7.05 L20.4 8.6 L18.1 10.9 L21 10.9 L21 13.1 L18.1 13.1 L20.4 15.4 L18.85 16.95 L15 13.1 L13.1 13.1 L13.1 15 L16.95 18.85 L15.4 20.4 L13.1 18.1 L13.1 21 L10.9 21 L10.9 18.1 L8.6 20.4 L7.05 18.85 L10.9 15 L10.9 13.1 L9 13.1 L5.15 16.95 L3.6 15.4 L5.9 13.1 L3 13.1 L3 10.9 L5.9 10.9 L3.6 8.6 L5.15 7.05 L9 10.9 L10.9 10.9 L10.9 9 L7.05 5.15 Z"/>
</svg>)";

            // A heap over a pour mouth: grains falling from a chute onto the
            // pile they make. The pile is the whole point of the object, and it
            // only exists while playing.
            case Icon::Granular:
                return R"(<svg viewBox="0 0 24 24">
<path d="M8 2.5 L16 2.5 L16 4.3 L8 4.3 Z"/>
<path d="M11.1 6 L12.9 6 L12.9 8.2 L11.1 8.2 Z"/>
<path d="M8.2 6.6 L9.8 6.6 L9.8 8.2 L8.2 8.2 Z"/>
<path d="M14.2 6.6 L15.8 6.6 L15.8 8.2 L14.2 8.2 Z"/>
<path d="M11.1 10 L12.9 10 L12.9 12 L11.1 12 Z"/>
<path d="M7.9 10.4 L9.5 10.4 L9.5 12 L7.9 12 Z"/>
<path d="M14.5 10.4 L16.1 10.4 L16.1 12 L14.5 12 Z"/>
<path d="M12 13.2 L21 20.5 L3 20.5 Z"/>
</svg>)";

            // A fence: two posts and a rail — a conveyor wall's point. Says
            // "this drags a barrier" before it is clicked.
            case Icon::WallPoint:
                return R"(<svg viewBox="0 0 24 24">
<path d="M4.5 5.5 L8 5.5 L8 20 L4.5 20 Z"/>
<path d="M16 5.5 L19.5 5.5 L19.5 20 L16 20 Z"/>
<path d="M2.5 9 L21.5 9 L21.5 12.2 L2.5 12.2 Z"/>
</svg>)";

            // A diamond handle with a core: a conveyor waypoint. Distinct from
            // the spline ring so a scene with both says which system a handle
            // belongs to before it is clicked.
            case Icon::ConveyorPoint:
                return R"(<svg viewBox="0 0 24 24">
<path d="M12 2.5 L21.5 12 L12 21.5 L2.5 12 Z M12 6.5 L6.5 12 L12 17.5 L17.5 12 Z"/>
<path d="M12 9.5 L14.5 12 L12 14.5 L9.5 12 Z"/>
</svg>)";

            // A ringed handle: a knot on a curve, small enough not to bury the
            // line it sits on. Drawn as a ring because a solid disc at the
            // sizes a spline is authored at reads as a blob.
            case Icon::SplinePoint:
            default:
                return R"(<svg viewBox="0 0 24 24">
<path d="M12 3 C16.97 3 21 7.03 21 12 C21 16.97 16.97 21 12 21 C7.03 21 3 16.97 3 12 C3 7.03 7.03 3 12 3 Z M12 6.5 C8.96 6.5 6.5 8.96 6.5 12 C6.5 15.04 8.96 17.5 12 17.5 C15.04 17.5 17.5 15.04 17.5 12 C17.5 8.96 15.04 6.5 12 6.5 Z"/>
<path d="M12 9.5 C13.38 9.5 14.5 10.62 14.5 12 C14.5 13.38 13.38 14.5 12 14.5 C10.62 14.5 9.5 13.38 9.5 12 C9.5 10.62 10.62 9.5 12 9.5 Z"/>
</svg>)";
        }
    }

    Icon iconFor(Object3D& object) {

        // Before everything: instrumentation is authored ON an object of some
        // other kind (a link, a mast, a wheel hub), so the sensor is the more
        // specific fact about a node that carries one.
        if (const auto sensor = SensorConfig::read(object); sensor && sensor->enabled) {
            return Icon::Sensor;
        }
        // Same reasoning: a sound is authored ON a node that may already be a
        // mesh or a light, and "there is a sound here" is the more specific
        // fact. After the sensor, which is the rarer authoring of the two.
        if (SoundConfig::isSound(object)) return Icon::Sound;
        // A joint is its own plain node — the entry is its whole identity.
        if (JointConfig::isJoint(object)) return Icon::Joint;
        // A vehicle is authored on a model that renders fine; the icon is the
        // sensor's argument — "this drives" is otherwise only discoverable by
        // selecting everything in turn.
        if (VehicleConfig::isVehicle(object)) return Icon::Vehicle;
        // Both are plain Groups whose userData entry IS their whole identity,
        // and both draw nothing of their own in edit mode — the marker is the
        // only way to click one.
        if (ParticleFieldConfig::isParticleField(object)) return Icon::Particles;
        if (GranularConfig::isGranular(object)) return Icon::Granular;
        // Before the type checks: a control point is an ordinary Object3D and
        // is told apart by its parent, not by what it is.
        if (SplineConfig::splineOf(object)) return Icon::SplinePoint;
        if (ConveyorConfig::conveyorOf(object)) return Icon::ConveyorPoint;
        if (ConveyorWallConfig::wallOf(object)) return Icon::WallPoint;
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

    // Over scene geometry, under the icons — the band every other editor
    // overlay draws in.
    constexpr int kSoundRingRenderOrder = 3000;
    constexpr int kSoundRingSegments = 72;
    // A ring bigger than this reads as a straight line across the viewport and
    // says nothing — and maxDistance defaults to miniaudio's 10000, which means
    // "unbounded" rather than "ten kilometres". Beyond the cap the ring is left
    // out; the number is still in the inspector.
    constexpr float kSoundRingMaxRadius = 1000.f;

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
        // Spline control points draw nothing either — the curve does — so they
        // are unclickable without an icon of their own.
        //
        // An instrumented object is the one case where the marker is NOT about
        // being invisible: a wheel hub with an IMU on it draws perfectly well.
        // The icon says the instrumentation is there, which is otherwise only
        // discoverable by selecting every object in turn.
        const auto sensor = SensorConfig::read(object);
        if (object.as<Camera>() || object.as<Light>() || SplineConfig::splineOf(object) ||
            ConveyorConfig::conveyorOf(object) || ConveyorWallConfig::wallOf(object) ||
            (sensor && sensor->enabled) || SoundConfig::isSound(object) ||
            JointConfig::isJoint(object) || VehicleConfig::isVehicle(object) ||
            // Both are empty Groups: they bound to nothing, so selectObject
            // builds no outline for them and the icon is the selection path.
            ParticleFieldConfig::isParticleField(object) ||
            GranularConfig::isGranular(object)) {
            owners.push_back(&object);
        }
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

    // --- retire markers whose owner changed kind ---------------------------
    // Authoring a sensor on a camera changes which glyph it wants, and the
    // geometry is baked at build time. Cheaper to rebuild the one marker than to
    // track authoring edits through the command stack.
    for (auto it = viewportMarkers_.begin(); it != viewportMarkers_.end();) {
        if (it->owner && it->icon != static_cast<int>(iconFor(*it->owner))) {
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

        const auto icon = iconFor(*owner);
        auto built = buildMarker(icon);
        // A marker that failed to parse is simply skipped; the object stays
        // selectable from the hierarchy, and the selftest asserts it parses.
        if (!built.node) continue;

        markers_->add(built.node);
        viewportMarkers_.push_back(ViewportMarker{owner, built.node,
                                                  std::move(built.materials),
                                                  static_cast<int>(icon)});
    }

    if (viewportMarkers_.empty()) return;

    // --- place, face and size them ----------------------------------------
    // Constant screen size, whichever projection is active — see
    // viewportWorldPerPixel(). Under perspective that is 2*d*tan(fov/2) per
    // frame height; under ortho the frustum height, the same everywhere.
    const float pixels = kMarkerPixels * contentScale_;
    const auto& viewQuaternion = viewCamera().quaternion;
    const auto* selected = selection_.get();

    Vector3 world;
    for (auto& marker : viewportMarkers_) {

        marker.owner->getWorldPosition(world);
        marker.node->position.copy(world);
        // Billboard: adopt the viewer's orientation outright, so the icon is
        // never edge-on regardless of how the owner is rotated.
        marker.node->quaternion.copy(viewQuaternion);

        const float scale = pixels * viewportWorldPerPixel(world);
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

// --------------------------------------------------------------- sound rings

void EditorApp::syncSoundRings() {

    // Only for the SELECTED sound: rings for every sound in the scene would
    // bury the scene they are drawn over, and the numbers are in the inspector
    // either way. Same rule the corner-radius handle and the camera frustum
    // follow.
    auto* selected = selection_.get();
    const auto config = selected ? SoundConfig::read(*selected) : std::nullopt;
    const bool wanted = config && config->positional;

    if (!wanted) {
        clearSoundRings();
        return;
    }

    // Keyed by uuid, not by pointer: a play/stop replaces the whole graph, and
    // a stale pointer here would outlive the node it named.
    char key[160];
    std::snprintf(key, sizeof(key), "%s|%g|%g", selected->uuid.c_str(),
                  static_cast<double>(config->minDistance),
                  static_cast<double>(config->maxDistance));

    if (!soundRings_) {
        auto material = LineBasicMaterial::create(
                LineBasicMaterial::Params().vertexColors(true).toneMapped(false));
        material->transparent = true;
        material->opacity = 0.7f;
        soundRings_ = LineSegments::create(BufferGeometry::create(), material);
        soundRings_->renderOrder = kSoundRingRenderOrder;
        // World-space vertices written in place, so a cached bound would be
        // wrong the moment the sound moves.
        soundRings_->frustumCulled = false;
        soundRings_->matrixAutoUpdate = false;
        soundRingsKey_.clear();
        overlay_->add(soundRings_);
    }

    if (soundRingsKey_ != key) {
        soundRingsKey_ = key;

        // maxDistance defaults to miniaudio's 10000 — "effectively unbounded".
        // A 10 km circle draws as a straight line across the viewport and says
        // nothing, so beyond a sane radius the ring is simply left out; the
        // inspector still shows the number.
        const float radii[2]{config->minDistance, config->maxDistance};
        const auto tint = theme::markerIdle();
        const float shade[2]{1.f, 0.45f};

        std::vector<float> positions;
        std::vector<float> colors;
        for (int ring = 0; ring < 2; ++ring) {
            const float radius = radii[ring];
            if (!(radius > 1e-3f) || radius > kSoundRingMaxRadius) continue;
            for (int i = 0; i < kSoundRingSegments; ++i) {
                const float a0 = math::TWO_PI * static_cast<float>(i) / kSoundRingSegments;
                const float a1 = math::TWO_PI * static_cast<float>(i + 1) / kSoundRingSegments;
                positions.insert(positions.end(),
                                 {radius * std::cos(a0), 0.f, radius * std::sin(a0),
                                  radius * std::cos(a1), 0.f, radius * std::sin(a1)});
                for (int v = 0; v < 2; ++v) {
                    colors.insert(colors.end(), {tint.x * shade[ring],
                                                 tint.y * shade[ring],
                                                 tint.z * shade[ring]});
                }
            }
        }

        // Replaced wholesale rather than rewritten: the buffer only changes
        // when a radius is edited, and the old geometry is disposed so the
        // renderer provably lets go of it (see SplineOverlay's writeSamples).
        const auto old = soundRings_->geometry();
        auto geometry = BufferGeometry::create();
        if (!positions.empty()) {
            geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
            geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));
        }
        soundRings_->setGeometry(geometry);
        if (old) old->dispose();
    }

    // Every frame, not only on a rebuild: applyAuthoringVisibility hides the
    // node for a --screenshot pass or a Play, and nothing else would turn it
    // back on.
    soundRings_->visible = soundRings_->geometry() &&
                           soundRings_->geometry()->hasAttribute("position");

    // Placed every frame too: the rings are local to the sound and the gizmo
    // can be dragging it. Position only — a ring that rolled with the node
    // would stop reading as a distance on the ground.
    Vector3 world;
    selected->getWorldPosition(world);
    soundRings_->matrix->makeTranslation(world.x, world.y, world.z);
    soundRings_->matrixWorldNeedsUpdate = true;
}

void EditorApp::clearSoundRings() {

    if (!soundRings_) return;
    soundRings_->removeFromParent();
    soundRings_.reset();
    soundRingsKey_.clear();
}

// -------------------------------------------------------------- joint helper

void EditorApp::syncJointHelper() {

    // Only for the SELECTED joint, the sound rings' rule: the helper answers
    // "where is the anchor and which way does the axis point" for the node
    // being edited, and a scene full of axes buries the scene.
    auto* selected = selection_.get();
    const auto config = selected ? JointConfig::read(*selected) : std::nullopt;

    if (!config) {
        clearJointHelper();
        return;
    }

    // Keyed by uuid, not by pointer (a play/stop replaces the whole graph),
    // plus the one field the picture is built from.
    char key[160];
    std::snprintf(key, sizeof(key), "%s|%d", selected->uuid.c_str(),
                  static_cast<int>(config->type));

    if (!jointHelper_) {
        auto material = LineBasicMaterial::create(
                LineBasicMaterial::Params().vertexColors(true).toneMapped(false));
        material->transparent = true;
        material->opacity = 0.9f;
        // The anchor usually sits ON a body — often inside it — and an axis
        // hidden by the mesh it hinges is an axis authored blind.
        material->depthTest = false;
        jointHelper_ = LineSegments::create(BufferGeometry::create(), material);
        jointHelper_->renderOrder = kSoundRingRenderOrder;
        jointHelper_->frustumCulled = false;
        jointHelper_->matrixAutoUpdate = false;
        jointHelperKey_.clear();
        overlay_->add(jointHelper_);
    }

    if (jointHelperKey_ != key) {
        jointHelperKey_ = key;

        // Built in UNIT space: the axis is ±1 along local X, and the frame
        // matrix below scales it to constant screen size. The X axis is the
        // hinge/slide axis by convention, so that line IS the authoring aid.
        std::vector<float> positions;
        std::vector<float> colors;
        const auto tint = theme::accent();

        const auto seg = [&](float ax, float ay, float az,
                             float bx, float by, float bz, float shade) {
            positions.insert(positions.end(), {ax, ay, az, bx, by, bz});
            for (int v = 0; v < 2; ++v) {
                colors.insert(colors.end(), {tint.x * shade, tint.y * shade, tint.z * shade});
            }
        };
        // A circle around the local X axis (in the YZ plane, `plane` == 0) or
        // around Z (XY plane) — the "this rotates" glyph.
        const auto circle = [&](int plane, float radius, float shade) {
            constexpr int segments = 32;
            for (int i = 0; i < segments; ++i) {
                const float a0 = math::TWO_PI * static_cast<float>(i) / segments;
                const float a1 = math::TWO_PI * static_cast<float>(i + 1) / segments;
                if (plane == 0) {
                    seg(0.f, radius * std::cos(a0), radius * std::sin(a0),
                        0.f, radius * std::cos(a1), radius * std::sin(a1), shade);
                } else {
                    seg(radius * std::cos(a0), radius * std::sin(a0), 0.f,
                        radius * std::cos(a1), radius * std::sin(a1), 0.f, shade);
                }
            }
        };

        const bool axial = config->type == JointConfig::Type::Revolute ||
                           config->type == JointConfig::Type::Prismatic ||
                           config->type == JointConfig::Type::Spherical;
        if (axial) {
            seg(-1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f);
            // Arrowhead on +X, so the axis reads as a direction, not a bar —
            // a positive drive target turns/slides towards it.
            seg(1.f, 0.f, 0.f, 0.86f, 0.06f, 0.f, 1.f);
            seg(1.f, 0.f, 0.f, 0.86f, -0.06f, 0.f, 1.f);
            if (config->type == JointConfig::Type::Prismatic) {
                seg(-1.f, 0.f, 0.f, -0.86f, 0.06f, 0.f, 1.f);
                seg(-1.f, 0.f, 0.f, -0.86f, -0.06f, 0.f, 1.f);
            }
        }
        if (config->type == JointConfig::Type::Revolute) {
            circle(0, 0.45f, 0.55f);// the rotation plane
        }
        if (config->type == JointConfig::Type::Spherical) {
            circle(0, 0.45f, 0.55f);
            circle(1, 0.45f, 0.55f);// two rings read as a ball
        }
        // The anchor cross, every type: the point the bodies are tied at.
        const float tick = axial ? 0.18f : 0.3f;
        if (!axial) seg(-tick, 0.f, 0.f, tick, 0.f, 0.f, 0.8f);
        seg(0.f, -tick, 0.f, 0.f, tick, 0.f, 0.8f);
        seg(0.f, 0.f, -tick, 0.f, 0.f, tick, 0.8f);

        // Replaced wholesale rather than rewritten: the buffer only changes
        // when the type does, and the old geometry is disposed so the renderer
        // provably lets go of it (see the sound rings above).
        const auto old = jointHelper_->geometry();
        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));
        jointHelper_->setGeometry(geometry);
        if (old) old->dispose();
    }

    // Every frame: applyAuthoringVisibility hides the node for a screenshot
    // pass or a Play, and nothing else would turn it back on.
    jointHelper_->visible = true;

    // Placed every frame — the gizmo can be dragging the node. Position AND
    // rotation, unlike the rings: the orientation is the payload here. Scale
    // is constant screen size, so the axis is legible at any zoom.
    selected->updateWorldMatrix(true, false);
    Vector3 position, scale;
    Quaternion rotation;
    selected->matrixWorld->decompose(position, rotation, scale);
    const float size = 70.f * contentScale_ * viewportWorldPerPixel(position);
    jointHelper_->matrix->compose(position, rotation, Vector3(size, size, size));
    jointHelper_->matrixWorldNeedsUpdate = true;
}

void EditorApp::clearJointHelper() {

    if (!jointHelper_) return;
    jointHelper_->removeFromParent();
    jointHelper_.reset();
    jointHelperKey_.clear();
}

// ------------------------------------------------------------- vehicle rings

void EditorApp::syncVehicleHelper() {

    // Only for the SELECTED vehicle, the joint helper's rule. The rings are
    // the proof of the picks: a circle of the effective radius lying exactly
    // on each resolved wheel mesh, plus a forward chevron at the chassis
    // centre — the two facts (which meshes, which way is forward) a car is
    // authored wrong without.
    auto* selected = selection_.get();
    const auto config = selected && VehicleConfig::isVehicle(*selected)
                                ? VehicleConfig::read(*selected)
                                : std::nullopt;
    if (!config) {
        clearVehicleHelper();
        return;
    }

    const auto geo = config->derived(*selected);
    if (!geo.valid) {
        // Nothing measurable to draw; the inspector names the missing pick.
        clearVehicleHelper();
        return;
    }

    // What Play will actually use, so the picture cannot flatter the config.
    const float radius = config->autoGeometry ? geo.wheelRadius
                                              : std::max(config->wheelRadius, 0.01f);
    float hubY = 0.f;
    for (const auto& hub : geo.hubs) hubY += 0.25f * hub.y;

    // Keyed by uuid plus every number the picture is built from — a rebuild
    // trigger, not a hash; the placement below is per-frame regardless.
    char key[256];
    std::snprintf(key, sizeof(key),
                  "%s|%.3f|%.3f|%.2f,%.2f,%.2f|%.2f,%.2f,%.2f|%.2f,%.2f,%.2f|%.2f,%.2f,%.2f",
                  selected->uuid.c_str(), static_cast<double>(radius), static_cast<double>(hubY),
                  static_cast<double>(geo.hubs[0].x), static_cast<double>(geo.hubs[0].y),
                  static_cast<double>(geo.hubs[0].z),
                  static_cast<double>(geo.hubs[1].x), static_cast<double>(geo.hubs[1].y),
                  static_cast<double>(geo.hubs[1].z),
                  static_cast<double>(geo.hubs[2].x), static_cast<double>(geo.hubs[2].y),
                  static_cast<double>(geo.hubs[2].z),
                  static_cast<double>(geo.hubs[3].x), static_cast<double>(geo.hubs[3].y),
                  static_cast<double>(geo.hubs[3].z));

    if (!vehicleHelper_) {
        auto material = LineBasicMaterial::create(
                LineBasicMaterial::Params().vertexColors(true).toneMapped(false));
        material->transparent = true;
        material->opacity = 0.9f;
        // The wheels sit inside their arches; a ring hidden by bodywork
        // proves nothing.
        material->depthTest = false;
        vehicleHelper_ = LineSegments::create(BufferGeometry::create(), material);
        vehicleHelper_->renderOrder = kSoundRingRenderOrder;
        vehicleHelper_->frustumCulled = false;
        vehicleHelper_->matrixAutoUpdate = false;
        vehicleHelperKey_.clear();
        overlay_->add(vehicleHelper_);
    }

    if (vehicleHelperKey_ != key) {
        vehicleHelperKey_ = key;

        // Built in the chassis frame, real metres — the rings claim "this is
        // the wheel", so they must lie on it at any zoom, unlike the
        // screen-sized joint helper.
        std::vector<float> positions;
        std::vector<float> colors;
        const auto tint = theme::accent();

        const auto seg = [&](const Vector3& a, const Vector3& b, float shade) {
            positions.insert(positions.end(), {a.x, a.y, a.z, b.x, b.y, b.z});
            for (int v = 0; v < 2; ++v) {
                colors.insert(colors.end(), {tint.x * shade, tint.y * shade, tint.z * shade});
            }
        };

        // One ring per wheel, in the YZ plane (the axle is chassis X).
        constexpr int segments = 48;
        for (const auto& hub : geo.hubs) {
            for (int i = 0; i < segments; ++i) {
                const float a0 = math::TWO_PI * static_cast<float>(i) / segments;
                const float a1 = math::TWO_PI * static_cast<float>(i + 1) / segments;
                seg(Vector3(hub.x, hub.y + radius * std::cos(a0), hub.z + radius * std::sin(a0)),
                    Vector3(hub.x, hub.y + radius * std::cos(a1), hub.z + radius * std::sin(a1)),
                    1.f);
            }
        }

        // The forward chevron: +Z is where W takes it.
        const float r = radius;
        seg(Vector3(0.f, hubY, -r), Vector3(0.f, hubY, 2.f * r), 0.6f);
        seg(Vector3(0.f, hubY, 2.f * r), Vector3(0.3f * r, hubY, 1.4f * r), 0.6f);
        seg(Vector3(0.f, hubY, 2.f * r), Vector3(-0.3f * r, hubY, 1.4f * r), 0.6f);

        // Replaced wholesale rather than rewritten, the sound rings' rule.
        const auto old = vehicleHelper_->geometry();
        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));
        vehicleHelper_->setGeometry(geometry);
        if (old) old->dispose();
    }

    // Every frame: applyAuthoringVisibility hides the node for a screenshot
    // pass or a Play, and nothing else would turn it back on.
    vehicleHelper_->visible = true;

    // Placed every frame at the derived chassis frame — the gizmo can be
    // dragging the model. Unit scale: the vertices are already metres.
    vehicleHelper_->matrix->compose(geo.position, geo.rotation, Vector3(1.f, 1.f, 1.f));
    vehicleHelper_->matrixWorldNeedsUpdate = true;
}

void EditorApp::clearVehicleHelper() {

    if (!vehicleHelper_) return;
    vehicleHelper_->removeFromParent();
    vehicleHelper_.reset();
    vehicleHelperKey_.clear();
}
