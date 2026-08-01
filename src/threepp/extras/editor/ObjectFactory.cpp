
#include "threepp/extras/editor/ObjectFactory.hpp"

#include "threepp/extras/editor/ConveyorConfig.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/ConeGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/geometries/TorusGeometry.hpp"
#include "threepp/lights/AmbientLight.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <string>
#include <unordered_set>

using namespace threepp;
using namespace threepp::editor;

namespace {

    std::shared_ptr<MeshStandardMaterial> defaultMaterial() {

        auto material = MeshStandardMaterial::create();
        // A mid grey with a little roughness reads as a surface under any
        // lighting rig; pure white blows out and pure diffuse looks like a
        // rendering error.
        material->color = Color(0xcccccc);
        material->roughness = 0.6f;
        material->metalness = 0.f;
        return material;
    }

}// namespace


std::string ObjectFactory::uniqueName(const Object3D& root, const std::string& base) {

    std::unordered_set<std::string> taken;
    // traverse() is non-const in Object3D, but naming genuinely only reads.
    const_cast<Object3D&>(root).traverse([&](Object3D& o) {
        if (!o.name.empty()) taken.insert(o.name);
    });

    if (!taken.count(base)) return base;

    for (int i = 2; i < 100000; ++i) {
        auto candidate = base + " " + std::to_string(i);
        if (!taken.count(candidate)) return candidate;
    }
    return base;
}

const char* ObjectFactory::label(Primitive type) {

    switch (type) {
        case Primitive::Box: return "Box";
        case Primitive::Sphere: return "Sphere";
        case Primitive::Plane: return "Plane";
        case Primitive::Cylinder: return "Cylinder";
        case Primitive::Cone: return "Cone";
        case Primitive::Torus: return "Torus";
    }
    return "Mesh";
}

const char* ObjectFactory::label(LightKind kind) {

    switch (kind) {
        case LightKind::Directional: return "Directional Light";
        case LightKind::Point: return "Point Light";
        case LightKind::Spot: return "Spot Light";
        case LightKind::Ambient: return "Ambient Light";
        case LightKind::Hemisphere: return "Hemisphere Light";
    }
    return "Light";
}

std::shared_ptr<Mesh> ObjectFactory::createPrimitive(Primitive type, const Object3D& root) {

    std::shared_ptr<BufferGeometry> geometry;
    // Sit new objects ON the ground plane rather than half-buried in it.
    float lift = 0.f;

    switch (type) {
        case Primitive::Box:
            geometry = BoxGeometry::create(1, 1, 1);
            lift = 0.5f;
            break;
        case Primitive::Sphere:
            geometry = SphereGeometry::create(0.5f, 32, 16);
            lift = 0.5f;
            break;
        case Primitive::Plane:
            geometry = PlaneGeometry::create(2, 2);
            break;
        case Primitive::Cylinder:
            geometry = CylinderGeometry::create(0.5f, 0.5f, 1, 32);
            lift = 0.5f;
            break;
        case Primitive::Cone:
            geometry = ConeGeometry::create(0.5f, 1, 32);
            lift = 0.5f;
            break;
        case Primitive::Torus:
            geometry = TorusGeometry::create(0.5f, 0.2f, 16, 48);
            lift = 0.7f;
            break;
    }

    auto mesh = Mesh::create(geometry, defaultMaterial());
    mesh->name = uniqueName(root, label(type));
    mesh->castShadow = true;
    mesh->receiveShadow = true;
    mesh->position.y = lift;

    if (type == Primitive::Plane) {
        // A plane authored in XY is a wall; the editor's is a floor.
        mesh->rotation.x = -math::PI / 2;
        mesh->castShadow = false;
    }

    return mesh;
}

std::shared_ptr<Object3D> ObjectFactory::createLight(LightKind kind, const Object3D& root) {

    std::shared_ptr<Object3D> light;

    switch (kind) {
        case LightKind::Directional: {
            auto l = DirectionalLight::create(0xffffff, 2.f);
            l->position.set(5, 10, 7.5f);
            l->castShadow = true;
            light = l;
            break;
        }
        case LightKind::Point: {
            auto l = PointLight::create(0xffffff, 10.f, 0.f, 2.f);
            l->position.set(0, 3, 0);
            l->castShadow = true;
            light = l;
            break;
        }
        case LightKind::Spot: {
            auto l = SpotLight::create(0xffffff, 20.f, 0.f, math::degToRad(35.f), 0.4f, 2.f);
            l->position.set(0, 5, 0);
            l->castShadow = true;
            light = l;
            break;
        }
        case LightKind::Ambient:
            light = AmbientLight::create(0xffffff, 0.4f);
            break;
        case LightKind::Hemisphere:
            light = HemisphereLight::create(0xbfd4ff, 0x6b5a44, 1.f);
            break;
    }

    light->name = uniqueName(root, label(kind));
    return light;
}

std::shared_ptr<Group> ObjectFactory::createGroup(const Object3D& root) {

    auto group = Group::create();
    group->name = uniqueName(root, "Group");
    return group;
}

std::shared_ptr<PerspectiveCamera> ObjectFactory::createCamera(const Object3D& root) {

    auto camera = PerspectiveCamera::create(50, 16.f / 9.f, 0.1f, 1000.f);
    camera->name = uniqueName(root, "Camera");
    camera->position.set(0, 2, 5);
    return camera;
}

std::shared_ptr<Group> ObjectFactory::createSpline(const Object3D& root) {

    auto spline = Group::create();
    spline->name = uniqueName(root, "Spline");
    SplineConfig{}.write(*spline);

    // An arc rather than a straight line: four collinear points draw something
    // indistinguishable from a segment, which tells the user nothing about what
    // they just added. Lifted clear of the ground plane for the same reason.
    static constexpr float positions[][3] = {
            {-3.f, 0.5f, 1.5f},
            {-1.f, 0.5f, -1.f},
            {1.f, 0.5f, -1.f},
            {3.f, 0.5f, 1.5f}};

    for (const auto& position : positions) {
        auto point = createSplinePoint(*spline);
        point->position.set(position[0], position[1], position[2]);
        spline->add(point);
    }

    return spline;
}

std::shared_ptr<Group> ObjectFactory::createConveyor(const Object3D& root) {

    auto conveyor = Group::create();
    conveyor->name = uniqueName(root, "Conveyor");
    ConveyorConfig{}.write(*conveyor);

    // A straight run at working height: the legs the frame generates need
    // ground under them to reach, and a conveyor on the floor tells the user
    // nothing about what they just added.
    static constexpr float positions[][3] = {
            {-1.5f, 0.75f, 0.f},
            {0.f, 0.75f, 0.f},
            {1.5f, 0.75f, 0.f}};

    for (const auto& position : positions) {
        auto point = createConveyorPoint(*conveyor);
        point->position.set(position[0], position[1], position[2]);
        conveyor->add(point);
    }

    return conveyor;
}

std::shared_ptr<Object3D> ObjectFactory::createConveyorPoint(const Object3D& conveyor) {

    auto point = Object3D::create();
    // Unique within the conveyor, like spline points: read against siblings.
    point->name = uniqueName(conveyor, "Waypoint");
    return point;
}

std::shared_ptr<Group> ObjectFactory::createConveyorWall(const Object3D& conveyor) {

    auto wall = Group::create();
    wall->name = uniqueName(conveyor, "Wall");
    ConveyorWallConfig{}.write(*wall);

    // The default is a PASSIVE guide following the OUTER edge of the belt for
    // most of its run — the wall every conveyor wants before any diverts. The
    // user then decides the rest: drag an end point along the belt to set the
    // length, drag any point toward the middle and that stretch sweeps inward
    // into a diverter (the built wall follows the path between its points —
    // see conveyor::followWall).
    namespace cv = threepp::conveyor;
    const auto config = ConveyorConfig::read(conveyor).value_or(ConveyorConfig{});
    const auto spec = config.spec(conveyor);
    const auto sampled = cv::resamplePath(spec.waypoints, spec.smooth, spec.samples);

    Vector3 first(-0.5f, 0.f, 0.35f), second(0.5f, 0.f, 0.35f);
    if (sampled.size() >= 2) {
        float total = 0.f;
        for (std::size_t i = 0; i + 1 < sampled.size(); ++i) {
            total += sampled[i].distanceTo(sampled[i + 1]);
        }

        // Which side is "outer": against the net turn of the path (cargo runs
        // wide on a bend, so that is where a guide earns its keep). A straight
        // path has no outer side; either will do.
        float netTurn = 0.f;
        for (std::size_t i = 1; i + 1 < spec.waypoints.size(); ++i) {
            const auto fillet = cv::cornerFillet(spec.waypoints, i);
            if (fillet.valid) netTurn += fillet.sweep;
        }
        const float side = netTurn > 1e-3f ? -1.f : 1.f;
        const float offset = std::max(config.width, 0.1f) * 0.5f + 0.03f;

        const Vector3 up(0, 1, 0);
        const auto edgeAt = [&](float s) {
            const Vector3 at = cv::pointAlong(sampled, s);
            const Vector3 ahead = cv::pointAlong(sampled, std::min(s + 0.2f, total));
            Vector3 tangent;
            tangent.subVectors(ahead, at);
            if (tangent.length() < 1e-5f) tangent.set(1, 0, 0);
            tangent.normalize();
            Vector3 lat;
            if (std::abs(tangent.dot(up)) > 0.999f) lat.set(0, 0, 1);
            else lat.crossVectors(tangent, up).normalize();
            Vector3 edge = at;
            edge.addScaledVector(lat, side * offset);
            return edge;
        };

        first = edgeAt(total * 0.06f);
        second = edgeAt(total * 0.94f);
    }

    auto p1 = createConveyorWallPoint(*wall);
    p1->position.copy(first);
    wall->add(p1);
    auto p2 = createConveyorWallPoint(*wall);
    p2->position.copy(second);
    wall->add(p2);

    return wall;
}

std::shared_ptr<Object3D> ObjectFactory::createConveyorWallPoint(const Object3D& wall) {

    auto point = Object3D::create();
    point->name = uniqueName(wall, "Wall Point");
    return point;
}

std::shared_ptr<Object3D> ObjectFactory::createSplinePoint(const Object3D& spline) {

    auto point = Object3D::create();
    // Unique within the spline, not within the scene: point names are read
    // against their siblings, and two splines both starting at "Point" is
    // clearer than one of them starting at "Point 5".
    point->name = uniqueName(spline, "Point");
    return point;
}
