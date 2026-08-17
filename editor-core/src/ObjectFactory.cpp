
#include "threepp/extras/editor/ObjectFactory.hpp"

#include "threepp/extras/editor/ConveyorConfig.hpp"
#include "threepp/extras/editor/FlockConfig.hpp"
#include "threepp/extras/editor/GranularConfig.hpp"
#include "threepp/extras/editor/JointConfig.hpp"
#include "threepp/extras/editor/ParticleFieldConfig.hpp"
#include "threepp/extras/editor/SoundConfig.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"
#include "threepp/extras/editor/TerrainConfig.hpp"
#include "threepp/extras/editor/TextConfig.hpp"
#include "threepp/extras/editor/TreeConfig.hpp"

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
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <random>
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

std::shared_ptr<Mesh> ObjectFactory::createText(const Object3D& root) {

    const TextConfig config;

    auto material = defaultMaterial();
    // Flat text (depth 0) is a sheet of one-sided triangles; from behind it
    // would simply not be there. Solid text keeps the setting harmlessly.
    material->side = Side::Double;
    // The shadow pass flips Front to Back to keep a surface out of its own
    // shadow map, but Double stays Double — so without this, the glyph caps
    // self-shadow into speckle. Back is what the flip would have picked.
    material->shadowSide = Side::Back;

    auto mesh = Mesh::create(config.buildGeometry(), material);
    config.write(*mesh);
    mesh->name = uniqueName(root, "Text");
    mesh->castShadow = true;
    mesh->receiveShadow = true;

    // Stood on the ground like the primitives. The geometry is centred, so
    // half its height is what is below the baseline of position.y = 0.
    mesh->geometry()->computeBoundingBox();
    const auto& box = mesh->geometry()->boundingBox;
    if (box && !box->isEmpty()) mesh->position.y = -box->min().y;

    return mesh;
}

std::shared_ptr<Mesh> ObjectFactory::createTerrain(const Object3D& root) {

    auto config = TerrainConfig::makeDefault();
    // A fresh seed per terrain, for createTree's reason: a second Add Terrain
    // is a different landscape, not the same hill twice. The seed lives in the
    // config, so the terrain stays deterministic through undo, save and reload.
    config.params.seed = std::random_device{}();

    const auto bake = config.bake();

    auto material = defaultMaterial();
    // The splat albedo carries the colour, so the tint has to be neutral or the
    // bands double-darken. Rough and non-metal: this is ground.
    material->color = Color::white;
    material->roughness = 0.93f;
    material->metalness = 0.f;

    auto mesh = Mesh::create(bake.geometry, material);
    mesh->name = uniqueName(root, "Terrain");
    // Ground receives; a heightfield casting into itself is all acne and no
    // information at editor shadow-map resolutions.
    mesh->castShadow = false;
    mesh->receiveShadow = true;
    config.write(*mesh);
    TerrainConfig::applyAlbedo(*mesh, bake.albedo, bake.dim);

    // The field runs 0..amplitude in local Y, so dropped in at the origin the
    // whole canvas would float above it and every primitive the Add menu makes
    // — all of which are lifted to stand ON y = 0 — would sit half buried.
    // Sink it by half the relief and the average ground IS the floor.
    mesh->position.y = -config.params.amplitude * 0.5f;

    return mesh;
}

std::shared_ptr<Group> ObjectFactory::createTree(const Object3D& root) {

    auto tree = Group::create();
    tree->name = uniqueName(root, "Tree");

    TreeConfig config;
    // Oak: the preset that most obviously reads as "a tree" the moment it
    // appears, and the one the vegetation demo opens on.
    vegetation::applyPreset(0, config.params);
    // A fresh seed per tree, so a second Add Tree is a DIFFERENT oak and not a
    // clone standing in the same wood. The seed lives in the config, so the
    // tree itself stays deterministic through undo, save and reload — only the
    // choice of individual is random, and only once.
    config.params.seed = std::random_device{}();
    config.write(*tree);

    // Built here rather than left to the sync pass, so the tree is a tree the
    // frame it appears — and so undo of the Add carries the meshes away with
    // it, AddObjectCommand owning the whole subtree.
    const auto geometries = config.build();

    // Named against their siblings rather than the scene, like spline control
    // points: every tree having a "Trunk" reads better than one of them
    // starting at "Trunk 7".
    auto trunk = Mesh::create(geometries.trunk, config.makeBarkMaterial());
    trunk->name = TreeConfig::label(TreeConfig::Part::Trunk);
    trunk->castShadow = true;
    trunk->receiveShadow = true;
    TreeConfig::markDerived(*trunk, TreeConfig::Part::Trunk);
    tree->add(trunk);

    auto leaves = Mesh::create(geometries.leaves, config.makeLeafMaterial());
    leaves->name = TreeConfig::label(TreeConfig::Part::Leaves);
    leaves->castShadow = true;
    leaves->receiveShadow = true;
    TreeConfig::markDerived(*leaves, TreeConfig::Part::Leaves);
    tree->add(leaves);

    return tree;
}

std::shared_ptr<Object3D> ObjectFactory::createSound(const Object3D& root) {

    // A plain Object3D: a sound has no geometry, and what makes it visible in
    // the viewport is the speaker marker keyed off the entry below.
    auto node = Object3D::create();
    SoundConfig{}.write(*node);
    node->name = uniqueName(root, "Sound");
    // Off the floor, where an emitter usually lives and where the marker is
    // not buried in the ground plane.
    node->position.y = 1.f;

    return node;
}

std::shared_ptr<Object3D> ObjectFactory::createJoint(const Object3D& root) {

    // A plain Object3D: the joint frame is the node's own transform, and what
    // makes it visible in the viewport is the hinge marker keyed off the entry
    // below. Left at the local origin — the natural anchor guess is "where the
    // parent body is", and the gizmo takes it from there.
    auto node = Object3D::create();
    JointConfig{}.write(*node);
    node->name = uniqueName(root, "Joint");

    return node;
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

    // The default is ONE SHORT segment on the edge at the START of the belt —
    // a piece, not a plan, sitting where building naturally begins. Building
    // a real wall is incremental from there: slide the segment along the belt
    // with the gizmo (the built wall follows the path between its points —
    // see conveyor::followWall — so it rides the edge wherever it is
    // dragged), then grow it downstream point by point with Insert After;
    // any point pulled toward the middle sweeps that stretch inward into a
    // diverter.
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

        // Placement TILES the line: walls come in SECTIONS with open stretches
        // between them (an opening in a guide is where a diverter feeds cargo
        // out), so each new wall continues after the furthest existing one on
        // the SAME side, a gap downstream. When that side runs out of belt,
        // the next wall starts over on the OTHER side — repeated Add Wall
        // guards one edge section by section, then the other. With no walls
        // yet: the start of the belt, on the outer side of the net turn
        // (where cargo runs wide on a bend).
        float furthestEnd = -1.f;
        float furthestSide = 0.f;
        for (const auto* existing : ConveyorConfig::wallNodes(conveyor)) {
            const auto points = ConveyorWallConfig::pointNodes(*existing);
            if (points.empty()) continue;
            Matrix4 local;
            local.compose(existing->position, existing->quaternion, existing->scale);
            for (const auto* endpoint : {points.front(), points.back()}) {
                Vector3 p = endpoint->position;
                p.applyMatrix4(local);
                const auto projected = cv::projectOntoPath(p, sampled);
                if (projected.station > furthestEnd) {
                    furthestEnd = projected.station;
                    furthestSide = projected.offset >= 0.f ? 1.f : -1.f;
                }
            }
        }

        float netTurn = 0.f;
        for (std::size_t i = 1; i + 1 < spec.waypoints.size(); ++i) {
            const auto fillet = cv::cornerFillet(spec.waypoints, i);
            if (fillet.valid) netTurn += fillet.sweep;
        }
        const float outer = netTurn > 1e-3f ? -1.f : 1.f;

        const float offset = std::max(config.width, 0.1f) * 0.5f + 0.03f;
        const float length = std::clamp(std::max(config.width, 0.1f) * 1.5f, 0.4f,
                                        total * 0.4f);
        const float margin = std::min(0.15f, total * 0.05f);
        constexpr float kGap = 0.5f;// the open stretch between tiled sections

        float side = outer;
        float start = margin;
        if (furthestEnd >= 0.f) {
            side = furthestSide != 0.f ? furthestSide : outer;
            start = furthestEnd + kGap;
            if (start + length > total) {
                // That side is built out — begin the other one.
                side = -side;
                start = margin;
            }
        }

        first = cv::pointOnPath(sampled, start, side * offset);
        second = cv::pointOnPath(sampled, start + length, side * offset);
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

std::shared_ptr<Group> ObjectFactory::createParticleField(const Object3D& root) {

    auto field = Group::create();
    field->name = uniqueName(root, "Particles");

    const auto config = ParticleFieldConfig::snow();
    config.write(*field);

    // The node IS the emitter frame, and the preset's spawn slab is a thin
    // ceiling the flakes fall from — at the floor it would pour the whole
    // column through the ground. Lifted to the top of one lifetime of fall.
    field->position.y = config.lifetime * -config.velocity.y;

    return field;
}

std::shared_ptr<Group> ObjectFactory::createGranular(const Object3D& root) {

    auto granular = Group::create();
    granular->name = uniqueName(root, "Granular");
    GranularConfig{}.write(*granular);

    // A chute pours DOWN from where it stands, so it needs to stand somewhere:
    // at the origin the grains spawn inside the floor.
    granular->position.y = 2.f;

    return granular;
}

std::shared_ptr<Group> ObjectFactory::createFlock(const Object3D& root) {

    auto flock = Group::create();
    flock->name = uniqueName(root, "Flock");
    FlockConfig{}.write(*flock);

    // The node's position is the territory's home — a loiter volume, not a
    // floor marker. At the origin the birds would orbit through the ground.
    // Exactly the default cruiseAltitude, so the helper's ground tick lands
    // on y=0 — a fresh flock over a ground-at-origin scene reads as placed
    // right, and a moved one shows its expectation.
    flock->position.y = 14.f;

    return flock;
}

std::shared_ptr<Object3D> ObjectFactory::createSplinePoint(const Object3D& spline) {

    auto point = Object3D::create();
    // Unique within the spline, not within the scene: point names are read
    // against their siblings, and two splines both starting at "Point" is
    // clearer than one of them starting at "Point 5".
    point->name = uniqueName(spline, "Point");
    return point;
}
