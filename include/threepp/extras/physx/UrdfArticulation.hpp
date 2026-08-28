// Import a URDF into a PhysX Articulation you can simulate — the C++ counterpart of the threepp.urdf
// Python helper, now sharing ONE parser (URDFLoader, xacro-capable) instead of a second XML reader.
//
//     PhysxWorld world(...);
//     auto robot = loadArticulation(world, "arm.urdf", {.fixedBase = true, .stiffness = 200, .damping = 20});
//     for (auto& m : robot.meshes) scene->add(m);   // render the link bodies (bound to the sim)
//     robot.articulation->setDriveTargets(q.data(), q.size());
//     world.step(dt);
//
// Builds the kinematic tree by forward kinematics, creating a reduced-coordinate Articulation: collision
// is box/sphere/cylinder(->capsule) directly and <mesh> by ONE convex hull of that mesh; mass comes from
// <inertial> (else default_density x shape volume); revolute/prismatic joints become DOFs (with limits +
// an optional PD drive); fixed joints are collapsed into their parent (the child's collision is dropped,
// as in the Python helper). An approximation (primitive/convex-hull collision), not a digital twin — but
// it turns "hand-build the robot" into one call, from C++ as well as Python.
#ifndef THREEPP_PHYSX_URDFARTICULATION_HPP
#define THREEPP_PHYSX_URDFARTICULATION_HPP

#include "threepp/extras/physx/Articulation.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CapsuleGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace threepp {

    struct URDFArticulationOptions {
        bool fixedBase = false;             // pin the root to the world (arms); leave false for free bodies
        Vector3 basePosition{0.f, 0.f, 0.f};// where the root link is placed
        Quaternion baseRotation{};          // orient the whole robot (e.g. URDF Z-up -> a Y-up world)
        float defaultDensity = 1000.f;      // used for links without an <inertial><mass>
        float stiffness = 0.f;              // PD drive on every actuated joint (0 stiffness = passive/force-controlled)
        float damping = 0.f;
        float maxForce = 1e6f;
        bool selfCollision = false;
        int solverPositionIterations = 12;
        bool renderVisuals = true;          // parent each link's <visual> under its collider so it renders as real meshes
        // Uniform length scale for the whole robot — a URDF drawn in millimetres
        // in a metre world is 0.001. Folded into the description before a single
        // actor exists (see scaleArticulationDesc), because a PhysX actor has no
        // scale of its own: shapes, joint frames and prismatic limits are built at
        // the scaled size instead. Masses stay as authored; inertia is derived
        // from the scaled shapes, so it follows. NOTE that a prismatic DOF then
        // solves in SCALED units, which is what a caller mirroring joint values
        // onto an unscaled kinematic model has to undo.
        float scale = 1.f;
        // xacro argument overrides, exactly as URDFLoader::setArgs takes them — the same
        // `name:=value` pairs the xacro CLI accepts.
        //
        // A parameterised description is not optional detail: UR's ur.urdf.xacro derives its
        // joint-limit, kinematics and visual yaml paths from $(arg ur_type), whose default
        // ("ur5x") names a config directory that does not exist. Building the articulation
        // without the caller's arguments therefore does not produce a slightly different robot,
        // it produces no robot at all — while the KINEMATIC path (EditorApp::rearticulateRobots,
        // which does pass RobotConfig::argMap()) renders the right one. Two rebuild paths
        // disagreeing about what one document says is the bug this exists to prevent.
        std::map<std::string, std::string> args;
    };

    struct URDFArticulationResult {
        std::unique_ptr<Articulation> articulation;
        std::vector<std::shared_ptr<Mesh>> meshes;// collider meshes, bound to the sim (add them to a scene)
        std::vector<std::string> jointNames;      // actuated joints, in add order (== drive-target order)
        // One handle per actuated joint, index-aligned with jointNames — the link
        // whose INBOUND joint that name refers to. A joint sensor (JointEncoder,
        // ForceTorqueSensor) takes an ArticulationLink; this is how a caller that
        // only knows a URDF joint name reaches the link to measure without
        // rebuilding the add-order bookkeeping the loader already did. These are
        // small copyable handles into the articulation; valid while it lives.
        std::vector<ArticulationLink> links;
        // Every URDF LINK name -> the articulation link that governs it, root and
        // fixed-collapsed links included (a fixed child maps to the link it was
        // welded into). What a caller needs to resolve "the body under this
        // visual node" — e.g. the editor associating a robot's link nodes with
        // their actors so an IMU or contact sensor authored on one resolves.
        std::vector<std::pair<std::string, ArticulationLink>> linkForName;

        // Why this went the way it did, straight from the URDFLoader that read
        // the file (see URDFLoader::diagnostics / lastError). The loader is
        // built INSIDE loadArticulation and dies with it, so without copying
        // these out a caller has no way to reach them at all - and the failure
        // path returns an empty result, which on its own says only "no".
        //
        // `error` is empty exactly when `articulation` is non-null.
        // `diagnostics` can be non-empty either way: a URDF that builds fine
        // still warns about a mesh it could not find.
        std::string error;
        std::vector<std::string> diagnostics;

        [[nodiscard]] std::size_t numDof() const { return jointNames.size(); }
    };

    namespace physx_detail {

        // Build the collider mesh for a parsed collision primitive; returns (mesh, volume). Volume converts
        // a target <inertial> mass to the density PhysX's updateMassAndInertia wants.
        inline std::pair<std::shared_ptr<Mesh>, float> makeColliderMesh(const URDFArticulationDesc::Collision& c) {
            using Shape = URDFArticulationDesc::Collision::Shape;
            auto mat = MeshStandardMaterial::create();
            std::shared_ptr<Mesh> mesh;
            float vol = 0.f;
            switch (c.shape) {
                case Shape::Box: {
                    const float x = 2.f * c.halfExtents.x, y = 2.f * c.halfExtents.y, z = 2.f * c.halfExtents.z;
                    mesh = Mesh::create(BoxGeometry::create(x, y, z), mat);
                    vol = std::abs(x * y * z);
                    break;
                }
                case Shape::Sphere: {
                    mesh = Mesh::create(SphereGeometry::create(c.radius), mat);
                    vol = 4.f / 3.f * math::PI * c.radius * c.radius * c.radius;
                    break;
                }
                case Shape::Capsule: {
                    const float len = 2.f * c.halfHeight;
                    mesh = Mesh::create(CapsuleGeometry::create(c.radius, len), mat);
                    vol = math::PI * c.radius * c.radius * len + 4.f / 3.f * math::PI * c.radius * c.radius * c.radius;
                    break;
                }
                case Shape::Hull: {
                    // A <mesh> collision: wrap the collected hull points in a
                    // geometry the link builder cooks into a convex shape. The
                    // material is hidden — like the None proxy, only the visual
                    // subtree should render. Volume is approximated from the
                    // point AABB (it only sets the density fallback, and the
                    // link usually carries an <inertial><mass> anyway).
                    auto geometry = BufferGeometry::create();
                    geometry->setAttribute("position",
                                           FloatBufferAttribute::create(c.hullPoints, 3));
                    geometry->computeBoundingBox();
                    mesh = Mesh::create(geometry, mat);
                    mat->visible = false;
                    if (geometry->boundingBox) {
                        Vector3 size;
                        geometry->boundingBox->getSize(size);
                        vol = std::max(std::abs(size.x * size.y * size.z), 1e-6f);
                    } else {
                        vol = 1e-6f;
                    }
                    break;
                }
                default: {// None -> a tiny invisible proxy so a frame-only link still becomes a body
                    mesh = Mesh::create(SphereGeometry::create(0.02f), mat);
                    mat->visible = false;
                    vol = 4.f / 3.f * math::PI * 0.02f * 0.02f * 0.02f;
                    break;
                }
            }
            return {mesh, vol};
        }

    }// namespace physx_detail

    // Import `path` (URDF or xacro) as a reduced-coordinate Articulation in `world`. Built at the zero joint
    // configuration; the articulation is NOT finalized for you only if the file is unreadable (result is empty).
    inline URDFArticulationResult loadArticulation(PhysxWorld& world, const std::filesystem::path& path,
                                                   const URDFArticulationOptions& opts = {}) {
        URDFArticulationResult result;
        URDFLoader loader;
        // The caller's xacro arguments, before anything is parsed — without them a parameterised
        // description expands to the file's own defaults, which is a different robot (or none).
        if (!opts.args.empty()) loader.setArgs(opts.args);
        // only load each link's <visual> mesh from disk when we will actually render it — otherwise this
        // dominates a large batch build (~0.45 s/env for a detailed arm that never renders in training).
        URDFArticulationDesc desc = loader.parseArticulation(path, opts.renderVisuals);
        // Copied out before anything else can go wrong, and on both paths: the
        // loader is a local and takes its account of the file with it.
        result.diagnostics = loader.diagnostics();
        if (desc.links.empty()) {
            result.error = loader.lastError();
            if (result.error.empty()) {
                // parseArticulation can come back empty without the XML being
                // at fault - a document that parses cleanly but has no single
                // root link. Saying so beats an empty string.
                result.error = "no single root link in " + path.string();
            }
            return result;// unreadable / no single root
        }

        // Units, before anything is built from the description.
        scaleArticulationDesc(desc, opts.scale);

        auto art = std::make_unique<Articulation>(world, opts.fixedBase, opts.solverPositionIterations, !opts.selfCollision);

        const std::size_t n = desc.links.size();
        std::vector<Matrix4> worldT(n);                  // each link's world transform at the zero config
        std::vector<ArticulationLink*> artLinkOf(n, nullptr);// the articulation link a child attaches to
        std::vector<ArticulationLink> linkStore;         // stable storage; reserve so &back() stays valid
        linkStore.reserve(n);

        Matrix4 baseT;
        baseT.compose(opts.basePosition, opts.baseRotation, Vector3(1.f, 1.f, 1.f));

        // Links whose mass comes from the density fallback rather than an
        // authored <inertial><mass>. Counted for the scale diagnostic below.
        std::size_t densityLinks = 0;

        for (std::size_t i = 0; i < n; ++i) {
            const auto& L = desc.links[i];
            Matrix4 parentWorld = (L.parent < 0) ? baseT : worldT[L.parent];
            worldT[i] = parentWorld.multiply(L.jointOrigin);// jointOrigin is identity for the root

            const bool isRoot = (L.parent < 0);
            if (!isRoot && L.jointType == Robot::JointType::Fixed) {
                artLinkOf[i] = artLinkOf[L.parent];// collapse: weld into parent, children attach to it
                continue;
            }

            auto [mesh, volume] = physx_detail::makeColliderMesh(L.collision);
            Matrix4 meshWorld = worldT[i];
            meshWorld.multiply(L.collision.origin);
            Vector3 mp, ms;
            Quaternion mq;
            meshWorld.decompose(mp, mq, ms);
            mesh->position.copy(mp);
            mesh->quaternion.copy(mq);

            const bool hasAuthoredMass = L.hasMass && volume > 1e-9f;
            if (!hasAuthoredMass) ++densityLinks;
            const float density = hasAuthoredMass ? (L.mass / volume) : opts.defaultDensity;

            ArticulationLink linkResult = [&]() -> ArticulationLink {
                if (isRoot) {
                    return art->addLink(nullptr, *mesh, density, {0.f, 0.f, 1.f}, {0.f, 0.f, 0.f},
                                        false, 0.f, 0.f, 0.f, 0.f, opts.maxForce, 0.f, "revolute", 0.f, nullptr);
                }
                Vector3 posI, sclI;
                Quaternion quatI;
                worldT[i].decompose(posI, quatI, sclI);
                Vector3 axisW = L.jointAxis;
                axisW.applyQuaternion(quatI).normalize();// joint axis -> world frame
                const bool limited = L.range.has_value();
                const float lo = limited ? L.range->min : 0.f;
                const float hi = limited ? L.range->max : 0.f;
                const std::string jt = (L.jointType == Robot::JointType::Prismatic) ? "prismatic" : "revolute";
                return art->addLink(artLinkOf[L.parent], *mesh, density,
                                    {axisW.x, axisW.y, axisW.z}, {posI.x, posI.y, posI.z},
                                    limited, lo, hi, opts.stiffness, opts.damping, opts.maxForce, 0.f, jt, 0.f, nullptr);
            }();
            if (!isRoot) {
                result.jointNames.push_back(L.jointName.empty() ? L.name : L.jointName);
                // Index-aligned with jointNames: a sensor that knows a joint name
                // gets the link to measure without redoing the add-order walk.
                result.links.push_back(linkResult);
            }

            linkStore.push_back(linkResult);
            artLinkOf[i] = &linkStore.back();

            if (opts.renderVisuals && L.visual) {
                // parent the visual under the collider, undoing the collision origin so it lands in the link
                // frame; hide the collider primitive so only the real mesh shows.
                Matrix4 inv = L.collision.origin;
                inv.invert();
                Vector3 hp, hs;
                Quaternion hq;
                inv.decompose(hp, hq, hs);
                auto holder = Group::create();
                holder->position.copy(hp);
                holder->quaternion.copy(hq);
                holder->scale.copy(hs);
                holder->add(L.visual);
                mesh->add(holder);
                if (mesh->material()) mesh->material()->visible = false;
            }
            result.meshes.push_back(mesh);
        }

        // A robot scaled UP whose masses come from the density fallback gets
        // heavier with the cube of the scale while its lever arms grow linearly:
        // the gravity torque on a joint grows with the FOURTH power, against
        // drive gains that are authored absolute numbers. Measured on a two-DOF
        // arm, doubling the scale multiplies the drive sag by 16, and a x4 robot
        // collapses outright. An authored <inertial><mass> is deliberately NOT
        // scaled (a millimetre CAD export authors its masses in kilograms), so
        // there the torque grows only linearly and the drive keeps up. Say which
        // regime this build is in rather than letting the robot sag in silence.
        if (opts.scale > 1.001f && densityLinks > 0) {
            char scaleText[32];
            std::snprintf(scaleText, sizeof(scaleText), "%.3g", static_cast<double>(opts.scale));
            result.diagnostics.push_back(
                    "scaled x" + std::string(scaleText) + " with " + std::to_string(densityLinks) +
                    " link(s) taking mass from density x volume - their gravity load grows with the "
                    "fourth power of the scale. If the drive sags, raise its stiffness, lower the "
                    "density, or author <inertial> masses in the URDF");
        }

        // Every desc link, root and fixed-collapsed alike, mapped to the
        // articulation link that governs it — artLinkOf already welded fixed
        // children into their parents, so this is a read-out, not new
        // bookkeeping.
        for (std::size_t i = 0; i < n; ++i) {
            if (!artLinkOf[i]) continue;
            result.linkForName.emplace_back(desc.links[i].name, *artLinkOf[i]);
            // Register the same mapping ON the articulation, so a caller holding
            // only the Articulation (the Python binding returns nothing else
            // per-link) can resolve a URDF link name to its ArticulationLink.
            // linkStore is pushed once per addLink in call order, so an index
            // into it IS the articulation's add-order link index.
            art->nameLink(desc.links[i].name,
                          static_cast<std::size_t>(artLinkOf[i] - linkStore.data()));
        }

        art->finalize();
        result.articulation = std::move(art);
        return result;
    }

}// namespace threepp

#endif// THREEPP_PHYSX_URDFARTICULATION_HPP
