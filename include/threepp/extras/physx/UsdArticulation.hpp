// Import a USD robot into a PhysX Articulation you can simulate — the USD counterpart of
// UrdfArticulation.hpp, and needing a build with THREEPP_WITH_USD.
//
//     PhysxWorld world(...);
//     auto robot = loadUsdArticulation(world, "crx10ia_l.usd");
//     for (auto& m : robot.meshes) scene->add(m);   // render the link bodies (bound to the sim)
//     robot.articulation->setDriveTargets(q.data(), q.size());
//     world.step(dt);
//
// The file decides everything it authors, so the options default to "what the file says": whether
// the base is fixed (a fixed joint to the world), drive gains and effort limits, joint limits, mass,
// centre of mass and principal inertia, and one convex hull per collider mesh. A file's "PhysX"
// variant is selected, which is where SimReady keeps solver iterations, self-collision and joint speed
// limits. Revolute and prismatic joints become DOFs; a fixed joint stays a link with its own mass. A
// link with no authored mass takes it from its colliders. A PhysxMimicJointAPI coupling becomes a PhysX
// mimic joint, so a gripper's follower fingers close with the finger its drive moves. What the
// description could not carry is in `diagnostics`: spherical/D6 joints, kinematic loops, a decomposition
// collider cooked as one hull. Stage-level metersPerUnit and kilogramsPerUnit are not applied.
#ifndef THREEPP_PHYSX_USDARTICULATION_HPP
#define THREEPP_PHYSX_USDARTICULATION_HPP

#include "threepp/extras/physx/Articulation.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/loaders/USDLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace threepp {

    struct USDArticulationOptions {
        // Unset means what the file says.
        std::optional<bool> fixedBase;
        std::optional<bool> selfCollision;       // the file's, else off
        std::optional<int> solverPositionIterations;// the file's, else 12
        std::optional<float> stiffness;          // override every drive, in N·m/rad or N/m
        std::optional<float> damping;
        std::optional<float> maxForce;

        Vector3 basePosition{0.f, 0.f, 0.f};// moves the whole robot from where the file puts it
        Quaternion baseRotation{};
        float defaultDensity = 1000.f;     // for links with neither mass nor density authored
        bool driveLimitsAreForces = true;  // USD's maxForce is a force, so enforce it as one (see Articulation)
        bool renderVisuals = true;         // parent each link's visual prims under its bound mesh
    };

    struct USDArticulationResult {
        std::unique_ptr<Articulation> articulation;
        std::vector<std::shared_ptr<Mesh>> meshes;// one per link, bound to the sim (add them to a scene)
        std::vector<std::string> jointNames;      // actuated joints, parents first (== drive-target order)
        // Index-aligned with jointNames: the link each joint moves.
        std::vector<ArticulationLink> links;
        // Every link's prim path and its leaf name -> its articulation link.
        std::vector<std::pair<std::string, ArticulationLink>> linkForName;
        // The couplings built, each between the joints moving two of `meshes`' links
        // (by index): q + gearing·qRef + offset = 0, in radians and metres.
        std::vector<USDArticulationDesc::Mimic> mimics;
        // Whatever the file holds outside the robot's bodies. Same world frame as `meshes`.
        std::shared_ptr<Group> scene;

        // `error` is empty exactly when `articulation` is non-null.
        std::string error;
        std::vector<std::string> diagnostics;

        [[nodiscard]] std::size_t numDof() const { return jointNames.size(); }
    };

    // Import `path` as a reduced-coordinate Articulation in `world`, added to it and in the pose the file
    // authors. Each joint's drive starts on the file's targetPosition.
    inline USDArticulationResult loadUsdArticulation(PhysxWorld& world, const std::filesystem::path& path,
                                                     const USDArticulationOptions& opts = {}) {
        using namespace ::physx;
        USDArticulationResult result;

        USDLoader loader;
        loader.setVariant("PhysX", "Enabled");
        USDResult usd = loader.loadFull(path);
        if (!usd.scene) {
            result.error = "could not load " + path.string();
            return result;
        }
        if (!usd.articulation) {
            result.error = "no UsdPhysics joints between two bodies in " + path.string();
            return result;
        }
        USDArticulationDesc& desc = *usd.articulation;
        result.diagnostics = desc.diagnostics;
        if (desc.links.empty()) {
            result.error = "no articulation root in " + path.string();
            return result;
        }

        auto art = std::make_unique<Articulation>(world, opts.fixedBase.value_or(desc.fixedBase),
                                                  opts.solverPositionIterations.value_or(desc.solverPositionIterations.value_or(12)),
                                                  !opts.selfCollision.value_or(desc.selfCollision.value_or(false)),
                                                  opts.driveLimitsAreForces);

        Matrix4 base;
        base.compose(opts.basePosition, opts.baseRotation, Vector3(1.f, 1.f, 1.f));

        const auto materialOf = [&](const USDArticulationDesc::Collider& c) -> PxMaterial* {
            return c.hasMaterial ? world.physics().createMaterial(c.staticFriction, c.dynamicFriction, c.restitution) : nullptr;
        };

        std::vector<ArticulationLink> handles;
        handles.reserve(desc.links.size());
        std::vector<float> authored;

        for (std::size_t i = 0; i < desc.links.size(); ++i) {
            auto& L = desc.links[i];

            auto hidden = MeshStandardMaterial::create();
            hidden->visible = false;
            std::shared_ptr<Mesh> mesh;
            if (L.colliders.empty()) {
                mesh = Mesh::create(SphereGeometry::create(0.02f), hidden);
            } else {
                auto geometry = BufferGeometry::create();
                geometry->setAttribute("position", FloatBufferAttribute::create(L.colliders.front().points, 3));
                mesh = Mesh::create(geometry, hidden);
            }
            Matrix4 world0;
            world0.compose(L.position, L.rotation, Vector3(1.f, 1.f, 1.f));
            world0.premultiply(base);
            Vector3 position, scale;
            Quaternion rotation;
            world0.decompose(position, rotation, scale);
            mesh->position.copy(position);
            mesh->quaternion.copy(rotation);

            Vector3 anchor = L.jointAnchor;
            anchor.applyMatrix4(base);
            Vector3 axis = L.jointAxis;
            axis.applyQuaternion(opts.baseRotation);

            const float density = L.density > 0.f ? L.density : opts.defaultDensity;
            const bool moving = L.parent >= 0 && L.jointType != Robot::JointType::Fixed;
            const bool prismatic = L.jointType == Robot::JointType::Prismatic;
            const auto& drive = L.drive;
            const float stiffness = opts.stiffness.value_or(drive ? drive->stiffness : 0.f);
            const float damping = opts.damping.value_or(drive ? drive->damping : 0.f);
            float maxForce = opts.maxForce.value_or(drive ? drive->maxForce : std::numeric_limits<float>::infinity());
            if (!std::isfinite(maxForce)) maxForce = std::numeric_limits<float>::max();

            PxMaterial* firstMaterial = L.colliders.empty() ? nullptr : materialOf(L.colliders.front());
            ArticulationLink link = art->addLink(
                    L.parent < 0 ? nullptr : &handles[L.parent], *mesh, density,
                    {axis.x, axis.y, axis.z}, {anchor.x, anchor.y, anchor.z},
                    moving && L.range.has_value(), L.range ? L.range->min : 0.f, L.range ? L.range->max : 0.f,
                    stiffness, damping, maxForce, drive ? drive->target : 0.f,
                    !moving ? "fixed" : prismatic ? "prismatic" : "revolute", L.jointFriction, firstMaterial,
                    drive && drive->acceleration ? PxArticulationDriveType::eACCELERATION : PxArticulationDriveType::eFORCE);
            if (firstMaterial) firstMaterial->release();// the shape keeps its own reference

            for (std::size_t c = 1; c < L.colliders.size(); ++c) {
                PxMaterial* material = materialOf(L.colliders[c]);
                art->addConvexHull(link, L.colliders[c].points.data(), L.colliders[c].points.size() / 3, material);
                if (material) material->release();
            }
            if (L.colliders.empty()) {
                // addLink needs a shape to build on; this one is not a collider.
                PxShape* proxy = nullptr;
                link.raw()->getShapes(&proxy, 1);
                proxy->setFlag(PxShapeFlag::eSIMULATION_SHAPE, false);
                proxy->setFlag(PxShapeFlag::eSCENE_QUERY_SHAPE, false);
            }

            // Mass in the order USD resolves it: whatever is authored wins, and the
            // colliders supply the rest.
            if (L.colliders.size() > 1) link.setDensityFromShapes(density);
            if (L.mass > 0.f) link.setMassFromShapes(L.mass);
            if (L.centerOfMass || L.diagonalInertia) {
                link.setMassProperties(link.raw()->getMass(),
                                       L.centerOfMass.value_or(fromPxVec3(link.raw()->getCMassLocalPose().p)),
                                       L.principalAxes,
                                       L.diagonalInertia.value_or(fromPxVec3(link.raw()->getMassSpaceInertiaTensor())));
            }

            if (moving) {
                if (L.maxJointVelocity) link.setMaxJointVelocity(*L.maxJointVelocity);
                if (L.armature) link.setArmature(*L.armature);
                result.jointNames.push_back(L.jointName);
                result.links.push_back(link);
                authored.push_back(L.jointValue);
            }

            if (opts.renderVisuals) {
                L.body->position.set(0.f, 0.f, 0.f);
                L.body->quaternion.identity();
                L.body->scale.copy(L.scale);
                mesh->add(L.body);
            }
            result.meshes.push_back(mesh);
            handles.push_back(link);

            art->nameLink(L.path, i);
            art->nameLink(L.path.substr(L.path.rfind('/') + 1), i);
            result.linkForName.emplace_back(L.path, link);
            result.linkForName.emplace_back(L.path.substr(L.path.rfind('/') + 1), link);
        }

        // Couplings go on before the articulation enters the scene: PhysX makes no
        // mimic joint on one that has.
        for (const auto& m : desc.mimics) {
            art->addMimicJoint(handles[static_cast<std::size_t>(m.link)], handles[static_cast<std::size_t>(m.referenceLink)],
                               m.gearing, m.offset, m.naturalFrequency, m.dampingRatio);
        }
        result.mimics = desc.mimics;

        art->finalize();

        // The tree's joints have served; each child now hangs from its bound mesh.
        for (const auto& L : desc.links) {
            if (L.jointNode && L.jointNode->parent) L.jointNode->parent->remove(*L.jointNode);
        }
        usd.robot->remove(*usd.scene);
        usd.scene->applyMatrix4(base);
        result.scene = usd.scene;

        // setJointPositions takes PhysX's cache order, which a branched tree does not share.
        const auto dofOrder = art->dofOrder();
        std::vector<float> positions(authored.size());
        for (std::size_t i = 0; i < authored.size(); ++i) positions[static_cast<std::size_t>(dofOrder[i])] = authored[i];
        art->setJointPositions(positions.data(), positions.size());
        for (std::size_t i = 0; i < result.meshes.size(); ++i) {
            result.meshes[i]->position.copy(handles[i].position());
            result.meshes[i]->quaternion.copy(handles[i].quaternion());
        }

        result.articulation = std::move(art);
        return result;
    }

}// namespace threepp

#endif// THREEPP_PHYSX_USDARTICULATION_HPP
