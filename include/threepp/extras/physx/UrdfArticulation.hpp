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
// is box/sphere/cylinder(->capsule) directly and <mesh> by its bounding box; mass comes from <inertial>
// (else default_density x shape volume); revolute/prismatic joints become DOFs (with limits + an optional
// PD drive); fixed joints are collapsed into their parent (the child's collision is dropped, as in the
// Python helper). An approximation (primitive/bbox collision), not a digital twin — but it turns
// "hand-build the robot" into one call, from C++ as well as Python.
#ifndef THREEPP_PHYSX_URDFARTICULATION_HPP
#define THREEPP_PHYSX_URDFARTICULATION_HPP

#include "threepp/extras/physx/Articulation.hpp"
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

#include <filesystem>
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
    };

    struct URDFArticulationResult {
        std::unique_ptr<Articulation> articulation;
        std::vector<std::shared_ptr<Mesh>> meshes;// collider meshes, bound to the sim (add them to a scene)
        std::vector<std::string> jointNames;      // actuated joints, in add order (== drive-target order)

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
        const URDFArticulationDesc desc = loader.parseArticulation(path);
        if (desc.links.empty()) return result;// unreadable / no single root

        auto art = std::make_unique<Articulation>(world, opts.fixedBase, opts.solverPositionIterations, !opts.selfCollision);

        const std::size_t n = desc.links.size();
        std::vector<Matrix4> worldT(n);                  // each link's world transform at the zero config
        std::vector<ArticulationLink*> artLinkOf(n, nullptr);// the articulation link a child attaches to
        std::vector<ArticulationLink> linkStore;         // stable storage; reserve so &back() stays valid
        linkStore.reserve(n);

        Matrix4 baseT;
        baseT.compose(opts.basePosition, opts.baseRotation, Vector3(1.f, 1.f, 1.f));

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

            const float density = (L.hasMass && volume > 1e-9f) ? (L.mass / volume) : opts.defaultDensity;

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
            if (!isRoot) result.jointNames.push_back(L.jointName.empty() ? L.name : L.jointName);

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

        art->finalize();
        result.articulation = std::move(art);
        return result;
    }

}// namespace threepp

#endif// THREEPP_PHYSX_URDFARTICULATION_HPP
