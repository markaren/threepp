// Reduced-coordinate articulation (a robot) built on PhysX, wired to the threepp
// scene graph through PhysxWorld. Add links (root first, then children, each with
// an inbound revolute/prismatic joint), then finalize() to add it to the world's
// scene; world.step() drives the bound visual meshes.
//
// This is the library type used by BOTH the Python bindings (bind_physx.cpp wraps
// the numpy/torch hot path on top) and C++ consumers (e.g. the URDF articulation
// loader, the RobotCell demo). State I/O here is plain C++ (std::vector / float*);
// the binding layer adapts it to numpy. CPU per-joint state is NOT valid under the
// direct-GPU path (use PhysxGpuBatch); the cpuOnly() guards enforce that.
#ifndef THREEPP_PHYSX_ARTICULATION_HPP
#define THREEPP_PHYSX_ARTICULATION_HPP

#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CapsuleGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Mesh.hpp"

#include <PxPhysicsAPI.h>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace threepp {

    namespace physx_detail {

        // Shortest-arc quaternion mapping unit vector `from` onto unit vector `to`.
        // Used to orient a revolute joint frame so its X axis (the eTWIST axis) lies
        // along the requested world hinge axis.
        inline ::physx::PxQuat shortestArc(::physx::PxVec3 from, ::physx::PxVec3 to) {
            using namespace ::physx;
            from.normalize();
            to.normalize();
            const float d = from.dot(to);
            if (d >= 1.f - 1e-6f) return PxQuat(PxIdentity);
            if (d <= -1.f + 1e-6f) {
                PxVec3 axis = PxVec3(1, 0, 0).cross(from);
                if (axis.magnitudeSquared() < 1e-6f) axis = PxVec3(0, 1, 0).cross(from);
                axis.normalize();
                return PxQuat(PxPi, axis);
            }
            const PxVec3 c = from.cross(to);
            PxQuat q(c.x, c.y, c.z, 1.f + d);
            q.normalize();
            return q;
        }

        // Box/Sphere/Capsule collider inferred from a mesh geometry (mirrors
        // PhysxWorld's private inferShape). localPose corrects the capsule axis
        // (threepp capsule is Y-aligned; PhysX capsule is X-aligned).
        struct LinkShape {
            ::physx::PxGeometryHolder geom;
            ::physx::PxTransform localPose{::physx::PxIdentity};
            bool valid = true;
        };
        inline LinkShape inferLinkShape(const BufferGeometry& g) {
            using namespace ::physx;
            LinkShape s;
            if (auto* b = dynamic_cast<const BoxGeometry*>(&g)) {
                s.geom = PxBoxGeometry(b->width * 0.5f, b->height * 0.5f, b->depth * 0.5f);
            } else if (auto* sp = dynamic_cast<const SphereGeometry*>(&g)) {
                s.geom = PxSphereGeometry(sp->radius);
            } else if (auto* c = dynamic_cast<const CapsuleGeometry*>(&g)) {
                s.geom = PxCapsuleGeometry(c->radius, c->length * 0.5f);
                s.localPose = PxTransform(PxQuat(-PxHalfPi, PxVec3(0, 0, 1)));
            } else {
                s.valid = false;
            }
            return s;
        }

    }// namespace physx_detail

    // Handle to one articulation link + its inbound joint (null for the root).
    // Valid only while its Articulation (and world) live.
    class ArticulationLink {
    public:
        ArticulationLink(::physx::PxArticulationLink* link,
                         ::physx::PxArticulationJointReducedCoordinate* joint,
                         ::physx::PxTransform creationPose,
                         ::physx::PxArticulationAxis::Enum axis = ::physx::PxArticulationAxis::eTWIST)
            : link_(link), joint_(joint), creationPose_(creationPose), axis_(axis) {}

        [[nodiscard]] bool isRoot() const { return joint_ == nullptr; }
        [[nodiscard]] Vector3 position() const { return fromPxVec3(link_->getGlobalPose().p); }
        [[nodiscard]] Quaternion quaternion() const { return fromPxQuat(link_->getGlobalPose().q); }

        // External force/impulse on this link (a PxArticulationLink is a PxRigidBody).
        // Use for perturbations — e.g. random shoves to train push recovery, or to
        // force-drive a cart link in a cart-pole on the CPU deployment path.
        void addForce(const Vector3& v) { link_->addForce(toPxVec3(v), ::physx::PxForceMode::eFORCE); }
        void addImpulse(const Vector3& v) { link_->addForce(toPxVec3(v), ::physx::PxForceMode::eIMPULSE); }

        // Operate on this joint's actual motion axis (eTWIST for revolute, eX for
        // prismatic) so the accessors are correct for both joint types.
        void setDriveTarget(float t) { joint()->setDriveTarget(axis_, t); }
        void setDriveVelocity(float v) { joint()->setDriveVelocity(axis_, v); }
        [[nodiscard]] float jointPosition() const { return joint()->getJointPosition(axis_); }
        [[nodiscard]] float jointVelocity() const { return joint()->getJointVelocity(axis_); }

        [[nodiscard]] ::physx::PxArticulationLink* raw() const { return link_; }
        [[nodiscard]] ::physx::PxTransform creationPose() const { return creationPose_; }

    private:
        ::physx::PxArticulationJointReducedCoordinate* joint() const {
            if (!joint_) throw std::runtime_error("ArticulationLink: the root link has no joint");
            return joint_;
        }
        ::physx::PxArticulationLink* link_;
        ::physx::PxArticulationJointReducedCoordinate* joint_;
        ::physx::PxTransform creationPose_;
        ::physx::PxArticulationAxis::Enum axis_;
    };

    // Reduced-coordinate articulation builder (a robot). Add links (root first,
    // then children with an inbound joint), then finalize() to add it to the
    // world's scene. world.step() drives the bound visual meshes.
    class Articulation {
    public:
        Articulation(PhysxWorld& world, bool fixedBase, int solverPositionIters, bool disableSelfCollision)
            : world_(world) {
            using namespace ::physx;
            art_ = world_.physics().createArticulationReducedCoordinate();
            if (!art_) throw std::runtime_error("createArticulationReducedCoordinate failed");
            art_->setArticulationFlag(PxArticulationFlag::eFIX_BASE, fixedBase);
            if (disableSelfCollision) art_->setArticulationFlag(PxArticulationFlag::eDISABLE_SELF_COLLISION, true);
            if (solverPositionIters > 0) art_->setSolverIterationCounts(static_cast<PxU32>(solverPositionIters), 1);
        }
        ~Articulation() {
            if (cache_) cache_->release();
            // If it was never added to a scene we still own it; once finalized the
            // scene owns it and releases it on world teardown (the world outlives us).
            if (art_ && !finalized_) art_->release();
        }
        Articulation(const Articulation&) = delete;
        Articulation& operator=(const Articulation&) = delete;

        // CPU state I/O (per-joint getters/setters, the cache) is NOT valid once the world
        // runs in direct-GPU mode — that state isn't synced. Fail loudly instead of silently
        // returning stale data; use PhysxGpuBatch for state I/O under direct_gpu.
        void cpuOnly(const char* what) const {
            if (world_.directGpuEnabled())
                throw std::runtime_error(std::string("Articulation.") + what +
                                         ": not valid under direct_gpu — use PhysxGpuBatch for state I/O");
        }

        // Episode reset: teleport the root to `pos` upright with zero velocity and
        // zero every joint position/velocity (back to the neutral build pose). The
        // bound visuals snap to the new state on the next world.step().
        void reset(const Vector3& pos, const Quaternion& quat = Quaternion()) {
            using namespace ::physx;
            if (!finalized_) throw std::runtime_error("Articulation.reset: finalize() first");
            cpuOnly("reset");
            art_->setRootGlobalPose(PxTransform(toPxVec3(pos), toPxQuat(quat)), false);
            if (!cache_) cache_ = art_->createCache();
            art_->zeroCache(*cache_);
            art_->applyCache(*cache_,
                             PxArticulationCacheFlag::ePOSITION | PxArticulationCacheFlag::eVELOCITY |
                                     PxArticulationCacheFlag::eROOT_VELOCITIES,
                             true);
        }

        ArticulationLink addLink(ArticulationLink* parent, Mesh& mesh, float density,
                                 const std::array<float, 3>& axis, const std::array<float, 3>& anchor,
                                 bool limited, float lower, float upper,
                                 float stiffness, float damping, float maxForce, float driveTarget,
                                 const std::string& jointType, float jointFriction,
                                 ::physx::PxMaterial* material) {
            using namespace ::physx;
            if (finalized_) throw std::runtime_error("Articulation.add_link: already finalized (no links after finalize)");
            auto* g = mesh.geometry().get();
            if (!g) throw std::runtime_error("Articulation.add_link: mesh has no geometry");
            const auto shape = physx_detail::inferLinkShape(*g);
            if (!shape.valid) throw std::runtime_error("Articulation.add_link: unsupported geometry (use Box/Sphere/Capsule)");

            mesh.updateMatrixWorld();
            Vector3 pos, scl;
            Quaternion rot;
            mesh.matrixWorld->decompose(pos, rot, scl);
            const PxTransform linkPose(toPxVec3(pos), toPxQuat(rot));

            PxArticulationLink* parentLink = parent ? parent->raw() : nullptr;
            PxArticulationLink* link = art_->createLink(parentLink, linkPose);
            if (!link) throw std::runtime_error("Articulation.add_link: createLink failed");
            if (!parentLink) rootLink_ = link;// the root link (for batched root_state)

            PxShape* s = world_.physics().createShape(shape.geom.any(),
                                                      material ? *material : world_.defaultMaterial(), true);
            s->setLocalPose(shape.localPose);
            link->attachShape(*s);
            s->release();
            PxRigidBodyExt::updateMassAndInertia(*link, density);

            // Revolute (hinge, rotation about the axis) or prismatic (slider,
            // translation along the axis). For revolute the motion DOF is eTWIST
            // (rotation about joint-frame X); for prismatic it's eX (translation
            // along joint-frame X) — same frame, different DOF.
            const bool prismatic = (jointType == "prismatic");
            const PxArticulationAxis::Enum ax = prismatic ? PxArticulationAxis::eX : PxArticulationAxis::eTWIST;

            PxArticulationJointReducedCoordinate* joint = nullptr;
            if (parentLink) {
                joint = link->getInboundJoint();
                joint->setJointType(prismatic ? PxArticulationJointType::ePRISMATIC
                                              : PxArticulationJointType::eREVOLUTE);
                // Joint frame in world space: origin at the anchor, X axis along the
                // hinge/slide axis (eTWIST rotates about X / eX translates along X).
                const PxTransform jointWorld(toPxVec3(Vector3(anchor[0], anchor[1], anchor[2])),
                                             physx_detail::shortestArc(PxVec3(1, 0, 0), toPxVec3(Vector3(axis[0], axis[1], axis[2]))));
                joint->setParentPose(parent->creationPose().getInverse() * jointWorld);
                joint->setChildPose(linkPose.getInverse() * jointWorld);
                joint->setMotion(ax, limited ? PxArticulationMotion::eLIMITED : PxArticulationMotion::eFREE);
                // Joint friction holds a joint static until the applied torque exceeds it —
                // which silently fakes balancing tasks (a near-upright pole's tiny gravity
                // torque never overcomes the default friction, so it never falls). Default to
                // frictionless so "free" joints are genuinely free.
                joint->setFrictionCoefficient(jointFriction);
                joint->setArmature(ax, 0.0f);
                if (limited) joint->setLimitParams(ax, PxArticulationLimit(lower, upper));
                if (stiffness > 0.f || damping > 0.f) {
                    joint->setDriveParams(ax,
                                          PxArticulationDrive(stiffness, damping, maxForce, PxArticulationDriveType::eFORCE));
                    joint->setDriveTarget(ax, driveTarget, false);// autowake=false: pre-scene
                }
                joints_.push_back(joint);
                jointAxes_.push_back(ax);// remember the motion axis so bulk I/O uses eX for prismatic
            }
            // A PxArticulationLink is a PxRigidActor, so the rigid-body bind path
            // syncs the visual mesh to the simulated link pose.
            world_.bind(mesh, *link);
            return ArticulationLink(link, joint, linkPose, ax);
        }

        void finalize() {
            if (finalized_) return;
            world_.scene().addArticulation(*art_);
            finalized_ = true;
        }

        // Set all joint positions (DOF order) and zero velocities, via the cache.
        // Use to place an articulation in a chosen configuration — e.g. start a
        // cart-pole hanging straight down for a swing-up demo.
        void setJointPositions(const float* pos, std::size_t count) {
            using namespace ::physx;
            if (!finalized_) throw std::runtime_error("Articulation.set_joint_positions: finalize() first");
            cpuOnly("set_joint_positions");
            if (!cache_) cache_ = art_->createCache();
            art_->zeroCache(*cache_);
            const PxU32 dof = art_->getDofs();
            const PxU32 n = std::min<PxU32>(dof, static_cast<PxU32>(count));
            for (PxU32 i = 0; i < n; ++i) cache_->jointPosition[i] = pos[i];
            art_->applyCache(*cache_, PxArticulationCacheFlag::ePOSITION | PxArticulationCacheFlag::eVELOCITY);
        }

        // Underlying PhysX articulation — used to assemble a PhysxGpuBatch over many
        // identical robots for GPU-resident vectorized stepping.
        [[nodiscard]] ::physx::PxArticulationReducedCoordinate* rawArt() const { return art_; }
        [[nodiscard]] bool finalized() const { return finalized_; }
        [[nodiscard]] std::size_t numDof() const { return joints_.size(); }

        // Batched joint I/O — one call reads/writes every revolute joint (in
        // add_link order). This is the hot path for vectorized RL: in the binding
        // it collapses ~36 pybind calls per robot per step to 3.
        [[nodiscard]] std::vector<float> jointPositions() const {
            cpuOnly("joint_positions");
            std::vector<float> a(joints_.size());
            for (std::size_t i = 0; i < joints_.size(); ++i)
                a[i] = joints_[i]->getJointPosition(jointAxes_[i]);// eX for prismatic, eTWIST for revolute
            return a;
        }
        [[nodiscard]] std::vector<float> jointVelocities() const {
            cpuOnly("joint_velocities");
            std::vector<float> a(joints_.size());
            for (std::size_t i = 0; i < joints_.size(); ++i)
                a[i] = joints_[i]->getJointVelocity(jointAxes_[i]);
            return a;
        }
        void setDriveTargets(const float* targets, std::size_t count) {
            using namespace ::physx;
            cpuOnly("set_drive_targets");
            const std::size_t n = std::min<std::size_t>(joints_.size(), count);
            // autowake=true on the last write: a policy that drives a settled robot
            // (e.g. after a reset/settle) must wake it, or the targets are ignored and
            // the articulation stays frozen at its rest pose. Use each joint's own axis —
            // eTWIST hardcoded here froze every PRISMATIC joint (e.g. parallel-jaw fingers).
            for (std::size_t i = 0; i < n; ++i)
                joints_[i]->setDriveTarget(jointAxes_[i], targets[i], i + 1 == n);
        }

        // For each joint (in add-order), its low-level DOF slot — the index it occupies
        // in the direct-GPU joint buffers (PhysxGpuBatch), which follow PhysX's cache
        // layout, NOT add-order. Use to reconcile a GPU-trained policy (GPU-DOF order)
        // with the CPU getters (add-order):
        // obs_gpu[dof_order[i]] = cpu_value[i];  cpu_target[i] = gpu_target[dof_order[i]].
        [[nodiscard]] std::vector<int> dofOrder() const {
            using namespace ::physx;
            const std::size_t n = joints_.size();
            // GPU/cache DOF order follows ascending low-level link index, one DOF per
            // revolute link. So a joint's GPU slot = the rank of its child link's index.
            std::vector<PxU32> linkIdx(n);
            for (std::size_t i = 0; i < n; ++i)
                linkIdx[i] = joints_[i]->getChildArticulationLink().getLinkIndex();
            std::vector<int> out(n);
            for (std::size_t i = 0; i < n; ++i) {
                int rank = 0;
                for (std::size_t j = 0; j < n; ++j)
                    if (linkIdx[j] < linkIdx[i]) ++rank;
                out[i] = rank;
            }
            return out;
        }

        // Root link world-frame spatial velocity [vx,vy,vz, wx,wy,wz].
        [[nodiscard]] std::array<float, 6> rootVelocity() const {
            cpuOnly("root_velocity");
            std::array<float, 6> p{};
            if (rootLink_) {
                const ::physx::PxVec3 lv = rootLink_->getLinearVelocity();
                const ::physx::PxVec3 av = rootLink_->getAngularVelocity();
                p = {lv.x, lv.y, lv.z, av.x, av.y, av.z};
            }
            return p;
        }

        // Root link world pose [px,py,pz, qx,qy,qz,qw].
        [[nodiscard]] std::array<float, 7> rootState() const {
            using namespace ::physx;
            cpuOnly("root_state");
            const PxTransform t = rootLink_ ? rootLink_->getGlobalPose() : PxTransform(PxIdentity);
            return {t.p.x, t.p.y, t.p.z, t.q.x, t.q.y, t.q.z, t.q.w};
        }

    private:
        PhysxWorld& world_;
        ::physx::PxArticulationReducedCoordinate* art_ = nullptr;
        ::physx::PxArticulationCache* cache_ = nullptr;
        ::physx::PxArticulationLink* rootLink_ = nullptr;
        std::vector<::physx::PxArticulationJointReducedCoordinate*> joints_;// non-root joints, add order
        std::vector<::physx::PxArticulationAxis::Enum> jointAxes_;// each joint's motion axis (eTWIST rev / eX prism)
        bool finalized_ = false;
    };

}// namespace threepp

#endif// THREEPP_PHYSX_ARTICULATION_HPP
