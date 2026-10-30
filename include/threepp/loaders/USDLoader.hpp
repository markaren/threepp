#pragma once

#include "threepp/loaders/MaterialVariants.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Robot.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace threepp {

    class Group;
    class Object3D;
    class Texture;

    /// What a USD file says about simulating its robot, independent of any
    /// physics engine (extras/physx/UsdArticulation.hpp builds a PhysX
    /// articulation from it). Links are ordered parents first. Poses are the
    /// loaded scene's world frame at joint zero, so a Z-up file reads Y-up here
    /// unless the loader ignores the up direction. Revolute quantities are in
    /// radians, not USD's degrees.
    struct USDArticulationDesc {
        struct Collider {
            std::string path;
            std::string approximation;// the file's physics:approximation
            std::vector<float> points;// x,y,z in the link frame
            // The bound physics material, when the file binds one.
            bool hasMaterial = false;
            float staticFriction = 0.f;
            float dynamicFriction = 0.f;
            float restitution = 0.f;
        };
        struct Drive {
            float stiffness = 0.f;// N·m/rad, or N/m for prismatic
            float damping = 0.f;  // N·m·s/rad, or N·s/m for prismatic
            float maxForce = 0.f; // N·m or N; +inf when the file sets no limit
            float target = 0.f;   // rad or m
            bool acceleration = false;// physics:type "acceleration", else "force"
        };
        /// A PhysxMimicJointAPI coupling: the joint moving `link` follows the
        /// joint moving `referenceLink` so that q + gearing·qRef + offset = 0,
        /// the rule PhysX enforces. In this description's units: USD authors
        /// the coupling with angles in degrees, so a revolute joint's offset is
        /// converted, and so is the gearing of a coupling between a revolute
        /// and a prismatic joint. naturalFrequency and dampingRatio both above
        /// zero make the coupling compliant; either at zero is a hard constraint.
        struct Mimic {
            std::string joint;         // prim path of the coupled joint
            std::string referenceJoint;// prim path of the joint it follows
            int link = -1;             // index into links: what the coupled joint moves
            int referenceLink = -1;    // index into links: what the reference joint moves
            float gearing = 1.f;
            float offset = 0.f;// rad or m
            float naturalFrequency = 0.f;// s^-1
            float dampingRatio = 0.f;
        };
        struct Link {
            std::string path;// prim path of the rigid body
            std::shared_ptr<Object3D> body;

            int parent = -1;// index into links; -1 = root
            std::string jointName;
            std::string jointPath;// prim path of the joint that moves this link
            Robot::JointType jointType = Robot::JointType::Fixed;
            std::shared_ptr<Object3D> jointNode;
            Vector3 jointAnchor;                   // world
            Vector3 jointAxis{1.f, 0.f, 0.f};      // world
            std::optional<Robot::JointRange> range;// radians or metres
            float jointValue = 0.f;                // the pose the file was authored in
            std::optional<Drive> drive;
            std::optional<float> maxJointVelocity;// rad/s or m/s, from physxJoint:*
            std::optional<float> armature;
            float jointFriction = 0.f;

            Vector3 position;     // link frame, world, without scale
            Quaternion rotation;  // link frame, world
            Vector3 scale{1, 1, 1};

            float mass = 0.f;   // 0 = not authored
            float density = 0.f;// 0 = not authored
            std::optional<Vector3> centerOfMass;// link frame
            Quaternion principalAxes;
            std::optional<Vector3> diagonalInertia;

            std::vector<Collider> colliders;
        };
        std::vector<Link> links;
        std::vector<Mimic> mimics;// couplings between two of the links' joints
        bool fixedBase = false;// a joint anchors the root to the world
        // From physxArticulation:*, when the file authors them.
        std::optional<bool> selfCollision;
        std::optional<int> solverPositionIterations;
        // What the description leaves out or approximates.
        std::vector<std::string> diagnostics;
    };

    struct USDResult {
        std::shared_ptr<Group> scene;
        MaterialVariants variants;///< Named material variants (empty in current implementation)
        /// HDR/equirect texture authored by the file's UsdLuxDomeLight, if any.
        /// Assign to `scene.background` and `scene.environment` for IBL. Null
        /// when the file has no DomeLight or its `inputs:texture:file` could
        /// not be resolved.
        std::shared_ptr<Texture> environment;
        /// The articulated robot, when the file has UsdPhysics joints
        /// (revolute, prismatic, fixed). Its joint nodes are spliced into
        /// `scene` between each joint's two bodies, and it holds `scene` as its
        /// only child: add `robot` (not `scene`) to your scene and drive it with
        /// setJointValue(). Joint parent/child names are the bodies' prim paths.
        /// Null when the file has no joints. A mimic coupling between joints is
        /// in `articulation`, not here: `robot` moves each joint on its own.
        std::shared_ptr<Robot> robot;
        /// The robot's mass, colliders and drives, set exactly when `robot` is.
        /// `body` and `jointNode` are the nodes `robot` is built from.
        std::optional<USDArticulationDesc> articulation;
    };

    class USDLoader {
    public:
        USDLoader();
        ~USDLoader();

        /// When true, the file-level upAxis metadata is ignored and the
        /// returned scene is left in its file-native orientation. Use this when
        /// the mesh is being composed into an outer system (URDF/SDF/MJCF) that
        /// owns the coordinate frame.
        USDLoader& setIgnoreUpDirection(bool ignore);

        /// Select `value` for every variant set named `set`, over the file's own
        /// selection. A set the file does not have is ignored.
        USDLoader& setVariant(const std::string& set, const std::string& value);

        /// Load a USD/USDA/USDC/USDZ file. Returns the scene root only.
        std::shared_ptr<Group> load(const std::filesystem::path& path);

        /// Load a USD file and return full result including (future) variant info.
        USDResult loadFull(const std::filesystem::path& path);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp
