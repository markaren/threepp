// The PlaySession that ships with the editor: PhysX rigid bodies driven by the
// per-object PhysicsConfig stored in userData.
//
// Header-only and PhysX-dependent, exactly like PhysxWorld itself — the threepp
// library proper never links PhysX, so this file is included only by builds that
// found the SDK (the editor sets THREEPP_EDITOR_WITH_PHYSX).
//
// What it does on start(): walk the scene, and for every object carrying an
// enabled PhysicsConfig create one actor of the requested body type and shape,
// then let PhysxWorld drive that object's transform. On stop() the world is
// destroyed; the editor then restores the pre-play snapshot, so nothing the
// simulation did to the scene survives.
//
// Soft bodies (Body::Soft) are deformable volumes, and PhysX simulates those on
// CUDA only. The world therefore switches to GPU dynamics as soon as the scene
// contains one — and falls back to a CPU world with the rigid bodies alone when
// no CUDA device is available, rather than failing the whole play attempt. A
// soft body rewrites its mesh's vertex positions every step, which the snapshot
// undoes on Stop exactly like a moved transform.

#ifndef THREEPP_EDITOR_PHYSICSPLAYSESSION_HPP
#define THREEPP_EDITOR_PHYSICSPLAYSESSION_HPP

#include "threepp/extras/editor/ArticulationConfig.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PlaySession.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"


#include "threepp/extras/physx/PhysxSoftBody.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/physx/UrdfArticulation.hpp"

// V-HACD convex decomposition, gated exactly like the physics play session
// itself: it is linked (and the macro defined) only for a build that found both
// the PhysX SDK and the v-hacd header. Without it, Shape::Pieces and the
// decomposed-Group paths fall back to a single hull with one log line.
#ifdef THREEPP_EDITOR_WITH_VHACD
#include "threepp/extras/physx/ConvexDecomposition.hpp"
#endif

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Clock.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CapsuleGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp::editor {

    class PhysicsPlaySession: public PlaySession {

    public:
        explicit PhysicsPlaySession(PhysxWorld::Settings settings = {})
            : settings_(settings) {
            // Leave smoothTimestep at its default (false): the EMA renders the
            // sim on a fictional clock and shows up as judder.
            settings_.smoothTimestep = false;
        }

        [[nodiscard]] std::string name() const override { return "Physics"; }

        // Bodies created on the last start(). Useful for a status readout.
        [[nodiscard]] std::size_t bodyCount() const { return bodyCount_; }

        // Of those, how many are deformable volumes.
        [[nodiscard]] std::size_t softBodyCount() const { return softBodyCount_; }

        // How many geometries this session ran V-HACD on since start(). A cached
        // decomposition (a second object sharing a geometry uuid and parameters)
        // does NOT bump it — which is exactly what a test asserting the cache
        // works reads. Reset by start().
        [[nodiscard]] std::size_t decompositionCookCount() const { return decompCookCount_; }

        // False when the scene asked for soft bodies but the machine could not
        // give us a CUDA context; the rigid half of the scene still ran.
        [[nodiscard]] bool gpuAvailable() const { return gpu_; }

        [[nodiscard]] PhysxWorld* world() { return world_.get(); }

        // The session currently playing, or nullptr.
        //
        // A script reaches its object's body through a free function
        // (threepp.editor.rigid_body_from_object), which has no context to
        // resolve against — the script session and this one are independent
        // PlaySessions that do not know about each other. The editor runs one
        // Play at a time, so "the session that is playing" is well defined and
        // is the seam. Set by start(), cleared by stop().
        [[nodiscard]] static PhysicsPlaySession* active() { return active_; }

        // The actor this session created for `object`, or for the nearest
        // ancestor of it that has one.
        //
        // NOT the same as PhysxWorld::findActor, which only knows the bodies it
        // was asked to bind() — and a static body is never bound, because there
        // is no pose to write back into the scene graph. A script asking about
        // the ground would get nothing. This map is the session's own record of
        // every actor it built, so the answer covers static, kinematic and
        // dynamic alike.
        [[nodiscard]] ::physx::PxRigidActor* findActor(const Object3D* object) const {

            for (const Object3D* o = object; o != nullptr; o = o->parent) {
                if (const auto it = actors_.find(o); it != actors_.end()) return it->second;
            }
            return nullptr;
        }

        // Every actor governing `object` — findActor, but plural.
        //
        // findActor answers with ONE actor because a handle needs one. An
        // EXCLUSION needs them all: a subtree collider is a dozen cooked meshes
        // under the single node the user authored, and ignoring only the first
        // of them in a scene query would ignore almost nothing. Walks up the
        // ancestry exactly as findActor does, then gathers every actor recorded
        // against the node it lands on.
        //
        // Empty for an articulated robot: its links belong to the articulation,
        // not to this registry (see findObject).
        [[nodiscard]] std::vector<const ::physx::PxRigidActor*> findActors(const Object3D* object) const {

            std::vector<const ::physx::PxRigidActor*> found;
            const Object3D* owner = nullptr;
            for (const Object3D* o = object; o != nullptr; o = o->parent) {
                if (actors_.find(o) != actors_.end()) {
                    owner = o;
                    break;
                }
            }
            if (!owner) return found;
            for (const auto& [actor, node] : objects_) {
                if (node == owner) found.push_back(actor);
            }
            return found;
        }

        // The object an actor belongs to — findActor read backwards.
        //
        // For anything handed an actor by PhysX rather than by the scene graph: a
        // contact report names two actors, a scene query names the one it hit,
        // and the script that has to be told what it touched wants the OBJECT.
        // The answer is the node the PhysicsConfig was authored on, i.e. exactly
        // the node findActor resolves TO, so the two directions agree by
        // construction. A subtree collider (one authored static body, many cooked
        // meshes) answers as its root for every actor it produced — a contact
        // with any of them is a contact with the one thing the user put physics
        // on.
        [[nodiscard]] Object3D* findObject(const ::physx::PxRigidActor* actor) const {

            if (!actor) return nullptr;
            if (const auto it = objects_.find(actor); it != objects_.end()) return it->second;

            // An articulation's links were never recorded here — an articulation
            // builds its own actors, and the registry above only holds what
            // createActor made. But the session knows its articulations, and a
            // link carries the articulation it belongs to, so the robot is one
            // pointer comparison away. Answering the ROBOT (not the link) is the
            // same contract findArticulation already keeps: the articulation
            // governs the whole subtree, and the robot is the node the user
            // authored.
            if (const auto* link = actor->is<::physx::PxArticulationLink>()) {
                const auto* owner = &link->getArticulation();
                for (const auto& played : articulations_) {
                    if (played->articulation && played->articulation->rawArt() == owner) {
                        return played->robot;
                    }
                }
            }
            return nullptr;
        }

        // A token that lives exactly as long as the world built by the last
        // start(). Handles handed to scripts keep a weak_ptr to it, so a body
        // still referenced after Stop reports that it is gone instead of
        // dereferencing a released actor.
        [[nodiscard]] std::weak_ptr<const void> lifetime() const { return lifetime_; }

        // One simulated robot: the visual Robot the editor authored, the PhysX
        // articulation driving it, and the joint-name <-> link mapping the joint
        // sensors resolve against. `links` and `jointNames` are index-aligned (the
        // articulation's add order); the visual robot's own joint order can differ,
        // which is exactly why linkFor() is keyed by NAME, not by index.
        struct PlayedArticulation {
            Robot* robot = nullptr;
            Articulation* articulation = nullptr;
            std::vector<std::string> jointNames;
            std::vector<ArticulationLink> links;

            [[nodiscard]] const ArticulationLink* linkFor(const std::string& jointName) const {
                for (std::size_t i = 0; i < jointNames.size() && i < links.size(); ++i) {
                    if (jointNames[i] == jointName) return &links[i];
                }
                return nullptr;
            }

            // Session-owned bookkeeping (not part of the read contract above).
            // owned: the articulation is destroyed with the session, before the
            // world it belongs to (see stop()). meshes: proxy colliders kept alive
            // but never added to the scene. visualIndexOfDof: DOF add-order ->
            // visual robot joint index, so update() mirrors through the name map.
            std::unique_ptr<Articulation> owned;
            std::vector<std::shared_ptr<Mesh>> meshes;
            std::vector<int> visualIndexOfDof;

            // The robot's uniform world scale, folded into the articulation at
            // build time. It makes the two sides of the mirror speak different
            // units for a PRISMATIC DOF: the articulation solves in SCENE units
            // (its shapes and joint frames were scaled), while Robot::setJointValue
            // slides along the joint axis in the URDF's OWN units, which the node's
            // scale then multiplies on the way to the screen. So a prismatic value
            // is divided by this coming back and multiplied by it going in. A
            // revolute DOF is an angle and rides through untouched.
            float lengthScale = 1.f;
            std::vector<char> prismaticDof;// DOF add-order, like visualIndexOfDof

            [[nodiscard]] bool isPrismatic(std::size_t dof) const {
                return dof < prismaticDof.size() && prismaticDof[dof] != 0;
            }
        };

        // The articulation governing `object`, or the nearest ancestor of it that
        // is a simulated robot — the same walk-up-parents contract as findActor,
        // so a sensor authored on a link resolves to the robot that owns it.
        [[nodiscard]] const PlayedArticulation* findArticulation(const Object3D* object) const {

            for (const Object3D* o = object; o != nullptr; o = o->parent) {
                for (const auto& played : articulations_) {
                    if (played->robot == o) return played.get();
                }
            }
            return nullptr;
        }

        // How many robots this session is simulating, for the status readout.
        [[nodiscard]] std::size_t articulationCount() const { return articulations_.size(); }

        // A session destroyed without a stop() — the editor tearing down mid-Play
        // — must not leave active() pointing at freed memory.
        ~PhysicsPlaySession() override {

            if (active_ == this) active_ = nullptr;
        }

        // Anything worth telling the user about a body that could not be
        // created — same hook the script session uses, so the editor routes
        // both into its log.
        void setLogger(std::function<void(const std::string&)> logger) {

            logger_ = std::move(logger);
        }

        void start(Scene& scene) override {

            bodyCount_ = 0;
            softBodyCount_ = 0;
            decompCookCount_ = 0;
            actors_.clear();
            objects_.clear();
            articulations_.clear();
            decompCache_.clear();

            // Collect first, create second: creating an actor binds the object,
            // and PhysxWorld writes transforms during step(), not during
            // traversal — but a stable list also keeps behaviour independent of
            // any graph edit a later hook might make.
            scene.updateMatrixWorld(true);

            // The robots to simulate: a Robot carrying both a urdf reference and an
            // articulation opt-in. Found first, because a link of one must NOT also
            // pick up a rigid body from its own PhysicsConfig — the articulation
            // link is the body for that node.
            std::vector<Robot*> articulatedRobots;
            scene.traverse([&](Object3D& object) {
                if (auto* robot = object.as<Robot>()) {
                    const auto robotConfig = RobotConfig::read(object);
                    const auto artConfig = ArticulationConfig::read(object);
                    if (robotConfig && !robotConfig->urdf.empty() && artConfig && artConfig->enabled) {
                        articulatedRobots.push_back(robot);
                    }
                }
            });

            std::vector<Object3D*> targets;
            bool wantsSoftBodies = false;
            scene.traverse([&](Object3D& object) {
                if (const auto config = PhysicsConfig::read(object); config && config->enabled) {
                    // A node inside an articulated robot's subtree is already a
                    // simulated link; a second rigid body over it would double the
                    // dynamics at that spot.
                    if (insideArticulatedRobot(object, articulatedRobots)) return;
                    targets.push_back(&object);
                    if (config->body == PhysicsConfig::Body::Soft) wantsSoftBodies = true;
                }
            });

            createWorld(wantsSoftBodies);
            lifetime_ = std::make_shared<const char>('\0');
            active_ = this;

            // Articulations before rigid bodies: they add themselves to the same
            // scene and share the world's material/timestep, and building one can
            // log — do it while the log context is the robot, not a stray box.
            for (auto* robot : articulatedRobots) {
                buildArticulation(*robot);
            }

            for (auto* object : targets) {
                const auto config = PhysicsConfig::read(*object);
                if (!config) continue;
                if (createActor(*object, *config)) ++bodyCount_;
            }
        }

        void update(float dt) override {

            if (!world_) return;
            world_->step(dt);
            // Mirror each articulation's solved state back onto the visual robot,
            // AFTER the step, so the inspector and any sensor read the pose the
            // frame ended on. The play snapshot undoes all of it on Stop.
            for (const auto& played : articulations_) mirrorArticulation(*played);
        }

        void stop() override {

            // Drop the token BEFORE the world: any handle a script is still
            // holding must read as dead from the moment the actors go.
            lifetime_.reset();
            if (active_ == this) active_ = nullptr;
            actors_.clear();
            objects_.clear();
            // Destroy the articulations while the world is still alive: an
            // Articulation releases itself back into the scene it belongs to, so
            // it must go before the world it was added to. (The sensor session,
            // which stops after us, already dropped its FT caches — it stops
            // world-touching the instant our lifetime token expires below.)
            articulations_.clear();
            world_.reset();
            bodyCount_ = 0;
            softBodyCount_ = 0;
        }

    private:
        void log(const std::string& message) {

            if (logger_) logger_(message);
        }

        // Is `object` this robot, or a descendant of one of the articulated
        // robots we are about to simulate? Walks up parents so a link mesh deep in
        // the URDF subtree is caught.
        static bool insideArticulatedRobot(const Object3D& object, const std::vector<Robot*>& robots) {

            for (const Object3D* o = &object; o != nullptr; o = o->parent) {
                for (const auto* robot : robots) {
                    if (o == robot) return true;
                }
            }
            return false;
        }

        // Build one reduced-coordinate articulation for a visual robot and wire it
        // to that robot so update() can mirror the solved joints back.
        //
        // Frame convention (verified against a test): both URDFLoader::load (the
        // visual Robot) and loadArticulation's forward kinematics consume URDF's
        // native frame with NO Z-up->Y-up rotation baked in — they share the same
        // originMatrix/parseInfo code. So the two agree at the origin, and the only
        // offset is the robot's own placement in the scene: pass basePosition /
        // baseRotation straight from the robot's decomposed WORLD matrix, no extra
        // rotation. A double-rotation here would poison every downstream reading.
        void buildArticulation(Robot& robot) {

            const auto robotConfig = RobotConfig::read(robot);
            const auto artConfig = ArticulationConfig::read(robot);
            if (!robotConfig || robotConfig->urdf.empty() || !artConfig || !artConfig->enabled) return;

            robot.updateWorldMatrix(true, false);
            Vector3 basePos, baseScale;
            Quaternion baseRot;
            robot.matrixWorld->decompose(basePos, baseRot, baseScale);

            // A PhysX link has no scale of its own, so a scaled robot is built at
            // the scaled size instead: the world scale is folded into the URDF
            // description (shapes, joint frames, prismatic limits) before any
            // actor exists. That is what makes "import a millimetre robot into a
            // metre scene, set 0.001, press Play" work.
            //
            // Uniform only. A sphere, a capsule and a joint frame have no way to
            // take a per-axis scale, and picking an axis would build something
            // that silently disagrees with what renders.
            const float scale = baseScale.x;
            const auto matches = [scale](float v) {
                return std::abs(v - scale) <= 1e-3f * std::max(1.f, std::abs(scale));
            };
            if (!matches(baseScale.y) || !matches(baseScale.z) || !(scale > 0.f)) {
                log("physics: \"" + robot.name + "\" has a non-uniform world scale - "
                    "PhysX articulation links can be scaled but not stretched, so it is not simulated");
                return;
            }

            URDFArticulationOptions opts;
            opts.fixedBase = artConfig->fixedBase;
            opts.basePosition = basePos;
            opts.baseRotation = baseRot;
            opts.defaultDensity = std::max(artConfig->density, 1e-3f);
            opts.stiffness = std::max(artConfig->stiffness, 0.f);
            opts.damping = std::max(artConfig->damping, 0.f);
            opts.maxForce = std::max(artConfig->maxForce, 0.f);
            opts.selfCollision = artConfig->selfCollision;
            opts.solverPositionIterations = std::max(artConfig->iterations, 1);
            opts.scale = scale;
            // The visual robot IS what renders; the articulation's colliders are
            // an invisible physical twin bound to the sim. Never load or add them.
            opts.renderVisuals = false;

            URDFArticulationResult built;
            try {
                built = loadArticulation(*world_, robotConfig->urdf, opts);
            } catch (const std::exception& e) {
                log("physics: \"" + robot.name + "\" could not be articulated - " + e.what());
                return;
            }
            if (!built.articulation) {
                log("physics: \"" + robot.name + "\" could not be articulated - the URDF at \"" +
                    robotConfig->urdf + "\" is unreadable");
                return;
            }

            auto played = std::make_unique<PlayedArticulation>();
            played->robot = &robot;
            played->articulation = built.articulation.get();
            played->jointNames = built.jointNames;
            played->links = built.links;

            // Map the articulation's DOF add-order onto the visual robot's joint
            // order. Never assume they match: the loader walks the URDF tree
            // breadth-first while the Robot's articulated-joint list follows the
            // <joint> declaration order. Both are keyed by URDF joint name, so the
            // name is the bridge.
            const auto info = robot.getArticulatedJointInfo();
            std::vector<int> visualIndexOfDof(built.jointNames.size(), -1);// dof -> visual joint index
            for (std::size_t d = 0; d < built.jointNames.size(); ++d) {
                for (std::size_t v = 0; v < info.size(); ++v) {
                    if (info[v].name == built.jointNames[d]) {
                        visualIndexOfDof[d] = static_cast<int>(v);
                        break;
                    }
                }
                if (visualIndexOfDof[d] < 0) {
                    log("physics: \"" + robot.name + "\" articulation joint \"" + built.jointNames[d] +
                        "\" has no matching visual joint - it will simulate but not drive the mesh");
                }
            }
            // And the reverse: a visual joint the articulation collapsed (a fixed
            // joint, or one the loader dropped) is worth one line too.
            for (const auto& j : info) {
                bool found = false;
                for (const auto& name : built.jointNames) found = found || name == j.name;
                if (!found) {
                    log("physics: \"" + robot.name + "\" joint \"" + j.name +
                        "\" is not an articulation DOF (fixed or collapsed) - not simulated");
                }
            }
            played->visualIndexOfDof = std::move(visualIndexOfDof);
            played->lengthScale = scale;

            // Which DOFs slide rather than turn, in the articulation's add order.
            // Read off the visual robot through the same name map, since that is
            // where the joint types live.
            played->prismaticDof.assign(built.jointNames.size(), 0);
            for (std::size_t d = 0; d < built.jointNames.size(); ++d) {
                const int v = played->visualIndexOfDof[d];
                if (v < 0 || static_cast<std::size_t>(v) >= info.size()) continue;
                played->prismaticDof[d] = info[v].type == Robot::JointType::Prismatic ? 1 : 0;
            }

            // Apply the authored pose to the DOFs, reordered through the name map:
            // the articulation is built at the URDF zero config, but the document's
            // pose is what the user placed the robot in. The document stores joint
            // values in the robot's native units, so a prismatic one crosses into
            // the articulation's scaled units here.
            const auto& joints = robotConfig->joints;
            std::vector<float> dofPose(built.jointNames.size(), 0.f);
            for (std::size_t d = 0; d < dofPose.size(); ++d) {
                const int v = played->visualIndexOfDof[d];
                if (v >= 0 && static_cast<std::size_t>(v) < joints.size()) {
                    dofPose[d] = played->isPrismatic(d) ? joints[v] * scale : joints[v];
                }
            }
            if (!dofPose.empty()) {
                built.articulation->setJointPositions(dofPose.data(), dofPose.size());
                // With a live drive, also target the authored pose so the PD holds
                // it rather than letting gravity pull the arm down from t=0.
                if (opts.stiffness > 0.f) {
                    built.articulation->setDriveTargets(dofPose.data(), dofPose.size());
                }
            }

            // The proxy colliders are bound to the sim and must stay alive for the
            // whole session, but they are NOT added to the scene: PhysxWorld::bind
            // writes their transforms wherever they are, and here they render
            // nowhere on purpose (the visual robot is what you see).
            played->meshes = std::move(built.meshes);

            // A sensor authored on a LINK (an IMU on a base, a contact pad on a
            // gripper) resolves its body by walking up the scene graph — but the
            // visual link nodes are never bound (the joint mirror drives them; a
            // world-pose write-back would fight the kinematic chain), so
            // findActor cannot see them. Associate each URDF link's visual node
            // with its articulation link instead: resolution only, no pose
            // writes. A fixed link welded away by the builder maps to the link
            // it collapsed into — which is where an "imu_link" mount actually
            // rides. The robot node itself maps to the root link, so a base IMU
            // can be authored on the robot without knowing its link names.
            for (const auto& [linkName, artLink] : built.linkForName) {
                if (auto* node = robot.getObjectByName(linkName)) {
                    world_->associate(*node, *artLink.raw());
                }
                if (artLink.isRoot()) world_->associate(robot, *artLink.raw());
            }

            played->owned = std::move(built.articulation);
            articulations_.push_back(std::move(played));
        }

        // Read the solved joint positions and root pose back onto the visual robot.
        void mirrorArticulation(PlayedArticulation& played) {

            if (!played.robot || !played.articulation) return;

            const auto positions = played.articulation->jointPositions();// DOF add-order
            for (std::size_t d = 0; d < positions.size() && d < played.visualIndexOfDof.size(); ++d) {
                const int v = played.visualIndexOfDof[d];
                if (v < 0) continue;
                // Back into the robot's own units (see PlayedArticulation::lengthScale).
                const float q = played.isPrismatic(d) && played.lengthScale > 0.f
                                        ? positions[d] / played.lengthScale
                                        : positions[d];
                played.robot->setJointValue(static_cast<std::size_t>(v), q);
            }

            // A floating base also moves the whole robot: write the solved root
            // world pose into the robot's local frame (through its parent's inverse
            // world matrix), so a walker or a drone actually travels.
            const auto artConfig = ArticulationConfig::read(*played.robot);
            if (artConfig && !artConfig->fixedBase) {
                const auto s = played.articulation->rootState();// px,py,pz, qx,qy,qz,qw
                const Vector3 worldPos(s[0], s[1], s[2]);
                const Quaternion worldRot(s[3], s[4], s[5], s[6]);
                Matrix4 worldMat;
                worldMat.compose(worldPos, worldRot, Vector3(1.f, 1.f, 1.f));
                if (played.robot->parent) {
                    played.robot->parent->updateWorldMatrix(true, false);
                    Matrix4 parentInv(*played.robot->parent->matrixWorld);
                    parentInv.invert();
                    worldMat.premultiply(parentInv);
                }
                Vector3 localPos, localScale;
                Quaternion localRot;
                worldMat.decompose(localPos, localRot, localScale);
                played.robot->position.copy(localPos);
                played.robot->quaternion.copy(localRot);
            }
        }

        // GPU dynamics is switched on only for a scene that actually needs it:
        // it costs a CUDA context (and the PhysX GPU DLLs) on every Play press,
        // and a machine without a CUDA device cannot create one at all. If the
        // GPU world fails to come up, the rigid bodies still get to run.
        void createWorld(bool wantsSoftBodies) {

            auto settings = settings_;
            gpu_ = settings.enableGpuDynamics || wantsSoftBodies;
            settings.enableGpuDynamics = gpu_;

            if (!gpu_) {
                world_ = std::make_unique<PhysxWorld>(settings);
                return;
            }
            try {
                world_ = std::make_unique<PhysxWorld>(settings);
            } catch (const std::exception& e) {
                gpu_ = false;
                settings.enableGpuDynamics = false;
                world_ = std::make_unique<PhysxWorld>(settings);
                log(std::string("physics: soft bodies need a CUDA GPU (") + e.what() +
                    ") - playing the rigid bodies only");
            }
        }

        // World-space pose + scale of an object, as PhysX wants it.
        struct Placement {
            ::physx::PxTransform pose;
            Vector3 scale{1, 1, 1};
        };

        static Placement placementOf(Object3D& object) {

            object.updateWorldMatrix(true, false);
            Vector3 position, scale;
            Quaternion rotation;
            object.matrixWorld->decompose(position, rotation, scale);
            return {toPxTransform(position, rotation), scale};
        }

        // Local-space AABB of the object's own geometry. Falls back to a unit
        // box for objects with no geometry (a Group used as a trigger volume).
        static Box3 localBounds(Object3D& object) {

            if (const auto geometry = object.geometry()) {
                if (!geometry->boundingBox) geometry->computeBoundingBox();
                if (geometry->boundingBox) return *geometry->boundingBox;
            }
            Box3 unit;
            unit.set(Vector3(-0.5f, -0.5f, -0.5f), Vector3(0.5f, 0.5f, 0.5f));
            return unit;
        }

        // Local-space AABB of a geometry-less Group: the union of every
        // descendant mesh's geometry bounds in the ROOT's frame (each sub-mesh's
        // relative transform applied; the root's own world transform excluded —
        // the actor carries that). An explicit Box/Sphere/Capsule authored on an
        // imported model means THAT shape fitted to the model, and sizing it
        // from the unit-box placeholder instead is indistinguishable from
        // ignoring the user's choice. Sub-meshes that own an actor are excluded,
        // the same rule the compound and static-subtree paths follow. Empty
        // (no usable sub-meshes) falls back to the unit box, which is still what
        // a bare Group used as a trigger volume wants.
        static Box3 subtreeLocalBounds(Object3D& root) {

            root.updateWorldMatrix(true, false);
            Matrix4 rootInv(*root.matrixWorld);
            rootInv.invert();

            Box3 bounds;
            bounds.makeEmpty();
            root.traverseType<Mesh>([&](Mesh& mesh) {
                const auto geometry = mesh.geometry();
                if (!geometry) return;
                if (ownsActor(mesh)) return;
                if (!geometry->boundingBox) geometry->computeBoundingBox();
                if (!geometry->boundingBox) return;
                mesh.updateWorldMatrix(true, false);
                Matrix4 rel(rootInv);
                rel.multiply(*mesh.matrixWorld);
                Box3 b = *geometry->boundingBox;
                b.applyMatrix4(rel);
                bounds.union_(b);
            });

            if (bounds.isEmpty()) {
                bounds.set(Vector3(-0.5f, -0.5f, -0.5f), Vector3(0.5f, 0.5f, 0.5f));
            }
            return bounds;
        }

        ::physx::PxMaterial* materialFor(const PhysicsConfig& config) {

            const float friction = std::max(config.friction, 0.f);
            const float restitution = std::clamp(config.restitution, 0.f, 1.f);
            return world_->physics().createMaterial(friction, friction, restitution);
        }

        // Resolve Shape::Auto against the actual geometry: the analytic shapes
        // when the geometry IS one of them, otherwise a triangle mesh for
        // static bodies and a convex hull for moving ones (PhysX dynamics
        // cannot use a triangle mesh).
        //
        // Everything that is not a primitive used to land on Box, i.e. on its
        // own AABB. For anything shaped — a road surface, a terrain patch, an
        // imported prop — that slab is not an approximation of the surface, it
        // is a different object: a flat surface's AABB is a razor at the minimum
        // half-extent, and a body dropped on it falls through beside it.
        static PhysicsConfig::Shape resolveShape(Object3D& object, const PhysicsConfig& config) {

            if (config.shape != PhysicsConfig::Shape::Auto) return config.shape;

            const auto geometry = object.geometry();
            // No geometry to read. Static bodies collide as their subtree (see
            // createActor); anything else keeps the unit-box placeholder.
            if (!geometry) return PhysicsConfig::Shape::Box;

            if (dynamic_cast<const SphereGeometry*>(geometry.get())) return PhysicsConfig::Shape::Sphere;
            if (dynamic_cast<const CapsuleGeometry*>(geometry.get())) return PhysicsConfig::Shape::Capsule;
            if (dynamic_cast<const BoxGeometry*>(geometry.get())) return PhysicsConfig::Shape::Box;

            // Kinematic counts as moving: a triangle mesh would come back as a
            // PxRigidStatic here, which is not something code can drive.
            return config.body == PhysicsConfig::Body::Static ? PhysicsConfig::Shape::TriMesh
                                                              : PhysicsConfig::Shape::Convex;
        }

        // Objects that get an actor of their own from start()'s walk. A subtree
        // collider must not cook them a second time.
        static bool ownsActor(const Object3D& object) {

            const auto config = PhysicsConfig::read(object);
            return config && config->enabled;
        }

        // Cook the descendants of a geometry-less static body as its collider.
        // Meshes carrying their own enabled PhysicsConfig are left alone —
        // start() is creating their actors — so putting physics on a spline
        // collides its generated tube, which is where a user naturally puts it.
        std::size_t addSubtree(Object3D& root, ::physx::PxMaterial* material) {

            const auto actors = world_->addStaticTrimeshTree(
                    root,
                    [&root](const Mesh& mesh) {
                        return &mesh != &root && mesh.geometry() != nullptr && !ownsActor(mesh);
                    },
                    material);
            // A subtree is many actors under one authored config. A script
            // asking the root for "its" body gets the first — enough to answer
            // is_static and where it is, which is all a static collider has.
            if (!actors.empty()) record(root, actors.front());
            // Backwards, though, ALL of them answer as the root: a contact
            // against the fourth cooked mesh of a spline's tube is a contact
            // with the spline, and there is nothing else to call it.
            for (auto* actor : actors) {
                if (actor) objects_.emplace(actor, &root);
            }
            return actors.size();
        }

        // Remember the actor governing `object`, so a script can find it later —
        // and the object governed by that actor, so something handed the actor
        // can name the object. Returns true so the creation sites can
        // `return record(...)`.
        bool record(Object3D& object, ::physx::PxRigidActor* actor) {

            if (actor) {
                actors_.emplace(&object, actor);
                objects_.emplace(actor, &object);
            }
            return actor != nullptr;
        }

        // A deformable volume: PhysX cooks a tetrahedral mesh from the object's
        // own geometry and then writes the deformed vertices back into it every
        // step, so the visual mesh IS the simulation output. addSoftBody bakes
        // the world matrix into the geometry and zeroes the local transform —
        // the play snapshot puts both back on Stop.
        bool createSoftBody(Object3D& object, const PhysicsConfig& config) {

            auto* mesh = object.as<Mesh>();
            if (!mesh) {
                log("physics: \"" + object.name + "\" is not a mesh - a soft body needs geometry to deform");
                return false;
            }
            if (!gpu_) return false;// already reported by createWorld

            const auto geometry = mesh->geometry();
            if (!geometry || !geometry->hasAttribute("position")) {
                log("physics: \"" + object.name + "\" has no geometry to cook a soft body from");
                return false;
            }

            // The tet cooker wants a closed surface. An open one (a plane, a
            // half-open imported shell) cooks into something arbitrary rather
            // than failing, so say so instead of silently simulating nonsense.
            if (!geometry->getIndex()) {
                log("physics: \"" + object.name + "\" has no index buffer - soft-body cooking needs a "
                                                  "connected triangle surface");
                return false;
            }

            auto* material = world_->createSoftBodyMaterial(
                    std::max(config.youngsModulus, 1e3f),
                    std::clamp(config.poissonsRatio, 0.f, 0.49f),
                    std::max(config.friction, 0.f));

            // Cooking is the expensive part, so N copies of one geometry cook
            // once. Only valid for an unscaled object: the cached path re-uses
            // the mesh cooked in local space and applies rotation + translation,
            // but NOT scale, so a scaled instance must cook its own.
            std::string cacheKey;
            const Vector3 scale = placementOf(object).scale;
            const auto isUnit = [](float v) { return std::abs(v - 1.f) < 1e-4f; };
            if (isUnit(scale.x) && isUnit(scale.y) && isUnit(scale.z)) {
                cacheKey = geometry->uuid;
            }

            try {
                const auto* body = world_->addSoftBody(
                        *mesh, material,
                        std::clamp(config.voxelResolution, 2, 64),
                        static_cast<unsigned>(std::clamp(config.solverIterations, 1, 255)),
                        config.selfCollision,
                        cacheKey,
                        std::max(config.mass, 0.f));
                if (!body) return false;
            } catch (const std::exception& e) {
                log("physics: \"" + object.name + "\" failed to cook as a soft body - " + e.what());
                return false;
            }

            ++softBodyCount_;
            return true;
        }

        // --- Compound / decomposed convex colliders --------------------------

        // The sub-meshes of `root` whose geometry should become collider hulls.
        // Skips the root itself (a Group has none anyway) and any descendant that
        // owns its own actor — putting physics on a Mesh nested in an imported
        // model gives that Mesh its own body, and a compound over the parent
        // must not cook it a second time. Same guard as the static subtree path.
        static std::vector<Mesh*> gatherSubMeshes(Object3D& root) {

            std::vector<Mesh*> out;
            root.traverseType<Mesh>([&](Mesh& mesh) {
                if (&mesh == &root) return;
                if (!mesh.geometry()) return;
                if (!mesh.geometry()->hasAttribute("position")) return;
                if (ownsActor(mesh)) return;
                out.push_back(&mesh);
            });
            return out;
        }

        // Decompose a geometry into convex hulls (each a flat x,y,z array in the
        // geometry's own local frame), cached on (uuid, parameters). Returns a
        // const reference into the cache so duplicates share the vectors. A cook
        // is timed and logged; a cache hit is silent and does not bump the cook
        // counter.
        //
        // Without V-HACD compiled in, or on a decomposition failure, falls back
        // to the geometry's raw positions as a single hull — the caller cooks
        // that into one convex mesh, i.e. the old single-hull behaviour, so the
        // body still simulates (just without the concavity).
        const std::vector<std::vector<float>>& decompose(const BufferGeometry& geometry,
                                                         const PhysicsConfig& config) {

            const std::string key = geometry.uuid + "|h=" + std::to_string(std::max(config.hulls, 1)) +
                                    ";v=" + std::to_string(std::max(config.hullVerts, 8)) +
                                    ";r=" + std::to_string(std::max(config.voxels, 10000));
            if (const auto it = decompCache_.find(key); it != decompCache_.end()) return it->second;

            std::vector<std::vector<float>> hulls;
            const auto* posAttr = geometry.getAttribute<float>("position");
            if (!posAttr) {
                // No positions to decompose. Cache an empty result so the caller
                // (appendHullParts / the Pieces-on-Mesh path) produces no shapes
                // and falls back gracefully, rather than dereferencing null here.
                auto [it, _] = decompCache_.emplace(key, std::move(hulls));
                return it->second;
            }
            const auto& positions = posAttr->array();

            // Build a u32 index buffer (the geometry's, or 0..N-1 for unindexed).
            std::vector<::physx::PxU32> indices;
            if (const auto* idxAttr = geometry.getIndex()) {
                const auto& src = idxAttr->array();
                indices.assign(src.begin(), src.end());
            } else {
                indices.resize(posAttr->count());
                for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<::physx::PxU32>(i);
            }

#ifdef THREEPP_EDITOR_WITH_VHACD
            const std::size_t tris = indices.size() / 3;
            // A triangle-count guard: V-HACD's cost scales with the mesh, and a
            // multi-hundred-thousand-triangle import would stall Play for tens of
            // seconds. Warn once and decompose the raw mesh anyway — a coarse
            // result beats an unbounded freeze, and the voxel step already caps
            // the real work.
            if (tris > 200000) {
                log("physics: decomposing a " + std::to_string(tris) +
                    "-triangle mesh - this may take a while");
            }
            ConvexDecompositionParams p;
            p.maxHulls = static_cast<std::uint32_t>(std::max(config.hulls, 1));
            p.maxVertsPerHull = static_cast<std::uint32_t>(std::clamp(config.hullVerts, 8, 64));
            p.voxelResolution = static_cast<std::uint32_t>(std::max(config.voxels, 10000));

            Clock clock;
            clock.start();
            hulls = decomposeConvex(positions.data(), posAttr->count(),
                                    indices.data(), indices.size(), p);
            const float seconds = clock.getElapsedTime();
            if (!hulls.empty()) {
                ++decompCookCount_;
                log("physics: decomposed a mesh into " + std::to_string(hulls.size()) +
                    " convex hull(s) in " + formatSeconds(seconds));
            }
#endif
            if (hulls.empty()) {
                // No V-HACD, or it failed: one hull from the raw positions.
                hulls.emplace_back(positions.begin(), positions.end());
            }

            auto [it, _] = decompCache_.emplace(key, std::move(hulls));
            return it->second;
        }

        // Turn each flat hull-point array into a cooked PxConvexMesh, positioned
        // by `localPose` and scaled by `scale`, and append the parts. Points are
        // in the hull's local frame; scale is applied by the cooked shape (via
        // PxMeshScale) so cached unscaled hulls serve any instance. Cooked meshes
        // are collected so the caller can release them after the actor is built.
        void appendHullParts(const std::vector<std::vector<float>>& hulls,
                             const ::physx::PxTransform& localPose, const Vector3& scale,
                             std::vector<PhysxWorld::ConvexPart>& parts,
                             std::vector<::physx::PxConvexMesh*>& cooked) {

            for (const auto& hull : hulls) {
                if (hull.size() < 12) continue;// < 4 points
                auto* mesh = world_->cookConvexHull(hull.data(), hull.size() / 3,
                                                    static_cast<unsigned>(64));
                if (!mesh) continue;
                cooked.push_back(mesh);
                parts.push_back({mesh, localPose, scale});
            }
        }

        // Build ONE compound convex actor for `root` out of its sub-meshes.
        // `decomposed` picks the shape of each part: false = one hull per
        // sub-mesh (Convex/Auto on a Group), true = V-HACD each sub-mesh into
        // several hulls (Pieces on a Group). `budgetPerMesh`, when decomposed,
        // divides the total hull budget across the sub-meshes so a five-part
        // model does not ask for `hulls` pieces five times over.
        bool buildCompoundFromSubMeshes(Object3D& root, const PhysicsConfig& config,
                                        bool decomposed, ::physx::PxMaterial* material) {

            using namespace ::physx;

            const auto meshes = gatherSubMeshes(root);
            if (meshes.empty()) return false;

            // The actor's world frame: root world position + rotation, NO scale.
            // Root scale (and each sub-mesh's own scale relative to the root) is
            // baked into the per-part hull scale below, so the actor itself is
            // unscaled — which is what PhysX wants (a rigid actor has no scale).
            root.updateWorldMatrix(true, false);
            Vector3 rootPos, rootScl;
            Quaternion rootRot;
            root.matrixWorld->decompose(rootPos, rootRot, rootScl);
            Matrix4 actorFrame;
            actorFrame.compose(rootPos, rootRot, Vector3(1.f, 1.f, 1.f));
            Matrix4 actorFrameInv(actorFrame);
            actorFrameInv.invert();

            // When decomposing, split the hull budget across the sub-meshes. The
            // floor of 1 wins over the budget: every sub-mesh must get at least
            // one hull or it would vanish from the collider (a hole a body falls
            // through), so a budget smaller than the sub-mesh count is rounded UP
            // to one-each rather than dropping parts. The warning fires when the
            // per-mesh share truncated, telling the user to raise the budget.
            PhysicsConfig perMesh = config;
            if (decomposed && meshes.size() > 1) {
                const int per = std::max(1, config.hulls / static_cast<int>(meshes.size()));
                if (per * static_cast<int>(meshes.size()) < config.hulls) {
                    log("physics: \"" + root.name + "\" hull budget " + std::to_string(config.hulls) +
                        " split across " + std::to_string(meshes.size()) + " sub-meshes (" +
                        std::to_string(per) + " each) - raise it to keep more detail");
                }
                perMesh.hulls = per;
            }

            std::vector<PhysxWorld::ConvexPart> parts;
            std::vector<PxConvexMesh*> cooked;
            for (auto* mesh : meshes) {
                mesh->updateWorldMatrix(true, false);
                // The sub-mesh relative to the actor frame: this carries the
                // sub-mesh's placement within the model AND any scale (root's and
                // its own), which we bake into the hull scale.
                Matrix4 rel;
                rel.multiplyMatrices(actorFrameInv, *mesh->matrixWorld);
                Vector3 relPos, relScl;
                Quaternion relRot;
                rel.decompose(relPos, relRot, relScl);
                const PxTransform localPose(toPxVec3(relPos), toPxQuat(relRot));

                if (decomposed) {
                    appendHullParts(decompose(*mesh->geometry(), perMesh), localPose, relScl, parts, cooked);
                } else {
                    // One hull per sub-mesh: the raw positions, cooked to a hull.
                    // gatherSubMeshes already filtered to meshes with a position
                    // attribute, but re-check rather than trust that at a distance.
                    const auto* posAttr = mesh->geometry()->getAttribute<float>("position");
                    if (!posAttr) continue;
                    if (auto* hull = world_->cookConvexHull(posAttr->array().data(), posAttr->count(),
                                                            static_cast<unsigned>(config.hullVerts <= 0 ? 64 : config.hullVerts))) {
                        cooked.push_back(hull);
                        parts.push_back({hull, localPose, relScl});
                    }
                }
            }

            if (parts.empty()) {
                for (auto* m : cooked) m->release();
                return false;
            }

            const bool dynamic = config.body != PhysicsConfig::Body::Static;
            const PxTransform actorPose(toPxVec3(rootPos), toPxQuat(rootRot));
            auto* actor = world_->addCompound(actorPose, parts, dynamic,
                                              1000.f, material);
            // attachShape keeps its own reference to each convex mesh, so our
            // local ones are done once the actor exists.
            for (auto* m : cooked) m->release();
            if (!actor) return false;

            if (dynamic) {
                auto* body = static_cast<PxRigidDynamic*>(actor);
                finishDynamic(*body, config);
                world_->bind(root, *body);
            }
            return record(root, actor);
        }

        // "1.23 s" / "840 ms" — a cook time a human reads at a glance.
        static std::string formatSeconds(float seconds) {
            char buf[32];
            if (seconds < 1.f) std::snprintf(buf, sizeof(buf), "%d ms", static_cast<int>(seconds * 1000.f));
            else std::snprintf(buf, sizeof(buf), "%.2f s", static_cast<double>(seconds));
            return buf;
        }

        bool createActor(Object3D& object, const PhysicsConfig& config) {

            using namespace ::physx;

            if (config.body == PhysicsConfig::Body::Soft) {
                return createSoftBody(object, config);
            }

            const auto placement = placementOf(object);
            const auto shape = resolveShape(object, config);
            const bool moving = config.body != PhysicsConfig::Body::Static;

            auto* material = materialFor(config);

            // TriMesh and Convex go through PhysxWorld's cookers, which need a
            // Mesh WITH geometry (they read it, and throw when it is absent).
            auto* mesh = object.as<Mesh>();
            if (mesh && !mesh->geometry()) mesh = nullptr;

            // A static body with nothing to collide with of its own COLLIDES AS
            // ITS SUBTREE. Physics authored on a spline group is the case this
            // exists for: the spline IS the tube as far as the user is
            // concerned, and the unit-box fallback below turned that into a
            // phantom 1 m cube at the spline's origin.
            if (!moving && !object.geometry() &&
                (config.shape == PhysicsConfig::Shape::Auto ||
                 config.shape == PhysicsConfig::Shape::TriMesh)) {
                if (addSubtree(object, material) > 0) return true;
                // Nothing under it: fall through to the placeholder, which is
                // still what a bare Group used as a trigger volume wants.
            }

            // Convex-pieces (V-HACD) on a Mesh: decompose its own geometry into
            // several hulls, welded into one compound actor. This is the shape a
            // concave prop wants — the mug that holds water, the pipe that is
            // hollow. A Group with Pieces is handled by the compound path below.
            if (config.shape == PhysicsConfig::Shape::Pieces && mesh) {
                mesh->updateWorldMatrix(true, false);
                Vector3 mpos, mscl;
                Quaternion mrot;
                mesh->matrixWorld->decompose(mpos, mrot, mscl);
                std::vector<PhysxWorld::ConvexPart> parts;
                std::vector<PxConvexMesh*> cooked;
                appendHullParts(decompose(*mesh->geometry(), config),
                                PxTransform(PxIdentity), mscl, parts, cooked);
                if (!parts.empty()) {
                    auto* actor = world_->addCompound(
                            PxTransform(toPxVec3(mpos), toPxQuat(mrot)), parts, moving, 1000.f, material);
                    for (auto* m : cooked) m->release();
                    if (actor) {
                        if (moving) {
                            auto* body = static_cast<PxRigidDynamic*>(actor);
                            finishDynamic(*body, config);
                            world_->bind(*mesh, *body);
                        }
                        return record(object, actor);
                    }
                } else {
                    for (auto* m : cooked) m->release();
                }
                // Nothing cooked (empty geometry, or every hull degenerate):
                // fall through to the analytic path below, which gives the body
                // a box of its bounds. Only reachable for genuinely unusable
                // geometry — decompose() otherwise always yields at least the
                // raw-positions single hull.
            }

            // A geometry-less Group gathers its sub-meshes into ONE compound
            // convex actor — the fix for the headline bug, where an imported
            // model (a Group with sub-meshes) fell back to a 1 m unit box.
            //   * Pieces on a Group  -> V-HACD each sub-mesh (many hulls each)
            //   * Convex on a Group  -> one hull per sub-mesh
            //   * Auto on a Group    -> one hull per sub-mesh, for a MOVING body
            //     (a static Group already collided as its trimesh subtree above)
            if (!object.geometry()) {
                const bool pieces = config.shape == PhysicsConfig::Shape::Pieces;
                const bool wantsCompound =
                        pieces ||
                        config.shape == PhysicsConfig::Shape::Convex ||
                        (config.shape == PhysicsConfig::Shape::Auto && moving);
                if (wantsCompound && buildCompoundFromSubMeshes(object, config, pieces, material)) {
                    return true;
                }
                // No usable sub-meshes: fall through to the unit-box placeholder,
                // which is still what a bare Group used as a trigger wants.
            }

            if (shape == PhysicsConfig::Shape::TriMesh && mesh) {
                // Triangle meshes are valid for static/kinematic actors only;
                // a dynamic request silently gets a static collider rather than
                // a PhysX assertion.
                return record(object, world_->addStaticTrimesh(*mesh, material));
            }
            if (shape == PhysicsConfig::Shape::Convex && mesh && moving) {
                auto* body = world_->addDynamicConvex(*mesh, 1000.f, material);
                if (!body) return false;
                finishDynamic(*body, config);
                return record(object, body);
            }
            if (shape == PhysicsConfig::Shape::Convex && mesh) {
                return record(object, world_->addStaticTrimesh(*mesh, material));
            }

            // Analytic shapes: build the geometry from the local bounds so an
            // off-centre mesh gets a correctly offset shape. A geometry-less
            // Group (an imported model with an explicit Box/Sphere/Capsule
            // override) is sized from its SUBTREE — the placeholder unit box is
            // only for a genuinely empty Group.
            const Box3 bounds = object.geometry() ? localBounds(object)
                                                  : subtreeLocalBounds(object);
            const Vector3 size = bounds.getSize();
            const Vector3 centre = bounds.getCenter();
            const Vector3 s = placement.scale;

            const float hx = std::max(std::abs(size.x * s.x) * 0.5f, 1e-4f);
            const float hy = std::max(std::abs(size.y * s.y) * 0.5f, 1e-4f);
            const float hz = std::max(std::abs(size.z * s.z) * 0.5f, 1e-4f);

            PxGeometryHolder geometry;
            PxTransform localPose(PxVec3(centre.x * s.x, centre.y * s.y, centre.z * s.z));

            switch (shape) {
                case PhysicsConfig::Shape::Sphere:
                    geometry = PxSphereGeometry(std::max({hx, hy, hz}));
                    break;
                case PhysicsConfig::Shape::Capsule: {
                    const float radius = std::max(hx, hz);
                    // PhysX capsules are X-aligned, threepp's are Y-aligned.
                    const float halfHeight = std::max(hy - radius, 1e-3f);
                    geometry = PxCapsuleGeometry(radius, halfHeight);
                    localPose = PxTransform(localPose.p, PxQuat(-PxHalfPi, PxVec3(0, 0, 1)));
                    break;
                }
                default:
                    geometry = PxBoxGeometry(hx, hy, hz);
                    break;
            }

            if (!moving) {
                PxRigidStatic* body = world_->physics().createRigidStatic(placement.pose);
                PxShape* pxShape = world_->physics().createShape(geometry.any(), *material, true);
                pxShape->setLocalPose(localPose);
                body->attachShape(*pxShape);
                pxShape->release();
                world_->scene().addActor(*body);
                return record(object, body);
            }

            PxRigidDynamic* body = world_->physics().createRigidDynamic(placement.pose);
            PxShape* pxShape = world_->physics().createShape(geometry.any(), *material, true);
            pxShape->setLocalPose(localPose);
            body->attachShape(*pxShape);
            pxShape->release();
            world_->scene().addActor(*body);
            finishDynamic(*body, config);
            // PhysxWorld writes the actor pose back into the object each step,
            // converting to parent-local space for nested nodes.
            world_->bind(object, *body);
            return record(object, body);
        }

        void finishDynamic(::physx::PxRigidDynamic& body, const PhysicsConfig& config) {

            using namespace ::physx;

            if (config.body == PhysicsConfig::Body::Kinematic) {
                body.setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
            }
            // Mass is authored directly (kg), so derive the inertia from it
            // rather than from a density the user never sees.
            PxRigidBodyExt::setMassAndUpdateInertia(body, std::max(config.mass, 1e-3f));
        }

        PhysxWorld::Settings settings_;
        std::unique_ptr<PhysxWorld> world_;
        std::function<void(const std::string&)> logger_;
        std::size_t bodyCount_ = 0;
        std::size_t softBodyCount_ = 0;
        std::size_t decompCookCount_ = 0;
        bool gpu_ = false;
        std::shared_ptr<const void> lifetime_;
        std::unordered_map<const Object3D*, ::physx::PxRigidActor*> actors_;
        // The same records the other way round (see findObject). Not the exact
        // inverse of actors_: a subtree collider puts every one of its actors in
        // here against the one root that owns them.
        std::unordered_map<const ::physx::PxRigidActor*, Object3D*> objects_;
        std::vector<std::unique_ptr<PlayedArticulation>> articulations_;

        // V-HACD is the expensive part of a Pieces cook (seconds on a dense
        // mesh), so N copies of one geometry decompose once — keyed on the
        // geometry uuid AND the parameters, since a different hull budget is a
        // different decomposition. Value is one flat x,y,z array per hull, in the
        // geometry's local (unscaled) frame; the caller applies scale at cook
        // time. Mirrors PhysxWorld's soft-body tet cache. Lives for the session.
        std::unordered_map<std::string, std::vector<std::vector<float>>> decompCache_;

        inline static PhysicsPlaySession* active_ = nullptr;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_PHYSICSPLAYSESSION_HPP
