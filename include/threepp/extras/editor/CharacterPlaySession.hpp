// The PlaySession that makes an authored character WALK: one PhysX capsule
// controller and one AnimationMixer per node carrying a CharacterConfig, plus
// the gait state machine that picks which clip belongs to the velocity the
// controller is actually travelling at.
//
// Header-only and PhysX-dependent, exactly like GranularPlaySession — included
// only by builds that found the SDK (the apps set THREEPP_EDITOR_WITH_PHYSX,
// and the registration sites carry the guard).
//
// The world is BORROWED from PhysicsPlaySession, with the granular session's
// contract verbatim: on the normal Stop it is still alive here (PlayController
// stops sessions in reverse registration order and physics goes last, so this
// session must be registered AFTER it), and the teardown paths that cannot
// promise that are guarded by PhysicsPlaySession's lifetime token. A scene with
// no physics gets one log line and characters that stand still, rather than a
// failed Play.
//
// WHY A PxController AND NOT A RIGID BODY. A character is not a physical
// object and simulating it as one is the classic mistake: a capsule with a
// mass tips over, bounces down stairs, and slides on any slope its friction
// cannot hold. PhysX's character controller is a kinematic collide-and-slide
// sweep with a step offset (walks over a kerb instead of into it) and a slope
// limit (climbs a ramp, slides off a cliff), which is what "walking" actually
// means. Gravity and jumping are integrated by this session, not by the
// solver.
//
// THE FEET DO NOT SLIDE, and that is the whole reason CharacterConfig measures
// clips at all. Every clip travels at the speed its animator authored; playing
// one at 1x while the controller moves at some other speed IS foot-sliding. So
// for each direction the session holds two clips, takes whichever one's own
// speed is nearer (in ratio) to what is being travelled, and time-scales away
// the remainder — the trick the FPS demo's bots use, here reading the numbers
// off the asset instead of a hand-measured table.
//
// AND THE CLIPS DO NOT MOVE THE CHARACTER. The controller owns the position,
// so the root bone's authored travel has to go: each frame the bone's offset
// from its bind pose is projected onto the body's own up axis and everything
// else is discarded. That keeps the gait's vertical bob (which is animation)
// and drops the horizontal travel (which is now the controller's job). The
// projection is done in the bone's parent frame against a constant axis, which
// is exact as long as the body stays upright — and a character controller
// keeps it upright by construction.

#ifndef THREEPP_EDITOR_CHARACTERPLAYSESSION_HPP
#define THREEPP_EDITOR_CHARACTERPLAYSESSION_HPP

#include "threepp/extras/editor/CharacterConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"

#include "threepp/animation/AnimationAction.hpp"
#include "threepp/animation/AnimationClip.hpp"
#include "threepp/animation/AnimationMixer.hpp"
#include "threepp/constants.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace threepp::editor {

    class CharacterPlaySession: public PlaySession {

    public:
        [[nodiscard]] std::string name() const override { return "Character"; }

        // A session destroyed without a stop() (the app torn down mid-Play)
        // must still release its controllers while the world is alive, or not
        // touch them at all.
        ~CharacterPlaySession() override { release(); }

        // --- wiring (set once, before the first Play) ------------------------

        // Where the controllers register. Borrowed; the token read from it is
        // what keeps a stopped world from being dereferenced.
        void setPhysics(PhysicsPlaySession* physics) { physics_ = physics; }

        void setLogger(std::function<void(const std::string&)> logger) {
            logger_ = std::move(logger);
        }

        // --- control ---------------------------------------------------------

        // One frame of intent, in terms the caller can supply without knowing
        // anything about the character: a move demand on the VIEW's axes, a
        // run modifier, a jump edge, and the yaw the view looks along.
        //
        // The editor polls its keys and calls drive() every frame while
        // playing, so releasing everything decelerates to Idle rather than
        // leaving the last demand latched.
        struct Input {
            float forward = 0.f;// -1 back .. +1 forward, along the view
            float strafe = 0.f; // -1 right .. +1 left, across the view
            bool run = false;
            // An EDGE, not a state: the session launches on the frame this is
            // first true and ignores it until the character lands again.
            bool jump = false;
            // Radians, three.js convention (yaw of +Z is 0, atan2(x, z)).
            float viewYaw = 0.f;
        };

        // Drives every played character. Plural for driveVehicles' reason: the
        // editor has one set of keys and a scene almost always has one
        // character; a scene with several gets them marching together, which is
        // at least honest.
        void drive(const Input& input) { input_ = input; }

        // --- readouts --------------------------------------------------------

        [[nodiscard]] std::size_t characterCount() const { return played_.size(); }

        // The scene had characters and this session could not simulate them.
        [[nodiscard]] bool declined() const { return declined_; }
        [[nodiscard]] const std::string& reason() const { return reason_; }

        // One simulated character.
        struct Played {

            Object3D* root = nullptr;
            CharacterConfig config;
            CharacterGeometry geo;

            // Owned by manager_, released with it.
            ::physx::PxController* controller = nullptr;

            std::unique_ptr<AnimationMixer> mixer;
            std::array<AnimationAction*, kGaitCount> actions{};
            AnimationAction* current = nullptr;

            // Root-motion pinning (see the header note).
            Object3D* rootBone = nullptr;
            Vector3 rootBoneBind;
            Vector3 rootBoneUp;// the body's up, in the bone's parent frame

            // Where the model's ORIGIN sits relative to its feet, in the
            // character's own yawed frame. Usually ~zero (a Mixamo model
            // stands on its origin), but a model whose root is at the hips
            // must not sink into the floor on the first frame.
            Vector3 originFromFeet;

            float yaw = 0.f;
            Vector3 velocity;      // world horizontal, m/s
            float verticalSpeed = 0.f;
            bool grounded = false;
            bool jumpLatched = false;// consumed the current jump edge
            float height = 0.f;
            float radius = 0.f;
            float walkSpeed = 0.f;
            float runSpeed = 0.f;

            [[nodiscard]] float speed() const {
                return std::sqrt(velocity.x * velocity.x + velocity.z * velocity.z);
            }
        };

        [[nodiscard]] const std::vector<std::unique_ptr<Played>>& characters() const {
            return played_;
        }

        // The character governing `object`, or the nearest ancestor of it that
        // is one — the same walk-up-parents contract findVehicle keeps.
        [[nodiscard]] const Played* findCharacter(const Object3D* object) const {

            for (const Object3D* o = object; o != nullptr; o = o->parent) {
                for (const auto& played : played_) {
                    if (played->root == o) return played.get();
                }
            }
            return nullptr;
        }

        // The character a chase camera should follow: the first in document
        // order. Null when the scene has none.
        [[nodiscard]] const Played* player() const {
            return played_.empty() ? nullptr : played_.front().get();
        }

        // --- PlaySession -----------------------------------------------------

        void start(Scene& scene) override {

            release();
            declined_ = false;
            reason_.clear();
            input_ = {};

            PhysxWorld* world = nullptr;
            worldLife_.reset();
            if (physics_) {
                world = physics_->world();
                worldLife_ = physics_->lifetime();
            }

            scene.updateMatrixWorld(true);

            std::vector<Object3D*> roots;
            scene.traverse([&](Object3D& object) {
                if (CharacterConfig::isCharacter(object)) roots.push_back(&object);
            });
            if (roots.empty()) return;

            if (!world) {
                decline("character: no physics this play - characters stand still");
                return;
            }

            manager_ = PxCreateControllerManager(world->scene());
            if (!manager_) {
                decline("character: PhysX declined to create a controller manager - "
                        "characters stand still");
                return;
            }

            for (auto* root : roots) build(*root, *world);
        }

        void update(float dt) override {

            if (dt <= 0.f) return;
            for (const auto& played : played_) step(*played, dt);
            // One edge per press: the caller keeps reporting `jump` while the
            // key is held, and holding it must not pogo.
            if (!input_.jump) {
                for (const auto& played : played_) played->jumpLatched = false;
            }
        }

        void stop() override { release(); }

    private:
        // --- tuning that is not worth an authoring field ---------------------

        // Below this the character is standing, not walking: the Idle clip.
        static constexpr float kMoveThreshold = 0.15f;
        // How forward/sideways the travel has to be to claim a direction. The
        // asymmetry is deliberate and comes from the FPS demo: a character
        // reads as "walking forward" over a wider cone than it reads as
        // "backpedalling".
        static constexpr float kForwardCone = 0.55f;
        static constexpr float kBackwardCone = -0.45f;
        // Clamp on the residual time scale, so a clip stretched to an absurd
        // speed looks wrong rather than broken.
        static constexpr float kTimeScaleMin = 0.55f;
        static constexpr float kTimeScaleMax = 1.7f;
        // Air control, as a fraction of the ground acceleration.
        static constexpr float kAirControl = 0.35f;
        // Downward bias while grounded, so the controller stays welded to the
        // floor across a seam instead of stepping off it every frame.
        static constexpr float kGroundStick = 2.f;
        // The sweep's minimum distance: below it PhysX skips the move.
        static constexpr float kMinMoveDist = 0.001f;

        static float wrapPi(float angle) {

            while (angle > math::PI) angle -= 2.f * math::PI;
            while (angle < -math::PI) angle += 2.f * math::PI;
            return angle;
        }

        // World matrix -> the node's local position/quaternion. Scale is left
        // alone: the session moves models, it does not resize them. (The same
        // helper PhysicsPlaySession uses for its vehicles.)
        static void writeLocalPose(Object3D& node, const Matrix4& world) {

            Matrix4 local(world);
            if (node.parent) {
                node.parent->updateWorldMatrix(true, false);
                Matrix4 parentInv(*node.parent->matrixWorld);
                parentInv.invert();
                local.premultiply(parentInv);
            }
            Vector3 position, scale;
            Quaternion rotation;
            local.decompose(position, rotation, scale);
            node.position.copy(position);
            node.quaternion.copy(rotation);
        }

        void build(Object3D& root, PhysxWorld& world) {

            const auto config = CharacterConfig::read(root);
            if (!config) return;

            auto geo = config->derived(root);
            if (!geo.valid) {
                log("character: \"" + root.name + "\" - " + geo.problem + " - not simulated");
                return;
            }
            // A partial derive is worth saying out loud but is never fatal: a
            // character with no clips still walks, T-posed.
            if (!geo.problem.empty()) {
                log("character: \"" + root.name + "\" - " + geo.problem);
            }

            auto played = std::make_unique<Played>();
            played->root = &root;
            played->config = *config;
            played->geo = geo;

            const bool autoGeo = config->autoGeometry;
            played->height = std::max(autoGeo ? geo.height : config->height, 0.1f);
            played->radius = std::clamp(autoGeo ? geo.radius : config->radius,
                                        0.02f, 0.45f * played->height);

            // Speeds: the clips' own numbers while auto is on, the authored
            // ones once it is off. A missing clip falls back to the authored
            // default rather than to zero, or the character would refuse to
            // move on a model that only shipped an idle.
            const float walkClip = geo.slot(Gait::Walk).speed;
            const float runClip = geo.slot(Gait::Run).speed;
            played->walkSpeed = std::max(
                    config->autoSpeeds && walkClip > 0.f ? walkClip : config->walkSpeed, 0.1f);
            played->runSpeed = std::max(
                    config->autoSpeeds && runClip > 0.f ? runClip : config->runSpeed,
                    played->walkSpeed);

            // --- the capsule ---------------------------------------------------
            root.updateWorldMatrix(true, false);
            Vector3 rootPos, rootScale;
            Quaternion rootRot;
            root.matrixWorld->decompose(rootPos, rootRot, rootScale);

            // Spawn facing the way it was placed, so a character posed to look
            // down a corridor does not snap north on Play.
            Vector3 facing(0.f, 0.f, 1.f);
            facing.applyQuaternion(rootRot);
            played->yaw = std::atan2(facing.x, facing.z);

            // The model origin relative to its feet, expressed in the
            // character's own frame so it rides the yaw.
            Vector3 offset;
            offset.subVectors(rootPos, geo.feet);
            Quaternion spawnInv;
            spawnInv.setFromAxisAngle(Vector3(0.f, 1.f, 0.f), -played->yaw);
            offset.applyQuaternion(spawnInv);
            played->originFromFeet = offset;

            ::physx::PxCapsuleControllerDesc desc;
            desc.radius = played->radius;
            // PhysX's height is the CYLINDER between the two caps, not the
            // standing height. A capsule whose radius is half its height has
            // no cylinder at all, which the clamp above already prevents.
            desc.height = std::max(played->height - 2.f * played->radius, 0.01f);
            desc.upDirection = ::physx::PxVec3(0.f, 1.f, 0.f);
            desc.slopeLimit = std::cos(std::clamp(config->slopeLimit, 0.f, 1.5f));
            desc.stepOffset = std::clamp(config->stepOffset, 0.f, 0.5f * played->height);
            // A tenth of the radius, PhysX's own rule of thumb. Too small and
            // the sweep re-resolves the same contact every frame, which reads
            // as a character vibrating against a slope.
            desc.contactOffset = std::clamp(0.1f * played->radius, 0.005f, 0.05f);
            desc.material = &world.defaultMaterial();
            desc.climbingMode = ::physx::PxCapsuleClimbingMode::eEASY;
            desc.position = ::physx::PxExtendedVec3(
                    geo.feet.x, geo.feet.y + 0.5 * static_cast<double>(played->height), geo.feet.z);

            played->controller = manager_->createController(desc);
            if (!played->controller) {
                log("character: \"" + root.name + "\" - PhysX declined the capsule - not simulated");
                return;
            }

            // --- the mixer -----------------------------------------------------
            played->mixer = std::make_unique<AnimationMixer>(root);
            for (std::size_t i = 0; i < kGaitCount; ++i) {
                const auto& slot = geo.gaits[i];
                if (!slot.clip) continue;
                auto* action = played->mixer->clipAction(slot.clip);
                if (!action) continue;
                const bool oneShot = i == gaitIndex(Gait::Jump);
                action->setLoop(oneShot ? Loop::Once : Loop::Repeat);
                action->setClampWhenFinished(oneShot);
                played->actions[i] = action;
            }
            // Start in the pose the character will hold: Idle if it has one,
            // else whatever it does have, so Play never opens on a T-pose when
            // the model shipped something better.
            played->current = firstAvailable(*played, {Gait::Idle, Gait::Walk, Gait::Run});
            if (played->current) {
                played->current->setEffectiveWeight(1.f);
                played->current->play();
            }

            // --- root-motion pinning ------------------------------------------
            played->rootBone = geo.rootBone;
            played->rootBoneBind = geo.rootBoneBind;
            if (played->rootBone && played->rootBone->parent) {
                auto* parent = played->rootBone->parent;
                parent->updateWorldMatrix(true, false);
                // The body's up, expressed in the bone's parent frame. The
                // frame yaws with the character, and yawing about the world up
                // leaves that axis fixed, so this is computed once.
                Matrix4 basisInv(*parent->matrixWorld);
                basisInv.elements[12] = 0.f;
                basisInv.elements[13] = 0.f;
                basisInv.elements[14] = 0.f;
                basisInv.invert();
                Vector3 up(0.f, 1.f, 0.f);
                up.applyMatrix4(basisInv);
                if (up.length() > 1e-6f) {
                    up.normalize();
                    played->rootBoneUp = up;
                } else {
                    played->rootBone = nullptr;// degenerate rig: leave the clip alone
                }
            }

            played_.push_back(std::move(played));
        }

        static AnimationAction* firstAvailable(const Played& played,
                                               std::initializer_list<Gait> order) {

            for (const Gait gait : order) {
                if (auto* action = played.actions[gaitIndex(gait)]) return action;
            }
            for (auto* action : played.actions) {
                if (action) return action;
            }
            return nullptr;
        }

        void step(Played& played, float dt) {

            if (!played.controller || !played.root) return;

            // --- what the input asks for, on the view's axes --------------------
            const float sy = std::sin(input_.viewYaw), cy = std::cos(input_.viewYaw);
            // forward = (sin, 0, cos); left is that turned a quarter turn, which
            // is (cos, 0, -sin) in this convention. Both are unit.
            Vector3 demand(input_.forward * sy + input_.strafe * cy, 0.f,
                           input_.forward * cy - input_.strafe * sy);
            const float demandLen = demand.length();
            if (demandLen > 1.f) demand.multiplyScalar(1.f / demandLen);

            const float top = input_.run ? played.runSpeed : played.walkSpeed;
            Vector3 target(demand);
            target.multiplyScalar(top);

            const float k = played.grounded ? played.config.accel
                                            : played.config.accel * kAirControl;
            const float alpha = std::min(1.f, dt * std::max(k, 0.1f));
            played.velocity.x += (target.x - played.velocity.x) * alpha;
            played.velocity.z += (target.z - played.velocity.z) * alpha;

            // --- gravity and the jump edge --------------------------------------
            const float g = std::max(played.config.gravity, 0.1f);
            if (input_.jump && played.grounded && !played.jumpLatched) {
                played.verticalSpeed = std::sqrt(
                        2.f * g * std::max(played.config.jumpHeight, 0.01f));
                played.grounded = false;
                played.jumpLatched = true;
                if (auto* jump = played.actions[gaitIndex(Gait::Jump)]) {
                    // reset() AND play(): the jump clip is Loop::Once, so a
                    // previous jump left it finished and deactivated. Rewinding
                    // a deactivated action shows nothing — the second jump in a
                    // row would be silent.
                    jump->reset();
                    jump->play();
                }
            } else if (played.grounded && played.verticalSpeed <= 0.f) {
                played.verticalSpeed = -kGroundStick;
            } else {
                played.verticalSpeed -= g * dt;
            }

            // --- move -----------------------------------------------------------
            const auto before = played.controller->getFootPosition();
            const ::physx::PxVec3 disp(played.velocity.x * dt,
                                       played.verticalSpeed * dt,
                                       played.velocity.z * dt);
            const auto flags = played.controller->move(disp, kMinMoveDist, dt,
                                                       ::physx::PxControllerFilters());
            const auto after = played.controller->getFootPosition();

            const bool down = flags.isSet(::physx::PxControllerCollisionFlag::eCOLLISION_DOWN);
            const bool up = flags.isSet(::physx::PxControllerCollisionFlag::eCOLLISION_UP);
            played.grounded = down;
            if (down && played.verticalSpeed < 0.f) played.verticalSpeed = 0.f;
            if (up && played.verticalSpeed > 0.f) played.verticalSpeed = 0.f;

            // Resync the horizontal velocity from what actually happened. With
            // nothing in the way this is the identity; walking into a wall it
            // is what stops the legs from running on the spot at full speed.
            played.velocity.x = static_cast<float>(after.x - before.x) / dt;
            played.velocity.z = static_cast<float>(after.z - before.z) / dt;

            // --- facing -----------------------------------------------------------
            float targetYaw = played.yaw;
            if (played.config.facing == CharacterConfig::Facing::Camera) {
                targetYaw = input_.viewYaw;
            } else if (demandLen > 0.01f) {
                targetYaw = std::atan2(demand.x, demand.z);
            }
            played.yaw += wrapPi(targetYaw - played.yaw) *
                          std::min(1.f, dt * std::max(played.config.turnRate, 0.1f));
            played.yaw = wrapPi(played.yaw);

            // --- place the model ---------------------------------------------------
            Quaternion rotation;
            rotation.setFromAxisAngle(Vector3(0.f, 1.f, 0.f), played.yaw);
            Vector3 origin(played.originFromFeet);
            origin.applyQuaternion(rotation);
            origin.add(Vector3(static_cast<float>(after.x), static_cast<float>(after.y),
                               static_cast<float>(after.z)));

            Matrix4 world;
            world.compose(origin, rotation, Vector3(1.f, 1.f, 1.f));
            writeLocalPose(*played.root, world);

            // --- gait ---------------------------------------------------------------
            chooseGait(played);
            if (played.mixer) played.mixer->update(dt);
            pinRootMotion(played);
        }

        void chooseGait(Played& played) {

            const float speed = played.speed();
            AnimationAction* want = played.actions[gaitIndex(Gait::Idle)];
            float timeScale = 1.f;

            const auto pick = [&](Gait slowGait, Gait fastGait) {
                auto* slow = played.actions[gaitIndex(slowGait)];
                auto* fast = played.actions[gaitIndex(fastGait)];
                const float slowSpeed = played.geo.slot(slowGait).speed;
                const float fastSpeed = played.geo.slot(fastGait).speed;

                AnimationAction* chosen = nullptr;
                float reference = 0.f;
                // Nearer in RATIO, not in difference: a clip authored at 1 m/s
                // played at 2 m/s looks twice as wrong as one authored at
                // 4 m/s played at 5, and the log distance is what says so.
                const bool haveSlow = slow && slowSpeed > 0.01f;
                const bool haveFast = fast && fastSpeed > 0.01f;
                if (haveSlow && haveFast) {
                    const bool preferFast = std::abs(std::log(speed / fastSpeed)) <
                                            std::abs(std::log(speed / slowSpeed));
                    chosen = preferFast ? fast : slow;
                    reference = preferFast ? fastSpeed : slowSpeed;
                } else if (haveSlow) {
                    chosen = slow;
                    reference = slowSpeed;
                } else if (haveFast) {
                    chosen = fast;
                    reference = fastSpeed;
                }
                if (chosen) timeScale = std::clamp(speed / reference, kTimeScaleMin, kTimeScaleMax);
                return chosen;
            };

            auto* jump = played.actions[gaitIndex(Gait::Jump)];
            if (!played.grounded && jump) {
                want = jump;
                timeScale = 1.f;
            } else if (speed > kMoveThreshold) {
                // Resolved in the CHARACTER's own frame, which is what makes a
                // strafing character strafe instead of sliding sideways on the
                // walk cycle.
                const float s = std::sin(played.yaw), c = std::cos(played.yaw);
                const float forwardness = (played.velocity.x * s + played.velocity.z * c) / speed;
                const float leftness = (played.velocity.x * c - played.velocity.z * s) / speed;

                AnimationAction* chosen = nullptr;
                if (forwardness > kForwardCone) {
                    chosen = pick(Gait::Walk, Gait::Run);
                } else if (forwardness < kBackwardCone) {
                    chosen = pick(Gait::WalkBack, Gait::RunBack);
                } else if (leftness > 0.f) {
                    chosen = pick(Gait::StrafeLeft, Gait::StrafeLeftFast);
                } else {
                    chosen = pick(Gait::StrafeRight, Gait::StrafeRightFast);
                }
                // A model that shipped no clip for this direction walks
                // forward rather than snapping to idle mid-stride.
                if (!chosen) {
                    chosen = pick(Gait::Walk, Gait::Run);
                }
                if (chosen) want = chosen;
            }

            if (!want) return;
            want->setEffectiveTimeScale(timeScale);
            if (want == played.current) return;

            want->reset();
            want->play();
            if (played.current) {
                played.current->crossFadeTo(want, std::max(played.config.blendTime, 0.f));
            } else {
                want->setEffectiveWeight(1.f);
            }
            played.current = want;
        }

        // Drop the clip's horizontal travel, keep its vertical bob. See the
        // header note for why the projection is exact.
        static void pinRootMotion(const Played& played) {

            if (!played.rootBone) return;
            Vector3 delta(played.rootBone->position);
            delta.sub(played.rootBoneBind);
            const float along = delta.dot(played.rootBoneUp);
            Vector3 kept(played.rootBoneUp);
            kept.multiplyScalar(along);
            kept.add(played.rootBoneBind);
            played.rootBone->position.copy(kept);
        }

        void release() {

            // The world may already be gone on a teardown that skipped the
            // controller (the app closed mid-Play): then the scene the
            // controllers registered with died with it and releasing them
            // would walk freed memory.
            const bool worldAlive = !worldLife_.expired();
            played_.clear();// mixers first; they only touch the scene graph
            if (manager_ && worldAlive) manager_->release();
            manager_ = nullptr;
            worldLife_.reset();
            input_ = {};
        }

        void decline(std::string reason) {

            declined_ = true;
            reason_ = std::move(reason);
            log(reason_);
        }

        void log(const std::string& message) const {

            if (logger_) logger_(message);
        }

        PhysicsPlaySession* physics_ = nullptr;
        std::function<void(const std::string&)> logger_;

        std::weak_ptr<const void> worldLife_;
        ::physx::PxControllerManager* manager_ = nullptr;
        std::vector<std::unique_ptr<Played>> played_;

        Input input_;
        bool declined_ = false;
        std::string reason_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_CHARACTERPLAYSESSION_HPP
