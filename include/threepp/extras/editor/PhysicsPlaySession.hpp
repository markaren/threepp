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

#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PlaySession.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"


#include "threepp/extras/physx/PhysxSoftBody.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CapsuleGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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

        // A token that lives exactly as long as the world built by the last
        // start(). Handles handed to scripts keep a weak_ptr to it, so a body
        // still referenced after Stop reports that it is gone instead of
        // dereferencing a released actor.
        [[nodiscard]] std::weak_ptr<const void> lifetime() const { return lifetime_; }

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
            actors_.clear();

            // Collect first, create second: creating an actor binds the object,
            // and PhysxWorld writes transforms during step(), not during
            // traversal — but a stable list also keeps behaviour independent of
            // any graph edit a later hook might make.
            std::vector<Object3D*> targets;
            bool wantsSoftBodies = false;
            scene.updateMatrixWorld(true);
            scene.traverse([&](Object3D& object) {
                if (const auto config = PhysicsConfig::read(object); config && config->enabled) {
                    targets.push_back(&object);
                    if (config->body == PhysicsConfig::Body::Soft) wantsSoftBodies = true;
                }
            });

            createWorld(wantsSoftBodies);
            lifetime_ = std::make_shared<const char>('\0');
            active_ = this;

            for (auto* object : targets) {
                const auto config = PhysicsConfig::read(*object);
                if (!config) continue;
                if (createActor(*object, *config)) ++bodyCount_;
            }
        }

        void update(float dt) override {

            if (world_) world_->step(dt);
        }

        void stop() override {

            // Drop the token BEFORE the world: any handle a script is still
            // holding must read as dead from the moment the actors go.
            lifetime_.reset();
            if (active_ == this) active_ = nullptr;
            actors_.clear();
            world_.reset();
            bodyCount_ = 0;
            softBodyCount_ = 0;
        }

    private:
        void log(const std::string& message) {

            if (logger_) logger_(message);
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
            return actors.size();
        }

        // Remember the actor governing `object`, so a script can find it later.
        // Returns true so the creation sites can `return record(...)`.
        bool record(const Object3D& object, ::physx::PxRigidActor* actor) {

            if (actor) actors_.emplace(&object, actor);
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
            // off-centre mesh gets a correctly offset shape.
            const Box3 bounds = localBounds(object);
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
        bool gpu_ = false;
        std::shared_ptr<const void> lifetime_;
        std::unordered_map<const Object3D*, ::physx::PxRigidActor*> actors_;
        inline static PhysicsPlaySession* active_ = nullptr;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_PHYSICSPLAYSESSION_HPP
