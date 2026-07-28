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

#ifndef THREEPP_EDITOR_PHYSICSPLAYSESSION_HPP
#define THREEPP_EDITOR_PHYSICSPLAYSESSION_HPP

#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PlaySession.hpp"
#include "threepp/extras/editor/SplineConfig.hpp"

#include "threepp/extras/curves/RoadGeometry.hpp"

#include "threepp/extras/physx/PhysxWorld.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CapsuleGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
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

        [[nodiscard]] PhysxWorld* world() { return world_.get(); }

        void start(Scene& scene) override {

            bodyCount_ = 0;
            world_ = std::make_unique<PhysxWorld>(settings_);

            // Collect first, create second: creating an actor binds the object,
            // and PhysxWorld writes transforms during step(), not during
            // traversal — but a stable list also keeps behaviour independent of
            // any graph edit a later hook might make.
            std::vector<Object3D*> targets;
            scene.updateMatrixWorld(true);
            scene.traverse([&](Object3D& object) {
                if (const auto config = PhysicsConfig::read(object); config && config->enabled) {
                    targets.push_back(&object);
                }
            });

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

            world_.reset();
            bodyCount_ = 0;
        }

    private:
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

        // A road generated by a spline: the derived child of a spline whose
        // config says Road. Tubes are not roads — a closed cross-section is
        // exactly what a triangle mesh handles well.
        static bool isDerivedRoad(const Object3D& object) {

            if (!SplineConfig::isDerived(object)) return false;
            if (!object.parent) return false;
            const auto spline = SplineConfig::read(*object.parent);
            return spline && spline->mesh == SplineConfig::MeshKind::Road;
        }

        // Objects that get an actor of their own from start()'s walk. A subtree
        // collider must not cook them a second time.
        static bool ownsActor(const Object3D& object) {

            const auto config = PhysicsConfig::read(object);
            return config && config->enabled;
        }

        // Depth a road collider is extruded below its visual surface. A road
        // surface has no thickness at all; a body moving fast enough passes
        // through one between substeps whatever the solver does about it.
        static constexpr float kRoadThickness = 0.25f;

        // A road collides as ONE static triangle mesh, cooked from a closed
        // solid built out of the road's OWN triangles (RoadGeometry::solid):
        // the surface, a copy of it pushed 0.25 m down, and a wall along every
        // boundary edge. So the collider is the surface the user is looking at,
        // triangle for triangle, and no cross-section of it can disagree with
        // the drawing however the road was trimmed.
        //
        // Static only — a triangle mesh cannot move — which is what lets this
        // be one shape rather than the convex decomposition a kinematic body
        // would need. The whole road is one shape whatever its shape.
        bool addRoadSolid(Mesh& mesh, ::physx::PxMaterial* material) {

            using namespace ::physx;

            const auto geometry = mesh.geometry();
            if (!geometry) return false;
            const auto solid = RoadGeometry::solid(*geometry, kRoadThickness);
            const auto* position = solid->getAttribute<float>("position");
            if (!position || position->count() < 4) return false;

            mesh.updateMatrixWorld();
            Vector3 translation, scale;
            Quaternion rotation;
            mesh.matrixWorld->decompose(translation, rotation, scale);

            // The scale rides along as a PxMeshScale rather than being baked
            // in: one mesh, cooked once, and the solid is built in the geometry
            // its own vertices are in.
            return world_->addStaticTrimesh(
                           *solid, PxTransform(toPxVec3(translation), toPxQuat(rotation)),
                           scale, material) != nullptr;
        }

        // Cook the descendants of a geometry-less static body as its collider.
        // Meshes carrying their own enabled PhysicsConfig are left alone —
        // start() is creating their actors — and a derived road goes through
        // the road solid above rather than a bare surface trimesh, so it
        // collides the same whether the config sat on it or on the group over
        // it — and so a body cannot end up under a road with no underside.
        std::size_t addSubtree(Object3D& root, ::physx::PxMaterial* material) {

            std::size_t added = 0;

            root.updateMatrixWorld();
            root.traverseType<Mesh>([&](Mesh& mesh) {
                if (&mesh == &root || ownsActor(mesh) || !mesh.geometry()) return;
                if (!isDerivedRoad(mesh)) return;
                if (addRoadSolid(mesh, material)) ++added;
            });

            const auto actors = world_->addStaticTrimeshTree(
                    root,
                    [&root](const Mesh& mesh) {
                        return &mesh != &root && mesh.geometry() != nullptr &&
                               !ownsActor(mesh) && !isDerivedRoad(mesh);
                    },
                    material);
            return added + actors.size();
        }

        bool createActor(Object3D& object, const PhysicsConfig& config) {

            using namespace ::physx;

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
            // exists for: the spline IS the road as far as the user is
            // concerned, and the unit-box fallback below turned that into a
            // phantom 1 m cube at the spline's origin.
            if (!moving && !object.geometry() &&
                (config.shape == PhysicsConfig::Shape::Auto ||
                 config.shape == PhysicsConfig::Shape::TriMesh)) {
                if (addSubtree(object, material) > 0) return true;
                // Nothing under it: fall through to the placeholder, which is
                // still what a bare Group used as a trigger volume wants.
            }

            // A spline-derived road collides as its solid whichever of the
            // mesh shapes was asked for — Auto, TriMesh and Convex all mean
            // "the surface I can see" here, and the solid is the only one of
            // the three that is exact AND safe to rest a body on (a bare
            // surface has no thickness, a hull swallows every bend). An
            // explicit primitive is left alone; so is a moving road, which is a
            // plank rather than a road and keeps the whole-mesh hull.
            if (mesh && !moving && isDerivedRoad(*mesh) &&
                (shape == PhysicsConfig::Shape::TriMesh || shape == PhysicsConfig::Shape::Convex)) {
                if (addRoadSolid(*mesh, material)) return true;
            }

            if (shape == PhysicsConfig::Shape::TriMesh && mesh) {
                // Triangle meshes are valid for static/kinematic actors only;
                // a dynamic request silently gets a static collider rather than
                // a PhysX assertion.
                return world_->addStaticTrimesh(*mesh, material) != nullptr;
            }
            if (shape == PhysicsConfig::Shape::Convex && mesh && moving) {
                auto* body = world_->addDynamicConvex(*mesh, 1000.f, material);
                if (!body) return false;
                finishDynamic(*body, config);
                return true;
            }
            if (shape == PhysicsConfig::Shape::Convex && mesh) {
                return world_->addStaticTrimesh(*mesh, material) != nullptr;
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
                return true;
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
            return true;
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
        std::size_t bodyCount_ = 0;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_PHYSICSPLAYSESSION_HPP
