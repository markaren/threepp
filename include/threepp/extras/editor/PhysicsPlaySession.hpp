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
        // own AABB. For anything shaped — a road ribbon, a terrain patch, an
        // imported prop — that slab is not an approximation of the surface, it
        // is a different object: a flat ribbon's AABB is a razor at the minimum
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

        // Largest angle one collider wedge spans. Coarser than the mesh's, on
        // purpose: a wedge is a solid a body rests on, not a silhouette.
        static constexpr float kRoadWedgeStep = 0.2618f;// 15 degrees

        // One PxRigidStatic per road, built from the road's PRIMITIVES — the
        // conveyor-belt treatment (examples/projects/Fish/ConveyorSystem):
        // a box per straight, and a bend tiled by convex ANNULAR WEDGES that
        // share each radial face exactly, where overlapping rectangles fight
        // each other on a tight bend.
        //
        // The primitives are recomputed from the spline config rather than read
        // off the mesh: the segmentation is deterministic, so the collider and
        // the surface the user is looking at come out of the same numbers — and
        // the shape count follows the road's SHAPE, a handful either way,
        // instead of following how finely the curve was sampled.
        bool addRoadPrimitives(Mesh& mesh, ::physx::PxMaterial* material) {

            using namespace ::physx;

            if (!mesh.parent) return false;
            const auto config = SplineConfig::read(*mesh.parent);
            if (!config) return false;
            const auto road = config->roadPath(*mesh.parent);
            if (!road || road->empty()) return false;

            const float half = std::max(config->width, 1e-3f) * 0.5f;

            mesh.updateMatrixWorld();
            Vector3 translation, scale;
            Quaternion rotation;
            mesh.matrixWorld->decompose(translation, rotation, scale);

            // Scale is BAKED INTO THE POINTS rather than carried as a
            // PxMeshScale: every piece is cooked once and attached once, so a
            // shared mesh with a per-shape scale buys nothing — and baking
            // leaves the actor's frame a pure rotation, in which the extrusion
            // below is a real 0.25 m rather than 0.25 m times whatever the node
            // was scaled by.
            const auto safe = [](float v) { return std::abs(v) < 1e-6f ? 1.f : v; };
            const Vector3 s{safe(scale.x), safe(scale.y), safe(scale.z)};
            const auto scaled = [&s](const Vector3& v) {
                return PxVec3(v.x * s.x, v.y * s.y, v.z * s.z);
            };
            // A box is a box under a uniform scale only. Anything else goes
            // through the hull path below, which is exact under any of them.
            const bool uniform = std::abs(s.x - s.y) < 1e-4f * std::abs(s.x) &&
                                 std::abs(s.x - s.z) < 1e-4f * std::abs(s.x);

            PxCookingParams params(world_->physics().getTolerancesScale());
            PxRigidStatic* body = world_->physics().createRigidStatic(
                    PxTransform(toPxVec3(translation), toPxQuat(rotation)));

            int shapes = 0;
            const auto attach = [&](const PxGeometry& geometry, const PxTransform& pose) {
                PxShape* shape = world_->physics().createShape(geometry, *material, true);
                shape->setLocalPose(pose);
                body->attachShape(*shape);
                shape->release();
                ++shapes;
            };
            const auto attachHull = [&](const PxVec3* points, PxU32 count) {
                PxConvexMeshDesc desc;
                desc.points.count = count;
                desc.points.stride = sizeof(PxVec3);
                desc.points.data = points;
                desc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
                PxConvexMesh* convex = PxCreateConvexMesh(params, desc);
                if (!convex) return;
                attach(PxConvexMeshGeometry(convex), PxTransform(PxIdentity));
                // The shape holds the reference now.
                convex->release();
            };

            for (const auto& primitive : road->primitives()) {

                const float length = primitive.length();
                if (length < 1e-4f) continue;

                if (primitive.kind == RoadPrimitive::Kind::Straight) {

                    // A slab whose TOP FACE is the road: the box is dropped
                    // half its thickness along its own normal, and butts
                    // against its neighbours rather than overlapping them.
                    Vector3 travel;
                    travel.copy(primitive.end).sub(primitive.start).normalize();
                    // The width axis is horizontal whatever the grade — a road
                    // rises with the slope but never banks.
                    Vector3 across{-travel.z, 0.f, travel.x};// cross(travel, up)
                    if (across.length() < 1e-6f) across.set(0.f, 0.f, 1.f);
                    across.normalize();
                    Vector3 normal;
                    normal.copy(across).cross(travel).normalize();

                    Vector3 center;
                    center.copy(primitive.start).add(primitive.end).multiplyScalar(0.5f);
                    center.addScaledVector(normal, -kRoadThickness * 0.5f);

                    if (uniform) {
                        Matrix4 basis;
                        basis.makeBasis(travel, normal, across);
                        Quaternion orientation;
                        orientation.setFromRotationMatrix(basis);
                        attach(PxBoxGeometry(length * 0.5f * s.x,
                                             kRoadThickness * 0.5f * s.x,
                                             half * s.x),
                               PxTransform(scaled(center), toPxQuat(orientation)));
                    } else {
                        PxVec3 hull[8];
                        int n = 0;
                        for (float u : {-length * 0.5f, length * 0.5f}) {
                            for (float v : {-half, half}) {
                                for (float w : {-kRoadThickness * 0.5f, kRoadThickness * 0.5f}) {
                                    Vector3 corner;
                                    corner.copy(center)
                                            .addScaledVector(travel, u)
                                            .addScaledVector(across, v)
                                            .addScaledVector(normal, w);
                                    hull[n++] = scaled(corner);
                                }
                            }
                        }
                        attachHull(hull, 8);
                    }
                    continue;
                }

                // The bend. Wedges tile the annulus between max(R - w/2, 0) and
                // R + w/2 with no gap and no overlap, since neighbours share a
                // radial face exactly. Where the bend is tighter than the half
                // width that inner radius is zero and the wedge is a pie slice
                // — six points rather than eight, and still convex.
                const float inner = std::max(primitive.radius - half, 0.f);
                const float outer = primitive.radius + half;
                const int steps = std::max(
                        1, static_cast<int>(std::ceil(std::abs(primitive.sweep) / kRoadWedgeStep)));
                for (int k = 0; k < steps; ++k) {

                    PxVec3 hull[8];
                    int n = 0;
                    // A pie slice's two inner points are the same point, and a
                    // hull with a duplicate in it is one PhysX has to clean up.
                    const auto push = [&hull, &n](const PxVec3& point) {
                        for (int q = 0; q < n; ++q) {
                            if ((hull[q] - point).magnitudeSquared() < 1e-12f) return;
                        }
                        hull[n++] = point;
                    };
                    for (int e = 0; e < 2; ++e) {
                        const float t = static_cast<float>(k + e) / static_cast<float>(steps);
                        const float angle = primitive.startAngle + primitive.sweep * t;
                        const float y = primitive.start.y + (primitive.end.y - primitive.start.y) * t;
                        const float cs = std::cos(angle), sn = std::sin(angle);
                        for (float radius : {inner, outer}) {
                            const Vector3 top{primitive.center.x + radius * cs, y,
                                              primitive.center.z + radius * sn};
                            push(scaled(top));
                            push(scaled(Vector3(top.x, top.y - kRoadThickness, top.z)));
                        }
                    }
                    attachHull(hull, static_cast<PxU32>(n));
                }
            }

            if (shapes == 0) {
                body->release();
                return false;
            }
            world_->scene().addActor(*body);
            return true;
        }

        // Cook the descendants of a geometry-less static body as its collider.
        // Meshes carrying their own enabled PhysicsConfig are left alone —
        // start() is creating their actors — and a derived road goes through
        // the primitive chain above rather than a triangle mesh, so it collides
        // the same whether the config sat on it or on the group over it.
        std::size_t addSubtree(Object3D& root, ::physx::PxMaterial* material) {

            std::size_t added = 0;

            root.updateMatrixWorld();
            root.traverseType<Mesh>([&](Mesh& mesh) {
                if (&mesh == &root || ownsActor(mesh) || !mesh.geometry()) return;
                if (!isDerivedRoad(mesh)) return;
                if (addRoadPrimitives(mesh, material)) ++added;
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

            // A spline-derived road collides as its primitive chain whichever
            // of the mesh shapes was asked for — Auto, TriMesh and Convex all
            // mean "the surface I can see" here, and the chain is the only one
            // of the three that is exact AND safe to rest a body on. An
            // explicit primitive is left alone; so is a moving road, which is a
            // plank rather than a road and keeps the whole-mesh hull.
            if (mesh && !moving && isDerivedRoad(*mesh) &&
                (shape == PhysicsConfig::Shape::TriMesh || shape == PhysicsConfig::Shape::Convex)) {
                if (addRoadPrimitives(*mesh, material)) return true;
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
