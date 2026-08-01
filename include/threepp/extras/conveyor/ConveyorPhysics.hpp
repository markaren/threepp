// Belt physics for conveyor specs: kinematic colliders driven by the
// fake-velocity trick, built into a borrowed PhysxWorld.
//
// Header-only and PhysX-dependent, exactly like PhysxWorld itself — the threepp
// library proper never links PhysX, so this file is included only by builds
// that found the SDK. The PhysX-free half (path model, spec, visual geometry)
// is ConveyorGeometry.hpp; this class consumes the SAME resampled path, so the
// colliders always match what is drawn.
//
// How a belt conveys: every straight run is a chain of kinematic boxes whose
// top face sits on the path. Each physics substep the box's kinematic target is
// advanced along travel (giving every contact the belt's surface velocity), and
// after the substep the box is teleported back to where it was — the surface
// never actually moves, but everything resting on it is dragged along. That
// works for rigid bodies AND for GPU-simulated deformable volumes (soft
// bodies), which is the reason the belts are kinematic dynamics rather than
// statics with a material velocity.
//
// A horizontal bend (an arc-centre waypoint) cannot be dragged linearly —
// every point needs its own direction — so the whole bend is ONE kinematic body
// ROTATING about the vertical axis through the arc centre, its surface tiled by
// convex annular wedges that share their radial faces (no overlap, no gap).
// The wedges are cooked GPU-compatible so deformables collide with them.
//
// Cleats are the exception to the teleport-back trick: a teleported-back wall
// un-does its push, so cleat bars GENUINELY travel along the path (kinematic
// targets marching forward), wrapping end→start with a plain teleport. The
// caller mirrors whatever visuals it owns onto cleatPose().
//
// The world is BORROWED. Whoever owns this object must destroy it while the
// world is still alive (the destructor unregisters its substep hooks and
// releases its actors), or call abandon() first when the world is already gone.

#ifndef THREEPP_CONVEYOR_PHYSICS_HPP
#define THREEPP_CONVEYOR_PHYSICS_HPP

#include "threepp/extras/conveyor/ConveyorGeometry.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace threepp::conveyor {

    class ConveyorPhysics {

    public:
        // Specs are consumed in WORLD space — resolve an authored conveyor's
        // waypoints through its matrixWorld before handing them over. Arc bends
        // assume the waypoint plane is horizontal (a yawed / translated conveyor
        // is fine; a pitched or rolled one distorts its bends).
        ConveyorPhysics(PhysxWorld& world, std::vector<ConveyorSpec> specs)
            : world_(&world), specs_(std::move(specs)) {

            // High-friction, low-restitution belt surface: contact friction is
            // the combination of both bodies' materials, and the belt side must
            // be grippy for anything to convey up a slope.
            beltMaterial_ = world_->physics().createMaterial(1.0f, 1.0f, 0.1f);

            for (std::size_t i = 0; i < specs_.size(); ++i) {
                const auto& spec = specs_[i];
                if (spec.waypoints.size() < 2) continue;
                if (spec.separator) {
                    buildWall(spec);
                } else {
                    buildBelts(spec, i);
                }
            }

            preHandle_ = world_->onPreSubstep([this](float dt) { preSubstep(dt); });
            postHandle_ = world_->onPostSubstep([this](float) { postSubstep(); });
        }

        ConveyorPhysics(const ConveyorPhysics&) = delete;
        ConveyorPhysics& operator=(const ConveyorPhysics&) = delete;

        ~ConveyorPhysics() {

            if (!world_) return;
            world_->removeSubstepCallback(preHandle_);
            world_->removeSubstepCallback(postHandle_);
            for (auto* actor : actors_) {
                world_->scene().removeActor(*actor);
                actor->release();
            }
        }

        // Give up every reference to the world WITHOUT touching it — for the
        // teardown path where the world is already destroyed (its actors and
        // callbacks died with it).
        void abandon() { world_ = nullptr; }

        // Global multiplier on every belt/cleat speed — live, no rebuild.
        float speedScale = 1.f;

        [[nodiscard]] const std::vector<ConveyorSpec>& specs() const { return specs_; }

        // Belt colliders built (straight segments + bend bodies + walls).
        [[nodiscard]] std::size_t beltCount() const { return belts_.size(); }

        // --- Cleat tracks, for visual mirroring ------------------------------
        //
        // The bars are colliders only; whoever owns visuals reads their poses
        // each frame. cleatCount is fixed for the track's lifetime.

        [[nodiscard]] std::size_t trackCount() const { return tracks_.size(); }

        // Which spec (constructor order) the track belongs to — bar dimensions
        // (cleatHeight × width) come from there.
        [[nodiscard]] std::size_t trackSpecIndex(std::size_t track) const {
            return tracks_[track].specIndex;
        }

        [[nodiscard]] std::size_t cleatCount(std::size_t track) const {
            return tracks_[track].cleats.size();
        }

        void cleatPose(std::size_t track, std::size_t i, Vector3& position,
                       Quaternion& rotation) const {

            const auto pose = tracks_[track].cleats[i].actor->getGlobalPose();
            position.set(pose.p.x, pose.p.y, pose.p.z);
            rotation.set(pose.q.x, pose.q.y, pose.q.z, pose.q.w);
        }

    private:
        static constexpr float kBeltThick = 0.08f;// belt collider thickness
        static constexpr float kWallThick = 0.04f;// separator wall thickness

        // A kinematic belt collider driven by the fake-velocity trick.
        struct Belt {
            ::physx::PxRigidDynamic* actor = nullptr;
            bool rotational = false;
            ::physx::PxVec3 velocity{0, 0, 0};// straight: world m/s
            float omega = 0.f;                // bend: rad/s about the actor's own +Y (arc centre)
            ::physx::PxTransform saved{::physx::PxIdentity};
        };

        // One travelling cleat bar.
        struct MovingCleat {
            ::physx::PxRigidDynamic* actor = nullptr;
            float offset = 0.f; // base arc-length along the track
            float prevS = -1.f; // last position; a backward jump = a wrap (teleport, no push)
        };

        // A cleated run: bars that genuinely travel along the centerline.
        struct CleatTrack {
            std::vector<Vector3> poly;// travel-ordered centerline the cleats ride
            float length = 0.f;
            float speed = 0.f;
            float height = 0.f;
            float rampLen = 0.f;// distance over which a bar folds flat at each end
            float phase = 0.f;  // advances with time, wrapped to [0, length)
            std::size_t specIndex = 0;
            std::vector<MovingCleat> cleats;
        };

        ::physx::PxRigidDynamic* makeKinematic(const ::physx::PxTransform& pose) {

            auto* actor = world_->physics().createRigidDynamic(pose);
            actor->setRigidBodyFlag(::physx::PxRigidBodyFlag::eKINEMATIC, true);
            return actor;
        }

        void commit(::physx::PxRigidDynamic* actor, ::physx::PxShape* shape) {

            actor->attachShape(*shape);
            shape->release();
            world_->scene().addActor(*actor);
            actors_.push_back(actor);
        }

        // One straight drag-segment a→b: a kinematic box whose top face sits on
        // the path. Boxes butt end-to-end (no length overlap): at a convex
        // crest an overhanging box would poke up through the next one.
        void addStraightSeg(const Vector3& a, const Vector3& b, float width, float speed) {

            using namespace ::physx;

            Vector3 d(b.x - a.x, b.y - a.y, b.z - a.z);
            const float len = d.length();
            if (len < 1e-3f) return;
            d.multiplyScalar(1.f / len);// full 3D travel direction (incl. slope)
            const PxQuat q = toPxQuat(segmentOrientation(a, b));
            const PxVec3 nrm = q.rotate(PxVec3(0, 1, 0));// offset down along the belt normal
            const PxVec3 center((a.x + b.x) * 0.5f - nrm.x * kBeltThick * 0.5f,
                                (a.y + b.y) * 0.5f - nrm.y * kBeltThick * 0.5f,
                                (a.z + b.z) * 0.5f - nrm.z * kBeltThick * 0.5f);
            auto* actor = makeKinematic(PxTransform(center, q));
            PxShape* shape = world_->physics().createShape(
                    PxBoxGeometry(len * 0.5f, kBeltThick * 0.5f, width * 0.5f), *beltMaterial_, true);
            commit(actor, shape);
            belts_.push_back({actor, false, PxVec3(d.x * speed, d.y * speed, d.z * speed), 0.f,
                              PxTransform(PxIdentity)});
        }

        // A horizontal bend A→B around centre C as ONE kinematic body rotating
        // about the vertical axis through C, tiled by convex annular wedges
        // (see the header note). omega = speed / radius, signed so the surface
        // flows A→B (a +Y turn moves a surface point to a LOWER path angle,
        // hence the minus).
        void addArcBelt(const Vector3& A, const Vector3& C, const Vector3& B,
                        float width, float speed, const Vector3& incoming) {

            using namespace ::physx;

            const float PI = static_cast<float>(math::PI);
            const Arc arc = computeArc(A, C, B, incoming);
            const float radius = 0.5f * (arc.radA + arc.radB);
            if (!arc.valid || radius < 1e-3f) {// degenerate — fall back to a straight chord
                addStraightSeg(A, B, width, speed);
                return;
            }
            const float deckY = 0.5f * (A.y + B.y);// horizontal bend
            auto* actor = makeKinematic(
                    PxTransform(PxVec3(C.x, deckY, C.z)));// identity rot: local +Y = world up

            // Cooked GPU-compatible so the wedges collide with GPU-simulated
            // deformable volumes (soft bodies) too.
            PxCookingParams cookParams(world_->physics().getTolerancesScale());
            cookParams.buildGPUData = true;
            const float halfW = width * 0.5f;
            const int steps = std::max(3, static_cast<int>(std::ceil(std::abs(arc.sweep) / (PI / 12.f))));
            bool any = false;
            for (int k = 0; k < steps; ++k) {
                // Convex wedge spanning [t0,t1]: 8 verts = {t0,t1} x {inner,outer}
                // x {top,bottom}, local to the actor origin (C at deckY).
                // Adjacent wedges share their radial face exactly.
                PxVec3 v[8];
                int n = 0;
                for (int e = 0; e < 2; ++e) {
                    const float t = static_cast<float>(k + e) / static_cast<float>(steps);
                    const float ang = arc.a0 + arc.sweep * t;
                    const float rc = arc.radA + (arc.radB - arc.radA) * t;
                    const float yTop = (A.y + (B.y - A.y) * t) - deckY;
                    const float cs = std::cos(ang), sn = std::sin(ang);
                    const float innerR = std::max(rc - halfW, 0.02f);
                    const float outerR = rc + halfW;
                    for (float rr : {innerR, outerR}) {
                        v[n++] = PxVec3(rr * cs, yTop, rr * sn);             // top
                        v[n++] = PxVec3(rr * cs, yTop - kBeltThick, rr * sn);// bottom
                    }
                }
                PxConvexMeshDesc cd;
                cd.points.count = 8;
                cd.points.stride = sizeof(PxVec3);
                cd.points.data = v;
                cd.flags = PxConvexFlag::eCOMPUTE_CONVEX | PxConvexFlag::eGPU_COMPATIBLE;
                PxConvexMesh* cm = PxCreateConvexMesh(cookParams, cd,
                                                      world_->physics().getPhysicsInsertionCallback());
                if (!cm) continue;
                PxShape* shape = world_->physics().createShape(PxConvexMeshGeometry(cm),
                                                               *beltMaterial_, true);
                actor->attachShape(*shape);
                shape->release();
                any = true;
            }
            if (!any) {// every wedge failed to cook — keep the path conveying
                actor->release();
                addStraightSeg(A, B, width, speed);
                return;
            }
            world_->scene().addActor(*actor);
            actors_.push_back(actor);
            const float omega = -std::copysign(speed / radius, arc.sweep);
            belts_.push_back({actor, true, PxVec3(0, 0, 0), omega, PxTransform(PxIdentity)});
        }

        // Belt colliders along the waypoint path: straight runs become
        // linearly-dragged boxes; each rounded corner becomes one
        // rotationally-driven bend body built from the SAME fillet the drawn
        // ribbon samples (cornerFillet), so collider and visual agree by
        // construction. `reverse` flips the whole path.
        void buildBelts(const ConveyorSpec& spec, std::size_t specIndex) {

            const float speed = spec.speed;
            bool corners = false;
            for (std::size_t i = 1; i + 1 < spec.waypoints.size(); ++i) {
                if (spec.waypoints[i].cornerRadius > 1e-4f) corners = true;
            }

            if (!corners) {
                // No bends: coarse straight segments along the (spline/raw)
                // centreline. A body spans several, so a curve needs far fewer
                // here than for a smooth-looking ribbon.
                auto pts = resamplePath(spec.waypoints, spec.smooth, 5);
                if (pts.size() >= 2) {
                    if (spec.reverse) std::reverse(pts.begin(), pts.end());
                    for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
                        addStraightSeg(pts[i], pts[i + 1], spec.width, speed);
                    }
                }
            } else {
                // Corner walk, mirroring resamplePath: straights between the
                // fillets' tangent points, one bend body per rounded corner.
                // Reversal flips travel by flipping the waypoint list; the
                // fillet geometry is direction-independent.
                std::vector<Waypoint> wps(spec.waypoints);
                if (spec.reverse) std::reverse(wps.begin(), wps.end());
                const std::size_t n = wps.size();
                Vector3 cursor = wps.front().pos;
                Vector3 incoming(0, 0, 0);
                for (std::size_t i = 1; i + 1 < n; ++i) {
                    const CornerFillet f = cornerFillet(wps, i);
                    if (!f.valid) {
                        addStraightSeg(cursor, wps[i].pos, spec.width, speed);
                        incoming.set(wps[i].pos.x - cursor.x, 0.f, wps[i].pos.z - cursor.z);
                        cursor = wps[i].pos;
                        continue;
                    }
                    addStraightSeg(cursor, f.t1, spec.width, speed);
                    incoming.set(f.t1.x - cursor.x, 0.f, f.t1.z - cursor.z);
                    addArcBelt(f.t1, f.centre, f.t2, spec.width, speed, incoming);
                    incoming.set(f.t2.x - f.t1.x, 0.f, f.t2.z - f.t1.z);
                    cursor = f.t2;
                }
                addStraightSeg(cursor, wps.back().pos, spec.width, speed);
            }

            // Travelling cleat bars per cleats-run. The box-belt colliders above
            // span the whole path regardless, so the surface conveys everywhere;
            // the bars add the pushing barrier an incline needs.
            if (spec.cleatHeight > 1e-3f && spec.cleatSpacing > 1e-3f) {
                for (const auto& run : resamplePathByKind(spec.waypoints, spec.smooth, spec.samples)) {
                    if (run.kind != SegKind::Cleats || run.pts.size() < 2) continue;
                    buildCleatTrack(run.pts, spec, specIndex);
                }
            }
        }

        void buildCleatTrack(const std::vector<Vector3>& pts, const ConveyorSpec& spec,
                             std::size_t specIndex) {

            using namespace ::physx;

            CleatTrack track;
            track.poly = pts;
            if (spec.reverse) std::reverse(track.poly.begin(), track.poly.end());// travel-ordered
            track.speed = spec.speed;
            track.height = spec.cleatHeight;
            track.specIndex = specIndex;
            for (std::size_t i = 0; i + 1 < track.poly.size(); ++i) {
                track.length += track.poly[i].distanceTo(track.poly[i + 1]);
            }
            if (track.length < 1e-3f) return;
            // Fold the bars flat within this distance of each end (a closed band
            // wraps around its pulleys flush with the belt).
            track.rampLen = std::min(2.f * spec.cleatHeight, 0.45f * track.length);

            // Evenly tiled so the spacing stays uniform across the wrap seam.
            for (float s : cleatOffsets(track.length, spec.cleatSpacing)) {
                Vector3 center;
                Quaternion q;
                const float fold = cleatFold(s, track.length, track.rampLen);
                cleatPoseAt(track.poly, s, spec.cleatHeight, fold, center, q);
                auto* actor = makeKinematic(toPxTransform(center, q));
                PxShape* shape = world_->physics().createShape(
                        PxBoxGeometry(kCleatThickness * 0.5f, spec.cleatHeight * 0.5f,
                                      spec.width * 0.5f),
                        *beltMaterial_, true);
                commit(actor, shape);
                track.cleats.push_back({actor, s, -1.f});
            }
            if (!track.cleats.empty()) tracks_.push_back(std::move(track));
        }

        // Separator: a collision-only vertical wall along the centerline —
        // static thin boxes per segment. Kinematic (like the belts) so it
        // reliably collides with GPU-simulated deformables, but never advanced,
        // so it stays put.
        void buildWall(const ConveyorSpec& spec) {

            using namespace ::physx;

            const auto pts = resamplePath(spec.waypoints, spec.smooth, 5);
            const float height = spec.wallHeight;
            for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
                const Vector3& a = pts[i];
                const Vector3& b = pts[i + 1];
                const Vector3 d(b.x - a.x, b.y - a.y, b.z - a.z);
                const float len = d.length();
                if (len < 1e-3f) continue;
                const PxQuat q = toPxQuat(segmentOrientation(a, b));
                const PxVec3 up = q.rotate(PxVec3(0, 1, 0));// frame normal — vertical on a flat run
                const PxVec3 center((a.x + b.x) * 0.5f + up.x * height * 0.5f,
                                    (a.y + b.y) * 0.5f + up.y * height * 0.5f,
                                    (a.z + b.z) * 0.5f + up.z * height * 0.5f);
                auto* actor = makeKinematic(PxTransform(center, q));
                PxShape* shape = world_->physics().createShape(
                        PxBoxGeometry(len * 0.5f, height * 0.5f, kWallThick * 0.5f),
                        *beltMaterial_, true);
                commit(actor, shape);
            }
        }

        // Fake surface velocity around each substep: straight belts translate
        // their kinematic target, bends rotate it about the arc centre; both
        // teleport back afterwards. Cleats genuinely travel (and are NOT
        // restored): a forward step uses setKinematicTarget so the contact gets
        // a push velocity; the end→start wrap uses setGlobalPose so it
        // teleports without flinging whatever it lands near.
        void preSubstep(float dt) {

            using namespace ::physx;

            for (auto& b : belts_) {
                b.saved = b.actor->getGlobalPose();
                PxTransform target = b.saved;
                if (b.rotational) {
                    target.q = b.saved.q * PxQuat(b.omega * speedScale * dt, PxVec3(0, 1, 0));
                } else {
                    target.p += b.velocity * (speedScale * dt);
                }
                b.actor->setKinematicTarget(target);
            }

            for (auto& tr : tracks_) {
                if (tr.length < 1e-4f) continue;
                tr.phase = std::fmod(tr.phase + tr.speed * speedScale * dt, tr.length);
                if (tr.phase < 0.f) tr.phase += tr.length;// negative speed still wraps into range
                for (auto& cl : tr.cleats) {
                    float s = std::fmod(cl.offset + tr.phase, tr.length);
                    if (s < 0.f) s += tr.length;
                    Vector3 center;
                    Quaternion q;
                    const float fold = cleatFold(s, tr.length, tr.rampLen);
                    cleatPoseAt(tr.poly, s, tr.height, fold, center, q);
                    const PxTransform pose = toPxTransform(center, q);
                    if (cl.prevS < 0.f || s < cl.prevS) cl.actor->setGlobalPose(pose);
                    else cl.actor->setKinematicTarget(pose);
                    cl.prevS = s;
                }
            }
        }

        void postSubstep() {

            for (auto& b : belts_) b.actor->setGlobalPose(b.saved);
        }

        PhysxWorld* world_;
        std::vector<ConveyorSpec> specs_;
        ::physx::PxMaterial* beltMaterial_ = nullptr;
        PhysxWorld::SubstepHandle preHandle_ = 0, postHandle_ = 0;

        std::vector<::physx::PxRigidDynamic*> actors_;// everything built, for teardown
        std::vector<Belt> belts_;
        std::vector<CleatTrack> tracks_;
    };

}// namespace threepp::conveyor

#endif// THREEPP_CONVEYOR_PHYSICS_HPP
