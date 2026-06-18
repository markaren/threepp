#include "ConveyorSystem.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace threepp;
using namespace ::physx;

namespace fs = std::filesystem;

namespace {

    constexpr float beltThick = 0.08f;// belt collider thickness
    constexpr float wallThick = 0.04f;// separator wall thickness

    // Point a given arc-length distance into a (travel-ordered) polyline; clamps to the
    // last point for short paths. Used to spawn fish a bit onto the belt, not at its edge.
    Vector3 pointAlong(const std::vector<Vector3>& path, float dist) {
        if (path.empty()) return Vector3();
        float acc = 0.f;
        for (size_t i = 0; i + 1 < path.size(); ++i) {
            const float seg = path[i].distanceTo(path[i + 1]);
            if (acc + seg >= dist) {
                Vector3 r;
                r.lerpVectors(path[i], path[i + 1], (dist - acc) / std::max(seg, 1e-6f));
                return r;
            }
            acc += seg;
        }
        return path.back();
    }

}// namespace

namespace fishsim {

    ConveyorSystem::ConveyorSystem(PhysxWorld& world, Scene& scene, fs::path convDir)
        : world_(world), scene_(scene), convDir_(std::move(convDir)) {

        // High-friction rigid material for the belt colliders. Contact friction with a
        // fish is the combination of this and the fish's deformable material, so the belt
        // side must be grippy too (esp. for slopes).
        beltMat_ = world_.physics().createMaterial(1.0f, 1.0f, 0.1f);

        // Visible belt surface for waypoint PATHS (the curved colliders are separate; this
        // ribbon is just the visual). It carries a scrolling modular-belt texture to fake
        // belt motion, so the material colour is white and the texture supplies the look;
        // each path clones the material + texture so belts scroll independently.
        beltVisualMat_ = MeshStandardMaterial::create();
        beltVisualMat_->color = Color(0xffffff);
        beltVisualMat_->roughness = 0.85f;
        beltVisualMat_->metalness = 0.f;
        beltVisualMat_->side = Side::Double;

        // Procedural modular-belt texture: a transverse groove per tile (reads clearly as
        // motion when scrolled along travel) plus a thin longitudinal module line. Tiled via
        // Repeat wrap; cloned per path for independent scroll offsets.
        beltTexture_ = DataTexture::create<unsigned char>(4, 64, 64);
        {
            auto& d = beltTexture_->image().data<unsigned char>();
            for (int y = 0; y < 64; ++y) {
                for (int x = 0; x < 64; ++x) {
                    const bool groove = (y < 5) || (x < 2);
                    const int i = (y * 64 + x) * 4;
                    d[i + 0] = groove ? 22 : 66;
                    d[i + 1] = groove ? 24 : 72;
                    d[i + 2] = groove ? 30 : 82;
                    d[i + 3] = 255;
                }
            }
        }
        beltTexture_->wrapS = TextureWrapping::Repeat;
        beltTexture_->wrapT = TextureWrapping::Repeat;
        beltTexture_->magFilter = Filter::Linear;
        beltTexture_->minFilter = Filter::Linear;
        beltTexture_->generateMipmaps = false;
        beltTexture_->colorSpace = ColorSpace::sRGB;
        beltTexture_->needsUpdate();

        // Roller-conveyor visuals: one rotating bank of cylinders per rollers run. Spun each
        // frame so the rollers turn in step with the kinematic belt drag underneath them (the
        // collider stays a continuous box surface; the rollers are the visible bed on top).
        rollerVisualMat_ = MeshStandardMaterial::create();
        rollerVisualMat_->color = Color(0x8a9097);
        rollerVisualMat_->roughness = 0.4f;
        rollerVisualMat_->metalness = 0.7f;
        rollerVisualMat_->flatShading = true;

        // Cleated belts: thin bars standing across the belt that GENUINELY travel along the
        // path at belt speed (wrapping at the ends), so they catch fish on an incline and carry
        // them up. The fake-velocity belt trick teleports its collider back each substep, which
        // works for friction (tangential drag) but not for a pushing barrier (a teleported-back
        // wall un-does its push), so cleats move for real instead.
        cleatVisualMat_ = MeshStandardMaterial::create();
        cleatVisualMat_->color = Color(0x202428);
        cleatVisualMat_->roughness = 0.6f;
        cleatVisualMat_->metalness = 0.3f;

        // Separator: a collision-only vertical wall along a waypoint centerline (guide rail /
        // lane divider) — a thin double-sided visual so it's placeable.
        wallVisualMat_ = MeshStandardMaterial::create();
        wallVisualMat_->color = Color(0xbfc6cc);
        wallVisualMat_->roughness = 0.7f;
        wallVisualMat_->metalness = 0.f;
        wallVisualMat_->transparent = true;
        wallVisualMat_->opacity = 0.55f;
        wallVisualMat_->side = Side::Double;

        typeOverrides_ = conveyor::loadTypeOverrides(convDir_ / "conveyor_types.json");

        load();
    }

    conveyor::ModelTemplate& ConveyorSystem::convTemplate(const std::string& name) {
        auto it = convTemplates_.find(name);
        if (it == convTemplates_.end()) {
            it = convTemplates_.emplace(name, conveyor::prepareModel(usdLoader_, convDir_ / (name + ".usd"))).first;
            if (auto ov = typeOverrides_.find(name); ov != typeOverrides_.end()) {
                it->second.kind = (ov->second == "curve") ? conveyor::BeltKind::Curve
                                                          : conveyor::BeltKind::Straight;
            }
        }
        return it->second;
    }

    void ConveyorSystem::addConveyorVisual(const conveyor::Piece& piece) {
        auto& t = convTemplate(piece.model);
        if (!t.group) {
            std::cerr << "Conveyor model missing: " << piece.model << std::endl;
            return;
        }
        auto vis = t.group->clone();
        vis->position.copy(piece.position);
        vis->rotation.set(0.f, math::degToRad(piece.yawDeg), 0.f);
        vis->scale.set(piece.scale, piece.scale, piece.scale);
        scene_.add(vis);
    }

    // Per-piece straight belt — used only when a layout has no waypoint path.
    void ConveyorSystem::addStraightBelt(const conveyor::Piece& piece) {
        if (piece.beltSpeed == 0.f) return;
        auto& t = convTemplate(piece.model);
        if (!t.group) return;
        const float yaw = math::degToRad(piece.yawDeg);
        const float deckWorldTop = piece.position.y + t.deckTopY * piece.scale;
        const float sx = std::max(t.deckSize.x * piece.scale, 0.2f);
        const float sz = std::max(t.deckSize.z * piece.scale, 0.2f);
        const PxQuat q(yaw, PxVec3(0, 1, 0));
        const PxVec3 center(piece.position.x, deckWorldTop - beltThick * 0.5f, piece.position.z);
        PxRigidDynamic* actor = world_.physics().createRigidDynamic(PxTransform(center, q));
        actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
        PxShape* shape = world_.physics().createShape(
                PxBoxGeometry(sx * 0.5f, beltThick * 0.5f, sz * 0.5f), *beltMat_, true);
        actor->attachShape(*shape);
        shape->release();
        world_.scene().addActor(*actor);
        const PxVec3 vel = q.rotate(PxVec3(piece.beltSpeed, 0, 0));
        belts_.push_back({actor, false, vel, 0.f, PxTransform(PxIdentity)});
        PxVec3 dir = vel;
        if (dir.magnitude() > 1e-4f) dir = dir.getNormalized();
        inlets_.emplace_back(center.x - dir.x * (sx * 0.4f), deckWorldTop + 0.35f, center.z - dir.z * (sx * 0.4f));
    }

    // One straight drag-segment a→b: a kinematic box whose top face sits on the path,
    // dragged LINEARLY along its travel direction (the fake-velocity trick). Boxes butt
    // end-to-end (no length overlap): at a convex crest an overhanging box would poke up
    // through the next and hit the fish.
    void ConveyorSystem::addStraightSeg(const Vector3& a, const Vector3& b, float width, float speed) {
        Vector3 d(b.x - a.x, b.y - a.y, b.z - a.z);
        const float len = d.length();
        if (len < 1e-3f) return;
        d.multiplyScalar(1.f / len);// full 3D travel direction (incl. slope)
        const PxQuat q = toPxQuat(conveyor::segmentOrientation(a, b));
        const PxVec3 nrm = q.rotate(PxVec3(0, 1, 0));// offset down along the belt normal
        const PxVec3 center((a.x + b.x) * 0.5f - nrm.x * beltThick * 0.5f,
                            (a.y + b.y) * 0.5f - nrm.y * beltThick * 0.5f,
                            (a.z + b.z) * 0.5f - nrm.z * beltThick * 0.5f);
        PxRigidDynamic* actor = world_.physics().createRigidDynamic(PxTransform(center, q));
        actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
        PxShape* shape = world_.physics().createShape(
                PxBoxGeometry(len * 0.5f, beltThick * 0.5f, width * 0.5f), *beltMat_, true);
        actor->attachShape(*shape);
        shape->release();
        world_.scene().addActor(*actor);
        belts_.push_back({actor, false, PxVec3(d.x * speed, d.y * speed, d.z * speed), 0.f, PxTransform(PxIdentity)});
    }

    // A horizontal bend A→B around centre C as ONE kinematic body that ROTATES about the
    // vertical axis through C — the rotational analogue of the straight belt. PhysX can't
    // drag a curved belt linearly (every point needs its own direction), so the whole bend is
    // one rigid body we rotate. Its surface is tiled by CONVEX ANNULAR WEDGES (radial side
    // faces) rather than rectangular boxes: wedges share each radial face exactly, tiling the
    // ring with no overlap and no gap (wide rectangles would mutually overlap on a tight bend).
    // Cooked GPU-compatible so they collide with the GPU-simulated fish. omega = beltSpeed/
    // radius, signed so the surface flows A→B (onPreSubstep rotates about local +Y; a +Y turn
    // moves a surface point to a LOWER path-angle, hence the minus).
    void ConveyorSystem::addArcBelt(const Vector3& A, const Vector3& C, const Vector3& B,
                                    float width, float speed, const Vector3& incoming) {
        const float PI = static_cast<float>(math::PI);
        const conveyor::Arc arc = conveyor::computeArc(A, C, B, incoming);
        const float radius = 0.5f * (arc.radA + arc.radB);
        if (!arc.valid || radius < 1e-3f) {// degenerate — fall back to a straight chord
            addStraightSeg(A, B, width, speed);
            return;
        }
        const float deckY = 0.5f * (A.y + B.y);// horizontal bend
        PxRigidDynamic* actor = world_.physics().createRigidDynamic(
                PxTransform(PxVec3(C.x, deckY, C.z)));// identity rot: local +Y = world up
        actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);

        PxCookingParams cookParams(world_.physics().getTolerancesScale());
        cookParams.buildGPUData = true;// deformable volumes are GPU-simulated
        const float halfW = width * 0.5f;
        const int steps = std::max(3, static_cast<int>(std::ceil(std::abs(arc.sweep) / (PI / 12.f))));
        for (int k = 0; k < steps; ++k) {
            // Convex wedge spanning [t0,t1]: 8 verts = {t0,t1}×{inner,outer}×{top,bottom},
            // local to the actor origin (C at deckY). Adjacent wedges share their radial
            // face, so they tile the ring with no overlap and no gap.
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
                    v[n++] = PxVec3(rr * cs, yTop, rr * sn);            // top
                    v[n++] = PxVec3(rr * cs, yTop - beltThick, rr * sn);// bottom
                }
            }
            PxConvexMeshDesc cd;
            cd.points.count = 8;
            cd.points.stride = sizeof(PxVec3);
            cd.points.data = v;
            cd.flags = PxConvexFlag::eCOMPUTE_CONVEX | PxConvexFlag::eGPU_COMPATIBLE;
            PxConvexMesh* cm = PxCreateConvexMesh(cookParams, cd, world_.physics().getPhysicsInsertionCallback());
            if (!cm) continue;
            PxShape* shape = world_.physics().createShape(PxConvexMeshGeometry(cm), *beltMat_, true);
            actor->attachShape(*shape);
            shape->release();
        }
        world_.scene().addActor(*actor);
        const float omega = -std::copysign(speed / radius, arc.sweep);
        belts_.push_back({actor, true, PxVec3(0, 0, 0), omega, PxTransform(PxIdentity)});
    }

    // Build belt colliders along an explicit waypoint path: straight runs become linearly-
    // dragged boxes; arc-CENTRE waypoints each become one rotationally-driven bend body.
    // `reverse` flips the whole path (travel direction + inlet).
    void ConveyorSystem::buildPathBelts(const std::vector<conveyor::Waypoint>& ctrl, float width, float speed,
                                        bool reverse, bool smooth, float rollerRadius,
                                        float cleatHeight, float cleatSpacing) {
        bool hasArc = false;
        for (const auto& w : ctrl) {
            if (w.arcCenter) {
                hasArc = true;
                break;
            }
        }

        if (!hasArc) {
            // No arcs: coarse straight segments along the (spline/raw) centreline. A fish
            // spans several, so a curve needs far fewer here than for a smooth-looking line.
            auto pts = conveyor::resamplePath(ctrl, smooth, 5);
            if (pts.size() >= 2) {
                if (reverse) std::ranges::reverse(pts);
                for (size_t i = 0; i + 1 < pts.size(); ++i) addStraightSeg(pts[i], pts[i + 1], width, speed);
            }
        } else {
            // Walk waypoints (mirroring resamplePath): straight runs between regular points,
            // a rotational bend body at each arc centre. Reversing the waypoint list flips
            // travel + inlet (arc centres stay between their neighbours).
            std::vector<conveyor::Waypoint> wps(ctrl);
            if (reverse) std::ranges::reverse(wps);
            const int n = static_cast<int>(wps.size());
            bool haveLast = false;
            Vector3 lastPt, incoming(0, 0, 0);
            for (int i = 0; i < n;) {
                if (wps[i].arcCenter) {
                    if (!haveLast || i + 1 >= n || wps[i + 1].arcCenter) {
                        ++i;
                        continue;
                    }
                    const Vector3 A = lastPt, C = wps[i].pos, B = wps[i + 1].pos;
                    addArcBelt(A, C, B, width, speed, incoming);
                    incoming.set(B.x - A.x, 0.f, B.z - A.z);
                    lastPt = B;
                    haveLast = true;
                    i += 2;
                } else {
                    const Vector3 p = wps[i].pos;
                    if (haveLast) {
                        addStraightSeg(lastPt, p, width, speed);
                        incoming.set(p.x - lastPt.x, 0.f, p.z - lastPt.z);
                    }
                    lastPt = p;
                    haveLast = true;
                    ++i;
                }
            }
        }

        // Visible surface, PER SEGMENT kind (runs share boundary points so they meet gap-free).
        // Flat/cleats runs get a scrolling ribbon; rollers runs a rotating bank; cleats runs
        // additionally get moving flight bars (real, travelling colliders). The box-belt
        // colliders above span the whole path regardless, so fish convey on every run.
        for (const auto& run : conveyor::resamplePathByKind(ctrl, smooth)) {
            if (run.pts.size() < 2) continue;
            if (run.kind == conveyor::SegRollers) {
                // A row of cylinders across the belt, spun each frame so the bed reads as
                // powered. Tops sit on the centreline (where the box collider's top also is),
                // so fish appear to ride on the rollers. One shared geometry per bank.
                const float spacing = conveyor::rollerSpacing(rollerRadius);
                auto cyl = CylinderGeometry::create(rollerRadius, rollerRadius, width, 12);
                RollerBank bank;
                bank.omega = (reverse ? 1.f : -1.f) * speed / std::max(rollerRadius, 1e-3f);
                for (const auto& r : conveyor::rollerTransforms(run.pts, rollerRadius, spacing)) {
                    auto roller = Mesh::create(cyl, rollerVisualMat_);
                    roller->position.copy(r.center);
                    roller->quaternion.copy(r.orientation);
                    scene_.add(roller);
                    bank.rollers.push_back(roller);
                    bank.base.push_back(r.orientation);
                }
                if (!bank.rollers.empty()) rollerBanks_.push_back(std::move(bank));
            } else {
                // Flat or cleats: a scrolling ribbon (one clone per run so each scrolls on its
                // own); UV.v is arc length (metres), so scrolling offset.y drags the pattern.
                constexpr float tileLen = 0.25f;// metres of belt per texture tile
                auto tex = beltTexture_->clone();
                tex->wrapS = tex->wrapT = TextureWrapping::Repeat;// ensure tiling survives clone
                tex->minFilter = tex->magFilter = Filter::Linear;
                tex->generateMipmaps = false;
                tex->colorSpace = ColorSpace::sRGB;
                tex->repeat.set(std::max(1.f, std::round(width / tileLen)), 1.f / tileLen);
                auto mat = beltVisualMat_->clone<MeshStandardMaterial>();
                mat->map = tex;
                scene_.add(Mesh::create(conveyor::ribbonGeometry(run.pts, width), mat));
                // offset.y is in tile units → scroll rate = speed[m/s] * repeat.y[tiles/m];
                // negative drags the pattern along travel, flipped for a reversed belt.
                beltVisuals_.push_back({tex, mat, (reverse ? 1.f : -1.f) * speed * tex->repeat.y});

                // Cleats: bars riding this run's (travel-ordered) centreline. Each is a kinematic
                // box collider (raised across the belt) + a visual; both are advanced along the
                // run every substep (see preSubstep), wrapping at its ends.
                if (run.kind == conveyor::SegCleats && cleatHeight > 1e-3f && cleatSpacing > 1e-3f) {
                    CleatTrack track;
                    track.poly = run.pts;
                    if (reverse) std::ranges::reverse(track.poly);// travel-ordered
                    track.speed = speed;
                    track.height = cleatHeight;
                    for (size_t i = 0; i + 1 < track.poly.size(); ++i)
                        track.length += track.poly[i].distanceTo(track.poly[i + 1]);
                    // Fold the bars flat within this distance of each end (a closed band wraps
                    // around its pulleys flush with the belt).
                    track.rampLen = std::min(2.f * cleatHeight, 0.45f * track.length);
                    auto box = BoxGeometry::create(conveyor::kCleatThickness, cleatHeight, width);
                    // Evenly tiled so the spacing stays uniform across the wrap seam (the band
                    // is a closed loop; advancing by phase mod length keeps every gap equal).
                    for (float s : conveyor::cleatOffsets(track.length, cleatSpacing)) {
                        Vector3 center;
                        Quaternion q;
                        const float fold = conveyor::cleatFold(s, track.length, track.rampLen);
                        conveyor::cleatPoseAt(track.poly, s, cleatHeight, fold, center, q);
                        PxRigidDynamic* actor = world_.physics().createRigidDynamic(toPxTransform(center, q));
                        actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
                        PxShape* shape = world_.physics().createShape(
                                PxBoxGeometry(conveyor::kCleatThickness * 0.5f, cleatHeight * 0.5f, width * 0.5f),
                                *beltMat_, true);
                        actor->attachShape(*shape);
                        shape->release();
                        world_.scene().addActor(*actor);
                        auto bar = Mesh::create(box, cleatVisualMat_);
                        bar->position.copy(center);
                        bar->quaternion.copy(q);
                        scene_.add(bar);
                        track.cleats.push_back({actor, bar, s});
                    }
                    if (!track.cleats.empty()) cleatTracks_.push_back(std::move(track));
                }
            }
        }

        // Inlet: spawn a bit downstream of the upstream edge so fish land fully on the belt
        // instead of half-off it. Walk the travel-ordered path forward ~a fish length.
        auto visPts = conveyor::resamplePath(ctrl, smooth);
        if (visPts.size() >= 2) {
            std::vector<Vector3> travel(visPts);
            if (reverse) std::ranges::reverse(travel);
            const Vector3 in = pointAlong(travel, 0.6f);
            inlets_.emplace_back(in.x, in.y + 0.35f, in.z);
        }
    }

    // Separator: a collision-only vertical wall along a waypoint centerline (guide rail / lane
    // divider) — STATIC thin boxes per segment (no belt velocity), plus a thin double-sided
    // visual so it's placeable. Reuses the same waypoint system.
    void ConveyorSystem::buildPathWall(const std::vector<conveyor::Waypoint>& ctrl, float height, bool smooth) {
        const std::vector<Vector3> pts = conveyor::resamplePath(ctrl, smooth, 5);
        for (size_t i = 0; i + 1 < pts.size(); ++i) {
            const Vector3& a = pts[i];
            const Vector3& b = pts[i + 1];
            const Vector3 d(b.x - a.x, b.y - a.y, b.z - a.z);
            const float len = d.length();
            if (len < 1e-3f) continue;
            const PxQuat q = toPxQuat(conveyor::segmentOrientation(a, b));
            const PxVec3 up = q.rotate(PxVec3(0, 1, 0));// frame normal — vertical on a flat belt
            const PxVec3 center((a.x + b.x) * 0.5f + up.x * height * 0.5f,
                                (a.y + b.y) * 0.5f + up.y * height * 0.5f,
                                (a.z + b.z) * 0.5f + up.z * height * 0.5f);
            // Kinematic (like the belts) so it reliably collides with the GPU-simulated
            // deformable fish, but never added to belts_[] → its target is never advanced,
            // so it stays put: an immovable collision-only wall.
            PxRigidDynamic* actor = world_.physics().createRigidDynamic(PxTransform(center, q));
            actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
            PxShape* shape = world_.physics().createShape(
                    PxBoxGeometry(len * 0.5f, height * 0.5f, wallThick * 0.5f), *beltMat_, true);
            actor->attachShape(*shape);
            shape->release();
            world_.scene().addActor(*actor);
        }
        const std::vector<Vector3> visPts = conveyor::resamplePath(ctrl, smooth);
        if (visPts.size() >= 2)
            scene_.add(Mesh::create(conveyor::wallGeometry(visPts, height), wallVisualMat_));
    }

    void ConveyorSystem::load() {
        const fs::path layoutPath = convDir_ / "layout.json";
        if (auto layout = conveyor::loadLayout(layoutPath); layout) {
            std::cout << "Loaded conveyor layout (" << layout->pieces.size() << " pieces, "
                      << layout->paths.size() << " paths) from " << layoutPath << std::endl;
            for (const auto& piece : layout->pieces) addConveyorVisual(piece);
            if (!layout->paths.empty()) {
                for (const auto& p : layout->paths) {
                    if (p.separator) buildPathWall(p.waypoints, p.wallHeight, p.smooth);
                    else buildPathBelts(p.waypoints, p.beltWidth, p.beltSpeed, p.reverse, p.smooth,
                                        p.rollerRadius, p.cleatHeight, p.cleatSpacing);
                }
            } else {
                for (const auto& piece : layout->pieces) addStraightBelt(piece);
            }
        } else {
            std::cout << "No layout.json — using a single default conveyor (ConveyorBelt_A06)." << std::endl;
            conveyor::Piece p;
            p.model = "ConveyorBelt_A06";
            p.beltSpeed = 0.6f;
            addConveyorVisual(p);
            addStraightBelt(p);
        }
        if (inlets_.empty()) inlets_.emplace_back(0.f, 1.f, 0.f);

        world_.onPreSubstep([this](float dt) { preSubstep(dt); });
        world_.onPostSubstep([this](float) { postSubstep(); });
    }

    // Fake surface velocity for every belt, around each physics substep: straight belts
    // translate their target, curved belts rotate it about the arc centre (the actor's own
    // origin). Both teleport back afterwards (postSubstep).
    void ConveyorSystem::preSubstep(float dt) {
        for (auto& b : belts_) {
            b.saved = b.actor->getGlobalPose();
            PxTransform target = b.saved;
            if (b.rotational) {
                target.q = b.saved.q * PxQuat(b.omega * beltSpeedScale * dt, PxVec3(0, 1, 0));
            } else {
                target.p += b.velocity * (beltSpeedScale * dt);
            }
            b.actor->setKinematicTarget(target);
        }
        // Cleats genuinely travel along their track (and are NOT restored in postSubstep), so
        // they carry fish up an incline. A forward step uses setKinematicTarget (gives the
        // contact a push velocity); the end→start wrap uses setGlobalPose so it teleports
        // without flinging whatever it lands near.
        for (auto& tr : cleatTracks_) {
            if (tr.length < 1e-4f) continue;
            tr.phase = std::fmod(tr.phase + tr.speed * beltSpeedScale * dt, tr.length);
            for (auto& cl : tr.cleats) {
                const float s = std::fmod(cl.offset + tr.phase, tr.length);
                Vector3 center;
                Quaternion q;
                // Slerp the bar flat (90°) at the ends so it folds over the pulley as it wraps.
                const float fold = conveyor::cleatFold(s, tr.length, tr.rampLen);
                conveyor::cleatPoseAt(tr.poly, s, tr.height, fold, center, q);
                const PxTransform pose = toPxTransform(center, q);
                if (cl.prevS < 0.f || s < cl.prevS) cl.actor->setGlobalPose(pose);
                else cl.actor->setKinematicTarget(pose);
                cl.prevS = s;
            }
        }
    }

    void ConveyorSystem::postSubstep() {
        for (auto& b : belts_) b.actor->setGlobalPose(b.saved);
    }

    void ConveyorSystem::update(float dt) {
        // Scroll each belt's texture to fake surface motion (same speed scale as the kinematic
        // drag, so the visual matches the physics).
        for (auto& bv : beltVisuals_) {
            bv.tex->offset.y += bv.scrollRate * beltSpeedScale * dt;
            bv.tex->offset.y -= std::floor(bv.tex->offset.y);// keep in [0,1) for float precision
            // The Vulkan renderer caches each material's uvTransform in a GPU MaterialDesc
            // buffer (shared by the deferred gbuffer AND the path tracer) and only re-uploads
            // it when Material::version() changes — a bare texture->offset write doesn't bump
            // it, so the scroll would never reach the GPU. needsUpdate() bumps the version so
            // the new offset flows through. (GLRenderer recomputes the UV matrix per draw, so
            // it never needed this.)
            bv.mat->needsUpdate();
        }

        // Spin each roller bank in step with the belt drag (same beltSpeedScale), about each
        // roller's own axis (local +Y), on top of its fixed width-axis orientation.
        for (auto& bank : rollerBanks_) {
            bank.angle += bank.omega * beltSpeedScale * dt;
            Quaternion spin;
            spin.setFromAxisAngle(Vector3(0, 1, 0), bank.angle);
            for (size_t i = 0; i < bank.rollers.size() && i < bank.base.size(); ++i) {
                Quaternion q;
                q.multiplyQuaternions(bank.base[i], spin);
                bank.rollers[i]->quaternion.copy(q);
            }
        }

        // Move each cleat mesh to its collider's (post-step) pose so the bars visibly ride up.
        for (auto& tr : cleatTracks_) {
            for (auto& cl : tr.cleats) {
                const PxTransform pose = cl.actor->getGlobalPose();
                cl.mesh->position.set(pose.p.x, pose.p.y, pose.p.z);
                cl.mesh->quaternion.set(pose.q.x, pose.q.y, pose.q.z, pose.q.w);
            }
        }
    }

}// namespace fishsim
