// Fish soft body demo — loads USDZ fish models from an external database
// and drops them as PhysX deformable volumes. Requires a CUDA-capable GPU.

#include "threepp/threepp.hpp"

#include "threepp/renderers/GLRenderer.hpp"
#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/physx/PhysxDebugRenderer.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/loaders/RGBELoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/loaders/USDLoader.hpp"

#include "ConveyorAssets.hpp"
#include "threepp/utils/BufferGeometryUtils.hpp"

#include <nlohmann/json.hpp>

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <unordered_map>
#include <vector>

using namespace threepp;
using namespace ::physx;

namespace fs = std::filesystem;

namespace {

    // Default data locations; override positionally on the command line (see main).
    const fs::path kDefaultFishDb = "C:/dev/FHF_FishProcessing/Database-Fish/Fish_database";
    const fs::path kDefaultParamsFile = "C:/dev/FHF_FishProcessing/Database-Fish/Physical_Parameters/parameters.json";

    std::vector<std::string> discoverFishIds(const fs::path& fishDb) {
        std::vector<std::string> ids;
        if (!fs::is_directory(fishDb)) return ids;
        for (const auto& entry : fs::directory_iterator(fishDb)) {
            if (entry.path().extension() == ".usdz") {
                ids.push_back(entry.path().stem().string());
            }
        }
        std::ranges::sort(ids);
        return ids;
    }

    // Per-specimen physical parameters parsed from parameters.json (keyed by the
    // same ID as the .usdz filename). Length drives the real-world scale, weight
    // the soft-body mass, friction the contact response.
    struct FishParams {
        std::string species;
        float lengthM = 0.4f;// total_length_mm / 1000
        float weightKg = 0.f;// 0 => keep PhysX default unit-density mass
        // Measured static friction (head-first) on each conveyor surface. A real
        // conveyor is the plastic module belt — much grippier than bare steel, so
        // that's the right value for fish riding the belt (esp. on slopes).
        float frictionSteel = 0.5f;
        float frictionPerfSteel = 0.5f;
        float frictionBelt = 0.5f;// plastic_module_belt
    };

    std::unordered_map<std::string, FishParams> discoverFishParams(const fs::path& paramsFile) {
        std::unordered_map<std::string, FishParams> out;
        std::ifstream f(paramsFile);
        if (!f) {
            std::cerr << "No parameters file at " << paramsFile << " (using defaults)" << std::endl;
            return out;
        }
        nlohmann::json j;
        try {
            f >> j;
        } catch (const std::exception& e) {
            std::cerr << "Failed to parse " << paramsFile << ": " << e.what() << std::endl;
            return out;
        }
        for (const auto& [id, v] : j.items()) {
            FishParams p;
            if (v.contains("species")) p.species = v["species"].get<std::string>();
            if (v.contains("dimensions") && v["dimensions"].contains("total_length_mm")) {
                p.lengthM = v["dimensions"]["total_length_mm"].get<float>() / 1000.f;
            }
            if (v.contains("weight_kg")) p.weightKg = v["weight_kg"].get<float>();
            if (auto sf = v.find("static_friction"); sf != v.end()) {
                if (sf->contains("steel")) p.frictionSteel = (*sf)["steel"].value("head_first", p.frictionSteel);
                if (sf->contains("perforated_steel")) p.frictionPerfSteel = (*sf)["perforated_steel"].value("head_first", p.frictionPerfSteel);
                if (sf->contains("plastic_module_belt")) p.frictionBelt = (*sf)["plastic_module_belt"].value("head_first", p.frictionBelt);
            }
            out[id] = p;
        }
        return out;
    }

    struct FishCache {
        std::shared_ptr<BufferGeometry> geometry;
        std::shared_ptr<Material> material;
    };

    FishCache loadFish(USDLoader& loader, const fs::path& fishDb, const std::string& fishId, float targetLengthM) {
        auto group = loader.load(fishDb / (fishId + ".usdz"));
        if (!group) throw std::runtime_error("Failed to load " + fishId);

        std::shared_ptr<BufferGeometry> geom;
        std::shared_ptr<Material> mat;
        group->traverseType<Mesh>([&](Mesh& m) {
            if (geom) return;
            geom = m.geometry();
            mat = m.material();
        });
        if (!geom) throw std::runtime_error("No mesh found in " + fishId);

        // Make the fish diffuse-dominant so the Vulkan path tracer albedo-demodulates
        // them (chit gate: specFrac < 0.3). Demod lets the denoiser smooth only the
        // lighting while keeping the skin/scale albedo crisp; otherwise it filters in
        // radiance space and washes the texture out. The usual culprit is authored
        // metalness (F0 -> albedo -> high spec fraction), so drop it; wet cod read
        // matte anyway. MeshPhysicalMaterial also derives from MeshStandardMaterial.
        if (auto stdMat = std::dynamic_pointer_cast<MeshStandardMaterial>(mat)) {
            stdMat->metalness = 0.f;
            stdMat->roughness = std::max(stdMat->roughness, 0.7f);
        }

        // Keep the visual geometry at full resolution. The physics mesh is
        // simplified internally by addSoftBody (remesh + voxelise into a coarse
        // tet cage), and these full-res vertices are skinned barycentrically
        // onto that cage every frame — so only the simulation is decimated, the
        // render model retains its original detail.
        geom = simplifyGeometry(*geom, 0.8f);

        Box3 bbox;
        bbox.setFromObject(*group, true);
        Vector3 size;
        bbox.getSize(size);
        const float maxDim = std::max({size.x, size.y, size.z});
        if (maxDim > 0.f) {
            const float s = targetLengthM / maxDim;
            geom->scale(s, s, s);
            geom->center();
        }
        
        return {geom, mat};
    }

    struct SpawnedFish {
        std::shared_ptr<Mesh> mesh;
        SoftBody* body;
    };

    SpawnedFish spawnFish(PhysxWorld& world, Scene& scene,
                          const FishCache& fish,
                          const Vector3& position,
                          PxDeformableVolumeMaterial* sbMat,
                          float yaw, int voxelRes, int solverIters,
                          const std::string& cacheKey, float mass) {
        auto geomClone = fish.geometry->clone();
        auto mesh = Mesh::create(geomClone, fish.material);
        mesh->position.copy(position);
        mesh->rotation.y = yaw;
        scene.add(mesh);
        auto* body = world.addSoftBody(*mesh, sbMat, voxelRes,
                                       static_cast<unsigned>(solverIters), false, cacheKey, mass);
        return {mesh, body};
    }

}// namespace

int main(int argc, char** argv) {

    // Data locations from the command line (all optional, positional):
    //   physx_fish [conveyorDir] [fishDir] [parametersFile]
    const fs::path conveyorDir = argc > 1 ? fs::path(argv[1])
                                          : fs::path(std::string(PROJECT_FOLDER) + "/data/models/conveyor");
    const fs::path fishDir = argc > 2 ? fs::path(argv[2]) : kDefaultFishDb;
    const fs::path paramsFile = argc > 3 ? fs::path(argv[3]) : kDefaultParamsFile;

    auto fishIds = discoverFishIds(fishDir);
    if (fishIds.empty()) {
        std::cerr << "No .usdz files found in " << fishDir << std::endl;
        return 1;
    }
    std::cout << "Found " << fishIds.size() << " fish models" << std::endl;

    auto fishParams = discoverFishParams(paramsFile);

    Canvas canvas("PhysX Fish Softbody", {{"aa", 4}, {"vsync", true}});
    auto renderer = createRenderer(canvas);

    if (auto pt = dynamic_cast<VulkanRenderer*>(renderer.get())) {
        pt->setSamplesPerPixel(4);
        pt->setDenoise(true);
        pt->setRestirDIEnabled(true);
        pt->setSilhouetteMsaaExtra(4);
    }

    Scene scene;
    RGBELoader hdrLoader;

    if (auto hdrTexture = hdrLoader.load(std::string(DATA_FOLDER) + "/textures/env/san_giuseppe_bridge/san_giuseppe_bridge_4k.hdr")) {
        scene.background = hdrTexture;
        scene.environment = hdrTexture;
    }

    auto camera = PerspectiveCamera::create(60, canvas.aspect(), 0.01f, 1000);
    camera->position.set(2.f, 5.f, 5.0f);

    OrbitControls controls{*camera, canvas};
    controls.target.set(0, 0.3f, 0);

    auto sun = DirectionalLight::create(0xffffff, 2.5f);
    sun->position.set(5, 10, 7);
    scene.add(sun);

    PhysxWorld::Settings settings;
    settings.enableGpuDynamics = true;
    PhysxWorld world(settings);

    PhysxDebugRenderer debugRenderer(world);
    debugRenderer.visible = false;
    scene.addRef(debugRenderer);

    // --- Conveyor system: rebuild the layout authored in the designer ----------
    // Each piece becomes a visual model (shared loader → identical placement to the
    // designer) plus, when beltSpeed != 0, a KINEMATIC fake-velocity belt: PhysX has
    // no surface velocity for deformables, so each substep we advance the belt box's
    // kinematic target by velocity*dt (friction drags the fish), then teleport it
    // back so the surface never actually moves.
    const fs::path convDir = conveyorDir;
    const fs::path layoutPath = convDir / "layout.json";
    constexpr float beltThick = 0.08f;
    // High-friction rigid material for the belt colliders. Contact friction with a
    // fish is the combination of this and the fish's deformable material, so the
    // belt side must be grippy too (esp. for slopes).
    PxMaterial* beltMat = world.physics().createMaterial(1.0f, 1.0f, 0.1f);

    // Visible belt surface for waypoint PATHS (the curved colliders are separate; this
    // ribbon is just the visual). It carries a scrolling modular-belt texture to fake
    // belt motion, so the material colour is white and the texture supplies the look;
    // each path clones the material + texture so belts scroll independently.
    auto beltVisualMat = MeshStandardMaterial::create();
    beltVisualMat->color = Color(0xffffff);
    beltVisualMat->roughness = 0.85f;
    beltVisualMat->metalness = 0.f;
    beltVisualMat->side = Side::Double;

    // Procedural modular-belt texture: a transverse groove per tile (reads clearly as
    // motion when scrolled along travel) plus a thin longitudinal module line. Tiled via
    // Repeat wrap; cloned per path for independent scroll offsets.
    auto beltTexture = DataTexture::create<unsigned char>(4, 64, 64);
    {
        auto& d = beltTexture->image().data<unsigned char>();
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
    beltTexture->wrapS = TextureWrapping::Repeat;
    beltTexture->wrapT = TextureWrapping::Repeat;
    beltTexture->magFilter = Filter::Linear;
    beltTexture->minFilter = Filter::Linear;
    beltTexture->generateMipmaps = false;
    beltTexture->colorSpace = ColorSpace::sRGB;
    beltTexture->needsUpdate();

    USDLoader usdLoader;
    auto typeOverrides = conveyor::loadTypeOverrides(convDir / "conveyor_types.json");
    std::unordered_map<std::string, conveyor::ModelTemplate> convTemplates;
    auto convTemplate = [&](const std::string& name) -> conveyor::ModelTemplate& {
        auto it = convTemplates.find(name);
        if (it == convTemplates.end()) {
            it = convTemplates.emplace(name, conveyor::prepareModel(usdLoader, convDir / (name + ".usd"))).first;
            if (auto ov = typeOverrides.find(name); ov != typeOverrides.end()) {
                it->second.kind = (ov->second == "curve") ? conveyor::BeltKind::Curve
                                                          : conveyor::BeltKind::Straight;
            }
        }
        return it->second;
    };

    struct Belt {
        PxRigidDynamic* actor;
        bool rotational = false;
        PxVec3 velocity{0, 0, 0};// straight: world m/s
        float omega = 0.f;       // curve: rad/s about the actor's own +Y (origin = arc centre)
        PxTransform saved{PxIdentity};
    };
    std::vector<Belt> belts;
    float beltSpeedScale = 1.f;// global multiplier (UI)
    std::vector<Vector3> inlets;// fish drop points, one per belt path

    // Per-path belt ribbons whose texture is scrolled each frame to fake surface motion.
    struct AnimatedBelt {
        std::shared_ptr<Texture> tex;
        std::shared_ptr<MeshStandardMaterial> mat;// bumped each frame so the scrolled
                                                  // offset re-uploads (see scroll loop)
        float scrollRate;// d(offset.y)/dt at beltSpeedScale = 1
    };
    std::vector<AnimatedBelt> beltVisuals;

    // Roller-conveyor visuals: one rotating bank of cylinders per rollers path. Spun each
    // frame at omega * beltSpeedScale so the rollers turn in step with the kinematic belt
    // drag underneath them (the collider stays a continuous box surface; the rollers are
    // the visible powered-roller bed on top).
    auto rollerVisualMat = MeshStandardMaterial::create();
    rollerVisualMat->color = Color(0x8a9097);
    rollerVisualMat->roughness = 0.4f;
    rollerVisualMat->metalness = 0.7f;
    rollerVisualMat->flatShading = true;
    struct RollerBank {
        std::vector<std::shared_ptr<Mesh>> rollers;
        std::vector<Quaternion> base;// per-roller axis orientation (parallel to rollers)
        float omega = 0.f;           // rad/s about each roller's own +Y at beltSpeedScale = 1
        float angle = 0.f;           // accumulated spin
    };
    std::vector<RollerBank> rollerBanks;

    // Cleated belts: thin bars standing across the belt that GENUINELY travel along the path
    // at belt speed (wrapping at the ends), so they catch fish on an incline and carry them
    // up. The fake-velocity belt trick teleports its collider back each substep, which works
    // for friction (tangential drag) but not for a pushing barrier (a teleported-back wall
    // un-does its push), so cleats move for real instead.
    auto cleatVisualMat = MeshStandardMaterial::create();
    cleatVisualMat->color = Color(0x202428);
    cleatVisualMat->roughness = 0.6f;
    cleatVisualMat->metalness = 0.3f;
    struct MovingCleat {
        PxRigidDynamic* actor;
        std::shared_ptr<Mesh> mesh;
        float offset;       // base arc-length along the track
        float prevS = -1.f; // last position; a backward jump = a wrap (teleport, no push)
    };
    struct CleatTrack {
        std::vector<Vector3> poly;// travel-ordered centerline the cleats ride
        float length = 0.f;       // total arc length
        float speed = 0.f;        // m/s at beltSpeedScale = 1
        float height = 0.f;
        float rampLen = 0.f;      // distance over which a bar rises/folds flat at each end
        float phase = 0.f;        // advances with time, wrapped to [0,length)
        std::vector<MovingCleat> cleats;
    };
    std::vector<CleatTrack> cleatTracks;

    auto addConveyorVisual = [&](const conveyor::Piece& piece) {
        auto& t = convTemplate(piece.model);
        if (!t.group) {
            std::cerr << "Conveyor model missing: " << piece.model << std::endl;
            return;
        }
        auto vis = t.group->clone();
        vis->position.copy(piece.position);
        vis->rotation.set(0.f, math::degToRad(piece.yawDeg), 0.f);
        vis->scale.set(piece.scale, piece.scale, piece.scale);
        scene.add(vis);
    };

    // Per-piece straight belt — used only when a layout has no waypoint path.
    auto addStraightBelt = [&](const conveyor::Piece& piece) {
        if (piece.beltSpeed == 0.f) return;
        auto& t = convTemplate(piece.model);
        if (!t.group) return;
        const float yaw = math::degToRad(piece.yawDeg);
        const float deckWorldTop = piece.position.y + t.deckTopY * piece.scale;
        const float sx = std::max(t.deckSize.x * piece.scale, 0.2f);
        const float sz = std::max(t.deckSize.z * piece.scale, 0.2f);
        const PxQuat q(yaw, PxVec3(0, 1, 0));
        const PxVec3 center(piece.position.x, deckWorldTop - beltThick * 0.5f, piece.position.z);
        PxRigidDynamic* actor = world.physics().createRigidDynamic(PxTransform(center, q));
        actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
        PxShape* shape = world.physics().createShape(
                PxBoxGeometry(sx * 0.5f, beltThick * 0.5f, sz * 0.5f), *beltMat, true);
        actor->attachShape(*shape);
        shape->release();
        world.scene().addActor(*actor);
        const PxVec3 vel = q.rotate(PxVec3(piece.beltSpeed, 0, 0));
        belts.push_back({actor, false, vel, 0.f, PxTransform(PxIdentity)});
        PxVec3 dir = vel;
        if (dir.magnitude() > 1e-4f) dir = dir.getNormalized();
        inlets.emplace_back(center.x - dir.x * (sx * 0.4f), deckWorldTop + 0.35f, center.z - dir.z * (sx * 0.4f));
    };

    // One straight drag-segment a→b: a kinematic box whose top face sits on the path,
    // dragged LINEARLY along its travel direction (the fake-velocity trick). Boxes
    // butt end-to-end (no length overlap): at a convex crest an overhanging box would
    // poke up through the next and hit the fish.
    auto addStraightSeg = [&](const Vector3& a, const Vector3& b, float width, float speed) {
        Vector3 d(b.x - a.x, b.y - a.y, b.z - a.z);
        const float len = d.length();
        if (len < 1e-3f) return;
        d.multiplyScalar(1.f / len);// full 3D travel direction (incl. slope)
        const PxQuat q = toPxQuat(conveyor::segmentOrientation(a, b));
        const PxVec3 nrm = q.rotate(PxVec3(0, 1, 0));// offset down along the belt normal
        const PxVec3 center((a.x + b.x) * 0.5f - nrm.x * beltThick * 0.5f,
                            (a.y + b.y) * 0.5f - nrm.y * beltThick * 0.5f,
                            (a.z + b.z) * 0.5f - nrm.z * beltThick * 0.5f);
        PxRigidDynamic* actor = world.physics().createRigidDynamic(PxTransform(center, q));
        actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
        PxShape* shape = world.physics().createShape(
                PxBoxGeometry(len * 0.5f, beltThick * 0.5f, width * 0.5f), *beltMat, true);
        actor->attachShape(*shape);
        shape->release();
        world.scene().addActor(*actor);
        belts.push_back({actor, false, PxVec3(d.x * speed, d.y * speed, d.z * speed), 0.f, PxTransform(PxIdentity)});
    };

    // A horizontal bend A→B around centre C as ONE kinematic body that ROTATES about the
    // vertical axis through C — the rotational analogue of the straight belt. PhysX can't
    // drag a curved belt linearly (every point needs its own direction), so the whole
    // bend is one rigid body we rotate. Its surface is tiled by CONVEX ANNULAR WEDGES
    // (radial side faces) rather than rectangular boxes: wedges share each radial face
    // exactly, tiling the ring with no overlap and no gap (wide rectangles would mutually
    // overlap on a tight bend). Cooked GPU-compatible so they collide with the GPU-
    // simulated fish. omega = beltSpeed/radius, signed so the surface flows A→B
    // (onPreSubstep rotates about local +Y; a +Y turn moves a surface point to a LOWER
    // path-angle, hence the minus).
    auto addArcBelt = [&](const Vector3& A, const Vector3& C, const Vector3& B,
                          float width, float speed, const Vector3& incoming) {
        const float PI = static_cast<float>(math::PI);
        const conveyor::Arc arc = conveyor::computeArc(A, C, B, incoming);
        const float radius = 0.5f * (arc.radA + arc.radB);
        if (!arc.valid || radius < 1e-3f) {// degenerate — fall back to a straight chord
            addStraightSeg(A, B, width, speed);
            return;
        }
        const float deckY = 0.5f * (A.y + B.y);// horizontal bend
        PxRigidDynamic* actor = world.physics().createRigidDynamic(
                PxTransform(PxVec3(C.x, deckY, C.z)));// identity rot: local +Y = world up
        actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);

        PxCookingParams cookParams(world.physics().getTolerancesScale());
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
                    v[n++] = PxVec3(rr * cs, yTop, rr * sn);             // top
                    v[n++] = PxVec3(rr * cs, yTop - beltThick, rr * sn); // bottom
                }
            }
            PxConvexMeshDesc cd;
            cd.points.count = 8;
            cd.points.stride = sizeof(PxVec3);
            cd.points.data = v;
            cd.flags = PxConvexFlag::eCOMPUTE_CONVEX | PxConvexFlag::eGPU_COMPATIBLE;
            PxConvexMesh* cm = PxCreateConvexMesh(cookParams, cd, world.physics().getPhysicsInsertionCallback());
            if (!cm) continue;
            PxShape* shape = world.physics().createShape(PxConvexMeshGeometry(cm), *beltMat, true);
            actor->attachShape(*shape);
            shape->release();
        }
        world.scene().addActor(*actor);
        const float omega = -std::copysign(speed / radius, arc.sweep);
        belts.push_back({actor, true, PxVec3(0, 0, 0), omega, PxTransform(PxIdentity)});
    };

    // Point a given arc-length distance into a (travel-ordered) polyline; clamps to the
    // last point for short paths. Used to spawn fish a bit onto the belt, not at its edge.
    auto pointAlong = [](const std::vector<Vector3>& path, float dist) -> Vector3 {
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
    };

    // Build belt colliders along an explicit waypoint path: straight runs become
    // linearly-dragged boxes; arc-CENTRE waypoints each become one rotationally-driven
    // bend body. `reverse` flips the whole path (travel direction + inlet).
    auto buildPathBelts = [&](const std::vector<conveyor::Waypoint>& ctrl, float width, float speed,
                              bool reverse, bool smooth, float rollerRadius,
                              float cleatHeight, float cleatSpacing) {
        bool hasArc = false;
        for (const auto& w : ctrl) {
            if (w.arcCenter) { hasArc = true; break; }
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
            // Walk waypoints (mirroring resamplePath): straight runs between regular
            // points, a rotational bend body at each arc centre. Reversing the waypoint
            // list flips travel + inlet (arc centres stay between their neighbours).
            std::vector<conveyor::Waypoint> wps(ctrl);
            if (reverse) std::ranges::reverse(wps);
            const int n = static_cast<int>(wps.size());
            bool haveLast = false;
            Vector3 lastPt, incoming(0, 0, 0);
            for (int i = 0; i < n;) {
                if (wps[i].arcCenter) {
                    if (!haveLast || i + 1 >= n || wps[i + 1].arcCenter) { ++i; continue; }
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

        // Visible surface, PER SEGMENT kind (runs share boundary points so they meet
        // gap-free). Flat/cleats runs get a scrolling ribbon; rollers runs a rotating bank;
        // cleats runs additionally get moving flight bars (real, travelling colliders). The
        // box-belt colliders above span the whole path regardless, so fish convey on every run.
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
                    auto roller = Mesh::create(cyl, rollerVisualMat);
                    roller->position.copy(r.center);
                    roller->quaternion.copy(r.orientation);
                    scene.add(roller);
                    bank.rollers.push_back(roller);
                    bank.base.push_back(r.orientation);
                }
                if (!bank.rollers.empty()) rollerBanks.push_back(std::move(bank));
            } else {
                // Flat or cleats: a scrolling ribbon (one clone per run so each scrolls on its
                // own); UV.v is arc length (metres), so scrolling offset.y drags the pattern.
                constexpr float tileLen = 0.25f;// metres of belt per texture tile
                auto tex = beltTexture->clone();
                tex->wrapS = tex->wrapT = TextureWrapping::Repeat;// ensure tiling survives clone
                tex->minFilter = tex->magFilter = Filter::Linear;
                tex->generateMipmaps = false;
                tex->colorSpace = ColorSpace::sRGB;
                tex->repeat.set(std::max(1.f, std::round(width / tileLen)), 1.f / tileLen);
                auto mat = beltVisualMat->clone<MeshStandardMaterial>();
                mat->map = tex;
                scene.add(Mesh::create(conveyor::ribbonGeometry(run.pts, width), mat));
                // offset.y is in tile units → scroll rate = speed[m/s] * repeat.y[tiles/m];
                // negative drags the pattern along travel, flipped for a reversed belt.
                beltVisuals.push_back({tex, mat, (reverse ? 1.f : -1.f) * speed * tex->repeat.y});

                // Cleats: bars riding this run's (travel-ordered) centreline. Each is a kinematic
                // box collider (raised across the belt) + a visual; both are advanced along the
                // run every substep (see onPreSubstep), wrapping at its ends.
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
                        PxRigidDynamic* actor = world.physics().createRigidDynamic(toPxTransform(center, q));
                        actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
                        PxShape* shape = world.physics().createShape(
                                PxBoxGeometry(conveyor::kCleatThickness * 0.5f, cleatHeight * 0.5f, width * 0.5f),
                                *beltMat, true);
                        actor->attachShape(*shape);
                        shape->release();
                        world.scene().addActor(*actor);
                        auto bar = Mesh::create(box, cleatVisualMat);
                        bar->position.copy(center);
                        bar->quaternion.copy(q);
                        scene.add(bar);
                        track.cleats.push_back({actor, bar, s});
                    }
                    if (!track.cleats.empty()) cleatTracks.push_back(std::move(track));
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
            inlets.emplace_back(in.x, in.y + 0.35f, in.z);
        }
    };

    // Separator: a collision-only vertical wall along a waypoint centerline (guide
    // rail / lane divider) — STATIC thin boxes per segment (no belt velocity), plus a
    // thin double-sided visual so it's placeable. Reuses the same waypoint system.
    auto wallVisualMat = MeshStandardMaterial::create();
    wallVisualMat->color = Color(0xbfc6cc);
    wallVisualMat->roughness = 0.7f;
    wallVisualMat->metalness = 0.f;
    wallVisualMat->transparent = true;
    wallVisualMat->opacity = 0.55f;
    wallVisualMat->side = Side::Double;
    constexpr float wallThick = 0.04f;
    auto buildPathWall = [&](const std::vector<conveyor::Waypoint>& ctrl, float height, bool smooth) {
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
            // deformable fish, but never added to belts[] → its target is never advanced,
            // so it stays put: an immovable collision-only wall.
            PxRigidDynamic* actor = world.physics().createRigidDynamic(PxTransform(center, q));
            actor->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
            PxShape* shape = world.physics().createShape(
                    PxBoxGeometry(len * 0.5f, height * 0.5f, wallThick * 0.5f), *beltMat, true);
            actor->attachShape(*shape);
            shape->release();
            world.scene().addActor(*actor);
        }
        const std::vector<Vector3> visPts = conveyor::resamplePath(ctrl, smooth);
        if (visPts.size() >= 2)
            scene.add(Mesh::create(conveyor::wallGeometry(visPts, height), wallVisualMat));
    };

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
    if (inlets.empty()) inlets.emplace_back(0.f, 1.f, 0.f);

    // Fake surface velocity for every belt, around each physics substep: straight
    // belts translate their target, curved belts rotate it about the arc centre
    // (the actor's own origin). Both teleport back afterwards.
    world.onPreSubstep([&](float dt) {
        for (auto& b : belts) {
            b.saved = b.actor->getGlobalPose();
            PxTransform target = b.saved;
            if (b.rotational) {
                target.q = b.saved.q * PxQuat(b.omega * beltSpeedScale * dt, PxVec3(0, 1, 0));
            } else {
                target.p += b.velocity * (beltSpeedScale * dt);
            }
            b.actor->setKinematicTarget(target);
        }
        // Cleats genuinely travel along their track (and are NOT restored in onPostSubstep),
        // so they carry fish up an incline. A forward step uses setKinematicTarget (gives the
        // contact a push velocity); the end→start wrap uses setGlobalPose so it teleports
        // without flinging whatever it lands near.
        for (auto& tr : cleatTracks) {
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
    });
    world.onPostSubstep([&](float) {
        for (auto& b : belts) b.actor->setGlobalPose(b.saved);
    });

    std::unordered_map<std::string, FishCache> fishCache;
    std::unordered_map<std::string, PxDeformableVolumeMaterial*> fishMats;
    std::vector<SpawnedFish> spawnedFish;

    std::mt19937 rng{42};
    int selectedFish = 0;
    int voxelRes = 8;
    int solverIters = 15;// soft-body solver iterations (lower = faster, less stiff)
    // GPU tet-skinning is supported on the GL forward renderer (vertex-shader tet
    // blend from a tet texture) and on the Vulkan path tracer (tet_skinning.comp
    // writes the deformed verts straight into the BLAS). Other renderers (WGPU/Cross)
    // fall back to CPU skin. The toggle stays enabled on supported renderers so it can
    // be turned off to A/B against the CPU path.
    const bool tetSkinSupported =
            dynamic_cast<GLRenderer*>(renderer.get()) != nullptr
#ifdef THREEPP_WITH_VULKAN
            || dynamic_cast<VulkanRenderer*>(renderer.get()) != nullptr
#endif
            ;
    bool gpuSkin = tetSkinSupported;// skin the visual on the GPU (tet texture) instead of CPU
    int beltSurface = 2;// 0=steel, 1=perforated steel, 2=plastic module belt (the belt)

    auto paramFor = [&](const std::string& id) -> const FishParams& {
        static const FishParams def{};
        auto it = fishParams.find(id);
        return it != fishParams.end() ? it->second : def;
    };

    // Measured contact friction for the currently-selected conveyor surface.
    auto frictionFor = [&](const std::string& id) -> float {
        const auto& p = paramFor(id);
        switch (beltSurface) {
            case 0: return p.frictionSteel;
            case 1: return p.frictionPerfSteel;
            default: return p.frictionBelt;
        }
    };

    // One soft-body material per (species, surface): the measured friction differs
    // per fish and per surface. Stiffness/damping aren't in the dataset (constant).
    auto materialFor = [&](const std::string& id) -> PxDeformableVolumeMaterial* {
        const std::string key = id + "#" + std::to_string(beltSurface);
        if (auto it = fishMats.find(key); it != fishMats.end()) return it->second;
        auto* m = world.createSoftBodyMaterial(1.0e4f, 0.45f, frictionFor(id));
        m->setDamping(1.0f);
        m->setElasticityDamping(0.025f);
        fishMats[key] = m;
        return m;
    };

    auto getFish = [&](int idx) -> const FishCache& {
        const auto& id = fishIds[idx];
        if (!fishCache.contains(id)) {
            std::cout << "Loading " << id << "..." << std::endl;
            fishCache[id] = loadFish(usdLoader, fishDir, id, paramFor(id).lengthM);
            std::cout << "  vertices: "
                      << fishCache[id].geometry->getAttribute<float>("position")->count()
                      << std::endl;
        }
        return fishCache.at(id);
    };

    auto dropFish = [&](const Vector3& pos) {
        const int idx = selectedFish;
        const auto& id = fishIds[idx];
        std::uniform_real_distribution<float> yawDist(-math::PI, math::PI);
        try {
            auto fish = spawnFish(world, scene, getFish(idx), pos,
                                  materialFor(id), yawDist(rng), voxelRes, solverIters, id,
                                  paramFor(id).weightKg);
            if (gpuSkin && tetSkinSupported && fish.body) fish.body->enableGpuSkinning();
            spawnedFish.push_back(fish);
        } catch (const std::exception& e) {
            std::cerr << "Spawn failed: " << e.what() << std::endl;
        }
    };

    // Drop one fish on each belt inlet to start; more stream in via auto-spawn.
    for (const auto& in : inlets) dropFish(in);

    bool autoSpawn = true;
    float spawnInterval = 2.5f;
    float spawnAccum = 0.f;
    int spawnInletIdx = 0;// round-robin over the belt inlets
    bool spawnPending = false;
    KeyAdapter spaceKey(KeyAdapter::KEY_PRESSED, [&](KeyEvent evt) {
        if (ImGui::GetIO().WantCaptureKeyboard) return;
        if (evt.key == Key::SPACE) spawnPending = true;
    });
    canvas.addKeyListener(spaceKey);

    IOCapture capture{};
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    capture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    canvas.setIOCapture(&capture);

    bool showDebug = false;
    float fps = 0.f, fpsAccum = 0.f;
    int fpsFrames = 0;

    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        const float width = 300 * ui.dpiScale();
        ImGui::SetNextWindowPos({float(canvas.size().width()) - width, 0}, 0, {0, 0});
        ImGui::SetNextWindowSize({width, 0}, 0);
        ImGui::Begin("Fish Softbody");
        ImGui::Text("FPS: %.1f", fps);
        ImGui::Text("Fish: %d", (int) spawnedFish.size());
        ImGui::Separator();
        if (ImGui::BeginCombo("Species", fishIds[selectedFish].c_str())) {
            for (int i = 0; i < static_cast<int>(fishIds.size()); ++i) {
                const bool isSelected = (i == selectedFish);
                if (ImGui::Selectable(fishIds[i].c_str(), isSelected)) {
                    selectedFish = i;
                }
                if (isSelected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        const FishParams& sel = paramFor(fishIds[selectedFish]);
        ImGui::Text("Species: %s", sel.species.empty() ? "(unknown)" : sel.species.c_str());
        ImGui::Text("Length: %.0f mm   Weight: %.2f kg", sel.lengthM * 1000.f, sel.weightKg);
        {
            const char* surfaces[] = {"Steel", "Perforated steel", "Plastic module belt"};
            ImGui::Combo("Belt surface", &beltSurface, surfaces, 3);
            const float fr = beltSurface == 0 ? sel.frictionSteel
                             : beltSurface == 1 ? sel.frictionPerfSteel
                                                : sel.frictionBelt;
            ImGui::Text("Friction (new fish): %.3f", fr);
        }
        ImGui::Separator();
        ImGui::Text("Belts: %d", (int) belts.size());
        ImGui::SliderFloat("Belt speed x", &beltSpeedScale, 0.f, 3.f);
        ImGui::Checkbox("Auto-spawn fish", &autoSpawn);
        ImGui::SliderFloat("Spawn interval (s)", &spawnInterval, 0.3f, 6.f);
        ImGui::SliderInt("Voxel resolution", &voxelRes, 5, 30);
        ImGui::SliderInt("Solver iterations", &solverIters, 4, 30);
        if (!tetSkinSupported) ImGui::BeginDisabled();
        ImGui::Checkbox("GPU skinning", &gpuSkin);
        if (!tetSkinSupported) ImGui::EndDisabled();
        if (!tetSkinSupported) ImGui::TextDisabled("(GPU skinning unsupported on this renderer; using CPU skin)");
        ImGui::TextDisabled("(voxel/solver/GPU-skin apply to NEW fish)");
        ImGui::Separator();
        ImGui::Text("SPACE drops a fish where you aim");
        if (ImGui::Checkbox("Show PhysX debug", &showDebug)) {
            debugRenderer.visible = showDebug;
            if (showDebug) debugRenderer.enableDefaults();
        }
        ImGui::End();
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    Clock clock;

    canvas.animate([&] {
        const float realDt = clock.getDelta();
        fpsAccum += realDt;
        ++fpsFrames;
        if (fpsAccum >= 0.5f) {
            fps = fpsFrames / fpsAccum;
            fpsAccum = 0.f;
            fpsFrames = 0;
        }

        if (spawnPending) {
            spawnPending = false;
            Vector3 dir;
            dir.subVectors(controls.target, camera->position).normalize();
            Vector3 spawn;
            spawn.copy(camera->position).addScaledVector(dir, 3.f);
            spawn.y = std::max(spawn.y, 0.8f);
            dropFish(spawn);
        }

        // Stream fish onto the conveyor inlet.
        if (autoSpawn && !inlets.empty()) {
            spawnAccum += realDt;
            if (spawnAccum >= spawnInterval) {
                spawnAccum = 0.f;
                std::uniform_real_distribution<float> j(-0.1f, 0.1f);
                Vector3 sp = inlets[spawnInletIdx % inlets.size()];
                ++spawnInletIdx;
                sp.x += j(rng);
                sp.z += j(rng);
                dropFish(sp);
            }
        }

        world.step(realDt);
        debugRenderer.update();

        // Despawn fish that have ridden off the end of the system.
        for (auto it = spawnedFish.begin(); it != spawnedFish.end();) {
            auto* geom = it->body->visualGeometry().get();
            if (geom->boundingSphere && geom->boundingSphere->center.y < -2.f) {
                world.removeSoftBody(it->body);
                it = spawnedFish.erase(it);
            } else {
                ++it;
            }
        }

        // Scroll each belt's texture to fake surface motion (same speed scale as the
        // kinematic drag, so the visual matches the physics).
        for (auto& bv : beltVisuals) {
            bv.tex->offset.y += bv.scrollRate * beltSpeedScale * realDt;
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
        for (auto& bank : rollerBanks) {
            bank.angle += bank.omega * beltSpeedScale * realDt;
            Quaternion spin;
            spin.setFromAxisAngle(Vector3(0, 1, 0), bank.angle);
            for (size_t i = 0; i < bank.rollers.size() && i < bank.base.size(); ++i) {
                Quaternion q;
                q.multiplyQuaternions(bank.base[i], spin);
                bank.rollers[i]->quaternion.copy(q);
            }
        }

        // Move each cleat mesh to its collider's (post-step) pose so the bars visibly ride up.
        for (auto& tr : cleatTracks) {
            for (auto& cl : tr.cleats) {
                const PxTransform pose = cl.actor->getGlobalPose();
                cl.mesh->position.set(pose.p.x, pose.p.y, pose.p.z);
                cl.mesh->quaternion.set(pose.q.x, pose.q.y, pose.q.z, pose.q.w);
            }
        }

        renderer->render(scene, *camera);
#ifdef THREEPP_PHYSX_CUDA_GL_INTEROP
        // Once the renderer has uploaded a fish's tet texture, hand its GL handle to
        // the soft body so subsequent frames feed it via CUDA-GL interop
        // (device->device) instead of the GPU->CPU->GPU bridge.
        if (auto* glr = dynamic_cast<GLRenderer*>(renderer.get())) {
            for (auto& f : spawnedFish) {
                if (f.body && f.body->needsInteropRegister()) {
                    if (auto* tex = f.body->interopTexture()) {
                        if (auto id = glr->getGlTextureId(*tex)) {
                            f.body->registerGlTexture(*id);
                        }
                    }
                }
            }
        }
#endif
        ui.render();
    });
}
