// 3D LIDAR SLAM demo.
//
// The window is split in half:
//   LEFT  — ground truth: an enclosed room with a robot driving a loop,
//           carrying a LidarSensor. The camera is user-orbitable.
//   RIGHT — the SLAM reconstruction: the accumulated point-cloud map plus the
//           estimated trajectory (orange) overlaid on the ground-truth
//           trajectory (green) so accumulated drift is visible.
//
// Both halves share one camera, so the two views line up 1:1 for comparison.
//
// The SLAM is a dependency-free, KISS-ICP-style frame-to-map pipeline:
//   1. The sensor returns world-space hit points; a small "driver" shim
//      de-frames them into the sensor's local frame using the inverse of the
//      ground-truth sensor pose. From that point on the estimator never sees
//      ground truth — it works only from sensor-relative measurements.
//   2. The local scan is voxel-downsampled.
//   3. A constant-velocity motion model predicts the next pose.
//   4. Linearized point-to-point ICP registers the scan against a voxel-hash
//      map, seeded by the prediction.
//   5. The registered scan is inserted into the map.
//
// The reusable pieces — the voxel-hash spatial index + downsampling
// (threepp/extras/pointcloud/VoxelGrid.hpp) and the point-to-point ICP
// (threepp/extras/pointcloud/Icp.hpp) — are first-party threepp components;
// this file is the application glue (scene, motion, the constant-velocity
// frame-to-map front-end, and visualisation) around them.
//
// Everything uses only threepp + the standard library + dear imgui (already
// vendored), so there are no new dependencies.

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/pointcloud/Icp.hpp"
#include "threepp/extras/pointcloud/MarchingCubes.hpp"
#include "threepp/extras/pointcloud/VoxelGrid.hpp"
#include "threepp/helpers/AxesHelper.hpp"
#include "threepp/helpers/GridHelper.hpp"
#include "threepp/helpers/LidarSensor.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/renderers/RendererFactory.hpp"
#include "threepp/threepp.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // -----------------------------------------------------------------------
    // Tunables
    // -----------------------------------------------------------------------
    constexpr unsigned int kDefaultFaceSize = 384;// cube-face resolution (live-tunable)
    constexpr float kSensorNear = 0.5f;
    constexpr float kSensorFar = 30.f;
    constexpr float kSensorHeight = 1.0f;       // mount height above the floor
    constexpr float kSelfReturnRange = 0.7f;    // drop returns nearer than this

    constexpr float kMapVoxel = 0.5f;           // map / NN resolution (>= ICP corr dist so NN is exact)
    constexpr size_t kMapVoxelCap = 20;         // max points kept per map voxel
    constexpr float kMapMinSpacing = 0.08f;     // dedup spacing within a voxel
    constexpr float kRegVoxel = 0.4f;           // ICP registration source (coarse = fast ICP)
    constexpr float kDownsampleVoxel = 0.25f;   // map insertion + display (dense = good map)
    // ICP itself uses threepp::IcpOptions defaults (20 iters, 0.5 m start corr,
    // 0.2 m min corr, 0.3 m robust sigma) — they match this map resolution.

    // Robot eases in from rest over this long; keeps inter-frame motion tiny
    // while the map is still sparse so bootstrap orientation stays accurate.
    constexpr float kRampTime = 1.5f;

    constexpr size_t kMaxLivePoints = 200000;   // live-scan display capacity
    constexpr size_t kMaxMapPoints = 400000;    // reconstruction display capacity
    constexpr size_t kMaxTrajPoints = 200000;

    // Map "meshification" (right pane can show: point cloud / occupancy cubes /
    // marching-cubes surface). Mesh views are rebuilt at most this often.
    constexpr size_t kMaxCubes = 120000;        // occupancy-cube instance capacity
    constexpr float kSurfaceCell = 0.3f;        // marching-cubes grid resolution
    constexpr float kSurfaceRadius = 0.45f;     // point splat radius (surface thickness)
    constexpr float kSurfaceIso = 0.5f;         // isolevel within the splat field
    constexpr float kMeshRebuildInterval = 0.5f;// seconds between active-view rebuilds
    constexpr size_t kMaxSurfaceVerts = 600000; // marching-cubes vertex capacity

    // Room: 30 (X) x 20 (Z), centred on the origin, 3 m walls.
    constexpr float kRoomHalfX = 15.f;
    constexpr float kRoomHalfZ = 10.f;
    constexpr float kWallH = 3.f;
    constexpr float kWallT = 0.3f;

    Vector3 translationOf(const Matrix4& m) {
        return {m.elements[12], m.elements[13], m.elements[14]};
    }

    float smoothstep01(float x) {
        const float c = std::clamp(x, 0.f, 1.f);
        return c * c * (3.f - 2.f * c);
    }

    // -----------------------------------------------------------------------
    // SLAM front-end: constant-velocity prediction + frame-to-map ICP over the
    // reusable threepp::VoxelGrid + threepp::icpPointToPoint primitives.
    // -----------------------------------------------------------------------
    class IcpSlam {
    public:
        IcpSlam() : map_(kMapVoxel, kMapVoxelCap, kMapMinSpacing) {}

        const VoxelGrid& map() const { return map_; }
        const Matrix4& pose() const { return t_; }

        // (Re)initialise the estimator. `initPose` anchors the SLAM global
        // frame to the ground-truth start pose so the estimated and true
        // trajectories overlay (their divergence is then pure drift).
        void reset(const Matrix4& initPose) {
            map_.clear();
            t_.copy(initPose);
            prev_.copy(initPose);
            prev2_.copy(initPose);
            first_ = true;
        }

        // Register `regSrc` (coarse, for a cheap ICP) against the map, then
        // insert `mapSrc` (dense, for a good map + display). Both are local-frame
        // points; the same estimated pose maps them into the map frame. Appends
        // genuinely new map points to `addedOut`. Returns the estimated pose.
        const Matrix4& process(const std::vector<Vector3>& regSrc,
                               const std::vector<Vector3>& mapSrc,
                               std::vector<Vector3>& addedOut) {
            if (first_) {
                first_ = false;// frame 0: trust the init pose, just seed the map
            } else {
                // Constant-velocity prediction in the body frame.
                Matrix4 inv;
                inv.copy(prev2_).invert();
                Matrix4 deltaBody;
                deltaBody.multiplyMatrices(inv, prev_);
                t_.multiplyMatrices(prev_, deltaBody);

                icpPointToPoint(regSrc, map_, t_);
            }

            prev2_.copy(prev_);
            prev_.copy(t_);

            Vector3 gp;
            for (const auto& sp : mapSrc) {
                gp = sp;
                gp.applyMatrix4(t_);
                if (map_.insert(gp)) addedOut.push_back(gp);
            }
            return t_;
        }

    private:
        VoxelGrid map_;
        Matrix4 t_;    // current estimated pose (local -> map)
        Matrix4 prev_; // pose at k-1
        Matrix4 prev2_;// pose at k-2
        bool first_{true};
    };

    // De-frame a scan into the sensor-local frame (via the inverse GT pose),
    // dropping misses and self-returns. This "driver shim" is the only place
    // ground truth enters the SLAM; the result is downsampled with the library's
    // voxelDownsample() before registration.
    std::vector<Vector3> deframeScan(const std::vector<LidarReturn>& cloud,
                                     const Matrix4& sensorInv, float minRange) {
        std::vector<Vector3> out;
        out.reserve(cloud.size());
        Vector3 p;
        for (const auto& r : cloud) {
            if (r.returnNo <= 0) continue;
            if (r.distance < minRange) continue;
            p = r.position;
            p.applyMatrix4(sensorInv);// world -> sensor-local
            out.push_back(p);
        }
        return out;
    }

    // -----------------------------------------------------------------------
    // Scene construction
    // -----------------------------------------------------------------------
    void addBox(Scene& scene, const std::shared_ptr<Material>& mat,
                float w, float h, float d, float x, float y, float z) {
        auto box = Mesh::create(BoxGeometry::create(w, h, d), mat);
        box->position.set(x, y, z);
        scene.add(box);
    }

    void buildRoom(Scene& scene) {
        auto floorMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x555a60)));
        auto wallMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x8a8f96)));
        auto pillarMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x6f8fb5)));

        // Floor (top surface at y = 0).
        addBox(scene, floorMat, kRoomHalfX * 2, 0.2f, kRoomHalfZ * 2, 0, -0.1f, 0);

        // Four enclosing walls. Vertical walls constrain X/Z/yaw; the floor
        // constrains height/pitch/roll — together full-rank for ICP.
        addBox(scene, wallMat, kRoomHalfX * 2, kWallH, kWallT, 0, kWallH / 2, kRoomHalfZ);
        addBox(scene, wallMat, kRoomHalfX * 2, kWallH, kWallT, 0, kWallH / 2, -kRoomHalfZ);
        addBox(scene, wallMat, kWallT, kWallH, kRoomHalfZ * 2, kRoomHalfX, kWallH / 2, 0);
        addBox(scene, wallMat, kWallT, kWallH, kRoomHalfZ * 2, -kRoomHalfX, kWallH / 2, 0);

        // Asymmetric pillars give a locally-unique skyline so ICP cannot lock
        // onto the wrong (symmetric) alignment. None sit on the robot path.
        struct P {
            float x, z, w, d, h;
        };
        const std::array<P, 10> pillars = {{
                {0, 0, 1.0f, 1.0f, 3.0f},
                {-3, 2, 0.6f, 0.6f, 3.0f},
                {4, -1, 0.8f, 0.8f, 3.0f},
                {-5, -3, 0.6f, 0.6f, 2.5f},
                {2, 3, 0.7f, 0.7f, 3.0f},
                {7, 0, 0.5f, 0.5f, 3.0f},
                {12, 7, 0.8f, 0.8f, 3.0f},
                {-12, -7, 0.8f, 0.8f, 3.0f},
                {12, -7, 0.6f, 0.6f, 3.0f},
                {-12, 7, 0.7f, 0.7f, 2.5f},
        }};
        for (const auto& p : pillars) {
            addBox(scene, pillarMat, p.w, p.h, p.d, p.x, p.h / 2, p.z);
        }

        scene.add(AmbientLight::create(0xffffff, 0.6f));
        auto dir = DirectionalLight::create(0xffffff, 0.8f);
        dir->position.set(10, 20, 10);
        scene.add(dir);
    }

    // A small visual robot (hidden from the LIDAR during scans).
    std::shared_ptr<Group> buildRobot() {
        auto robot = Group::create();

        auto bodyMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x222831)));
        auto mastMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0x444b53)));
        auto sensorMat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}.color(Color(0xff5533)));

        auto body = Mesh::create(BoxGeometry::create(0.6f, 0.4f, 0.8f), bodyMat);
        body->position.y = 0.2f;
        robot->add(body);

        auto mast = Mesh::create(BoxGeometry::create(0.1f, 0.6f, 0.1f), mastMat);
        mast->position.y = 0.7f;
        robot->add(mast);

        auto sensor = Mesh::create(BoxGeometry::create(0.26f, 0.14f, 0.26f), sensorMat);
        sensor->position.y = kSensorHeight;
        robot->add(sensor);

        robot->add(AxesHelper::create(1.2f));
        return robot;
    }

    std::shared_ptr<Points> makePointCloud(float size) {
        auto geom = BufferGeometry::create();
        geom->setAttribute("position", FloatBufferAttribute::create(std::vector<float>(kMaxMapPoints * 3), 3));
        geom->setAttribute("color", FloatBufferAttribute::create(std::vector<float>(kMaxMapPoints * 3), 3));
        geom->getAttribute<float>("position")->setUsage(DrawUsage::Dynamic);
        geom->getAttribute<float>("color")->setUsage(DrawUsage::Dynamic);
        geom->setDrawRange(0, 0);// nothing drawn until points are written
        auto mat = PointsMaterial::create(PointsMaterial::Params{}.size(size).vertexColors(true));
        auto pts = Points::create(geom, mat);
        pts->frustumCulled = false;
        return pts;
    }

    std::shared_ptr<Line> makeTrajectory(const Color& color) {
        auto geom = BufferGeometry::create();
        geom->setAttribute("position", FloatBufferAttribute::create(std::vector<float>(kMaxTrajPoints * 3), 3));
        geom->getAttribute<float>("position")->setUsage(DrawUsage::Dynamic);
        geom->setDrawRange(0, 0);
        auto mat = LineBasicMaterial::create(LineBasicMaterial::Params{}.color(color));
        auto line = Line::create(geom, mat);
        line->frustumCulled = false;
        return line;
    }

    void appendVertex(BufferGeometry& geom, int& count, const Vector3& p) {
        if (static_cast<size_t>(count) >= kMaxTrajPoints) return;
        auto* attr = geom.getAttribute<float>("position");
        attr->setXYZ(count, p.x, p.y, p.z);
        ++count;
        attr->needsUpdate();
        geom.setDrawRange(0, count);
    }

    // Headless correctness check for the core estimator: build a map, define a
    // known sensor pose, synthesise the matching local scan, and confirm ICP
    // recovers the pose. Needs no GL context. Run: lidar_slam --selftest
    int runSelfTest() {
        VoxelGrid map(kMapVoxel, kMapVoxelCap, kMapMinSpacing);
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> ux(-10.f, 10.f), uy(0.f, 3.f), uz(-10.f, 10.f);

        std::vector<Vector3> mapPts;
        for (int i = 0; i < 4000; ++i) {
            Vector3 p(ux(rng), uy(rng), uz(rng));
            mapPts.push_back(p);
            map.insert(p);
        }

        // Ground-truth sensor pose (map <- local): 0.2 m translation, 3 deg yaw.
        Quaternion q;
        q.setFromAxisAngle(Vector3(0, 1, 0), 3.f * math::DEG2RAD);
        Matrix4 tKnown;
        tKnown.compose(Vector3(0.2f, 0.05f, -0.15f), q, Vector3(1, 1, 1));
        Matrix4 tInv;
        tInv.copy(tKnown).invert();

        // Local scan = tKnown^-1 * map points, plus realistic range noise.
        std::normal_distribution<float> noise(0.f, 0.01f);
        std::vector<Vector3> src;
        for (const auto& m : mapPts) {
            Vector3 s = m;
            s.applyMatrix4(tInv);
            s.x += noise(rng);
            s.y += noise(rng);
            s.z += noise(rng);
            src.push_back(s);
        }

        // Seed at identity (offset is small) and let ICP recover tKnown.
        Matrix4 t;
        icpPointToPoint(src, map, t);

        // Compare full pose (rotation + translation) via probe points.
        float maxErr = 0.f;
        const std::array<Vector3, 4> probes = {{{5, 1, 5}, {-5, 2, -5}, {5, 0, -5}, {0, 3, 0}}};
        for (const auto& pr : probes) {
            Vector3 a = pr;
            a.applyMatrix4(t);
            Vector3 b = pr;
            b.applyMatrix4(tKnown);
            maxErr = std::max(maxErr, a.sub(b).length());
        }

        const Vector3 tp = translationOf(t);
        std::cout << "recovered translation = (" << tp.x << ", " << tp.y << ", " << tp.z << ")\n";
        std::cout << "expected  translation = (0.2, 0.05, -0.15)\n";
        std::cout << "max probe-point error = " << maxErr << " m\n";

        const bool pass = maxErr < 0.03f;
        std::cout << (pass ? "SELFTEST PASS" : "SELFTEST FAIL") << std::endl;
        return pass ? 0 : 1;
    }

    // Colour by height: blue (floor) -> red (ceiling).
    Color heightColor(float y) {
        const float t = std::clamp(y / kWallH, 0.f, 1.f);
        Color c;
        c.setHSL(0.66f * (1.f - t), 0.85f, 0.5f);
        return c;
    }

}// namespace

int main(int argc, char** argv) {

    if (argc > 1 && std::string(argv[1]) == "--selftest") {
        return runSelfTest();
    }

    Canvas canvas("Lidar SLAM", {{"antialiasing", 4}});
    auto renderer = createRenderer(canvas);
    renderer->setScissorTest(true);
    renderer->shadowMap().enabled = false;

    // --- Ground-truth scene (left) ---
    Scene sceneLeft;
    sceneLeft.background = Color(0x1d2330);
    buildRoom(sceneLeft);

    auto robot = buildRobot();
    sceneLeft.add(robot);

    // LIDAR model + scan resolution are both live-selectable; changing either
    // rebuilds the sensor (resolution is a constructor parameter).
    const char* modelNames[] = {"OS1-64", "OS0-128", "HDL-32E", "VLP-16"};
    int currentModel = 0;
    const int faceSizeOptions[] = {192, 256, 384, 512};
    const char* faceSizeNames[] = {"192", "256", "384", "512"};
    int currentFaceIdx = 2;// 384
    auto makeLidar = [&](int model, unsigned int faceSize) {
        std::unique_ptr<LidarSensor> s;
        switch (model) {
            case 1: s = std::make_unique<LidarSensor>(LidarModel::OS0_128(), faceSize, kSensorNear, kSensorFar); break;
            case 2: s = std::make_unique<LidarSensor>(LidarModel::HDL32E(), faceSize, kSensorNear, kSensorFar); break;
            case 3: s = std::make_unique<LidarSensor>(LidarModel::VLP16(), faceSize, kSensorNear, kSensorFar); break;
            default: s = std::make_unique<LidarSensor>(LidarModel::OS1_64(), faceSize, kSensorNear, kSensorFar); break;
        }
        return s;
    };

    auto lidar = makeLidar(currentModel, static_cast<unsigned int>(faceSizeOptions[currentFaceIdx]));
    lidar->rangeNoise = 0.02f;
    sceneLeft.addRef(*lidar);

    auto livePoints = makePointCloud(0.06f);
    sceneLeft.add(livePoints);

    // --- Reconstruction scene (right) ---
    Scene sceneRight;
    sceneRight.background = Color(0x0a0c10);
    sceneRight.add(GridHelper::create(60, 60, Color(0x224422), Color(0x1a2a1a)));
    sceneRight.add(AxesHelper::create(2.0f));

    // Lights for the shaded mesh views (point cloud + lines are unlit, so these
    // only affect the cubes / surface).
    sceneRight.add(AmbientLight::create(0xffffff, 0.7f));
    auto rightDir = DirectionalLight::create(0xffffff, 0.7f);
    rightDir->position.set(8, 20, 12);
    sceneRight.add(rightDir);

    // View 0: raw point cloud.
    auto mapPoints = makePointCloud(0.05f);
    sceneRight.add(mapPoints);

    // View 1: occupancy cubes (one voxel-sized box per occupied cell).
    const float cubeSize = static_cast<float>(kMapVoxel);
    auto cubesMesh = InstancedMesh::create(
            BoxGeometry::create(cubeSize, cubeSize, cubeSize),
            MeshStandardMaterial::create(MeshStandardMaterial::Params{}.roughness(0.9f).metalness(0.f)),
            kMaxCubes);
    cubesMesh->setCount(0);
    cubesMesh->frustumCulled = false;
    cubesMesh->visible = false;
    sceneRight.add(cubesMesh);

    // View 2: marching-cubes surface. Attributes are preallocated and updated
    // in place each rebuild (replacing them would churn the GL buffers and cause
    // intermittent bad frames).
    auto surfaceGeom = BufferGeometry::create();
    surfaceGeom->setAttribute("position", FloatBufferAttribute::create(std::vector<float>(kMaxSurfaceVerts * 3), 3));
    surfaceGeom->setAttribute("normal", FloatBufferAttribute::create(std::vector<float>(kMaxSurfaceVerts * 3), 3));
    surfaceGeom->setAttribute("color", FloatBufferAttribute::create(std::vector<float>(kMaxSurfaceVerts * 3), 3));
    surfaceGeom->getAttribute<float>("position")->setUsage(DrawUsage::Dynamic);
    surfaceGeom->getAttribute<float>("normal")->setUsage(DrawUsage::Dynamic);
    surfaceGeom->getAttribute<float>("color")->setUsage(DrawUsage::Dynamic);
    surfaceGeom->setDrawRange(0, 0);
    auto surfaceMat = MeshStandardMaterial::create(
            MeshStandardMaterial::Params{}.roughness(0.85f).metalness(0.f).vertexColors(true));
    surfaceMat->side = Side::Double;
    auto surfaceMesh = Mesh::create(surfaceGeom, surfaceMat);
    surfaceMesh->frustumCulled = false;
    surfaceMesh->visible = false;
    sceneRight.add(surfaceMesh);

    auto gtTraj = makeTrajectory(Color(0x33ff66)); // ground truth
    auto estTraj = makeTrajectory(Color(0xffaa22));// SLAM estimate
    sceneRight.add(gtTraj);
    sceneRight.add(estTraj);

    // --- Shared camera + controls ---
    auto camera = PerspectiveCamera::create(55, canvas.aspect(), 0.1f, 400.f);
    camera->position.set(0, 28, 34);
    OrbitControls controls{*camera, canvas};
    controls.target.set(0, 0, 0);

    // --- SLAM ---
    IcpSlam slam;
    bool slamInitialised = false;

    int mapCount = 0;
    int gtCount = 0;
    int estCount = 0;
    float motionElapsed = 0.f;// time since (re)start, drives the speed ease-in

    // Right-pane map view: 0 = point cloud, 1 = occupancy cubes, 2 = surface.
    int viewMode = 0;
    int meshBuiltMode = -1;  // which view the cube/surface mesh currently shows
    int meshBuiltCount = -1; // mapCount at the last mesh rebuild
    float meshBuildTime = -1.f;

    auto resetReconstruction = [&] {
        slamInitialised = false;
        mapCount = 0;
        gtCount = 0;
        estCount = 0;
        motionElapsed = 0.f;// ease in again so the rebuilt map bootstraps cleanly
        mapPoints->geometry()->setDrawRange(0, 0);
        gtTraj->geometry()->setDrawRange(0, 0);
        estTraj->geometry()->setDrawRange(0, 0);
        cubesMesh->setCount(0);
        surfaceMesh->geometry()->setDrawRange(0, 0);
        meshBuiltMode = -1;
        meshBuiltCount = -1;
    };

    // --- Robot motion ---
    const float pathRx = 10.f, pathRz = 6.f;
    float pathAngle = 0.f;
    float speed = 1.2f;// m/s
    bool paused = false;
    bool showLive = true;

    auto rebuildLidar = [&] {
        sceneLeft.remove(*lidar);
        auto next = makeLidar(currentModel, static_cast<unsigned int>(faceSizeOptions[currentFaceIdx]));
        next->rangeNoise = lidar->rangeNoise;
        next->position.copy(lidar->position);
        next->rotation.copy(lidar->rotation);
        lidar = std::move(next);
        sceneLeft.addRef(*lidar);
        resetReconstruction();
    };

    // --- Map mesh rebuilds (occupancy cubes / marching-cubes surface) ---
    std::vector<Vector3> scratchCenters;
    auto rebuildCubes = [&] {
        scratchCenters.clear();
        slam.map().collectVoxelCenters(scratchCenters);
        const size_t n = std::min(scratchCenters.size(), kMaxCubes);
        Matrix4 m;// identity; only the translation is updated below
        for (size_t i = 0; i < n; ++i) {
            m.setPosition(scratchCenters[i]);
            cubesMesh->setMatrixAt(i, m);
            cubesMesh->setColorAt(i, heightColor(scratchCenters[i].y));
        }
        cubesMesh->setCount(n);
        cubesMesh->instanceMatrix()->needsUpdate();
        if (cubesMesh->instanceColor()) cubesMesh->instanceColor()->needsUpdate();
    };

    std::vector<Vector3> scratchPts;
    auto rebuildSurface = [&] {
        scratchPts.clear();
        slam.map().collect(scratchPts);
        const ScalarField field = splatPointsToField(scratchPts, kSurfaceCell, kSurfaceRadius);
        const IsoMesh iso = marchingCubes(field, kSurfaceIso);
        auto& geom = *surfaceMesh->geometry();
        auto* pos = geom.getAttribute<float>("position");
        auto* nrm = geom.getAttribute<float>("normal");
        auto* col = geom.getAttribute<float>("color");

        size_t nv = std::min(iso.positions.size(), kMaxSurfaceVerts);
        nv -= nv % 3;// whole triangles only
        Color c;
        for (size_t i = 0; i < nv; ++i) {
            pos->setXYZ(i, iso.positions[i].x, iso.positions[i].y, iso.positions[i].z);
            nrm->setXYZ(i, iso.normals[i].x, iso.normals[i].y, iso.normals[i].z);
            c = heightColor(iso.positions[i].y);
            col->setXYZ(i, c.r, c.g, c.b);
        }
        const int cnt = static_cast<int>(nv) * 3;
        pos->updateRange.offset = 0;
        pos->updateRange.count = cnt;
        pos->needsUpdate();
        nrm->updateRange.offset = 0;
        nrm->updateRange.count = cnt;
        nrm->needsUpdate();
        col->updateRange.offset = 0;
        col->updateRange.count = cnt;
        col->needsUpdate();
        geom.setDrawRange(0, static_cast<int>(nv));
    };

    // --- UI ---
    float drift = 0.f;
    float scanMs = 0.f;
    float slamMs = 0.f;
    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        ImGui::SetNextWindowPos({0, 0}, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize({0, 0}, ImGuiCond_FirstUseEver);
        ImGui::Begin("SLAM");
        ImGui::Text("Left: ground truth   Right: reconstruction");
        ImGui::Separator();
        ImGui::Checkbox("Pause robot", &paused);
        ImGui::SameLine();
        ImGui::Checkbox("Show live scan", &showLive);
        ImGui::SliderFloat("Speed (m/s)", &speed, 0.2f, 2.5f);
        ImGui::SliderFloat("Range noise (m)", &lidar->rangeNoise, 0.f, 0.1f);

        int prevModel = currentModel;
        ImGui::Combo("LIDAR", &currentModel, modelNames, 4);
        if (currentModel != prevModel) rebuildLidar();

        int prevFace = currentFaceIdx;
        ImGui::Combo("Scan res", &currentFaceIdx, faceSizeNames, 4);
        if (currentFaceIdx != prevFace) rebuildLidar();

        const char* viewNames[] = {"Point cloud", "Occupancy cubes", "Surface (marching cubes)"};
        ImGui::Combo("Map view", &viewMode, viewNames, 3);

        if (ImGui::Button("Reset SLAM")) resetReconstruction();
        ImGui::Separator();
        ImGui::Text("Map points : %d", mapCount);
        ImGui::Text("Drift      : %.3f m", drift);
        ImGui::Text("Scan  : %.1f ms", scanMs);
        ImGui::Text("SLAM  : %.1f ms", slamMs);
        ImGui::End();
    });

    IOCapture capture;
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    capture.preventScrollEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    canvas.setIOCapture(&capture);

    canvas.onWindowResize([&](WindowSize size) {
        renderer->setSize(size);
    });

    Clock clock;
    float prevTime = 0.f;
    std::vector<LidarReturn> cloud;
    std::vector<Vector3> added;

    canvas.animate([&] {
        const float now = clock.getElapsedTime();
        float dt = now - prevTime;
        prevTime = now;
        dt = std::min(dt, 0.05f);

        // --- Advance the ground-truth robot pose along the loop ---
        // Ease in from rest so the constant-velocity model and the still-sparse
        // map have an easy time during bootstrap (otherwise the first frames
        // mis-estimate orientation).
        if (!paused) {
            motionElapsed += dt;
            const float ramp = smoothstep01(motionElapsed / kRampTime);
            const float avgR = 0.5f * (pathRx + pathRz);
            pathAngle += speed * ramp * dt / std::max(avgR, 0.1f);
        }
        const float px = pathRx * std::cos(pathAngle);
        const float pz = pathRz * std::sin(pathAngle);
        const float vx = -pathRx * std::sin(pathAngle);
        const float vz = pathRz * std::cos(pathAngle);
        const float yaw = std::atan2(vx, vz);// face along the tangent

        lidar->position.set(px, kSensorHeight, pz);
        lidar->rotation.set(0, yaw, 0);
        robot->position.set(px, 0, pz);
        robot->rotation.set(0, yaw, 0);

        // Make the ground-truth sensor pose current, then capture it. This is
        // the matrix the cube-face scan uses; its inverse de-frames returns.
        lidar->updateMatrixWorld(true);
        Matrix4 mGt;
        mGt.copy(*lidar->matrixWorld);
        Matrix4 mInv;
        mInv.copy(mGt).invert();

        // --- Scan (the robot must not see itself) ---
        const float scanT0 = clock.getElapsedTime();
        robot->visible = false;
        livePoints->visible = false;
        lidar->scan(*renderer, sceneLeft, cloud);
        robot->visible = true;
        livePoints->visible = showLive;
        scanMs = (clock.getElapsedTime() - scanT0) * 1000.f;

        // --- Live point cloud (left), coloured by range ---
        {
            auto& geom = *livePoints->geometry();
            auto* pos = geom.getAttribute<float>("position");
            auto* col = geom.getAttribute<float>("color");
            Color c;
            int i = 0;
            for (const auto& r : cloud) {
                if (r.returnNo <= 0) continue;
                if (static_cast<size_t>(i) >= kMaxLivePoints) break;
                pos->setXYZ(i, r.position.x, r.position.y, r.position.z);
                c.setHSL(0.33f * (1.f - std::min(r.distance / kSensorFar, 1.f)), 1.f, 0.5f);
                col->setXYZ(i, c.r, c.g, c.b);
                ++i;
            }
            geom.setDrawRange(0, i);
            // Upload only the valid prefix, not the whole preallocated buffer.
            pos->updateRange.offset = 0;
            pos->updateRange.count = i * 3;
            col->updateRange.offset = 0;
            col->updateRange.count = i * 3;
            pos->needsUpdate();
            col->needsUpdate();
        }

        // --- De-frame to sensor-local + voxel downsample ---
        // Coarse source for a cheap ICP; dense source for a good map + display.
        const auto localScan = deframeScan(cloud, mInv, kSelfReturnRange);
        const auto regSrc = voxelDownsample(localScan, kRegVoxel);
        const auto mapSrc = voxelDownsample(localScan, kDownsampleVoxel);

        // --- SLAM update ---
        const float slamT0 = clock.getElapsedTime();
        if (!slamInitialised) {
            slam.reset(mGt);// anchor SLAM frame to GT start so paths overlay
            slamInitialised = true;
        }
        added.clear();
        const Matrix4& estPose = slam.process(regSrc, mapSrc, added);
        slamMs = (clock.getElapsedTime() - slamT0) * 1000.f;

        // --- Grow the reconstruction display from newly added map points ---
        {
            const int prevCount = mapCount;
            auto& geom = *mapPoints->geometry();
            auto* pos = geom.getAttribute<float>("position");
            auto* col = geom.getAttribute<float>("color");
            for (const auto& q : added) {
                if (static_cast<size_t>(mapCount) >= kMaxMapPoints) break;
                pos->setXYZ(mapCount, q.x, q.y, q.z);
                const Color c = heightColor(q.y);
                col->setXYZ(mapCount, c.r, c.g, c.b);
                ++mapCount;
            }
            if (mapCount > prevCount) {
                geom.setDrawRange(0, mapCount);
                // Upload only the freshly appended tail, not the whole buffer.
                const int off = prevCount * 3;
                const int cnt = (mapCount - prevCount) * 3;
                pos->updateRange.offset = off;
                pos->updateRange.count = cnt;
                col->updateRange.offset = off;
                col->updateRange.count = cnt;
                pos->needsUpdate();
                col->needsUpdate();
            }
        }

        // --- Trajectories + drift ---
        const Vector3 gtPos = translationOf(mGt);
        const Vector3 estPos = translationOf(estPose);
        appendVertex(*gtTraj->geometry(), gtCount, gtPos);
        appendVertex(*estTraj->geometry(), estCount, estPos);
        drift = (estPos.clone().sub(gtPos)).length();

        // --- Map view: point cloud / occupancy cubes / surface ---
        mapPoints->visible = (viewMode == 0);
        cubesMesh->visible = (viewMode == 1);
        surfaceMesh->visible = (viewMode == 2);
        if (viewMode == 0) {
            meshBuiltMode = 0;// nothing to rebuild
        } else {
            // Rebuild the active mesh on switch, then throttled as the map grows.
            const bool modeChanged = (viewMode != meshBuiltMode);
            const bool grew = (mapCount - meshBuiltCount) > 100;
            const bool throttleOk = (now - meshBuildTime) > kMeshRebuildInterval;
            if (modeChanged || (grew && throttleOk)) {
                if (viewMode == 1) rebuildCubes();
                else rebuildSurface();
                meshBuiltMode = viewMode;
                meshBuiltCount = mapCount;
                meshBuildTime = now;
            }
        }

        // --- Split-screen render (independent, correctly-proportioned views) ---
        const auto size = canvas.size();
        const int w = size.width();
        const int h = size.height();
        const int halfW = w / 2;
        camera->aspect = static_cast<float>(halfW) / static_cast<float>(h);
        camera->updateProjectionMatrix();
        controls.update();

        renderer->setViewport(0, 0, halfW, h);
        renderer->setScissor(0, 0, halfW, h);
        renderer->render(sceneLeft, *camera);

        renderer->setViewport(halfW, 0, w - halfW, h);
        renderer->setScissor(halfW, 0, w - halfW, h);
        renderer->render(sceneRight, *camera);

        // UI spans the whole window.
        renderer->setViewport(0, 0, w, h);
        renderer->setScissor(0, 0, w, h);
        ui.render();
    });
}
