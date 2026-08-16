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
//
// `--bench` runs the whole thing non-interactively at a fixed timestep and
// prints trajectory error, so a front-end change is an A/B measurement rather
// than an impression. See the Bench block below for what it has already
// established about where this estimator breaks — in particular, that the
// break is an observability failure and not something a motion sensor fixes.

#include "threepp/extras/imgui/RendererSettings.hpp"
#include "threepp/extras/pointcloud/Icp.hpp"
#include "threepp/extras/pointcloud/MarchingCubes.hpp"
#include "threepp/extras/pointcloud/VoxelGrid.hpp"
#include "threepp/helpers/AxesHelper.hpp"
#include "threepp/helpers/GridHelper.hpp"
#include "threepp/helpers/LidarSensor.hpp"
#ifdef THREEPP_WITH_VULKAN
#include "threepp/helpers/PathTracedLidarSensor.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#endif
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/renderers/RendererFactory.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
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

    constexpr float kMapVoxel = 0.5f;           // map / NN resolution
    constexpr size_t kMapVoxelCap = 20;         // max points kept per map voxel (display density)
    constexpr float kMapMinSpacing = 0.08f;     // dedup spacing within a voxel
    constexpr float kRegVoxel = 0.4f;           // ICP registration source (coarse = fast ICP)
    constexpr float kDownsampleVoxel = 0.25f;   // map insertion + display (dense = good map)
    // The registration NN target is a second, lean grid over the same points.
    // The display map's ~17 points per voxel (cap 20 / 0.08 m) are clumps of
    // per-scan noise: ICP's inner loop scans every clump member per query, and
    // correspondences into a clump pull the pose toward the map's own
    // accumulated error. Few well-spread points per voxel register 3x faster
    // AND drift less (measured: mean drift 0.121 -> 0.024 m on the loop).
    constexpr size_t kRegMapCap = 4;            // points per voxel in the NN target
    constexpr float kRegMapMinSpacing = 0.15f;  // spread them out
    // ICP: corr gates / sigma are the IcpOptions defaults (0.5 m start, 0.2 m
    // floor, 0.3 m sigma) — matched to this map resolution. Iterations and
    // convergence tolerance are relaxed from the "iterate until numerically
    // still" defaults: refining a 2 cm-noise scan below 1 mm is wasted frames.
    constexpr int kIcpMaxIterations = 10;
    constexpr float kIcpTolerance = 1e-3f;      // metres / radians

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
        IcpSlam() : map_(kMapVoxel, kMapVoxelCap, kMapMinSpacing),
                    regMap_(kMapVoxel, kRegMapCap, kRegMapMinSpacing) {
            opts_.maxIterations = kIcpMaxIterations;
            opts_.translationTolerance = kIcpTolerance;
            opts_.rotationTolerance = kIcpTolerance;
        }

        const VoxelGrid& map() const { return map_; }
        const Matrix4& pose() const { return t_; }
        const IcpResult& lastIcp() const { return lastIcp_; }

        // (Re)initialise the estimator. `initPose` anchors the SLAM global
        // frame to the ground-truth start pose so the estimated and true
        // trajectories overlay (their divergence is then pure drift).
        void reset(const Matrix4& initPose) {
            map_.clear();
            regMap_.clear();
            t_.copy(initPose);
            prev_.copy(initPose);
            prev2_.copy(initPose);
            first_ = true;
        }

        // Register `regSrc` (coarse, for a cheap ICP) against the lean
        // registration grid, then insert `mapSrc` (dense, for a good map +
        // display) into both grids. Both are local-frame points; the same
        // estimated pose maps them into the map frame. Appends genuinely new
        // map points to `addedOut`. Returns the estimated pose.
        const Matrix4& process(const std::vector<Vector3>& regSrc,
                               const std::vector<Vector3>& mapSrc,
                               std::vector<Vector3>& addedOut,
                               const Matrix4* bodyDeltaOverride = nullptr) {
            if (first_) {
                first_ = false;// frame 0: trust the init pose, just seed the map
            } else {
                // Prediction: the body-frame motion since the last frame. With
                // no better information that is the constant-velocity guess
                // (replay the previous frame's delta); a motion sensor supplies
                // a measured delta instead.
                Matrix4 deltaBody;
                if (bodyDeltaOverride) {
                    deltaBody.copy(*bodyDeltaOverride);
                } else {
                    Matrix4 inv;
                    inv.copy(prev2_).invert();
                    deltaBody.multiplyMatrices(inv, prev_);
                }
                t_.multiplyMatrices(prev_, deltaBody);

                lastIcp_ = icpPointToPoint(regSrc, regMap_, t_, opts_);
            }

            prev2_.copy(prev_);
            prev_.copy(t_);

            Vector3 gp;
            for (const auto& sp : mapSrc) {
                gp = sp;
                gp.applyMatrix4(t_);
                if (map_.insert(gp)) addedOut.push_back(gp);
                regMap_.insert(gp);// its own cap/spacing picks a sparser subset
            }
            return t_;
        }

    private:
        VoxelGrid map_;   // dense: display + meshing
        VoxelGrid regMap_;// lean: ICP nearest-neighbour target
        IcpOptions opts_;
        Matrix4 t_;    // current estimated pose (local -> map)
        Matrix4 prev_; // pose at k-1
        Matrix4 prev2_;// pose at k-2
        IcpResult lastIcp_{};
        bool first_{true};
    };

    // De-frame a scan into the sensor-local frame (via the inverse GT pose),
    // dropping misses, self-returns and anything past the detector's range.
    // This "driver shim" is the only place ground truth enters the SLAM; the
    // result is downsampled with the library's voxelDownsample() before
    // registration.
    std::vector<Vector3> deframeScan(const std::vector<LidarReturn>& cloud,
                                     const Matrix4& sensorInv, float minRange, float maxRange) {
        std::vector<Vector3> out;
        out.reserve(cloud.size());
        Vector3 p;
        for (const auto& r : cloud) {
            if (r.returnNo <= 0) continue;
            if (r.distance < minRange || r.distance > maxRange) continue;
            p = r.position;
            p.applyMatrix4(sensorInv);// world -> sensor-local
            out.push_back(p);
        }
        return out;
    }

    // Yaw of a pose, in radians. The robot only ever rotates about Y, so this
    // is its heading.
    float yawOf(const Matrix4& m) {
        return std::atan2(m.elements[8], m.elements[10]);
    }

    float wrapPi(float a) {
        while (a > math::PI) a -= 2.f * math::PI;
        while (a < -math::PI) a += 2.f * math::PI;
        return a;
    }

    // -----------------------------------------------------------------------
    // Drift bench: a non-interactive, fixed-dt run that scripts the stress the
    // interactive demo exposes through the "Max range" slider, and prints the
    // trajectory error it produces. Fixed dt matters — with real frame times a
    // front-end A/B measures the machine's frame pacing as much as the
    // estimator.
    //
    //   lidar_slam --bench [--frames N] [--range R] [--range-at F]
    //              [--oracle] [--min-h H]
    //
    // What it already established (so nobody re-runs it to find out):
    //
    //   range 30 (control)      meanDrift 0.021 m   maxDrift 0.107 m   yaw 0.49 deg
    //   range 8 from frame 300  meanDrift 2.267 m   maxDrift 9.206 m   yaw 1.09 deg
    //
    // The failure is NOT the motion model and NOT the ground plane:
    //
    //   --oracle  seeds ICP with the exact ground-truth body delta — a perfect,
    //             noiseless, drift-free motion sensor, i.e. the upper bound on
    //             what any IMU could ever contribute. It buys 9.21 -> 8.05 m.
    //             Effectively nothing: the prediction was never the problem.
    //   --min-h   drops registration points below a sensor-frame height, so
    //             -0.9 removes the floor. Also changes nothing (9.09 m).
    //
    // What it IS: at short range the robot drives along the +Z wall with
    // nothing in range ahead or behind, so X becomes unobservable. Per-axis,
    // estimated Z tracks truth to millimetres while X stays pinned near 4.5 as
    // truth travels 8 m away. Point-to-point ICP cannot see this coming — its
    // translation Jacobian is the identity for every correspondence, so the
    // translation block of the normal matrix is (sum w) * I, isotropic no
    // matter what the geometry looks like. Detecting the degenerate direction
    // needs surface normals (point-to-plane), which VoxelGrid does not carry;
    // only then is there a null space for a motion prior to fill.
    // -----------------------------------------------------------------------
    struct Bench {
        bool on = false;
        int frames = 900;   // ~15 s at the fixed 60 Hz step
        float range = 8.f;  // detector max range applied at `rangeAt`
        int rangeAt = 300;  // frame the range change lands on
        bool oracle = false;// seed ICP with the TRUE body delta (see above)
        float minRegHeight = -1e9f;// drop registration points below this sensor-frame height
    };

    Bench parseBench(int argc, char** argv) {
        Bench b;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto val = [&](const char* flag) -> const char* {
                return (a == flag && i + 1 < argc) ? argv[++i] : nullptr;
            };
            if (a == "--bench") b.on = true;
            else if (a == "--oracle") b.oracle = true;
            else if (const char* s = val("--frames")) b.frames = std::atoi(s);
            else if (const char* s = val("--range")) b.range = static_cast<float>(std::atof(s));
            else if (const char* s = val("--range-at")) b.rangeAt = std::atoi(s);
            else if (const char* s = val("--min-h")) b.minRegHeight = static_cast<float>(std::atof(s));
        }
        return b;
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

        // The mast + marker must top out *below* the sensor origin so no mesh
        // encloses it. On Vulkan the robot is in the path-traced TLAS, and an
        // enclosing mesh would make every beam hit it at ~0 m (the whole scan
        // becomes self-returns). The AxesHelper is an unlit overlay (never in
        // the TLAS), so it marks the sensor without being scanned.
        auto mast = Mesh::create(BoxGeometry::create(0.1f, 0.45f, 0.1f), mastMat);
        mast->position.y = 0.625f;// spans 0.40 .. 0.85
        robot->add(mast);

        auto marker = Mesh::create(BoxGeometry::create(0.24f, 0.1f, 0.24f), sensorMat);
        marker->position.y = 0.85f;// top at 0.90, clears the sensor at kSensorHeight (1.0)
        robot->add(marker);

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
    const Bench bench = parseBench(argc, argv);

    Canvas canvas("Lidar SLAM", {{"antialiasing", 4}, {"vsync", false}});
    auto renderer = createRenderer(canvas);
    renderer->setScissorTest(true);
    renderer->shadowMap().enabled = false;

    // Pick the LIDAR sensor by backend: the raster LidarSensor (GL) is
    // cube-face based and unsupported on Vulkan, which uses the ray-traced
    // PathTracedLidarSensor instead. Everything downstream consumes LidarReturn,
    // so the SLAM + meshing is identical on both.
    bool isVulkan = false;
#ifdef THREEPP_WITH_VULKAN
    auto* vk = dynamic_cast<VulkanRenderer*>(renderer.get());
    isVulkan = (vk != nullptr);
#endif

    // --- Ground-truth scene (left) ---
    Scene sceneLeft;
    sceneLeft.background = Color(0x1d2330);
    buildRoom(sceneLeft);

    auto robot = buildRobot();
    sceneLeft.add(robot);

    // LIDAR model (and, on the raster backend, scan resolution) are live-
    // selectable; changing either rebuilds the sensor.
    const char* modelNames[] = {"OS1-64", "OS0-128", "HDL-32E", "VLP-16"};
    int currentModel = 0;
    const int faceSizeOptions[] = {192, 256, 384, 512};
    const char* faceSizeNames[] = {"192", "256", "384", "512"};
    int currentFaceIdx = 2;// 384
    auto makeModel = [&]() -> LidarModel {
        switch (currentModel) {
            case 1: return LidarModel::OS0_128();
            case 2: return LidarModel::HDL32E();
            case 3: return LidarModel::VLP16();
            default: return LidarModel::OS1_64();
        }
    };

    // Exactly one of these is live; `sensorObj` is the active sensor's Object3D
    // (its pose is driven each frame).
    std::unique_ptr<LidarSensor> rasterLidar;
#ifdef THREEPP_WITH_VULKAN
    std::unique_ptr<PathTracedLidarSensor> ptLidar;
#endif
    Object3D* sensorObj = nullptr;
    // Detector max range, live-tunable and shared by both backends: on Vulkan it
    // shortens the traced beam (params.maxRange), on GL it gates the returns the
    // SLAM is allowed to see. Shortening it is the demo's sharpest stress — once
    // the scan no longer reaches far enough to see structure along the direction
    // of travel, that axis stops being observable and the estimate walks away
    // from truth along it (see the Bench block above).
    float detectorRange = kSensorFar;
    auto buildSensor = [&] {
#ifdef THREEPP_WITH_VULKAN
        if (isVulkan) {
            ptLidar = std::make_unique<PathTracedLidarSensor>(makeModel(), detectorRange);
            ptLidar->params.referenceRange = 5.f;
            ptLidar->params.detectorThreshold = 0.005f;
            sensorObj = ptLidar.get();
            sceneLeft.addRef(*ptLidar);
            return;
        }
#endif
        rasterLidar = std::make_unique<LidarSensor>(
                makeModel(), static_cast<unsigned int>(faceSizeOptions[currentFaceIdx]), kSensorNear, kSensorFar);
        rasterLidar->rangeNoise.stddev = 0.02f;
        sensorObj = rasterLidar.get();
        sceneLeft.addRef(*rasterLidar);
    };
    buildSensor();

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

    // The reconstruction overlays (map points + trajectories) live in the RIGHT
    // scene on both backends. On raster this is a true split (each pane its own
    // raster render); on Vulkan the left pane is the deferred-rendered ground truth
    // and the right pane is an overlay-only compose (points + lines) of a second
    // render into the right scissor region.
    Scene& reconScene = sceneRight;

    // View 0: raw point cloud.
    auto mapPoints = makePointCloud(0.05f);
    reconScene.add(mapPoints);

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
    reconScene.add(gtTraj);
    reconScene.add(estTraj);

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

    auto rebuildSensor = [&] {
        const float noise = rasterLidar ? rasterLidar->rangeNoise.stddev : 0.02f;
        if (sensorObj) sceneLeft.remove(*sensorObj);
        rasterLidar.reset();
#ifdef THREEPP_WITH_VULKAN
        ptLidar.reset();
#endif
        buildSensor();// pose is re-applied next frame
        if (rasterLidar) rasterLidar->rangeNoise.stddev = noise;
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
    RendererSettingsUi ui(canvas, *renderer, [&] {
        ImGui::Text("Left: ground truth   Right: reconstruction");
        ImGui::Text("Sensor: %s", isVulkan ? "path-traced (Vulkan)" : "raster (cube-face)");
        ImGui::Separator();
        ImGui::Checkbox("Pause robot", &paused);
        ImGui::SameLine();
        ImGui::Checkbox("Show live scan", &showLive);
        ImGui::SliderFloat("Speed (m/s)", &speed, 0.2f, 2.5f);

        int prevModel = currentModel;
        ImGui::Combo("LIDAR", &currentModel, modelNames, 4);
        if (currentModel != prevModel) rebuildSensor();

        if (rasterLidar) {
            int prevFace = currentFaceIdx;
            ImGui::Combo("Scan res", &currentFaceIdx, faceSizeNames, 4);
            if (currentFaceIdx != prevFace) rebuildSensor();
            ImGui::SliderFloat("Range noise (m)", &rasterLidar->rangeNoise.stddev, 0.f, 0.1f);
        }
        // On GL the raster sensor's depth far plane is fixed at kSensorFar, so
        // there is nothing to see past it; only the ray-traced backend can be
        // pushed further out.
        if (ImGui::SliderFloat("Max range (m)", &detectorRange, 5.f, isVulkan ? 50.f : kSensorFar)) {
#ifdef THREEPP_WITH_VULKAN
            if (ptLidar) ptLidar->params.maxRange = detectorRange;
#endif
        }

        if (isVulkan) {
            // The path-traced backend can't draw the dynamic occupancy/surface
            // meshes in the reconstruction scene; it shows the point cloud.
            ImGui::TextWrapped("Map view: point cloud (occupancy/surface meshing needs a raster backend)");
        } else {
            const char* viewNames[] = {"Point cloud", "Occupancy cubes", "Surface (marching cubes)"};
            ImGui::Combo("Map view", &viewMode, viewNames, 3);
        }

        if (ImGui::Button("Reset SLAM")) resetReconstruction();
        ImGui::Separator();
        ImGui::Text("Map points : %d", mapCount);
        ImGui::Text("Drift      : %.3f m", drift);
        ImGui::Text("Scan  : %.1f ms", scanMs);
        ImGui::Text("SLAM  : %.1f ms", slamMs);
    }, "SLAM");

    canvas.onWindowResize([&](WindowSize size) {
        renderer->setSize(size);
    });

    Clock clock;
    float prevTime = 0.f;
    std::vector<LidarReturn> cloud;
    std::vector<Vector3> added;

    // Bench accumulators (unused in the interactive run).
    Matrix4 prevGt;
    bool hasPrevGt = false;
    int benchFrame = 0;
    double driftSum = 0.0;
    float driftMax = 0.f;
    float yawErrMax = 0.f;

    canvas.animate([&] {
        const float now = clock.getElapsedTime();
        float dt = now - prevTime;
        prevTime = now;
        dt = std::min(dt, 0.05f);

        if (bench.on) {
            dt = 1.f / 60.f;// fixed step: the A/B must measure the estimator, not frame pacing
            if (benchFrame == bench.rangeAt) {
                detectorRange = bench.range;
#ifdef THREEPP_WITH_VULKAN
                if (ptLidar) ptLidar->params.maxRange = detectorRange;
#endif
                std::cout << "# frame " << benchFrame << ": max range -> " << detectorRange << " m\n";
            }
        }

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

        sensorObj->position.set(px, kSensorHeight, pz);
        sensorObj->rotation.set(0, yaw, 0);
        robot->position.set(px, 0, pz);
        robot->rotation.set(0, yaw, 0);

        // Make the ground-truth sensor pose current, then capture it. Its
        // inverse de-frames the world-space returns back to sensor-local.
        sensorObj->updateMatrixWorld(true);
        Matrix4 mGt;
        mGt.copy(*sensorObj->matrixWorld);
        Matrix4 mInv;
        mInv.copy(mGt).invert();

        // View geometry: both backends split the window in half (ground truth
        // left, reconstruction right), so each pane gets the half-width aspect.
        // On Vulkan the left pane is the deferred-rendered view and the right pane is
        // an overlay-only compose of a second render into the right region.
        const auto size = canvas.size();
        const int w = size.width();
        const int h = size.height();
        const int halfW = w / 2;
        camera->aspect = static_cast<float>(halfW) / static_cast<float>(h);
        camera->updateProjectionMatrix();
        controls.update();

        // --- Scan ---
        if (isVulkan) {
#ifdef THREEPP_WITH_VULKAN
            // Path-traced scan reads the renderer's TLAS, so render the ground-
            // truth scene first — into the LEFT pane (the Vulkan view). The right
            // pane (reconstruction) is composed afterwards in the render step.
            // Points are overlay-only (never in the TLAS); the robot is, but its
            // near self-returns are dropped below.
            livePoints->visible = showLive;
            renderer->setViewport(0, 0, halfW, h);
            renderer->setScissor(0, 0, halfW, h);
            renderer->render(sceneLeft, *camera);
            const float t0 = clock.getElapsedTime();
            ptLidar->scan(*vk, cloud);
            scanMs = (clock.getElapsedTime() - t0) * 1000.f;
#endif
        } else {
            // Raster cube-face scan renders its own views; hide the robot +
            // overlay so the sensor doesn't see itself.
            const float t0 = clock.getElapsedTime();
            robot->visible = false;
            livePoints->visible = false;
            rasterLidar->scan(*renderer, sceneLeft, cloud);
            robot->visible = true;
            livePoints->visible = showLive;
            scanMs = (clock.getElapsedTime() - t0) * 1000.f;
        }

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
        // Vulkan keeps the robot in the TLAS, so use a wider self-return cut to
        // drop its near hits (the path stays >1.5 m from real geometry).
        const float selfRange = isVulkan ? 1.1f : kSelfReturnRange;
        const auto localScan = deframeScan(cloud, mInv, selfRange, detectorRange);
        const auto mapSrc = voxelDownsample(localScan, kDownsampleVoxel);
        // Chained: decimating the 0.25 m result at 0.4 m ≈ decimating the raw
        // ~50k-point scan at 0.4 m, minus one full pass over it.
        auto regSrc = voxelDownsample(mapSrc, kRegVoxel);
        if (bench.minRegHeight > -1e8f) {
            regSrc.erase(std::remove_if(regSrc.begin(), regSrc.end(),
                                        [&](const Vector3& p) { return p.y < bench.minRegHeight; }),
                         regSrc.end());
        }

        // --- SLAM update ---
        const float slamT0 = clock.getElapsedTime();
        if (!slamInitialised) {
            slam.reset(mGt);// anchor SLAM frame to GT start so paths overlay
            slamInitialised = true;
        }
        added.clear();
        // Oracle diagnostic: the exact body-frame motion since the previous
        // frame, i.e. a noiseless, drift-free motion sensor.
        Matrix4 oracleDelta;
        bool haveOracle = false;
        if (bench.oracle && hasPrevGt) {
            Matrix4 inv;
            inv.copy(prevGt).invert();
            oracleDelta.multiplyMatrices(inv, mGt);
            haveOracle = true;
        }
        prevGt.copy(mGt);
        hasPrevGt = true;
        const Matrix4& estPose = slam.process(regSrc, mapSrc, added, haveOracle ? &oracleDelta : nullptr);
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

        if (bench.on) {
            const float yawErr = std::abs(wrapPi(yawOf(estPose) - yawOf(mGt))) * math::RAD2DEG;
            driftSum += drift;
            driftMax = std::max(driftMax, drift);
            yawErrMax = std::max(yawErrMax, yawErr);
            if (benchFrame % 60 == 0) {
                // Per-axis, not just the magnitude: the whole diagnosis turned
                // on seeing estimated Z track truth while X ran away.
                const Vector3 err = estPos.clone().sub(gtPos);
                std::cout << "frame " << benchFrame
                          << "  gt (" << gtPos.x << ", " << gtPos.z << ")"
                          << "  est (" << estPos.x << ", " << estPos.z << ")"
                          << "  drift " << drift << " m"
                          << " (dx " << err.x << ", dy " << err.y << ", dz " << err.z << ")"
                          << "  yawErr " << yawErr << " deg"
                          << "  scanPts " << localScan.size()
                          << "  icp " << slam.lastIcp().iterations << "it/"
                          << slam.lastIcp().correspondences << "corr" << std::endl;
            }
            if (++benchFrame >= bench.frames) {
                std::cout << "BENCH range=" << bench.range << " at=" << bench.rangeAt
                          << " frames=" << bench.frames
                          << "  meanDrift " << (driftSum / benchFrame) << " m"
                          << "  maxDrift " << driftMax << " m"
                          << "  maxYawErr " << yawErrMax << " deg" << std::endl;
                canvas.close();
            }
        }

        // --- Map view: point cloud / occupancy cubes / surface ---
        if (isVulkan) viewMode = 0;// mesh views are raster-only; never show them on Vulkan
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

        // --- Render ---
        if (!isVulkan) {
            // Raster: ground truth left (Vulkan already rendered its left
            // pane in the scan step above).
            renderer->setViewport(0, 0, halfW, h);
            renderer->setScissor(0, 0, halfW, h);
            renderer->render(sceneLeft, *camera);
        }
        // Reconstruction right pane: a full raster render on GL, an overlay-only
        // compose on Vulkan (the renderer draws Points/Lines into the right
        // scissor region beside the left pane, leaving its temporal accumulation untouched).
        renderer->setViewport(halfW, 0, w - halfW, h);
        renderer->setScissor(halfW, 0, w - halfW, h);
        renderer->render(sceneRight, *camera);

        // UI spans the whole window.
        renderer->setViewport(0, 0, w, h);
        renderer->setScissor(0, 0, w, h);
        ui.render();
    });
}
