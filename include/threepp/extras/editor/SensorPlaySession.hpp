// The PlaySession that turns authored SensorConfig entries into live sensors.
//
// Header-only and PhysX-dependent, exactly like PhysicsPlaySession — the threepp
// library proper never links PhysX, so this file is included only by builds that
// found the SDK (the editor sets THREEPP_EDITOR_WITH_PHYSX). Everything that is
// PhysX-free lives in SensorConfig, which the inspector and the tests use on
// every platform.
//
// Sensors are PLAY-TIME CONSTRUCTS. They are built from userData at Play and
// dropped at Stop; nothing about a live sensor is ever serialized. That is not a
// limitation, it is the determinism story: a fresh sensor re-seeds its PRNG from
// the authored seed, so pressing Play twice on the same scene produces the same
// noise, the same cloud and the same recording. A sensor that survived across
// plays would carry its random-walked bias with it and quietly make the second
// run different from the first.
//
// Two drive models, because the suite has two (see Sensor.hpp):
//
//   pushed  — Imu, ContactSensor: registered with the PhysxWorld that
//             PhysicsPlaySession built, sampled from its fixed-substep loop the
//             instant body state is fresh. This session does not step physics
//             and does not touch the world's timestep.
//   pulled  — DepthSensor, LidarSensor: a scan needs a Renderer, which the
//             physics step loop has no business holding, so update() drives
//             their clock and rate gate itself and scans when scanDue() says so.
//
// The world is borrowed, never owned: PlayController stops sessions in
// registration order, and physics is registered FIRST, so by the time stop()
// runs here the PhysxWorld may already be gone. Hence the lifetime token — the
// world pointer is only ever dereferenced while PhysicsPlaySession's token is
// still alive.
//
// Editor furniture is hidden for the duration of every scan. A depth camera
// pointed at the viewport grid measures the grid; the sensor rig itself lives
// outside that hidden subtree so it is never hidden from itself.

#ifndef THREEPP_EDITOR_SENSORPLAYSESSION_HPP
#define THREEPP_EDITOR_SENSORPLAYSESSION_HPP

#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/PlaySession.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"

#include "threepp/extras/sensors/ContactSensor.hpp"
#include "threepp/extras/sensors/Imu.hpp"

#include "threepp/helpers/DepthSensor.hpp"
#include "threepp/helpers/LidarSensor.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/renderers/Renderer.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

namespace threepp::editor {

    class SensorPlaySession: public PlaySession {

    public:
        // A short history of one scalar channel, sized for an ImGui::PlotLines
        // call: `values` is a fixed ring and `offset` is the oldest sample, which
        // is exactly the (values, count, offset) triple that widget takes.
        //
        // Lives here rather than in the panel because the SESSION owns the drain:
        // drain() empties the sensor's ring, so whoever calls it is the only one
        // who can feed a plot. Two readers draining the same sensor would each
        // see half the data.
        struct Trace {
            static constexpr int capacity = 512;

            std::array<float, capacity> values{};
            int count = 0; // samples held, <= capacity
            int offset = 0;// index of the oldest sample

            void push(float v) {
                const int at = (count < capacity) ? count : offset;
                values[static_cast<std::size_t>(at)] = v;
                if (count < capacity) {
                    ++count;
                } else {
                    offset = (offset + 1) % capacity;
                }
            }

            void clear() {
                count = 0;
                offset = 0;
            }
        };

        static constexpr int maxTraces = 6;

        struct Entry {
            // The authored object, by uuid AND by pointer. The uuid is what
            // survives; the pointer is only valid between start() and stop(),
            // which is the same contract Sensor itself has with its node.
            std::string uuid;
            std::string label;
            SensorConfig config;
            Object3D* node = nullptr;

            std::unique_ptr<Imu> imu;
            std::unique_ptr<ContactSensor> contact;
            // Object3D subclasses: parented into the rig, so they follow the
            // scene's world-matrix update and are never saved or picked.
            std::unique_ptr<DepthSensor> depth;
            std::unique_ptr<LidarSensor> lidar;

            // Why this entry is not measuring, when it is not. Empty = fine.
            std::string status;

            // Counters for the readout.
            std::size_t samples = 0;// measurements drained so far
            std::size_t scans = 0;  // completed vision scans
            double lastTime = 0.0;  // sim time of the newest measurement

            // Latest vision output, world space. Rewritten in place per scan.
            std::vector<Vector3> cloud;       // depth
            std::vector<LidarReturn> returns; // lidar

            // Contact latch, for the readout's touch light.
            bool inContact = false;
            float contactForce = 0.f;

            std::array<Trace, maxTraces> traces{};
            std::array<const char*, maxTraces> traceNames{};
            int traceCount = 0;

            // CSV recording. One file per sensor; opened on the first drained
            // measurement after Record goes on, closed and flushed on Stop.
            std::ofstream csv;
            std::filesystem::path csvPath;
            std::size_t rows = 0;

            [[nodiscard]] bool live() const {
                return imu || contact || depth || lidar;
            }

            // Points the newest scan produced, whichever vision kind this is.
            [[nodiscard]] std::size_t pointCount() const {
                return depth ? cloud.size() : returns.size();
            }
        };

        [[nodiscard]] std::string name() const override { return "Sensors"; }

        // The editor being torn down mid-Play never gets a stop(). The rig
        // outlives this session by construction (it is editor furniture), so the
        // sensor nodes have to come out of its children list before they die.
        ~SensorPlaySession() override {

            for (const auto& entry : entries_) {
                if (entry->depth) entry->depth->removeFromParent();
                if (entry->lidar) entry->lidar->removeFromParent();
            }
        }

        // --- wiring (set once, before the first Play) ------------------------

        // Where the pushed sensors register. Borrowed; the token this session
        // reads from it is what keeps a stopped world from being dereferenced.
        void setPhysics(PhysicsPlaySession* physics) { physics_ = physics; }

        // The renderer the vision sensors scan with. Without one they are still
        // built (and say so) but never scan — a headless test, or a build whose
        // backend cannot do a depth pass.
        void setRenderer(Renderer* renderer) { renderer_ = renderer; }

        // Where the sensor nodes are parented: an editor-only Group, so they are
        // excluded from every export and snapshot and never picked. MUST NOT be
        // inside the subtree passed to setHiddenDuringScan.
        void setRig(Object3D* rig) { rig_ = rig; }

        // Editor furniture (grid, gizmo, markers, the point-cloud overlay) that
        // a sensor must not measure. Hidden for the duration of each scan and
        // restored immediately afterwards.
        void setHiddenDuringScan(Object3D* node) { hidden_ = node; }

        void setLogger(std::function<void(const std::string&)> logger) {
            logger_ = std::move(logger);
        }

        // --- state -----------------------------------------------------------

        [[nodiscard]] const std::vector<std::unique_ptr<Entry>>& entries() const { return entries_; }

        [[nodiscard]] std::size_t sensorCount() const { return entries_.size(); }

        // Sensors that actually came up. An authored entry whose body could not
        // be resolved counts in sensorCount() and not here.
        [[nodiscard]] std::size_t liveCount() const {
            return static_cast<std::size_t>(
                    std::count_if(entries_.begin(), entries_.end(),
                                  [](const std::unique_ptr<Entry>& e) { return e->live(); }));
        }

        // The clock every measurement is stamped with: the physics world's
        // accumulated substep time while one exists, otherwise this session's own
        // accumulation of the frame delta. Sim time, never wall time.
        [[nodiscard]] double simTime() const { return simTime_; }

        // --- recording -------------------------------------------------------

        void setRecordDirectory(const std::filesystem::path& dir) { recordDir_ = dir; }
        [[nodiscard]] const std::filesystem::path& recordDirectory() const { return recordDir_; }

        // Arming is legal while stopped: the files are opened on the first
        // measurement of the next Play, so "Record then Play" captures from t=0.
        void setRecording(bool on) {
            if (recording_ == on) return;
            recording_ = on;
            if (!on) closeFiles();
        }
        [[nodiscard]] bool recording() const { return recording_; }

        [[nodiscard]] std::size_t recordedRows() const {
            std::size_t rows = 0;
            for (const auto& entry : entries_) rows += entry->rows;
            return rows;
        }

        // --- PlaySession -----------------------------------------------------

        void start(Scene& scene) override {

            scene_ = &scene;
            entries_.clear();
            simTime_ = 0.0;
            world_ = nullptr;
            worldLife_.reset();

            if (physics_) {
                world_ = physics_->world();
                worldLife_ = physics_->lifetime();
            }

            // Collect first, build second: registering a sensor can throw, and a
            // stable list keeps the build order (and therefore every seeded
            // stream) independent of anything a hook might do to the graph.
            scene.updateMatrixWorld(true);
            std::vector<Object3D*> targets;
            scene.traverse([&](Object3D& object) {
                if (const auto config = SensorConfig::read(object); config && config->enabled) {
                    targets.push_back(&object);
                }
            });

            for (auto* node : targets) {
                const auto config = SensorConfig::read(*node);
                if (!config) continue;
                build(*node, *config);
            }
        }

        void update(float dt) override {

            if (entries_.empty()) return;

            // One clock for the whole rig. The pushed sensors are stamped by the
            // world itself; reading the same number here keeps a scan's timestamp
            // on the same time base as the IMU sample beside it.
            if (auto* world = liveWorld()) {
                simTime_ = world->simTime();
            } else {
                simTime_ += static_cast<double>(dt);
            }

            scanAll(dt);
            drainAll();
        }

        void stop() override {

            // Unregister only while the world is provably alive: physics stops
            // first, and a stopped session has already destroyed it.
            if (auto* world = liveWorld()) {
                for (const auto& entry : entries_) {
                    if (entry->imu) world->unregisterSensor(entry->imu.get());
                    if (entry->contact) world->unregisterSensor(entry->contact.get());
                }
            }

            closeFiles();

            // Detach the rig nodes before the sensors are destroyed: the rig
            // survives Stop (it is editor furniture) and must not be left
            // holding raw pointers into freed sensors.
            for (const auto& entry : entries_) {
                if (entry->depth) entry->depth->removeFromParent();
                if (entry->lidar) entry->lidar->removeFromParent();
            }

            entries_.clear();
            scene_ = nullptr;
            world_ = nullptr;
            worldLife_.reset();
            simTime_ = 0.0;
        }

    private:
        void log(const std::string& message) {

            if (logger_) logger_(message);
        }

        // The borrowed world, or nullptr once PhysicsPlaySession has stopped.
        [[nodiscard]] PhysxWorld* liveWorld() const {

            if (!world_) return nullptr;
            return worldLife_.expired() ? nullptr : world_;
        }

        static std::string labelFor(const Object3D& node) {

            if (!node.name.empty()) return node.name;
            return "sensor " + node.uuid.substr(0, 8);
        }

        NoiseModel gyroNoise(const SensorConfig& config) const {

            NoiseModel model;
            const float d = std::max(config.gyroNoiseDensity, 0.f);
            const float w = std::max(config.gyroRandomWalk, 0.f);
            model.whiteNoiseDensity.set(d, d, d);
            model.randomWalk.set(w, w, w);
            model.seed = config.streamSeed(0);
            return model;
        }

        NoiseModel accelNoise(const SensorConfig& config) const {

            NoiseModel model;
            const float d = std::max(config.accelNoiseDensity, 0.f);
            const float w = std::max(config.accelRandomWalk, 0.f);
            model.whiteNoiseDensity.set(d, d, d);
            model.randomWalk.set(w, w, w);
            model.seed = config.streamSeed(1);
            return model;
        }

        static RangeNoiseModel rangeNoise(const SensorConfig& config) {

            RangeNoiseModel model;
            model.stddev = std::max(config.rangeStddev, 0.f);
            model.stddevPerMetre = std::max(config.rangeStddevPerMetre, 0.f);
            model.bias = config.rangeBias;
            model.seed = config.streamSeed(2);
            return model;
        }

        static LidarModel beamModel(SensorConfig::Beams beams) {

            switch (beams) {
                case SensorConfig::Beams::HDL32E: return LidarModel::HDL32E();
                case SensorConfig::Beams::OS1_64: return LidarModel::OS1_64();
                case SensorConfig::Beams::OS0_128: return LidarModel::OS0_128();
                case SensorConfig::Beams::Dense:  // handled by the caller
                case SensorConfig::Beams::VLP16:
                default: return LidarModel::VLP16();
            }
        }

        void build(Object3D& node, const SensorConfig& config) {

            auto entry = std::make_unique<Entry>();
            entry->uuid = node.uuid;
            entry->label = labelFor(node);
            entry->config = config;
            entry->node = &node;

            switch (config.type) {
                case SensorConfig::Type::Imu: buildImu(*entry, node, config); break;
                case SensorConfig::Type::Contact: buildContact(*entry, node, config); break;
                case SensorConfig::Type::Depth: buildDepth(*entry, config); break;
                case SensorConfig::Type::Lidar: buildLidar(*entry, config); break;
                case SensorConfig::Type::Encoder:
                case SensorConfig::Type::ForceTorque:
                    // Both measure an ARTICULATION joint, and the editor's
                    // physics session builds rigid bodies only — there is no
                    // joint to read. Authored and saved, not simulated.
                    entry->status = std::string(SensorConfig::label(config.type)) +
                                    " needs an articulated robot - authored, not simulated";
                    break;
            }

            if (!entry->status.empty()) log("sensor: \"" + entry->label + "\" - " + entry->status);
            entries_.push_back(std::move(entry));
        }

        void buildImu(Entry& entry, Object3D& node, const SensorConfig& config) {

            auto* world = liveWorld();
            if (!world) {
                entry.status = "no physics world - an IMU rides a rigid body";
                return;
            }

            auto imu = std::make_unique<Imu>(node, static_cast<double>(std::max(config.rateHz, 0.f)));
            imu->gyroNoise = gyroNoise(config);
            imu->accelNoise = accelNoise(config);
            try {
                // onRegister resolves the rigid body and throws when there is
                // none, which is the mistake worth surfacing at Play rather than
                // as a sensor that silently reads zeros.
                world->registerSensor(imu.get());
            } catch (const std::exception& e) {
                entry.status = e.what();
                return;
            }
            entry.imu = std::move(imu);

            entry.traceNames = {"gyro x", "gyro y", "gyro z", "accel x", "accel y", "accel z"};
            entry.traceCount = 6;
        }

        void buildContact(Entry& entry, Object3D& node, const SensorConfig& config) {

            auto* world = liveWorld();
            if (!world) {
                entry.status = "no physics world - a contact sensor rides a rigid body";
                return;
            }

            auto contact = std::make_unique<ContactSensor>(
                    node, static_cast<double>(std::max(config.rateHz, 0.f)));
            try {
                world->registerSensor(contact.get());
            } catch (const std::exception& e) {
                entry.status = e.what();
                return;
            }
            entry.contact = std::move(contact);

            entry.traceNames = {"force (N)", "touching", nullptr, nullptr, nullptr, nullptr};
            entry.traceCount = 2;
        }

        void buildDepth(Entry& entry, const SensorConfig& config) {

            if (!rig_) {
                entry.status = "no sensor rig - the editor did not wire one up";
                return;
            }

            const auto width = static_cast<unsigned>(
                    std::clamp(config.width, 8, SensorConfig::maxImageSize));
            const auto height = static_cast<unsigned>(
                    std::clamp(config.height, 8, SensorConfig::maxImageSize));

            auto depth = std::make_unique<DepthSensor>(
                    std::clamp(config.fovY, 1.f, 179.f), width, height,
                    std::max(config.nearPlane, 1e-3f),
                    std::max(config.farPlane, std::max(config.nearPlane, 1e-3f) + 1e-3f));
            depth->name = entry.label + " (depth)";
            depth->rangeNoise = rangeNoise(config);
            depth->resetNoise();
            depth->setRateHz(static_cast<double>(std::max(config.rateHz, 0.f)));
            rig_->addRef(*depth);
            entry.depth = std::move(depth);

            entry.traceNames = {"points", nullptr, nullptr, nullptr, nullptr, nullptr};
            entry.traceCount = 1;
            if (!renderer_) entry.status = "no renderer - built but not scanning";
        }

        void buildLidar(Entry& entry, const SensorConfig& config) {

            if (!rig_) {
                entry.status = "no sensor rig - the editor did not wire one up";
                return;
            }

            const auto faceSize = static_cast<unsigned>(
                    std::clamp(config.faceSize, 16, SensorConfig::maxFaceSize));
            const float near = std::max(config.nearPlane, 1e-3f);
            const float far = std::max(config.farPlane, near + 1e-3f);

            auto lidar = (config.beams == SensorConfig::Beams::Dense)
                                 ? std::make_unique<LidarSensor>(faceSize, near, far)
                                 : std::make_unique<LidarSensor>(beamModel(config.beams),
                                                                 faceSize, near, far);
            lidar->name = entry.label + " (lidar)";
            lidar->rangeNoise = rangeNoise(config);
            lidar->resetNoise();
            lidar->setRateHz(static_cast<double>(std::max(config.rateHz, 0.f)));
            rig_->addRef(*lidar);
            entry.lidar = std::move(lidar);

            entry.traceNames = {"returns", nullptr, nullptr, nullptr, nullptr, nullptr};
            entry.traceCount = 1;
            if (!renderer_) entry.status = "no renderer - built but not scanning";
        }

        // Park the sensor node on the authored object's world pose. Scale is
        // dropped deliberately: a scaled sensor would scale its own ray
        // directions, and "the sensor is 2x bigger" is not a thing a sensor is.
        void placeVision(Entry& entry) {

            Object3D* sensorNode = entry.depth ? static_cast<Object3D*>(entry.depth.get())
                                               : static_cast<Object3D*>(entry.lidar.get());
            if (!sensorNode || !entry.node) return;

            entry.node->updateWorldMatrix(true, false);
            Vector3 position, scale;
            Quaternion rotation;
            entry.node->matrixWorld->decompose(position, rotation, scale);

            sensorNode->position.copy(position);
            sensorNode->quaternion.copy(rotation);
            sensorNode->scale.set(1.f, 1.f, 1.f);
            // The rig is an identity-transformed child of the scene root, so the
            // local pose above IS the world pose. Force it down into the child
            // cameras now — the scan reads their world matrices directly.
            sensorNode->updateWorldMatrix(true, true);
        }

        void scanAll(float dt) {

            if (!renderer_ || !scene_) return;

            // Arm the gate and latch the clock on the pulled sensors. They are
            // deliberately NOT registered with the world (that would tick them
            // twice); tick() is the same entry point the world would use.
            bool any = false;
            for (const auto& entry : entries_) {
                if (entry->depth) {
                    entry->depth->tick(static_cast<double>(dt), simTime_);
                    any = any || entry->depth->scanDue();
                }
                if (entry->lidar) {
                    entry->lidar->tick(static_cast<double>(dt), simTime_);
                    any = any || entry->lidar->scanDue();
                }
            }
            if (!any) return;

            // A depth camera aimed at the viewport grid measures the grid. Hide
            // the editor's furniture for exactly as long as the scans take.
            const bool wasVisible = hidden_ ? hidden_->visible : true;
            if (hidden_) hidden_->visible = false;

            for (const auto& entry : entries_) {
                if (entry->depth && entry->depth->scanDue()) {
                    placeVision(*entry);
                    entry->depth->scan(*renderer_, *scene_, entry->cloud);
                    entry->lastTime = entry->depth->lastScanTime();
                    ++entry->scans;
                    entry->traces[0].push(static_cast<float>(entry->cloud.size()));
                    recordVision(*entry);
                }
                if (entry->lidar && entry->lidar->scanDue()) {
                    placeVision(*entry);
                    entry->lidar->scan(*renderer_, *scene_, entry->returns);
                    entry->lastTime = entry->lidar->lastScanTime();
                    ++entry->scans;
                    entry->traces[0].push(static_cast<float>(entry->returns.size()));
                    recordVision(*entry);
                }
            }

            if (hidden_) hidden_->visible = wasVisible;
        }

        void drainAll() {

            for (const auto& entry : entries_) {
                if (entry->imu) drainImu(*entry);
                if (entry->contact) drainContact(*entry);
            }
        }

        void drainImu(Entry& entry) {

            entry.imu->drain(imuScratch_);
            if (imuScratch_.empty()) return;

            for (const auto& sample : imuScratch_) {
                entry.traces[0].push(sample.angularVelocity.x);
                entry.traces[1].push(sample.angularVelocity.y);
                entry.traces[2].push(sample.angularVelocity.z);
                entry.traces[3].push(sample.linearAcceleration.x);
                entry.traces[4].push(sample.linearAcceleration.y);
                entry.traces[5].push(sample.linearAcceleration.z);
                if (openFile(entry, "t,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z")) {
                    entry.csv << sample.t << ','
                              << sample.angularVelocity.x << ',' << sample.angularVelocity.y << ','
                              << sample.angularVelocity.z << ','
                              << sample.linearAcceleration.x << ',' << sample.linearAcceleration.y
                              << ',' << sample.linearAcceleration.z << '\n';
                    ++entry.rows;
                }
            }
            entry.samples += imuScratch_.size();
            entry.lastTime = imuScratch_.back().t;
        }

        void drainContact(Entry& entry) {

            entry.contact->drain(contactScratch_);
            if (contactScratch_.empty()) return;

            for (const auto& sample : contactScratch_) {
                const float force = sample.force.length();
                entry.traces[0].push(force);
                entry.traces[1].push(sample.inContact ? 1.f : 0.f);
                if (openFile(entry, "t,in_contact,force,fx,fy,fz,points")) {
                    entry.csv << sample.t << ',' << (sample.inContact ? 1 : 0) << ',' << force << ','
                              << sample.force.x << ',' << sample.force.y << ',' << sample.force.z
                              << ',' << sample.pointCount << '\n';
                    ++entry.rows;
                }
            }
            entry.samples += contactScratch_.size();
            const auto& last = contactScratch_.back();
            entry.inContact = last.inContact;
            entry.contactForce = last.force.length();
            entry.lastTime = last.t;
        }

        // One row per scan: time, count and the cloud's gross shape. A row per
        // RETURN would be tens of thousands of lines a second and is a point-
        // cloud format's job, not a time series'.
        void recordVision(Entry& entry) {

            if (!recording_) return;
            if (!openFile(entry, "t,points,centroid_x,centroid_y,centroid_z,range_min,range_max")) return;

            std::size_t count = 0;
            Vector3 sum;
            float rmin = 0.f, rmax = 0.f;

            if (entry.depth) {
                Vector3 origin;
                entry.depth->getWorldPosition(origin);
                for (const auto& p : entry.cloud) {
                    sum.add(p);
                    const float r = p.distanceTo(origin);
                    if (count == 0 || r < rmin) rmin = r;
                    if (count == 0 || r > rmax) rmax = r;
                    ++count;
                }
            } else if (entry.lidar) {
                for (const auto& r : entry.returns) {
                    sum.add(r.position);
                    if (count == 0 || r.distance < rmin) rmin = r.distance;
                    if (count == 0 || r.distance > rmax) rmax = r.distance;
                    ++count;
                }
            }
            if (count > 0) sum.multiplyScalar(1.f / static_cast<float>(count));

            entry.csv << entry.lastTime << ',' << count << ',' << sum.x << ',' << sum.y << ','
                      << sum.z << ',' << rmin << ',' << rmax << '\n';
            ++entry.rows;
        }

        // Opens the entry's CSV on first use, writing `header`. Returns false
        // when recording is off or the file could not be opened (reported once).
        bool openFile(Entry& entry, const char* header) {

            if (!recording_) return false;
            if (entry.csv.is_open()) return true;

            std::error_code ec;
            std::filesystem::create_directories(recordDir_, ec);

            entry.csvPath = recordDir_ / (sanitize(entry.label) + "_" +
                                          entry.uuid.substr(0, 8) + ".csv");
            entry.csv.open(entry.csvPath, std::ios::out | std::ios::trunc);
            if (!entry.csv.is_open()) {
                log("sensor: cannot write " + entry.csvPath.string());
                // Turn recording off wholesale rather than retry the open on
                // every measurement of every frame.
                recording_ = false;
                return false;
            }
            // Default ostream precision is 6 SIGNIFICANT digits, which turns a
            // sim time of 12.3456789 s into 12.3457 — a millisecond-resolution
            // dataset quantized to a tenth of a millisecond for no reason.
            entry.csv << std::setprecision(9);
            entry.csv << header << '\n';
            return true;
        }

        void closeFiles() {

            for (const auto& entry : entries_) {
                if (!entry->csv.is_open()) continue;
                // The time series says how the cloud behaved; this says what the
                // last one WAS. Written on close so a recording of any length
                // costs one cloud on disk instead of one per scan.
                dumpCloud(*entry);
                entry->csv.flush();
                entry->csv.close();
            }
        }

        static void dumpCloud(Entry& entry) {

            if (entry.pointCount() == 0) return;

            auto path = entry.csvPath;
            path.replace_filename(entry.csvPath.stem().string() + "_cloud.csv");
            std::ofstream out(path, std::ios::out | std::ios::trunc);
            if (!out.is_open()) return;

            out << std::setprecision(9);
            out << "t,x,y,z,range\n";
            if (entry.depth) {
                Vector3 origin;
                entry.depth->getWorldPosition(origin);
                for (const auto& p : entry.cloud) {
                    out << entry.lastTime << ',' << p.x << ',' << p.y << ',' << p.z << ','
                        << p.distanceTo(origin) << '\n';
                }
            } else {
                for (const auto& r : entry.returns) {
                    out << entry.lastTime << ',' << r.position.x << ',' << r.position.y << ','
                        << r.position.z << ',' << r.distance << '\n';
                }
            }
        }

        // A label becomes a filename, and object names carry whatever the user
        // typed. Keep the readable characters, fold the rest to '_'.
        static std::string sanitize(const std::string& text) {

            std::string out;
            for (const char c : text) {
                const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                (c >= '0' && c <= '9') || c == '-' || c == '.';
                out += ok ? c : '_';
            }
            if (out.empty()) out = "sensor";
            return out;
        }

        PhysicsPlaySession* physics_ = nullptr;
        Renderer* renderer_ = nullptr;
        Object3D* rig_ = nullptr;
        Object3D* hidden_ = nullptr;
        Scene* scene_ = nullptr;
        std::function<void(const std::string&)> logger_;

        PhysxWorld* world_ = nullptr;
        std::weak_ptr<const void> worldLife_;

        std::vector<std::unique_ptr<Entry>> entries_;
        double simTime_ = 0.0;

        bool recording_ = false;
        std::filesystem::path recordDir_ = std::filesystem::temp_directory_path() / "threepp-sensors";

        // Drain targets, reused so a 1 kHz sensor does not allocate per frame.
        std::vector<ImuSample> imuScratch_;
        std::vector<ContactSample> contactScratch_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SENSORPLAYSESSION_HPP
