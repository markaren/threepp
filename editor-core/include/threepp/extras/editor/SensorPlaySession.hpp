// The PlaySession that turns authored SensorConfig entries into live sensors.
//
// This header is PhysX-FREE, and that is the point of its shape. The vision
// sensors — depth, lidar and the colour camera — are renderer constructs: a
// scan needs a Renderer and a Scene, not a physics world, so a build without
// the PhysX SDK still authors, plays, overlays and records them. The four body/joint sensors (IMU,
// contact, encoder, force/torque) do need the world, so their construction and
// draining are virtual hooks: this class answers with a status line naming the
// missing build, and PhysxSensorPlaySession overrides them with the real thing.
//
// The split is by CLASS, not by #ifdef. A header-only class whose inline
// methods change under a per-target macro is an ODR violation waiting for two
// targets to disagree — the V-HACD macro rode exactly that path once — so each
// header compiles the same way in every TU, and the one #ifdef left is the
// editor choosing WHICH class to construct. Entry still carries the body
// sensors' pointers (forward-declared, shared_ptr so destruction never needs
// the complete type) because the panel, the overlay and the tests all consume
// ONE entry list.
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
//             instant body state is fresh. Lives in PhysxSensorPlaySession.
//   pulled  — DepthSensor, LidarSensor, CameraSensor: a scan needs a Renderer,
//             which the physics step loop has no business holding, so update()
//             drives their clock and rate gate itself and scans when scanDue()
//             says so. The two RANGING ones share a fire-on-one-frame,
//             deliver-on-a-later-one protocol (see scanAll) because a traced
//             scan means waiting on a fence; a colour capture is a render plus
//             a readback, both synchronous, and takes its own shorter path.
//
// Editor furniture is hidden for the duration of every scan and every capture.
// A depth camera pointed at the viewport grid measures the grid and a colour
// camera photographs it; the sensor rig itself lives outside that hidden
// subtree so it is never hidden from itself.

#ifndef THREEPP_EDITOR_SENSORPLAYSESSION_HPP
#define THREEPP_EDITOR_SENSORPLAYSESSION_HPP

#include "threepp/extras/editor/PlaySession.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/helpers/CameraSensor.hpp"
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
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

namespace threepp {

    // The body/joint sensors are PhysX types; only their pointers appear here.
    // shared_ptr rather than unique_ptr so ~Entry never needs the complete
    // types — the deleter is captured where PhysxSensorPlaySession constructs
    // them, which is the one place that includes their headers.
    class ContactSensor;
    class ForceTorqueSensor;
    class Imu;
    class JointEncoder;

}// namespace threepp

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

        // --- what a reader that is NOT the panel sees ------------------------
        //
        // drain() empties a sensor's ring, so the session is the only party that
        // may call it (see Trace). That is precisely why a second reader has to
        // be served from HERE: these are the measurements the session keeps
        // after draining them, so a play script holding a handle
        // (threepp.editor.imu_from_object) reads the same seeded, post-noise
        // stream the plots and the CSV are built from without stealing a sample
        // from either of them.
        //
        // Deliberately not the sensors' own sample types: those live in
        // PhysX-including headers and this class is PhysX-free by design. The
        // fields are the ones a control loop closes on; the CSV stays the place
        // to go for the manifold points and the edge flags.
        struct ImuReading {
            double t = 0.0;
            Vector3 angularVelocity;// rad/s, sensor frame
            Vector3 acceleration;   // specific force, m/s^2, sensor frame
        };

        struct EncoderReading {
            double t = 0.0;
            float position = 0.f;// rad (revolute) or m (prismatic)
            float velocity = 0.f;
        };

        struct WrenchReading {
            double t = 0.0;
            Vector3 force; // N, joint child frame
            Vector3 torque;// N*m, joint child frame
        };

        struct ContactReading {
            double t = 0.0;
            bool touching = false;// the LATCH, which survives the pair falling asleep
            Vector3 force;        // mean force over the interval (N); zero while asleep
        };

        // A short tail of retained measurements, carrying a monotonic sequence
        // number so several readers can each hold their own cursor: "what is new
        // since I last looked" must not depend on who else looked. Bounded like
        // the sensor's own ring — a reader that stops reading loses the oldest
        // samples rather than growing the session without limit.
        template<class T>
        class History {

        public:
            static constexpr std::size_t capacity = 256;

            void push(const T& value) {
                // Allocated on first use, so an entry that is not this kind of
                // sensor (and a vision entry, which is none of them) pays nothing.
                if (buf_.empty()) buf_.resize(capacity);
                buf_[total_ % capacity] = value;
                ++total_;
            }

            // Sequence number one past the newest sample: a cursor equal to it
            // has seen everything.
            [[nodiscard]] std::size_t total() const { return total_; }

            // Oldest sequence number still retained. A cursor older than this
            // missed samples and is fast-forwarded, rather than being handed a
            // window that silently starts in the wrong place.
            [[nodiscard]] std::size_t oldest() const {
                return total_ > capacity ? total_ - capacity : 0;
            }

            [[nodiscard]] bool empty() const { return total_ == 0; }

            // Precondition: !empty().
            [[nodiscard]] const T& newest() const { return buf_[(total_ - 1) % capacity]; }

            // Samples at or after `cursor`, oldest-first. Returns the cursor to
            // read from next time.
            std::size_t since(std::size_t cursor, std::vector<T>& out) const {
                out.clear();
                for (std::size_t i = std::max(cursor, oldest()); i < total_; ++i) {
                    out.push_back(buf_[i % capacity]);
                }
                return total_;
            }

        private:
            std::vector<T> buf_;
            std::size_t total_ = 0;
        };

        struct Entry {
            // The authored object, by uuid AND by pointer. The uuid is what
            // survives; the pointer is only valid between start() and stop(),
            // which is the same contract Sensor itself has with its node.
            std::string uuid;
            std::string label;
            SensorConfig config;
            Object3D* node = nullptr;

            std::shared_ptr<Imu> imu;
            std::shared_ptr<ContactSensor> contact;
            std::shared_ptr<JointEncoder> encoder;
            std::shared_ptr<ForceTorqueSensor> forceTorque;
            // Object3D subclasses: parented into the rig, so they follow the
            // scene's world-matrix update and are never saved or picked.
            std::unique_ptr<DepthSensor> depth;
            std::unique_ptr<LidarSensor> lidar;
            // Vision too, but not RANGING: a picture, not a cloud. Kept in its
            // own slot rather than folded into withVision() because it shares
            // none of the ranging scan protocol - no fire/deliver pair, no
            // RangeNoiseModel, no points.
            std::unique_ptr<CameraSensor> camera;

            // Why this entry is not measuring, when it is not. Empty = fine.
            std::string status;
            // Consecutive colour captures that returned nothing. A Vulkan
            // camera legitimately misses its first few (the secondary view
            // serving it warms up over a frame or two), so a status is only
            // worth raising once the misses stop looking like warm-up.
            int captureMisses = 0;

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

            // Retained measurements for readers outside the panel (see History).
            // Filled where the drain happens, so a handle never touches a live
            // sensor — or the PhysX state behind it — at all.
            History<ImuReading> imuReadings;
            History<EncoderReading> encoderReadings;
            History<WrenchReading> wrenchReadings;
            History<ContactReading> contactReadings;

            // CSV recording. One file per sensor; opened on the first drained
            // measurement after Record goes on, closed and flushed on Stop.
            std::ofstream csv;
            std::filesystem::path csvPath;
            std::size_t rows = 0;

            [[nodiscard]] bool live() const {
                return imu || contact || encoder || forceTorque || depth || lidar || camera;
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
        // The body sensors' unregistration is PhysxSensorPlaySession's half —
        // its destructor body runs before this one.
        ~SensorPlaySession() override {

            // A session destroyed without a stop() must not leave active()
            // pointing at freed memory.
            if (active_ == this) active_ = nullptr;

            for (const auto& entry : entries_) {
                detachVision(*entry);
            }
        }

        // --- wiring (set once, before the first Play) ------------------------

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

        // --- the seam a play script reaches its sensors through ---------------

        // The sensor session currently playing, or nullptr.
        //
        // The same seam as PhysicsPlaySession::active(), for the same reason: a
        // script reaches its object's sensors through a free function
        // (threepp.editor.imu_from_object) that has no context to resolve
        // against. The editor runs one Play at a time, so "the session that is
        // playing" is well defined. Set by start(), cleared by stop().
        [[nodiscard]] static SensorPlaySession* active() { return active_; }

        // A token that lives exactly as long as the entries the last start()
        // built. Handles handed to scripts keep a weak_ptr to it, so one held
        // across a Stop reports that it is gone rather than reading a freed
        // Entry.
        [[nodiscard]] std::weak_ptr<const void> lifetime() const { return lifetime_; }

        // The live sensors of `type` authored on `object`, or on the nearest
        // ancestor of it that carries one — the same walk-up contract as
        // PhysicsPlaySession::findActor, so a script sited on a child of an
        // instrumented link still finds the sensor measuring it.
        //
        // A list because ONE authored entry can be several live sensors: the
        // all-joints encoder fans out to one per DOF, each its own Entry (see
        // PhysxSensorPlaySession::buildEncoderFanout).
        [[nodiscard]] std::vector<const Entry*> findSensors(const Object3D* object,
                                                            SensorConfig::Type type) const {

            std::vector<const Entry*> found;
            for (const Object3D* o = object; o != nullptr; o = o->parent) {
                for (const auto& entry : entries_) {
                    if (entry->node == o && entry->config.type == type && entry->live()) {
                        found.push_back(entry.get());
                    }
                }
                if (!found.empty()) break;// nearest ancestor wins, as for a body
            }
            return found;
        }

        // The clock every measurement is stamped with: the physics world's
        // accumulated substep time while one exists (worldClock), otherwise this
        // session's own accumulation of the frame delta. Sim time, never wall
        // time.
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

            // Kill the previous token BEFORE the entries it pointed into, then
            // mint a fresh one: a handle from the previous Play stays dead even
            // though the same session object is starting again.
            lifetime_.reset();
            entries_.clear();
            simTime_ = 0.0;

            lifetime_ = std::make_shared<const char>('\0');
            active_ = this;

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
            if (!worldClock(simTime_)) {
                simTime_ += static_cast<double>(dt);
            }

            scanAll(dt);
            captureCameras(dt);
            drainBodies();
        }

        void stop() override {

            // PhysxSensorPlaySession has already unregistered the body sensors
            // (its stop() runs first and calls down here); what is left is the
            // renderer half and the shared bookkeeping.
            //
            // Drop the token BEFORE the entries: a handle a script is still
            // holding must read as dead from the moment the sensors go.
            lifetime_.reset();
            if (active_ == this) active_ = nullptr;

            closeFiles();

            // Detach the rig nodes before the sensors are destroyed: the rig
            // survives Stop (it is editor furniture) and must not be left
            // holding raw pointers into freed sensors.
            for (const auto& entry : entries_) {
                detachVision(*entry);
            }

            entries_.clear();
            scene_ = nullptr;
            simTime_ = 0.0;
        }

    protected:
        // --- the PhysX seam --------------------------------------------------
        //
        // The three places a body/joint sensor differs from a vision one. The
        // defaults are what a build without the SDK does; PhysxSensorPlaySession
        // overrides all three.

        // Build the sensors for one authored non-vision entry (the encoder
        // fan-out can produce several). Here: one entry that says which build it
        // is waiting for — authoring is respected everywhere, simulation is not
        // pretended anywhere.
        virtual void buildBodySensors(Object3D& node, const SensorConfig& config) {

            auto entry = makeEntry(node, config);
            entry->status = std::string(SensorConfig::label(config.type)) +
                            " needs the PhysX build - authored and saved, not run";
            commit(std::move(entry));
        }

        // Drain the pushed sensors into traces/CSV. Nothing to drain here.
        virtual void drainBodies() {}

        // Report the physics world's clock, if one is running. False leaves the
        // session accumulating the frame delta itself.
        virtual bool worldClock(double& /*time*/) { return false; }

        // --- shared machinery for the subclass -------------------------------

        std::unique_ptr<Entry> makeEntry(Object3D& node, const SensorConfig& config) {

            auto entry = std::make_unique<Entry>();
            entry->uuid = node.uuid;
            entry->label = labelFor(node);
            entry->config = config;
            entry->node = &node;
            return entry;
        }

        // Adopt a built entry: a non-empty status is worth a console line at
        // Play — it is the moment the user is looking.
        void commit(std::unique_ptr<Entry> entry) {

            if (!entry->status.empty()) log("sensor: \"" + entry->label + "\" - " + entry->status);
            entries_.push_back(std::move(entry));
        }

        void log(const std::string& message) {

            if (logger_) logger_(message);
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

        std::vector<std::unique_ptr<Entry>> entries_;

    private:
        static std::string labelFor(const Object3D& node) {

            if (!node.name.empty()) return node.name;
            return "sensor " + node.uuid.substr(0, 8);
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

            if (!SensorConfig::isVision(config.type)) {
                buildBodySensors(node, config);
                return;
            }

            // A pinhole hosted on a camera reads its frustum FROM the camera,
            // whatever the flat string says: the Camera section of the
            // inspector is where those numbers are edited, and a stale
            // userData copy must not out-vote it. Everywhere else the config
            // is the only frustum there is.
            SensorConfig effective = config;
            if (SensorConfig::isPinhole(config.type)) {
                if (const auto* camera = node.as<PerspectiveCamera>()) {
                    effective.fovY = camera->fov;
                    effective.nearPlane = camera->nearPlane;
                    effective.farPlane = camera->farPlane;
                }
            }

            auto entry = makeEntry(node, effective);
            if (config.type == SensorConfig::Type::Depth) {
                buildDepth(*entry, effective);
            } else if (config.type == SensorConfig::Type::Camera) {
                buildCamera(*entry, effective);
            } else {
                buildLidar(*entry, effective);
            }
            commit(std::move(entry));
        }

        // The one vision sensor this entry carries, handed to `fn` with its
        // cloud. Depth and lidar share the whole scan protocol (see
        // TracedRasterVisionSensor), so everything that drives them is written
        // once against fn(sensor, cloud) and dispatched here.
        template<class Fn>
        static void withVision(Entry& entry, Fn&& fn) {
            if (entry.depth) {
                fn(*entry.depth, entry.cloud);
            } else if (entry.lidar) {
                fn(*entry.lidar, entry.returns);
            }
        }

        // Visit every point of the entry's newest cloud as (world position,
        // range from the sensor), whichever vision kind this is — the depth
        // cloud carries positions only, so its ranges are measured here.
        template<class Fn>
        static void forEachPoint(const Entry& entry, Fn&& fn) {
            if (entry.depth) {
                Vector3 origin;
                entry.depth->getWorldPosition(origin);
                for (const auto& p : entry.cloud) fn(p, p.distanceTo(origin));
            } else if (entry.lidar) {
                for (const auto& r : entry.returns) fn(r.position, r.distance);
            }
        }

        // The shared tail of a vision build: seeded noise from the authored
        // config, the rate gate, parenting into the rig, and the one trace a
        // vision entry plots. Written once because the sensors share
        // TracedRasterVisionSensor — only the construction above it differs.
        template<class SensorT>
        void adoptVision(Entry& entry, std::unique_ptr<SensorT>& slot,
                         std::unique_ptr<SensorT> sensor,
                         const SensorConfig& config, const char* traceName) {

            sensor->rangeNoise = rangeNoise(config);
            sensor->resetNoise();
            sensor->setRateHz(static_cast<double>(std::max(config.rateHz, 0.f)));
            rig_->addRef(*sensor);
            slot = std::move(sensor);

            entry.traceNames = {traceName, nullptr, nullptr, nullptr, nullptr, nullptr};
            entry.traceCount = 1;
            if (!renderer_) entry.status = "no renderer - built but not scanning";
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
            adoptVision(entry, entry.depth, std::move(depth), config, "points");
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
            adoptVision(entry, entry.lidar, std::move(lidar), config, "returns");
        }

        // The colour camera. Deliberately NOT routed through adoptVision: that
        // helper's whole body is the ranging contract (seeded RangeNoiseModel,
        // resetNoise), and a picture has no range to corrupt. What it does
        // share — the rate gate and the rig parenting — is two lines.
        void buildCamera(Entry& entry, const SensorConfig& config) {

            if (!rig_) {
                entry.status = "no sensor rig - the editor did not wire one up";
                return;
            }

            const auto width = static_cast<unsigned>(
                    std::clamp(config.width, 8, SensorConfig::maxImageSize));
            const auto height = static_cast<unsigned>(
                    std::clamp(config.height, 8, SensorConfig::maxImageSize));
            const float near = std::max(config.nearPlane, 1e-3f);
            const float far = std::max(config.farPlane, near + 1e-3f);

            auto camera = std::make_unique<CameraSensor>(
                    std::clamp(config.fovY, 1.f, 179.f), width, height, near, far);
            camera->name = entry.label + " (camera)";
            camera->setRateHz(static_cast<double>(std::max(config.rateHz, 0.f)));
            rig_->addRef(*camera);
            entry.camera = std::move(camera);

            // Mean luminance, which is the one scalar worth plotting for an
            // image: it says at a glance whether the camera is looking at the
            // scene or at the inside of a link.
            entry.traceNames = {"brightness", nullptr, nullptr, nullptr, nullptr, nullptr};
            entry.traceCount = 1;
            if (!renderer_) entry.status = "no renderer - built but not capturing";
        }

        // The rig node this entry's vision sensor lives on, whichever kind it
        // is. Every vision sensor IS an Object3D; only the ranging pair also
        // carries a cloud, which is why withVision() cannot answer this.
        [[nodiscard]] static Object3D* visionNode(Entry& entry) {
            if (entry.depth) return entry.depth.get();
            if (entry.lidar) return entry.lidar.get();
            if (entry.camera) return entry.camera.get();
            return nullptr;
        }

        static void detachVision(Entry& entry) {
            if (auto* node = visionNode(entry)) node->removeFromParent();
        }

        // Park the sensor node on the authored object's world pose. Scale is
        // dropped deliberately: a scaled sensor would scale its own ray
        // directions, and "the sensor is 2x bigger" is not a thing a sensor is.
        void placeVision(Entry& entry) {

            if (!entry.node) return;

            auto* sensorNode = visionNode(entry);
            if (!sensorNode) return;

            entry.node->updateWorldMatrix(true, false);
            Vector3 position, scale;
            Quaternion rotation;
            entry.node->matrixWorld->decompose(position, rotation, scale);

            sensorNode->position.copy(position);
            sensorNode->quaternion.copy(rotation);
            sensorNode->scale.set(1.f, 1.f, 1.f);
            // The rig is an identity-transformed child of the scene root, so
            // the local pose above IS the world pose. Force it down into the
            // child cameras now — the scan reads their world matrices
            // directly.
            sensorNode->updateWorldMatrix(true, true);
        }

        // A vision scan is FIRED on one frame and DELIVERED on a later one.
        //
        // That is not an optimisation detail, it is the only way a rate-gated
        // sensor can be free of hitches on a ray-traced backend: taking delivery
        // means waiting on a GPU fence, and that fence sits behind every frame
        // already queued — measured at ~28 ms for a 1.2 ms VLP-16 trace on an
        // RTX 4070 with two frames in flight. A 10 Hz LIDAR would deliver that
        // stall ten times a second, which is exactly what the Hover Arena
        // example felt like. So: collect what an earlier frame fired (by then a
        // memcpy), then fire whatever the gate says is due.
        //
        // The cloud a panel or the overlay reads is therefore one frame old, at
        // the pose the beams were fired from. A real sensor has latency too; a
        // frame of it at 60 Hz is 16 ms, and the alternative is a 28 ms stall.
        // On a raster backend the fire IS the scan (six framebuffer reads that
        // block anyway), so the delivery lands on the same frame and nothing
        // about GL behaviour changes.
        void scanAll(float dt) {

            if (!renderer_ || !scene_) return;

            // --- deliveries owed from an earlier frame ------------------------
            // scanReady is a fence poll, never a wait: a scan that has not
            // landed yet simply stays owed for another frame, and the entry
            // keeps the cloud it already had rather than blinking empty.
            for (const auto& entry : entries_) {
                withVision(*entry, [&](auto& sensor, auto& cloud) {
                    if (sensor.scanReady(*renderer_)) deliver(*entry, sensor, cloud);
                });
            }

            // Arm the gate and latch the clock on the pulled sensors. They are
            // deliberately NOT registered with the world (that would tick them
            // twice); tick() is the same entry point the world would use.
            // A sensor with a scan still owed does not fire again: the second
            // fire would throw the first away, and an UNGATED sensor (rateHz 0
            // = every frame) would then spend its whole life firing scans
            // nobody ever sees. The gate stays armed, so it fires on the frame
            // after delivery instead.
            bool any = false;
            for (const auto& entry : entries_) {
                withVision(*entry, [&](auto& sensor, auto&) {
                    sensor.tick(static_cast<double>(dt), simTime_);
                    any = any || (sensor.scanDue() && !sensor.scanPending());
                });
            }
            if (!any) return;

            // A depth camera aimed at the viewport grid measures the grid. Hide
            // the editor's furniture for exactly as long as the scans take.
            const bool wasVisible = hidden_ ? hidden_->visible : true;
            if (hidden_) hidden_->visible = false;

            // scanBegin answers whether the cloud is ALREADY in hand (raster) or
            // owed by a later frame (Vulkan). Deliberately not a fence poll: a
            // poll that happened to succeed would land the same scan on frame N
            // in one run and N+1 in the next, and "which frame did this sensor
            // report on" would become a property of GPU scheduling.
            for (const auto& entry : entries_) {
                withVision(*entry, [&](auto& sensor, auto& cloud) {
                    if (!sensor.scanDue() || sensor.scanPending()) return;
                    placeVision(*entry);
                    if (sensor.scanBegin(*renderer_, *scene_, cloud)) {
                        deliver(*entry, sensor, cloud);
                    }
                });
            }

            if (hidden_) hidden_->visible = wasVisible;
        }

        // One delivered scan: stamp, count, plot, record. Shared by the two
        // places delivery can happen (immediately on a raster backend, a frame
        // later on Vulkan) so a scan is booked identically either way.
        template<class SensorT, class CloudT>
        void deliver(Entry& entry, SensorT& sensor, CloudT& cloud) {
            if (!sensor.scanCollect(*renderer_, cloud)) return;
            entry.lastTime = sensor.lastScanTime();
            ++entry.scans;
            entry.traces[0].push(static_cast<float>(cloud.size()));
            recordVision(entry);
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
            forEachPoint(entry, [&](const Vector3& p, float r) {
                sum.add(p);
                if (count == 0 || r < rmin) rmin = r;
                if (count == 0 || r > rmax) rmax = r;
                ++count;
            });
            if (count > 0) sum.multiplyScalar(1.f / static_cast<float>(count));

            entry.csv << entry.lastTime << ',' << count << ',' << sum.x << ',' << sum.y << ','
                      << sum.z << ',' << rmin << ',' << rmax << '\n';
            ++entry.rows;
        }

        // The colour cameras' half of the frame loop, and a much shorter story
        // than scanAll's: no fence to poll and no fire/deliver split, because
        // the sensor hides the split inside capture(). On a raster backend the
        // capture is a render plus a readback, both synchronous; on Vulkan it
        // collects the frame the last render() drew from the sensor's own
        // secondary view, so the picture is one frame old and the first
        // capture or two of a Play return false while the view warms up.
        // The furniture-hiding window matters on the raster path — a wrist
        // camera pointed at the floor would otherwise photograph the viewport
        // grid. On Vulkan the pixels were drawn before this call, but the
        // furniture is overlay-pass work, which secondary views never draw.
        //
        // Empty-handed captures a camera gets before its status calls that a
        // failure. Vulkan warm-up takes two or three; ten is comfortably past
        // any of them while still inside a Play's first second.
        static constexpr int kCaptureGraceFrames = 10;

        void captureCameras(float dt) {

            if (!renderer_ || !scene_) return;

            bool any = false;
            for (const auto& entry : entries_) {
                if (!entry->camera) continue;
                entry->camera->tick(static_cast<double>(dt), simTime_);
                any = any || entry->camera->captureDue();
            }
            if (!any) return;

            const bool wasVisible = hidden_ ? hidden_->visible : true;
            if (hidden_) hidden_->visible = false;

            for (const auto& entry : entries_) {
                if (!entry->camera || !entry->camera->captureDue()) continue;
                placeVision(*entry);
                if (entry->camera->capture(*renderer_, *scene_)) {
                    entry->lastTime = entry->camera->lastCaptureTime();
                    ++entry->scans;
                    entry->traces[0].push(meanLuminance(entry->camera->image()));
                    // A warm-up note must not outlive the warm-up.
                    entry->status.clear();
                    entry->captureMisses = 0;
                    recordCamera(*entry);
                } else if (entry->camera->frames() == 0 &&
                           ++entry->captureMisses > kCaptureGraceFrames &&
                           entry->status.empty()) {
                    // Said once, at the moment the user is looking, and then
                    // left standing in the panel: a camera that never produces
                    // a frame is a fact about the run, not a per-frame event
                    // to spam the console with. The grace window is what keeps
                    // a Vulkan view's first few warm-up misses out of it.
                    entry->status = "no frame captured - the renderer is not "
                                    "producing images for this camera";
                    log("sensor: \"" + entry->label + "\" - " + entry->status);
                }
            }

            if (hidden_) hidden_->visible = wasVisible;
        }

        // Rec. 601 luma over the whole frame, 0-255. Cheap enough at every
        // authored resolution to not be worth sub-sampling.
        static float meanLuminance(const std::vector<unsigned char>& rgb) {

            if (rgb.size() < 3) return 0.f;
            double sum = 0.0;
            const std::size_t pixels = rgb.size() / 3;
            for (std::size_t i = 0; i < pixels; ++i) {
                sum += 0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2];
            }
            return static_cast<float>(sum / static_cast<double>(pixels));
        }

        // A camera's recording is the FRAMES: one PNG per capture beside a CSV
        // indexing them. That is what a perception dataset is, and it is the
        // reason to prefer this over the ranging sensors' one-row-per-scan
        // summary — a row saying "the picture was this bright" trains nothing.
        void recordCamera(Entry& entry) {

            if (!recording_ || !entry.camera) return;
            if (!openFile(entry, "t,frame,file,mean_luma")) return;

            const auto stem = sanitize(entry.label) + "_" + entry.uuid.substr(0, 8) + "_" +
                              frameNumber(entry.camera->frames());
            const auto file = recordDir_ / (stem + ".png");
            if (!entry.camera->writeImage(file)) {
                log("sensor: cannot write " + file.string());
                recording_ = false;
                return;
            }

            entry.csv << entry.lastTime << ',' << entry.camera->frames() << ','
                      << stem << ".png," << meanLuminance(entry.camera->image()) << '\n';
            ++entry.rows;
        }

        // Zero-padded so the frames sort in capture order in a file browser and
        // in a glob, which is how every tool that eats an image sequence finds
        // them.
        static std::string frameNumber(std::size_t frame) {

            auto text = std::to_string(frame);
            if (text.size() < 6) text.insert(0, 6 - text.size(), '0');
            return text;
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
            forEachPoint(entry, [&](const Vector3& p, float r) {
                out << entry.lastTime << ',' << p.x << ',' << p.y << ',' << p.z << ','
                    << r << '\n';
            });
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

        Renderer* renderer_ = nullptr;
        Object3D* rig_ = nullptr;
        Object3D* hidden_ = nullptr;
        Scene* scene_ = nullptr;
        std::function<void(const std::string&)> logger_;

        double simTime_ = 0.0;

        bool recording_ = false;
        std::filesystem::path recordDir_ = std::filesystem::temp_directory_path() / "threepp-sensors";

        std::shared_ptr<const void> lifetime_;
        inline static SensorPlaySession* active_ = nullptr;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SENSORPLAYSESSION_HPP
