// The PhysX half of the sensor session: body and joint sensors, live.
//
// SensorPlaySession (the base, PhysX-free) owns the entry list, the vision
// sensors, the clock and the recording; this class fills in the three seams the
// base leaves open — building the IMU/contact/encoder/force-torque sensors
// against the world PhysicsPlaySession built, draining them into the traces,
// and reading the world's substep clock so every measurement shares one time
// base. Header-only and PhysX-dependent, exactly like PhysicsPlaySession — the
// threepp library proper never links PhysX, so this file is included only by
// builds that found the SDK (the editor sets THREEPP_EDITOR_WITH_PHYSX).
//
// The world is borrowed, never owned: PlayController stops sessions in
// registration order, and physics is registered FIRST, so by the time stop()
// runs here the PhysxWorld may already be gone. Hence the lifetime token — the
// world pointer is only ever dereferenced while PhysicsPlaySession's token is
// still alive.

#ifndef THREEPP_EDITOR_PHYSXSENSORPLAYSESSION_HPP
#define THREEPP_EDITOR_PHYSXSENSORPLAYSESSION_HPP

#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/SensorPlaySession.hpp"

#include "threepp/extras/sensors/ContactSensor.hpp"
#include "threepp/extras/sensors/ForceTorqueSensor.hpp"
#include "threepp/extras/sensors/Imu.hpp"
#include "threepp/extras/sensors/JointEncoder.hpp"

#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace threepp::editor {

    class PhysxSensorPlaySession: public SensorPlaySession {

    public:
        // The editor being torn down mid-Play never gets a stop(). This runs
        // before the base destructor detaches the vision nodes, so the entries
        // (and their sensors) are still alive here.
        ~PhysxSensorPlaySession() override {

            releaseBodySensors();
        }

        // Where the pushed sensors register. Borrowed; the token this session
        // reads from it is what keeps a stopped world from being dereferenced.
        void setPhysics(PhysicsPlaySession* physics) { physics_ = physics; }

        // --- PlaySession -----------------------------------------------------

        void start(Scene& scene) override {

            // Latch the world before the base runs the builds — buildBodySensors
            // registers against it.
            world_ = nullptr;
            worldLife_.reset();
            if (physics_) {
                world_ = physics_->world();
                worldLife_ = physics_->lifetime();
            }

            SensorPlaySession::start(scene);
        }

        void stop() override {

            // Sessions stop in registration order, physics FIRST — so by the time
            // this runs the PhysxWorld (and the PhysX SDK behind it) is usually
            // already gone. Release the body sensors accordingly, then let the
            // base do the renderer half and drop the entries.
            releaseBodySensors();

            SensorPlaySession::stop();

            world_ = nullptr;
            worldLife_.reset();
        }

    protected:
        // --- the PhysX seam, filled in ---------------------------------------

        void buildBodySensors(Object3D& node, const SensorConfig& config) override {

            // One authored entry normally becomes one sensor. The whole-robot
            // encoder (joint == allJoints) is the exception: an object carries
            // ONE sensor entry, so covering every DOF from the robot's root has
            // to fan out — one live encoder per joint, each its own Entry so the
            // traces, the counters and the CSV files stay per-joint.
            if (config.type == SensorConfig::Type::Encoder &&
                config.joint == SensorConfig::allJoints) {
                buildEncoderFanout(node, config);
                return;
            }

            auto entry = makeEntry(node, config);

            switch (config.type) {
                case SensorConfig::Type::Imu: buildImu(*entry, node, config); break;
                case SensorConfig::Type::Contact: buildContact(*entry, node, config); break;
                case SensorConfig::Type::Encoder: buildEncoder(*entry, node, config); break;
                case SensorConfig::Type::ForceTorque: buildForceTorque(*entry, node, config); break;
                default: break;// vision never routes here
            }

            commit(std::move(entry));
        }

        void drainBodies() override {

            for (const auto& entry : entries_) {
                if (entry->imu) drainImu(*entry);
                if (entry->contact) drainContact(*entry);
                if (entry->encoder) drainEncoder(*entry);
                if (entry->forceTorque) drainForceTorque(*entry);
            }
        }

        bool worldClock(double& time) override {

            if (auto* world = liveWorld()) {
                time = world->simTime();
                return true;
            }
            return false;
        }

    private:
        // The borrowed world, or nullptr once PhysicsPlaySession has stopped.
        [[nodiscard]] PhysxWorld* liveWorld() const {

            if (!world_) return nullptr;
            return worldLife_.expired() ? nullptr : world_;
        }

        // Two paths, and the difference is a class of bug this codebase cares
        // about:
        //
        //  world alive  — a mid-play teardown that reached us before physics.
        //                 Unregister cleanly; the Force/Torque sensor releases
        //                 its PxArticulationCache through onUnregister while the
        //                 SDK still exists.
        //  world gone   — the normal Stop. The cache's memory went with the SDK,
        //                 so calling release() on it would touch a freed
        //                 allocator. Abandon the pointer instead: the SDK teardown
        //                 already reclaimed the buffer, so nothing leaks that the
        //                 process does not, and the destructor stays a no-op.
        void releaseBodySensors() {

            if (auto* world = liveWorld()) {
                for (const auto& entry : entries_) {
                    if (entry->imu) world->unregisterSensor(entry->imu.get());
                    if (entry->contact) world->unregisterSensor(entry->contact.get());
                    if (entry->encoder) world->unregisterSensor(entry->encoder.get());
                    if (entry->forceTorque) world->unregisterSensor(entry->forceTorque.get());
                }
            } else {
                for (const auto& entry : entries_) {
                    if (entry->forceTorque) entry->forceTorque->abandonCache();
                }
            }
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

        // Resolve the played articulation a joint sensor rides, or set a status
        // that names exactly what is missing. Returns nullptr on any failure.
        const PhysicsPlaySession::PlayedArticulation* resolveArticulation(Entry& entry, Object3D& node) {

            if (!liveWorld()) {
                entry.status = "no physics world - a joint sensor rides an articulated robot";
                return nullptr;
            }
            if (!physics_) {
                entry.status = "no physics session - a joint sensor rides an articulated robot";
                return nullptr;
            }
            const auto* played = physics_->findArticulation(&node);
            if (!played) {
                entry.status = "no articulated robot - enable Simulate in the Robot section";
            }
            return played;
        }

        // Resolve the articulation link ONE joint sensor measures, or set a
        // status that names exactly what is missing. Returns nullptr on any
        // failure.
        const ArticulationLink* resolveJoint(Entry& entry, Object3D& node, const SensorConfig& config) {

            const auto* played = resolveArticulation(entry, node);
            if (!played) return nullptr;
            if (config.joint.empty()) {
                entry.status = "no joint chosen - pick one in the Sensor section";
                return nullptr;
            }
            if (config.joint == SensorConfig::allJoints) {
                // Only the Force/Torque sensor gets here - the encoder's
                // all-joints fan-out is intercepted in buildBodySensors() before
                // any entry reaches this resolver.
                entry.status = "All joints is an encoder thing - a load cell sits in one joint";
                return nullptr;
            }
            const auto* link = played->linkFor(config.joint);
            if (!link) {
                entry.status = "joint \"" + config.joint + "\" is not a simulated DOF of this robot";
                return nullptr;
            }
            return link;
        }

        NoiseModel encoderPositionNoise(const SensorConfig& config, int streamIndex) const {

            // A joint encoder's error is dominated by quantization (encoderResolution
            // below), so the position noise is left at the clean default here — no
            // authored density/walk fields exist for it in this pass. Still seed it
            // from a free stream index so a future field is reproducible and does not
            // collide with the IMU's 0/1 or a range channel's 2. `streamIndex` is the
            // DOF index under the all-joints fan-out (0 for a single encoder, so the
            // authored seed keeps its meaning), so seven encoders from one authored
            // seed do not draw correlated noise.
            NoiseModel model;
            model.seed = config.streamSeed(3 + streamIndex);
            return model;
        }

        void buildEncoder(Entry& entry, Object3D& node, const SensorConfig& config) {

            const auto* link = resolveJoint(entry, node, config);
            if (!link) return;
            buildEncoderFor(entry, node, config, *link, 0);
        }

        // The construction shared by the single named encoder and the all-joints
        // fan-out.
        void buildEncoderFor(Entry& entry, Object3D& node, const SensorConfig& config,
                             const ArticulationLink& link, int streamIndex) {

            auto encoder = std::make_unique<JointEncoder>(
                    node, link, static_cast<double>(std::max(config.rateHz, 0.f)));
            encoder->resolution = std::max(config.encoderResolution, 0.f);
            encoder->positionNoise = encoderPositionNoise(config, streamIndex);
            try {
                liveWorld()->registerSensor(encoder.get());
            } catch (const std::exception& e) {
                entry.status = e.what();
                return;
            }
            entry.encoder = std::move(encoder);

            entry.traceNames = {"position", "velocity", nullptr, nullptr, nullptr, nullptr};
            entry.traceCount = 2;
        }

        // One authored entry, one live encoder per articulated DOF. Each joint
        // gets its own Entry: its own traces, its own sample counter, and its
        // own CSV — the label carries the joint name, which is what keeps the
        // readout rows and the recorded files apart when they all share a node.
        void buildEncoderFanout(Object3D& node, const SensorConfig& config) {

            // Resolve the robot once: a failure is ONE entry carrying the
            // reason, not a copy of it for every DOF that does not exist.
            auto probe = makeEntry(node, config);
            const auto* played = resolveArticulation(*probe, node);
            if (played && played->jointNames.empty()) {
                probe->status = "the robot has no simulated DOFs";
            }
            if (!played || played->jointNames.empty()) {
                commit(std::move(probe));
                return;
            }

            for (std::size_t i = 0; i < played->jointNames.size() && i < played->links.size(); ++i) {
                auto entry = makeEntry(node, config);
                entry->label += " " + played->jointNames[i];
                entry->config.joint = played->jointNames[i];// which joint this one became
                buildEncoderFor(*entry, node, config, played->links[i], static_cast<int>(i));
                commit(std::move(entry));
            }
        }

        void buildForceTorque(Entry& entry, Object3D& node, const SensorConfig& config) {

            const auto* link = resolveJoint(entry, node, config);
            if (!link) return;

            // resolveJoint proved the articulation exists; fetch it for the sensor,
            // which needs the Articulation to build its state cache.
            const auto* played = physics_->findArticulation(&node);
            if (!played || !played->articulation) {
                entry.status = "no articulation to read a wrench from";
                return;
            }

            auto ft = std::make_unique<ForceTorqueSensor>(
                    node, *played->articulation, *link,
                    static_cast<double>(std::max(config.rateHz, 0.f)));
            try {
                // onRegister creates the PxArticulationCache; unregisterSensor (in
                // stop(), while the world is alive) releases it before the
                // articulation is destroyed, so there is no use-after-free on Stop.
                liveWorld()->registerSensor(ft.get());
            } catch (const std::exception& e) {
                entry.status = e.what();
                return;
            }
            entry.forceTorque = std::move(ft);

            entry.traceNames = {"force x", "force y", "force z", "torque x", "torque y", "torque z"};
            entry.traceCount = 6;
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

        void drainEncoder(Entry& entry) {

            entry.encoder->drain(encoderScratch_);
            if (encoderScratch_.empty()) return;

            for (const auto& sample : encoderScratch_) {
                entry.traces[0].push(sample.position);
                entry.traces[1].push(sample.velocity);
                if (openFile(entry, "t,position,velocity")) {
                    entry.csv << sample.t << ',' << sample.position << ',' << sample.velocity << '\n';
                    ++entry.rows;
                }
            }
            entry.samples += encoderScratch_.size();
            entry.lastTime = encoderScratch_.back().t;
        }

        void drainForceTorque(Entry& entry) {

            entry.forceTorque->drain(wrenchScratch_);
            if (wrenchScratch_.empty()) return;

            for (const auto& sample : wrenchScratch_) {
                entry.traces[0].push(sample.force.x);
                entry.traces[1].push(sample.force.y);
                entry.traces[2].push(sample.force.z);
                entry.traces[3].push(sample.torque.x);
                entry.traces[4].push(sample.torque.y);
                entry.traces[5].push(sample.torque.z);
                if (openFile(entry, "t,fx,fy,fz,tx,ty,tz")) {
                    entry.csv << sample.t << ','
                              << sample.force.x << ',' << sample.force.y << ',' << sample.force.z << ','
                              << sample.torque.x << ',' << sample.torque.y << ',' << sample.torque.z << '\n';
                    ++entry.rows;
                }
            }
            entry.samples += wrenchScratch_.size();
            entry.lastTime = wrenchScratch_.back().t;
        }

        PhysicsPlaySession* physics_ = nullptr;

        PhysxWorld* world_ = nullptr;
        std::weak_ptr<const void> worldLife_;

        // Drain targets, reused so a 1 kHz sensor does not allocate per frame.
        std::vector<ImuSample> imuScratch_;
        std::vector<ContactSample> contactScratch_;
        std::vector<JointSample> encoderScratch_;
        std::vector<WrenchSample> wrenchScratch_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_PHYSXSENSORPLAYSESSION_HPP
