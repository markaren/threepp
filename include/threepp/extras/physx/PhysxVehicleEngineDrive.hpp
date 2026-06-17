
#ifndef THREEPP_PHYSX_VEHICLE_ENGINE_DRIVE_HPP
#define THREEPP_PHYSX_VEHICLE_ENGINE_DRIVE_HPP

#include "threepp/extras/physx/PhysxWorld.hpp"

#include <PxPhysicsAPI.h>
#include <vehicle2/PxVehicleAPI.h>

#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

namespace threepp {

    // Engine-driven 4-wheel vehicle built on the full PxVehicle2 component stack:
    // engine (torque curve + revs) → clutch → multi-ratio gearbox (+ autobox) →
    // differential → wheels. This is the "fully featured" sibling of PhysxVehicle
    // (which is direct-drive, no engine). It produces real engine RPM, automatic or
    // manual gear shifts, clutch slip, and engine braking — telemetry the HUD and
    // procedural audio can read directly instead of synthesising.
    //
    // Structurally it mirrors PhysxVehicle: it creates its own PxRigidDynamic chassis
    // + 4 wheel shapes, wires the standard component sequence, and steps via
    // PhysxWorld::onPreSubstep so the substep cadence stays in sync with PhysX. Only
    // the drive components differ — DirectDrive{CommandResponse,ActuationState,
    // Drivetrain} are replaced by EngineDrive{CommandResponse,ActuationState,
    // Drivetrain} and a MultiWheelDriveDifferential is inserted between command
    // response and actuation (both downstream components consume the differential
    // torque split).
    //
    // Frame convention is threepp-native: lng=+Z (forward), lat=+X (right), vrt=+Y (up).
    // SI units: meters, kg, seconds. Engine speeds are radians/second internally;
    // engineRPM() converts for display.
    class PhysxVehicleEngineDrive final
        : public ::physx::vehicle2::PxVehiclePhysXActorBeginComponent,
          public ::physx::vehicle2::PxVehiclePhysXRoadGeometrySceneQueryComponent,
          public ::physx::vehicle2::PxVehicleSuspensionComponent,
          public ::physx::vehicle2::PxVehicleTireComponent,
          public ::physx::vehicle2::PxVehicleEngineDriveCommandResponseComponent,
          public ::physx::vehicle2::PxVehicleMultiWheelDriveDifferentialStateComponent,
          public ::physx::vehicle2::PxVehicleEngineDriveActuationStateComponent,
          public ::physx::vehicle2::PxVehicleEngineDrivetrainComponent,
          public ::physx::vehicle2::PxVehicleWheelComponent,
          public ::physx::vehicle2::PxVehicleRigidBodyComponent,
          public ::physx::vehicle2::PxVehiclePhysXConstraintComponent,
          public ::physx::vehicle2::PxVehiclePhysXActorEndComponent {

    public:
        struct Settings {
            // ── Chassis ──────────────────────────────────────────────────────
            float chassisWidth = 1.9f;
            float chassisHeight = 1.3f;
            float chassisLength = 4.4f;
            float chassisMass = 1600.f;

            // ── Wheels ───────────────────────────────────────────────────────
            float wheelRadius = 0.36f;
            float wheelHalfWidth = 0.16f;
            float wheelMass = 25.f;

            float trackWidth = 1.65f;
            float wheelbase = 2.7f;

            // ── Suspension (the shock absorbers) ─────────────────────────────
            float suspensionTravelDist = 0.32f;
            float suspensionStiffness = 38'000.f;
            float suspensionDamping = 5000.f;
            // Wheel pose at maximum compression, chassis-frame Y. Negative => below center.
            float suspensionAttachmentY = -0.35f;

            // ── Brakes / steering ────────────────────────────────────────────
            float maxBrakeTorque = 6000.f; // N*m at full brake, all wheels
            float maxHandbrakeTorque = 9000.f;// N*m, rear wheels only
            float maxSteerAngleRad = 0.6f; // ~34 deg, front wheels (must be <= PI)

            // ── Engine ───────────────────────────────────────────────────────
            float enginePeakTorque = 520.f;  // N*m at the torque-curve peak
            float engineMoi = 1.0f;          // kg*m^2 — engine rotational inertia
            float engineIdleRpm = 900.f;     // rpm
            float engineMaxRpm = 6200.f;     // rpm (redline)
            float engineDampingFullThrottle = 0.15f;
            float engineDampingZeroThrottleClutchEngaged = 2.0f;
            float engineDampingZeroThrottleClutchDisengaged = 0.35f;

            // ── Clutch ───────────────────────────────────────────────────────
            float clutchStrength = 10.f;     // max clutch coupling torque*time
            unsigned clutchEstimateIterations = 5;

            // ── Gearbox ──────────────────────────────────────────────────────
            // ratios[0] reverse (negative), [1] neutral (0), [2..] forward (descending).
            // The applied ratio is ratios[gear] * finalRatio.
            std::vector<float> gearRatios = {-4.0f, 0.0f, 4.2f, 2.6f, 1.7f, 1.25f, 0.95f, 0.78f};
            float finalDriveRatio = 4.0f;
            float gearSwitchTime = 0.5f;     // seconds to change gear

            // ── Autobox (automatic transmission) ─────────────────────────────
            // Normalised engine-rev thresholds (engineRev/maxRev) to shift up / down.
            float autoboxUpRatio = 0.65f;
            float autoboxDownRatio = 0.25f;
            float autoboxLatency = 0.4f;     // min seconds between automatic shifts

            // ── Differential ─────────────────────────────────────────────────
            // Which wheels receive engine torque. Default = all-wheel drive.
            // Indices: 0 = front-right, 1 = front-left, 2 = rear-right, 3 = rear-left.
            std::array<bool, 4> drivenWheels{true, true, true, true};

            // ── Tire ─────────────────────────────────────────────────────────
            float tireFriction = 2.0f;
            float lateralStiffness = 90'000.f;
            float longitudinalStiffness = 100'000.f;
            float wheelDampingRate = 0.25f;  // lower than direct-drive: the engine/
                                             // gearbox now provide the coasting drag

            float chassisLinearDamping = 0.1f;
            float chassisAngularDamping = 0.5f;

            // ── Spawn ────────────────────────────────────────────────────────
            Vector3 spawnPosition{0, 1.0f, 0};
            Quaternion spawnRotation;

            // ── Sticky tire (decisive low-speed stops) — see PhysxVehicle docs ─
            float stickySpeedThreshold = 0.2f;
            float stickyTimeThreshold = 0.1f;
            float stickyDampingLongitudinal = 1.0f;
            float stickyDampingLateral = 1.0f;

            // ── Vehicle sub-stepping (stiff wheel-spin ODE) — see PhysxVehicle docs ─
            unsigned subStepCountLowSpeed = 8;
            unsigned subStepCountHighSpeed = 2;
            float subStepThresholdSpeed = 6.f;// m/s

            // ── Tire-slip denominator floors — see PhysxVehicle docs ─────────
            float minLongSlipDenominatorActive = 2.0f;
            float minLongSlipDenominatorPassive = 4.0f;
            float minLatSlipDenominator = 1.0f;
        };

        enum class TransmissionMode { Automatic,
                                      Manual };

        // Driver-facing gear selector (like P/R/N/D). In Drive with Automatic mode
        // the autobox picks the forward gear; in Manual it uses shiftUp/shiftDown.
        enum class Direction { Reverse,
                               Neutral,
                               Drive };

        explicit PhysxVehicleEngineDrive(PhysxWorld& world, const Settings& s = Settings())
            : world_(&world), settings_(s) {

            using namespace ::physx;
            using namespace ::physx::vehicle2;

            if (settings_.gearRatios.size() < 3) {
                throw std::runtime_error("PhysxVehicleEngineDrive: need at least reverse+neutral+1 forward gear");
            }

            if (!PxInitVehicleExtension(world.foundation())) {
                throw std::runtime_error("PxInitVehicleExtension failed");
            }
            initialised_ = true;

            buildAxleDescription();
            buildRigidBodyParams();
            buildWheelParams();
            buildSuspensionParams();
            buildTireParams();
            buildEngineDriveParams();
            buildCommandResponseParams();
            buildSimContext();

            createPhysxActor();
            createPhysxConstraints();
            buildComponentSequence();

            stepCallback_ = [this](float dt) { stepVehicle(dt); };
            world_->onPreSubstep(stepCallback_);
        }

        ~PhysxVehicleEngineDrive() {
            using namespace ::physx;

            if (constraintsCreated_) {
                ::physx::vehicle2::PxVehicleConstraintsDestroy(physxConstraints_);
                constraintsCreated_ = false;
            }
            if (chassisActor_) {
                chassisActor_->getScene()->removeActor(*chassisActor_);
                chassisActor_->release();
                chassisActor_ = nullptr;
            }
            if (wheelShapeMaterial_) {
                wheelShapeMaterial_->release();
                wheelShapeMaterial_ = nullptr;
            }
            if (initialised_) {
                ::physx::vehicle2::PxCloseVehicleExtension();
                initialised_ = false;
            }
        }

        PhysxVehicleEngineDrive(const PhysxVehicleEngineDrive&) = delete;
        PhysxVehicleEngineDrive& operator=(const PhysxVehicleEngineDrive&) = delete;

        // ── Inputs ───────────────────────────────────────────────────────────

        void setThrottle(float v) { commands_.throttle = std::clamp(v, 0.f, 1.f); }
        void setBrake(float v) { commands_.brakes[0] = std::clamp(v, 0.f, 1.f); }
        void setHandbrake(float v) { commands_.brakes[1] = std::clamp(v, 0.f, 1.f); }
        void setSteer(float v) { commands_.steer = std::clamp(v, -1.f, 1.f); }
        // Clutch pedal: 0 = engaged (driving), 1 = disengaged (decoupled). The
        // automatic mode keeps this at 0 and lets the gearbox switch-time model shifts.
        void setClutch(float v) { transmissionCommands_.clutch = std::clamp(v, 0.f, 1.f); }

        void setTransmissionMode(TransmissionMode m) {
            mode_ = m;
            applyTransmissionCommand();
        }
        TransmissionMode transmissionMode() const { return mode_; }

        void setDirection(Direction d) {
            direction_ = d;
            applyTransmissionCommand();
        }
        Direction direction() const { return direction_; }

        // Manual shifting (only takes effect in Manual mode + Drive). Clamps within
        // the forward-gear range.
        void shiftUp() {
            manualForwardGear_ = std::min<::physx::PxU32>(manualForwardGear_ + 1, lastForwardGearIndex());
            applyTransmissionCommand();
        }
        void shiftDown() {
            manualForwardGear_ = std::max<::physx::PxU32>(manualForwardGear_ - 1, firstForwardGearIndex());
            applyTransmissionCommand();
        }

        // ── Readouts ───────────────────────────────────────────────────────────

        ::physx::PxRigidDynamic* chassisActor() const { return chassisActor_; }
        ::physx::PxTransform chassisPose() const { return rigidBodyState_.pose; }

        ::physx::PxTransform wheelLocalPose(int i) const { return wheelLocalPoses_[i].localPose; }

        // Forward speed in chassis longitudinal direction (m/s).
        float forwardSpeed() const { return rigidBodyState_.getLongitudinalSpeed(simContext_.frame); }

        float wheelRotationAngle(int i) const { return wheelRigidBody1dStates_[i].rotationAngle; }
        float wheelAngularSpeed(int i) const { return wheelRigidBody1dStates_[i].rotationSpeed; }

        float tireLongitudinalSlip(int i) const {
            return tireSlipStates_[i].slips[::physx::vehicle2::PxVehicleTireDirectionModes::eLONGITUDINAL];
        }
        float tireLateralSlip(int i) const {
            return tireSlipStates_[i].slips[::physx::vehicle2::PxVehicleTireDirectionModes::eLATERAL];
        }

        // Suspension compression (m): 0 = full droop … suspensionTravelDist = bottomed.
        float suspensionJounce(int i) const { return suspensionStates_[i].jounce; }
        float suspensionJounceSpeed(int i) const { return suspensionStates_[i].jounceSpeed; }
        bool wheelGrounded(int i) const { return roadGeometryStates_[i].hitState; }

        // ── Engine / transmission telemetry ──────────────────────────────────

        // Engine rotation speed (rad/s) and its rpm equivalent.
        float engineRotationSpeed() const { return engineState_.rotationSpeed; }
        float engineRpm() const { return engineState_.rotationSpeed * (60.f / kTwoPi); }
        float engineIdleRpm() const { return settings_.engineIdleRpm; }
        float engineMaxRpm() const { return settings_.engineMaxRpm; }
        // Normalised tacho needle in [0,1].
        float engineRpmFraction() const {
            return std::clamp(engineRpm() / std::max(settings_.engineMaxRpm, 1.f), 0.f, 1.f);
        }

        // Raw gearbox index (0 = reverse, neutralGear = neutral, higher = forward).
        int currentGearIndex() const { return static_cast<int>(gearboxState_.currentGear); }
        int targetGearIndex() const { return static_cast<int>(gearboxState_.targetGear); }
        bool gearShiftInProgress() const { return gearboxState_.currentGear != gearboxState_.targetGear; }
        int neutralGearIndex() const { return static_cast<int>(gearboxParams_.neutralGear); }

        // Human-facing label: "R", "N", "1".."N".
        std::string gearLabel() const {
            const ::physx::PxU32 g = gearboxState_.currentGear;
            if (g < gearboxParams_.neutralGear) return "R";
            if (g == gearboxParams_.neutralGear) return "N";
            return std::to_string(g - gearboxParams_.neutralGear);
        }

        // Clutch slip (rad/s) — engine/wheel speed mismatch; spikes on shifts.
        float clutchSlip() const { return clutchSlipState_.clutchSlip; }

        const Settings& settings() const { return settings_; }

    private:
        static constexpr float kTwoPi = 6.28318530717958647692f;
        static constexpr float kRpmToRadPerSec = kTwoPi / 60.f;

        ::physx::PxU32 firstForwardGearIndex() const { return gearboxParams_.neutralGear + 1; }
        ::physx::PxU32 lastForwardGearIndex() const { return gearboxParams_.nbRatios - 1; }

        // Translate the (mode, direction, manual gear) selector into the PhysX
        // transmission command's target gear. Drive+Automatic uses the special
        // eAUTOMATIC_GEAR sentinel so the autobox drives the shifts.
        void applyTransmissionCommand() {
            using TCS = ::physx::vehicle2::PxVehicleEngineDriveTransmissionCommandState;
            switch (direction_) {
                case Direction::Reverse:
                    transmissionCommands_.targetGear = 0;// reverse gear index
                    break;
                case Direction::Neutral:
                    transmissionCommands_.targetGear = gearboxParams_.neutralGear;
                    break;
                case Direction::Drive:
                    transmissionCommands_.targetGear =
                            (mode_ == TransmissionMode::Automatic)
                                    ? static_cast<::physx::PxU32>(TCS::eAUTOMATIC_GEAR)
                                    : manualForwardGear_;
                    break;
            }
        }

        // Autobox is only consulted when cruising forward under automatic control.
        bool useAutobox() const {
            return mode_ == TransmissionMode::Automatic && direction_ == Direction::Drive;
        }

        // ── Construction helpers ──────────────────────────────────────────────

        void buildAxleDescription() {
            using namespace ::physx;
            const PxU32 frontIds[2] = {0, 1};
            const PxU32 rearIds[2] = {2, 3};
            axleDesc_.setToDefault();
            axleDesc_.addAxle(2, frontIds);
            axleDesc_.addAxle(2, rearIds);
        }

        void buildRigidBodyParams() {
            using namespace ::physx;
            rigidBodyParams_.mass = settings_.chassisMass;
            const float W = settings_.chassisWidth;
            const float H = settings_.chassisHeight;
            const float L = settings_.chassisLength;
            const float m = settings_.chassisMass;
            rigidBodyParams_.moi = PxVec3(
                    m / 12.f * (H * H + L * L),
                    m / 12.f * (W * W + L * L),
                    m / 12.f * (W * W + H * H));
            rigidBodyState_ = ::physx::vehicle2::PxVehicleRigidBodyState();
            rigidBodyState_.pose = toPxTransform(settings_.spawnPosition, settings_.spawnRotation);
        }

        void buildWheelParams() {
            using namespace ::physx;
            for (PxU32 i = 0; i < 4; ++i) {
                wheelParams_[i].radius = settings_.wheelRadius;
                wheelParams_[i].halfWidth = settings_.wheelHalfWidth;
                wheelParams_[i].mass = settings_.wheelMass;
                wheelParams_[i].moi = 0.5f * settings_.wheelMass * settings_.wheelRadius * settings_.wheelRadius;
                wheelParams_[i].dampingRate = settings_.wheelDampingRate;
            }
        }

        void buildSuspensionParams() {
            using namespace ::physx;
            const float halfTrack = settings_.trackWidth * 0.5f;
            const float halfBase = settings_.wheelbase * 0.5f;
            const float ay = settings_.suspensionAttachmentY;

            const PxVec3 wheelPos[4] = {
                    PxVec3(+halfTrack, ay, +halfBase),// 0 front-right
                    PxVec3(-halfTrack, ay, +halfBase),// 1 front-left
                    PxVec3(+halfTrack, ay, -halfBase),// 2 rear-right
                    PxVec3(-halfTrack, ay, -halfBase),// 3 rear-left
            };

            for (PxU32 i = 0; i < 4; ++i) {
                suspensionParams_[i].suspensionAttachment = PxTransform(wheelPos[i]);
                suspensionParams_[i].suspensionTravelDir = PxVec3(0, -1, 0);
                suspensionParams_[i].suspensionTravelDist = settings_.suspensionTravelDist;
                suspensionParams_[i].wheelAttachment = PxTransform(PxIdentity);

                suspensionForceParams_[i].stiffness = settings_.suspensionStiffness;
                suspensionForceParams_[i].damping = settings_.suspensionDamping;
                suspensionForceParams_[i].sprungMass = settings_.chassisMass * 0.25f;

                suspensionComplianceParams_[i].wheelToeAngle.clear();
                suspensionComplianceParams_[i].wheelCamberAngle.clear();
                suspensionComplianceParams_[i].suspForceAppPoint.clear();
                suspensionComplianceParams_[i].tireForceAppPoint.clear();

                suspensionLimitParams_[i].restitution = 0.f;
                suspensionLimitParams_[i].directionForSuspensionLimitConstraint =
                        ::physx::vehicle2::PxVehiclePhysXSuspensionLimitConstraintParams::eROAD_GEOMETRY_NORMAL;
            }

            suspensionStateCalcParams_.suspensionJounceCalculationType =
                    ::physx::vehicle2::PxVehicleSuspensionJounceCalculationType::eRAYCAST;
            suspensionStateCalcParams_.limitSuspensionExpansionVelocity = false;
        }

        void buildTireParams() {
            using namespace ::physx;
            for (PxU32 i = 0; i < 4; ++i) {
                tireForceParams_[i].latStiffX = 0.f;
                tireForceParams_[i].latStiffY = settings_.lateralStiffness;
                tireForceParams_[i].longStiff = settings_.longitudinalStiffness;
                tireForceParams_[i].camberStiff = 0.f;
                tireForceParams_[i].restLoad = settings_.chassisMass * 0.25f * 9.81f;

                tireForceParams_[i].frictionVsSlip[0][0] = 0.f;
                tireForceParams_[i].frictionVsSlip[0][1] = 1.f;
                tireForceParams_[i].frictionVsSlip[1][0] = 0.1f;
                tireForceParams_[i].frictionVsSlip[1][1] = 1.f;
                tireForceParams_[i].frictionVsSlip[2][0] = 1.f;
                tireForceParams_[i].frictionVsSlip[2][1] = 1.f;

                tireForceParams_[i].loadFilter[0][0] = 0.f;
                tireForceParams_[i].loadFilter[0][1] = 0.23f;
                tireForceParams_[i].loadFilter[1][0] = 3.f;
                tireForceParams_[i].loadFilter[1][1] = 3.f;
            }

            materialFriction_.material = &world_->defaultMaterial();
            materialFriction_.friction = settings_.tireFriction;
            for (PxU32 i = 0; i < 4; ++i) {
                perWheelMaterialFriction_[i].materialFrictions = &materialFriction_;
                perWheelMaterialFriction_[i].nbMaterialFrictions = 1;
                perWheelMaterialFriction_[i].defaultFriction = settings_.tireFriction;
            }
        }

        void buildEngineDriveParams() {
            using namespace ::physx;
            using namespace ::physx::vehicle2;

            // Engine: a simple flat-topped torque curve (normalised torque vs
            // normalised revs). Peak in the mid range, tailing off toward idle and
            // redline so the autobox has a reason to keep revs in the meat of the band.
            engineParams_.torqueCurve.clear();
            engineParams_.torqueCurve.addPair(0.0f, 0.72f);
            engineParams_.torqueCurve.addPair(0.33f, 1.0f);
            engineParams_.torqueCurve.addPair(0.66f, 0.96f);
            engineParams_.torqueCurve.addPair(1.0f, 0.74f);
            engineParams_.moi = settings_.engineMoi;
            engineParams_.peakTorque = settings_.enginePeakTorque;
            engineParams_.idleOmega = settings_.engineIdleRpm * kRpmToRadPerSec;
            engineParams_.maxOmega = settings_.engineMaxRpm * kRpmToRadPerSec;
            engineParams_.dampingRateFullThrottle = settings_.engineDampingFullThrottle;
            engineParams_.dampingRateZeroThrottleClutchEngaged = settings_.engineDampingZeroThrottleClutchEngaged;
            engineParams_.dampingRateZeroThrottleClutchDisengaged = settings_.engineDampingZeroThrottleClutchDisengaged;

            engineState_.setToDefault();
            engineState_.rotationSpeed = engineParams_.idleOmega;

            // Clutch. Fully qualify the accuracy mode — the enum name also exists in
            // the legacy `physx::` vehicle API, so the bare name is ambiguous here.
            clutchParams_.accuracyMode = ::physx::vehicle2::PxVehicleClutchAccuracyMode::eESTIMATE;
            clutchParams_.estimateIterations = std::max(1u, settings_.clutchEstimateIterations);
            clutchResponseParams_.maxResponse = settings_.clutchStrength;
            clutchResponseState_.setToDefault();
            clutchSlipState_.setToDefault();
            throttleResponseState_.setToDefault();

            // Gearbox. neutralGear is index 1 (ratios[0] = reverse, [1] = neutral).
            const PxU32 nb = static_cast<PxU32>(std::min<size_t>(settings_.gearRatios.size(),
                                                                 PxVehicleGearboxParams::eMAX_NB_GEARS));
            gearboxParams_.neutralGear = 1;
            gearboxParams_.nbRatios = nb;
            gearboxParams_.finalRatio = settings_.finalDriveRatio;
            gearboxParams_.switchTime = settings_.gearSwitchTime;
            for (PxU32 i = 0; i < nb; ++i) gearboxParams_.ratios[i] = settings_.gearRatios[i];

            gearboxState_.setToDefault();
            // Start in first forward gear so the car is immediately drivable.
            gearboxState_.currentGear = firstForwardGearIndex();
            gearboxState_.targetGear = firstForwardGearIndex();

            // Autobox: per-gear up/down rev thresholds. Indices match gear indices.
            for (PxU32 i = 0; i < PxVehicleGearboxParams::eMAX_NB_GEARS; ++i) {
                autoboxParams_.upRatios[i] = settings_.autoboxUpRatio;
                autoboxParams_.downRatios[i] = settings_.autoboxDownRatio;
            }
            // Never auto-upshift out of the top gear; never auto-downshift below first.
            autoboxParams_.upRatios[lastForwardGearIndex()] = 1.0f;// unreachable => no upshift
            autoboxParams_.downRatios[firstForwardGearIndex()] = 0.0f;// => no downshift below 1st
            autoboxParams_.latency = settings_.autoboxLatency;
            autoboxState_.setToDefault();

            // Differential: split torque equally across the driven wheels.
            diffParams_.setToDefault();
            int nDriven = 0;
            for (bool d : settings_.drivenWheels) nDriven += d ? 1 : 0;
            if (nDriven == 0) {// guard: default to AWD if mis-configured
                settings_.drivenWheels = {true, true, true, true};
                nDriven = 4;
            }
            const float each = 1.f / static_cast<float>(nDriven);
            for (PxU32 i = 0; i < 4; ++i) {
                const float r = settings_.drivenWheels[i] ? each : 0.f;
                diffParams_.torqueRatios[i] = r;
                diffParams_.aveWheelSpeedRatios[i] = r;
            }
            diffState_.setToDefault();
            constraintGroupState_.setToDefault();
        }

        void buildCommandResponseParams() {
            using namespace ::physx;
            using namespace ::physx::vehicle2;

            // Brake command 0: foot brake (all wheels). Brake command 1: handbrake
            // (rear wheels only). Two separate brake response params, one per command.
            std::memset(&brakeResponseParams_, 0, sizeof(brakeResponseParams_));
            brakeResponseParams_.maxResponse = settings_.maxBrakeTorque;
            for (PxU32 i = 0; i < 4; ++i) brakeResponseParams_.wheelResponseMultipliers[i] = 1.f;

            std::memset(&handbrakeResponseParams_, 0, sizeof(handbrakeResponseParams_));
            handbrakeResponseParams_.maxResponse = settings_.maxHandbrakeTorque;
            handbrakeResponseParams_.wheelResponseMultipliers[0] = 0.f;
            handbrakeResponseParams_.wheelResponseMultipliers[1] = 0.f;
            handbrakeResponseParams_.wheelResponseMultipliers[2] = 1.f;
            handbrakeResponseParams_.wheelResponseMultipliers[3] = 1.f;

            // Pack both into the contiguous array actually handed to the command-
            // response component (index 0 = foot brake, 1 = handbrake). Without this
            // the array stays zero-initialised and NEITHER brake produces torque.
            brakeResponseParamArray_[0] = brakeResponseParams_;
            brakeResponseParamArray_[1] = handbrakeResponseParams_;

            std::memset(&steerResponseParams_, 0, sizeof(steerResponseParams_));
            steerResponseParams_.maxResponse = settings_.maxSteerAngleRad;
            steerResponseParams_.wheelResponseMultipliers[0] = 1.f;
            steerResponseParams_.wheelResponseMultipliers[1] = 1.f;
            steerResponseParams_.wheelResponseMultipliers[2] = 0.f;
            steerResponseParams_.wheelResponseMultipliers[3] = 0.f;

            commands_.brakes[0] = 0.f;
            commands_.brakes[1] = 0.f;
            commands_.nbBrakes = 2;
            commands_.throttle = 0.f;
            commands_.steer = 0.f;

            transmissionCommands_.clutch = 0.f;
            manualForwardGear_ = firstForwardGearIndex();
            applyTransmissionCommand();

            physxSteerState_.setToDefault();
            physxConstraints_.setToDefault();
            physxRoadGeomQueryParams_.defaultFilterData = PxQueryFilterData(
                    PxQueryFlag::eSTATIC | PxQueryFlag::eDYNAMIC | PxQueryFlag::ePREFILTER);
            physxRoadGeomQueryParams_.filterDataEntries = nullptr;
            physxRoadGeomQueryParams_.filterCallback = &queryFilter_;
            physxRoadGeomQueryParams_.roadGeometryQueryType =
                    ::physx::vehicle2::PxVehiclePhysXRoadGeometryQueryType::eRAYCAST;
        }

        void buildSimContext() {
            using namespace ::physx;
            using namespace ::physx::vehicle2;
            simContext_.setToDefault();
            simContext_.frame.lngAxis = PxVehicleAxes::ePosZ;
            simContext_.frame.latAxis = PxVehicleAxes::ePosX;
            simContext_.frame.vrtAxis = PxVehicleAxes::ePosY;
            simContext_.scale.scale = 1.f;
            simContext_.gravity = PxVec3(0, -9.81f, 0);
            simContext_.physxScene = &world_->scene();
            simContext_.physxUnitCylinderSweepMesh = nullptr;
            simContext_.physxActorUpdateMode = ::physx::vehicle2::PxVehiclePhysXActorUpdateMode::eAPPLY_VELOCITY;

            auto& sticky = simContext_.tireStickyParams;
            sticky.stickyParams[PxVehicleTireDirectionModes::eLONGITUDINAL].thresholdSpeed = settings_.stickySpeedThreshold;
            sticky.stickyParams[PxVehicleTireDirectionModes::eLONGITUDINAL].thresholdTime = settings_.stickyTimeThreshold;
            sticky.stickyParams[PxVehicleTireDirectionModes::eLONGITUDINAL].damping = settings_.stickyDampingLongitudinal;
            sticky.stickyParams[PxVehicleTireDirectionModes::eLATERAL].thresholdSpeed = settings_.stickySpeedThreshold;
            sticky.stickyParams[PxVehicleTireDirectionModes::eLATERAL].thresholdTime = settings_.stickyTimeThreshold;
            sticky.stickyParams[PxVehicleTireDirectionModes::eLATERAL].damping = settings_.stickyDampingLateral;

            auto& slip = simContext_.tireSlipParams;
            slip.minActiveLongSlipDenominator = settings_.minLongSlipDenominatorActive;
            slip.minPassiveLongSlipDenominator = settings_.minLongSlipDenominatorPassive;
            slip.minLatSlipDenominator = settings_.minLatSlipDenominator;
        }

        void createPhysxActor() {
            using namespace ::physx;
            auto& physics = world_->physics();

            chassisActor_ = physics.createRigidDynamic(rigidBodyState_.pose);
            chassisActor_->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
            chassisActor_->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, false);
            chassisActor_->setLinearDamping(settings_.chassisLinearDamping);
            chassisActor_->setAngularDamping(settings_.chassisAngularDamping);

            const PxBoxGeometry chassisGeom(
                    settings_.chassisWidth * 0.5f,
                    settings_.chassisHeight * 0.5f,
                    settings_.chassisLength * 0.5f);
            PxShape* chassisShape = physics.createShape(chassisGeom, world_->defaultMaterial(), true);
            chassisActor_->attachShape(*chassisShape);
            chassisShape->release();

            wheelShapeMaterial_ = physics.createMaterial(0.f, 0.f, 0.f);
            const float halfTrack = settings_.trackWidth * 0.5f;
            const float halfBase = settings_.wheelbase * 0.5f;
            const float ay = settings_.suspensionAttachmentY;
            const PxVec3 wheelInitial[4] = {
                    PxVec3(+halfTrack, ay, +halfBase),
                    PxVec3(-halfTrack, ay, +halfBase),
                    PxVec3(+halfTrack, ay, -halfBase),
                    PxVec3(-halfTrack, ay, -halfBase),
            };
            const PxSphereGeometry wheelGeom(settings_.wheelRadius);
            for (PxU32 i = 0; i < 4; ++i) {
                PxShape* ws = physics.createShape(wheelGeom, *wheelShapeMaterial_, true);
                ws->setLocalPose(PxTransform(wheelInitial[i]));
                ws->setFlag(PxShapeFlag::eSCENE_QUERY_SHAPE, false);
                ws->setFlag(PxShapeFlag::eSIMULATION_SHAPE, false);
                chassisActor_->attachShape(*ws);
                wheelShapeLocalPoses_[i] = ws->getLocalPose();
                physxActor_.wheelShapes[i] = ws;
                ws->release();
            }
            physxActor_.rigidBody = chassisActor_;

            PxRigidBodyExt::setMassAndUpdateInertia(*chassisActor_, settings_.chassisMass);
            world_->scene().addActor(*chassisActor_);

            queryFilter_.ignored = chassisActor_;
        }

        void createPhysxConstraints() {
            ::physx::vehicle2::PxVehicleConstraintsCreate(
                    axleDesc_, world_->physics(), *chassisActor_, physxConstraints_);
            constraintsCreated_ = true;
        }

        void buildComponentSequence() {
            using namespace ::physx::vehicle2;
            componentSequence_.add(static_cast<PxVehiclePhysXActorBeginComponent*>(this));
            componentSequence_.add(static_cast<PxVehiclePhysXRoadGeometrySceneQueryComponent*>(this));

            // Sub-step the integration-heavy components (suspension … rigid body),
            // exactly as PhysxVehicle does. The engine-drive command response +
            // differential sit inside the group between tire and actuation: the
            // differential output (torque split) is consumed by both actuation and
            // drivetrain, so it must run before them.
            substepGroupHandle_ = componentSequence_.beginSubstepGroup(
                    static_cast<::physx::PxU8>(std::max(1u, settings_.subStepCountLowSpeed)));
            componentSequence_.add(static_cast<PxVehicleSuspensionComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleTireComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleEngineDriveCommandResponseComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleMultiWheelDriveDifferentialStateComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleEngineDriveActuationStateComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleEngineDrivetrainComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleWheelComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleRigidBodyComponent*>(this));
            componentSequence_.endSubstepGroup();

            componentSequence_.add(static_cast<PxVehiclePhysXConstraintComponent*>(this));
            componentSequence_.add(static_cast<PxVehiclePhysXActorEndComponent*>(this));
        }

        void stepVehicle(float dt) {
            const float fwdSpeed = std::abs(rigidBodyState_.getLongitudinalSpeed(simContext_.frame));
            const unsigned nSub = fwdSpeed < settings_.subStepThresholdSpeed
                                          ? settings_.subStepCountLowSpeed
                                          : settings_.subStepCountHighSpeed;
            componentSequence_.setSubsteps(substepGroupHandle_,
                                           static_cast<::physx::PxU8>(std::max(1u, nSub)));
            componentSequence_.update(dt, simContext_);
            for (::physx::PxU32 i = 0; i < 4; ++i) {
                if (auto* shape = physxActor_.wheelShapes[i]) {
                    shape->setLocalPose(wheelLocalPoses_[i].localPose);
                }
            }
        }

        // ── Component getDataFor* overrides ───────────────────────────────────

        void getDataForPhysXActorBeginComponent(
                const ::physx::vehicle2::PxVehicleAxleDescription*& axleDescription,
                const ::physx::vehicle2::PxVehicleCommandState*& commands,
                const ::physx::vehicle2::PxVehicleEngineDriveTransmissionCommandState*& transmissionCommands,
                const ::physx::vehicle2::PxVehicleGearboxParams*& gearParams,
                const ::physx::vehicle2::PxVehicleGearboxState*& gearState,
                const ::physx::vehicle2::PxVehicleEngineParams*& engineParams,
                ::physx::vehicle2::PxVehiclePhysXActor*& physxActor,
                ::physx::vehicle2::PxVehiclePhysXSteerState*& physxSteerState,
                ::physx::vehicle2::PxVehiclePhysXConstraints*& physxConstraints,
                ::physx::vehicle2::PxVehicleRigidBodyState*& rigidBodyState,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleWheelRigidBody1dState>& wheelRigidBody1dStates,
                ::physx::vehicle2::PxVehicleEngineState*& engineState) override {
            axleDescription = &axleDesc_;
            commands = &commands_;
            transmissionCommands = &transmissionCommands_;
            gearParams = &gearboxParams_;
            gearState = &gearboxState_;
            engineParams = &engineParams_;
            physxActor = &physxActor_;
            physxSteerState = &physxSteerState_;
            physxConstraints = &physxConstraints_;
            rigidBodyState = &rigidBodyState_;
            wheelRigidBody1dStates.setData(wheelRigidBody1dStates_.data());
            engineState = &engineState_;
        }

        void getDataForPhysXRoadGeometrySceneQueryComponent(
                const ::physx::vehicle2::PxVehicleAxleDescription*& axleDescription,
                const ::physx::vehicle2::PxVehiclePhysXRoadGeometryQueryParams*& roadGeomParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::PxReal>& steerResponseStates,
                const ::physx::vehicle2::PxVehicleRigidBodyState*& rigidBodyState,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleWheelParams>& wheelParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleSuspensionParams>& suspensionParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehiclePhysXMaterialFrictionParams>& materialFrictionParams,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleRoadGeometryState>& roadGeometryStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehiclePhysXRoadGeometryQueryState>& physxRoadGeometryStates) override {
            axleDescription = &axleDesc_;
            roadGeomParams = &physxRoadGeomQueryParams_;
            steerResponseStates.setData(steerResponseStates_.data());
            rigidBodyState = &rigidBodyState_;
            wheelParams.setData(wheelParams_.data());
            suspensionParams.setData(suspensionParams_.data());
            materialFrictionParams.setData(perWheelMaterialFriction_.data());
            roadGeometryStates.setData(roadGeometryStates_.data());
            physxRoadGeometryStates.setEmpty();
        }

        void getDataForSuspensionComponent(
                const ::physx::vehicle2::PxVehicleAxleDescription*& axleDescription,
                const ::physx::vehicle2::PxVehicleRigidBodyParams*& rigidBodyParams,
                const ::physx::vehicle2::PxVehicleSuspensionStateCalculationParams*& suspensionStateCalculationParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::PxReal>& steerResponseStates,
                const ::physx::vehicle2::PxVehicleRigidBodyState*& rigidBodyState,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleWheelParams>& wheelParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleSuspensionParams>& suspensionParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleSuspensionComplianceParams>& suspensionComplianceParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleSuspensionForceParams>& suspensionForceParams,
                ::physx::vehicle2::PxVehicleSizedArrayData<const ::physx::vehicle2::PxVehicleAntiRollForceParams>& antiRollForceParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleRoadGeometryState>& wheelRoadGeomStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleSuspensionState>& suspensionStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleSuspensionComplianceState>& suspensionComplianceStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleSuspensionForce>& suspensionForces,
                ::physx::vehicle2::PxVehicleAntiRollTorque*& antiRollTorque) override {
            axleDescription = &axleDesc_;
            rigidBodyParams = &rigidBodyParams_;
            suspensionStateCalculationParams = &suspensionStateCalcParams_;
            steerResponseStates.setData(steerResponseStates_.data());
            rigidBodyState = &rigidBodyState_;
            wheelParams.setData(wheelParams_.data());
            suspensionParams.setData(suspensionParams_.data());
            suspensionComplianceParams.setData(suspensionComplianceParams_.data());
            suspensionForceParams.setData(suspensionForceParams_.data());
            antiRollForceParams.setEmpty();
            wheelRoadGeomStates.setData(roadGeometryStates_.data());
            suspensionStates.setData(suspensionStates_.data());
            suspensionComplianceStates.setData(suspensionComplianceStates_.data());
            suspensionForces.setData(suspensionForces_.data());
            antiRollTorque = nullptr;
        }

        void getDataForTireComponent(
                const ::physx::vehicle2::PxVehicleAxleDescription*& axleDescription,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::PxReal>& steerResponseStates,
                const ::physx::vehicle2::PxVehicleRigidBodyState*& rigidBodyState,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleWheelActuationState>& actuationStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleWheelParams>& wheelParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleSuspensionParams>& suspensionParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleTireForceParams>& tireForceParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleRoadGeometryState>& roadGeomStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleSuspensionState>& suspensionStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleSuspensionComplianceState>& suspensionComplianceStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleSuspensionForce>& suspensionForces,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleWheelRigidBody1dState>& wheelRigidBody1DStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleTireGripState>& tireGripStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleTireDirectionState>& tireDirectionStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleTireSpeedState>& tireSpeedStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleTireSlipState>& tireSlipStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleTireCamberAngleState>& tireCamberAngleStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleTireStickyState>& tireStickyStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleTireForce>& tireForces) override {
            axleDescription = &axleDesc_;
            steerResponseStates.setData(steerResponseStates_.data());
            rigidBodyState = &rigidBodyState_;
            actuationStates.setData(actuationStates_.data());
            wheelParams.setData(wheelParams_.data());
            suspensionParams.setData(suspensionParams_.data());
            tireForceParams.setData(tireForceParams_.data());
            roadGeomStates.setData(roadGeometryStates_.data());
            suspensionStates.setData(suspensionStates_.data());
            suspensionComplianceStates.setData(suspensionComplianceStates_.data());
            suspensionForces.setData(suspensionForces_.data());
            wheelRigidBody1DStates.setData(wheelRigidBody1dStates_.data());
            tireGripStates.setData(tireGripStates_.data());
            tireDirectionStates.setData(tireDirectionStates_.data());
            tireSpeedStates.setData(tireSpeedStates_.data());
            tireSlipStates.setData(tireSlipStates_.data());
            tireCamberAngleStates.setData(tireCamberAngleStates_.data());
            tireStickyStates.setData(tireStickyStates_.data());
            tireForces.setData(tireForces_.data());
        }

        void getDataForEngineDriveCommandResponseComponent(
                const ::physx::vehicle2::PxVehicleAxleDescription*& axleDescription,
                ::physx::vehicle2::PxVehicleSizedArrayData<const ::physx::vehicle2::PxVehicleBrakeCommandResponseParams>& brakeResponseParams,
                const ::physx::vehicle2::PxVehicleSteerCommandResponseParams*& steerResponseParams,
                ::physx::vehicle2::PxVehicleSizedArrayData<const ::physx::vehicle2::PxVehicleAckermannParams>& ackermannParams,
                const ::physx::vehicle2::PxVehicleGearboxParams*& gearboxParams,
                const ::physx::vehicle2::PxVehicleClutchCommandResponseParams*& clutchResponseParams,
                const ::physx::vehicle2::PxVehicleEngineParams*& engineParams,
                const ::physx::vehicle2::PxVehicleRigidBodyState*& rigidBodyState,
                const ::physx::vehicle2::PxVehicleEngineState*& engineState,
                const ::physx::vehicle2::PxVehicleAutoboxParams*& autoboxParams,
                const ::physx::vehicle2::PxVehicleCommandState*& commands,
                const ::physx::vehicle2::PxVehicleEngineDriveTransmissionCommandState*& transmissionCommands,
                ::physx::vehicle2::PxVehicleArrayData<::physx::PxReal>& brakeResponseStates,
                ::physx::vehicle2::PxVehicleEngineDriveThrottleCommandResponseState*& throttleResponseState,
                ::physx::vehicle2::PxVehicleArrayData<::physx::PxReal>& steerResponseStates,
                ::physx::vehicle2::PxVehicleGearboxState*& gearboxResponseState,
                ::physx::vehicle2::PxVehicleClutchCommandResponseState*& clutchResponseState,
                ::physx::vehicle2::PxVehicleAutoboxState*& autoboxState) override {
            axleDescription = &axleDesc_;
            brakeResponseParams.setDataAndCount(brakeResponseParamArray_.data(),
                                                static_cast<::physx::PxU32>(brakeResponseParamArray_.size()));
            steerResponseParams = &steerResponseParams_;
            ackermannParams.setEmpty();
            gearboxParams = &gearboxParams_;
            clutchResponseParams = &clutchResponseParams_;
            engineParams = &engineParams_;
            rigidBodyState = &rigidBodyState_;
            engineState = &engineState_;
            autoboxParams = useAutobox() ? &autoboxParams_ : nullptr;
            commands = &commands_;
            transmissionCommands = &transmissionCommands_;
            brakeResponseStates.setData(brakeResponseStates_.data());
            throttleResponseState = &throttleResponseState_;
            steerResponseStates.setData(steerResponseStates_.data());
            gearboxResponseState = &gearboxState_;
            clutchResponseState = &clutchResponseState_;
            autoboxState = useAutobox() ? &autoboxState_ : nullptr;
        }

        void getDataForMultiWheelDriveDifferentialStateComponent(
                const ::physx::vehicle2::PxVehicleAxleDescription*& axleDescription,
                const ::physx::vehicle2::PxVehicleMultiWheelDriveDifferentialParams*& differentialParams,
                ::physx::vehicle2::PxVehicleDifferentialState*& differentialState) override {
            axleDescription = &axleDesc_;
            differentialParams = &diffParams_;
            differentialState = &diffState_;
        }

        void getDataForEngineDriveActuationStateComponent(
                const ::physx::vehicle2::PxVehicleAxleDescription*& axleDescription,
                const ::physx::vehicle2::PxVehicleGearboxParams*& gearboxParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::PxReal>& brakeResponseStates,
                const ::physx::vehicle2::PxVehicleEngineDriveThrottleCommandResponseState*& throttleResponseState,
                const ::physx::vehicle2::PxVehicleGearboxState*& gearboxState,
                const ::physx::vehicle2::PxVehicleDifferentialState*& differentialState,
                const ::physx::vehicle2::PxVehicleClutchCommandResponseState*& clutchResponseState,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleWheelActuationState>& actuationStates) override {
            axleDescription = &axleDesc_;
            gearboxParams = &gearboxParams_;
            brakeResponseStates.setData(brakeResponseStates_.data());
            throttleResponseState = &throttleResponseState_;
            gearboxState = &gearboxState_;
            differentialState = &diffState_;
            clutchResponseState = &clutchResponseState_;
            actuationStates.setData(actuationStates_.data());
        }

        void getDataForEngineDrivetrainComponent(
                const ::physx::vehicle2::PxVehicleAxleDescription*& axleDescription,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleWheelParams>& wheelParams,
                const ::physx::vehicle2::PxVehicleEngineParams*& engineParams,
                const ::physx::vehicle2::PxVehicleClutchParams*& clutchParams,
                const ::physx::vehicle2::PxVehicleGearboxParams*& gearboxParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::PxReal>& brakeResponseStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleWheelActuationState>& actuationStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleTireForce>& tireForces,
                const ::physx::vehicle2::PxVehicleEngineDriveThrottleCommandResponseState*& throttleResponseState,
                const ::physx::vehicle2::PxVehicleClutchCommandResponseState*& clutchResponseState,
                const ::physx::vehicle2::PxVehicleDifferentialState*& differentialState,
                const ::physx::vehicle2::PxVehicleWheelConstraintGroupState*& constraintGroupState,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleWheelRigidBody1dState>& wheelRigidBody1dStates,
                ::physx::vehicle2::PxVehicleEngineState*& engineState,
                ::physx::vehicle2::PxVehicleGearboxState*& gearboxState,
                ::physx::vehicle2::PxVehicleClutchSlipState*& clutchState) override {
            axleDescription = &axleDesc_;
            wheelParams.setData(wheelParams_.data());
            engineParams = &engineParams_;
            clutchParams = &clutchParams_;
            gearboxParams = &gearboxParams_;
            brakeResponseStates.setData(brakeResponseStates_.data());
            actuationStates.setData(actuationStates_.data());
            tireForces.setData(tireForces_.data());
            throttleResponseState = &throttleResponseState_;
            clutchResponseState = &clutchResponseState_;
            differentialState = &diffState_;
            constraintGroupState = nullptr;// MultiWheel differential has no wheel constraint groups
            wheelRigidBody1dStates.setData(wheelRigidBody1dStates_.data());
            engineState = &engineState_;
            gearboxState = &gearboxState_;
            clutchState = &clutchSlipState_;
        }

        void getDataForWheelComponent(
                const ::physx::vehicle2::PxVehicleAxleDescription*& axleDescription,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::PxReal>& steerResponseStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleWheelParams>& wheelParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleSuspensionParams>& suspensionParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleWheelActuationState>& actuationStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleSuspensionState>& suspensionStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleSuspensionComplianceState>& suspensionComplianceStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleTireSpeedState>& tireSpeedStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleWheelRigidBody1dState>& wheelRigidBody1dStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleWheelLocalPose>& wheelLocalPoses) override {
            axleDescription = &axleDesc_;
            steerResponseStates.setData(steerResponseStates_.data());
            wheelParams.setData(wheelParams_.data());
            suspensionParams.setData(suspensionParams_.data());
            actuationStates.setData(actuationStates_.data());
            suspensionStates.setData(suspensionStates_.data());
            suspensionComplianceStates.setData(suspensionComplianceStates_.data());
            tireSpeedStates.setData(tireSpeedStates_.data());
            wheelRigidBody1dStates.setData(wheelRigidBody1dStates_.data());
            wheelLocalPoses.setData(wheelLocalPoses_.data());
        }

        void getDataForRigidBodyComponent(
                const ::physx::vehicle2::PxVehicleAxleDescription*& axleDescription,
                const ::physx::vehicle2::PxVehicleRigidBodyParams*& rigidBodyParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleSuspensionForce>& suspensionForces,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleTireForce>& tireForces,
                const ::physx::vehicle2::PxVehicleAntiRollTorque*& antiRollTorque,
                ::physx::vehicle2::PxVehicleRigidBodyState*& rigidBodyState) override {
            axleDescription = &axleDesc_;
            rigidBodyParams = &rigidBodyParams_;
            suspensionForces.setData(suspensionForces_.data());
            tireForces.setData(tireForces_.data());
            antiRollTorque = nullptr;
            rigidBodyState = &rigidBodyState_;
        }

        void getDataForPhysXConstraintComponent(
                const ::physx::vehicle2::PxVehicleAxleDescription*& axleDescription,
                const ::physx::vehicle2::PxVehicleRigidBodyState*& rigidBodyState,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleSuspensionParams>& suspensionParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehiclePhysXSuspensionLimitConstraintParams>& suspensionLimitParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleSuspensionState>& suspensionStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleSuspensionComplianceState>& suspensionComplianceStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleRoadGeometryState>& wheelRoadGeomStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleTireDirectionState>& tireDirectionStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleTireStickyState>& tireStickyStates,
                ::physx::vehicle2::PxVehiclePhysXConstraints*& constraints) override {
            axleDescription = &axleDesc_;
            rigidBodyState = &rigidBodyState_;
            suspensionParams.setData(suspensionParams_.data());
            suspensionLimitParams.setData(suspensionLimitParams_.data());
            suspensionStates.setData(suspensionStates_.data());
            suspensionComplianceStates.setData(suspensionComplianceStates_.data());
            wheelRoadGeomStates.setData(roadGeometryStates_.data());
            tireDirectionStates.setData(tireDirectionStates_.data());
            tireStickyStates.setData(tireStickyStates_.data());
            constraints = &physxConstraints_;
        }

        void getDataForPhysXActorEndComponent(
                const ::physx::vehicle2::PxVehicleAxleDescription*& axleDescription,
                const ::physx::vehicle2::PxVehicleRigidBodyState*& rigidBodyState,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleWheelParams>& wheelParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::PxTransform>& wheelShapeLocalPoses,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleWheelRigidBody1dState>& wheelRigidBody1dStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleWheelLocalPose>& wheelLocalPoses,
                const ::physx::vehicle2::PxVehicleGearboxState*& gearState,
                const ::physx::PxReal*& throttle,
                ::physx::vehicle2::PxVehiclePhysXActor*& physxActor) override {
            axleDescription = &axleDesc_;
            rigidBodyState = &rigidBodyState_;
            wheelParams.setData(wheelParams_.data());
            wheelShapeLocalPoses.setData(wheelShapeLocalPoses_.data());
            wheelRigidBody1dStates.setData(wheelRigidBody1dStates_.data());
            wheelLocalPoses.setData(wheelLocalPoses_.data());
            gearState = &gearboxState_;
            throttle = &commands_.throttle;
            physxActor = &physxActor_;
        }

        // ── Internal types ────────────────────────────────────────────────────

        struct ChassisQueryFilter : public ::physx::PxQueryFilterCallback {
            ::physx::PxRigidActor const* ignored = nullptr;

            ::physx::PxQueryHitType::Enum preFilter(
                    const ::physx::PxFilterData&,
                    const ::physx::PxShape*,
                    const ::physx::PxRigidActor* actor,
                    ::physx::PxHitFlags&) override {
                return actor == ignored ? ::physx::PxQueryHitType::eNONE : ::physx::PxQueryHitType::eBLOCK;
            }

            ::physx::PxQueryHitType::Enum postFilter(
                    const ::physx::PxFilterData&,
                    const ::physx::PxQueryHit&,
                    const ::physx::PxShape*,
                    const ::physx::PxRigidActor*) override {
                return ::physx::PxQueryHitType::eBLOCK;
            }
        };

        // ── State ─────────────────────────────────────────────────────────────

        PhysxWorld* world_ = nullptr;
        Settings settings_;
        bool initialised_ = false;

        std::function<void(float)> stepCallback_;

        ::physx::PxRigidDynamic* chassisActor_ = nullptr;
        ::physx::PxMaterial* wheelShapeMaterial_ = nullptr;

        TransmissionMode mode_ = TransmissionMode::Automatic;
        Direction direction_ = Direction::Drive;
        ::physx::PxU32 manualForwardGear_ = 2;// set properly in buildCommandResponseParams

        ::physx::vehicle2::PxVehicleAxleDescription axleDesc_{};
        ::physx::vehicle2::PxVehicleRigidBodyParams rigidBodyParams_{};
        ::physx::vehicle2::PxVehicleRigidBodyState rigidBodyState_{};

        std::array<::physx::vehicle2::PxVehicleWheelParams, 4> wheelParams_{};
        std::array<::physx::vehicle2::PxVehicleSuspensionParams, 4> suspensionParams_{};
        std::array<::physx::vehicle2::PxVehicleSuspensionComplianceParams, 4> suspensionComplianceParams_{};
        std::array<::physx::vehicle2::PxVehicleSuspensionForceParams, 4> suspensionForceParams_{};
        std::array<::physx::vehicle2::PxVehiclePhysXSuspensionLimitConstraintParams, 4> suspensionLimitParams_{};
        ::physx::vehicle2::PxVehicleSuspensionStateCalculationParams suspensionStateCalcParams_{};
        std::array<::physx::vehicle2::PxVehicleTireForceParams, 4> tireForceParams_{};

        std::array<::physx::vehicle2::PxVehicleSuspensionState, 4> suspensionStates_{};
        std::array<::physx::vehicle2::PxVehicleSuspensionComplianceState, 4> suspensionComplianceStates_{};
        std::array<::physx::vehicle2::PxVehicleSuspensionForce, 4> suspensionForces_{};
        std::array<::physx::vehicle2::PxVehicleTireDirectionState, 4> tireDirectionStates_{};
        std::array<::physx::vehicle2::PxVehicleTireSpeedState, 4> tireSpeedStates_{};
        std::array<::physx::vehicle2::PxVehicleTireSlipState, 4> tireSlipStates_{};
        std::array<::physx::vehicle2::PxVehicleTireGripState, 4> tireGripStates_{};
        std::array<::physx::vehicle2::PxVehicleTireCamberAngleState, 4> tireCamberAngleStates_{};
        std::array<::physx::vehicle2::PxVehicleTireStickyState, 4> tireStickyStates_{};
        std::array<::physx::vehicle2::PxVehicleTireForce, 4> tireForces_{};

        std::array<::physx::vehicle2::PxVehicleWheelActuationState, 4> actuationStates_{};
        std::array<::physx::vehicle2::PxVehicleWheelRigidBody1dState, 4> wheelRigidBody1dStates_{};
        std::array<::physx::vehicle2::PxVehicleWheelLocalPose, 4> wheelLocalPoses_{};
        std::array<::physx::PxTransform, 4> wheelShapeLocalPoses_{
                ::physx::PxTransform(::physx::PxIdentity),
                ::physx::PxTransform(::physx::PxIdentity),
                ::physx::PxTransform(::physx::PxIdentity),
                ::physx::PxTransform(::physx::PxIdentity)};

        std::array<::physx::PxReal, 4> brakeResponseStates_{};
        std::array<::physx::PxReal, 4> steerResponseStates_{};

        // ── Engine-drive params + state ──
        ::physx::vehicle2::PxVehicleEngineParams engineParams_{};
        ::physx::vehicle2::PxVehicleClutchParams clutchParams_{};
        ::physx::vehicle2::PxVehicleClutchCommandResponseParams clutchResponseParams_{};
        ::physx::vehicle2::PxVehicleGearboxParams gearboxParams_{};
        ::physx::vehicle2::PxVehicleAutoboxParams autoboxParams_{};
        ::physx::vehicle2::PxVehicleMultiWheelDriveDifferentialParams diffParams_{};

        ::physx::vehicle2::PxVehicleEngineState engineState_{};
        ::physx::vehicle2::PxVehicleGearboxState gearboxState_{};
        ::physx::vehicle2::PxVehicleClutchCommandResponseState clutchResponseState_{};
        ::physx::vehicle2::PxVehicleClutchSlipState clutchSlipState_{};
        ::physx::vehicle2::PxVehicleAutoboxState autoboxState_{};
        ::physx::vehicle2::PxVehicleDifferentialState diffState_{};
        ::physx::vehicle2::PxVehicleEngineDriveThrottleCommandResponseState throttleResponseState_{};
        ::physx::vehicle2::PxVehicleWheelConstraintGroupState constraintGroupState_{};

        // Foot brake (cmd 0) + handbrake (cmd 1) response params, packed contiguously
        // so they can be handed to the command-response component as a sized array.
        ::physx::vehicle2::PxVehicleBrakeCommandResponseParams brakeResponseParams_{};
        ::physx::vehicle2::PxVehicleBrakeCommandResponseParams handbrakeResponseParams_{};
        std::array<::physx::vehicle2::PxVehicleBrakeCommandResponseParams, 2> brakeResponseParamArray_{};

        ::physx::vehicle2::PxVehicleSteerCommandResponseParams steerResponseParams_{};

        ::physx::vehicle2::PxVehicleCommandState commands_{};
        ::physx::vehicle2::PxVehicleEngineDriveTransmissionCommandState transmissionCommands_{};

        ::physx::vehicle2::PxVehiclePhysXActor physxActor_{};
        ::physx::vehicle2::PxVehiclePhysXSteerState physxSteerState_{};
        ::physx::vehicle2::PxVehiclePhysXConstraints physxConstraints_{};
        bool constraintsCreated_ = false;
        ::physx::vehicle2::PxVehiclePhysXRoadGeometryQueryParams physxRoadGeomQueryParams_;
        ::physx::vehicle2::PxVehiclePhysXMaterialFriction materialFriction_{};
        std::array<::physx::vehicle2::PxVehiclePhysXMaterialFrictionParams, 4> perWheelMaterialFriction_{};
        std::array<::physx::vehicle2::PxVehicleRoadGeometryState, 4> roadGeometryStates_{};

        ChassisQueryFilter queryFilter_{};

        ::physx::vehicle2::PxVehiclePhysXSimulationContext simContext_{};
        ::physx::vehicle2::PxVehicleComponentSequence componentSequence_{};
        ::physx::PxU8 substepGroupHandle_ = 0;
    };

}// namespace threepp

#endif//THREEPP_PHYSX_VEHICLE_ENGINE_DRIVE_HPP
