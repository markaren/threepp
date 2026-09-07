
#ifndef THREEPP_PHYSX_VEHICLE_ENGINE_DRIVE_HPP
#define THREEPP_PHYSX_VEHICLE_ENGINE_DRIVE_HPP

#include "threepp/extras/physx/PhysxVehicleBase.hpp"

#include <string>
#include <vector>

namespace threepp {

    // Engine-drive tuning: overrides several shared defaults (a heavier chassis
    // with grippier tires) and adds the engine/clutch/gearbox/autobox/differential
    // stack. Engine speeds are radians/second internally; the vehicle's engineRpm()
    // converts for display.
    struct PhysxVehicleEngineDriveSettings : PhysxVehicleBaseSettings {
        PhysxVehicleEngineDriveSettings() {
            chassisWidth = 1.9f;
            chassisHeight = 1.3f;
            chassisLength = 4.4f;
            chassisMass = 1600.f;
            wheelRadius = 0.36f;
            wheelHalfWidth = 0.16f;
            trackWidth = 1.65f;
            wheelbase = 2.7f;
            suspensionTravelDist = 0.32f;
            suspensionStiffness = 38'000.f;
            suspensionDamping = 5000.f;
            suspensionAttachmentY = -0.35f;
            maxBrakeTorque = 6000.f;
            drivenWheels = {true, true, true, true};// which wheels receive engine torque; default = all-wheel drive
            tireFriction = 2.0f;
            lateralStiffness = 90'000.f;
            longitudinalStiffness = 100'000.f;
            wheelDampingRate = 0.25f;// lower than direct-drive: the engine/gearbox now provide the coasting drag
        }

        // ── Brakes ───────────────────────────────────────────────────────
        float maxHandbrakeTorque = 9000.f;// N*m, rear wheels only

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
    };

    // Engine-driven 4-wheel vehicle built on the full PxVehicle2 component stack:
    // engine (torque curve + revs) → clutch → multi-ratio gearbox (+ autobox) →
    // differential → wheels. This is the "fully featured" sibling of PhysxVehicle
    // (which is direct-drive, no engine). It produces real engine RPM, automatic or
    // manual gear shifts, clutch slip, and engine braking — telemetry the HUD and
    // procedural audio can read directly instead of synthesising.
    //
    // PhysxVehicleBase owns the chassis/suspension/tire machinery and the stepping
    // scaffold; only the drive components live here — DirectDrive{CommandResponse,
    // ActuationState,Drivetrain} are replaced by EngineDrive{CommandResponse,
    // ActuationState,Drivetrain} and a MultiWheelDriveDifferential is inserted
    // between command response and actuation (both downstream components consume
    // the differential torque split).
    //
    // Frame convention is threepp-native: lng=+Z (forward), lat=+X (right), vrt=+Y (up).
    // SI units: meters, kg, seconds. Engine speeds are radians/second internally;
    // engineRPM() converts for display.
    class PhysxVehicleEngineDrive final
        : public PhysxVehicleBase<PhysxVehicleEngineDriveSettings>,
          public ::physx::vehicle2::PxVehicleEngineDriveCommandResponseComponent,
          public ::physx::vehicle2::PxVehicleMultiWheelDriveDifferentialStateComponent,
          public ::physx::vehicle2::PxVehicleEngineDriveActuationStateComponent,
          public ::physx::vehicle2::PxVehicleEngineDrivetrainComponent {

    public:
        using Settings = PhysxVehicleEngineDriveSettings;

        enum class TransmissionMode { Automatic,
                                      Manual };

        // Driver-facing gear selector (like P/R/N/D). In Drive with Automatic mode
        // the autobox picks the forward gear; in Manual it uses shiftUp/shiftDown.
        enum class Direction { Reverse,
                               Neutral,
                               Drive };

        explicit PhysxVehicleEngineDrive(PhysxWorld& world, const Settings& s = Settings())
            : PhysxVehicleBase(world, validated(s)) {

            buildEngineDriveParams();
            buildCommandResponseParams();
            completeSetup();
        }

        // ── Inputs (throttle/brake/steer are inherited from PhysxVehicleBase) ─

        void setHandbrake(float v) { commands_.brakes[1] = std::clamp(v, 0.f, 1.f); }
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

    private:
        static constexpr float kTwoPi = 6.28318530717958647692f;
        static constexpr float kRpmToRadPerSec = kTwoPi / 60.f;

        // Validated pass-through so the gear-ratio check runs BEFORE the base
        // constructor initialises the PhysX vehicle extension.
        static const Settings& validated(const Settings& s) {
            if (s.gearRatios.size() < 3) {
                throw std::runtime_error("PhysxVehicleEngineDrive: need at least reverse+neutral+1 forward gear");
            }
            return s;
        }

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

            buildSteerResponseParams();

            initCommandState(2);// foot brake + handbrake

            transmissionCommands_.clutch = 0.f;
            manualForwardGear_ = firstForwardGearIndex();
            applyTransmissionCommand();
        }

        void addDrivetrainComponents() override {
            using namespace ::physx::vehicle2;
            // The engine-drive command response + differential sit inside the
            // substep group between tire and actuation: the differential output
            // (torque split) is consumed by both actuation and drivetrain, so it
            // must run before them.
            componentSequence_.add(static_cast<PxVehicleEngineDriveCommandResponseComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleMultiWheelDriveDifferentialStateComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleEngineDriveActuationStateComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleEngineDrivetrainComponent*>(this));
        }

        // ── Component getDataFor* overrides (drive-specific) ─────────────────

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

        // ── State ─────────────────────────────────────────────────────────────

        TransmissionMode mode_ = TransmissionMode::Automatic;
        Direction direction_ = Direction::Drive;
        ::physx::PxU32 manualForwardGear_ = 2;// set properly in buildCommandResponseParams

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

        ::physx::vehicle2::PxVehicleEngineDriveTransmissionCommandState transmissionCommands_{};
    };

}// namespace threepp

#endif//THREEPP_PHYSX_VEHICLE_ENGINE_DRIVE_HPP
