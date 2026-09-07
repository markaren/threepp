
#ifndef THREEPP_PHYSX_VEHICLE_HPP
#define THREEPP_PHYSX_VEHICLE_HPP

#include "threepp/extras/physx/PhysxVehicleBase.hpp"

namespace threepp {

    // Direct-drive tuning. The shared defaults in PhysxVehicleBaseSettings ARE the
    // direct-drive tuning, so only the drive command itself is added here.
    struct PhysxVehicleSettings : PhysxVehicleBaseSettings {
        // Drive parameters. Throttle torque goes to the wheels selected by
        // drivenWheels (default = rear-wheel drive).
        float maxThrottleTorque = 1500.f;// N*m at full throttle
    };

    // Direct-drive 4-wheel vehicle built on PxVehicle2 components: throttle torque
    // is applied straight to the driven wheels — no engine/clutch/gearbox model.
    // PhysxVehicleBase owns the chassis/suspension/tire machinery and the stepping
    // scaffold; this class adds the DirectDrive command-response/actuation/
    // drivetrain components. The user supplies command inputs (throttle/brake/
    // steer/gear) and reads back chassis pose, per-wheel local poses, and forward
    // speed.
    //
    // Frame convention is threepp-native: lng=+Z (forward), lat=+X (right), vrt=+Y (up).
    // SI units: meters, kg, seconds.
    class PhysxVehicle final
        : public PhysxVehicleBase<PhysxVehicleSettings>,
          public ::physx::vehicle2::PxVehicleDirectDriveCommandResponseComponent,
          public ::physx::vehicle2::PxVehicleDirectDriveActuationStateComponent,
          public ::physx::vehicle2::PxVehicleDirectDrivetrainComponent {

    public:
        using Settings = PhysxVehicleSettings;

        enum class Gear : int { Reverse = 0,
                                Neutral = 1,
                                Forward = 2 };

        explicit PhysxVehicle(PhysxWorld& world, const Settings& s = Settings())
            : PhysxVehicleBase(world, s) {

            buildCommandResponseParams();
            completeSetup();
        }

        // -- Inputs (throttle/brake/steer are inherited from PhysxVehicleBase) --

        void setGear(Gear g) { transmissionCommands_.gear = static_cast<::physx::vehicle2::PxVehicleDirectDriveTransmissionCommandState::Enum>(g); }

        // -- Readouts --

        Gear gear() const { return static_cast<Gear>(transmissionCommands_.gear); }

    private:
        // ---- Construction helpers ----

        void buildCommandResponseParams() {
            using namespace ::physx;

            std::memset(&throttleResponseParams_, 0, sizeof(throttleResponseParams_));
            throttleResponseParams_.maxResponse = settings_.maxThrottleTorque;
            for (PxU32 i = 0; i < 4; ++i) {
                throttleResponseParams_.wheelResponseMultipliers[i] = settings_.drivenWheels[i] ? 1.f : 0.f;
            }

            buildSteerResponseParams();

            std::memset(&brakeResponseParams_, 0, sizeof(brakeResponseParams_));
            brakeResponseParams_.maxResponse = settings_.maxBrakeTorque;
            for (PxU32 i = 0; i < 4; ++i) {
                brakeResponseParams_.wheelResponseMultipliers[i] = 1.f;
            }

            initCommandState(1);// single brake command

            transmissionCommands_.gear = ::physx::vehicle2::PxVehicleDirectDriveTransmissionCommandState::eFORWARD;
        }

        void addDrivetrainComponents() override {
            using namespace ::physx::vehicle2;
            componentSequence_.add(static_cast<PxVehicleDirectDriveCommandResponseComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleDirectDriveActuationStateComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleDirectDrivetrainComponent*>(this));
        }

        // ---- Component getDataFor* overrides (drive-specific) ----

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
            transmissionCommands = nullptr;
            gearParams = nullptr;
            gearState = nullptr;
            engineParams = nullptr;
            physxActor = &physxActor_;
            physxSteerState = &physxSteerState_;
            physxConstraints = &physxConstraints_;
            rigidBodyState = &rigidBodyState_;
            wheelRigidBody1dStates.setData(wheelRigidBody1dStates_.data());
            engineState = nullptr;
        }

        void getDataForDirectDriveCommandResponseComponent(
                const ::physx::vehicle2::PxVehicleAxleDescription*& axleDescription,
                ::physx::vehicle2::PxVehicleSizedArrayData<const ::physx::vehicle2::PxVehicleBrakeCommandResponseParams>& brakeResponseParams,
                const ::physx::vehicle2::PxVehicleDirectDriveThrottleCommandResponseParams*& throttleResponseParams,
                const ::physx::vehicle2::PxVehicleSteerCommandResponseParams*& steerResponseParams,
                ::physx::vehicle2::PxVehicleSizedArrayData<const ::physx::vehicle2::PxVehicleAckermannParams>& ackermannParams,
                const ::physx::vehicle2::PxVehicleCommandState*& commands,
                const ::physx::vehicle2::PxVehicleDirectDriveTransmissionCommandState*& transmissionCommands,
                const ::physx::vehicle2::PxVehicleRigidBodyState*& rigidBodyState,
                ::physx::vehicle2::PxVehicleArrayData<::physx::PxReal>& brakeResponseStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::PxReal>& throttleResponseStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::PxReal>& steerResponseStates) override {
            axleDescription = &axleDesc_;
            brakeResponseParams.setDataAndCount(&brakeResponseParams_, 1);
            throttleResponseParams = &throttleResponseParams_;
            steerResponseParams = &steerResponseParams_;
            ackermannParams.setEmpty();
            commands = &commands_;
            transmissionCommands = &transmissionCommands_;
            rigidBodyState = &rigidBodyState_;
            brakeResponseStates.setData(brakeResponseStates_.data());
            throttleResponseStates.setData(throttleResponseStates_.data());
            steerResponseStates.setData(steerResponseStates_.data());
        }

        void getDataForDirectDriveActuationStateComponent(
                const ::physx::vehicle2::PxVehicleAxleDescription*& axleDescription,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::PxReal>& brakeResponseStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::PxReal>& throttleResponseStates,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleWheelActuationState>& actuationStates) override {
            axleDescription = &axleDesc_;
            brakeResponseStates.setData(brakeResponseStates_.data());
            throttleResponseStates.setData(throttleResponseStates_.data());
            actuationStates.setData(actuationStates_.data());
        }

        void getDataForDirectDrivetrainComponent(
                const ::physx::vehicle2::PxVehicleAxleDescription*& axleDescription,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::PxReal>& brakeResponseStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::PxReal>& throttleResponseStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleWheelParams>& wheelParams,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleWheelActuationState>& actuationStates,
                ::physx::vehicle2::PxVehicleArrayData<const ::physx::vehicle2::PxVehicleTireForce>& tireForces,
                ::physx::vehicle2::PxVehicleArrayData<::physx::vehicle2::PxVehicleWheelRigidBody1dState>& wheelRigidBody1dStates) override {
            axleDescription = &axleDesc_;
            brakeResponseStates.setData(brakeResponseStates_.data());
            throttleResponseStates.setData(throttleResponseStates_.data());
            wheelParams.setData(wheelParams_.data());
            actuationStates.setData(actuationStates_.data());
            tireForces.setData(tireForces_.data());
            wheelRigidBody1dStates.setData(wheelRigidBody1dStates_.data());
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
            gearState = nullptr;
            throttle = &commands_.throttle;
            physxActor = &physxActor_;
        }

        // ---- State ----

        std::array<::physx::PxReal, 4> throttleResponseStates_{};

        ::physx::vehicle2::PxVehicleDirectDriveThrottleCommandResponseParams throttleResponseParams_{};
        ::physx::vehicle2::PxVehicleBrakeCommandResponseParams brakeResponseParams_{};

        ::physx::vehicle2::PxVehicleDirectDriveTransmissionCommandState transmissionCommands_{};
    };

}// namespace threepp

#endif//THREEPP_PHYSX_VEHICLE_HPP
