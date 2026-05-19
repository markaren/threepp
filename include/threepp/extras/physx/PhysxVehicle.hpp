
#ifndef THREEPP_PHYSX_VEHICLE_HPP
#define THREEPP_PHYSX_VEHICLE_HPP

#include "threepp/extras/physx/PhysxWorld.hpp"

#include <PxPhysicsAPI.h>
#include <vehicle2/PxVehicleAPI.h>

#include <array>
#include <stdexcept>

namespace threepp {

    // Direct-drive 4-wheel vehicle built on PxVehicle2 components. Creates its own
    // PxRigidDynamic chassis + 4 wheel shapes, wires the standard 10-component
    // sequence, and steps via PhysxWorld::onPreSubstep so the substep cadence stays
    // in sync with PhysX. The user supplies command inputs (throttle/brake/steer/gear)
    // and reads back chassis pose, per-wheel local poses, and forward speed.
    //
    // Frame convention is threepp-native: lng=+Z (forward), lat=+X (right), vrt=+Y (up).
    // SI units: meters, kg, seconds.
    class PhysxVehicle final
        : public ::physx::vehicle2::PxVehiclePhysXActorBeginComponent,
          public ::physx::vehicle2::PxVehiclePhysXRoadGeometrySceneQueryComponent,
          public ::physx::vehicle2::PxVehicleSuspensionComponent,
          public ::physx::vehicle2::PxVehicleTireComponent,
          public ::physx::vehicle2::PxVehicleDirectDriveCommandResponseComponent,
          public ::physx::vehicle2::PxVehicleDirectDriveActuationStateComponent,
          public ::physx::vehicle2::PxVehicleDirectDrivetrainComponent,
          public ::physx::vehicle2::PxVehicleWheelComponent,
          public ::physx::vehicle2::PxVehicleRigidBodyComponent,
          public ::physx::vehicle2::PxVehiclePhysXConstraintComponent,
          public ::physx::vehicle2::PxVehiclePhysXActorEndComponent {

    public:
        struct Settings {
            // Chassis box dimensions (meters).
            float chassisWidth = 1.8f;
            float chassisHeight = 1.0f;
            float chassisLength = 4.5f;

            // Chassis mass (kg). Inertia computed from a box of these dims and this mass.
            float chassisMass = 1500.f;

            // Wheel geometry (meters / kg).
            float wheelRadius = 0.4f;
            float wheelHalfWidth = 0.15f;
            float wheelMass = 25.f;

            // Track + wheelbase (defaults derived from chassis dims). Track is X-spread,
            // wheelbase is Z-spread between the two axles.
            float trackWidth = 1.6f;
            float wheelbase = 2.8f;

            // Suspension travel distance (meters), spring stiffness (N/m), damping (Ns/m).
            float suspensionTravelDist = 0.3f;
            float suspensionStiffness = 35'000.f;
            float suspensionDamping = 4500.f;

            // Vertical offset (chassis frame) of suspension attachment - the wheel pose
            // at maximum compression. Negative => below chassis center.
            float suspensionAttachmentY = -0.4f;

            // Drive parameters.
            float maxThrottleTorque = 1500.f;// N*m at full throttle, applied to rear wheels
            float maxBrakeTorque = 5000.f;   // N*m at full brake, applied to all wheels
            float maxSteerAngleRad = 0.6f;   // ~34 deg, applied to front wheels (must be <= PI)

            // Spawn pose.
            Vector3 spawnPosition{0, 1.0f, 0};
            Quaternion spawnRotation;

            // Drive layout: which wheels receive throttle. Default = rear-wheel drive.
            // Indices: 0 = front-right, 1 = front-left, 2 = rear-right, 3 = rear-left.
            std::array<bool, 4> drivenWheels{false, false, true, true};

            // Tire friction on the default ground material. PhysX scales the linear
            // tire forces by mu * load, so this is a hard ceiling on grip.
            float tireFriction = 1.5f;

            // Wheel spin damping (N*m*s/rad). Models engine braking + rolling
            // resistance; higher = car decelerates faster when coasting.
            float wheelDampingRate = 2.0f;

            // Linear/angular damping applied to the chassis actor. Linear models
            // aerodynamic drag; angular damps yaw oscillations after sharp turns.
            float chassisLinearDamping = 0.1f;
            float chassisAngularDamping = 0.5f;

            // Lateral cornering stiffness (N / rad of slip). Determines how quickly
            // the tire responds to steering. Realistic cars: 40k-150k.
            float lateralStiffness = 80'000.f;

            // Longitudinal stiffness (N / unit slip ratio). Determines how quickly
            // the tire responds to throttle/brake. Pre-friction-cap value.
            float longitudinalStiffness = 50'000.f;

            // Sticky-tire engagement. Once the tire's slip speed has stayed below
            // `stickySpeedThreshold` for `stickyTimeThreshold` seconds (without a
            // drive torque applied), a velocity constraint pins the contact patch
            // to the road, killing residual drift that the slip-based tire force
            // can't fully resolve at near-zero speeds. PhysX's stock defaults
            // (time = 1.0 s, lateral damping = 0.1) let the car visibly creep
            // forward and slide sideways while stopping — tighter values here
            // make stops decisive without hurting normal driving (sticky only
            // engages near 0 m/s).
            float stickySpeedThreshold = 0.2f;     // m/s; same as PhysX default
            float stickyTimeThreshold = 0.1f;      // s;   PhysX default is 1.0
            float stickyDampingLongitudinal = 1.0f;// same as PhysX default
            float stickyDampingLateral = 1.0f;     // PhysX default is 0.1
        };

        enum class Gear : int { Reverse = 0,
                                Neutral = 1,
                                Forward = 2 };

        explicit PhysxVehicle(PhysxWorld& world, const Settings& s = Settings())
            : world_(&world), settings_(s) {

            using namespace ::physx;
            using namespace ::physx::vehicle2;

            if (!PxInitVehicleExtension(world.foundation())) {
                throw std::runtime_error("PxInitVehicleExtension failed");
            }
            initialised_ = true;

            buildAxleDescription();
            buildRigidBodyParams();
            buildWheelParams();
            buildSuspensionParams();
            buildTireParams();
            buildCommandResponseParams();
            buildSimContext();

            createPhysxActor();
            createPhysxConstraints();
            buildComponentSequence();

            stepCallback_ = [this](float dt) { stepVehicle(dt); };
            world_->onPreSubstep(stepCallback_);
        }

        ~PhysxVehicle() {
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

        PhysxVehicle(const PhysxVehicle&) = delete;
        PhysxVehicle& operator=(const PhysxVehicle&) = delete;

        // -- Inputs --

        void setThrottle(float v) { commands_.throttle = std::clamp(v, 0.f, 1.f); }
        void setBrake(float v) { commands_.brakes[0] = std::clamp(v, 0.f, 1.f); }
        void setSteer(float v) { commands_.steer = std::clamp(v, -1.f, 1.f); }
        void setGear(Gear g) { transmissionCommands_.gear = static_cast<::physx::vehicle2::PxVehicleDirectDriveTransmissionCommandState::Enum>(g); }

        // -- Readouts --

        ::physx::PxRigidDynamic* chassisActor() const { return chassisActor_; }
        ::physx::PxTransform chassisPose() const { return rigidBodyState_.pose; }

        ::physx::PxTransform wheelLocalPose(int i) const {
            return wheelLocalPoses_[i].localPose;
        }

        // Forward speed in chassis longitudinal direction (m/s).
        float forwardSpeed() const {
            return rigidBodyState_.getLongitudinalSpeed(simContext_.frame);
        }

        // Wheel spin angle in radians, in (-2π, 2π).
        float wheelRotationAngle(int i) const {
            return wheelRigidBody1dStates_[i].rotationAngle;
        }

        // Wheel spin rate in rad/s.
        float wheelAngularSpeed(int i) const {
            return wheelRigidBody1dStates_[i].rotationSpeed;
        }

        Gear gear() const { return static_cast<Gear>(transmissionCommands_.gear); }

        const Settings& settings() const { return settings_; }

    private:
        // ---- Construction helpers ----

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
            // Box inertia: I_x = m/12 (h^2 + L^2), I_y = m/12 (W^2 + L^2), I_z = m/12 (W^2 + h^2)
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
                // Spin damping models engine/drivetrain friction + rolling resistance.
                // Without it the car coasts forever after throttle release.
                wheelParams_[i].dampingRate = settings_.wheelDampingRate;
            }
        }

        void buildSuspensionParams() {
            using namespace ::physx;
            const float halfTrack = settings_.trackWidth * 0.5f;
            const float halfBase = settings_.wheelbase * 0.5f;
            const float ay = settings_.suspensionAttachmentY;

            // Wheel id -> chassis-space (X, Z) position. Y from settings_.suspensionAttachmentY.
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

                // Empty compliance lookup tables -> zero compliance (toe/camber/offsets all 0).
                suspensionComplianceParams_[i].wheelToeAngle.clear();
                suspensionComplianceParams_[i].wheelCamberAngle.clear();
                suspensionComplianceParams_[i].suspForceAppPoint.clear();
                suspensionComplianceParams_[i].tireForceAppPoint.clear();

                // Constraint params for the PhysXConstraintComponent. eROAD_GEOMETRY_NORMAL
                // pushes wheels back to the ground along the road normal — recommended by the
                // PhysX docs over eSUSPENSION (which can self-right the car when on its side).
                // restitution=0 disables the velocity restitution model.
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
                // latStiffX = 0 -> tire lateral stiffness is independent of load
                // (uses latStiffY directly for all loads). Simpler tuning.
                tireForceParams_[i].latStiffX = 0.f;
                tireForceParams_[i].latStiffY = settings_.lateralStiffness;
                tireForceParams_[i].longStiff = settings_.longitudinalStiffness;
                tireForceParams_[i].camberStiff = 0.f;
                tireForceParams_[i].restLoad = settings_.chassisMass * 0.25f * 9.81f;

                // friction vs slip: flat unit profile across slip range (linear ramp 0->1, plateau, linear).
                tireForceParams_[i].frictionVsSlip[0][0] = 0.f;
                tireForceParams_[i].frictionVsSlip[0][1] = 1.f;
                tireForceParams_[i].frictionVsSlip[1][0] = 0.1f;
                tireForceParams_[i].frictionVsSlip[1][1] = 1.f;
                tireForceParams_[i].frictionVsSlip[2][0] = 1.f;
                tireForceParams_[i].frictionVsSlip[2][1] = 1.f;

                // load filter: identity (input load = filtered load) at low and high points.
                tireForceParams_[i].loadFilter[0][0] = 0.f;
                tireForceParams_[i].loadFilter[0][1] = 0.23f;
                tireForceParams_[i].loadFilter[1][0] = 3.f;
                tireForceParams_[i].loadFilter[1][1] = 3.f;
            }

            // Default ground material maps to settings_.tireFriction.
            materialFriction_.material = &world_->defaultMaterial();
            materialFriction_.friction = settings_.tireFriction;
            for (PxU32 i = 0; i < 4; ++i) {
                perWheelMaterialFriction_[i].materialFrictions = &materialFriction_;
                perWheelMaterialFriction_[i].nbMaterialFrictions = 1;
                perWheelMaterialFriction_[i].defaultFriction = settings_.tireFriction;
            }
        }

        void buildCommandResponseParams() {
            using namespace ::physx;

            std::memset(&throttleResponseParams_, 0, sizeof(throttleResponseParams_));
            throttleResponseParams_.maxResponse = settings_.maxThrottleTorque;
            for (PxU32 i = 0; i < 4; ++i) {
                throttleResponseParams_.wheelResponseMultipliers[i] = settings_.drivenWheels[i] ? 1.f : 0.f;
            }

            std::memset(&steerResponseParams_, 0, sizeof(steerResponseParams_));
            steerResponseParams_.maxResponse = settings_.maxSteerAngleRad;
            // Front wheels (ids 0, 1) steer; rear wheels do not.
            steerResponseParams_.wheelResponseMultipliers[0] = 1.f;
            steerResponseParams_.wheelResponseMultipliers[1] = 1.f;
            steerResponseParams_.wheelResponseMultipliers[2] = 0.f;
            steerResponseParams_.wheelResponseMultipliers[3] = 0.f;

            std::memset(&brakeResponseParams_, 0, sizeof(brakeResponseParams_));
            brakeResponseParams_.maxResponse = settings_.maxBrakeTorque;
            for (PxU32 i = 0; i < 4; ++i) {
                brakeResponseParams_.wheelResponseMultipliers[i] = 1.f;
            }

            commands_.brakes[0] = 0.f;
            commands_.brakes[1] = 0.f;
            commands_.nbBrakes = 1;
            commands_.throttle = 0.f;
            commands_.steer = 0.f;

            transmissionCommands_.gear = ::physx::vehicle2::PxVehicleDirectDriveTransmissionCommandState::eFORWARD;

            physxSteerState_.setToDefault();
            physxConstraints_.setToDefault();
            // ePREFILTER required so queryFilter_.preFilter() is actually invoked;
            // without it the wheel rays hit the chassis box from inside (suspension
            // attachment is below chassis center) and the vehicle "rests" on itself.
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
            simContext_.physxUnitCylinderSweepMesh = nullptr;// raycast mode -> not needed
            simContext_.physxActorUpdateMode = ::physx::vehicle2::PxVehiclePhysXActorUpdateMode::eAPPLY_VELOCITY;

            // Override the stock sticky-tire defaults — see Settings docs above.
            auto& sticky = simContext_.tireStickyParams;
            sticky.stickyParams[PxVehicleTireDirectionModes::eLONGITUDINAL].thresholdSpeed = settings_.stickySpeedThreshold;
            sticky.stickyParams[PxVehicleTireDirectionModes::eLONGITUDINAL].thresholdTime = settings_.stickyTimeThreshold;
            sticky.stickyParams[PxVehicleTireDirectionModes::eLONGITUDINAL].damping = settings_.stickyDampingLongitudinal;
            sticky.stickyParams[PxVehicleTireDirectionModes::eLATERAL].thresholdSpeed = settings_.stickySpeedThreshold;
            sticky.stickyParams[PxVehicleTireDirectionModes::eLATERAL].thresholdTime = settings_.stickyTimeThreshold;
            sticky.stickyParams[PxVehicleTireDirectionModes::eLATERAL].damping = settings_.stickyDampingLateral;
        }

        void createPhysxActor() {
            using namespace ::physx;
            auto& physics = world_->physics();

            chassisActor_ = physics.createRigidDynamic(rigidBodyState_.pose);
            chassisActor_->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
            chassisActor_->setRigidBodyFlag(PxRigidBodyFlag::eENABLE_CCD, false);
            chassisActor_->setLinearDamping(settings_.chassisLinearDamping);
            chassisActor_->setAngularDamping(settings_.chassisAngularDamping);

            // Chassis box.
            const PxBoxGeometry chassisGeom(
                    settings_.chassisWidth * 0.5f,
                    settings_.chassisHeight * 0.5f,
                    settings_.chassisLength * 0.5f);
            PxShape* chassisShape = physics.createShape(chassisGeom, world_->defaultMaterial(), true);
            chassisActor_->attachShape(*chassisShape);
            chassisShape->release();

            // Per-wheel sphere shapes. eSCENE_QUERY_SHAPE off so wheel raycasts ignore them
            // (the prefilter callback also rejects the chassis actor as a belt-and-suspenders).
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
            // Sticky-tire + suspension-limit constraints. Without these, low-speed brake
            // commands compute a tire sticky state but it is never applied to PhysX, so
            // the car keeps slip-creeping under full brake. Must run after the chassis
            // actor has been added to the scene (helper registers PxConstraints with it).
            ::physx::vehicle2::PxVehicleConstraintsCreate(
                    axleDesc_, world_->physics(), *chassisActor_, physxConstraints_);
            constraintsCreated_ = true;
        }

        void buildComponentSequence() {
            using namespace ::physx::vehicle2;
            componentSequence_.add(static_cast<PxVehiclePhysXActorBeginComponent*>(this));
            componentSequence_.add(static_cast<PxVehiclePhysXRoadGeometrySceneQueryComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleSuspensionComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleTireComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleDirectDriveCommandResponseComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleDirectDriveActuationStateComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleDirectDrivetrainComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleWheelComponent*>(this));
            componentSequence_.add(static_cast<PxVehicleRigidBodyComponent*>(this));
            // Convert sticky-tire + suspension-limit states into PxConstraint impulses.
            // Goes after rigid-body update so it sees the up-to-date pose, before
            // ActorEnd which writes velocities back onto the PxRigidBody.
            componentSequence_.add(static_cast<PxVehiclePhysXConstraintComponent*>(this));
            componentSequence_.add(static_cast<PxVehiclePhysXActorEndComponent*>(this));
        }

        void stepVehicle(float dt) {
            componentSequence_.update(dt, simContext_);
            // PxVehicle's component sequence updates wheelLocalPoses_ but doesn't
            // push them onto the chassis-attached wheel shapes — debug visualization
            // would otherwise stay frozen at the initial rest pose. Sync explicitly.
            for (::physx::PxU32 i = 0; i < 4; ++i) {
                if (auto* shape = physxActor_.wheelShapes[i]) {
                    shape->setLocalPose(wheelLocalPoses_[i].localPose);
                }
            }
        }

        // ---- Component getDataFor* overrides ----

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
            gearState = nullptr;
            throttle = &commands_.throttle;
            physxActor = &physxActor_;
        }

        // ---- Internal types ----

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

        // ---- State ----

        PhysxWorld* world_ = nullptr;
        Settings settings_;
        bool initialised_ = false;

        std::function<void(float)> stepCallback_;

        ::physx::PxRigidDynamic* chassisActor_ = nullptr;
        ::physx::PxMaterial* wheelShapeMaterial_ = nullptr;

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
        std::array<::physx::PxReal, 4> throttleResponseStates_{};
        std::array<::physx::PxReal, 4> steerResponseStates_{};

        ::physx::vehicle2::PxVehicleDirectDriveThrottleCommandResponseParams throttleResponseParams_{};
        ::physx::vehicle2::PxVehicleSteerCommandResponseParams steerResponseParams_{};
        ::physx::vehicle2::PxVehicleBrakeCommandResponseParams brakeResponseParams_{};

        ::physx::vehicle2::PxVehicleCommandState commands_{};
        ::physx::vehicle2::PxVehicleDirectDriveTransmissionCommandState transmissionCommands_{};

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
    };

}// namespace threepp

#endif//THREEPP_PHYSX_VEHICLE_HPP
