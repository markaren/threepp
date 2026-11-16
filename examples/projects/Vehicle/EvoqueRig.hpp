#ifndef THREEPP_EVOQUE_RIG_HPP
#define THREEPP_EVOQUE_RIG_HPP

// The Range Rover Evoque as a PhysxVehicle: chassis and tyre settings sized to
// the model, and the visual rig (body, four wheels driven from the PhysX wheel
// poses, brake and reverse lamps). Shared by physx_vehicle and
// vehicle_playground.
//
// Model: "2015 Land-Rover Range Rover Evoque Coupe" by Ddiaz Design,
// https://sketchfab.com/3d-models/2015-land-rover-range-rover-evoque-coupe-23ef38b95ebb4d6fb679248817e8b211,
// CC-BY-NC-SA-4.0. kCredit is the attribution line to show where the demo is shared.

#include "threepp/threepp.hpp"

#include "threepp/extras/physx/PhysxVehicle.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/loaders/ModelLoader.hpp"

#include <array>
#include <string>
#include <vector>

namespace threepp::evoque {

    inline constexpr const char* kModelPath = "/models/gltf/2015_land-rover_range_rover_evoque_coupe/scene.gltf";

    inline constexpr const char* kCredit =
            "Car: \"2015 Land-Rover Range Rover Evoque Coupe\" by Ddiaz Design (Sketchfab), CC BY-NC-SA 4.0";

    // Chassis and tyre settings matching the model; the caller sets the spawn pose.
    inline PhysxVehicle::Settings vehicleSettings() {
        PhysxVehicle::Settings settings;
        // The model is ~2.1 x 1.6 x 4.4 m.
        settings.chassisWidth = 1.95f;
        settings.chassisHeight = 1.4f;
        settings.chassisLength = 4.4f;
        settings.wheelbase = 2.66f;
        settings.trackWidth = 1.65f;
        // 4WD splits drive torque across all four wheels: twice the grip
        // envelope before the tyres break loose at low speed. The Evoque is AWD.
        settings.drivenWheels = {true, true, true, true};
        // Direct drive (no gearbox), so the torque has to do all the work.
        settings.maxThrottleTorque = 1500.f;
        settings.wheelDampingRate = 1.5f;
        // Stickier tyres and more longitudinal stiffness, so the contact patch
        // transmits force faster than the engine can spin the wheel.
        settings.tireFriction = 2.0f;
        settings.longitudinalStiffness = 100'000.f;
        return settings;
    }

    // The visual car. `chassis` is bound to the vehicle's chassis actor; the body
    // and the wheel rigs hang under it.
    class Rig {
    public:
        std::shared_ptr<Group> chassis = Group::create();
        std::shared_ptr<Group> body;
        std::array<std::shared_ptr<Group>, 4> wheels;

        // dataFolder is the asset root (DATA_FOLDER in the examples).
        Rig(PhysxWorld& world, PhysxVehicle& vehicle, const std::string& dataFolder) {
            world.bind(*chassis, *vehicle.chassisActor());

            ModelLoader modelLoader;
            body = modelLoader.load(dataFolder + kModelPath);
            // The model's native units are roughly mm. Its wheel contacts sit at
            // y=0 in model space; the offset lines them up with the rest height
            // of the PhysX chassis.
            body->scale.set(100.f, 100.f, 100.f);
            body->position.y = -1.f;
            chassis->add(body);

            buildWheels();
            buildLamps();
        }

        // Wheel rigs from the PhysX wheel poses (chassis space), lamps from the
        // commands. Call once per frame after the world has stepped.
        void update(const PhysxVehicle& vehicle, bool braking) {
            for (int i = 0; i < 4; ++i) {
                const auto wp = vehicle.wheelLocalPose(i);
                wheels[i]->position.set(wp.p.x, wp.p.y, wp.p.z);
                wheels[i]->quaternion.set(wp.q.x, wp.q.y, wp.q.z, wp.q.w);
            }
            // needsUpdate() re-uploads the material on some backends, so only
            // when a lamp changes state.
            setLamp(brakeLightMat_, braking, brakeOn_);
            setLamp(reverseLightMat_, vehicle.gear() == PhysxVehicle::Gear::Reverse, reverseOn_);
        }

    private:
        std::shared_ptr<MeshStandardMaterial> brakeLightMat_, reverseLightMat_;
        bool brakeOn_ = false, reverseOn_ = false;

        // The .gltf's per-wheel "pivot" nodes have identity transforms (the wheel
        // positions are baked into the geometry), so the hub comes from the
        // combined bounding box and each wheel mesh is cloned into a rig whose
        // origin is the hub. PhysX wheel index -> model tag.
        void buildWheels() {
            const char* wheelTag[4] = {"WheelFL", "WheelFR", "WheelBL", "WheelBR"};
            std::array<std::vector<Mesh*>, 4> wheelParts;
            body->traverseType<Mesh>([&](Mesh& m) {
                for (int i = 0; i < 4; ++i) {
                    if (m.name.find(wheelTag[i]) != std::string::npos) {
                        wheelParts[i].push_back(&m);
                        return;
                    }
                }
            });

            for (int i = 0; i < 4; ++i) {
                wheels[i] = Group::create();
                chassis->add(wheels[i]);

                if (wheelParts[i].empty()) continue;

                Box3 combined;
                combined.makeEmpty();
                for (auto* part : wheelParts[i]) {
                    part->geometry()->computeBoundingBox();
                    const auto& b = *part->geometry()->boundingBox;
                    combined.expandByPoint(b.min());
                    combined.expandByPoint(b.max());
                }
                const Vector3 hub = combined.getCenter();

                for (auto* part : wheelParts[i]) {
                    // Clone before hiding the original: Object3D::copy inherits `visible`.
                    auto cloned = std::dynamic_pointer_cast<Mesh>(part->clone());
                    part->visible = false;
                    if (!cloned) continue;
                    cloned->visible = true;
                    // A vertex v ends up at 100*(v - hub) in rig space: the hub at
                    // the rig origin, the geometry in meters.
                    cloned->position.set(-hub.x, -hub.y, -hub.z);
                    wheels[i]->add(cloned);
                }
            }
        }

        // The model's "lights_position_back" (tail/brake) and "lights_reverse"
        // groups share one emissive "glows" material; each group gets its own
        // copy so the two can light independently. emissiveIntensity 1 is the
        // model's stock glow.
        void buildLamps() {
            body->traverseType<Mesh>([&](Mesh& m) {
                auto retag = [&m](std::shared_ptr<MeshStandardMaterial>& slot) {
                    if (!slot) {
                        if (auto* orig = m.materialAs<MeshStandardMaterial>()) {
                            slot = std::dynamic_pointer_cast<MeshStandardMaterial>(orig->clone());
                        }
                    }
                    if (slot) m.setMaterial(slot);
                };
                if (m.name.find("lights_position_back") != std::string::npos) {
                    retag(brakeLightMat_);
                } else if (m.name.find("lights_reverse") != std::string::npos) {
                    retag(reverseLightMat_);
                }
            });
        }

        static void setLamp(const std::shared_ptr<MeshStandardMaterial>& mat, bool on, bool& wasOn) {
            if (!mat || on == wasOn) return;
            wasOn = on;
            mat->emissiveIntensity = on ? 5.f : 1.f;
            mat->needsUpdate();
        }
    };

}// namespace threepp::evoque

#endif//THREEPP_EVOQUE_RIG_HPP
