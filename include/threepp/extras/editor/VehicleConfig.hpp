// A drivable vehicle authored ON an imported model's root node.
//
// The authoring is deliberately small: point at the four wheel meshes, press
// Play, drive. Everything geometric — chassis dimensions, wheel radius, track,
// wheelbase, where the suspension attaches — is DERIVED from the four picked
// wheels while the `auto` flag is on, so an imported car drives on its first
// Play without a single number typed. Overriding any geometry field flips the
// flag and the inspector seeds every geometry field with the derived values,
// so nothing jumps.
//
// Encoding follows the PhysicsConfig/JointConfig template: one flat
// `key=value;` string under userData["vehicle"], every key written on every
// write, unknown keys ignored on read. The four wheel NAMES are user-typed and
// free to contain the flat format's `;`/`=` delimiters, so each rides a plain
// userData key of its own (vehicleWheelFR/FL/RR/RL) — the same escape hatch
// JointConfig::bodyKey and the sound file use.
//
// The PRESENCE of the entry is what makes the node a vehicle (SoundConfig's
// rule), so write() always writes, defaults included; unticking "Simulate as
// Vehicle" erases all five keys.
//
// Wheel order everywhere is the runtime's: 0 = front-right, 1 = front-left,
// 2 = rear-right, 3 = rear-left. Four wheels only — the runtime is a
// std::array<4>; trucks and trailers are out of scope (a trailer is a second
// vehicle plus an authored joint). Angles are stored in RADIANS, lengths in
// metres; only the inspector's widgets speak degrees. Nothing here depends on
// PhysX: the struct is authoring data, and turning it into a live
// PhysxVehicle is PhysicsPlaySession's job.

#ifndef THREEPP_EDITOR_VEHICLECONFIG_HPP
#define THREEPP_EDITOR_VEHICLECONFIG_HPP

#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"

#include <array>
#include <optional>
#include <string>

namespace threepp {

    class Object3D;

}

namespace threepp::editor {

    // What Play needs beyond the stored scalars: the wheels resolved to nodes,
    // the chassis frame the vehicle spawns in, and the geometry read off the
    // model. Derived fresh at every use — the model is the source of truth.
    struct VehicleGeometry {

        bool valid = false;
        // Why not, when !valid — one console line's worth.
        std::string problem;

        // Resolved wheel nodes, FR/FL/RR/RL. All non-null when valid.
        std::array<Object3D*, 4> wheels{};

        // The chassis frame, world space: origin at the derived chassis
        // centre, +Z forward (rear axle towards front axle), +X right, +Y up.
        // This is where the chassis actor spawns, whatever the auto flag says
        // — the frame is not authorable, the wheels ARE it.
        Vector3 position;
        Quaternion rotation;

        // Derived geometry, metres, meaningful when valid.
        float chassisWidth = 0.f;
        float chassisHeight = 0.f;
        float chassisLength = 0.f;
        float wheelRadius = 0.f;
        float wheelWidth = 0.f;
        float trackWidth = 0.f;
        float wheelbase = 0.f;
        // Suspension attachment height (chassis frame). Includes the static
        // settle compensation, so the car rests at its authored ride height
        // rather than sagging on the first Play — see derived().
        float suspensionY = 0.f;

        // Actual per-wheel hub centres in the chassis frame. The runtime
        // places wheels symmetrically; the difference to these is what the
        // play session's per-wheel mirror offsets absorb, so an asymmetric
        // model still renders exactly where it was authored.
        std::array<Vector3, 4> hubs{};
    };

    struct VehicleConfig {

        enum class Drive {
            Direct,// throttle torque straight to the wheels — no gearbox
            Engine // engine/clutch/gearbox chain, automatic all the way
        };

        enum class Driven {
            All, // AWD: the forgiving default — grip before wheelspin
            Rear,// RWD
            Front// FWD
        };

        Drive drive = Drive::Direct;
        Driven driven = Driven::All;

        // Geometry is read off the four picked wheels while this is on; the
        // stored fields below only act once a field is overridden (which
        // flips it, seeding all of them from the derived values).
        bool autoGeometry = true;
        float chassisWidth = 1.8f;
        float chassisHeight = 1.f;
        float chassisLength = 4.5f;
        float wheelRadius = 0.4f;
        float wheelWidth = 0.3f;// full width; the runtime wants the half
        float trackWidth = 1.6f;
        float wheelbase = 2.8f;
        float suspensionY = -0.4f;// attachment height, chassis frame

        // Always authored.
        float mass = 1500.f;
        float suspensionTravel = 0.3f;
        float suspensionStiffness = 35'000.f;
        float suspensionDamping = 4500.f;
        // The Vehicle demo's proven street tuning, not the base's 1.5: direct
        // drive has no gearbox multiplying torque away, so grip is what keeps
        // the first Play from being a burnout.
        float tireFriction = 2.f;
        float maxBrakeTorque = 5000.f;
        float maxSteerAngle = 0.6f;    // radians; ~34 degrees
        float throttleTorque = 1500.f; // N*m at full throttle, Direct only

        // The four wheel references, FR/FL/RR/RL: a descendant node's name,
        // or "name#N" for the N-th node of that name in document order — how
        // duplicates are told apart (imported assets repeat names freely).
        // The node may be a Group: a wheel authored as an assembly (rim +
        // tire + caliper) measures as its whole subtree, which is usually
        // what is right. Not part of encode()/decode() — each lives on its
        // own wheelKeys entry.
        std::array<std::string, 4> wheels{};

        static constexpr const char* userDataKey = "vehicle";
        static constexpr const char* wheelKeys[4] = {
                "vehicleWheelFR", "vehicleWheelFL",
                "vehicleWheelRR", "vehicleWheelRL"};
        static constexpr const char* wheelLabels[4] = {
                "Front Right", "Front Left", "Rear Right", "Rear Left"};

        [[nodiscard]] std::string encode() const;
        [[nodiscard]] static std::optional<VehicleConfig> decode(const std::string& text);

        // nullopt when the object carries no vehicle entry (or an unreadable
        // one). A present entry is what makes the node a vehicle, so the
        // wheel names are filled in from their keys on the way out.
        [[nodiscard]] static std::optional<VehicleConfig> read(const Object3D& object);

        // Whether this node is an authored vehicle at all — the predicate the
        // inspector section, the wheel-ring helper and the play session share.
        [[nodiscard]] static bool isVehicle(const Object3D& object);

        // Writes the flat entry (defaults included — presence is the node's
        // identity) plus the four wheel keys; an empty wheel name erases its
        // key, so a half-picked vehicle leaves the smaller document.
        void write(Object3D& object) const;

        // All five entries, so removing the authoring is one call.
        static void erase(Object3D& object);

        // Resolve this config's wheels against `root`'s descendants and read
        // the geometry off them. Non-const root: measuring needs the world
        // matrices up to date. Always derives — the caller picks derived vs
        // authored numbers by the auto flag; the frame and the hubs are
        // needed either way. `problem` says what is missing when !valid.
        [[nodiscard]] VehicleGeometry derived(Object3D& root) const;

        static const char* label(Drive drive);
        static const char* label(Driven driven);

        static constexpr Drive drives[] = {Drive::Direct, Drive::Engine};
        static constexpr Driven drivens[] = {Driven::All, Driven::Rear, Driven::Front};

        bool operator==(const VehicleConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_VEHICLECONFIG_HPP
