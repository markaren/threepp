// Whether — and how — a URDF robot is SIMULATED during Play, stored on the robot.
//
// A RobotConfig already records the URDF a robot came from and its authored
// joint pose; that is enough to render it and drive its joints by hand. This is
// the separate opt-in that turns the same robot into a PhysX reduced-coordinate
// articulation while playing: the joints become real DOFs, gravity and contact
// act on the links, and the joint sensors (Encoder, Force/Torque) have something
// to measure. Stop restores the pre-play snapshot, so nothing the simulation did
// survives — exactly like the rigid bodies PhysicsConfig authors.
//
// WHY THIS IS SEPARATE FROM RobotConfig. RobotConfig's fields are a filesystem
// PATH and a vector of joint values — neither rides the flat `key=value;` format
// (a Windows path carries the delimiters, and a vector has no fixed width), so it
// uses two plain userData entries instead. This config is all SCALARS, so it
// follows the PhysicsConfig/SensorConfig template exactly: one flat, deterministic
// `key=value;` string in `userData["articulation"]`, every key written on every
// write, unknown keys ignored on read, and a disabled entry removed entirely.
// The PRESENCE of the entry is the signal — a robot with an articulation entry is
// simulated, one without it is a kinematic prop the physics session ignores.
//
// Defaults lean toward the common editor case: a fixed-base arm that should hold
// the pose you authored. So fixedBase is true and the drive is stiff enough to
// hold a link against gravity out of the box; a free-flying robot (a quadruped,
// a drone) is the opt-out, and a passive/force-controlled robot sets stiffness
// to zero.
//
// Nothing here depends on PhysX. The struct is authoring data; turning it into a
// live articulation is PhysicsPlaySession's job, and any other runtime is free to
// read the same field and build something else from it.

#ifndef THREEPP_EDITOR_ARTICULATIONCONFIG_HPP
#define THREEPP_EDITOR_ARTICULATIONCONFIG_HPP

#include <optional>
#include <string>

namespace threepp {

    class Object3D;

}

namespace threepp::editor {

    struct ArticulationConfig {

        // Presence of the userData entry is what "simulate this robot" means, so
        // read() returns nullopt when there is no entry and a config with
        // enabled == true whenever there is one. write() with enabled == false
        // erases the entry, matching PhysicsConfig/SensorConfig.
        bool enabled = false;

        // Pin the root link to the world. True is the arm case — the base is
        // bolted to a table and only the joints move. A quadruped or a drone
        // wants a free-floating base, so set this false.
        bool fixedBase = true;

        // The per-joint PD drive that holds the authored pose. Stiff enough by
        // default that a horizontal arm link does not sag under gravity the
        // instant Play starts; a passive robot sets stiffness to 0 (the joints go
        // free and the drive target is not written).
        float stiffness = 500.f;
        float damping = 50.f;
        float maxForce = 1e6f;// per-joint effort ceiling for the drive

        // Whether links of the same robot collide with each other. Off by
        // default: URDF collision is approximated by primitives/bounding boxes
        // here, and adjacent links' boxes overlap at the joint, so self-collision
        // would fight the pose rather than protect it.
        bool selfCollision = false;

        // Solver position iterations. Higher holds a stiff pose better under load
        // at the cost of step time.
        int iterations = 12;

        // Fallback density (kg/m^3) for links whose URDF gives no <inertial><mass>.
        float density = 1000.f;

        static constexpr const char* userDataKey = "articulation";

        [[nodiscard]] std::string encode() const;
        [[nodiscard]] static std::optional<ArticulationConfig> decode(const std::string& text);

        // nullopt when the object carries no articulation entry (or an unreadable
        // one). A present entry means "simulate", so a returned config always has
        // enabled == true.
        [[nodiscard]] static std::optional<ArticulationConfig> read(const Object3D& object);

        // Writes the entry; `enabled == false` removes it, so a robot that is not
        // simulated leaves no trace in the saved file.
        void write(Object3D& object) const;

        static void erase(Object3D& object);

        bool operator==(const ArticulationConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_ARTICULATIONCONFIG_HPP
