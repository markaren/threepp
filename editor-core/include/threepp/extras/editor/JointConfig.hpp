// A joint authored as its own scene node, stored on that node itself.
//
// The node's TRANSFORM is the joint frame: origin at the anchor, local X along
// the hinge/slide axis — the same convention Articulation::addLink and the
// runtime Joint wrapper use. That is what makes the existing transform gizmo
// the anchor/axis editor; no bespoke handles. The node's PARENT is body A (the
// nearest ancestor with a rigid body governs it, exactly like a sensor), and
// the OTHER body is referenced by name — the same by-name convention
// SensorConfig::joint and userData["editorFollow"] follow. An empty name means
// "the world": a pendulum pivot, a door frame bolted to nothing.
//
// Encoding follows the PhysicsConfig/SensorConfig template: one flat
// `key=value;` string under userData["joint"], every key written on every
// write, unknown keys ignored on read. The body NAME is the one field a user
// types freely — it may contain the flat format's `;`/`=` delimiters — so it
// rides a plain userData key of its own (userData["jointBody"]), the same
// escape hatch RobotConfig's path and SoundConfig's file use.
//
// The PRESENCE of the entry is what makes a plain Object3D a joint at all
// (SoundConfig's rule), so write() always writes, defaults included; removing
// a joint means deleting the node.
//
// Angular values are stored in RADIANS and linear ones in metres — native
// units, like RobotConfig's jointValues; only the inspector's widgets speak
// degrees. Nothing here depends on PhysX: the struct is authoring data, and
// turning it into a live constraint is PhysicsPlaySession's job.

#ifndef THREEPP_EDITOR_JOINTCONFIG_HPP
#define THREEPP_EDITOR_JOINTCONFIG_HPP

#include <optional>
#include <string>

namespace threepp {

    class Object3D;

}

namespace threepp::editor {

    struct JointConfig {

        enum class Type {
            Fixed,    // welds body A to body B
            Revolute, // hinge about the node's local X
            Prismatic,// slider along the node's local X
            Spherical,// ball socket, optional swing cone
            Distance  // tether: keeps the anchors within [lower, upper] metres
        };

        // Revolute is the default: the joint someone reaches for first (a
        // door, a wheel, a pendulum), and the one whose axis the node's
        // transform most obviously depicts.
        Type type = Type::Revolute;

        // The OTHER body, by scene-object name; resolved against the played
        // scene like every persisted reference (see header note). Empty means
        // the world. Not part of encode()/decode() — it lives on bodyKey.
        std::string body;

        // Limits, in the node's own units: radians about X for Revolute,
        // metres along X for Prismatic. For Spherical, `limited` enables the
        // swing cone (coneY/coneZ half-angles, radians; twist stays free).
        // Distance ignores `limited`: lower/upper ARE the constraint (min/max
        // metres), because an unlimited tether would constrain nothing.
        // Defaults give a ±45° hinge the moment Limited is ticked.
        bool limited = false;
        float lower = -0.785398f;
        float upper = 0.785398f;
        float coneY = 0.785398f;
        float coneZ = 0.785398f;

        // PD drive toward `target` (radians/metres), plus a feed-forward
        // `velocity`. Zero stiffness AND damping = no drive, which is the
        // default: an authored joint is passive until it is told otherwise.
        // Each half gates its own input — `target` acts through stiffness and
        // `velocity` through damping (the drive force is stiffness·(target−x)
        // + damping·(velocity−v)) — which is why the inspector seeds BOTH when
        // its Driven box is ticked. Force mode, so the numbers mean the same
        // as ArticulationConfig's. For Distance the stiffness/damping pair is
        // the tether's spring.
        float stiffness = 0.f;
        float damping = 0.f;
        float maxForce = 1e6f;// effort ceiling for the drive
        float target = 0.f;
        float velocity = 0.f;

        // The joint breaks apart for good past this solver force/torque.
        // Zero = unbreakable, the default.
        float breakForce = 0.f;
        float breakTorque = 0.f;

        // Whether the two jointed bodies still collide with each other. Off
        // by default: bodies meeting at a joint overlap at the anchor, and
        // contacts there fight the constraint.
        bool collide = false;

        static constexpr const char* userDataKey = "joint";
        // The other body's name, on its own plain key (see the header note).
        static constexpr const char* bodyKey = "jointBody";

        [[nodiscard]] std::string encode() const;
        [[nodiscard]] static std::optional<JointConfig> decode(const std::string& text);

        // nullopt when the object carries no joint entry (or an unreadable
        // one). A present entry is what makes the node a joint, so `body` is
        // filled in from bodyKey on the way out.
        [[nodiscard]] static std::optional<JointConfig> read(const Object3D& object);

        // Whether this node is an authored joint at all — the predicate the
        // inspector section, the viewport marker and the play session share.
        [[nodiscard]] static bool isJoint(const Object3D& object);

        // Writes both entries, defaults included (presence is the node's
        // identity — see the header note). An empty body erases bodyKey, so
        // "jointed to the world" leaves the smaller document.
        void write(Object3D& object) const;

        // Both entries, so removing the authoring is one call.
        static void erase(Object3D& object);

        static const char* label(Type type);

        static constexpr Type types[] = {
                Type::Fixed, Type::Revolute, Type::Prismatic,
                Type::Spherical, Type::Distance};

        bool operator==(const JointConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_JOINTCONFIG_HPP
