// What the editor needs to rebuild an articulated robot from a saved scene.
//
// A Robot loaded from URDF is a live object: it owns a joint table, the original
// pose of every joint node, and the axis/limit data needed to drive them. None
// of that survives the three.js JSON, which knows only about transforms — a
// saved robot comes back as a plain Object3D subtree, correctly posed but
// frozen.
//
// So the scene stores a reference instead: the URDF path it came from, plus the
// current joint values. On load the editor re-imports the file and transplants
// the live Robot over the frozen placeholder, keeping its uuid and placement.
// The geometry is still written to the document, so a scene without its URDF
// (or opened by another tool) renders exactly as it was saved — it just cannot
// be re-jointed.
//
// Unlike PhysicsConfig and AnimationConfig this does NOT pack into one
// key=value string: a Windows path contains the characters that format uses as
// delimiters. Two plain userData entries avoid the escaping problem entirely.

#ifndef THREEPP_EDITOR_ROBOTCONFIG_HPP
#define THREEPP_EDITOR_ROBOTCONFIG_HPP

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace threepp {

    class Object3D;
    class Robot;

}

namespace threepp::editor {

    struct RobotConfig {

        // Source file, stored verbatim.
        std::string urdf;
        // One value per articulated DOF, always in radians/metres — the joint's
        // native unit — so the file does not depend on the inspector's choice
        // of degrees for display.
        std::vector<float> joints;
        // URDF ships collision geometry alongside the visual meshes, and
        // URDFLoader builds both. The collision hulls are wireframe stand-ins
        // that sit right on top of the real meshes, so they start hidden and
        // this is the per-robot opt-in. Persisted because a play/stop cycle
        // rebuilds the robot, and an inspection aid that vanishes when you
        // press play is worse than useless.
        bool showColliders = false;

        static constexpr const char* urdfKey = "urdf";
        static constexpr const char* jointsKey = "jointValues";
        static constexpr const char* collidersKey = "showColliders";

        [[nodiscard]] static std::string encodeJoints(const std::vector<float>& values);
        [[nodiscard]] static std::vector<float> decodeJoints(const std::string& text);

        // nullopt when the object carries no urdf reference.
        [[nodiscard]] static std::optional<RobotConfig> read(const Object3D& object);

        void write(Object3D& object) const;

        static void erase(Object3D& object);
    };

    // Transplant a freshly re-imported Robot over the frozen placeholder a
    // document round trip left behind, in place: `robot` takes the placeholder's
    // uuid, name, transform, visibility and userData, is posed from the
    // placeholder's RobotConfig, and is swapped into the placeholder's parent at
    // the same child index. The placeholder is detached.
    //
    // Descendant userData is preserved too — a sensor or a physics entry authored
    // on a LINK or on one of its meshes, not on the root, would otherwise be
    // silently dropped, because only the root's userData is part of RobotConfig.
    // Each placeholder descendant's non-empty userData is re-applied to the node
    // at the SAME POSITION in the fresh robot's pre-order walk, provided the two
    // still agree on the name; a name that disagrees falls back to a lookup by
    // name (first match wins). Position rather than name because a URDF's visual
    // and collision groups and their meshes are unnamed, and those are the nodes a
    // viewport click selects. What cannot be placed is reported through `log`.
    //
    // This is the headless core of EditorApp::rearticulateRobots, factored out so
    // it can be tested without the app; the app calls it inside its traversal.
    void transplantRobot(Object3D& placeholder, const std::shared_ptr<Robot>& robot,
                         const std::function<void(const std::string&)>& log = {});

}// namespace threepp::editor

#endif//THREEPP_EDITOR_ROBOTCONFIG_HPP
