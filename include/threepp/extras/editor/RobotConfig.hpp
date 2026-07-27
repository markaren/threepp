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

#include <optional>
#include <string>
#include <vector>

namespace threepp {

    class Object3D;

}

namespace threepp::editor {

    struct RobotConfig {

        // Source file, stored verbatim.
        std::string urdf;
        // One value per articulated DOF, always in radians/metres — the joint's
        // native unit — so the file does not depend on the inspector's choice
        // of degrees for display.
        std::vector<float> joints;

        static constexpr const char* urdfKey = "urdf";
        static constexpr const char* jointsKey = "jointValues";

        [[nodiscard]] static std::string encodeJoints(const std::vector<float>& values);
        [[nodiscard]] static std::vector<float> decodeJoints(const std::string& text);

        // nullopt when the object carries no urdf reference.
        [[nodiscard]] static std::optional<RobotConfig> read(const Object3D& object);

        void write(Object3D& object) const;

        static void erase(Object3D& object);
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_ROBOTCONFIG_HPP
