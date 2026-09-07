// The document's record of where a robot came from and how it stands.
//
// A Robot loaded from URDF is a live object: it owns a joint table, the original
// pose of every joint node, and the axis/limit data needed to drive them. The
// table itself now travels in the document (ObjectExporter's "threeppRobot"
// block), so a saved robot loads back live — this config is NOT what makes it
// articulate. It is what everything else needs:
//
//   - the URDF path, for the paths that genuinely must re-read the file — the
//     PhysX articulation (inertia and collision data are not in the scene
//     graph) and transplantRobot, which revives documents written before the
//     articulation block existed;
//   - the joint values, so the inspector and every rebuild agree on the pose
//     the document was saved in;
//   - the import-time xacro arguments, without which the file describes a
//     different robot (see below).
//
// Unlike PhysicsConfig and AnimationConfig this does NOT pack into one
// key=value string: a Windows path contains the characters that format uses as
// delimiters. Two plain userData entries avoid the escaping problem entirely.

#ifndef THREEPP_EDITOR_ROBOTCONFIG_HPP
#define THREEPP_EDITOR_ROBOTCONFIG_HPP

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
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
        // this is the per-robot opt-in. Persisted so the choice survives a
        // save/reload and a legacy re-articulation alike — an inspection aid
        // that resets itself is worse than useless.
        bool showColliders = false;
        // The xacro arguments the robot was imported with — ONLY the ones that
        // were explicitly set, never the file's own defaults. A description that
        // derives one default from another (UR's joint limits path is built from
        // ur_type) has to keep deriving it, and a captured default would freeze
        // the derivation at whatever it happened to be on the day of the import.
        // In the declaration order of the file, which is the order the import
        // dialog showed them in.
        //
        // Stored as two kinds of userData entry rather than one packed string,
        // for the reason at the top of this file: an argument value is very often
        // a path. The key names live in Xacro.hpp, because ObjectLoader has to
        // read the same entries when it re-imports a linked asset and it knows
        // nothing about the editor.
        std::vector<std::pair<std::string, std::string>> xacroArgs;

        static constexpr const char* urdfKey = "urdf";
        static constexpr const char* jointsKey = "jointValues";
        static constexpr const char* collidersKey = "showColliders";

        // The same arguments in the shape URDFLoader::setArgs wants. Every path
        // that rebuilds the robot goes through this, so "rebuilt with the
        // arguments it was imported with" is one call rather than a convention
        // each rebuild site has to remember.
        [[nodiscard]] std::map<std::string, std::string> argMap() const;

        [[nodiscard]] static std::string encodeJoints(const std::vector<float>& values);
        [[nodiscard]] static std::vector<float> decodeJoints(const std::string& text);

        // nullopt when the object carries no urdf reference.
        [[nodiscard]] static std::optional<RobotConfig> read(const Object3D& object);

        void write(Object3D& object) const;

        static void erase(Object3D& object);
    };

    // Make the frozen placeholder a document round trip left behind articulate
    // again, by moving `donor`'s JOINT TABLE onto the subtree the document
    // already carries. Returns the live Robot now standing in the placeholder's
    // place — same uuid, name, transform, layers and userData, same child index
    // under the same parent, posed from the placeholder's RobotConfig — or
    // nullptr if there was nothing to do.
    //
    // The donor is read, not planted. It is a fresh import of the same URDF, and
    // the only thing taken from it is what the document cannot express: each
    // joint's axis, type, limits and rest pose, plus which link is the end
    // effector. The NODES those apply to are looked up in the placeholder's own
    // subtree — links by name, joint nodes as the parent of their child link —
    // so everything authored into the robot survives: a camera bolted to the
    // wrist, a collider deleted, a material retouched, a sensor on an unnamed
    // mesh. Planting the donor instead, which is what this used to do, deleted
    // all of that on every Stop.
    //
    // Documents written since the articulation extension (ObjectExporter's
    // "threeppRobot" block) never come here: they load live and need no file at
    // all. This is the path for the ones written before it, and for a node that
    // carries nothing but a urdf reference.
    //
    // When the document's subtree has drifted so far from the file that not one
    // joint resolves — an empty placeholder, or a URDF rebuilt from scratch —
    // the donor's own subtree is planted after all and `log` says so, because a
    // robot that cannot be driven is worse than one that lost its annotations.
    //
    // This is the headless core of EditorApp::rearticulateRobots, factored out so
    // it can be tested without the app; the app calls it inside its traversal.
    std::shared_ptr<Robot> transplantRobot(Object3D& placeholder, const std::shared_ptr<Robot>& donor,
                                           const std::function<void(const std::string&)>& log = {});

}// namespace threepp::editor

#endif//THREEPP_EDITOR_ROBOTCONFIG_HPP
