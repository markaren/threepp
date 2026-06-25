
#ifndef THREEPP_URDFLOADER_HPP
#define THREEPP_URDFLOADER_HPP

#include "threepp/loaders/Loader.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Robot.hpp"

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace threepp {

    class Group;
    class Object3D;

    // Flat, articulation-ready description of a URDF — what a physics articulation builder needs that
    // the kinematic Robot doesn't carry: per-link COLLISION primitives and INERTIAL mass. Produced by
    // URDFLoader::parseArticulation, consumed by extras/physx/loadArticulation. Links are in root-first
    // topological order; `parent` indexes into `links` (-1 = root). Joint axis/origin are in the URDF's
    // native frame (the consumer applies forward kinematics to place each link in the world).
    struct URDFArticulationDesc {
        struct Collision {
            enum class Shape { None,
                               Box,
                               Sphere,
                               Capsule };
            Shape shape = Shape::None;
            Vector3 halfExtents;   // Box: half-size per axis
            float radius = 0.f;    // Sphere / Capsule
            float halfHeight = 0.f;// Capsule: half the cylinder length (caps added on top)
            Matrix4 origin;        // collision frame in the link frame (<collision><origin>)
        };
        struct Link {
            std::string name;
            std::string jointName;                        // name of the inbound joint ("" for the root)
            int parent = -1;                              // index into links; -1 = root
            Robot::JointType jointType = Robot::JointType::Fixed;
            Vector3 jointAxis{1.f, 0.f, 0.f};             // in the joint (child link) frame
            std::optional<Robot::JointRange> range;       // joint limits (none = free/continuous)
            Matrix4 jointOrigin;                          // parent-link frame -> this link frame (<joint><origin>)
            Collision collision;
            bool hasMass = false;
            float mass = 0.f;                             // from <inertial><mass>; else density fallback
            std::shared_ptr<Object3D> visual;             // link-frame visual subtree (may be null)
        };
        std::vector<Link> links;
    };

    class URDFLoader {

    public:
        explicit URDFLoader();

        // By default, the loader uses ModelLoader to load geometry files referenced in the URDF.
        URDFLoader& setGeometryLoader(std::shared_ptr<Loader<Group>> loader);

        // Set xacro argument overrides, equivalent to passing `name:=value` on the xacro CLI.
        // These override <xacro:arg default="..."/> values and are available as ${name} / $(arg name).
        URDFLoader& setArgs(std::map<std::string, std::string> args);

        std::shared_ptr<Robot> load(const std::filesystem::path& path);

        std::shared_ptr<Robot> parse(const std::filesystem::path& baseDir, const std::string& xml);

        // Parse a URDF (xacro supported, same as load) into a flat articulation description for the
        // physics articulation builder (extras/physx/loadArticulation). Returns links in root-first
        // order; the description is empty if the file can't be read or has no single root link.
        URDFArticulationDesc parseArticulation(const std::filesystem::path& path);

        ~URDFLoader();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_URDFLOADER_HPP
