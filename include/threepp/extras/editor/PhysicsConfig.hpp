// Per-object physics authoring, stored on the object itself.
//
// The editor writes this into `object.userData["physics"]`, which means it
// travels with the scene through ObjectExporter/ObjectLoader for free — a saved
// scene carries its rigid-body setup with no sidecar file and no schema
// extension.
//
// Encoding: a single `key=value;key=value` string. userData round-trips scalars
// only (bool / int / float / double / string — see ObjectExporter::writeUserData),
// so a nested object is not an option; one flat, deterministic, human-readable
// string is. Unknown keys are ignored on read, which is what makes the format
// extensible.
//
// Nothing here depends on PhysX. The struct is just authoring data; turning it
// into actors is PhysicsPlaySession's job, and any other runtime is free to read
// the same field and do something else with it.

#ifndef THREEPP_EDITOR_PHYSICSCONFIG_HPP
#define THREEPP_EDITOR_PHYSICSCONFIG_HPP

#include <optional>
#include <string>

namespace threepp {

    class Object3D;

}

namespace threepp::editor {

    struct PhysicsConfig {

        enum class Body {
            Static,   // never moves; collides with everything
            Dynamic,  // simulated
            Kinematic,// moved by code, pushes dynamics, ignores forces
            Soft      // deformable volume; the mesh itself bends and squashes
        };

        enum class Shape {
            Auto,   // Box/Sphere/Capsule when the geometry says so, else Convex/TriMesh
            Box,
            Sphere,
            Capsule,
            Convex,
            TriMesh // static/kinematic only — PhysX dynamics need a convex shape
        };

        bool enabled = false;
        Body body = Body::Dynamic;
        Shape shape = Shape::Auto;
        float mass = 1.f;       // kg, dynamic and soft bodies
        float friction = 0.5f;  // static == dynamic friction
        float restitution = 0.2f;// rigid bodies only

        // --- Body::Soft only. The Shape field is ignored for a soft body: its
        // collider is always a tetrahedral volume cooked from the mesh itself.

        float youngsModulus = 1e6f; // Pa. Stiffness: ~1e4 jelly, 1e6 rubber, 1e8 hard.
        float poissonsRatio = 0.45f;// 0 squashes freely, ->0.5 preserves volume.
        int voxelResolution = 10;   // simulation-mesh cells along the longest axis
        int solverIterations = 20;  // per step; more = stiffer, slower
        bool selfCollision = false; // let folds of the body collide with each other

        static constexpr const char* userDataKey = "physics";

        [[nodiscard]] std::string encode() const;
        [[nodiscard]] static std::optional<PhysicsConfig> decode(const std::string& text);

        // nullopt when the object carries no physics entry (or an unreadable one).
        [[nodiscard]] static std::optional<PhysicsConfig> read(const Object3D& object);

        // Writes the entry; `enabled == false` removes it, so a disabled body
        // leaves no trace in the saved file.
        void write(Object3D& object) const;

        static void erase(Object3D& object);

        static const char* label(Body body);
        static const char* label(Shape shape);
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_PHYSICSCONFIG_HPP
