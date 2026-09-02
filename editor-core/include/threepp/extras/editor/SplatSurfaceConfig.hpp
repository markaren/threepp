// Per-cloud surface-bake authoring: "this scan is a thing you can stand on".
//
// Stored on the SplatCloud node as one flat `key=value;…` string under
// userData["splatSurface"], exactly like PhysicsConfig — so the day splat
// clouds serialize, the bake setting travels with them for free.
//
// THE MESH IS NEVER STORED. splats::bakeSurface is deterministic (see its
// header: sequential fusion in pose-list order), so the config plus the cloud
// plus the node's world transform reproduce the same triangles bit for bit;
// writing a few hundred thousand of them into a document would be storing a
// pure function of data the document already has. The bake is memoized in
// memory instead (SplatSurfaceCache), keyed on exactly those inputs.
//
// The knobs are the three a user actually turns. Everything else in
// SurfaceBakeOptions — truncation, weight floor, fringe erode, outlier
// tolerance — was measured into its default by the P1 sweep
// (plans/splat-surface-bake.md) and is not authorable: a wrong value there
// produces a plausible-looking mesh that is quietly worse, which is the kind of
// knob a UI should not offer.

#ifndef THREEPP_EDITOR_SPLATSURFACECONFIG_HPP
#define THREEPP_EDITOR_SPLATSURFACECONFIG_HPP

#include <optional>
#include <string>

namespace threepp {

    class Object3D;

}

namespace threepp::editor {

    struct SplatSurfaceConfig {

        bool enabled = false;

        // Metres. 0 = derive from the cloud's own robust fit (radius / 256,
        // clamped to [5 mm, 10 cm]) — SurfaceBakeOptions::voxelSize's default,
        // which is the right answer for a scan whose scale nobody has measured.
        // It is also the accuracy the whole bake is quoted against: a collider
        // and a lidar return agree with the splats to within one voxel.
        float voxelSize = 0.f;

        // Islands smaller than this many surface CELLS are dropped — an area in
        // disguise (voxelSize^2 per cell), which is why it is authored next to
        // the voxel and not in metres. See SurfaceBakeOptions.
        int minComponentVoxels = 256;

        // Capture viewpoints. 0 = SurfaceBakeOptions' own 26. More poses cost
        // linearly (render AND fuse) and buy corroboration: the TSDF's weight
        // floor of 2 means every triangle was seen by two of them.
        int poseCount = 0;

        // Where those viewpoints stand: false orbits the scan from outside,
        // true stands INSIDE it and looks out
        // (SurfaceBakeOptions::PoseSet::Interior). A scan of the inside of a
        // room baked with the orbit reconstructs the outside of its walls — free
        // space carved outside, colliders on the outer skin, the walkable volume
        // never observed — which is a mistake nothing about the result announces
        // except that a robot cannot walk in it.
        bool interior = false;

        // How the surface is built. Auto takes the direct point route for a
        // cloud imported as a point cloud (SplatImportConfig::pointCloud) and
        // the depth-fusion bake for a Gaussian scan; the other two force one.
        // The point route needs no renderer at all (splats::buildPointSurface);
        // the fusion bake needs Vulkan (splats::bakeSurface).
        enum Method : int { Auto = 0, Fusion = 1, Points = 2 };
        int method = Auto;

        // What the surface IS in the physics world. Static (the default) cooks
        // the triangles as they are: exact, and free for a scan of a site. A
        // moving body cannot carry a triangle mesh in PhysX, so Dynamic and
        // Kinematic split the surface into convex hulls (V-HACD, `hulls` of
        // them at most) and weld those into one rigid actor at the cloud's
        // pose: Dynamic is simulated and the cloud follows it; Kinematic is
        // driven by the cloud's own transform and pushes dynamics aside.
        enum Body : int { Static = 0, Dynamic = 1, Kinematic = 2 };
        int body = Static;

        // Kilograms, for a Dynamic body. Authored directly, like
        // PhysicsConfig::mass; the inertia follows from the hulls.
        float mass = 10.f;

        // V-HACD budget for a moving body. More hulls follow concavities more
        // closely and cost more at cook time and per contact.
        int hulls = 16;

        static constexpr const char* userDataKey = "splatSurface";

        [[nodiscard]] std::string encode() const;
        [[nodiscard]] static std::optional<SplatSurfaceConfig> decode(const std::string& text);

        // nullopt when the object carries no surface entry.
        [[nodiscard]] static std::optional<SplatSurfaceConfig> read(const Object3D& object);

        // `enabled == false` removes the entry, so a cloud nobody baked leaves
        // no trace — the PhysicsConfig rule.
        void write(Object3D& object) const;

        static void erase(Object3D& object);

        bool operator==(const SplatSurfaceConfig& other) const {
            return enabled == other.enabled && voxelSize == other.voxelSize &&
                   minComponentVoxels == other.minComponentVoxels && poseCount == other.poseCount &&
                   interior == other.interior && method == other.method && body == other.body &&
                   mass == other.mass && hulls == other.hulls;
        }
        bool operator!=(const SplatSurfaceConfig& other) const { return !(*this == other); }
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SPLATSURFACECONFIG_HPP
