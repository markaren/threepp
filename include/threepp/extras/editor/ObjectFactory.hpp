// Everything the editor's "Add ▸" menu can create.
//
// Each factory hands back a ready-to-use object: sensible dimensions, a
// MeshStandardMaterial, shadow flags on, and a name that is unique within the
// scene it is about to join ("Box", "Box 2", "Box 3", ...). Nothing is added to
// the scene here — the caller wraps the result in an AddObjectCommand so the
// creation is undoable.

#ifndef THREEPP_EDITOR_OBJECTFACTORY_HPP
#define THREEPP_EDITOR_OBJECTFACTORY_HPP

#include <memory>
#include <string>

namespace threepp {

    class Group;
    class Mesh;
    class Object3D;
    class PerspectiveCamera;

}// namespace threepp

namespace threepp::editor {

    enum class Primitive {
        Box,
        Sphere,
        Plane,
        Cylinder,
        Cone,
        Torus
    };

    enum class LightKind {
        Directional,
        Point,
        Spot,
        Ambient,
        Hemisphere
    };

    class ObjectFactory {

    public:
        // `root` is only read — for the name search and nothing else.
        static std::shared_ptr<Mesh> createPrimitive(Primitive type, const Object3D& root);
        static std::shared_ptr<Object3D> createLight(LightKind kind, const Object3D& root);
        // A Mesh carrying TextConfig, its geometry already built from the
        // default content. Double-sided, because flat text seen from behind
        // must read as text and not vanish.
        static std::shared_ptr<Mesh> createText(const Object3D& root);
        // A plain Object3D carrying a default SoundConfig — a sound has no
        // geometry of its own; the viewport speaker marker is what shows it.
        // The file is attached afterwards, from the inspector.
        static std::shared_ptr<Object3D> createSound(const Object3D& root);
        // A plain Object3D carrying a default JointConfig. Its transform IS
        // the joint frame (anchor + axis), so it is added at its parent's
        // origin and moved with the ordinary gizmo; the parent chain is body
        // A and the other body is chosen in the inspector. The viewport hinge
        // marker is what shows it.
        static std::shared_ptr<Object3D> createJoint(const Object3D& root);
        static std::shared_ptr<Group> createGroup(const Object3D& root);
        static std::shared_ptr<PerspectiveCamera> createCamera(const Object3D& root);
        // A Group carrying SplineConfig, with four control-point children
        // forming a gentle arc — the curve has to show what it is the moment it
        // appears. See SplineConfig: the children ARE the control points.
        static std::shared_ptr<Group> createSpline(const Object3D& root);

        // A new control point for `spline`, named uniquely within it. Nothing
        // is attached — the caller wraps it in an AddObjectCommand, at whatever
        // index the insertion calls for.
        static std::shared_ptr<Object3D> createSplinePoint(const Object3D& spline);

        // A Group carrying ConveyorConfig, with three waypoint children forming
        // a straight run at working height — enough for the generated belt,
        // frame and drums to say what it is the moment it appears. See
        // ConveyorConfig: the children ARE the waypoints.
        static std::shared_ptr<Group> createConveyor(const Object3D& root);

        // A new waypoint for `conveyor` — same contract as createSplinePoint.
        static std::shared_ptr<Object3D> createConveyorPoint(const Object3D& conveyor);

        // A wall attached to `conveyor`: a Group carrying ConveyorWallConfig
        // with two points forming a DIVERTER angled across the belt at the
        // path's midpoint — the obvious starting shape; drag the points from
        // there. The base snaps to the deck, so the points only matter in plan.
        static std::shared_ptr<Group> createConveyorWall(const Object3D& conveyor);

        // A new point for `wall` — same contract as the other point factories.
        static std::shared_ptr<Object3D> createConveyorWallPoint(const Object3D& wall);

        // "Box" if free, else "Box 2", "Box 3", ... Matching is exact, so a
        // user-typed "Box copy" never blocks "Box".
        static std::string uniqueName(const Object3D& root, const std::string& base);

        // Menu labels, also used as the name bases above.
        static const char* label(Primitive type);
        static const char* label(LightKind kind);

        static constexpr Primitive primitives[] = {
                Primitive::Box, Primitive::Sphere, Primitive::Plane,
                Primitive::Cylinder, Primitive::Cone, Primitive::Torus};

        static constexpr LightKind lights[] = {
                LightKind::Directional, LightKind::Point, LightKind::Spot,
                LightKind::Ambient, LightKind::Hemisphere};
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_OBJECTFACTORY_HPP
