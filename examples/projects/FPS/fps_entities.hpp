// ============================================================================
//  FPS demo — game entity structs
//  Included inside namespace {} in main.cpp — not a standalone header.
//  Requires: fps_constants.hpp, PhysX and threepp types
// ============================================================================

// A range prop: a real PhysX dynamic body with a mesh collider, its home pose,
// and — for the steel plates — whether it is currently down.
//
// The whole demo turns on this being genuine rigid-body physics rather than a
// scripted knock-over animation: a hit applies an impulse AT THE CONTACT
// POINT, so where you hit a plate decides whether it tips, spins or barely
// rocks, and a crate stack collapses the way its own masses say it should.
struct Dynamic {
    // Plates score and pop back up; props are scenery you can knock about.
    enum class Kind { Plate, Prop };

    std::shared_ptr<Mesh> mesh;
    PxRigidDynamic* body;
    Vector3 home;
    Quaternion homeRot;
    Kind kind = Kind::Prop;
    // Latched once the plate has tipped past kPlateDownCos, so one fall scores
    // once however long it lies there.
    bool down = false;
    float resetIn = 0.f;// counts down while a downed plate waits to stand up
};

// Short-lived visual (tracer / flash / spark).
struct Ephemeral {
    std::shared_ptr<Object3D> obj;
    float ttl;
};

// Spent brass tossed from the ejection port. Cheap ballistic prop (gravity +
// tumble + damped bounce, TTL-recycled) — NOT a physics body: spawning a rigid
// actor ~9x/s would churn the scene.
struct Casing {
    std::shared_ptr<Mesh> mesh;
    Vector3 vel;
    Vector3 spinAxis;
    float spinRate = 0.f;
    float ttl = 0.f;
    float groundY = 0.02f;// resolved at spawn (player may stand on a platform)
    bool tinked = false;  // brass-tink sfx fires once, on the first bounce
};

// A short-lived burst of camera-facing billboard sprites per hit (dust on
// surfaces, blood on enemies), integrated on the CPU and faded via opacity.
struct ParticleBurst {
    std::shared_ptr<Group> group;
    std::shared_ptr<SpriteMaterial> mat;
    std::vector<std::shared_ptr<Sprite>> sprites;
    std::vector<Vector3> pos, vel;
    float ttl, life, gravity, drag;
    float maxOpacity = 1.f;// caps the fade-in peak; < 1 reads as a hazy/translucent cloud
};
