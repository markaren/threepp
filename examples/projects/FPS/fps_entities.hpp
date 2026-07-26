// ============================================================================
//  FPS demo — game entity structs
//  Included inside namespace {} in main.cpp — not a standalone header.
//  Requires: fps_constants.hpp, PhysX and threepp types
// ============================================================================

// Dynamic physics prop (crate): mesh + body + home pose for restart.
struct Dynamic {
    std::shared_ptr<Mesh> mesh;
    PxRigidDynamic* body;
    Vector3 home;
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

// Named animation actions for the SWAT enemy. swat.glb ships the full Mixamo
// rifle locomotion set, so the bots get a 6-way pick (see the AI loop) instead
// of the run-or-idle pair they used to slide around on.
struct EnemyAnims {
    AnimationAction* idle = nullptr;     // rifle aiming idle
    AnimationAction* walk = nullptr;     // walking
    AnimationAction* run = nullptr;      // rifle run
    AnimationAction* walkBack = nullptr; // walking backwards
    AnimationAction* runBack = nullptr;  // run backwards
    AnimationAction* strafeL = nullptr;  // strafe left
    AnimationAction* strafeR = nullptr;  // strafe right
    AnimationAction* strafeLFast = nullptr;// "strafe (2)" — fast left
    AnimationAction* strafeRFast = nullptr;// "strafe"     — fast right
    AnimationAction* fire = nullptr;     // firing rifle (additive overlay)
    AnimationAction* reload = nullptr;   // reloading
    AnimationAction* hit = nullptr;      // hit reaction
};

// Pooled SWAT visual: the 14 MB GLB is loaded once per slot at startup and
// slots are recycled across enemy spawns (threepp has no skinned-mesh clone).
struct EnemySlot {
    std::shared_ptr<Group> rig;    // scene-level group, position driven from physics
    std::shared_ptr<Object3D> model;// swat.glb scene (child of rig)
    std::shared_ptr<Group> rifle;  // rifle.glb instance, pinned to the right hand
    std::unique_ptr<AnimationMixer> mixer;
    EnemyAnims anims;
    AnimationAction* current = nullptr;
    Object3D* handBone = nullptr;
    Object3D* leftHandBone = nullptr;
    Object3D* hipsBone = nullptr;
    Vector3 hipsBind;
    Vector3 muzzleLocal;// rifle-local barrel tip (tracer/flash origin)
    std::shared_ptr<Sprite> flash;// per-slot muzzle-flash sprite (own material)
    std::shared_ptr<SpriteMaterial> flashMat;
    float flashT = 0.f;// >0 while the flash sprite is showing; independent of burst state
    float fireW = 0.f;// smoothed additive fire-overlay weight
    bool inUse = false;
};
