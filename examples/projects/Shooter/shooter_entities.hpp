// ============================================================================
//  Shooter — game entity structs
//  Included inside namespace {} in main.cpp — not a standalone header.
//  Requires: shooter_constants.hpp (kEnemyHp), PhysX and threepp types
// ============================================================================

// ========================================================================
//  Game entities
// ========================================================================

// One jointed limb of a death ragdoll (spawned when the enemy dies).
struct RagdollPart {
    std::shared_ptr<Mesh> mesh;
    PxRigidDynamic* body = nullptr;
};

struct Enemy {
    std::shared_ptr<Mesh> visual;// capsule body (added to scene)
    std::shared_ptr<MeshStandardMaterial> mat;
    PxRigidDynamic* body = nullptr;
    int hp = kEnemyHp;
    bool alive = true;
    float deadTtl = 0.f;
    float attackCd = 0.f;
    // death ragdoll: the live capsule becomes the torso, these are the limbs
    // jointed to it; both vectors are populated in killEnemy, freed in removeEnemy.
    std::vector<RagdollPart> parts;
    std::vector<PxJoint*> joints;
};

struct Ephemeral {// short-lived visual (tracer / flash / spark)
    std::shared_ptr<Object3D> obj;
    float ttl;
};

// Dynamic physics prop (crate or barrel): mesh + body + home pose for restart.
struct Dynamic {
    std::shared_ptr<Mesh> mesh;
    PxRigidDynamic* body;
    Vector3 home;
};

// Named animation action pointers for all locomotion and combat states.
struct PlayerAnims {
    AnimationAction* idle = nullptr;    // rifle aiming idle
    AnimationAction* walk = nullptr;    // walking
    AnimationAction* walkBack = nullptr;// walking backwards
    AnimationAction* run = nullptr;     // rifle run
    AnimationAction* runBack = nullptr; // run backwards
    AnimationAction* strafeL = nullptr; // strafe left
    AnimationAction* strafeR = nullptr; // strafe right
    AnimationAction* fire = nullptr;    // firing rifle
    AnimationAction* reload = nullptr;  // reloading
    AnimationAction* jump = nullptr;    // rifle jump
    AnimationAction* hit = nullptr;     // hit reaction
    AnimationAction* grenade = nullptr; // toss grenade (additive overlay)
};

// Active grenade projectile: mesh + physics body + remaining fuse time.
struct Grenade {
    std::shared_ptr<Mesh> mesh;
    PxRigidDynamic* body;
    float fuse;
};
