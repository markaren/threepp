// A player-controlled character authored ON an imported skinned model's root.
//
// The authoring is deliberately small, for VehicleConfig's reason: import a
// Mixamo-style character, tick "Simulate as Character", press Play, walk. The
// capsule, the gait speeds and WHICH CLIP IS WHICH are all DERIVED from the
// model and from the clips it carries, so nothing has to be typed for the
// first Play to work.
//
// Deriving the clip roles is the part worth explaining. Names are useless as a
// key — one pack calls its clips "walking" / "left strafe walking" / "left
// strafe", another calls them "strafe left" / "strafe (2)" — so the roles are
// read off the MOTION instead: every clip's root bone carries the travel its
// animator authored, and measuring it says both which way that clip goes (in
// the model's own frame: +Z forward, +X left) and how fast. Forward, backward
// and the two strafes fall out of the direction; within each direction the
// slower clip is the walk and the faster one is the run. A clip that does not
// travel is a candidate for Idle or Jump, and those two ARE picked by name,
// since standing still and jumping on the spot look identical to a ruler.
//
// The measured speed is not a curiosity: playing a clip at 1x while the
// character travels at some OTHER speed is exactly what foot-sliding is. The
// play session picks whichever clip is closest in ratio to the speed actually
// being travelled and time-scales away the remainder (the same trick the FPS
// demo's bots use), which needs the authored number for each clip.
//
// Encoding follows the PhysicsConfig/VehicleConfig template: one flat
// `key=value;` string under userData["character"], every key written on every
// write, unknown keys ignored on read. The per-gait clip NAMES are user-typed
// and free to contain the flat format's `;`/`=` delimiters, so each rides a
// plain userData key of its own (characterClipIdle, ...) — the same escape
// hatch VehicleConfig::wheelKeys and the sound file use.
//
// The PRESENCE of the entry is what makes the node a character (SoundConfig's
// rule), so write() always writes, defaults included; unticking "Simulate as
// Character" erases every key.
//
// Lengths are metres, angles RADIANS (only the inspector's widgets speak
// degrees), speeds m/s. Nothing here depends on PhysX: the struct is authoring
// data, and turning it into a live capsule controller is
// CharacterPlaySession's job.

#ifndef THREEPP_EDITOR_CHARACTERCONFIG_HPP
#define THREEPP_EDITOR_CHARACTERCONFIG_HPP

#include "threepp/math/Vector3.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace threepp {

    class AnimationClip;
    class Object3D;

}// namespace threepp

namespace threepp::editor {

    // The locomotion roles the controller blends between. Order is the storage
    // order of everything indexed by gait, so only ever append.
    enum class Gait {
        Idle,
        Walk,
        Run,
        WalkBack,
        RunBack,
        StrafeLeft,
        StrafeLeftFast,
        StrafeRight,
        StrafeRightFast,
        Jump
    };

    inline constexpr std::size_t kGaitCount = 10;

    inline constexpr std::size_t gaitIndex(Gait gait) { return static_cast<std::size_t>(gait); }

    // What Play needs beyond the stored scalars: the model measured, the root
    // bone found, and every gait resolved to a clip with its own authored
    // ground speed. Derived fresh at every use — the model is the source of
    // truth. `problem` says what is missing when !valid.
    struct CharacterGeometry {

        bool valid = false;
        std::string problem;

        // The model's standing size, metres, measured in its own frame.
        float height = 0.f;
        // Capsule radius: the model's half-DEPTH, floored at human proportion.
        // Half-WIDTH is not usable — a skinned character's bind pose is a T,
        // so its X extent is an arm span, not a body (see derived()).
        float radius = 0.f;
        // World position of the model's lowest point, i.e. where its feet are
        // standing right now. The capsule spawns centred height/2 above it.
        Vector3 feet;

        // The root-motion bone: the node whose translation track carries each
        // clip's travel. The play session pins its horizontal position every
        // frame, because the CONTROLLER owns where the character is; the
        // vertical component is the gait's bob and rides through untouched.
        Object3D* rootBone = nullptr;
        // Its authored local position, i.e. what to pin back to.
        Vector3 rootBoneBind;

        struct Slot {
            std::shared_ptr<AnimationClip> clip;
            // The clip's OWN ground speed, m/s (0 for a clip that stands
            // still). See the header note: this is what stops the feet
            // sliding.
            float speed = 0.f;
        };
        std::array<Slot, kGaitCount> gaits;

        [[nodiscard]] const Slot& slot(Gait gait) const { return gaits[gaitIndex(gait)]; }

        // How many gaits actually resolved — the inspector's readout, and the
        // one number a test asserts to say the auto-match worked.
        [[nodiscard]] std::size_t resolvedCount() const;
    };

    struct CharacterConfig {

        enum class Facing {
            // Yaw follows the VIEW. W/S walk and backpedal along the camera's
            // forward, A/D genuinely strafe — which is what a locomotion pack's
            // strafe and backward clips are FOR. The default.
            Camera,
            // Yaw turns towards travel, so every direction is the forward gait
            // and the strafe clips never play. Cheaper to control, and the
            // right answer for an NPC or a top-down game.
            Movement
        };

        Facing facing = Facing::Camera;

        // The capsule is measured off the model while this is on; the stored
        // fields below only act once one is overridden (which flips it, and
        // the inspector seeds both from the derived values so nothing jumps).
        bool autoGeometry = true;
        float height = 1.8f;
        float radius = 0.3f;

        // Same rule for the speeds, read off the clips.
        bool autoSpeeds = true;
        float walkSpeed = 1.6f;
        float runSpeed = 4.4f;

        // Always authored.
        float mass = 75.f;
        // Apex of a standing jump, metres. Converted to a launch velocity
        // against `gravity`, so the two stay consistent by construction.
        float jumpHeight = 0.9f;
        // Deliberately heavier than 9.81: a character falling at survey
        // gravity floats, and every game engine's controller does this.
        float gravity = 18.f;
        // Kerb height the capsule walks over instead of into.
        float stepOffset = 0.35f;
        // Steeper than this and the character slides rather than climbs.
        float slopeLimit = 0.87f;// radians, ~50 degrees
        // Exponential yaw approach, 1/s. High enough to feel immediate,
        // low enough that a flicked camera does not snap the model.
        float turnRate = 14.f;
        // Ground acceleration towards the demanded velocity, 1/s (same
        // exponential form). Air control is a fraction of it (see the session).
        float accel = 14.f;
        // Crossfade between gaits, seconds.
        float blendTime = 0.18f;

        // Per-gait clip overrides. Empty means "auto-match" — which is the
        // normal case and the whole point (see the header). A name that does
        // not resolve leaves that gait unfilled and says so, rather than
        // silently falling back to a guess the user did not ask for.
        std::array<std::string, kGaitCount> clips{};

        static constexpr const char* userDataKey = "character";
        static constexpr const char* clipKeys[kGaitCount] = {
                "characterClipIdle", "characterClipWalk", "characterClipRun",
                "characterClipWalkBack", "characterClipRunBack",
                "characterClipStrafeLeft", "characterClipStrafeLeftFast",
                "characterClipStrafeRight", "characterClipStrafeRightFast",
                "characterClipJump"};
        static constexpr const char* gaitLabels[kGaitCount] = {
                "Idle", "Walk", "Run", "Walk Back", "Run Back",
                "Strafe Left", "Strafe Left (fast)",
                "Strafe Right", "Strafe Right (fast)", "Jump"};

        [[nodiscard]] std::string encode() const;
        [[nodiscard]] static std::optional<CharacterConfig> decode(const std::string& text);

        // nullopt when the object carries no character entry. A present entry
        // is what makes the node a character, so the clip names are filled in
        // from their keys on the way out.
        [[nodiscard]] static std::optional<CharacterConfig> read(const Object3D& object);

        // Whether this node is an authored character at all — the predicate
        // the inspector section, the play session and the physics session's
        // exclusion walk all share.
        [[nodiscard]] static bool isCharacter(const Object3D& object);

        // Writes the flat entry (defaults included — presence is the node's
        // identity) plus the clip keys; an empty clip name erases its key, so
        // an all-automatic character leaves the smaller document.
        void write(Object3D& object) const;

        static void erase(Object3D& object);

        // Measure `root` and classify the clips it carries. Non-const root:
        // measuring needs the world matrices up to date. Always derives — the
        // caller picks derived vs authored numbers by the auto flags, and the
        // clips and the root bone are needed either way.
        //
        // `clips` are the model's own animations (Object3D::animations on the
        // root, or whatever the importer handed back).
        [[nodiscard]] CharacterGeometry derived(
                Object3D& root,
                const std::vector<std::shared_ptr<AnimationClip>>& clips) const;

        // The overload that reads the clips off the node itself — what every
        // caller that is not a test wants.
        [[nodiscard]] CharacterGeometry derived(Object3D& root) const;

        static const char* label(Facing facing);

        static constexpr Facing facings[] = {Facing::Camera, Facing::Movement};

        bool operator==(const CharacterConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_CHARACTERCONFIG_HPP
