// Particle-field authoring, stored on the object itself.
//
// The authored node is a plain Group carrying `userData["particles"]` — one
// flat `key=value;…` string like PhysicsConfig. The node's TRANSFORM is the
// emitter frame: EmitterParams are field-local and the field's matrixWorld is
// applied downstream by every consumer, so moving the Group moves the weather
// and changes not one authored number (see ParticleField.hpp, "WHERE THE
// PARTICLES ARE"). EmitterParams::spawnCenter is therefore NOT authored here —
// the node's own position is it.
//
// A ParticleField object is NEVER a document node. It is built from this config
// by the editor overlay (edit mode) and by the play session (play mode), both
// through the same builder, and it is discarded with them: the type is
// Vulkan-only, has no ObjectLoader case, and would export as its zero-area
// placeholder Mesh (ObjectExporter::isUnexportable blocks it for that reason).
// What a saved document carries is this string and nothing else.
//
// Authored fields are always Ownership::Renderer + WSemantic::Radius. The
// closed-form device emitter is the only mode that runs with no application
// code driving CPU submits, so there is no ownership knob — and w carrying the
// per-particle radius is what makes `sizejitter` free rather than a second
// buffer. TracedRepr, BillboardRepr::blending, Config::attributes, orientations
// and billboard textures are deliberately absent: each is inert, write-once or
// needs asset plumbing that does not exist yet.
//
// CAPACITY, RADIUS, PROXY and DENSITYRES are STRUCTURAL — a field is created
// once at its final capacity and never resized, and a density volume's
// resolution is latched at first enable (ParticleField.hpp churn contract).
// structuralKey() is the string those four hash into, and it is what the
// overlay compares to decide "rebuild the field" versus "push new parameters".

#ifndef THREEPP_EDITOR_PARTICLEFIELDCONFIG_HPP
#define THREEPP_EDITOR_PARTICLEFIELDCONFIG_HPP

#include "threepp/math/Color.hpp"
#include "threepp/math/Vector3.hpp"

#include <optional>
#include <string>

namespace threepp {

    class Object3D;

}// namespace threepp

namespace threepp::editor {

    struct ParticleFieldConfig {

        // The per-particle proxy the mesh representation draws. `Flake` is an
        // octahedron at `radius`, which is what a snow flake reads as at the
        // handful of pixels it covers; `None` leaves the field to its billboard
        // and density representations.
        enum class Proxy {
            None,
            Sphere,
            Flake
        };

        // ── Structural: changing any of these rebuilds the field ────────────
        int   capacity = 20000;
        float radius   = 0.01f;// Config::uniformRadius
        Proxy proxy    = Proxy::Flake;
        int   densityResolution = 128;// latched at first enable

        // ── Emitter (mutable; one setEmitter call) ──────────────────────────
        Vector3 velocity{0.f, -1.f, 0.f};
        float   speedSpread = 0.f;
        // Falling snow and rain are at terminal velocity, i.e. ZERO accel.
        Vector3 accel{0.f, 0.f, 0.f};
        Vector3 wind{0.f, 0.f, 0.f};
        // Birth slab, field-local, and also the toroidal wrap period when
        // `follow` is on. A THIN slab swept over velocity * lifetime is the
        // steady-state cloud; a slab the size of the volume gives a triangular
        // density ramp instead of even snowfall.
        Vector3 spawnHalfExtent{5.f, 0.1f, 5.f};
        float   driftAmplitude = 0.f;
        float   driftFrequency = 0.f;
        float   driftGrowth    = 0.f;
        float   driftScale     = 0.f;
        float   lifetime       = 4.f;
        float   lifetimeJitter = 0.f;
        float   dutyCycle      = 1.f;
        float   size           = 0.02f;
        float   sizeJitter     = 0.f;
        bool    follow         = false;
        // Snap lattice of the follow centre. Choose an integer number of
        // density voxels when the field also carries a DensityRepr on the same
        // centre, or the haze re-phases against its own lattice and swims.
        float   followSnap     = 4.f;
        int     seed           = 20260812;

        // ── Surface landing (emitter sub-block) ─────────────────────────────
        bool  surface           = false;
        float surfaceRest       = 3.f;
        float surfaceRestJitter = 0.6f;
        float surfaceFade       = 1.2f;
        float surfaceSplash     = 0.f;// > 0 = rain rings instead of resting drops
        float surfaceSplashGrow = 7.f;
        // A particle is a solid with a radius; 0 buries half of it.
        float surfaceBias       = 0.f;
        int   surfaceResolution = 256;

        // ── Billboard representation (mutable) ──────────────────────────────
        bool  billboard          = false;
        float billboardSize      = 1.f;
        Color colorHot{1.00f, 0.72f, 0.34f};
        Color colorCool{1.00f, 0.16f, 0.02f};
        float billboardIntensity = 1.f;
        float billboardSoftness  = 0.45f;
        float billboardFade      = 1.6f;
        float billboardJitter    = 0.45f;
        float billboardTaper     = 0.55f;
        float billboardStretch   = 0.f;// seconds of travel to smear over
        float billboardStretchMax = 24.f;
        float billboardNearFade  = 0.f;
        float billboardGlow      = 0.f;// 0 skips the per-field bloom chain entirely
        float billboardGlowThreshold = 0.f;

        // ── Mesh representation (mutable apart from `proxy`) ────────────────
        bool  mesh         = false;
        float meshLodFar   = 0.f;
        float meshLodFade  = 0.f;
        float meshNearCull = 0.f;

        // ── Density representation (mutable apart from `densityResolution`) ─
        bool    density = false;
        float   sigma   = 1.f;// sigma_t per particle, 1/m
        Color   albedo{1.f, 1.f, 1.f};
        float   anisotropy = 0.f;
        // WORLD half-size of the volume box. The scene-wide budget is 4 density
        // volumes (kMaxDensityFields); a fifth is reported, not dropped.
        Vector3 densityHalfExtent{5.f, 5.f, 5.f};
        float   emissiveIntensity = 0.f;// 0 = the exact no-op, no emissive path
        float   tempBottom  = 1900.f;
        float   tempTop     = 800.f;
        float   tempFalloff = 1.6f;

        static constexpr const char* userDataKey = "particles";
        // The scene-wide density budget (ParticleFieldPass kMaxDensityFields).
        static constexpr int maxDensityFields = 4;

        [[nodiscard]] std::string encode() const;
        // Never nullopt: an empty or unparsable string decodes to the defaults,
        // which is what makes a hand-edited document degrade instead of fail.
        // Every value is CLAMPED here to the range the ParticleField setters
        // enforce, so nothing decoded from a document can make them throw.
        [[nodiscard]] static std::optional<ParticleFieldConfig> decode(const std::string& text);

        // nullopt when the object carries no particle entry.
        [[nodiscard]] static std::optional<ParticleFieldConfig> read(const Object3D& object);
        void write(Object3D& object) const;
        static void erase(Object3D& object);

        [[nodiscard]] static bool isParticleField(const Object3D& object);

        // The four structural fields, and only those, as one comparable string.
        // Equal keys mean an existing field can be re-parameterised in place;
        // different keys mean it must be destroyed and rebuilt (a vkDeviceWaitIdle
        // and a cleared TAA history — edit-time only, never per slider tick).
        [[nodiscard]] std::string structuralKey() const;

        // ── Presets ─────────────────────────────────────────────────────────
        // Four points in the same parameter space, not four shaders. Numbers
        // lifted from the shipped references: snow/rain from vulkan_snow.cpp,
        // embers from FireEffect, motes from ParticleField.hpp's archetype note.
        [[nodiscard]] static ParticleFieldConfig snow();
        [[nodiscard]] static ParticleFieldConfig rain();
        [[nodiscard]] static ParticleFieldConfig embers();
        [[nodiscard]] static ParticleFieldConfig motes();

        [[nodiscard]] static const char* label(Proxy proxy);

        bool operator==(const ParticleFieldConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_PARTICLEFIELDCONFIG_HPP
