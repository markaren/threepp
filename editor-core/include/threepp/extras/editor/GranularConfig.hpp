// Granular-particle authoring, stored on the object itself.
//
// The authored node is a plain Group carrying `userData["granular"]` — one flat
// `key=value;…` string like PhysicsConfig. The node's TRANSFORM is the chute
// frame: grains are poured from a slab at the node's origin along `emitVelocity`
// (-Y by default), so aiming a chute is the ordinary gizmo and nothing here
// carries a world position.
//
// The simulation is PhysX PBD (PbdParticles), which is CUDA-ONLY — it throws
// without PhysxWorld::cudaContextManager(). That is a PLAY-time constraint, not
// an authoring one: this config is just strings, compiles and round-trips with
// no PhysX in the build, and a session on a CPU world declines with one log
// line while the rigids keep playing.
//
// PbdParticles is not an Object3D — rendering the grains is the application's
// job — so `visual` picks how the play session draws them: an InstancedMesh
// (GL-capable) or a ParticleField MeshRepr (Vulkan). `Auto` resolves to the one
// the running backend supports.
//
// maxNeighborhood, selfCollision and speculativeCcd stay at the library
// defaults and get no key: the first sizes a GPU allocation and the other two
// are the difference between a pile and grains leaving the world.

#ifndef THREEPP_EDITOR_GRANULARCONFIG_HPP
#define THREEPP_EDITOR_GRANULARCONFIG_HPP

#include "threepp/math/Color.hpp"
#include "threepp/math/Vector3.hpp"

#include <optional>
#include <string>

namespace threepp {

    class Object3D;

}// namespace threepp

namespace threepp::editor {

    struct GranularConfig {

        enum class Visual {
            Auto,     // field on Vulkan, instanced on GL
            Instanced,// InstancedMesh — the GL-capable path
            Field     // ParticleField MeshRepr over the host ring
        };

        // ── Sim ─────────────────────────────────────────────────────────────
        // Resting distance between two touching grains: the one knob that
        // matters, since the render radius is spacing * 0.5 and everything else
        // derives from it.
        float spacing = 0.06f;
        // Position iterations. A deep pile keeps sinking into itself with too few.
        int iterations = 8;
        int capacity = 100000;
        // Velocity clamp; 0 = derive from spacing. Not optional in the library
        // — an emitter overlapping existing grains depenetrates explosively.
        float maxVelocity = 0.f;

        // ── Material (PxPBDMaterial; there is no restitution) ───────────────
        float friction = 0.4f;// the repose angle of a heap is its internal friction
        float damping = 0.f;
        float adhesion = 0.f; // sticks to RIGIDS
        float cohesion = 0.f; // sticks to OTHER GRAINS — the pile-angle knob
        float viscosity = 0.f;
        float gravityScale = 1.f;

        // ── Emitter (the node frame is the chute frame) ─────────────────────
        // Half-size of the pour mouth in the chute's XZ plane. Grains are placed
        // on a LATTICE inside a thin slab and jittered by a fraction of a cell,
        // never sampled uniformly: PBD depenetrates an overlap violently.
        float emitExtentX = 0.2f;
        float emitExtentZ = 0.2f;
        float rate = 5000.f;// grains per second, accumulated fractionally
        Vector3 emitVelocity{0.f, -1.f, 0.f};
        float mass = 0.f;   // 0 = the library's 1 kg default
        float emitFor = 0.f;// seconds; 0 = pour until capacity
        float jitter = 0.2f;// fraction of a lattice cell

        // ── Visual ──────────────────────────────────────────────────────────
        Visual visual = Visual::Auto;
        Color color{0.85f, 0.74f, 0.51f};
        float roughness = 0.8f;

        static constexpr const char* userDataKey = "granular";

        [[nodiscard]] std::string encode() const;
        // Never nullopt: an empty or unparsable string decodes to the defaults,
        // clamped to the ranges PbdParticles::Settings expects.
        [[nodiscard]] static std::optional<GranularConfig> decode(const std::string& text);

        // nullopt when the object carries no granular entry.
        [[nodiscard]] static std::optional<GranularConfig> read(const Object3D& object);
        void write(Object3D& object) const;
        static void erase(Object3D& object);

        [[nodiscard]] static bool isGranular(const Object3D& object);

        [[nodiscard]] static const char* label(Visual visual);

        bool operator==(const GranularConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_GRANULARCONFIG_HPP
