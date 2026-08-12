// The PlaySession that makes an authored granular chute POUR: one PhysX PBD
// particle system in the world PhysicsPlaySession built, one group + one chute
// emitter + one visual per node carrying a GranularConfig.
//
// Header-only and PhysX-dependent, exactly like ConveyorPlaySession — included
// only by builds that found the SDK (the apps set THREEPP_EDITOR_WITH_PHYSX,
// and the registration sites carry the guard).
//
// The world is BORROWED, with the conveyor's contract verbatim: on the normal
// Stop it is still alive here (PlayController stops sessions in reverse
// registration order, physics last), and the teardown paths that cannot promise
// that are guarded by PhysicsPlaySession's lifetime token — when the token is
// dead the particle system is abandoned rather than asked to release a buffer
// through a destroyed CUDA context.
//
// TWO WAYS TO DECLINE, both of them loud and neither of them fatal:
//
//   • no physics this play at all — the grains stay decorative, like a
//     conveyor with no belts;
//   • a world with no CUDA context. PBD is a CUDA-only PhysX feature with NO
//     CPU path (PbdParticles.hpp:8-12: the constructor throws), so a machine
//     without a GPU — or one whose GPU world failed to come up and fell back to
//     the CPU — gets one log line while the rigid half of the scene plays on.
//
// ONE PbdParticles for the whole play, and the sim block of the FIRST authored
// config wins. Spacing, the neighbourhood grid and the iteration counts are
// physically global to a particle system — everything in one system collides
// with everything else in it, which is the point — so mixed grain sizes would
// mean one solver each and no interaction between the piles. A second chute
// asking for something different is told so and pours the first one's grains.
//
// The VISUALS go to the SCENE ROOT, not under the authored node, and that is
// not a style choice: the solver's positions are WORLD space and both visuals
// push them in unmodified, so the node they hang from must have an identity
// world matrix. A chute node has a transform by construction (it stands over
// something), and a child of it would carry that transform twice. They are
// scene content either way — the play snapshot was taken before start(), so
// Stop restores a document that never saw them.

#ifndef THREEPP_EDITOR_GRANULARPLAYSESSION_HPP
#define THREEPP_EDITOR_GRANULARPLAYSESSION_HPP

#include "threepp/extras/editor/GranularConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"

#include "threepp/extras/physx/ChuteEmitter.hpp"
#include "threepp/extras/physx/GranularVisual.hpp"
#include "threepp/extras/physx/PhysxParticles.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/geometries/IcosahedronGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/renderers/Renderer.hpp"
#include "threepp/scenes/Scene.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <algorithm>
#include <cstddef>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace threepp::editor {

    class GranularPlaySession: public PlaySession {

    public:
        [[nodiscard]] std::string name() const override { return "Granular"; }

        // A session destroyed without a stop() (the app torn down mid-Play)
        // must still release the solver while its world is alive, or say it
        // cannot.
        ~GranularPlaySession() override { release(); }

        // --- wiring (set once, before the first Play) ------------------------

        // Where the particle system registers. Borrowed; the token this session
        // reads from it is what keeps a stopped world from being dereferenced.
        void setPhysics(PhysicsPlaySession* physics) { physics_ = physics; }

        // The renderer whose BACKEND resolves Visual::Auto. Borrowed and only
        // ever asked what it is; null reads as "not Vulkan", which is what a
        // headless run is — and the instanced visual works there.
        void setRenderer(Renderer* renderer) { renderer_ = renderer; }

        void setLogger(std::function<void(const std::string&)> logger) { logger_ = std::move(logger); }

        // --- readouts --------------------------------------------------------

        // Authored chutes the last start() saw, whether or not any of them
        // poured. Kept across stop(), like the conveyor's count: a caller reads
        // it to report what the episode contained.
        [[nodiscard]] std::size_t granularNodeCount() const { return granularNodeCount_; }

        // Groups actually built — zero on every machine that declined.
        [[nodiscard]] std::size_t groupCount() const { return entries_.size(); }

        // Live grains across every group, right now.
        [[nodiscard]] unsigned activeGrainCount() const {
            return particles_ ? particles_->active() : 0u;
        }

        // The scene had chutes and this session did not simulate them. `reason`
        // is the line that went to the log, kept for a test that has to assert
        // WHICH way a machine declined.
        [[nodiscard]] bool declined() const { return declined_; }
        [[nodiscard]] const std::string& reason() const { return reason_; }

        // Which visual Visual::Auto resolves to here — the backend gate, asked
        // the same way the particle session asks it.
        [[nodiscard]] bool fieldVisuals() const {
#ifdef THREEPP_WITH_VULKAN
            return dynamic_cast<const VulkanRenderer*>(renderer_) != nullptr;
#else
            return false;
#endif
        }

        // --- PlaySession -----------------------------------------------------

        void start(Scene& scene) override {

            release();
            granularNodeCount_ = 0;
            declined_ = false;
            reason_.clear();

            PhysxWorld* world = nullptr;
            worldLife_.reset();
            if (physics_) {
                world = physics_->world();
                worldLife_ = physics_->lifetime();
            }

            // Collect first, build second, like the particle session: the build
            // order fixes every seeded stream, so it must not depend on what
            // the traversal is walking into.
            scene.updateMatrixWorld(true);
            struct Authored {
                Object3D* node;
                GranularConfig config;
            };
            std::vector<Authored> authored;
            scene.traverse([&](Object3D& object) {
                if (const auto config = GranularConfig::read(object)) {
                    authored.push_back(Authored{&object, *config});
                }
            });

            granularNodeCount_ = authored.size();
            if (authored.empty()) return;

            if (!world) {
                decline("granular: no physics this play - grains stay decorative");
                return;
            }
            if (!world->cudaContextManager()) {
                // The documented guard. PhysicsPlaySession escalates to GPU
                // dynamics for exactly this scene, so arriving here means the
                // escalation failed and fell back to a CPU world — which it
                // already said out loud.
                decline("granular: grains need a CUDA GPU - skipping");
                return;
            }

            const auto& first = authored.front().config;
            PbdParticles::Settings settings;
            settings.spacing = first.spacing;
            settings.solverIterations = static_cast<unsigned>(first.iterations);
            settings.maxVelocity = first.maxVelocity;
            for (std::size_t i = 1; i < authored.size(); ++i) {
                const auto& other = authored[i].config;
                if (other.spacing == first.spacing && other.iterations == first.iterations &&
                    other.maxVelocity == first.maxVelocity) continue;
                log("granular: \"" + authored[i].node->name +
                    "\" asks for a different grain size - one solver runs the scene, so the "
                    "first chute's sim settings win");
            }

            try {
                particles_ = std::make_unique<PbdParticles>(*world, settings);
            } catch (const std::exception& e) {
                decline(std::string("granular: the PBD solver declined to start (") + e.what() +
                        ") - skipping");
                return;
            }

            const float radius = particles_->solidRestOffset();
            for (const auto& owner : authored) {
                Entry entry;
                entry.uuid = owner.node->uuid;
                entry.node = owner.node;
                entry.config = owner.config;
                // One seed per chute, so two of them scatter differently — and
                // the same seed every episode, so one of them scatters the same.
                const auto seed = kSeed + static_cast<unsigned>(entries_.size()) * 977u;

                PbdParticles::MaterialSpec material;
                material.friction = owner.config.friction;
                material.damping = owner.config.damping;
                material.adhesion = owner.config.adhesion;
                material.cohesion = owner.config.cohesion;
                material.viscosity = owner.config.viscosity;
                material.gravityScale = owner.config.gravityScale;
                // Adhesion with a zero radius scale is a no-op, and the config
                // has no key for the scale: derive it rather than authoring a
                // number whose only legal value is "not zero".
                if (owner.config.adhesion > 0.f) material.adhesionRadiusScale = 1.f;

                const auto capacity = static_cast<unsigned>(std::max(1, owner.config.capacity));
                try {
                    entry.group = &particles_->addGroup(capacity, material);
                } catch (const std::exception& e) {
                    log("granular: \"" + owner.node->name + "\" could not be created - " + e.what());
                    continue;
                }

                ChuteEmitter::Settings chute;
                chute.halfExtentX = owner.config.emitExtentX;
                chute.halfExtentZ = owner.config.emitExtentZ;
                chute.spacing = settings.spacing;// the solver's, not this node's
                chute.jitter = owner.config.jitter;
                chute.seed = seed;
                entry.emitter = std::make_unique<ChuteEmitter>(chute);

                entry.visual = makeVisual(owner.config, radius, capacity, seed);
                if (entry.visual) {
                    entry.visual->object()->name = "Grains (play)";
                    scene.add(entry.visual->object());
                }
                entries_.push_back(std::move(entry));
            }
        }

        void update(float dt) override {

            if (!particles_) return;

            // Emitting is a device write to the buffer the solver owns, so it
            // has to happen BETWEEN simulate() calls. It does: the physics
            // session is registered before this one and has already stepped the
            // world this frame.
            for (auto& entry : entries_) {
                if (!entry.group || !entry.node || !entry.emitter) continue;
                // emitFor 0 = pour until capacity; the buffer's own tail stops
                // it either way (Group::emit clamps rather than corrupting).
                if (entry.config.emitFor > 0.f && t_ >= entry.config.emitFor) continue;
                const unsigned room = entry.group->capacity() - entry.group->active();
                if (room == 0) continue;

                // Every frame: the chute is an ordinary node, and physics, a
                // script or an animation can be moving it.
                entry.node->updateWorldMatrix(true, false);
                const auto& burst =
                        entry.emitter->tick(dt, entry.config.rate, *entry.node->matrixWorld, room);
                if (burst.empty()) continue;

                // The launch direction is authored in the chute's frame, which
                // is what makes aiming one the ordinary gizmo. Its SPEED is
                // not scaled by the node: a chute made twice as wide pours
                // through a wider mouth, not twice as hard.
                Vector3 velocity = entry.config.emitVelocity;
                if (const float speed = velocity.length(); speed > 0.f) {
                    velocity.transformDirection(*entry.node->matrixWorld).multiplyScalar(speed);
                }
                entry.group->emit(burst.data(), static_cast<unsigned>(burst.size()), velocity,
                                  entry.config.mass);
            }
            t_ += dt;

            // ONE drain for every group, then the visuals read their own slice
            // of it (PbdParticles::pull is one CUDA lock and one synchronize).
            particles_->pull();
            for (auto& entry : entries_) {
                if (entry.visual && entry.group) {
                    entry.visual->update(entry.group->positions(), entry.group->active());
                }
            }
        }

        void stop() override { release(); }

    private:
        struct Entry {
            // The authored node by uuid AND by pointer: the uuid is what
            // survives a Stop, the pointer is valid only between start() and
            // stop().
            std::string uuid;
            Object3D* node = nullptr;
            GranularConfig config;
            PbdParticles::Group* group = nullptr;// owned by particles_
            std::unique_ptr<ChuteEmitter> emitter;
            std::unique_ptr<IGrainVisual> visual;
        };

        // Fixed, so an episode is repeatable as far as the solver allows. (GPU
        // PhysX is not bit-deterministic; the emitter's scatter is.)
        static constexpr unsigned kSeed = 20260812u;

        [[nodiscard]] std::unique_ptr<IGrainVisual> makeVisual(const GranularConfig& config,
                                                               float radius, unsigned capacity,
                                                               unsigned seed) const {

            // detail 0 = a 20-face icosahedron, about half the triangles of even
            // a coarse UV sphere — which matters capacity times over.
            auto geometry = IcosahedronGeometry::create(radius, 0);
            auto material = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}
                            .color(config.color)
                            .roughness(config.roughness)
                            .metalness(0.f)
                            .flatShading(true));

            const bool field = config.visual == GranularConfig::Visual::Field ||
                               (config.visual == GranularConfig::Visual::Auto && fieldVisuals());
            if (field) {
                return std::make_unique<FieldGrainVisual>(geometry, material, capacity, seed,
                                                          radius);
            }
            return std::make_unique<InstancedGrainVisual>(geometry, material, capacity, seed);
        }

        void release() {

            // The world may already be gone on a teardown that skipped the
            // controller (the app closed mid-Play, destructor order): then the
            // actor, the buffers and the CUDA context died with it, and the
            // particle system must not free through any of them.
            if (particles_ && worldLife_.expired()) particles_->abandon();

            for (auto& entry : entries_) {
                if (entry.visual) entry.visual->object()->removeFromParent();
            }
            // Before the solver: an Entry holds a Group that belongs to it.
            entries_.clear();
            particles_.reset();
            worldLife_.reset();
            t_ = 0.f;
        }

        void decline(std::string reason) {

            declined_ = true;
            reason_ = std::move(reason);
            log(reason_);
        }

        void log(const std::string& message) const {

            if (logger_) logger_(message);
        }

        PhysicsPlaySession* physics_ = nullptr;
        Renderer* renderer_ = nullptr;
        std::function<void(const std::string&)> logger_;

        std::weak_ptr<const void> worldLife_;
        std::unique_ptr<PbdParticles> particles_;
        std::vector<Entry> entries_;

        std::size_t granularNodeCount_ = 0;
        bool declined_ = false;
        std::string reason_;
        float t_ = 0.f;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_GRANULARPLAYSESSION_HPP
