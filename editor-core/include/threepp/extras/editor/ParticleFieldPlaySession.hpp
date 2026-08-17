// The PlaySession that makes an authored particle field RUN.
//
// One ParticleField per node carrying a ParticleFieldConfig, built through the
// same ParticleFieldBuild the editor's edit-mode preview uses — so what the
// weather looks like while it is authored and what it looks like under Play (or
// under the player, in CI) cannot drift into two implementations of "snow".
// PhysX-free and unguarded: a particle field needs a renderer, not a world, so
// this session exists in every build the way SensorPlaySession's vision half
// does.
//
// The CLOCK is the whole difference from the preview. The overlay runs on wall
// time so snow falls while somebody is dragging a slider; this session starts at
// t = 0 on every start() and advances by the frame delta it is handed, so two
// episodes of the same document are the same weather — the emitter is a
// stateless closed form f(seed, slot, t) (ParticleField.hpp), so nothing else is
// needed to make an episode reproducible.
//
// VULKAN ONLY, and it says so out loud. The type draws nothing on OpenGL by
// decision, so on a GL session no field is built at all — but the authored nodes
// are COUNTED regardless, because "the document has three fields and this
// backend drew none of them" is a fact worth reporting and "reports zero" is the
// regression the count exists to make visible (PlayerCore.hpp).
//
// The fields are added to the SCENE ROOT, not under the authored node, for the
// same reason ConveyorPlaySession's travelling bars are: their world pose is
// pushed straight in. A field's matrixAutoUpdate is off and its `matrix` IS its
// world matrix (see ParticleFieldBuild::setWorldMatrix, which also re-anchors
// the density volume — the emitter is field-local but DensityRepr::center is
// world), and a parent with a transform of its own would multiply that matrix a
// second time. They are scene content either way: the play snapshot was taken
// before start(), so Stop restores a document that never saw them.

#ifndef THREEPP_EDITOR_PARTICLEFIELDPLAYSESSION_HPP
#define THREEPP_EDITOR_PARTICLEFIELDPLAYSESSION_HPP

#include "threepp/extras/editor/ParticleFieldBuild.hpp"
#include "threepp/extras/editor/ParticleFieldConfig.hpp"
#include "threepp/extras/editor/PlaySession.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/ParticleField.hpp"
#include "threepp/renderers/Renderer.hpp"
#include "threepp/scenes/Scene.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace threepp::editor {

    class ParticleFieldPlaySession: public PlaySession {

    public:
        [[nodiscard]] std::string name() const override { return "Particles"; }

        // A session destroyed without a stop() (the app torn down mid-Play) must
        // still let go of the fields it parented into a scene that is about to
        // die.
        ~ParticleFieldPlaySession() override { release(); }

        // --- wiring (set once, before the first Play) ------------------------

        // The renderer whose BACKEND decides whether there are visuals at all.
        // Borrowed and only ever asked what it is; null is a supported state and
        // reads as "not Vulkan", which is what a headless run is.
        void setRenderer(Renderer* renderer) { renderer_ = renderer; }

        // Where the camera is, for the toroidal follow box (EmitterParams::
        // follow) — a function rather than a node because the editor's viewport
        // camera is whichever of several the current view is using.
        //
        // Unset leaves the follow centre where the authored node put it, which
        // is the right answer for a run with nobody looking: a field that wraps
        // about a viewpoint that does not exist has no viewpoint to wrap about.
        void setViewpoint(std::function<Vector3()> viewpoint) { viewpoint_ = std::move(viewpoint); }

        void setLogger(std::function<void(const std::string&)> logger) { logger_ = std::move(logger); }

        // --- readouts --------------------------------------------------------

        // Authored nodes the last start() saw, on EVERY backend. Not the number
        // of fields built — see the header note.
        [[nodiscard]] std::size_t fieldNodeCount() const { return fieldNodeCount_; }

        // Fields actually built, i.e. zero unless this session is drawing on
        // Vulkan.
        [[nodiscard]] std::size_t liveFieldCount() const { return entries_.size(); }

        // Whether a field would be drawn if one were authored: the backend gate,
        // asked the same way the editor's preview asks it.
        [[nodiscard]] bool visualsAvailable() const {
#ifdef THREEPP_WITH_VULKAN
            return dynamic_cast<const VulkanRenderer*>(renderer_) != nullptr;
#else
            return false;
#endif
        }

        // Seconds of THIS episode. Zero at every start(), and the number every
        // field's emitter is evaluated at.
        [[nodiscard]] float emitterTime() const { return t_; }

        // The field built for the node with this uuid, or nullptr — by uuid
        // rather than by pointer because the uuid is what survives a Stop, the
        // same contract the sensor session's entries keep.
        [[nodiscard]] const ParticleField* fieldFor(const std::string& uuid) const {

            for (const auto& entry : entries_) {
                if (entry.uuid == uuid) return entry.field.get();
            }
            return nullptr;
        }

        // --- PlaySession -----------------------------------------------------

        void start(Scene& scene) override {

            release();
            t_ = 0.f;

            // Collect first, build second: creating a field is a structural
            // scene change (the churn contract), and a stable list keeps the
            // build order — and therefore every seeded stream — independent of
            // anything the traversal might otherwise be walking into.
            scene.updateMatrixWorld(true);
            struct Authored {
                Object3D* node;
                ParticleFieldConfig config;
            };
            std::vector<Authored> authored;
            scene.traverse([&](Object3D& object) {
                if (const auto config = ParticleFieldConfig::read(object)) {
                    authored.push_back(Authored{&object, *config});
                }
            });

            fieldNodeCount_ = authored.size();
            if (authored.empty()) return;

            if (!visualsAvailable()) {
                // Once per Play, at the moment the user is looking, and then
                // never again: the backend is a fact about the run, not a
                // per-frame event. The nodes are still counted above.
                log("particles: " + std::to_string(authored.size()) +
                    " field(s) need the Vulkan backend - skipping visuals");
                return;
            }

            for (const auto& owner : authored) {
                Entry entry;
                entry.uuid = owner.node->uuid;
                entry.node = owner.node;
                entry.follow = owner.config.follow;
                entry.field = ParticleFieldBuild::create(owner.config);
                entry.field->name = "Particles (play)";
                // Placed before the first frame is drawn, so a field is never
                // seen for one frame at the origin.
                owner.node->updateWorldMatrix(true, false);
                ParticleFieldBuild::setWorldMatrix(*entry.field, *owner.node->matrixWorld);
                scene.add(entry.field);
                entries_.push_back(std::move(entry));
            }
        }

        void update(float dt) override {

            // Advanced on every backend: the episode's clock is what makes it
            // deterministic, and it is not the visuals' property.
            t_ += dt;
            if (entries_.empty()) return;

            Vector3 viewpoint;
            const bool viewing = static_cast<bool>(viewpoint_);
            if (viewing) viewpoint = viewpoint_();

            for (auto& entry : entries_) {
                if (!entry.node) continue;

                // Every frame: physics, a script or an animation can be moving
                // the authored node, and the emitter is field-local — the node's
                // world matrix is the whole placement.
                entry.node->updateWorldMatrix(true, false);
                ParticleFieldBuild::setWorldMatrix(*entry.field, *entry.node->matrixWorld);

                entry.field->setEmitterTime(t_, dt);
                if (entry.follow && viewing) {
                    ParticleFieldBuild::setFollowCenter(*entry.field, viewpoint);
                }
            }
        }

        void stop() override { release(); }

    private:
        struct Entry {
            // The authored node by uuid AND by pointer: the uuid is what
            // survives, the pointer is valid only between start() and stop().
            std::string uuid;
            Object3D* node = nullptr;
            std::shared_ptr<ParticleField> field;
            bool follow = false;
        };

        // Let go of the fields. The snapshot restore drops the whole scene they
        // were added to, so this is not what cleans the document up — it is what
        // keeps this session from holding the last reference to a field after
        // the graph around it is gone.
        void release() {

            for (auto& entry : entries_) {
                if (entry.field) entry.field->removeFromParent();
            }
            entries_.clear();
        }

        void log(const std::string& message) const {

            if (logger_) logger_(message);
        }

        Renderer* renderer_ = nullptr;
        std::function<Vector3()> viewpoint_;
        std::function<void(const std::string&)> logger_;

        std::vector<Entry> entries_;
        // Kept across stop(), like the conveyor's count: a caller reads it to
        // report what the episode contained.
        std::size_t fieldNodeCount_ = 0;
        float t_ = 0.f;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_PARTICLEFIELDPLAYSESSION_HPP
