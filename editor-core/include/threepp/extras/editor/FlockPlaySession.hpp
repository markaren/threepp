// The PlaySession that makes an authored flock FLY: one Flock per node
// carrying a FlockConfig, built when Play starts and ticked every frame.
//
// Header-only and dependency-free beyond threepp core — no PhysX, no
// renderer, no guard. The birds work identically on GL, Vulkan and headless,
// which is the Flock's own design contract (Flock.hpp's RENDERING note).
//
// The Flock goes to the SCENE ROOT, not under the authored node, for the
// granular session's reason verbatim: the simulation is world-space and the
// mesh wants an identity parent (Flock.hpp: "THE FLOCK NODE SHOULD STAY AT
// IDENTITY"). The authored node's transform contributes exactly one thing —
// its world position becomes the territory's `home`. The birds are scene
// content either way: the play snapshot was taken before start(), so Stop
// restores a document that never saw them.
//
// PERCHES BAKE AGAINST THE SCENE AS PLAY FINDS IT, amortised. bakePerches()
// starts the work-unit-budgeted bake and Flock::update() advances it, so a
// heavy scene costs a few frames of birds-not-landing-yet instead of a hitch
// on the Play button. Zero perches is a normal answer (a sky-only scene) and
// the flock simply flies — PerchIndex's own contract.

#ifndef THREEPP_EDITOR_FLOCKPLAYSESSION_HPP
#define THREEPP_EDITOR_FLOCKPLAYSESSION_HPP

#include "threepp/extras/editor/FlockConfig.hpp"
#include "threepp/extras/editor/PlaySession.hpp"

#include "threepp/extras/fauna/Flock.hpp"

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/scenes/Scene.hpp"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace threepp::editor {

    class FlockPlaySession: public PlaySession {

    public:
        [[nodiscard]] std::string name() const override { return "Flock"; }

        // --- wiring (set once, before the first Play) ------------------------

        // Which meshes the perch bake may sample. THE EDITOR MUST SET THIS to
        // exclude its overlay meshes: gizmo handles, light markers and waypoint
        // pucks live in the same scene graph the bake traverses, and a marker
        // hovering at altitude becomes the "highest sampled surface" in its
        // column — the whole flock then chases a phantom floor (measured:
        // birds at y=430 over a flat template scene). Same contract as
        // Flock::setPerchFilter, which is where it lands.
        void setMeshFilter(std::function<bool(const Mesh&)> filter) { filter_ = std::move(filter); }

        void start(Scene& scene) override {

            scene_ = &scene;

            // A node placed this very frame has a stale matrixWorld until the
            // next render; home must not read it one frame old.
            scene.updateMatrixWorld();

            // Collect first, add after — adding during traverse would walk the
            // birds we are in the middle of creating.
            struct Entry {
                FlockConfig config;
                Vector3 home;
            };
            std::vector<Entry> entries;
            scene.traverse([&](Object3D& node) {
                if (!node.visible) return;// a hidden flock node plays silent
                const auto config = FlockConfig::read(node);
                if (!config) return;
                Vector3 home;
                home.setFromMatrixPosition(*node.matrixWorld);
                entries.push_back({*config, home});
            });

            for (const auto& entry : entries) {
                auto flock = Flock::create(entry.config.makeParams(entry.home));
                if (filter_) flock->setPerchFilter(filter_);
                scene.add(flock);
                // Amortised on purpose — see the header note. The flock
                // excludes its own mesh from the bake itself.
                flock->bakePerches(scene);
                flocks_.push_back(std::move(flock));
            }
        }

        void update(float dt) override {

            for (const auto& flock : flocks_) flock->update(dt);
        }

        void stop() override {

            if (scene_) {
                for (const auto& flock : flocks_) scene_->remove(*flock);
            }
            flocks_.clear();
            scene_ = nullptr;
        }

    private:
        Scene* scene_ = nullptr;
        std::function<bool(const Mesh&)> filter_;
        std::vector<std::shared_ptr<Flock>> flocks_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_FLOCKPLAYSESSION_HPP
