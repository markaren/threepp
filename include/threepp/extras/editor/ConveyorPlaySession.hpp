// The PlaySession that makes authored conveyors RUN: belt colliders in the
// world PhysicsPlaySession built, and the visual motion (belt texture scroll,
// roller/drum spin, travelling cleat bars) on the derived meshes the editor
// generated.
//
// Header-only and PhysX-dependent, exactly like PhysicsPlaySession — included
// only by builds that found the SDK (the editor sets THREEPP_EDITOR_WITH_PHYSX).
//
// The world is borrowed, never owned — the same contract as
// PhysxSensorPlaySession: on the normal Stop it is still alive here
// (PlayController stops sessions in REVERSE registration order, physics last),
// and the teardown paths that cannot promise that are guarded by
// PhysicsPlaySession's lifetime token — when the token is dead the sim is
// abandoned rather than asked to release actors in a destroyed world.
//
// Everything visual this session touches (texture offsets, roller quaternions,
// hidden preview bars, its own travelling bars added to the scene) is undone by
// the play snapshot on Stop, like any other session mutation.

#ifndef THREEPP_EDITOR_CONVEYORPLAYSESSION_HPP
#define THREEPP_EDITOR_CONVEYORPLAYSESSION_HPP

#include "threepp/extras/conveyor/ConveyorPhysics.hpp"
#include "threepp/extras/editor/ConveyorConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace threepp::editor {

    class ConveyorPlaySession: public PlaySession {

    public:
        [[nodiscard]] std::string name() const override { return "Conveyor"; }

        // Where the belts register. Borrowed; the token this session reads from
        // it is what keeps a stopped world from being dereferenced.
        void setPhysics(PhysicsPlaySession* physics) { physics_ = physics; }

        // Conveyors picked up on the last start(), for a status readout.
        [[nodiscard]] std::size_t conveyorCount() const { return conveyorCount_; }

        // Belt colliders the sim built, exposed for tests.
        [[nodiscard]] std::size_t beltCount() const {
            return sim_ ? sim_->beltCount() : 0;
        }

        // Global speed multiplier (belt drag, texture scroll, roller spin,
        // cleat travel — all of it, so the visuals never disagree with the
        // physics).
        void setSpeedScale(float scale) {
            speedScale_ = scale;
            if (sim_) sim_->speedScale = scale;
        }
        [[nodiscard]] float speedScale() const { return speedScale_; }

        ~ConveyorPlaySession() override { release(); }

        void start(Scene& scene) override {

            conveyorCount_ = 0;
            scene_ = &scene;

            PhysxWorld* world = nullptr;
            worldLife_.reset();
            if (physics_) {
                world = physics_->world();
                worldLife_ = physics_->lifetime();
            }
            if (!world) return;// no physics this play — conveyors stay decorative

            scene.updateMatrixWorld(true);

            // Resolve every authored conveyor to a WORLD-space spec, and gather
            // the visual moving parts from its derived group.
            std::vector<conveyor::ConveyorSpec> specs;
            scene.traverse([&](Object3D& object) {
                const auto config = ConveyorConfig::read(object);
                if (!config) return;

                auto spec = config->spec(object);
                object.updateWorldMatrix(true, false);
                // Corner radii are authored in the conveyor's local units; a
                // scaled group scales its bends with it (mean horizontal scale
                // — the fillet lives in the XZ plane).
                Vector3 p, s;
                Quaternion q;
                object.matrixWorld->decompose(p, q, s);
                const float radiusScale = (std::abs(s.x) + std::abs(s.z)) * 0.5f;
                for (auto& wp : spec.waypoints) {
                    wp.pos.applyMatrix4(*object.matrixWorld);
                    wp.cornerRadius *= radiusScale;
                }
                collectVisuals(object, *config);
                specs.push_back(std::move(spec));
                ++conveyorCount_;
            });

            if (specs.empty()) return;

            sim_ = std::make_unique<conveyor::ConveyorPhysics>(*world, std::move(specs));
            sim_->speedScale = speedScale_;

            spawnCleatBars(scene);
        }

        void update(float dt) override {

            if (!sim_) return;
            const float scale = sim_->speedScale;

            // Scroll each belt texture to fake surface motion — same speed
            // scale as the kinematic drag, so the visual matches the physics.
            for (auto& belt : beltVisuals_) {
                belt.texture->offset.y += belt.scrollRate * scale * dt;
                belt.texture->offset.y -= std::floor(belt.texture->offset.y);// keep in [0,1)
                // The Vulkan renderer caches each material's uvTransform in a
                // GPU MaterialDesc buffer and re-uploads it only when
                // Material::version() changes — a bare texture offset write
                // never reaches the GPU without this. (GL recomputes the UV
                // matrix per draw and does not need it.)
                belt.material->needsUpdate();
            }

            // Spin rollers and end drums about their own axis (local +Y), on
            // top of the fixed width-axis orientation.
            for (auto& bank : spinners_) {
                bank.angle += bank.omega * scale * dt;
                Quaternion spin;
                spin.setFromAxisAngle(Vector3(0, 1, 0), bank.angle);
                for (auto& part : bank.parts) {
                    part.mesh->quaternion.multiplyQuaternions(part.base, spin);
                }
            }

            // Move each travelling bar to its collider's pose so the cleats
            // visibly ride along (and up an incline).
            for (auto& track : barMeshes_) {
                for (std::size_t i = 0; i < track.meshes.size(); ++i) {
                    Vector3 p;
                    Quaternion q;
                    sim_->cleatPose(track.track, i, p, q);
                    track.meshes[i]->position.copy(p);
                    track.meshes[i]->quaternion.copy(q);
                }
            }
        }

        void stop() override { release(); }

    private:
        // A belt ribbon material whose texture is scrolled each frame.
        struct BeltVisual {
            std::shared_ptr<Texture> texture;
            std::shared_ptr<Material> material;// version-bumped so the scroll re-uploads
            float scrollRate = 0.f;            // d(offset.y)/dt at speedScale 1
        };

        // Rollers / drums of one conveyor, spun in step with its belt.
        struct SpinningPart {
            Mesh* mesh = nullptr;
            Quaternion base;// fixed width-axis orientation the spin rides on
        };
        struct SpinnerBank {
            std::vector<SpinningPart> parts;
            float omega = 0.f;// rad/s at speedScale 1
            float angle = 0.f;
        };

        // The session-owned meshes mirroring one cleat track's bar colliders.
        struct BarTrack {
            std::size_t track = 0;
            std::vector<std::shared_ptr<Mesh>> meshes;
        };

        void collectVisuals(Object3D& conveyorNode, const ConveyorConfig& config) {

            auto* derived = ConveyorConfig::derivedGroup(conveyorNode);
            if (!derived) return;

            const float direction = config.reverse ? 1.f : -1.f;
            std::unordered_set<Material*> seenBeltMaterials;
            SpinnerBank rollers, drums;
            rollers.omega = direction * config.speed / std::max(config.rollerRadius, 1e-3f);
            const auto profile = conveyor::FrameProfile::forWidth(config.width);
            drums.omega = direction * config.speed / std::max(profile.drumRadius, 1e-3f);

            derived->traverseType<Mesh>([&](Mesh& mesh) {
                const auto role = ConveyorConfig::roleOf(mesh);
                if (role == "belt") {
                    // Every ribbon of one conveyor shares one material — scroll
                    // it once, not once per run.
                    auto material = mesh.material();
                    if (!material || !seenBeltMaterials.insert(material.get()).second) return;
                    if (auto* withMap = material->as<MaterialWithMap>(); withMap && withMap->map) {
                        // offset.y is in tile units → scroll rate =
                        // speed [m/s] * repeat.y [tiles/m]; negative drags the
                        // pattern along travel, flipped for a reversed belt.
                        beltVisuals_.push_back({withMap->map, material,
                                                direction * config.speed * withMap->map->repeat.y});
                    }
                } else if (role == "roller") {
                    rollers.parts.push_back({&mesh, mesh.quaternion});
                } else if (role == "drum") {
                    drums.parts.push_back({&mesh, mesh.quaternion});
                } else if (role == "cleat") {
                    // Static preview bars; the travelling colliders (and the
                    // session's own bar meshes) take over during play.
                    mesh.visible = false;
                }
            });

            if (!rollers.parts.empty()) spinners_.push_back(std::move(rollers));
            if (!drums.parts.empty()) spinners_.push_back(std::move(drums));
        }

        // One mesh per travelling bar collider, added to the SCENE ROOT so the
        // world-space actor poses can be copied straight in.
        void spawnCleatBars(Scene& scene) {

            auto material = MeshStandardMaterial::create();
            material->color = Color(0x202428);
            material->roughness = 0.6f;
            material->metalness = 0.3f;

            for (std::size_t t = 0; t < sim_->trackCount(); ++t) {
                const auto& spec = sim_->specs()[sim_->trackSpecIndex(t)];
                auto geometry = BoxGeometry::create(conveyor::kCleatThickness,
                                                    spec.cleatHeight, spec.width);
                BarTrack track;
                track.track = t;
                for (std::size_t i = 0; i < sim_->cleatCount(t); ++i) {
                    auto bar = Mesh::create(geometry, material);
                    bar->name = "Cleat (play)";
                    bar->castShadow = true;
                    Vector3 p;
                    Quaternion q;
                    sim_->cleatPose(t, i, p, q);
                    bar->position.copy(p);
                    bar->quaternion.copy(q);
                    scene.add(bar);
                    track.meshes.push_back(std::move(bar));
                }
                barMeshes_.push_back(std::move(track));
            }
        }

        void release() {

            // The world may already be gone on a teardown that skipped the
            // controller (editor closed mid-Play, destructor order): then the
            // actors and substep callbacks died with it, and the sim must not
            // touch either.
            if (sim_ && worldLife_.expired()) sim_->abandon();
            sim_.reset();
            worldLife_.reset();

            for (auto& track : barMeshes_) {
                for (auto& mesh : track.meshes) mesh->removeFromParent();
            }
            barMeshes_.clear();
            beltVisuals_.clear();
            spinners_.clear();
            scene_ = nullptr;
        }

        PhysicsPlaySession* physics_ = nullptr;
        std::weak_ptr<const void> worldLife_;
        Scene* scene_ = nullptr;
        std::unique_ptr<conveyor::ConveyorPhysics> sim_;
        float speedScale_ = 1.f;
        std::size_t conveyorCount_ = 0;

        std::vector<BeltVisual> beltVisuals_;
        std::vector<SpinnerBank> spinners_;
        std::vector<BarTrack> barMeshes_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_CONVEYORPLAYSESSION_HPP
