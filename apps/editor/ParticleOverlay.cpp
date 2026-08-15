// Particle-field previews: the weather an authored node describes, falling
// while it is authored.
//
// Derived state, like every other sync pass here — the undoable step is the
// userData edit, and this follows the config wherever undo/redo/load leaves
// it. What makes it different from the conveyor's is the CHURN CONTRACT
// (ParticleField.hpp): a field is created once at its final capacity and is
// never resized, and creating or destroying one costs a vkDeviceWaitIdle and a
// cleared TAA history. So the change detection is in two tiers — a structural
// key (capacity, radius, proxy, density resolution) that rebuilds, and the
// encoded config that is pushed into the live field in place — and a slider
// drag pays neither.
//
// Nothing here is a document node. A ParticleField has no ObjectLoader case
// and would export as its zero-area placeholder Mesh, so the previews live
// under the editor-only overlay and are dropped with it; what a save carries
// is the userData string and nothing else.
//
// VULKAN ONLY. The type renders no particles on OpenGL by decision, so on a GL
// editor no field is built at all — the map stays empty, the inspector says so
// in as many words, and the spawn-box helper below is the whole picture. That
// helper is also the only part that draws for a granular node, whose grains
// are a PhysX PBD simulation that exists only while playing.

#include "EditorApp.hpp"
#include "EditorTheme.hpp"

#include "threepp/extras/editor/FlockConfig.hpp"
#include "threepp/extras/editor/GranularConfig.hpp"
#include "threepp/extras/editor/ParticleFieldBuild.hpp"
#include "threepp/extras/editor/ParticleFieldConfig.hpp"

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/ParticleField.hpp"
#include "threepp/scenes/Scene.hpp"

#ifdef THREEPP_WITH_VULKAN
#include "threepp/renderers/VulkanRenderer.hpp"
#endif

#include <imgui.h>// theme colours are ImVec4

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Over scene geometry, under the marker icons — the band the sound rings
    // and the joint helper draw in.
    constexpr int kHelperRenderOrder = 3000;

    // The arrow says WHICH WAY, never how fast: a rain field flies at 9 m/s and
    // an ember at 1.4, and drawing either to scale makes one of them useless.
    float arrowLength(const Vector3& extent) {

        const float reach = std::max({extent.x, extent.y, extent.z});
        return std::max(reach * 0.35f, 0.5f);
    }

    void appendSegment(std::vector<float>& out, const Vector3& a, const Vector3& b) {

        out.insert(out.end(), {a.x, a.y, a.z, b.x, b.y, b.z});
    }

    // The twelve edges of [-extent, +extent] about the node's origin: the birth
    // slab, which is the one authored number a weather field cannot be judged
    // without (a slab the size of the volume gives a density ramp, not snowfall).
    void appendBox(std::vector<float>& out, const Vector3& extent) {

        const float x = extent.x, y = extent.y, z = extent.z;
        const Vector3 corners[8]{{-x, -y, -z}, {x, -y, -z}, {x, -y, z}, {-x, -y, z},
                                 {-x, y, -z}, {x, y, -z}, {x, y, z}, {-x, y, z}};
        const int edges[12][2]{{0, 1}, {1, 2}, {2, 3}, {3, 0},
                               {4, 5}, {5, 6}, {6, 7}, {7, 4},
                               {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        for (const auto& edge : edges) appendSegment(out, corners[edge[0]], corners[edge[1]]);
    }

    // Direction plus a two-barb head, from the node's origin. `direction` need
    // not be normalised; a zero vector draws nothing.
    void appendArrow(std::vector<float>& out, const Vector3& direction, float length) {

        Vector3 dir = direction;
        if (dir.length() < 1e-5f) return;
        dir.normalize();

        Vector3 tip = dir;
        tip.multiplyScalar(length);
        appendSegment(out, Vector3(0.f, 0.f, 0.f), tip);

        // Any axis not parallel to the flight gives a barb plane.
        Vector3 lateral(0.f, 1.f, 0.f);
        if (std::abs(dir.dot(lateral)) > 0.99f) lateral.set(1.f, 0.f, 0.f);
        lateral.cross(dir).normalize();

        const float barb = length * 0.18f;
        for (const float side : {1.f, -1.f}) {
            Vector3 back = tip;
            back.addScaledVector(dir, -barb * 1.6f).addScaledVector(lateral, barb * side);
            appendSegment(out, tip, back);
        }
    }

}// namespace


bool EditorApp::particlePreviewAvailable() const {

#ifdef THREEPP_WITH_VULKAN
    return dynamic_cast<const VulkanRenderer*>(renderer_.get()) != nullptr;
#else
    return false;
#endif
}

void EditorApp::syncParticleOverlays(float dt) {

    if (!particles_) return;

    // --- who is authored this frame ---------------------------------------
    // The density budget is scene-wide (ParticleFieldPass kMaxDensityFields) and
    // is counted on EVERY backend: the inspector's warning is about the document,
    // not about what this session happens to be able to draw.
    struct Authored {
        Object3D* node;
        ParticleFieldConfig config;
    };
    static thread_local std::vector<Authored> owners;
    owners.clear();
    int densityFields = 0;
    document_.scene().traverse([&](Object3D& object) {
        if (document_.isEditorOnly(object)) return;
        const auto config = ParticleFieldConfig::read(object);
        if (!config) return;
        if (config->density) ++densityFields;
        owners.push_back(Authored{&object, *config});
    });
    particleDensityCount_ = densityFields;

    syncParticleHelper();

    if (!particlePreviewAvailable()) return;

    // --- retire previews whose node or entry is gone ----------------------
    // Undo of an Add lands here, and so does erasing the userData entry: both
    // leave a field in the scene that nothing describes any more.
    for (auto it = particlePreviews_.begin(); it != particlePreviews_.end();) {
        const bool authored = std::any_of(owners.begin(), owners.end(),
                                          [&](const Authored& a) { return a.node->uuid == it->first; });
        if (authored) {
            ++it;
            continue;
        }
        if (it->second.field) it->second.field->removeFromParent();
        it = particlePreviews_.erase(it);
    }

    if (owners.empty()) return;

    // Wall clock, so snow falls while it is being authored — and frozen for the
    // duration of a Play, where the session owns the clock and the previews are
    // parked out of its way.
    if (!isPlaying()) particleTime_ += dt;

    Vector3 cameraWorld;
    viewCamera().getWorldPosition(cameraWorld);

    for (const auto& owner : owners) {

        const auto structural = ParticleFieldBuild::structuralKey(owner.config);
        const auto encoded = owner.config.encode();

        auto& preview = particlePreviews_[owner.node->uuid];
        if (!preview.field || preview.structuralKey != structural) {
            // The one path that pays the structural cost. Edit time only, and
            // only for the four fields a field's identity is made of.
            if (preview.field) preview.field->removeFromParent();
            preview.field = ParticleFieldBuild::create(owner.config);
            preview.field->name = "__editor_particle_preview";
            particles_->add(preview.field);
            preview.structuralKey = structural;
            // create() applied the config already.
            preview.mutableKey = encoded;
        } else if (preview.mutableKey != encoded) {
            preview.mutableKey = encoded;
            ParticleFieldBuild::apply(*preview.field, owner.config);
        }

        // Every frame: the gizmo can be dragging the node, and the emitter is
        // field-local, so the node's world matrix is the whole placement.
        owner.node->updateMatrixWorld();
        ParticleFieldBuild::setWorldMatrix(*preview.field, *owner.node->matrixWorld);

        if (isPlaying()) {
            // Parked, not removed: a field with liveCount 0 skips its emit
            // dispatch and costs one entry, while removing it would be a
            // structural change on every Play and every Stop.
            preview.field->setLiveCount(0);
            continue;
        }
        preview.field->setLiveCount(preview.field->capacity());
        preview.field->setEmitterTime(particleTime_, dt);
        if (owner.config.follow) {
            ParticleFieldBuild::setFollowCenter(*preview.field, cameraWorld);
        }
    }
}

void EditorApp::clearParticleOverlays() {

    for (auto& [uuid, preview] : particlePreviews_) {
        if (preview.field) preview.field->removeFromParent();
    }
    particlePreviews_.clear();
    particleDensityCount_ = 0;
    if (particleHelper_) {
        particleHelper_->removeFromParent();
        particleHelper_.reset();
        particleHelperKey_.clear();
    }
}

// ------------------------------------------------------------- spawn box aid

void EditorApp::syncParticleHelper() {

    // Only for the SELECTED node, the sound rings' rule: a box per field would
    // bury the scene, and the numbers are in the inspector either way. Drawn on
    // every backend — where the particles are born is an authoring fact, not a
    // rendering one, and on GL it is the only picture there is.
    //
    // The flock shares the helper: its territory is the same kind of authored
    // extent as a spawn slab, and while playing the birds themselves are the
    // picture, so edit mode is exactly when the circle earns its place.
    auto* selected = selection_.get();
    const auto particles = selected ? ParticleFieldConfig::read(*selected) : std::nullopt;
    const auto granular = selected && !particles ? GranularConfig::read(*selected) : std::nullopt;
    const auto flock = selected && !particles && !granular ? FlockConfig::read(*selected)
                                                           : std::nullopt;

    if (!particles && !granular && !flock) {
        if (particleHelper_) {
            particleHelper_->removeFromParent();
            particleHelper_.reset();
            particleHelperKey_.clear();
        }
        return;
    }

    // Keyed by uuid (a play/stop replaces the whole graph) plus the numbers the
    // picture is built from — a rebuild trigger, not a hash; the placement below
    // runs every frame regardless. The flock rides the same two slots: extent
    // carries (roamRadius, cruiseAltitude, altitudeSpread), flight the wind.
    const Vector3 extent = particles ? particles->spawnHalfExtent
                           : granular
                                   ? Vector3(granular->emitExtentX, 0.f, granular->emitExtentZ)
                                   : Vector3(flock->roamRadius, flock->cruiseAltitude,
                                             flock->altitudeSpread);
    Vector3 flight = particles ? particles->velocity
                     : granular ? granular->emitVelocity
                                : Vector3(flock->windX, 0.f, flock->windZ);
    if (particles) flight.add(particles->wind);

    char key[224];
    std::snprintf(key, sizeof(key), "%s|%.4f,%.4f,%.4f|%.4f,%.4f,%.4f",
                  selected->uuid.c_str(),
                  static_cast<double>(extent.x), static_cast<double>(extent.y),
                  static_cast<double>(extent.z), static_cast<double>(flight.x),
                  static_cast<double>(flight.y), static_cast<double>(flight.z));

    if (!particleHelper_) {
        auto material = LineBasicMaterial::create(
                LineBasicMaterial::Params().vertexColors(true).toneMapped(false));
        material->transparent = true;
        material->opacity = 0.9f;
        // A spawn slab is usually a ceiling with the whole scene under it; an
        // outline hidden by what it is raining on is an outline authored blind.
        material->depthTest = false;
        particleHelper_ = LineSegments::create(BufferGeometry::create(), material);
        particleHelper_->renderOrder = kHelperRenderOrder;
        particleHelper_->frustumCulled = false;
        particleHelper_->matrixAutoUpdate = false;
        particleHelperKey_.clear();
        overlay_->add(particleHelper_);
    }

    if (particleHelperKey_ != key) {
        particleHelperKey_ = key;

        std::vector<float> box, arrow;
        if (particles) {
            appendBox(box, extent);
        } else if (granular) {
            // A chute has no vertical extent to draw: the pour mouth is a
            // rectangle in the node's own XZ plane.
            const Vector3 corners[4]{{-extent.x, 0.f, -extent.z}, {extent.x, 0.f, -extent.z},
                                     {extent.x, 0.f, extent.z}, {-extent.x, 0.f, extent.z}};
            for (int i = 0; i < 4; ++i) appendSegment(box, corners[i], corners[(i + 1) % 4]);
        } else {
            // The territory: the soft roam edge as a circle in the node's XZ
            // plane (extent.x = roamRadius), a second ring at 0.75× where the
            // bounds force starts to bite, and the SAME full circle again at
            // expected-ground level (cruiseAltitude = extent.y below) — the
            // loiter volume floats, but an editor camera looks down at a
            // scene, and extents read against the ground it will be judged
            // over. A drop line with a ground tick joins the two; the tick
            // landing off the actual floor is the one spatial fact a
            // misplaced flock node gets wrong. The band rings at
            // ±cruiseAltitude·spread (extent.z) show the ALTITUDE BAND the
            // birds hold: each bird prefers ground + cruiseAltitude·(1 ±
            // spread), so with the ground where the tick claims, the flock
            // flies between these two rings.
            constexpr int kSegments = 48;
            const auto appendRing = [&](float radius, float y) {
                for (int i = 0; i < kSegments; ++i) {
                    const float a0 = static_cast<float>(i) * math::TWO_PI / kSegments;
                    const float a1 = static_cast<float>(i + 1) * math::TWO_PI / kSegments;
                    appendSegment(box,
                                  {radius * std::cos(a0), y, radius * std::sin(a0)},
                                  {radius * std::cos(a1), y, radius * std::sin(a1)});
                }
            };
            appendRing(extent.x, 0.f);
            appendRing(extent.x * 0.75f, 0.f);
            appendRing(extent.x, -extent.y);
            const float band = extent.y * extent.z;
            if (band > 0.01f) {
                appendRing(extent.x * 0.9f, band);
                appendRing(extent.x * 0.9f, -band);
            }
            appendSegment(box, {0.f, 0.f, 0.f}, {0.f, -extent.y, 0.f});
            const float tick = std::max(extent.x * 0.05f, 0.5f);
            appendSegment(box, {-tick, -extent.y, 0.f}, {tick, -extent.y, 0.f});
            appendSegment(box, {0.f, -extent.y, -tick}, {0.f, -extent.y, tick});
        }
        appendArrow(arrow, flight, arrowLength(extent));

        std::vector<float> positions = box;
        positions.insert(positions.end(), arrow.begin(), arrow.end());

        const auto tint = theme::accent();
        std::vector<float> colors;
        colors.reserve(positions.size());
        const auto shadeFor = [&](std::size_t floats, float shade) {
            for (std::size_t v = 0; v < floats / 3; ++v) {
                colors.insert(colors.end(), {tint.x * shade, tint.y * shade, tint.z * shade});
            }
        };
        shadeFor(box.size(), 0.6f);
        shadeFor(arrow.size(), 1.f);

        // Replaced wholesale rather than rewritten: the buffer only changes when
        // an extent or the flight direction is edited, and the old geometry is
        // disposed so the renderer provably lets go of it (the sound rings' rule).
        const auto old = particleHelper_->geometry();
        auto geometry = BufferGeometry::create();
        geometry->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        geometry->setAttribute("color", FloatBufferAttribute::create(colors, 3));
        particleHelper_->setGeometry(geometry);
        if (old) old->dispose();
    }

    // Every frame: applyAuthoringVisibility hides the node for a screenshot pass
    // or a Play, and nothing else would turn it back on.
    particleHelper_->visible = true;

    // The whole world matrix, scale included: the extents are metres in the
    // node's own frame, which is exactly what the emitter reads them as.
    selected->updateWorldMatrix(true, false);
    particleHelper_->matrix->copy(*selected->matrixWorld);
    particleHelper_->matrixWorldNeedsUpdate = true;
}
