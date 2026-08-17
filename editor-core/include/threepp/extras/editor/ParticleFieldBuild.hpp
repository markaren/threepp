// One ParticleFieldConfig -> one ParticleField, for everybody.
//
// The editor's edit-mode preview and the play session build the same field
// from the same entry through these functions, so what a scene looks like
// while it is authored and what it looks like when it runs cannot drift into
// two implementations of "snow". Header-only and PhysX-free: the only thing it
// needs is the ParticleField type, which compiles on every backend (it simply
// draws nothing outside Vulkan).
//
// The split between create() and apply() IS the churn contract
// (ParticleField.hpp): a field is created once at its final capacity and never
// resized, so the four STRUCTURAL fields — capacity, radius, proxy,
// densityResolution, i.e. exactly what ParticleFieldConfig::structuralKey()
// hashes — force a destroy-and-rebuild, and everything else is pushed into the
// live field in place. Never rebuild per slider tick.
//
// Three parameters have no authored key and are derived here instead, because
// each is a RELATION rather than a value someone should be typing:
//
//   BillboardRepr::lodNear / lodFade — the mesh->billboard cross-dissolve. The
//     quad must fade in over exactly the band the proxy shrinks out over
//     (MeshRepr::lodFar / lodFade), or a flake crossing it pops or draws
//     twice. Two independent keys are two ways to author that band wrong.
//   BillboardRepr::stretchMaxScreen — a safety cap in NDC, not a look: a near
//     particle projects its correctly capped world streak to a bar across the
//     frame. On whenever the streak is.
//   BillboardRepr::splashRingWidth — left at its own default, which is the
//     value the shipped rain capture uses.

#ifndef THREEPP_EDITOR_PARTICLEFIELDBUILD_HPP
#define THREEPP_EDITOR_PARTICLEFIELDBUILD_HPP

#include "threepp/extras/editor/ParticleFieldConfig.hpp"

#include "threepp/geometries/OctahedronGeometry.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/ParticleField.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>

namespace threepp::editor {

    struct ParticleFieldBuild {

        // Fraction of the frame HEIGHT a stretched sprite may cover. Between
        // the shipped rain (0.045) and ember (0.06) captures: long enough to
        // read as motion smear, short enough never to draw the eye.
        static constexpr float streakScreenCap = 0.05f;

        // A field at its final capacity, in the only mode authored fields use:
        // Ownership::Renderer (the closed-form device emitter — no application
        // code has to feed it) with WSemantic::Radius (w carries the particle's
        // own radius, which is what makes sizeJitter free rather than a second
        // buffer). Every representation and the emitter are applied before it
        // returns, so the caller only has to place it and advance its clock.
        [[nodiscard]] static std::shared_ptr<ParticleField> create(const ParticleFieldConfig& config) {

            ParticleField::Config cfg;
            cfg.capacity = static_cast<std::uint32_t>(std::max(1, config.capacity));
            cfg.ownership = ParticleField::Ownership::Renderer;
            cfg.wSemantic = ParticleField::WSemantic::Radius;
            cfg.uniformRadius = config.radius;

            auto field = ParticleField::create(cfg);
            // A field's Mesh geometry is the zero-area placeholder, so its
            // bounding sphere says nothing about where the particles are — the
            // examples all turn culling off for the same reason.
            field->frustumCulled = false;
            // The transform is pushed in from the authored node every frame.
            field->matrixAutoUpdate = false;
            apply(*field, config);
            return field;
        }

        // The mutable half: everything a live field can take without being
        // rebuilt. setBillboardRepr / setDensityRepr are called only on the
        // FIRST enable — they are the calls that make the colours and the
        // latched volume resolution explicit — and the repr structs are
        // mutated directly afterwards.
        //
        // Every value read here is one ParticleFieldConfig::decode() has
        // already clamped to the range these setters enforce, which is why
        // there is no second clamp expression and no catch.
        static void apply(ParticleField& field, const ParticleFieldConfig& config) {

            field.setEmitter(emitterOf(config));

            // --- mesh proxy ---------------------------------------------
            auto& mr = field.meshRepr();
            if (config.mesh && config.proxy != ParticleFieldConfig::Proxy::None) {
                if (!mr.geometry) {
                    // Authored at the field's uniformRadius, which the proxy
                    // then scales by w / uniformRadius per particle. Both are
                    // structural, so this geometry outlives every mutable edit.
                    std::shared_ptr<BufferGeometry> proxy;
                    if (config.proxy == ParticleFieldConfig::Proxy::Sphere) {
                        proxy = SphereGeometry::create(config.radius, 8, 6);
                    } else {
                        proxy = OctahedronGeometry::create(config.radius, 0);
                    }
                    field.setMeshRepr(std::move(proxy),
                                      MeshStandardMaterial::create(
                                              MeshStandardMaterial::Params{}
                                                      .color(Color(0.96f, 0.97f, 1.00f))
                                                      .roughness(0.85f)
                                                      .metalness(0.f)));
                }
                mr.enabled = true;
                mr.lodFar = config.meshLodFar;
                mr.lodFade = config.meshLodFade;
                mr.nearCull = config.meshNearCull;
            } else {
                mr.enabled = false;
            }

            // --- billboard ----------------------------------------------
            auto& br = field.billboardRepr();
            if (config.billboard) {
                if (!br.enabled) {
                    field.setBillboardRepr(config.colorHot, config.colorCool,
                                           config.billboardIntensity, config.billboardSize);
                } else {
                    br.colorHot = config.colorHot;
                    br.colorCool = config.colorCool;
                    br.intensity = config.billboardIntensity;
                    br.sizeScale = config.billboardSize;
                }
                br.softness = config.billboardSoftness;
                br.fadePower = config.billboardFade;
                br.brightJitter = config.billboardJitter;
                br.sizeTaper = config.billboardTaper;
                br.stretchSeconds = config.billboardStretch;
                br.stretchMax = config.billboardStretchMax;
                br.stretchMaxScreen = config.billboardStretch > 0.f ? streakScreenCap : 0.f;
                br.nearFade = config.billboardNearFade;
                br.glow = config.billboardGlow;
                br.glowThreshold = config.billboardGlowThreshold;
                // The cross-dissolve, derived rather than authored: the quad
                // fades in over exactly the band the proxy shrinks out over.
                // Off entirely when only one representation is drawing, since
                // then there is nothing to hand over to.
                if (mr.enabled && config.meshLodFar > 0.f) {
                    br.lodNear = std::max(config.meshLodFar - config.meshLodFade, 0.f);
                    br.lodFade = config.meshLodFade;
                } else {
                    br.lodNear = 0.f;
                    br.lodFade = 0.f;
                }
            } else {
                br.enabled = false;
            }

            // --- density volume -----------------------------------------
            auto& dr = field.densityRepr();
            if (config.density) {
                if (!dr.enabled) {
                    // The resolution is LATCHED the frame the volume is
                    // allocated, which is what makes densityResolution
                    // structural — a change to it arrives as a rebuilt field,
                    // never as a resize.
                    field.setDensityRepr(worldPosition(field), config.densityHalfExtent,
                                         config.sigma,
                                         static_cast<std::uint32_t>(config.densityResolution));
                }
                dr.sigmaPerParticle = config.sigma;
                dr.albedo = config.albedo;
                dr.anisotropy = config.anisotropy;
                dr.halfExtent = config.densityHalfExtent;
                dr.emissiveIntensity = config.emissiveIntensity;
                dr.tempBottomK = config.tempBottom;
                dr.tempTopK = config.tempTop;
                dr.tempFalloff = config.tempFalloff;
            } else {
                dr.enabled = false;
            }
        }

        // Place the field in the world from the authored node's own matrix.
        //
        // Also re-anchors the density box, and that is the whole reason this is
        // a function rather than two lines at each call site: the emitter is
        // FIELD-LOCAL but DensityRepr::center is WORLD, so moving the node has
        // to move the volume or the haze stays behind. Re-scattered from
        // scratch every frame, so moving the box costs nothing.
        static void setWorldMatrix(ParticleField& field, const Matrix4& matrixWorld) {

            field.matrix->copy(matrixWorld);
            field.matrixWorldNeedsUpdate = true;
            if (field.densityRepr().enabled) {
                field.densityRepr().center.setFromMatrixPosition(matrixWorld);
            }
        }

        // Move the toroidal follow box (EmitterParams::follow) onto the camera.
        //
        // The centre is SNAPPED inside setFollowCenter, and the snapped point
        // is what anything that has to agree with the wrap box must use — above
        // all this field's own density volume, which swims against its own
        // lattice if it is re-derived from the unsnapped camera. Y is left
        // where the node put it: the wrap is lateral only.
        static void setFollowCenter(ParticleField& field, const Vector3& cameraWorldPosition) {

            field.setFollowCenter(cameraWorldPosition);
            if (field.densityRepr().enabled) {
                auto& centre = field.densityRepr().center;
                centre.set(field.followCenter().x, centre.y, field.followCenter().z);
            }
        }

        // What the overlay and the play session compare to decide "rebuild"
        // versus "push new parameters". Forwards to the config so the two
        // cannot answer differently.
        [[nodiscard]] static std::string structuralKey(const ParticleFieldConfig& config) {

            return config.structuralKey();
        }

        // The authored payload of an Ownership::Renderer field, as one struct.
        // spawnCenter is deliberately left at the origin: the node's transform
        // IS the emitter frame (see ParticleFieldConfig).
        [[nodiscard]] static ParticleField::EmitterParams emitterOf(const ParticleFieldConfig& config) {

            ParticleField::EmitterParams e;
            e.spawnHalfExtent = config.spawnHalfExtent;
            e.velocity = config.velocity;
            e.speedSpread = config.speedSpread;
            e.accel = config.accel;
            e.wind = config.wind;
            e.driftAmplitude = config.driftAmplitude;
            e.driftFrequency = config.driftFrequency;
            e.driftGrowth = config.driftGrowth;
            e.driftScale = config.driftScale;
            e.lifetime = config.lifetime;
            e.lifetimeJitter = config.lifetimeJitter;
            e.dutyCycle = config.dutyCycle;
            e.size = config.size;
            e.sizeJitter = config.sizeJitter;
            e.follow = config.follow;
            e.followSnap = config.followSnap;
            e.seed = static_cast<std::uint32_t>(config.seed);

            auto& sf = e.surface;
            sf.enabled = config.surface;
            sf.resolution = static_cast<std::uint32_t>(config.surfaceResolution);
            sf.bias = config.surfaceBias;
            sf.restSeconds = config.surfaceRest;
            sf.restJitter = config.surfaceRestJitter;
            sf.fadeSeconds = config.surfaceFade;
            sf.splashSeconds = config.surfaceSplash;
            sf.splashGrow = config.surfaceSplashGrow;
            // extent and the search band stay 0, i.e. "derive them": the bake
            // then covers exactly the spawn slab's lateral box, which is also
            // the toroidal wrap period, and searches the band one lifetime of
            // fall below its ceiling. Both are what the field can occupy, and
            // neither is a number anyone should be maintaining by hand.

            return e;
        }

    private:
        [[nodiscard]] static Vector3 worldPosition(const ParticleField& field) {

            Vector3 position;
            position.setFromMatrixPosition(*field.matrixWorld);
            return position;
        }
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_PARTICLEFIELDBUILD_HPP
