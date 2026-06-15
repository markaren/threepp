
#ifndef THREEPP_DRIVE_CARRIG_HPP
#define THREEPP_DRIVE_CARRIG_HPP

// A car built from scratch out of threepp primitives — no imported model.
//
//   - body : stacked / tapered boxes (lower body, greenhouse cabin, bonnet,
//            bumpers) + chrome trim + emissive lamp lenses.
//   - wheels: cylinder tyre + rim + hub, one Group per corner, driven each frame
//            by the vehicle's per-wheel local pose (steer + spin + jounce baked in).
//   - struts: a visible coil-over per corner — a helical spring tube + damper rod
//            spanning a fixed chassis mount to the moving wheel hub. The spring is
//            modelled at unit height and scaled to the live strut length, so the
//            coils visibly bunch under compression (the "shock absorbers").
//   - lights: two headlight SpotLights (+ glowing lenses) for night driving, plus
//            brake / reverse / indicator lamps driven from vehicle telemetry.
//
// All visuals live under one root Group that the demo binds to the PhysX chassis
// actor, so the whole car follows the simulation.

#include "threepp/threepp.hpp"

#include "threepp/extras/physx/PhysxVehicleEngineDrive.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <vector>

namespace drive {

    using namespace threepp;

    class CarRig {
    public:
        struct Config {
            float bodyWidth = 1.9f;
            float bodyHeight = 1.3f;
            float bodyLength = 4.4f;
            float wheelRadius = 0.36f;
            float wheelHalfWidth = 0.16f;
            float trackWidth = 1.65f;
            float wheelbase = 2.7f;
            float suspensionAttachmentY = -0.35f;
            float suspensionTravelDist = 0.32f;
            Color paint = Color(0xb8412e);// warm rally red
        };

        explicit CarRig(const Config& cfg) : cfg_(cfg) {
            root_ = Group::create();
            buildBody();
            buildWheels();
            buildStruts();
            buildLights();
        }

        [[nodiscard]] std::shared_ptr<Group> root() const { return root_; }

        void setHeadlights(bool on) {
            headlightsOn_ = on;
            for (auto& sl : headlights_)
                if (sl) sl->intensity = on ? headlightIntensity_ : 0.f;
            if (headlampMat_) {
                headlampMat_->emissiveIntensity = on ? 6.f : 0.4f;
                headlampMat_->needsUpdate();
            }
        }
        [[nodiscard]] bool headlightsOn() const { return headlightsOn_; }

        // Update visuals from the vehicle. turnSignal: -1 left, 0 none, +1 right.
        void update(const PhysxVehicleEngineDrive& v, float dt,
                    float brakeCmd, int turnSignal) {

            // ── Wheels: copy the per-wheel local pose straight from the vehicle ──
            for (int i = 0; i < 4; ++i) {
                const auto wp = v.wheelLocalPose(i);
                wheelRigs_[i]->position.set(wp.p.x, wp.p.y, wp.p.z);
                wheelRigs_[i]->quaternion.set(wp.q.x, wp.q.y, wp.q.z, wp.q.w);
            }

            // ── Struts: stretch each coil-over from its chassis mount to the live
            //    hub position (translation only — the strut doesn't spin). ────────
            const Vector3 yAxis{0.f, 1.f, 0.f};
            for (int i = 0; i < 4; ++i) {
                const auto wp = v.wheelLocalPose(i);
                Vector3 hub(wp.p.x, wp.p.y, wp.p.z);
                Vector3 dir = hub.clone().sub(strutMount_[i]);
                float len = dir.length();
                if (len < 1e-4f) len = 1e-4f;
                dir.multiplyScalar(1.f / len);
                struts_[i]->position.copy(strutMount_[i]);
                struts_[i]->quaternion.setFromUnitVectors(yAxis, dir);
                struts_[i]->scale.set(1.f, len, 1.f);
            }

            // ── Lamps ────────────────────────────────────────────────────────
            const bool brakeOn = brakeCmd > 0.05f;
            const bool reverseOn = v.direction() == PhysxVehicleEngineDrive::Direction::Reverse;
            setEmissive(brakeMat_, brakeOn ? 6.f : (headlightsOn_ ? 1.4f : 0.25f));
            setEmissive(reverseMat_, reverseOn ? 5.f : 0.0f);

            // Indicators: ~1.5 Hz square-wave blink.
            blinkPhase_ += dt;
            if (blinkPhase_ > 1.f) blinkPhase_ -= 1.f;
            const bool blinkOn = blinkPhase_ < 0.33f;
            setEmissive(blinkerLMat_, (turnSignal < 0 && blinkOn) ? 6.f : 0.f);
            setEmissive(blinkerRMat_, (turnSignal > 0 && blinkOn) ? 6.f : 0.f);
        }

    private:
        // ── Body ──────────────────────────────────────────────────────────────
        void buildBody() {
            auto paintMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(cfg_.paint).metalness(0.55f).roughness(0.38f));
            auto glassMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color(0x10141b)).metalness(0.2f).roughness(0.08f));
            auto trimMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color(0x0a0a0a)).metalness(0.3f).roughness(0.6f));
            auto chromeMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color(0xc8ccd0)).metalness(0.9f).roughness(0.22f));

            const float W = cfg_.bodyWidth, H = cfg_.bodyHeight, L = cfg_.bodyLength;

            // Lower body: the main mass, sitting around the chassis centre.
            auto lower = Mesh::create(BoxGeometry::create(W, H * 0.5f, L * 0.96f), paintMat);
            lower->position.y = -0.05f;
            lower->castShadow = lower->receiveShadow = true;
            root_->add(lower);

            // Sill/rocker (darker) to break the slab and read as a separate volume.
            auto sill = Mesh::create(BoxGeometry::create(W * 1.01f, H * 0.18f, L * 0.7f), trimMat);
            sill->position.y = -H * 0.27f;
            root_->add(sill);

            // Bonnet + boot: thin slabs front and rear, a touch lower than the cabin.
            auto bonnet = Mesh::create(BoxGeometry::create(W * 0.96f, H * 0.16f, L * 0.30f), paintMat);
            bonnet->position.set(0.f, H * 0.22f, L * 0.30f);
            bonnet->castShadow = true;
            root_->add(bonnet);
            auto boot = Mesh::create(BoxGeometry::create(W * 0.96f, H * 0.16f, L * 0.22f), paintMat);
            boot->position.set(0.f, H * 0.22f, -L * 0.34f);
            boot->castShadow = true;
            root_->add(boot);

            // Greenhouse / cabin: a narrower, taller glass box set back a little.
            auto cabin = Mesh::create(BoxGeometry::create(W * 0.82f, H * 0.42f, L * 0.42f), glassMat);
            cabin->position.set(0.f, H * 0.42f, -L * 0.02f);
            cabin->castShadow = true;
            root_->add(cabin);
            // Roof cap (painted) on top of the glass.
            auto roof = Mesh::create(BoxGeometry::create(W * 0.78f, H * 0.06f, L * 0.36f), paintMat);
            roof->position.set(0.f, H * 0.63f, -L * 0.02f);
            root_->add(roof);

            // Bumpers.
            auto fBumper = Mesh::create(BoxGeometry::create(W * 0.98f, H * 0.16f, L * 0.06f), trimMat);
            fBumper->position.set(0.f, -H * 0.12f, L * 0.49f);
            root_->add(fBumper);
            auto rBumper = Mesh::create(BoxGeometry::create(W * 0.98f, H * 0.16f, L * 0.06f), trimMat);
            rBumper->position.set(0.f, -H * 0.12f, -L * 0.49f);
            root_->add(rBumper);

            // Front grille trim.
            auto grille = Mesh::create(BoxGeometry::create(W * 0.5f, H * 0.12f, 0.04f), chromeMat);
            grille->position.set(0.f, 0.0f, L * 0.5f);
            root_->add(grille);

            // ── Emissive lamp lenses ──
            const float lampZf = L * 0.49f;// front face
            const float lampZr = -L * 0.49f;// rear face
            const float lampX = W * 0.34f;

            headlampMat_ = makeEmissive(Color(0xfff4d6), 0.4f);
            addLens(headlampMat_, {+lampX, 0.05f, lampZf}, {0.34f, 0.20f, 0.05f});
            addLens(headlampMat_, {-lampX, 0.05f, lampZf}, {0.34f, 0.20f, 0.05f});

            brakeMat_ = makeEmissive(Color(0xff2a17), 0.25f);
            addLens(brakeMat_, {+lampX, 0.08f, lampZr}, {0.40f, 0.16f, 0.05f});
            addLens(brakeMat_, {-lampX, 0.08f, lampZr}, {0.40f, 0.16f, 0.05f});

            reverseMat_ = makeEmissive(Color(0xf2f5ff), 0.0f);
            addLens(reverseMat_, {+lampX * 0.45f, 0.05f, lampZr}, {0.16f, 0.10f, 0.05f});
            addLens(reverseMat_, {-lampX * 0.45f, 0.05f, lampZr}, {0.16f, 0.10f, 0.05f});

            blinkerLMat_ = makeEmissive(Color(0xff8a00), 0.0f);
            blinkerRMat_ = makeEmissive(Color(0xff8a00), 0.0f);
            // Left side (-X) front + rear; right side (+X) front + rear.
            addLens(blinkerLMat_, {-W * 0.46f, -0.02f, lampZf * 0.97f}, {0.06f, 0.10f, 0.14f});
            addLens(blinkerLMat_, {-W * 0.46f, -0.02f, lampZr * 0.97f}, {0.06f, 0.10f, 0.14f});
            addLens(blinkerRMat_, {+W * 0.46f, -0.02f, lampZf * 0.97f}, {0.06f, 0.10f, 0.14f});
            addLens(blinkerRMat_, {+W * 0.46f, -0.02f, lampZr * 0.97f}, {0.06f, 0.10f, 0.14f});
        }

        // ── Wheels ──────────────────────────────────────────────────────────────
        void buildWheels() {
            auto tyreMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color(0x0c0d0f)).metalness(0.f).roughness(0.92f));
            auto rimMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color(0xb9bdc4)).metalness(0.85f).roughness(0.30f));
            auto hubMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color(0x2a2d31)).metalness(0.7f).roughness(0.4f));

            const float r = cfg_.wheelRadius;
            const float hw = cfg_.wheelHalfWidth;

            // Cylinder axis is Y by default; rotate so the wheel spins about local X
            // (the vehicle's lateral axis), matching the wheel local pose convention.
            auto tyreGeo = CylinderGeometry::create(r, r, hw * 2.f, 28);
            tyreGeo->rotateZ(math::PI * 0.5f);
            auto rimGeo = CylinderGeometry::create(r * 0.6f, r * 0.6f, hw * 2.04f, 18);
            rimGeo->rotateZ(math::PI * 0.5f);
            auto hubGeo = CylinderGeometry::create(r * 0.22f, r * 0.22f, hw * 2.1f, 10);
            hubGeo->rotateZ(math::PI * 0.5f);

            for (int i = 0; i < 4; ++i) {
                auto rig = Group::create();
                auto tyre = Mesh::create(tyreGeo, tyreMat);
                tyre->castShadow = true;
                auto rim = Mesh::create(rimGeo, rimMat);
                auto hub = Mesh::create(hubGeo, hubMat);
                // A couple of spoke bars so the wheel spin reads clearly.
                for (int s = 0; s < 4; ++s) {
                    auto spoke = Mesh::create(BoxGeometry::create(hw * 1.6f, r * 1.05f, r * 0.10f), rimMat);
                    spoke->rotation.x = math::PI * 0.5f;// lay the bar across the wheel face
                    spoke->rotation.y = 0.f;
                    spoke->rotation.z = static_cast<float>(s) * math::PI * 0.25f;
                    rim->add(spoke);
                }
                rig->add(tyre);
                rig->add(rim);
                rig->add(hub);
                wheelRigs_[i] = rig;
                root_->add(rig);
            }
        }

        // ── Suspension struts (visible coil-overs) ───────────────────────────────
        void buildStruts() {
            auto springMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color(0xd24b2a)).metalness(0.6f).roughness(0.4f));
            auto rodMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color(0x9aa0a8)).metalness(0.9f).roughness(0.25f));

            const float halfTrack = cfg_.trackWidth * 0.5f;
            const float halfBase = cfg_.wheelbase * 0.5f;
            // Mount the top of each strut a little above the wheel's compressed rest
            // height, near the chassis — the spring tower.
            const float mountY = cfg_.suspensionAttachmentY + cfg_.suspensionTravelDist + 0.12f;
            const Vector3 mounts[4] = {
                    {+halfTrack, mountY, +halfBase},
                    {-halfTrack, mountY, +halfBase},
                    {+halfTrack, mountY, -halfBase},
                    {-halfTrack, mountY, -halfBase},
            };

            const float coilR = cfg_.wheelRadius * 0.32f;
            auto springGeo = makeCoilSpring(6.0f, coilR, coilR * 0.16f, 160, 6);
            auto rodGeo = CylinderGeometry::create(coilR * 0.28f, coilR * 0.28f, 1.f, 8);
            rodGeo->translate(0.f, 0.5f, 0.f);// base at y=0, tip at y=1 (matches spring)

            for (int i = 0; i < 4; ++i) {
                strutMount_[i] = mounts[i];
                auto strut = Group::create();
                strut->add(Mesh::create(springGeo, springMat));
                strut->add(Mesh::create(rodGeo, rodMat));
                struts_[i] = strut;
                root_->add(strut);
            }
        }

        // ── Lights ────────────────────────────────────────────────────────────
        void buildLights() {
            const float W = cfg_.bodyWidth, L = cfg_.bodyLength;
            const float lampX = W * 0.34f;
            const float lampZ = L * 0.49f;

            for (int i = 0; i < 2; ++i) {
                const float sx = (i == 0) ? +lampX : -lampX;
                // SpotLight(color, intensity, distance, angle, penumbra, decay).
                auto sl = SpotLight::create(Color(0xfff2d8), 0.f, 70.f,
                                            math::degToRad(34.f), 0.45f, 1.4f);
                sl->position.set(sx, 0.05f, lampZ);
                sl->castShadow = false;// keep cheap; PT lights the cone regardless
                // Aim forward (+Z) and slightly down via a target parented to the car.
                auto tgt = Object3D::create();
                tgt->position.set(sx, -0.5f, lampZ + 12.f);
                root_->add(tgt);
                sl->setTarget(*tgt);// target is owned by root_, outlives the light
                root_->add(sl);
                headlights_[i] = sl;
            }
            setHeadlights(false);
        }

        // ── small builders / helpers ─────────────────────────────────────────
        std::shared_ptr<MeshStandardMaterial> makeEmissive(const Color& c, float intensity) {
            auto m = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color(0x080808)).metalness(0.f).roughness(0.5f));
            m->emissive = c;
            m->emissiveIntensity = intensity;
            return m;
        }
        static void setEmissive(const std::shared_ptr<MeshStandardMaterial>& m, float intensity) {
            if (!m) return;
            if (std::abs(m->emissiveIntensity - intensity) > 1e-3f) {
                m->emissiveIntensity = intensity;
                m->needsUpdate();
            }
        }
        void addLens(const std::shared_ptr<MeshStandardMaterial>& mat,
                     const Vector3& pos, const Vector3& size) {
            auto lens = Mesh::create(BoxGeometry::create(size.x, size.y, size.z), mat);
            lens->position.copy(pos);
            root_->add(lens);
        }

        // Build a helical spring tube of unit height (y in [0,1]) so it can be scaled
        // to the live strut length — the coils bunch up as the strut compresses.
        static std::shared_ptr<BufferGeometry> makeCoilSpring(
                float turns, float coilRadius, float wireRadius,
                int tubularSegments, int radialSegments) {
            std::vector<float> pos, nrm;
            std::vector<unsigned int> idx;

            const float twoPi = 6.28318530718f;
            auto centre = [&](float t, Vector3& c, Vector3& tan) {
                const float ang = t * turns * twoPi;
                c.set(coilRadius * std::cos(ang), t, coilRadius * std::sin(ang));
                // dC/dt
                tan.set(-coilRadius * turns * twoPi * std::sin(ang), 1.f,
                        coilRadius * turns * twoPi * std::cos(ang));
                tan.normalize();
            };

            const Vector3 ref{0.f, 1.f, 0.f};
            Vector3 c{}, tan{}, n{}, b{};
            for (int i = 0; i <= tubularSegments; ++i) {
                const float t = static_cast<float>(i) / static_cast<float>(tubularSegments);
                centre(t, c, tan);
                // Orthonormal ring frame perpendicular to the tangent.
                n.copy(ref).cross(tan);
                if (n.length() < 1e-4f) n.set(1.f, 0.f, 0.f);
                n.normalize();
                b.copy(tan).cross(n).normalize();
                for (int j = 0; j <= radialSegments; ++j) {
                    const float a = static_cast<float>(j) / static_cast<float>(radialSegments) * twoPi;
                    const float ca = std::cos(a), sa = std::sin(a);
                    Vector3 off(n.x * ca + b.x * sa, n.y * ca + b.y * sa, n.z * ca + b.z * sa);
                    pos.push_back(c.x + off.x * wireRadius);
                    pos.push_back(c.y + off.y * wireRadius);
                    pos.push_back(c.z + off.z * wireRadius);
                    nrm.push_back(off.x);
                    nrm.push_back(off.y);
                    nrm.push_back(off.z);
                }
            }
            const int ring = radialSegments + 1;
            for (int i = 0; i < tubularSegments; ++i) {
                for (int j = 0; j < radialSegments; ++j) {
                    const auto a = static_cast<unsigned int>(i * ring + j);
                    const auto b2 = static_cast<unsigned int>((i + 1) * ring + j);
                    idx.push_back(a);
                    idx.push_back(b2);
                    idx.push_back(a + 1);
                    idx.push_back(a + 1);
                    idx.push_back(b2);
                    idx.push_back(b2 + 1);
                }
            }
            auto geo = BufferGeometry::create();
            geo->setIndex(idx);
            geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
            geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
            return geo;
        }

        Config cfg_;
        std::shared_ptr<Group> root_;
        std::array<std::shared_ptr<Group>, 4> wheelRigs_{};
        std::array<std::shared_ptr<Group>, 4> struts_{};
        std::array<Vector3, 4> strutMount_{};

        std::shared_ptr<MeshStandardMaterial> headlampMat_, brakeMat_, reverseMat_, blinkerLMat_, blinkerRMat_;
        std::array<std::shared_ptr<SpotLight>, 2> headlights_{};
        float headlightIntensity_ = 12.f;
        bool headlightsOn_ = false;
        float blinkPhase_ = 0.f;
    };

}// namespace drive

#endif//THREEPP_DRIVE_CARRIG_HPP
