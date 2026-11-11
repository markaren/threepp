// A small CPU cloth solver for flags, banners, sails and curtains.
//
//     auto geometry = PlaneGeometry::create(1.5f, 1.f, 30, 20);
//     ClothMesh flag(*geometry);
//     for (int i : flag.particlesWhere([](const Vector3& p) { return p.x < -0.74f; })) flag.cloth().pin(i);
//     flag.cloth().setWind({6.f, 0.f, 0.f});
//     canvas.animate([&] { flag.update(clock.getDelta()); renderer->render(*scene, *camera); });
//
// TWO CLASSES. Cloth is the solver: particles, triangles, pins, colliders and
// wind, std::vector and threepp math only. It knows nothing of geometries,
// renderers or a physics engine. ClothMesh binds a Cloth to a BufferGeometry:
// it welds the geometry's vertices into particles and writes positions and
// normals back each update. The geometry is a plain animated mesh to both
// renderers.
//
// METHOD: XPBD with small steps (Macklin et al. 2019): `substeps` substeps per
// step, one Gauss-Seidel pass over the constraints in each. Stretch is a
// distance constraint per unique triangle edge. Bending is a distance
// constraint between the two vertices opposite each interior edge. Compliance
// is in m/N, 0 = rigid; stiffness therefore does not depend on the substep
// count or the frame rate.
//
// WIND acts on each triangle from the air's velocity v relative to it, split
// across and along the sheet (n its normal, A its area, q = 0.5 * rho):
//
//   across: q * normalDrag * A * (v . n) |v . n|. The cross-flow law: the
//           normal force of a plate in separated flow goes as the square of
//           the velocity across it (Hoerner, Fluid-Dynamic Drag, ch. 3; G. I.
//           Taylor 1952 split the forces on long flexible bodies the same
//           way). normalDrag 1.2 is a square plate broadside on. A sheet
//           edge-on to the wind feels little; that is what lets a flag stream.
//   along:  q * tangentDrag * A * |v_t| v_t, skin friction. 0.015 counts both
//           faces of a rough fabric: about twice a smooth flat plate's
//           turbulent friction, Cf near 0.004 a side at Re 1e6.
//
// `gustiness` adds a deterministic variation of speed and direction that
// travels downwind across the cloth, which keeps a flag from settling into
// one shape.
//
// FLAPPING. A quasi-steady pressure on each triangle damps the waves a real
// flag flies in; nothing in it feeds them. The solver adds them as a FORCING,
// not as a flutter instability: a pressure across the sheet that travels
// downstream as a wave,
//
//   q * flapping * |v_t|^2 * min(1, s / sqrt(area)) * sin(k s - omega t),
//
// with v_t the flow along the sheet and s the distance downstream from the
// cloth's upstream edge. Its scales come from the flow and the cloth, not from
// the scene: crests travel at 0.6 times the mean flow speed U, and the
// wavelength is 1.1 times the square root of the cloth's area. Flags flap with
// bending waves that run from root to tip at roughly the flow speed, with a
// wavelength that scales with the flag (Raynaud & Mulleners, "On the wake and
// flapping dynamics of different aspect ratio flags", arXiv 2602.06237); 0.6
// leans toward the slower waves they find on low aspect ratio flags. A 1.1 x
// 0.8 m flag then flaps at about 0.6 U per metre: 3 Hz in 5 m/s, 6 Hz in
// 10 m/s, 9 Hz in 15 m/s (fL/U near 0.65, inside the 0.1 to 1 flags show).
// The amplitude grows from nothing at the upstream edge over the cloth's own
// size, and goes with the dynamic pressure of the flow ALONG the sheet, so a
// cloth hanging across a light wind is barely driven and one streaming in a
// fresh one is driven hard. `flapping` is the one calibrated number: that
// fraction of the dynamic pressure, 0.3, set so the fly of a metre-long flag
// swings a few tens of centimetres. What this is not: the flag does not choose
// its own frequency or wavelength, there is no onset speed (real flag fabric
// flutters in any wind that holds it out), and the cloth's wake and the
// sheltering of one part by another are not modelled.
//
// SPACE: the solver works in whatever space the positions are given in.
// ClothMesh uses the geometry's own space, so gravity, wind, pins and
// colliders are in the mesh's local frame. Keep the mesh unrotated and
// unscaled, or express those in its frame.
//
// LIMITS: no self-collision, no tearing, no collision against meshes (spheres,
// capsules and planes only). Topology is fixed at construction.
//
// DETERMINISM: single-threaded and order-fixed; the same binary, inputs and
// dt sequence give bit-identical positions.
//
// Header-only, dependency-free beyond threepp core.

#ifndef THREEPP_CLOTH_HPP
#define THREEPP_CLOTH_HPP

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/constants.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace threepp {

    class Cloth {

    public:
        struct Params {
            Vector3 gravity{0.f, -9.81f, 0.f};
            // Substeps per step(); each runs one pass over the constraints.
            int substeps = 12;
            // A step() longer than this is clamped (a hitch must not launch
            // the cloth).
            float maxStep = 1.f / 30.f;
            // Mass per area, kg/m2. Flag polyester is about 0.1-0.15.
            float arealDensity = 0.15f;
            // XPBD compliances, m/N. 0 = inextensible / as stiff as the
            // substep count allows.
            float stretchCompliance = 0.f;
            float bendCompliance = 0.5f;
            // Fraction of velocity lost per second to internal friction.
            float damping = 0.15f;
            // Air: density kg/m3, pressure coefficient across the sheet and
            // friction coefficient along it (see WIND).
            float airDensity = 1.2f;
            float normalDrag = 1.2f;
            float tangentDrag = 0.015f;
            // Amplitude of the flapping pressure wave, as a fraction of the
            // dynamic pressure of the flow along the sheet (see FLAPPING).
            float flapping = 0.3f;
            // 0 = steady wind; 1 = speed varies by about +-50 % and direction
            // by about +-25 degrees, in waves that travel downwind.
            float gustiness = 0.35f;
            // Tangential velocity kept on contact with a collider (1 = no
            // friction).
            float colliderSlip = 0.6f;
            // Colliders are padded by this much, so the rendered sheet does
            // not z-fight the surface it rests on.
            float thickness = 0.01f;
        };

        Params params;

        // positions: one per particle. indices: three per triangle.
        Cloth(std::vector<Vector3> positions, std::vector<unsigned int> indices, const Params& p = {})
            : params(p), x_(std::move(positions)), tris_(std::move(indices)) {

            if (tris_.size() % 3 != 0) throw std::invalid_argument("Cloth: index count is not a multiple of 3");
            for (const unsigned int i : tris_) {
                if (i >= x_.size()) throw std::invalid_argument("Cloth: index out of range");
            }

            const size_t n = x_.size();
            prev_ = x_;
            v_.assign(n, Vector3{});
            normals_.assign(n, Vector3{0.f, 0.f, 1.f});
            force_.assign(n, Vector3{});
            air_.assign(n, Vector3{});
            pinned_.assign(n, 0);
            pinFrom_ = x_;
            pinTo_ = x_;

            // Lumped mass: a third of each adjacent triangle.
            mass_.assign(n, 0.f);
            for (size_t t = 0; t + 2 < tris_.size(); t += 3) {
                const unsigned int a = tris_[t], b = tris_[t + 1], c = tris_[t + 2];
                const float area = 0.5f * Vector3().crossVectors(x_[b] - x_[a], x_[c] - x_[a]).length();
                restArea_ += area;
                const float m = params.arealDensity * area / 3.f;
                mass_[a] += m;
                mass_[b] += m;
                mass_[c] += m;
            }
            invMass_.resize(n);
            for (size_t i = 0; i < n; ++i) invMass_[i] = mass_[i] > 0.f ? 1.f / mass_[i] : 0.f;

            buildConstraints();
            computeNormals();
        }

        [[nodiscard]] size_t particleCount() const { return x_.size(); }
        [[nodiscard]] size_t triangleCount() const { return tris_.size() / 3; }
        [[nodiscard]] size_t stretchConstraintCount() const { return stretch_.size(); }
        [[nodiscard]] size_t bendConstraintCount() const { return bend_.size(); }

        [[nodiscard]] const std::vector<Vector3>& positions() const { return x_; }
        [[nodiscard]] const std::vector<Vector3>& velocities() const { return v_; }
        // Area-weighted vertex normals, refreshed by step().
        [[nodiscard]] const std::vector<Vector3>& normals() const { return normals_; }
        [[nodiscard]] const std::vector<unsigned int>& indices() const { return tris_; }

        // ── Pins ─────────────────────────────────────────────────────────

        // Holds a particle where it is now.
        void pin(size_t i) {
            pinned_.at(i) = 1;
            pinFrom_[i] = pinTo_[i] = x_[i];
            v_[i].set(0.f, 0.f, 0.f);
        }

        // Holds a particle at `target`. Called every frame with a moving
        // target, the particle is carried there over the next step()'s
        // substeps, so a moving pole does not jerk the cloth.
        void pin(size_t i, const Vector3& target) {
            if (!pinned_.at(i)) {
                pinned_[i] = 1;
                pinFrom_[i] = x_[i];
            }
            pinTo_[i] = target;
        }

        void unpin(size_t i) { pinned_.at(i) = 0; }

        [[nodiscard]] bool isPinned(size_t i) const { return pinned_.at(i) != 0; }

        // ── Wind ─────────────────────────────────────────────────────────

        // Air velocity, m/s, in the solver's space.
        void setWind(const Vector3& w) { wind_ = w; }
        [[nodiscard]] const Vector3& wind() const { return wind_; }

        // Replaces the built-in wind (and its gusts) by a field: air velocity
        // at a position and time, sampled at every particle once per step().
        // Pass nullptr to go back to setWind().
        void setWindField(std::function<Vector3(const Vector3& position, float time)> f) { windField_ = std::move(f); }

        // ── Colliders ────────────────────────────────────────────────────
        // Each add returns a handle for the matching set, so a collider can
        // follow something that moves.

        size_t addSphere(const Vector3& center, float radius) {
            capsules_.push_back({center, center, radius});
            return capsules_.size() - 1;
        }

        size_t addCapsule(const Vector3& a, const Vector3& b, float radius) {
            capsules_.push_back({a, b, radius});
            return capsules_.size() - 1;
        }

        void setSphere(size_t handle, const Vector3& center, float radius) {
            capsules_.at(handle) = {center, center, radius};
        }

        void setCapsule(size_t handle, const Vector3& a, const Vector3& b, float radius) {
            capsules_.at(handle) = {a, b, radius};
        }

        // The half-space behind a plane: the cloth stays on the side `normal`
        // points to.
        size_t addPlane(const Vector3& point, const Vector3& normal) {
            Vector3 n = normal;
            n.normalize();
            planes_.push_back({n, n.dot(point)});
            return planes_.size() - 1;
        }

        void setPlane(size_t handle, const Vector3& point, const Vector3& normal) {
            Vector3 n = normal;
            n.normalize();
            planes_.at(handle) = {n, n.dot(point)};
        }

        void clearColliders() {
            capsules_.clear();
            planes_.clear();
        }

        // ── Stepping ─────────────────────────────────────────────────────

        [[nodiscard]] float time() const { return time_; }

        // Advances the cloth by dt seconds (clamped to params.maxStep).
        void step(float dt) {

            dt = std::min(dt, params.maxStep);
            if (!(dt > 0.f)) return;

            const int substeps = std::max(1, params.substeps);
            const float h = dt / static_cast<float>(substeps);
            const float keep = std::exp(-std::max(0.f, params.damping) * h);
            const size_t n = x_.size();

            // The air is sampled once per step, at the particles: it changes
            // over metres and tenths of a second, the substeps over
            // millimetres and milliseconds. So is the flow that sets the
            // flapping wave's rate.
            sampleAir();
            sampleFlow();

            for (int s = 0; s < substeps; ++s) {

                const float a = static_cast<float>(s + 1) / static_cast<float>(substeps);

                flapPhase_ += flapOmega_ * h;
                if (flapPhase_ > math::TWO_PI) flapPhase_ -= math::TWO_PI;
                accumulateAir();

                for (size_t i = 0; i < n; ++i) {
                    prev_[i] = x_[i];
                    if (pinned_[i]) {
                        x_[i].lerpVectors(pinFrom_[i], pinTo_[i], a);
                        continue;
                    }
                    if (invMass_[i] == 0.f) continue;
                    v_[i].addScaledVector(params.gravity, h);
                    v_[i].addScaledVector(force_[i], h * invMass_[i]);
                    x_[i].addScaledVector(v_[i], h);
                }

                const float h2 = h * h;
                solveDistance(stretch_, params.stretchCompliance / h2);
                solveDistance(bend_, params.bendCompliance / h2);
                collide();

                for (size_t i = 0; i < n; ++i) {
                    v_[i].subVectors(x_[i], prev_[i]).divideScalar(h);
                    if (!pinned_[i]) v_[i].multiplyScalar(keep);
                }

                time_ += h;
            }

            for (size_t i = 0; i < n; ++i) {
                if (pinned_[i]) pinFrom_[i] = pinTo_[i];
            }

            computeNormals();
        }

    private:
        struct Distance {
            unsigned int a, b;
            float rest;
        };

        struct Capsule {
            Vector3 a, b;
            float radius;
        };

        struct Plane {
            Vector3 n;
            float d;
        };

        std::vector<Vector3> x_, prev_, v_, normals_, force_, air_, pinFrom_, pinTo_;
        std::vector<float> mass_, invMass_;
        std::vector<std::uint8_t> pinned_;
        std::vector<unsigned int> tris_;
        std::vector<Distance> stretch_, bend_;
        std::vector<Capsule> capsules_;
        std::vector<Plane> planes_;
        Vector3 wind_;
        std::function<Vector3(const Vector3&, float)> windField_;
        float time_ = 0.f;

        // The flapping wave's speed as a fraction of the flow's, and its
        // wavelength over the square root of the cloth's area (FLAPPING).
        static constexpr float kWaveSpeed = 0.6f;
        static constexpr float kWaveLength = 1.1f;

        float restArea_ = 0.f;
        // Per step: the mean flow over the cloth relative to it, and the
        // cloth's upstream edge along it.
        Vector3 flowDir_{1.f, 0.f, 0.f};
        float flowSpeed_ = 0.f, flowEdge_ = 0.f;
        // The flapping wave: phase (advanced per substep), rate, wavenumber
        // and the inverse of the length its amplitude grows over.
        float flapPhase_ = 0.f, flapOmega_ = 0.f, flapK_ = 0.f, flapGrow_ = 0.f;

        [[nodiscard]] float w(size_t i) const { return pinned_[i] ? 0.f : invMass_[i]; }

        void buildConstraints() {

            // Each undirected edge, with the vertex opposite it in the first
            // triangle that used it. A second triangle on the same edge gives
            // the bend pair; a third (non-manifold) is ignored for bending.
            struct EdgeInfo {
                unsigned int opposite;
                bool paired;
            };
            std::unordered_map<std::uint64_t, EdgeInfo> edges;
            edges.reserve(tris_.size() * 2);

            const auto key = [](unsigned int a, unsigned int b) {
                if (a > b) std::swap(a, b);
                return (static_cast<std::uint64_t>(a) << 32) | b;
            };

            for (size_t t = 0; t + 2 < tris_.size(); t += 3) {
                const unsigned int tri[3] = {tris_[t], tris_[t + 1], tris_[t + 2]};
                for (int e = 0; e < 3; ++e) {
                    const unsigned int a = tri[e], b = tri[(e + 1) % 3], c = tri[(e + 2) % 3];
                    if (a == b) continue;
                    const auto k = key(a, b);
                    const auto it = edges.find(k);
                    if (it == edges.end()) {
                        edges.emplace(k, EdgeInfo{c, false});
                        stretch_.push_back({a, b, x_[a].distanceTo(x_[b])});
                    } else if (!it->second.paired && it->second.opposite != c) {
                        it->second.paired = true;
                        const unsigned int o = it->second.opposite;
                        bend_.push_back({o, c, x_[o].distanceTo(x_[c])});
                    }
                }
            }
        }

        void solveDistance(const std::vector<Distance>& constraints, float alpha) {

            for (const Distance& c : constraints) {
                const float wa = w(c.a), wb = w(c.b);
                const float wsum = wa + wb;
                if (wsum == 0.f) continue;
                Vector3 d;
                d.subVectors(x_[c.b], x_[c.a]);
                const float len = d.length();
                if (len < 1e-9f) continue;
                const float lambda = (len - c.rest) / ((wsum + alpha) * len);
                x_[c.a].addScaledVector(d, wa * lambda);
                x_[c.b].addScaledVector(d, -wb * lambda);
            }
        }

        // The mean air over the cloth relative to its particles, the cloth's
        // upstream edge along it, and from them the flapping wave's rate.
        void sampleFlow() {

            Vector3 mean;
            for (size_t i = 0; i < x_.size(); ++i) mean.add(air_[i]).sub(v_[i]);
            if (!x_.empty()) mean.divideScalar(static_cast<float>(x_.size()));
            flowSpeed_ = mean.length();
            if (flowSpeed_ > 1e-4f) flowDir_.copy(mean).divideScalar(flowSpeed_);

            flowEdge_ = std::numeric_limits<float>::max();
            for (const Vector3& p : x_) flowEdge_ = std::min(flowEdge_, p.dot(flowDir_));

            const float size = std::sqrt(std::max(restArea_, 1e-8f));
            flapK_ = math::TWO_PI / (kWaveLength * size);
            flapOmega_ = flapK_ * kWaveSpeed * flowSpeed_;
            flapGrow_ = 1.f / size;
        }

        void sampleAir() {

            if (windField_) {
                for (size_t i = 0; i < x_.size(); ++i) air_[i] = windField_(x_[i], time_);
                return;
            }

            const float speed = wind_.length();
            const float g = params.gustiness;
            if (speed < 1e-6f || g <= 0.f) {
                std::fill(air_.begin(), air_.end(), wind_);
                return;
            }

            // Phases advance with distance travelled downwind, so the pattern
            // moves across the cloth at the wind's own speed.
            Vector3 dir = wind_;
            dir.divideScalar(speed);
            Vector3 side;
            side.crossVectors(dir, std::abs(dir.y) < 0.9f ? Vector3{0.f, 1.f, 0.f} : Vector3{1.f, 0.f, 0.f});
            side.normalize();
            Vector3 up;
            up.crossVectors(side, dir);

            const float t = time_;
            const float slow = 0.20f * std::sin(0.31f * speed * t + 0.4f);

            for (size_t i = 0; i < x_.size(); ++i) {
                const float along = x_[i].dot(dir) - speed * t;
                const float across = x_[i].dot(side);

                const float s = 0.50f * std::sin(0.9f * along + 0.6f * across) +
                                0.30f * std::sin(2.3f * along - 1.1f * across + 1.7f) + slow;
                const float yaw = 0.45f * std::sin(1.4f * along + 2.1f) + 0.25f * std::sin(3.7f * along + 1.3f * across);
                const float pitch = 0.30f * std::sin(1.9f * along + 0.8f * across + 4.2f);

                Vector3& out = air_[i];
                out.copy(dir);
                out.addScaledVector(side, g * yaw);
                out.addScaledVector(up, g * pitch);
                out.multiplyScalar(speed * (1.f + g * s));
            }
        }

        void accumulateAir() {

            std::fill(force_.begin(), force_.end(), Vector3{});
            if (params.normalDrag <= 0.f && params.tangentDrag <= 0.f && params.flapping <= 0.f) return;

            const float q = 0.5f * params.airDensity;
            Vector3 e1, e2, n, rel, tangential, f, centre;

            for (size_t tri = 0; tri + 2 < tris_.size(); tri += 3) {
                const unsigned int a = tris_[tri], b = tris_[tri + 1], c = tris_[tri + 2];
                e1.subVectors(x_[b], x_[a]);
                e2.subVectors(x_[c], x_[a]);
                n.crossVectors(e1, e2);
                const float twiceArea = n.length();
                if (twiceArea < 1e-12f) continue;
                n.divideScalar(twiceArea);
                const float area = 0.5f * twiceArea;

                rel.copy(air_[a]).add(air_[b]).add(air_[c]);
                rel.sub(v_[a]).sub(v_[b]).sub(v_[c]);
                rel.divideScalar(3.f);

                const float vn = rel.dot(n);
                tangential.copy(rel).addScaledVector(n, -vn);
                const float vt2 = tangential.lengthSq();

                // The flapping wave (see FLAPPING).
                centre.copy(x_[a]).add(x_[b]).add(x_[c]).divideScalar(3.f);
                const float s = centre.dot(flowDir_) - flowEdge_;
                const float flap = params.flapping * q * vt2 * std::min(1.f, s * flapGrow_) * std::sin(flapK_ * s - flapPhase_);

                f.copy(n).multiplyScalar(area * (q * params.normalDrag * vn * std::abs(vn) + flap));
                f.addScaledVector(tangential, q * params.tangentDrag * area * std::sqrt(vt2));
                f.divideScalar(3.f);

                force_[a].add(f);
                force_[b].add(f);
                force_[c].add(f);
            }
        }

        void collide() {

            if (capsules_.empty() && planes_.empty()) return;

            const float pad = params.thickness;
            const float slip = std::clamp(params.colliderSlip, 0.f, 1.f);
            Vector3 ab, closest, d, normal;

            for (size_t i = 0; i < x_.size(); ++i) {
                if (pinned_[i] || invMass_[i] == 0.f) continue;
                Vector3& p = x_[i];

                for (const Capsule& c : capsules_) {
                    ab.subVectors(c.b, c.a);
                    const float len2 = ab.lengthSq();
                    const float u = len2 > 0.f ? std::clamp(Vector3().subVectors(p, c.a).dot(ab) / len2, 0.f, 1.f) : 0.f;
                    closest.copy(c.a).addScaledVector(ab, u);
                    d.subVectors(p, closest);
                    const float r = c.radius + pad;
                    const float dist2 = d.lengthSq();
                    if (dist2 >= r * r) continue;
                    const float dist = std::sqrt(dist2);
                    if (dist > 1e-9f) normal.copy(d).divideScalar(dist);
                    else
                        normal.set(0.f, 1.f, 0.f);
                    p.copy(closest).addScaledVector(normal, r);
                    applyFriction(i, normal, slip);
                }

                for (const Plane& pl : planes_) {
                    const float depth = pl.d + pad - pl.n.dot(p);
                    if (depth <= 0.f) continue;
                    p.addScaledVector(pl.n, depth);
                    applyFriction(i, pl.n, slip);
                }
            }
        }

        // Shortens this substep's motion along the contact surface.
        void applyFriction(size_t i, const Vector3& normal, float slip) {
            if (slip >= 1.f) return;
            Vector3 motion;
            motion.subVectors(x_[i], prev_[i]);
            const float mn = motion.dot(normal);
            Vector3 tangential = motion;
            tangential.addScaledVector(normal, -mn);
            x_[i].addScaledVector(tangential, slip - 1.f);
        }

        void computeNormals() {

            std::fill(normals_.begin(), normals_.end(), Vector3{});
            Vector3 e1, e2, n;
            for (size_t t = 0; t + 2 < tris_.size(); t += 3) {
                const unsigned int a = tris_[t], b = tris_[t + 1], c = tris_[t + 2];
                e1.subVectors(x_[b], x_[a]);
                e2.subVectors(x_[c], x_[a]);
                n.crossVectors(e1, e2);// length = 2 * area: the weight
                normals_[a].add(n);
                normals_[b].add(n);
                normals_[c].add(n);
            }
            for (Vector3& nn : normals_) {
                const float len = nn.length();
                if (len > 1e-12f) nn.divideScalar(len);
                else
                    nn.set(0.f, 0.f, 1.f);
            }
        }
    };


    // A Cloth bound to a BufferGeometry. Vertices that share a position (a UV
    // seam, split normals) become one particle, so the sheet does not come
    // apart along them. The geometry needs a position attribute and triangles
    // (indexed or not); a normal attribute is added if it has none.
    class ClothMesh {

    public:
        explicit ClothMesh(BufferGeometry& geometry, const Cloth::Params& params = {}, float weldDistance = 1e-5f)
            : geometry_(&geometry) {

            position_ = geometry.getAttribute<float>("position");
            if (!position_) throw std::invalid_argument("ClothMesh: geometry has no position attribute");

            const size_t vertexCount = static_cast<size_t>(position_->count());
            const auto& pos = position_->array();

            // Weld by quantised position.
            std::vector<Vector3> particles;
            vertexToParticle_.resize(vertexCount);
            {
                struct KeyHash {
                    size_t operator()(const std::array<std::int64_t, 3>& k) const {
                        std::uint64_t h = 1469598103934665603ull;
                        for (const std::int64_t v : k) h = (h ^ static_cast<std::uint64_t>(v)) * 1099511628211ull;
                        return static_cast<size_t>(h);
                    }
                };
                std::unordered_map<std::array<std::int64_t, 3>, unsigned int, KeyHash> seen;
                seen.reserve(vertexCount);
                const float inv = 1.f / std::max(weldDistance, 1e-12f);
                for (size_t i = 0; i < vertexCount; ++i) {
                    const Vector3 p{pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]};
                    const std::array<std::int64_t, 3> k{std::llround(p.x * inv), std::llround(p.y * inv), std::llround(p.z * inv)};
                    const auto [it, inserted] = seen.emplace(k, static_cast<unsigned int>(particles.size()));
                    if (inserted) particles.push_back(p);
                    vertexToParticle_[i] = it->second;
                }
            }

            std::vector<unsigned int> indices;
            if (const auto* index = geometry.getIndex()) {
                const auto& src = index->array();
                indices.reserve(src.size());
                for (const auto v : src) indices.push_back(vertexToParticle_[static_cast<size_t>(v)]);
            } else {
                indices.reserve(vertexCount);
                for (size_t i = 0; i < vertexCount; ++i) indices.push_back(vertexToParticle_[i]);
            }
            indices.resize(indices.size() - indices.size() % 3);

            cloth_ = std::make_unique<Cloth>(std::move(particles), std::move(indices), params);

            normal_ = geometry.getAttribute<float>("normal");
            if (!normal_) {
                geometry.setAttribute("normal", FloatBufferAttribute::create(std::vector<float>(vertexCount * 3), 3));
                normal_ = geometry.getAttribute<float>("normal");
            }
            // Set once, before the first upload: the GL backend reads the
            // usage hint when it creates the buffer.
            position_->setUsage(DrawUsage::Dynamic);
            normal_->setUsage(DrawUsage::Dynamic);

            write();
        }

        [[nodiscard]] Cloth& cloth() { return *cloth_; }
        [[nodiscard]] const Cloth& cloth() const { return *cloth_; }

        // The particle a geometry vertex was welded into.
        [[nodiscard]] unsigned int particleOf(size_t vertex) const { return vertexToParticle_.at(vertex); }

        // Particles whose position satisfies a predicate, for pinning an edge
        // or a corner without knowing the vertex layout.
        [[nodiscard]] std::vector<size_t> particlesWhere(const std::function<bool(const Vector3&)>& predicate) const {
            std::vector<size_t> out;
            const auto& x = cloth_->positions();
            for (size_t i = 0; i < x.size(); ++i) {
                if (predicate(x[i])) out.push_back(i);
            }
            return out;
        }

        // Steps the cloth and writes the geometry.
        void update(float dt) {
            cloth_->step(dt);
            write();
        }

        // Writes the cloth's positions and normals into the geometry without
        // stepping (after moving particles by hand, say).
        void write() {
            const auto& x = cloth_->positions();
            const auto& n = cloth_->normals();
            auto& pos = position_->array();
            auto& nrm = normal_->array();
            for (size_t i = 0; i < vertexToParticle_.size(); ++i) {
                const unsigned int p = vertexToParticle_[i];
                pos[i * 3] = x[p].x;
                pos[i * 3 + 1] = x[p].y;
                pos[i * 3 + 2] = x[p].z;
                nrm[i * 3] = n[p].x;
                nrm[i * 3 + 1] = n[p].y;
                nrm[i * 3 + 2] = n[p].z;
            }
            position_->needsUpdate();
            normal_->needsUpdate();
            // The bounds follow the cloth, so frustum culling and the ray
            // tracer's structures see where it is now.
            geometry_->computeBoundingBox();
            geometry_->computeBoundingSphere();
        }

    private:
        BufferGeometry* geometry_;
        TypedBufferAttribute<float>* position_ = nullptr;
        TypedBufferAttribute<float>* normal_ = nullptr;
        std::vector<unsigned int> vertexToParticle_;
        std::unique_ptr<Cloth> cloth_;
    };

}// namespace threepp

#endif//THREEPP_CLOTH_HPP
