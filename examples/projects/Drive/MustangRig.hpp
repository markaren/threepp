
#ifndef THREEPP_DRIVE_MUSTANGRIG_HPP
#define THREEPP_DRIVE_MUSTANGRIG_HPP

// A car rig built around an imported glTF/GLB model (a 1967 Ford Mustang Shelby
// GT500) instead of hand-rolled primitives. It mirrors CarRig's public surface
// — root(), setHeadlights()/headlightsOn(), update(vehicle, dt, brake, signal) —
// so the Drive demo can swap one for the other.
//
// The wrinkle vs. CarRig is that nothing about the model's dimensions is known
// ahead of time, and the PhysX vehicle needs those numbers (track, wheelbase,
// wheel radius, chassis extents) to build its collider and suspension. So the
// flow is two-phase:
//
//   1. MustangRig::measure(model)  — walk the loaded scene, find the four tyres,
//      and read the real geometry out of their world-space transforms. Returns
//      a Measurements the demo feeds straight into PhysxVehicleEngineDrive.
//   2. MustangRig(model, meas, …)  — clone each corner's tyre + rim into a wheel
//      group driven every frame from vehicle.wheelLocalPose(i) (the originals in
//      the body are hidden), park the body on a pivot so its centre of mass sits
//      on the chassis actor origin, and wire up headlight spots + emissive brake
//      / headlamp lenses.
//
// Everything is measured from the model, so a differently-scaled or differently-
// centred export still lands on its wheels — there are no magic constants baked
// against this one file.

#include "threepp/threepp.hpp"

#include "threepp/extras/physx/PhysxVehicleEngineDrive.hpp"
#include "threepp/materials/MeshPhysicalMaterial.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace drive {

    using namespace threepp;

    class MustangRig {
    public:
        // Object3D layer the glass is tagged onto. On Vulkan the demo passes this
        // to renderer.setOverlayLayer() so the windows draw as a post-shade tint
        // over the full-quality image instead of re-tracing the scene behind them.
        // Harmless on GL/WebGPU (the camera still renders layer 0 as normal).
        static constexpr unsigned kOverlayLayer = 1;

        // Real-world geometry read off the loaded model, in the model's own space.
        struct Measurements {
            Vector3 chassisCenter;   // point that should map to the chassis actor origin
            float chassisWidth = 1.9f;
            float chassisHeight = 1.3f;
            float chassisLength = 4.6f;
            float wheelRadius = 0.35f;
            float wheelHalfWidth = 0.16f;
            float trackWidth = 1.6f;
            float wheelbase = 2.7f;
            // Wheel (axle) centres in model space, indexed to the PhysX wheel order:
            // 0 front-right, 1 front-left, 2 rear-right, 3 rear-left.
            std::array<Vector3, 4> wheelCenter{};
            // Axle height relative to chassisCenter (chassis-frame Y at the wheels).
            float wheelCenterYRel = -0.45f;
            bool valid = false;
        };

        // ── Phase 1: measure the model so the vehicle can be configured ──────────
        static Measurements measure(Object3D& model) {
            model.updateMatrixWorld(true);

            Measurements m;

            Box3 full;
            full.setFromObject(model);
            Vector3 fullCenter, fullSize;
            full.getCenter(fullCenter);
            full.getSize(fullSize);
            m.chassisWidth = fullSize.x;
            m.chassisHeight = fullSize.y;
            m.chassisLength = fullSize.z;

            // Collect the four tyres: axle centre (node origin) + radial size.
            std::vector<Vector3> tyrePos;
            std::vector<Vector3> tyreSize;
            model.traverseType<Mesh>([&](Mesh& mesh) {
                if (!isTyre(mesh.name)) return;
                Vector3 p;
                mesh.getWorldPosition(p);
                Box3 b;
                b.setFromObject(mesh);
                Vector3 s;
                b.getSize(s);
                tyrePos.push_back(p);
                tyreSize.push_back(s);
            });

            if (tyrePos.size() < 4) {
                // Degenerate export: fall back to sensible defaults so the demo
                // still runs (the body will simply sit on invisible wheels).
                m.chassisCenter = fullCenter;
                return m;
            }

            // Chassis centre: symmetric in X, wheel-centred in Z, body-centred in Y.
            float meanZ = 0.f, meanY = 0.f;
            for (const auto& p : tyrePos) {
                meanZ += p.z;
                meanY += p.y;
            }
            meanZ /= static_cast<float>(tyrePos.size());
            meanY /= static_cast<float>(tyrePos.size());
            m.chassisCenter = {fullCenter.x, fullCenter.y, meanZ};

            // Sort the four tyres into FR / FL / RR / RL about the chassis centre.
            for (const auto& p : tyrePos) {
                const bool front = p.z > m.chassisCenter.z;
                const bool right = p.x > m.chassisCenter.x;
                const int idx = (front ? 0 : 2) + (right ? 0 : 1);
                m.wheelCenter[static_cast<size_t>(idx)] = p;
            }

            // Track = mean of front & rear track; wheelbase = front axle Z − rear axle Z.
            const float frontTrack = std::abs(m.wheelCenter[0].x - m.wheelCenter[1].x);
            const float rearTrack = std::abs(m.wheelCenter[2].x - m.wheelCenter[3].x);
            m.trackWidth = 0.5f * (frontTrack + rearTrack);
            const float frontZ = 0.5f * (m.wheelCenter[0].z + m.wheelCenter[1].z);
            const float rearZ = 0.5f * (m.wheelCenter[2].z + m.wheelCenter[3].z);
            m.wheelbase = std::abs(frontZ - rearZ);

            // Radius / half-width from the tyre bounding boxes (axle along X).
            float radius = 0.f, halfWidth = 0.f;
            for (const auto& s : tyreSize) {
                radius += 0.25f * (s.y + s.z);// mean of the two radial half-extents
                halfWidth += 0.5f * s.x;
            }
            m.wheelRadius = radius / static_cast<float>(tyreSize.size());
            m.wheelHalfWidth = halfWidth / static_cast<float>(tyreSize.size());

            m.wheelCenterYRel = meanY - m.chassisCenter.y;
            m.valid = true;
            return m;
        }

        // ── Phase 2: build the visual rig around the loaded model ────────────────
        MustangRig(std::shared_ptr<Group> model, const Measurements& meas)
            : meas_(meas) {

            root_ = Group::create();

            // Wheel groups first, parked at rest so the very first frame (before
            // update()) already looks right. They live directly under root_ and
            // are driven from wheelLocalPose — NOT under the body pivot.
            for (int i = 0; i < 4; ++i) {
                wheelGroups_[i] = Group::create();
                wheelGroups_[i]->position.copy(meas.wheelCenter[i]).sub(meas.chassisCenter);
                root_->add(wheelGroups_[i]);
            }

            // Walk the model while it is still in its own (measured) space: clone
            // wheel meshes into their groups (hiding the originals) and capture the
            // light materials / lens positions. Doing this before the body pivot is
            // applied keeps every world position consistent with `meas`.
            model->updateMatrixWorld(true);
            model->traverseType<Mesh>([&](Mesh& mesh) {
                mesh.castShadow = true;
                tameGlass(mesh);

                if (isTyre(mesh.name) || isRim(mesh.name)) {
                    adoptWheelMesh(mesh);
                    return;
                }
                if (auto* mat = mesh.materialAs<MeshStandardMaterial>()) {
                    if (isBrakeMat(mat->name)) brakeMats_.push_back(mat);
                    if (isHeadlampMat(mat->name)) headlampMats_.push_back(mat);
                }
                if (isHeadlightLens(mesh.name)) {
                    // Use the mesh's world BBOX CENTRE, not getWorldPosition():
                    // the body meshes carry baked geometry under identity nodes,
                    // so getWorldPosition() returns the model origin for every one
                    // of them — which would stack both headlights at the chassis
                    // centre. The bbox centre is the actual lamp location.
                    Box3 b;
                    b.setFromObject(mesh);
                    Vector3 c;
                    b.getCenter(c);
                    headlightLensWorld_.push_back(c);
                }
            });

            buildHeadlights();

            // Body: park the whole model on a pivot so its chassis centre lands on
            // the root (= chassis actor) origin. The hidden wheel originals ride
            // along with it; the visible clones stay in the wheel groups.
            bodyPivot_ = Group::create();
            bodyPivot_->position.copy(meas.chassisCenter).negate();
            root_->add(bodyPivot_);
            bodyPivot_->add(model);

            setHeadlights(false);
            applyBrake(false);
        }

        [[nodiscard]] std::shared_ptr<Group> root() const { return root_; }

        void setHeadlights(bool on) {
            headlightsOn_ = on;
            for (auto& sl : headlights_)
                if (sl) sl->intensity = on ? headlightIntensity_ : 0.f;
            for (auto* m : headlampMats_) setEmissive(m, on ? 5.f : 0.4f);
        }
        [[nodiscard]] bool headlightsOn() const { return headlightsOn_; }

        // The two headlight spots (may contain nulls if the model had no lens
        // meshes). Exposed so the demo can attach SpotLightHelpers to inspect
        // the beam placement / angle.
        [[nodiscard]] const std::array<std::shared_ptr<SpotLight>, 2>& headlights() const {
            return headlights_;
        }

        // Update visuals from the vehicle. turnSignal is accepted for API parity
        // with CarRig; this model has no dedicated indicator geometry.
        void update(const PhysxVehicleEngineDrive& v, float /*dt*/,
                    float brakeCmd, int /*turnSignal*/) {
            for (int i = 0; i < 4; ++i) {
                const auto wp = v.wheelLocalPose(i);
                wheelGroups_[i]->position.set(wp.p.x, wp.p.y, wp.p.z);
                wheelGroups_[i]->quaternion.set(wp.q.x, wp.q.y, wp.q.z, wp.q.w);
            }
            applyBrake(brakeCmd > 0.05f);
        }

    private:
        // ── Wheel adoption: clone the rendered mesh into its wheel group ─────────
        void adoptWheelMesh(Mesh& mesh) {
            Vector3 wp, ws;
            Quaternion wq;
            mesh.getWorldPosition(wp);
            mesh.getWorldQuaternion(wq);
            mesh.getWorldScale(ws);

            const int i = nearestCorner(wp);
            auto clone = mesh.clone<Mesh>(false);
            clone->position.copy(wp).sub(meas_.wheelCenter[i]);// offset from the axle
            clone->quaternion.copy(wq);
            clone->scale.copy(ws);
            clone->castShadow = true;
            clone->visible = true;
            wheelGroups_[static_cast<size_t>(i)]->add(clone);

            mesh.visible = false;// hide the static original that lives in the body
        }

        int nearestCorner(const Vector3& worldPos) const {
            int best = 0;
            float bestD = std::numeric_limits<float>::max();
            for (int i = 0; i < 4; ++i) {
                const float d = worldPos.distanceToSquared(meas_.wheelCenter[i]);
                if (d < bestD) {
                    bestD = d;
                    best = i;
                }
            }
            return best;
        }

        // ── Headlights: a spot per lens, aimed DOWN at the road ahead ────────────
        void buildHeadlights() {
            if (headlightLensWorld_.empty()) return;
            // Road surface in chassis-local Y (a touch below the wheel contact so
            // the beam clearly rakes the tarmac rather than the horizon).
            const float roadY = meas_.wheelCenterYRel - meas_.wheelRadius - 0.05f;
            const size_t n = std::min<size_t>(2, headlightLensWorld_.size());
            for (size_t i = 0; i < n; ++i) {
                Vector3 local = headlightLensWorld_[i].clone().sub(meas_.chassisCenter);
                // Narrower cone (a real low-beam is tight); aim it at a point on
                // the road ~9 m ahead so the whole cone lands on the tarmac
                // instead of half of it spilling into the sky.
                auto sl = SpotLight::create(Color(0xfff2d8), 0.f, 70.f,
                                            math::degToRad(24.f), 0.35f, 0.4f);
                sl->position.copy(local);
                sl->castShadow = false;
                auto tgt = Object3D::create();
                tgt->position.set(local.x, roadY, local.z + 9.f);
                root_->add(tgt);
                sl->setTarget(*tgt);
                root_->add(sl);
                headlights_[i] = sl;
            }
        }

        // The car windows import as refractive glass (transmission, ior 1.5).
        // Every renderer's see-through path has a problem with them:
        //   • Vulkan's deferred glass BSDF sprays the bright HDRI sun into
        //     per-facet dots on the big flat panels, and everything seen through
        //     it is a RE-TRACED ray — lower detail and mis-coloured (the RT path
        //     drops the grass's vertex colours) vs. the primary raster view.
        //   • GL's transmission pass washed the view out.
        // The robust cross-backend answer is to NOT treat them as refractive
        // glass at all: render a plain transparent tint. On GL/WebGPU that is a
        // normal alpha blend over the already-shaded opaque scene (full detail,
        // correct colour). On Vulkan a plain transparent still hits the deferred
        // blend/re-trace, so we ALSO tag the windows onto the overlay layer
        // (kOverlayLayer): the renderer then draws them post-shade as a flat
        // tint composited over the full-quality image, depth-tested against the
        // G-buffer (so the dashboard still occludes them) — no re-trace, no dots.
        static bool tameGlass(Mesh& mesh) {
            bool glass = false;
            for (const auto& m : mesh.materials()) {
                auto* pm = dynamic_cast<MeshPhysicalMaterial*>(m.get());
                if (!pm || pm->transmission <= 0.f) continue;
                pm->transmission = 0.f;
                pm->transparent = true;
                pm->opacity = 0.16f;                    // subtle tint strength
                pm->color = Color(0.58f, 0.70f, 0.84f); // cool glass tint
                pm->roughness = 0.15f;
                pm->metalness = 0.f;
                pm->side = Side::Front;                 // one blend, not front+back
                pm->depthWrite = false;
                pm->needsUpdate();
                glass = true;
            }
            if (glass) mesh.layers.enable(kOverlayLayer);
            return glass;
        }

        void applyBrake(bool on) {
            if (on == brakeOn_) return;
            brakeOn_ = on;
            for (auto* m : brakeMats_)
                setEmissive(m, on ? 6.f : (headlightsOn_ ? 1.6f : 0.6f));
        }

        static void setEmissive(MeshStandardMaterial* m, float intensity) {
            if (!m) return;
            if (std::abs(m->emissiveIntensity - intensity) > 1e-3f) {
                m->emissiveIntensity = intensity;
                m->needsUpdate();
            }
        }

        // ── Name matching (case-insensitive substrings) ─────────────────────────
        static std::string upper(std::string s) {
            std::ranges::transform(s, s.begin(), [](unsigned char c) { return std::toupper(c); });
            return s;
        }
        static bool has(const std::string& hay, const char* needle) {
            return upper(hay).find(needle) != std::string::npos;
        }
        static bool isTyre(const std::string& n) { return has(n, "TYRE") || has(n, "TIRE"); }
        static bool isRim(const std::string& n) { return has(n, "WHEEL") && !has(n, "STEERING"); }
        static bool isHeadlightLens(const std::string& n) { return has(n, "HEADLIGHT"); }
        static bool isBrakeMat(const std::string& n) { return has(n, "EMISS") || has(n, "BRAKE"); }
        static bool isHeadlampMat(const std::string& n) { return has(n, "LIGHT_GLASS"); }

        Measurements meas_;
        std::shared_ptr<Group> root_, bodyPivot_;
        std::array<std::shared_ptr<Group>, 4> wheelGroups_{};

        std::array<std::shared_ptr<SpotLight>, 2> headlights_{};
        std::vector<Vector3> headlightLensWorld_;
        std::vector<MeshStandardMaterial*> brakeMats_, headlampMats_;

        float headlightIntensity_ = 48.f;
        bool headlightsOn_ = false;
        bool brakeOn_ = false;
    };

}// namespace drive

#endif//THREEPP_DRIVE_MUSTANGRIG_HPP
