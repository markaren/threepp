// Sensor Rig — the whole proprioceptive suite on one robot.
//
// A fixed-base two-joint leg stomps the floor on a sine drive. That single
// motion gives every sensor something real to say, and each one says something
// the others cannot:
//
//   Imu               on the shin: centripetal load through the swing, then a
//                     sharp spike the instant the foot lands.
//   JointEncoder      on hip and knee: the angle a controller would actually
//                     read — quantized to whole encoder ticks, not the exact
//                     value the simulator holds. Turn the resolution down and
//                     watch the reading go visibly stair-stepped while the true
//                     angle underneath stays smooth. This is the difference
//                     between a sensor and a getter.
//   ForceTorqueSensor on the hip joint: the wrench the base carries. Rises with
//                     the leg's reach and jumps on impact.
//   ContactSensor     on the shin: foot-down, plus the contact force.
//
// The point of running them together is that they disagree in useful ways —
// contact goes true a substep before the F/T spike peaks, and the IMU sees the
// deceleration the encoders never show at all.
//
// Controls: SPACE pause/resume, R reset the episode, mouse to orbit.

#include "threepp/threepp.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/extras/physx/Articulation.hpp"
#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/sensors/ContactSensor.hpp"
#include "threepp/extras/sensors/ForceTorqueSensor.hpp"
#include "threepp/extras/sensors/Imu.hpp"
#include "threepp/extras/sensors/JointEncoder.hpp"

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using namespace ::physx;

namespace {

    // 240 Hz physics: fast enough that a 200 Hz IMU and a 100 Hz encoder are
    // genuinely rate-gated rather than sampling every substep.
    constexpr float kSubstep = 1.f / 240.f;

    std::shared_ptr<MeshStandardMaterial> matte(const Color& c, float rough = 0.7f) {
        auto m = MeshStandardMaterial::create();
        m->color = c;
        m->roughness = rough;
        m->metalness = 0.05f;
        return m;
    }

    std::shared_ptr<Mesh> limb(float w, float h, float d, const Color& c) {
        auto mesh = Mesh::create(BoxGeometry::create(w, h, d), matte(c));
        mesh->castShadow = true;
        mesh->receiveShadow = true;
        return mesh;
    }

    // Fixed-length scrolling history for ImGui::PlotLines, which wants a flat
    // float array. A deque plus a scratch buffer keeps the copy trivial.
    class Trace {
    public:
        explicit Trace(std::size_t capacity = 400) : cap_(capacity) { flat_.reserve(cap_); }

        void push(float v) {
            v_.push_back(v);
            while (v_.size() > cap_) v_.pop_front();
        }
        void clear() {
            v_.clear();
            flat_.clear();
        }

        // Flattened view, plus the range to plot against. A fixed range would
        // hide small signals; an auto range makes the shape readable.
        const float* data() {
            flat_.assign(v_.begin(), v_.end());
            return flat_.empty() ? nullptr : flat_.data();
        }
        [[nodiscard]] int size() const { return static_cast<int>(v_.size()); }
        [[nodiscard]] float min() const { return v_.empty() ? 0.f : *std::min_element(v_.begin(), v_.end()); }
        [[nodiscard]] float max() const { return v_.empty() ? 1.f : *std::max_element(v_.begin(), v_.end()); }

    private:
        std::size_t cap_;
        std::deque<float> v_;
        std::vector<float> flat_;
    };

    // Plot with a little headroom so a flat line does not fill the whole box.
    // The label goes ABOVE rather than in PlotLines' own slot, which sits to the
    // right and would eat a third of an already narrow panel.
    void plot(const char* label, Trace& t, float height = 40.f) {
        const float lo = t.min(), hi = t.max();
        const float pad = std::max(1e-3f, (hi - lo) * 0.15f);
        ImGui::TextUnformatted(label);
        // The plot's own label is hidden ("##"), so every call would otherwise
        // land on the same ImGui ID and the plots would share hover state.
        // Scope each one by its visible label instead.
        ImGui::PushID(label);
        ImGui::PlotLines("##p", t.data(), t.size(), 0, nullptr, lo - pad, hi + pad,
                         ImVec2(-1, height));
        ImGui::PopID();
    }

}// namespace


int main() {

    Canvas canvas("Sensor Rig", {{"aa", 4}, {"vsync", true}, {"size", WindowSize{1280, 860}}});
    auto renderer = createRenderer(canvas);
    renderer->shadowMap().enabled = true;

    auto scene = Scene::create();
    scene->background = Color(0x9fb8d4);

    auto camera = PerspectiveCamera::create(55, canvas.aspect(), 0.1f, 200);
    camera->position.set(3.8f, 2.2f, 4.2f);

    OrbitControls controls{*camera, canvas};
    controls.target.set(0, 0.9f, 0);

    auto sun = DirectionalLight::create(0xffffff, 2.2f);
    sun->position.set(6, 12, 8);
    sun->castShadow = true;
    sun->shadow->mapSize.set(2048, 2048);
    sun->shadow->bias = -0.0004f;
    {
        auto* c = sun->shadow->camera->as<OrthographicCamera>();
        c->left = -8;
        c->right = 8;
        c->top = 8;
        c->bottom = -8;
        sun->shadow->camera->nearPlane = 1.f;
        sun->shadow->camera->farPlane = 40.f;
        sun->shadow->camera->updateProjectionMatrix();
    }
    scene->add(sun);
    scene->add(AmbientLight::create(0xb9cbe0, 0.55f));

    // ---- world ------------------------------------------------------------

    PhysxWorld::Settings settings;
    settings.fixedTimestep = kSubstep;
    settings.maxSubSteps = 8;
    PhysxWorld world(settings);

    auto ground = Mesh::create(BoxGeometry::create(40, 1, 40), matte(Color(0x6b7a55)));
    ground->position.y = -0.5f;
    ground->receiveShadow = true;
    scene->add(ground);
    world.addStatic(*ground);

    auto grid = GridHelper::create(40, 40, Color::gray, Color(0x808080));
    grid->position.y = 0.01f;
    scene->add(grid);

    // ---- the leg ----------------------------------------------------------
    //
    // base (fixed root) --hip--> thigh --knee--> shin
    //
    // Both joints hinge about Z, so the leg works in the XY plane and the whole
    // thing reads clearly from the default camera.

    constexpr float kThighLen = 0.9f;
    constexpr float kShinLen = 0.9f;
    // Hip height is the whole gait: fully extended the leg reaches
    // kThighLen + kShinLen = 1.8 m, so putting the hip a little BELOW that
    // drives the foot into the floor at the bottom of each stride instead of
    // waving it in the air. That overlap is what produces a contact event and
    // the impact spike the sensors are here to show.
    constexpr float kHipY = 1.78f;

    auto baseMesh = limb(0.5f, 0.4f, 0.5f, Color(0x39424e));
    baseMesh->position.set(0.f, kHipY + 0.3f, 0.f);
    scene->add(baseMesh);

    auto thighMesh = limb(0.18f, kThighLen, 0.18f, Color(0xC8783C));
    thighMesh->position.set(0.f, kHipY - kThighLen * 0.5f, 0.f);
    scene->add(thighMesh);

    auto shinMesh = limb(0.16f, kShinLen, 0.16f, Color(0xD9A05B));
    shinMesh->position.set(0.f, kHipY - kThighLen - kShinLen * 0.5f, 0.f);
    scene->add(shinMesh);

    Articulation leg(world, /*fixedBase*/ true, /*solverPositionIters*/ 32,
                     /*disableSelfCollision*/ true);

    auto base = leg.addLink(nullptr, *baseMesh, 800.f, {0, 0, 1}, {0, kHipY, 0},
                            false, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, "revolute", 0.f, nullptr);

    // Stiff position drives: the leg follows the commanded sine closely enough
    // that the interesting signal is the impact, not drive lag.
    auto thigh = leg.addLink(&base, *thighMesh, 900.f, {0, 0, 1}, {0, kHipY, 0},
                             /*limited*/ false, 0.f, 0.f,
                             /*stiffness*/ 4000.f, /*damping*/ 450.f,
                             /*maxForce*/ 1e6f, /*driveTarget*/ 0.f,
                             "revolute", 0.f, nullptr);

    auto shin = leg.addLink(&thigh, *shinMesh, 900.f, {0, 0, 1},
                            {0, kHipY - kThighLen, 0},
                            /*limited*/ false, 0.f, 0.f,
                            /*stiffness*/ 3000.f, /*damping*/ 350.f,
                            /*maxForce*/ 1e6f, /*driveTarget*/ 0.f,
                            "revolute", 0.f, nullptr);
    leg.finalize();

    // ---- the sensors ------------------------------------------------------

    // IMU on the shin: the fastest-moving part, so it sees both the swing's
    // centripetal term and the landing spike.
    Imu imu(*shinMesh, /*rateHz*/ 200.0);

    // Encoders on both joints, at a bus-like 100 Hz. 1024 counts/rev is a
    // mid-range industrial incremental encoder.
    JointEncoder hipEnc(*thighMesh, thigh, /*rateHz*/ 100.0);
    JointEncoder kneeEnc(*shinMesh, shin, /*rateHz*/ 100.0);
    hipEnc.setCountsPerRev(1024);
    kneeEnc.setCountsPerRev(1024);

    // Load cell at the hip — it carries the whole leg plus whatever the foot
    // pushes against.
    ForceTorqueSensor hipFt(*thighMesh, leg, thigh, /*rateHz*/ 200.0);

    // Foot contact. The shin's lower end is the foot.
    ContactSensor foot(*shinMesh);

    world.registerSensor(&imu);
    world.registerSensor(&hipEnc);
    world.registerSensor(&kneeEnc);
    world.registerSensor(&hipFt);
    world.registerSensor(&foot);

    // ---- gait + UI state --------------------------------------------------

    bool paused = false;
    bool noisy = true;    // MEMS-class IMU noise on by default
    int cprChoice = 1;    // index into kCpr
    float strideHz = 0.6f;// stomp cycle rate
    float reach = 0.45f;  // hip amplitude (rad)
    float tuck = 1.0f;    // knee amplitude (rad) — must clear the floor
    double simClock = 0.0;

    constexpr std::array<int, 4> kCpr{0, 1024, 128, 32};// 0 = ideal
    constexpr std::array<const char*, 4> kCprLabels{"ideal (continuous)", "1024 cpr",
                                                    "128 cpr", "32 cpr (coarse)"};

    const auto applyEncoderRes = [&] {
        if (kCpr[cprChoice] == 0) {
            hipEnc.resolution = 0.f;
            kneeEnc.resolution = 0.f;
        } else {
            hipEnc.setCountsPerRev(kCpr[cprChoice]);
            kneeEnc.setCountsPerRev(kCpr[cprChoice]);
        }
    };

    const auto applyNoise = [&] {
        if (noisy) {
            // Restore the MEMS-class defaults a fresh Imu ships with.
            imu.gyroNoise = NoiseModel{Vector3(0.005f, 0.005f, 0.005f),
                                       Vector3(4e-5f, 4e-5f, 4e-5f),
                                       Vector3(0.f, 0.f, 0.f), 0x9E3779B97F4A7C15ULL};
            imu.accelNoise = NoiseModel{Vector3(0.06f, 0.06f, 0.06f),
                                        Vector3(4e-3f, 4e-3f, 4e-3f),
                                        Vector3(0.f, 0.f, 0.f), 0xBF58476D1CE4E5B9ULL};
        } else {
            imu.gyroNoise = NoiseModel{};
            imu.accelNoise = NoiseModel{};
        }
        imu.reset();
    };
    applyNoise();

    // Traces, all fed once per rendered frame from latest().
    Trace trAccel, trHip, trHipTrue, trKnee, trForce, trTorque, trContact;
    const auto clearTraces = [&] {
        for (Trace* t: {&trAccel, &trHip, &trHipTrue, &trKnee, &trForce,
                        &trTorque, &trContact}) {
            t->clear();
        }
    };

    // Drive the gait from a pre-substep hook so the commanded angle advances on
    // the physics clock, not the render clock — the sensor traces then line up
    // with a fixed time base whatever the frame rate does.
    world.onPreSubstep([&](float dt) {
        if (paused) return;
        simClock += dt;
        const auto phase = static_cast<float>(simClock) * strideHz * 2.f * math::PI;
        // Hip sweeps the leg fore/aft; the knee tucks on the way up so the foot
        // clears the floor and lands flat-ish on the way down.
        thigh.setDriveTarget(reach * std::sin(phase));
        shin.setDriveTarget(-tuck * (1.f - std::cos(phase)) * 0.5f);
    });

    const auto resetEpisode = [&] {
        simClock = 0.0;
        leg.reset(Vector3(0.f, kHipY + 0.3f, 0.f));
        imu.reset();
        hipEnc.reset();
        kneeEnc.reset();
        hipFt.reset();
        foot.reset();
        clearTraces();
    };

    KeyAdapter keys(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent evt) {
        if (ImGui::GetIO().WantCaptureKeyboard) return;
        if (evt.key == Key::SPACE) paused = !paused;
        if (evt.key == Key::R) resetEpisode();
    });
    canvas.addKeyListener(keys);

    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        const float w = 330 * ui.dpiScale();
        ImGui::SetNextWindowPos({float(canvas.size().width()) - w, 0}, 0, {0, 0});
        // Full window height rather than auto-fit: the panel is taller than most
        // windows, and auto-fit silently clips the bottom section instead of
        // giving it a scrollbar.
        ImGui::SetNextWindowSize({w, float(canvas.size().height())}, 0);
        ImGui::Begin("Sensor Rig");

        ImGui::TextWrapped("One leg, four sensors. SPACE pause, R reset.");
        ImGui::Separator();

        ImGui::SliderFloat("stride", &strideHz, 0.1f, 1.5f, "%.2f Hz");
        ImGui::SliderFloat("reach", &reach, 0.1f, 1.0f, "%.2f rad");
        ImGui::SliderFloat("tuck", &tuck, 0.2f, 1.4f, "%.2f rad");
        if (ImGui::Button(paused ? "Resume (SPACE)" : "Pause (SPACE)")) paused = !paused;
        ImGui::SameLine();
        if (ImGui::Button("Reset (R)")) resetEpisode();

        // --- contact ---
        ImGui::Separator();
        const bool down = foot.inContact();
        ImGui::TextColored(down ? ImVec4(0.4f, 0.9f, 0.4f, 1.f) : ImVec4(0.6f, 0.6f, 0.6f, 1.f),
                           "ContactSensor: %s", down ? "FOOT DOWN" : "airborne");
        if (const auto c = foot.latest()) {
            ImGui::Text("  force %.0f N   points %u/%u", c->force.length(),
                        c->pointCount, c->observedPoints);
        }
        plot("foot down (0 / 1)", trContact, 30.f);
        ImGui::TextDisabled("latched: stays true when the pair sleeps");

        // --- imu ---
        ImGui::Separator();
        ImGui::Text("Imu (200 Hz, shin frame)");
        if (ImGui::Checkbox("MEMS noise", &noisy)) applyNoise();
        if (const auto s = imu.latest()) {
            ImGui::Text("  accel %6.2f %6.2f %6.2f  m/s2", s->linearAcceleration.x,
                        s->linearAcceleration.y, s->linearAcceleration.z);
            ImGui::Text("  gyro  %6.2f %6.2f %6.2f  rad/s", s->angularVelocity.x,
                        s->angularVelocity.y, s->angularVelocity.z);
        }
        plot("|accel| (spikes = footfalls)", trAccel);

        // --- encoders ---
        ImGui::Separator();
        ImGui::Text("JointEncoder (100 Hz)");
        if (ImGui::Combo("resolution", &cprChoice, kCprLabels.data(),
                         static_cast<int>(kCprLabels.size()))) {
            applyEncoderRes();
        }
        if (const auto h = hipEnc.latest()) {
            ImGui::Text("  hip  %+.4f rad   (true %+.4f)", h->position, thigh.jointPosition());
            ImGui::Text("  hip  %+.3f rad/s (differentiated)", h->velocity);
        }
        if (const auto k = kneeEnc.latest()) {
            ImGui::Text("  knee %+.4f rad   (true %+.4f)", k->position, shin.jointPosition());
        }
        plot("hip: encoder reading", trHip);
        plot("hip: true joint angle", trHipTrue);
        plot("knee: encoder reading", trKnee);
        ImGui::TextDisabled("drop to 32 cpr: the read trace stair-steps");
        ImGui::TextDisabled("while the true one stays smooth");

        // --- force/torque ---
        ImGui::Separator();
        ImGui::Text("ForceTorqueSensor (hip joint)");
        if (const auto f = hipFt.latest()) {
            ImGui::Text("  |F| %7.1f N    |T| %7.1f Nm", f->force.length(), f->torque.length());
        }
        plot("|F|", trForce);
        plot("|T|", trTorque);

        ImGui::End();
    });

    IOCapture cap;
    cap.preventMouseEvent = []{
        return ImGui::GetIO().WantCaptureMouse;
    };
    cap.preventScrollEvent = []{
        return ImGui::GetIO().WantCaptureMouse;
    };
    canvas.setIOCapture(&cap);

    Clock clock;
    canvas.animate([&] {
        const float dt = std::min(clock.getDelta(), 0.05f);
        if (!paused) world.step(dt);

        // Feed the traces from the read side, exactly as a controller would.
        if (const auto s = imu.latest()) trAccel.push(s->linearAcceleration.length());
        if (const auto h = hipEnc.latest()) trHip.push(h->position);
        trHipTrue.push(thigh.jointPosition());
        if (const auto k = kneeEnc.latest()) trKnee.push(k->position);
        if (const auto f = hipFt.latest()) {
            trForce.push(f->force.length());
            trTorque.push(f->torque.length());
        }
        trContact.push(foot.inContact() ? 1.f : 0.f);

        renderer->render(*scene, *camera);
        ui.render();
    });

    // Sensors are non-owning observers of the world; unhook before either dies.
    world.unregisterSensor(&imu);
    world.unregisterSensor(&hipEnc);
    world.unregisterSensor(&kneeEnc);
    world.unregisterSensor(&hipFt);
    world.unregisterSensor(&foot);
}
