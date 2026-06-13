// RLtools x threepp — live deep-reinforcement-learning demo.
//
// A Soft-Actor-Critic (SAC) agent from RLtools (https://rl.tools) learns the
// classic Pendulum swing-up task. Training runs on a background thread; the
// threepp scene continuously renders the *current* policy controlling a pendulum
// that obeys the very same dynamics the agent trains on. Watch it go from random
// flailing to a crisp swing-up-and-balance over the first ~minute.
//
// The RLtools engine is wrapped behind RLPendulumTrainer (plain C++); none of its
// template machinery leaks into this file.

#include "RLPendulumTrainer.hpp"

#include "threepp/canvas/Monitor.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <deque>
#include <random>
#include <thread>
#include <vector>

using namespace threepp;
using rldemo::RLPendulumTrainer;

namespace {

    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kPivotY = 0.65f;// height of the pivot above the world origin
    constexpr float kArmLen = RLPendulumTrainer::kL;

    // Match RLtools' angle_normalize (python-style modulo) so the displayed reward
    // tracks the training objective exactly.
    float angleNormalize(float x) {
        const float twoPi = 2 * kPi;
        float m = std::fmod(x + kPi, twoPi);
        if (m < 0) m += twoPi;
        return m - kPi;
    }

    // Bob position in the pivot frame: theta = 0 -> straight up, theta = PI -> down.
    Vector3 bobOffset(float theta) {
        return {std::sin(theta) * kArmLen, std::cos(theta) * kArmLen, 0.f};
    }

}// namespace

int main() {

    Canvas canvas("RLtools x threepp - SAC Pendulum", {{"aa", 4}, {"size", WindowSize{1280, 800}}});
    auto renderer = createRenderer(canvas);
    renderer->autoClear = true;

    auto scene = Scene::create();
    scene->background = Color(0x222a33);

    auto camera = PerspectiveCamera::create(55, canvas.aspect(), 0.1f, 100);
    camera->position.set(0.f, 0.7f, 5.0f);

    OrbitControls controls{*camera, canvas};
    controls.target.set(0.f, 0.55f, 0.f);
    controls.update();

    // ---- lighting -------------------------------------------------------------
    scene->add(HemisphereLight::create(0xbfd4ff, 0x202830, 1.1f));
    auto sun = DirectionalLight::create(0xffffff, 1.4f);
    sun->position.set(3.f, 6.f, 4.f);
    scene->add(sun);

    // ---- static scaffolding ---------------------------------------------------
    auto grid = GridHelper::create(10, 20, 0x3a4654, 0x2c343d);
    grid->position.y = -0.6f;
    scene->add(grid);

    auto stand = Mesh::create(
            CylinderGeometry::create(0.035f, 0.05f, kPivotY + 0.6f, 24),
            MeshStandardMaterial::create({{"color", Color(0x4a5360)}, {"roughness", 0.7f}, {"metalness", 0.2f}}));
    stand->position.set(0.f, (kPivotY - 0.6f) / 2.f, 0.f);
    scene->add(stand);

    auto hub = Mesh::create(
            SphereGeometry::create(0.07f, 24, 16),
            MeshStandardMaterial::create({{"color", Color(0x9aa6b4)}, {"roughness", 0.4f}, {"metalness", 0.6f}}));
    hub->position.set(0.f, kPivotY, 0.f);
    scene->add(hub);

    // goal ring at the upright target
    auto goalMat = MeshStandardMaterial::create({{"color", Color(0x2f6b3a)}, {"emissive", Color(0x102810)}});
    goalMat->transparent = true;
    goalMat->opacity = 0.85f;
    auto goal = Mesh::create(TorusGeometry::create(0.17f, 0.022f, 16, 48), goalMat);
    goal->position.set(0.f, kPivotY + kArmLen, 0.f);
    scene->add(goal);

    // ---- the pendulum (a rotating pivot group) --------------------------------
    auto armPivot = Group::create();
    armPivot->position.set(0.f, kPivotY, 0.f);
    scene->add(armPivot);

    auto arm = Mesh::create(
            CylinderGeometry::create(0.028f, 0.028f, kArmLen, 20),
            MeshStandardMaterial::create({{"color", Color(0xced6e0)}, {"roughness", 0.5f}, {"metalness", 0.3f}}));
    arm->position.set(0.f, kArmLen / 2.f, 0.f);// span pivot -> bob along local +Y
    armPivot->add(arm);

    auto bobMat = MeshStandardMaterial::create({{"color", Color(0xffffff)}, {"roughness", 0.35f}, {"metalness", 0.1f}});
    auto bob = Mesh::create(SphereGeometry::create(0.13f, 32, 24), bobMat);
    bob->position.set(0.f, kArmLen, 0.f);
    armPivot->add(bob);

    // torque indicator arrow (tangential push applied by the policy)
    auto torqueArrow = ArrowHelper::create(Vector3(1, 0, 0), Vector3(), 0.001f, Color(0xffd23f));
    scene->add(torqueArrow);

    // fading swing-up trail of the bob
    auto trailMat = LineBasicMaterial::create();
    trailMat->color = Color(0x6fd0ff);
    trailMat->transparent = true;
    trailMat->opacity = 0.6f;
    auto trailGeom = BufferGeometry::create();
    auto trail = Line::create(trailGeom, trailMat);
    scene->add(trail);
    std::deque<Vector3> trailPts;

    // ---- HUD text -------------------------------------------------------------
    FontLoader fontLoader;
    const auto font = fontLoader.defaultFont();
    const float cs = monitor::contentScale().first;

    auto makeHud = [&](float px, float py, float sizePx, const Color& col,
                       TextSprite::HorizontalAlignment ha = TextSprite::HorizontalAlignment::Left) {
        auto t = TextSprite::create(font, sizePx * cs);
        t->setColor(col);
        t->setVerticalAlignment(TextSprite::VerticalAlignment::Below);
        t->setHorizontalAlignment(ha);
        t->screenSpace = true;
        t->screenAnchor.set(0.f, 1.f);// top-left
        t->position.set(px, py, 0.f);
        scene->add(t);
        return t;
    };

    auto titleHud = makeHud(14.f, -12.f, 30.f, Color(0x8fe3ff));
    titleHud->setText("RLtools  x  threepp");
    auto subHud = makeHud(14.f, -46.f, 17.f, Color(0xc8d2dc));
    subHud->setText("3D pendulum = a live test of the current policy (restarts every ~10s)");
    auto stepHud = makeHud(14.f, -78.f, 20.f, Color(0xffffff));
    auto statusHud = makeHud(14.f, -108.f, 22.f, Color(0xffd23f));
    auto telemetryHud = makeHud(14.f, -140.f, 17.f, Color(0xc8d2dc));

    // ---- trainer + background training thread ---------------------------------
    RLPendulumTrainer trainer(/*seed*/ 1);

    std::atomic<bool> running{true};
    std::atomic<bool> paused{false};
    std::atomic<int> resetSeedReq{-1};
    std::thread trainerThread([&] {
        while (running.load()) {
            const int rs = resetSeedReq.exchange(-1);
            if (rs >= 0) trainer.reset(static_cast<unsigned>(rs));
            if (!paused.load() && !trainer.done()) {
                trainer.trainStep();
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    });

    // ---- viz / episode state --------------------------------------------------
    std::mt19937 rng(1234);
    auto resetEpisode = [&](float& theta, float& thetaDot, float& prevTheta) {
        std::uniform_real_distribution<float> d(-0.05f, 0.05f);
        std::uniform_real_distribution<float> dv(-0.2f, 0.2f);
        theta = kPi + d(rng);// start hanging down for a dramatic swing-up
        thetaDot = dv(rng);
        prevTheta = theta;
    };

    float theta, thetaDot, prevTheta;
    resetEpisode(theta, thetaDot, prevTheta);
    int episodeStep = 0;
    float episodeReturn = 0.f;
    int episodesDone = 0;
    float lastAction = 0.f;
    std::vector<float> returnHistory;
    int simSpeed = 1;       // physics-step multiplier (speeds up the view, not training)
    bool nextEpisodeReq = false;

    double simAccum = 0.0;

    auto physicsStep = [&] {
        const float c = std::cos(theta), s = std::sin(theta);
        const float a = trainer.policyAction(c, s, thetaDot);// policy torque in [-1, 1]
        lastAction = a;
        const float u = RLPendulumTrainer::kMaxTorque * std::clamp(a, -1.f, 1.f);

        // reward (RLtools' pendulum cost), from the pre-step state + action
        const float an = angleNormalize(theta);
        episodeReturn += -(an * an + 0.1f * thetaDot * thetaDot + 0.001f * (u * u));

        // integrate identical dynamics
        float newthdot = thetaDot + (3.f * RLPendulumTrainer::kG / (2.f * RLPendulumTrainer::kL) * std::sin(theta) + 3.f / (RLPendulumTrainer::kM * RLPendulumTrainer::kL * RLPendulumTrainer::kL) * u) * RLPendulumTrainer::kDt;
        newthdot = std::clamp(newthdot, -RLPendulumTrainer::kMaxSpeed, RLPendulumTrainer::kMaxSpeed);
        prevTheta = theta;
        theta += newthdot * RLPendulumTrainer::kDt;
        thetaDot = newthdot;

        // trail sample (in pivot frame -> world)
        trailPts.push_back(bobOffset(theta) + armPivot->position);
        while (trailPts.size() > 64) trailPts.pop_front();

        if (++episodeStep >= RLPendulumTrainer::kEpisodeSteps || nextEpisodeReq) {
            nextEpisodeReq = false;
            returnHistory.push_back(episodeReturn);
            if (returnHistory.size() > 240) returnHistory.erase(returnHistory.begin());
            ++episodesDone;
            episodeReturn = 0.f;
            episodeStep = 0;
            trailPts.clear();
            resetEpisode(theta, thetaDot, prevTheta);
        }
    };

    // ---- ImGui control panel --------------------------------------------------
    float stepsPerSec = 0.f;
    long lastStepCount = 0;
    double spsTimer = 0.0;

    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        ImGui::SetNextWindowPos({0, 0}, ImGuiCond_Once, {1, 0});
        const auto vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos({vp->WorkPos.x + vp->WorkSize.x - 10, vp->WorkPos.y + 10}, ImGuiCond_Always, {1, 0});
        ImGui::SetNextWindowSize({340, 0}, ImGuiCond_Always);
        ImGui::Begin("RLtools - Soft Actor-Critic", nullptr, ImGuiWindowFlags_NoCollapse);

        ImGui::TextWrapped("A SAC agent (rl.tools) trains in the background (counted in steps below). "
                           "The 3D pendulum is a live test of its CURRENT policy, auto-restarted every "
                           "~10s: it flails early and swings up cleanly once trained. The plot shows each "
                           "test run's score climbing as it learns.");
        ImGui::Separator();

        const long sc = trainer.stepCount();
        const long lim = RLPendulumTrainer::stepLimit();
        ImGui::Text("Training step");
        ImGui::ProgressBar(lim > 0 ? static_cast<float>(sc) / static_cast<float>(lim) : 0.f,
                           {-1, 0}, (std::to_string(sc) + " / " + std::to_string(lim)).c_str());
        ImGui::Text("Throughput: %.0f steps/s", stepsPerSec);
        if (trainer.done()) {
            ImGui::TextColored(ImVec4(0.22f, 0.83f, 0.29f, 1.f), "Converged - policy trained.");
        } else {
            const float eta = stepsPerSec > 1.f ? static_cast<float>(lim - sc) / stepsPerSec : 0.f;
            ImGui::TextColored(ImVec4(1.f, 0.82f, 0.25f, 1.f), "~%.0fs to converge (automatic)", eta);
        }
        ImGui::Spacing();

        ImGui::Text("Angle from upright: %+6.1f deg", angleNormalize(theta) * 180.f / kPi);
        ImGui::Text("Angular velocity : %+6.2f rad/s", thetaDot);
        ImGui::Text("Policy torque     : %+5.2f  (x%.0f Nm)", lastAction, RLPendulumTrainer::kMaxTorque);
        ImGui::Text("This run's score  : %8.1f", episodeReturn);
        ImGui::Text("Test runs shown   : %d", episodesDone);
        ImGui::Spacing();

        if (!returnHistory.empty()) {
            ImGui::PlotLines("##returns", returnHistory.data(), static_cast<int>(returnHistory.size()),
                             0, "score per test run (up = better)", -1700.f, 0.f, {-1, 70});
        }
        ImGui::Separator();

        bool p = paused.load();
        if (ImGui::Button(p ? "Resume training" : "Pause training", {-1, 0})) paused.store(!p);
        if (ImGui::Button("Restart training (new policy)", {-1, 0})) {
            std::uniform_int_distribution<int> sd(1, 100000);
            resetSeedReq.store(sd(rng));
            returnHistory.clear();
            episodesDone = 0;
        }
        if (ImGui::Button("Skip to next episode (view only)", {-1, 0})) nextEpisodeReq = true;
        ImGui::SliderInt("View speed", &simSpeed, 1, 8, "%dx");

        ImGui::End();
    });

    IOCapture capture;
    capture.preventMouseEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    capture.preventKeyboardEvent = [] { return ImGui::GetIO().WantCaptureKeyboard; };
    capture.preventScrollEvent = [] { return ImGui::GetIO().WantCaptureMouse; };
    canvas.setIOCapture(&capture);

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer->setSize(size);
    });

    Clock clock;
    canvas.animate([&] {
        const float dt = clock.getDelta();

        // advance the simulation at the pendulum's native 20 Hz (x view speed)
        simAccum += static_cast<double>(dt) * simSpeed;
        int guard = 0;
        while (simAccum >= RLPendulumTrainer::kDt && guard++ < 32) {
            physicsStep();
            simAccum -= RLPendulumTrainer::kDt;
        }

        // smooth sub-step interpolation of the rendered angle
        const float frac = static_cast<float>(std::clamp(simAccum / RLPendulumTrainer::kDt, 0.0, 1.0));
        const float renderTheta = prevTheta + (theta - prevTheta) * frac;
        armPivot->rotation.z = -renderTheta;

        // bob colour encodes torque sign/magnitude (blue = CCW, red = CW)
        Color tc;
        if (lastAction >= 0) tc.lerpColors(Color(0xffffff), Color(0xff3b30), std::min(1.f, lastAction));
        else tc.lerpColors(Color(0xffffff), Color(0x3b7bff), std::min(1.f, -lastAction));
        bobMat->color.copy(tc);

        // torque arrow: tangential push at the bob
        const Vector3 bobWorld = bobOffset(renderTheta) + armPivot->position;
        const float mag = std::abs(lastAction);
        if (mag > 0.02f) {
            Vector3 tangent(std::cos(renderTheta), -std::sin(renderTheta), 0.f);
            if (lastAction < 0) tangent.negate();
            torqueArrow->position.copy(bobWorld);
            torqueArrow->setDirection(tangent.normalize());
            torqueArrow->setLength(0.25f + 0.55f * mag, 0.12f, 0.08f);
            torqueArrow->visible = true;
        } else {
            torqueArrow->visible = false;
        }

        // goal ring turns green when balanced upright
        const bool balanced = std::abs(angleNormalize(theta)) < 0.2f && std::abs(thetaDot) < 1.5f;
        goalMat->color.copy(balanced ? Color(0x37d34a) : Color(0x2f6b3a));
        goalMat->emissive.copy(balanced ? Color(0x123d16) : Color(0x102810));

        // trail
        if (trailPts.size() >= 2) {
            std::vector<Vector3> pts(trailPts.begin(), trailPts.end());
            trailGeom->setFromPoints(pts);
            trail->visible = true;
        } else {
            trail->visible = false;
        }

        // throughput readout
        spsTimer += dt;
        if (spsTimer >= 0.4) {
            const long sc = trainer.stepCount();
            stepsPerSec = static_cast<float>(sc - lastStepCount) / static_cast<float>(spsTimer);
            lastStepCount = sc;
            spsTimer = 0.0;
        }

        // HUD
        const long sc = trainer.stepCount();
        const long lim = RLPendulumTrainer::stepLimit();
        const int pct = lim > 0 ? static_cast<int>(100 * sc / lim) : 0;
        stepHud->setText("Training step  " + std::to_string(sc) + " / " + std::to_string(lim) + "   (" + std::to_string(pct) + "%)");

        std::string status;
        Color statusCol;
        if (trainer.done()) {
            status = "TRAINED - this is the final controller";
            statusCol = Color(0x37d34a);
        } else if (sc < RLPendulumTrainer::warmupSteps()) {
            status = "WARMUP - random actions (it flails)";
            statusCol = Color(0xff9f1c);
        } else {
            status = "LEARNING - policy improving live";
            statusCol = Color(0xffd23f);
        }
        statusHud->setText(status);
        statusHud->setColor(statusCol);

        char tele[160];
        std::snprintf(tele, sizeof(tele), "angle %+6.1f deg   |w| %4.2f   torque %+4.2f   return %7.1f",
                      angleNormalize(theta) * 180.f / kPi, std::abs(thetaDot), lastAction, episodeReturn);
        telemetryHud->setText(tele);

        renderer->render(*scene, *camera);
        ui.render();
    });

    // ---- shutdown -------------------------------------------------------------
    running.store(false);
    if (trainerThread.joinable()) trainerThread.join();

    return 0;
}
