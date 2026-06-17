// RLtools x threepp — live deep-reinforcement-learning demo (multi-pendulum).
//
// A single Soft-Actor-Critic (SAC) agent from RLtools (https://rl.tools) learns the
// classic Pendulum swing-up task. Training runs on a background thread; the threepp
// scene renders a whole FIELD of pendulums, all driven by that one current policy
// from different starting angles. Early on they flail in unison; as the shared
// policy learns, the whole field swings up and balances. It's a vivid way to see a
// single controller generalize across initial conditions.
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
#include <random>
#include <thread>
#include <vector>

using namespace threepp;
using rldemo::RLPendulumTrainer;

namespace {

    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kPivotY = 0.65f;// height of each pivot above its base
    constexpr float kArmLen = RLPendulumTrainer::kL;

    // field layout
    constexpr int kCols = 5;
    constexpr int kRows = 3;
    constexpr int kCount = kCols * kRows;
    constexpr float kDX = 1.7f;// spacing in X
    constexpr float kDZ = 1.7f;// spacing in Z

    // Match RLtools' angle_normalize (python-style modulo) so the displayed reward
    // tracks the training objective exactly.
    float angleNormalize(float x) {
        const float twoPi = 2 * kPi;
        float m = std::fmod(x + kPi, twoPi);
        if (m < 0) m += twoPi;
        return m - kPi;
    }

    struct Pendulum {
        float theta = kPi, thetaDot = 0.f, prevTheta = kPi;
        float ret = 0.f;       // current run's accumulated return
        float lastAction = 0.f;// last policy torque in [-1, 1]
        std::shared_ptr<Group> pivot;
        std::shared_ptr<MeshStandardMaterial> bobMat;
        std::shared_ptr<MeshStandardMaterial> goalMat;
    };

}// namespace

int main() {

    // HUD design units are logical pixels on a 96-dpi display. GLFW window
    // coordinates are physical pixels on Windows/X11, so every HUD design unit —
    // text AND the offsets between lines — is multiplied by the monitor content
    // scale; scaling only the text (as before) left the lines spaced for 96 dpi
    // and they overlapped on hi-dpi displays. On macOS window coordinates are
    // already logical points (the renderer compensates via pixelRatio), uiScale = 1.
    float uiScale = 1.f;
#ifndef __APPLE__
    uiScale = monitor::contentScale().first;
#endif

    // scale the window with the HUD, but keep it on screen
    const auto screen = monitor::monitorSize();
    const int winW = std::min(static_cast<int>(1280 * uiScale), screen.width() * 9 / 10);
    const int winH = std::min(static_cast<int>(800 * uiScale), screen.height() * 9 / 10);
    Canvas canvas("RLtools x threepp - SAC Pendulum field", {{"aa", 4}, {"size", WindowSize{winW, winH}}});
    auto renderer = createRenderer(canvas);
    renderer->autoClear = true;

    auto scene = Scene::create();
    scene->background = Color(0x222a33);

    auto camera = PerspectiveCamera::create(55, canvas.aspect(), 0.1f, 100);
    camera->position.set(0.f, 2.8f, 9.5f);

    OrbitControls controls{*camera, canvas};
    controls.target.set(0.f, 0.45f, 0.f);
    controls.update();

    // ---- lighting -------------------------------------------------------------
    scene->add(HemisphereLight::create(0xbfd4ff, 0x202830, 1.1f));
    auto sun = DirectionalLight::create(0xffffff, 1.4f);
    sun->position.set(3.f, 6.f, 4.f);
    scene->add(sun);

    auto grid = GridHelper::create(16, 32, 0x3a4654, 0x2c343d);
    grid->position.y = -0.6f;
    scene->add(grid);

    // ---- shared geometry + static materials (one set, reused by every pendulum)
    const auto standGeo = CylinderGeometry::create(0.03f, 0.045f, kPivotY + 0.6f, 16);
    const auto hubGeo = SphereGeometry::create(0.06f, 16, 12);
    const auto armGeo = CylinderGeometry::create(0.026f, 0.026f, kArmLen, 14);
    const auto bobGeo = SphereGeometry::create(0.12f, 24, 16);
    const auto goalGeo = TorusGeometry::create(0.16f, 0.02f, 12, 32);

    const auto standMat = MeshStandardMaterial::create({{"color", Color(0x4a5360)}, {"roughness", 0.7f}, {"metalness", 0.2f}});
    const auto hubMat = MeshStandardMaterial::create({{"color", Color(0x9aa6b4)}, {"roughness", 0.4f}, {"metalness", 0.6f}});
    const auto armMat = MeshStandardMaterial::create({{"color", Color(0xced6e0)}, {"roughness", 0.5f}, {"metalness", 0.3f}});

    std::vector<Pendulum> pends;
    pends.reserve(kCount);
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
            const float x = (c - (kCols - 1) / 2.f) * kDX;
            const float z = (r - (kRows - 1) / 2.f) * kDZ;

            auto stand = Mesh::create(standGeo, standMat);
            stand->position.set(x, (kPivotY - 0.6f) / 2.f, z);
            scene->add(stand);

            auto hub = Mesh::create(hubGeo, hubMat);
            hub->position.set(x, kPivotY, z);
            scene->add(hub);

            auto pivot = Group::create();
            pivot->position.set(x, kPivotY, z);
            scene->add(pivot);

            auto arm = Mesh::create(armGeo, armMat);
            arm->position.set(0.f, kArmLen / 2.f, 0.f);// span pivot -> bob along local +Y
            pivot->add(arm);

            auto bobMat = MeshStandardMaterial::create({{"color", Color(0xffffff)}, {"roughness", 0.35f}, {"metalness", 0.1f}});
            auto bob = Mesh::create(bobGeo, bobMat);
            bob->position.set(0.f, kArmLen, 0.f);
            pivot->add(bob);

            auto goalMat = MeshStandardMaterial::create({{"color", Color(0x2f6b3a)}, {"emissive", Color(0x102810)}});
            goalMat->transparent = true;
            goalMat->opacity = 0.85f;
            auto goal = Mesh::create(goalGeo, goalMat);
            goal->position.set(x, kPivotY + kArmLen, z);
            scene->add(goal);

            Pendulum p;
            p.pivot = pivot;
            p.bobMat = bobMat;
            p.goalMat = goalMat;
            pends.push_back(p);
        }
    }

    // ---- HUD text -------------------------------------------------------------
    FontLoader fontLoader;
    const auto font = fontLoader.defaultFont();

    auto makeHud = [&](float px, float py, float sizePx, const Color& col) {
        auto t = TextSprite::create(font, sizePx * uiScale);
        t->setColor(col);
        t->setVerticalAlignment(TextSprite::VerticalAlignment::Below);
        t->screenSpace = true;
        t->screenAnchor.set(0.f, 1.f);// top-left
        t->position.set(px * uiScale, py * uiScale, 0.f);
        scene->add(t);
        return t;
    };

    auto titleHud = makeHud(14.f, -12.f, 30.f, Color(0x8fe3ff));
    titleHud->setText("RLtools  x  threepp");
    auto subHud = makeHud(14.f, -46.f, 17.f, Color(0xc8d2dc));
    subHud->setText("One SAC policy controlling " + std::to_string(kCount) + " pendulums (each from a different start, restarts every ~10s)");
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
    auto resetEpisode = [&](Pendulum& p) {
        std::uniform_real_distribution<float> d(-0.45f, 0.45f);// spread of starts -> variety across the field
        std::uniform_real_distribution<float> dv(-0.6f, 0.6f);
        p.theta = kPi + d(rng);// start near hanging-down for a dramatic swing-up
        p.thetaDot = dv(rng);
        p.prevTheta = p.theta;
        p.ret = 0.f;
    };
    for (auto& p : pends) resetEpisode(p);

    int episodeStep = 0;        // shared: all pendulums step and reset together
    int episodesDone = 0;
    std::vector<float> returnHistory;// mean score per wave
    bool nextEpisodeReq = false;
    int simSpeed = 1;           // physics-step multiplier (speeds the view, not training)
    double simAccum = 0.0;

    // values computed each frame, surfaced to the ImGui panel
    int solvedCount = 0;
    float meanReturn = 0.f;

    auto physicsStep = [&] {
        for (auto& p : pends) {
            const float c = std::cos(p.theta), s = std::sin(p.theta);
            const float a = trainer.policyAction(c, s, p.thetaDot);// policy torque in [-1, 1]
            p.lastAction = a;
            const float u = RLPendulumTrainer::kMaxTorque * std::clamp(a, -1.f, 1.f);

            const float an = angleNormalize(p.theta);
            p.ret += -(an * an + 0.1f * p.thetaDot * p.thetaDot + 0.001f * (u * u));

            float newthdot = p.thetaDot + (3.f * RLPendulumTrainer::kG / (2.f * RLPendulumTrainer::kL) * std::sin(p.theta) + 3.f / (RLPendulumTrainer::kM * RLPendulumTrainer::kL * RLPendulumTrainer::kL) * u) * RLPendulumTrainer::kDt;
            newthdot = std::clamp(newthdot, -RLPendulumTrainer::kMaxSpeed, RLPendulumTrainer::kMaxSpeed);
            p.prevTheta = p.theta;
            p.theta += newthdot * RLPendulumTrainer::kDt;
            p.thetaDot = newthdot;
        }

        if (++episodeStep >= RLPendulumTrainer::kEpisodeSteps || nextEpisodeReq) {
            nextEpisodeReq = false;
            float mean = 0.f;
            for (auto& p : pends) mean += p.ret;
            mean /= static_cast<float>(pends.size());
            returnHistory.push_back(mean);
            if (returnHistory.size() > 240) returnHistory.erase(returnHistory.begin());
            ++episodesDone;
            episodeStep = 0;
            for (auto& p : pends) resetEpisode(p);
        }
    };

    // ---- ImGui control panel --------------------------------------------------
    float stepsPerSec = 0.f;
    long lastStepCount = 0;
    double spsTimer = 0.0;

    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        const auto vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos({vp->WorkPos.x + vp->WorkSize.x - 10, vp->WorkPos.y + 10}, ImGuiCond_Always, {1, 0});
        ImGui::SetNextWindowSize({340, 0}, ImGuiCond_Always);
        ImGui::Begin("RLtools - Soft Actor-Critic", nullptr, ImGuiWindowFlags_NoCollapse);

        ImGui::TextWrapped("A single SAC agent (rl.tools) trains in the background (counted in steps below). "
                           "Every pendulum in the field is driven by that one CURRENT policy from a different "
                           "start, and they all restart together every ~10s. Early = flailing; once trained, "
                           "the whole field swings up. The plot is the field's mean score per run.");
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

        ImGui::Text("Pendulums upright : %d / %d", solvedCount, kCount);
        ImGui::Text("Mean run score    : %8.1f", meanReturn);
        ImGui::Text("Test runs shown   : %d", episodesDone);
        ImGui::Spacing();

        if (!returnHistory.empty()) {
            ImGui::PlotLines("##returns", returnHistory.data(), static_cast<int>(returnHistory.size()),
                             0, "mean score per run (up = better)", -1700.f, 0.f, {-1, 70});
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
        if (ImGui::Button("Skip to next run (view only)", {-1, 0})) nextEpisodeReq = true;
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
        while (simAccum >= RLPendulumTrainer::kDt && guard++ < 12) {
            physicsStep();
            simAccum -= RLPendulumTrainer::kDt;
        }

        const float frac = static_cast<float>(std::clamp(simAccum / RLPendulumTrainer::kDt, 0.0, 1.0));

        solvedCount = 0;
        meanReturn = 0.f;
        float meanAngleDeg = 0.f;
        for (auto& p : pends) {
            const float renderTheta = p.prevTheta + (p.theta - p.prevTheta) * frac;
            p.pivot->rotation.z = -renderTheta;

            // bob colour encodes torque sign/magnitude (blue = CCW, red = CW)
            Color tc;
            if (p.lastAction >= 0) tc.lerpColors(Color(0xffffff), Color(0xff3b30), std::min(1.f, p.lastAction));
            else tc.lerpColors(Color(0xffffff), Color(0x3b7bff), std::min(1.f, -p.lastAction));
            p.bobMat->color.copy(tc);

            const float an = angleNormalize(p.theta);
            const bool balanced = std::abs(an) < 0.2f && std::abs(p.thetaDot) < 1.5f;
            p.goalMat->color.copy(balanced ? Color(0x37d34a) : Color(0x2f6b3a));
            p.goalMat->emissive.copy(balanced ? Color(0x123d16) : Color(0x102810));

            if (balanced) ++solvedCount;
            meanReturn += p.ret;
            meanAngleDeg += std::abs(an) * 180.f / kPi;
        }
        meanReturn /= static_cast<float>(pends.size());
        meanAngleDeg /= static_cast<float>(pends.size());

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
            status = "WARMUP - random actions (they flail)";
            statusCol = Color(0xff9f1c);
        } else {
            status = "LEARNING - policy improving live";
            statusCol = Color(0xffd23f);
        }
        statusHud->setText(status);
        statusHud->setColor(statusCol);

        char tele[160];
        std::snprintf(tele, sizeof(tele), "upright %2d / %d    mean |angle| %5.1f deg    mean score %7.1f",
                      solvedCount, kCount, meanAngleDeg, meanReturn);
        telemetryHud->setText(tele);

        renderer->render(*scene, *camera);
        ui.render();
    });

    // ---- shutdown -------------------------------------------------------------
    running.store(false);
    if (trainerThread.joinable()) trainerThread.join();

    return 0;
}
