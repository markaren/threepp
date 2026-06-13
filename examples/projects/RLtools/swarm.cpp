// RLtools x threepp — parallel-rollout RL swarm (Stage 1, CPU).
//
// A SINGLE Soft-Actor-Critic agent (rl.tools) learns Pendulum swing-up, but its
// experience is collected from a FIELD of 64 environments stepped in parallel — the
// "thousands of envs feeding one learner" pattern (Isaac-Gym/Brax-style) at toy
// scale and cross-platform. The pendulums you see ARE the training environments:
// a background thread steps all 64, feeds their transitions to the learner, and
// runs SAC gradient steps; the render thread just visualizes the published states.
//
// Toggle "Deterministic showcase" to instead drive the field with the latest policy
// deterministically (clean, no exploration noise) — the same view the rltools_pendulum
// demo shows.
//
// The learner is behind RLSwarmTrainer (plain C++); env dynamics are in pendulum_env.hpp
// (the part that, in Stage 2, moves to a Vulkan compute shader).

#include "RLSwarmTrainer.hpp"
#include "pendulum_env.hpp"

#include "threepp/canvas/Monitor.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

using namespace threepp;
using rldemo::RLSwarmTrainer;

namespace {
    constexpr int M = RLSwarmTrainer::kNumEnvs;// 64
    constexpr int kCols = 8, kRows = 8;
    constexpr float kPivotY = 0.65f;
    constexpr float kArmLen = rlenv::kL;
    constexpr float kDX = 1.7f, kDZ = 1.7f;

    struct Vis {
        std::shared_ptr<Group> pivot;
        std::shared_ptr<MeshStandardMaterial> bobMat;
        std::shared_ptr<MeshStandardMaterial> goalMat;
        float dispTheta = rlenv::kPi, dispThetaDot = 0.f, dispAction = 0.f;
    };
    struct Pub {
        float theta = rlenv::kPi, thetaDot = 0.f, action = 0.f;
    };

    // deterministic policy eval (mean episode return over E fresh envs) — the honest
    // "how good is the policy" learning-curve signal (vs noisy exploratory returns).
    template<class RNG>
    float deterministicEval(RLSwarmTrainer& trainer, RNG& rng) {
        constexpr int E = 8;
        std::vector<rlenv::State> s(E);
        for (auto& st : s) st = rlenv::sampleInitial(rng);
        std::vector<float> obs(E * 3), act(E), ret(E, 0.f);
        for (int k = 0; k < rlenv::kEpisodeSteps; ++k) {
            for (int i = 0; i < E; ++i) rlenv::observe(s[i], &obs[i * 3]);
            trainer.policyActionsDet(obs.data(), act.data(), E);
            for (int i = 0; i < E; ++i) {
                ret[i] += rlenv::reward(s[i], act[i]);
                s[i] = rlenv::step(s[i], act[i]);
            }
        }
        return std::accumulate(ret.begin(), ret.end(), 0.f) / E;
    }
}// namespace

int main() {
    Canvas canvas("RLtools x threepp - parallel RL swarm", {{"aa", 4}, {"size", WindowSize{1280, 800}}});
    auto renderer = createRenderer(canvas);
    renderer->autoClear = true;

    auto scene = Scene::create();
    scene->background = Color(0x20262e);

    auto camera = PerspectiveCamera::create(50, canvas.aspect(), 0.1f, 200);
    camera->position.set(0.f, 7.5f, 17.f);
    OrbitControls controls{*camera, canvas};
    controls.target.set(0.f, 0.2f, 0.f);
    controls.update();

    scene->add(HemisphereLight::create(0xbfd4ff, 0x202830, 1.15f));
    auto sun = DirectionalLight::create(0xffffff, 1.3f);
    sun->position.set(4.f, 8.f, 6.f);
    scene->add(sun);
    auto grid = GridHelper::create(26, 26, 0x3a4654, 0x2c343d);
    grid->position.y = -0.6f;
    scene->add(grid);

    // shared geometry + static arm material
    const auto armGeo = CylinderGeometry::create(0.024f, 0.024f, kArmLen, 12);
    const auto bobGeo = SphereGeometry::create(0.11f, 20, 14);
    const auto goalGeo = TorusGeometry::create(0.15f, 0.018f, 10, 28);
    const auto armMat = MeshStandardMaterial::create({{"color", Color(0xced6e0)}, {"roughness", 0.5f}, {"metalness", 0.3f}});

    std::vector<Vis> field(M);
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
            const int idx = r * kCols + c;
            const float x = (c - (kCols - 1) / 2.f) * kDX;
            const float z = (r - (kRows - 1) / 2.f) * kDZ;
            auto pivot = Group::create();
            pivot->position.set(x, kPivotY, z);
            scene->add(pivot);
            auto arm = Mesh::create(armGeo, armMat);
            arm->position.set(0.f, kArmLen / 2.f, 0.f);
            pivot->add(arm);
            auto bobMat = MeshStandardMaterial::create({{"color", Color(0xffffff)}, {"roughness", 0.35f}, {"metalness", 0.1f}});
            auto bob = Mesh::create(bobGeo, bobMat);
            bob->position.set(0.f, kArmLen, 0.f);
            pivot->add(bob);
            auto goalMat = MeshStandardMaterial::create({{"color", Color(0x2f6b3a)}, {"emissive", Color(0x0f2410)}});
            goalMat->transparent = true;
            goalMat->opacity = 0.8f;
            auto goal = Mesh::create(goalGeo, goalMat);
            goal->position.set(x, kPivotY + kArmLen, z);
            scene->add(goal);
            field[idx].pivot = pivot;
            field[idx].bobMat = bobMat;
            field[idx].goalMat = goalMat;
        }
    }

    // HUD
    FontLoader fontLoader;
    const auto font = fontLoader.defaultFont();
    const float cs = monitor::contentScale().first;
    auto makeHud = [&](float px, float py, float sizePx, const Color& col) {
        auto t = TextSprite::create(font, sizePx * cs);
        t->setColor(col);
        t->setVerticalAlignment(TextSprite::VerticalAlignment::Below);
        t->screenSpace = true;
        t->screenAnchor.set(0.f, 1.f);
        t->position.set(px, py, 0.f);
        scene->add(t);
        return t;
    };
    auto titleHud = makeHud(14.f, -12.f, 28.f, Color(0x8fe3ff));
    titleHud->setText("RLtools  x  threepp  -  parallel RL swarm");
    auto subHud = makeHud(14.f, -44.f, 16.f, Color(0xc8d2dc));
    subHud->setText("One SAC learner, " + std::to_string(M) + " environments collected in parallel - the field IS the training data");
    auto stepHud = makeHud(14.f, -74.f, 19.f, Color(0xffffff));
    auto statusHud = makeHud(14.f, -103.f, 21.f, Color(0xffd23f));
    auto telemetryHud = makeHud(14.f, -133.f, 16.f, Color(0xc8d2dc));

    // ---- trainer + shared published state -------------------------------------
    RLSwarmTrainer trainer(/*seed*/ 1);
    std::atomic<bool> running{true};
    std::vector<Pub> published(M);
    std::mutex pubMutex;
    std::vector<float> returnHistory;// deterministic eval returns
    std::mutex histMutex;
    std::atomic<int> uprightTrain{0};// envs currently balanced (training collection)

    std::thread trainerThread([&] {
        using clk = std::chrono::steady_clock;
        std::mt19937 rng(7), evalRng(99);
        std::vector<rlenv::State> states(M), next(M);
        std::vector<int> epStep(M);
        std::uniform_int_distribution<int> phase(0, rlenv::kEpisodeSteps - 1);
        for (int i = 0; i < M; ++i) {
            states[i] = rlenv::sampleInitial(rng);
            epStep[i] = phase(rng);
        }
        std::vector<float> obs(M * 3), nextObs(M * 3), actions(M), rewards(M);
        std::vector<unsigned char> trunc(M);
        long tick = 0;
        while (running.load()) {
            const auto tickStart = clk::now();

            for (int i = 0; i < M; ++i) rlenv::observe(states[i], &obs[i * 3]);
            trainer.rolloutActions(obs.data(), actions.data(), M);
            int upright = 0;
            for (int i = 0; i < M; ++i) {
                rewards[i] = rlenv::reward(states[i], actions[i]);
                next[i] = rlenv::step(states[i], actions[i]);
                rlenv::observe(next[i], &nextObs[i * 3]);
                ++epStep[i];
                trunc[i] = epStep[i] >= rlenv::kEpisodeSteps ? 1 : 0;
                if (std::fabs(rlenv::angleNormalize(states[i].theta)) < 0.2f && std::fabs(states[i].thetaDot) < 1.5f) ++upright;
            }
            trainer.addTransitions(obs.data(), actions.data(), rewards.data(), nextObs.data(), trunc.data(), M);
            {
                std::lock_guard<std::mutex> lock(pubMutex);
                for (int i = 0; i < M; ++i) published[i] = {states[i].theta, states[i].thetaDot, actions[i]};
            }
            for (int i = 0; i < M; ++i) {
                if (trunc[i]) {
                    states[i] = rlenv::sampleInitial(rng);
                    epStep[i] = 0;
                } else {
                    states[i] = next[i];
                }
            }
            uprightTrain.store(upright);

            // pack gradient steps into the rest of a ~50 ms (20 Hz) tick so the swarm
            // runs near real-time while training proceeds.
            while (!trainer.done() && std::chrono::duration<double>(clk::now() - tickStart).count() < 0.045)
                trainer.update(1);

            if (++tick % 20 == 0) {
                const float r = deterministicEval(trainer, evalRng);
                std::lock_guard<std::mutex> lock(histMutex);
                returnHistory.push_back(r);
                if (returnHistory.size() > 300) returnHistory.erase(returnHistory.begin());
            }

            const auto target = tickStart + std::chrono::milliseconds(50);
            if (clk::now() < target) std::this_thread::sleep_until(target);
        }
    });

    // ---- showcase (deterministic) local sim state -----------------------------
    bool showTrainingEnvs = true;
    bool showcaseMode = false;// checkbox: when true -> deterministic showcase
    bool prevShowTraining = true;
    std::mt19937 showRng(123);
    std::vector<rlenv::State> showStates(M), showPrev(M);
    std::vector<int> showEp(M);
    std::vector<float> showObs(M * 3), showAct(M);
    double showAccum = 0.0;
    auto resetShowcase = [&] {
        std::uniform_int_distribution<int> phase(0, rlenv::kEpisodeSteps - 1);
        for (int i = 0; i < M; ++i) {
            showStates[i] = rlenv::sampleInitial(showRng);
            showPrev[i] = showStates[i];
            showEp[i] = phase(showRng);
        }
        showAccum = 0.0;
    };

    // ---- ImGui ----------------------------------------------------------------
    float gradPerSec = 0.f;
    long lastGrad = 0;
    double spsTimer = 0.0;
    int uprightShown = 0;

    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        const auto vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos({vp->WorkPos.x + vp->WorkSize.x - 10, vp->WorkPos.y + 10}, ImGuiCond_Always, {1, 0});
        ImGui::SetNextWindowSize({350, 0}, ImGuiCond_Always);
        ImGui::Begin("RLtools - parallel SAC", nullptr, ImGuiWindowFlags_NoCollapse);

        ImGui::TextWrapped("One SAC learner trains from %d environments stepped in parallel on a background "
                           "thread. The field you see IS the live training data (exploratory while learning). "
                           "Toggle the showcase to view the current policy deterministically.",
                           M);
        ImGui::Separator();

        const long sc = trainer.gradientSteps();
        const long lim = RLSwarmTrainer::updateBudget();
        ImGui::Text("Gradient steps");
        ImGui::ProgressBar(lim > 0 ? static_cast<float>(sc) / static_cast<float>(lim) : 0.f,
                           {-1, 0}, (std::to_string(sc) + " / " + std::to_string(lim)).c_str());
        ImGui::Text("Throughput : %.0f grad/s   |   %ld transitions", gradPerSec, trainer.transitionsCollected());
        if (trainer.done()) ImGui::TextColored(ImVec4(0.22f, 0.83f, 0.29f, 1.f), "Converged - policy trained.");
        else if (!trainer.pastWarmup()) ImGui::TextColored(ImVec4(1.f, 0.62f, 0.11f, 1.f), "Warmup - random exploration");
        else {
            const float eta = gradPerSec > 1.f ? static_cast<float>(lim - sc) / gradPerSec : 0.f;
            ImGui::TextColored(ImVec4(1.f, 0.82f, 0.25f, 1.f), "~%.0fs to converge (automatic)", eta);
        }
        ImGui::Spacing();
        ImGui::Text("Pendulums upright : %d / %d", showTrainingEnvs ? uprightTrain.load() : uprightShown, M);
        ImGui::Spacing();

        {
            std::lock_guard<std::mutex> lock(histMutex);
            if (!returnHistory.empty())
                ImGui::PlotLines("##ret", returnHistory.data(), static_cast<int>(returnHistory.size()),
                                 0, "deterministic policy score (up = better)", -1700.f, 0.f, {-1, 70});
        }
        ImGui::Separator();
        ImGui::Checkbox("Deterministic showcase (no exploration)", &showcaseMode);
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
        showTrainingEnvs = !showcaseMode;

        if (showTrainingEnvs != prevShowTraining) {
            if (!showTrainingEnvs) resetShowcase();
            prevShowTraining = showTrainingEnvs;
        }

        if (showTrainingEnvs) {
            std::vector<Pub> snap(M);
            {
                std::lock_guard<std::mutex> lock(pubMutex);
                snap = published;
            }
            const float k = std::min(1.f, 12.f * dt);
            for (int i = 0; i < M; ++i) {
                field[i].dispTheta += (snap[i].theta - field[i].dispTheta) * k;
                field[i].dispThetaDot = snap[i].thetaDot;
                field[i].dispAction = snap[i].action;
            }
        } else {
            showAccum += dt;
            int guard = 0;
            while (showAccum >= rlenv::kDt && guard++ < 8) {
                for (int i = 0; i < M; ++i) rlenv::observe(showStates[i], &showObs[i * 3]);
                trainer.policyActionsDet(showObs.data(), showAct.data(), M);
                for (int i = 0; i < M; ++i) {
                    showPrev[i] = showStates[i];
                    showStates[i] = rlenv::step(showStates[i], showAct[i]);
                    if (++showEp[i] >= rlenv::kEpisodeSteps) {
                        showStates[i] = rlenv::sampleInitial(showRng);
                        showPrev[i] = showStates[i];
                        showEp[i] = 0;
                    }
                }
                showAccum -= rlenv::kDt;
            }
            const float frac = static_cast<float>(std::clamp(showAccum / rlenv::kDt, 0.0, 1.0));
            for (int i = 0; i < M; ++i) {
                field[i].dispTheta = showPrev[i].theta + (showStates[i].theta - showPrev[i].theta) * frac;
                field[i].dispThetaDot = showStates[i].thetaDot;
                field[i].dispAction = showAct[i];
            }
        }

        // apply visuals
        int upright = 0;
        for (int i = 0; i < M; ++i) {
            field[i].pivot->rotation.z = -field[i].dispTheta;
            Color tc;
            if (field[i].dispAction >= 0) tc.lerpColors(Color(0xffffff), Color(0xff3b30), std::min(1.f, field[i].dispAction));
            else tc.lerpColors(Color(0xffffff), Color(0x3b7bff), std::min(1.f, -field[i].dispAction));
            field[i].bobMat->color.copy(tc);
            const bool bal = std::abs(rlenv::angleNormalize(field[i].dispTheta)) < 0.2f && std::abs(field[i].dispThetaDot) < 1.5f;
            field[i].goalMat->color.copy(bal ? Color(0x37d34a) : Color(0x2f6b3a));
            field[i].goalMat->emissive.copy(bal ? Color(0x123d16) : Color(0x0f2410));
            if (bal) ++upright;
        }
        uprightShown = upright;

        spsTimer += dt;
        if (spsTimer >= 0.4) {
            const long sc = trainer.gradientSteps();
            gradPerSec = static_cast<float>(sc - lastGrad) / static_cast<float>(spsTimer);
            lastGrad = sc;
            spsTimer = 0.0;
        }

        const long sc = trainer.gradientSteps();
        const long lim = RLSwarmTrainer::updateBudget();
        const int pct = lim > 0 ? static_cast<int>(100 * sc / lim) : 0;
        stepHud->setText("Gradient steps  " + std::to_string(sc) + " / " + std::to_string(lim) + "   (" + std::to_string(pct) + "%)");
        std::string status;
        Color statusCol;
        if (trainer.done()) { status = "TRAINED - final policy"; statusCol = Color(0x37d34a); }
        else if (!trainer.pastWarmup()) { status = "WARMUP - random actions"; statusCol = Color(0xff9f1c); }
        else { status = "LEARNING - one policy, " + std::to_string(M) + " envs"; statusCol = Color(0xffd23f); }
        statusHud->setText(status);
        statusHud->setColor(statusCol);
        char tele[160];
        std::snprintf(tele, sizeof(tele), "%s view   |   upright %2d / %d   |   %ld transitions collected",
                      showTrainingEnvs ? "training-envs" : "deterministic", upright, M, trainer.transitionsCollected());
        telemetryHud->setText(tele);

        renderer->render(*scene, *camera);
        ui.render();
    });

    running.store(false);
    if (trainerThread.joinable()) trainerThread.join();
    return 0;
}
