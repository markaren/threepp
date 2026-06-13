// RLtools x threepp — parallel-rollout RL swarm (Stage 1, CPU), env-generic.
//
// A SINGLE Soft-Actor-Critic learner is fed by a FIELD of 64 environments stepped in
// parallel ("many envs -> one learner"). The TASK is chosen by one line below
// (`using Env = ...`): pendulum, acrobot, ... The learner, the parallel-rollout
// plumbing and the visualization are all task-agnostic — each Env supplies its
// dynamics + its renderable links (env_pendulum.hpp / env_acrobot.hpp).

#include "env_acrobot.hpp"
#include "env_pendulum.hpp"
#include "RLSwarmTrainer.hpp"

#include "threepp/canvas/Monitor.hpp"
#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

using namespace threepp;

// ===== choose the task here ==================================================
using Env = rldemo::PendulumEnv;
// using Env = rldemo::AcrobotEnv;
// =============================================================================
using Trainer = rldemo::RLSwarmTrainer<Env::kObsDim, Env::kActDim>;

namespace {
    constexpr int M = Trainer::kNumEnvs;// 64
    constexpr int kCols = 8, kRows = 8;
    constexpr int O = Env::kObsDim, A = Env::kActDim;
    const float kBaseY = Env::kReach;          // pivot height so the mechanism clears the floor
    const float kSpacing = 2.4f * Env::kReach;  // grid spacing scales with the task's size
}// namespace

int main() {
    Canvas canvas(std::string("RLtools x threepp - swarm: ") + Env::kName, {{"aa", 4}, {"size", WindowSize{1280, 800}}});
    auto renderer = createRenderer(canvas);

    auto scene = Scene::create();
    scene->background = Color(0x20262e);

    const float gridW = kSpacing * kCols;
    auto camera = PerspectiveCamera::create(50, canvas.aspect(), 0.1f, 1000);
    camera->position.set(0.f, gridW * 0.42f, gridW * 1.15f);
    OrbitControls controls{*camera, canvas};
    controls.target.set(0.f, kBaseY * 0.3f, 0.f);
    controls.update();

    scene->add(HemisphereLight::create(0xbfd4ff, 0x202830, 1.2f));
    auto sun = DirectionalLight::create(0xffffff, 1.3f);
    sun->position.set(0.4f * gridW, gridW, 0.6f * gridW);
    scene->add(sun);
    auto grid = GridHelper::create(static_cast<unsigned>(gridW * 1.4f), static_cast<unsigned>(gridW * 1.4f), 0x3a4654, 0x2c343d);
    grid->position.y = -0.6f;
    scene->add(grid);

    // grid cell centres
    std::vector<float> cellX(M), cellZ(M);
    for (int r = 0; r < kRows; ++r)
        for (int c = 0; c < kCols; ++c) {
            const int i = r * kCols + c;
            cellX[i] = (c - (kCols - 1) / 2.f) * kSpacing;
            cellZ[i] = (r - (kRows - 1) / 2.f) * kSpacing;
        }

    // one InstancedMesh of unit-length cylinders per renderable link
    const float thick = 0.05f * Env::kReach;
    auto linkMat = MeshStandardMaterial::create({{"color", Color(0xced6e0)}, {"roughness", 0.5f}, {"metalness", 0.3f}});
    std::vector<std::shared_ptr<InstancedMesh>> links(Env::kRods);
    for (int r = 0; r < Env::kRods; ++r) {
        links[r] = InstancedMesh::create(CylinderGeometry::create(thick, thick, 1.f, 8), linkMat, M);
        links[r]->instanceMatrix()->setUsage(DrawUsage::Dynamic);
        scene->add(links[r]);
    }

    // End-effector spheres at each link's distal joint (pendulum -> bob; acrobot ->
    // elbow + tip). Two meshes per joint so the sphere turns green when that env is
    // "solved" (Env::upright) -> the field becomes a live at-a-glance success meter.
    // (A two-mesh scale toggle, not per-instance color, so it works on every backend.)
    const float bobR = 0.12f * Env::kReach;
    auto bobRestMat = MeshStandardMaterial::create({{"color", Color(0x6fd0ff)}, {"roughness", 0.35f}, {"metalness", 0.1f}});
    auto bobGoalMat = MeshStandardMaterial::create({{"color", Color(0x39d353)}, {"roughness", 0.4f}, {"metalness", 0.1f}, {"emissive", Color(0x12491f)}});
    std::vector<std::shared_ptr<InstancedMesh>> bobsRest(Env::kRods), bobsGoal(Env::kRods);
    for (int r = 0; r < Env::kRods; ++r) {
        bobsRest[r] = InstancedMesh::create(SphereGeometry::create(bobR, 14, 10), bobRestMat, M);
        bobsGoal[r] = InstancedMesh::create(SphereGeometry::create(bobR, 14, 10), bobGoalMat, M);
        bobsRest[r]->instanceMatrix()->setUsage(DrawUsage::Dynamic);
        bobsGoal[r]->instanceMatrix()->setUsage(DrawUsage::Dynamic);
        scene->add(bobsRest[r]);
        scene->add(bobsGoal[r]);
    }

    // HUD
    FontLoader fontLoader;
    const auto font = fontLoader.defaultFont();
    const float cs = monitor::contentScale().first;
    auto makeHud = [&](float py, float sz, const Color& col) {
        auto t = TextSprite::create(font, sz * cs);
        t->setColor(col);
        t->setVerticalAlignment(TextSprite::VerticalAlignment::Below);
        t->screenSpace = true;
        t->screenAnchor.set(0.f, 1.f);
        t->position.set(14.f, py, 0.f);
        scene->add(t);
        return t;
    };
    auto titleHud = makeHud(-12.f, 28.f, Color(0x8fe3ff));
    titleHud->setText(std::string("RLtools x threepp - ") + Env::kName + " swarm");
    auto subHud = makeHud(-44.f, 16.f, Color(0xc8d2dc));
    subHud->setText("One SAC learner, " + std::to_string(M) + " environments collected in parallel");
    auto stepHud = makeHud(-74.f, 19.f, Color(0xffffff));
    auto statusHud = makeHud(-103.f, 21.f, Color(0xffd23f));

    // ---- trainer + background collect/train thread ----------------------------
    Trainer trainer(1);
    std::atomic<bool> running{true};
    std::vector<typename Env::State> published(M);
    std::mutex pubMutex;
    std::vector<float> returnHistory;
    std::mutex histMutex;
    std::atomic<int> uprightCount{0};

    std::thread trainerThread([&] {
        using clk = std::chrono::steady_clock;
        std::mt19937 rng(7), evalRng(99);
        std::vector<typename Env::State> states(M), next(M);
        std::vector<int> epStep(M);
        std::uniform_int_distribution<int> phase(0, Env::kEpisodeSteps - 1);
        for (int i = 0; i < M; ++i) {
            states[i] = Env::sampleInitial(rng);
            epStep[i] = phase(rng);
        }
        std::vector<float> obs(M * O), nextObs(M * O), act(M * A), rew(M);
        std::vector<unsigned char> trunc(M);
        long tick = 0;
        while (running.load()) {
            const auto t0 = clk::now();
            for (int i = 0; i < M; ++i) Env::observe(states[i], &obs[i * O]);
            trainer.rolloutActions(obs.data(), act.data(), M);
            int up = 0;
            for (int i = 0; i < M; ++i) {
                rew[i] = Env::reward(states[i], &act[i * A]);
                next[i] = Env::step(states[i], &act[i * A]);
                Env::observe(next[i], &nextObs[i * O]);
                ++epStep[i];
                trunc[i] = epStep[i] >= Env::kEpisodeSteps ? 1 : 0;
                if (Env::upright(states[i])) ++up;
            }
            trainer.addTransitions(obs.data(), act.data(), rew.data(), nextObs.data(), trunc.data(), M);
            {
                std::lock_guard<std::mutex> lock(pubMutex);
                published = states;
            }
            for (int i = 0; i < M; ++i) {
                if (trunc[i]) { states[i] = Env::sampleInitial(rng); epStep[i] = 0; }
                else states[i] = next[i];
            }
            uprightCount.store(up);

            while (!trainer.done() && std::chrono::duration<double>(clk::now() - t0).count() < 0.045)
                trainer.update(1);

            if (++tick % 20 == 0) {
                // deterministic eval: mean per-episode return over a few fresh envs
                constexpr int E = 8;
                std::vector<typename Env::State> s(E);
                for (auto& st : s) st = Env::sampleInitial(evalRng);
                std::vector<float> o(E * O), a(E * A), ret(E, 0.f);
                for (int k = 0; k < Env::kEpisodeSteps; ++k) {
                    for (int i = 0; i < E; ++i) Env::observe(s[i], &o[i * O]);
                    trainer.policyActionsDet(o.data(), a.data(), E);
                    for (int i = 0; i < E; ++i) {
                        ret[i] += Env::reward(s[i], &a[i * A]);
                        s[i] = Env::step(s[i], &a[i * A]);
                    }
                }
                std::lock_guard<std::mutex> lock(histMutex);
                returnHistory.push_back(std::accumulate(ret.begin(), ret.end(), 0.f) / E);
                if (returnHistory.size() > 300) returnHistory.erase(returnHistory.begin());
            }
            std::this_thread::sleep_until(t0 + std::chrono::milliseconds(50));
        }
    });

    float gradPerSec = 0.f;
    long lastGrad = 0;
    double spsTimer = 0.0;

    ImguiFunctionalContext ui(canvas, *renderer, [&] {
        const auto vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos({vp->WorkPos.x + vp->WorkSize.x - 10, vp->WorkPos.y + 10}, ImGuiCond_Always, {1, 0});
        ImGui::SetNextWindowSize({340, 0}, ImGuiCond_Always);
        ImGui::Begin("RLtools - parallel SAC", nullptr, ImGuiWindowFlags_NoCollapse);
        ImGui::TextWrapped("Task: %s.  One SAC learner trains from %d environments stepped in parallel; the field "
                           "you see IS the live training data.", Env::kName, M);
        ImGui::Separator();
        const long sc = trainer.gradientSteps();
        const long lim = Trainer::updateBudget();
        ImGui::ProgressBar(lim > 0 ? static_cast<float>(sc) / lim : 0.f, {-1, 0},
                           (std::to_string(sc) + " / " + std::to_string(lim) + " grad steps").c_str());
        ImGui::Text("Learner: %.0f grad/s   |   %ld transitions", gradPerSec, trainer.transitionsCollected());
        ImGui::Text("At goal : %d / %d", uprightCount.load(), M);
        if (trainer.done()) ImGui::TextColored(ImVec4(0.22f, 0.83f, 0.29f, 1.f), "Converged.");
        else if (!trainer.pastWarmup()) ImGui::TextColored(ImVec4(1.f, 0.62f, 0.11f, 1.f), "Warmup");
        else ImGui::TextColored(ImVec4(1.f, 0.82f, 0.25f, 1.f), "Learning...");
        {
            std::lock_guard<std::mutex> lock(histMutex);
            if (!returnHistory.empty())
                ImGui::PlotLines("##ret", returnHistory.data(), static_cast<int>(returnHistory.size()), 0,
                                 "deterministic score (up = better)", FLT_MAX, FLT_MAX, {-1, 70});
        }
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

    std::vector<typename Env::State> snap(M);
    Quaternion q;
    Clock clock;
    canvas.animate([&] {
        const float dt = clock.getDelta();
        {
            std::lock_guard<std::mutex> lock(pubMutex);
            snap = published;
        }
        // place each env's links + end-effector spheres from its published state
        const Quaternion qi;// identity (spheres don't rotate)
        for (int r = 0; r < Env::kRods; ++r) {
            for (int i = 0; i < M; ++i) {
                float x0, y0, x1, y1;
                Env::rod(snap[i], r, x0, y0, x1, y1);
                const float ax = cellX[i] + x0, ay = kBaseY + y0;
                const float bx = cellX[i] + x1, by = kBaseY + y1;
                const float ddx = bx - ax, ddy = by - ay;
                const float len = std::sqrt(ddx * ddx + ddy * ddy);
                q.setFromAxisAngle({0.f, 0.f, 1.f}, std::atan2(-ddx, ddy));
                Matrix4 m;
                m.compose({(ax + bx) / 2.f, (ay + by) / 2.f, cellZ[i]}, q, {1.f, std::max(1e-3f, len), 1.f});
                links[r]->setMatrixAt(i, m);

                // sphere at the distal joint; show it in the rest- or goal-colored mesh
                const bool solved = Env::upright(snap[i]);
                Matrix4 show, hide;
                show.compose({bx, by, cellZ[i]}, qi, {1.f, 1.f, 1.f});
                hide.compose({bx, by, cellZ[i]}, qi, {0.f, 0.f, 0.f});// collapsed -> invisible
                bobsGoal[r]->setMatrixAt(i, solved ? show : hide);
                bobsRest[r]->setMatrixAt(i, solved ? hide : show);
            }
            links[r]->instanceMatrix()->needsUpdate();
            bobsRest[r]->instanceMatrix()->needsUpdate();
            bobsGoal[r]->instanceMatrix()->needsUpdate();
        }

        spsTimer += dt;
        if (spsTimer >= 0.4) {
            const long sc = trainer.gradientSteps();
            gradPerSec = static_cast<float>(sc - lastGrad) / static_cast<float>(spsTimer);
            lastGrad = sc;
            spsTimer = 0.0;
        }
        const long sc = trainer.gradientSteps();
        const long lim = Trainer::updateBudget();
        const int pct = lim > 0 ? static_cast<int>(100 * sc / lim) : 0;
        stepHud->setText("Grad steps  " + std::to_string(sc) + " / " + std::to_string(lim) + "   (" + std::to_string(pct) + "%)");
        statusHud->setText(trainer.done() ? "TRAINED" : (trainer.pastWarmup() ? "LEARNING - one policy, 64 envs" : "WARMUP"));

        renderer->render(*scene, *camera);
        ui.render();
    });

    running.store(false);
    if (trainerThread.joinable()) trainerThread.join();
    return 0;
}
