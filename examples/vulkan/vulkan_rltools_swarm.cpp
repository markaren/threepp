// RLtools x threepp — GPU rollout swarm (Stage 2, Vulkan compute), env-generic.
//
// The ENTIRE rollout runs on the GPU: a single fused compute shader steps N
// environments in parallel (one thread per env), running the actor MLP forward +
// sample_and_squash + dynamics, and emits a transition each. Those transitions are
// read back and fed to the SAME CPU SAC learner (RLSwarmTrainer) — the learner never
// changed; only the rollout moved to the GPU. Cross-vendor (Vulkan), reusing the
// RF-DETR VkInfer harness.
//
// The TASK is chosen by ONE line below (`using Env = ...`), exactly like the CPU
// swarm. Each Env owns its dynamics twice: once in C++ (env_*.hpp, for the learner +
// viz) and once in GLSL (swarm_rollout.comp, behind ENV_*, compiled to a per-task
// SPIR-V variant). The host wires the matching variant + dims from the Env.
//
// Prints a self-test (GPU action == CPU forward) and a live learning curve to the
// console, so the pipeline is verifiable without watching the window.

#include "VkInfer.hpp"// rfdetr/VkInfer.hpp (added to include path)

#include "env_acrobot.hpp"
#include "env_pendulum.hpp"
#include "RLSwarmTrainer.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

// one rollout-shader SPIR-V variant per task; the matching one is selected below.
#include "threepp/renderers/vulkan/shaders/swarm_rollout.comp.pendulum.spv.h"// kSwarmPendulumSpv
#include "threepp/renderers/vulkan/shaders/swarm_rollout.comp.acrobot.spv.h" // kSwarmAcrobotSpv

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
#include <type_traits>
#include <vector>

using namespace threepp;

// ===== choose the task here ==================================================
using Env = rldemo::PendulumEnv;
// using Env = rldemo::AcrobotEnv;
// =============================================================================
using Trainer = rldemo::RLSwarmTrainer<Env::kObsDim, Env::kActDim>;

namespace {
    constexpr int N_TOTAL = 256;// environments stepped on the GPU per dispatch
    constexpr int N_VIS = 256;  // visualized (instanced)
    constexpr int kCols = 16, kRows = 16;
    constexpr int O = Env::kObsDim, A = Env::kActDim;
    constexpr int SD = Env::kStateDim;       // dynamics vars per env in the GPU state buffer
    constexpr int SSTRIDE = SD + 1;          // + epStep
    constexpr int TRANS = 2 * O + A + 2;      // obs, action, reward, nextObs, trunc
    constexpr int R = Env::kRods;
    constexpr int WCOUNT = Trainer::kWeightCount;
    constexpr int H = Trainer::kHidden;
    constexpr bool kIsPendulum = std::is_same_v<Env, rldemo::PendulumEnv>;

    // The shader gets all dynamics constants + episode length from the single-source
    // envdyn file, so the push-constant only carries the per-dispatch flags.
    struct PC {
        uint32_t n;
        uint32_t stochastic;
        uint32_t warmup;
    };

    // CPU reference forward (deterministic, action 0) — for the GPU parity self-test.
    // Mirrors the shader's actor MLP: OBS->64->64->2*ACT, action0 = tanh(output[0]).
    float forwardDet(const float* w, const float* obs) {
        const float* W1 = w;
        const float* b1 = w + H * O;
        const float* W2 = b1 + H;
        const float* b2 = W2 + H * H;
        const float* W3 = b2 + H;
        const float* b3 = W3 + (2 * A) * H;
        float h1[H], h2[H];
        for (int o = 0; o < H; ++o) {
            float s = b1[o];
            for (int i = 0; i < O; ++i) s += W1[o * O + i] * obs[i];
            h1[o] = s > 0 ? s : 0;
        }
        for (int o = 0; o < H; ++o) {
            float s = b2[o];
            for (int i = 0; i < H; ++i) s += W2[o * H + i] * h1[i];
            h2[o] = s > 0 ? s : 0;
        }
        float mean = b3[0];
        for (int i = 0; i < H; ++i) mean += W3[i] * h2[i];
        return std::tanh(mean);
    }

    PC makePC(uint32_t stochastic, uint32_t warmup) {
        return PC{(uint32_t) N_TOTAL, stochastic, warmup};
    }
}// namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);// unbuffered so console validation shows up live
    Canvas canvas(std::string("RLtools x threepp - GPU swarm (Vulkan compute): ") + Env::kName,
                  {{"vsync", true}, {"size", WindowSize{1280, 800}}});
    VulkanRenderer renderer(canvas);
    renderer.outputColorSpace = ColorSpace::sRGB;

    auto scene = Scene::create();
    scene->background = Color(0x20262e);

    const float reach = Env::kReach;
    const float baseY = reach;            // pivot height so the mechanism clears the floor
    const float spacing = 2.4f * reach;   // grid spacing scales with the task's size
    const float gridW = spacing * kCols;
    auto camera = PerspectiveCamera::create(50, canvas.aspect(), 0.1f, std::max(400.f, gridW * 3.f));
    camera->position.set(0.f, gridW * 0.40f, gridW * 1.15f);
    OrbitControls controls{*camera, canvas};
    controls.target.set(0.f, baseY * 0.3f, 0.f);
    controls.update();

    scene->add(HemisphereLight::create(0xbfd4ff, 0x202830, 1.2f));
    auto sun = DirectionalLight::create(0xffffff, 1.3f);
    sun->position.set(0.4f * gridW, gridW, 0.6f * gridW);
    scene->add(sun);
    auto grid = GridHelper::create(static_cast<unsigned>(gridW * 1.4f), static_cast<unsigned>(gridW * 1.4f), 0x3a4654, 0x2c343d);
    grid->position.y = -0.6f;
    scene->add(grid);

    // grid cell centres
    std::vector<float> cellX(N_VIS), cellZ(N_VIS);
    for (int r = 0; r < kRows; ++r)
        for (int c = 0; c < kCols; ++c) {
            const int i = r * kCols + c;
            if (i >= N_VIS) break;
            cellX[i] = (c - (kCols - 1) / 2.f) * spacing;
            cellZ[i] = (r - (kRows - 1) / 2.f) * spacing;
        }

    // one InstancedMesh of unit-length cylinders per renderable link (any Env)
    const float thick = 0.05f * reach;
    auto linkMat = MeshStandardMaterial::create({{"color", Color(0xced6e0)}, {"roughness", 0.5f}, {"metalness", 0.3f}});
    std::vector<std::shared_ptr<InstancedMesh>> links(R);
    for (int r = 0; r < R; ++r) {
        links[r] = InstancedMesh::create(CylinderGeometry::create(thick, thick, 1.f, 8), linkMat, N_VIS);
        links[r]->instanceMatrix()->setUsage(DrawUsage::Dynamic);
        scene->add(links[r]);
    }

    // End-effector spheres at each link's distal joint (pendulum -> bob; acrobot ->
    // elbow + tip), turning green when that env is "solved" (Env::upright) so the
    // swarm is a live at-a-glance success meter. Two meshes per joint + a scale toggle
    // (not per-instance color, which the Vulkan backend doesn't carry).
    const float bobR = 0.12f * reach;
    auto bobRestMat = MeshStandardMaterial::create({{"color", Color(0x6fd0ff)}, {"roughness", 0.35f}, {"metalness", 0.1f}});
    auto bobGoalMat = MeshStandardMaterial::create({{"color", Color(0x39d353)}, {"roughness", 0.4f}, {"metalness", 0.1f}, {"emissive", Color(0x12491f)}});
    std::vector<std::shared_ptr<InstancedMesh>> bobsRest(R), bobsGoal(R);
    for (int r = 0; r < R; ++r) {
        bobsRest[r] = InstancedMesh::create(SphereGeometry::create(bobR, 14, 10), bobRestMat, N_VIS);
        bobsGoal[r] = InstancedMesh::create(SphereGeometry::create(bobR, 14, 10), bobGoalMat, N_VIS);
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
    auto titleHud = makeHud(-12.f, 27.f, Color(0x8fe3ff));
    titleHud->setText(std::string("RLtools x threepp - GPU rollout swarm: ") + Env::kName);
    auto subHud = makeHud(-43.f, 16.f, Color(0xc8d2dc));
    subHud->setText(std::to_string(N_TOTAL) + " environments stepped per dispatch on the GPU; one CPU SAC learner");
    auto stepHud = makeHud(-72.f, 19.f, Color(0xffffff));
    auto statusHud = makeHud(-100.f, 20.f, Color(0xffd23f));

    // ---- VkInfer compute harness + buffers ------------------------------------
    rfdetr::VkInfer vk(static_cast<VkDevice>(renderer.nativeDevice()),
                       static_cast<VkPhysicalDevice>(renderer.nativePhysicalDevice()),
                       static_cast<VkQueue>(renderer.nativeGraphicsQueue()),
                       renderer.graphicsQueueFamily());
    auto weightsBuf = vk.createOwned({(uint32_t) WCOUNT});
    auto stateBuf = vk.createOwned({(uint32_t) (N_TOTAL * SSTRIDE)});
    auto noiseBuf = vk.createOwned({(uint32_t) (N_TOTAL * A)});
    auto initBuf = vk.createOwned({(uint32_t) (N_TOTAL * SD)});
    auto outBuf = vk.createOwned({(uint32_t) (N_TOTAL * TRANS)});

    // pick the rollout-shader variant compiled for this task
    const uint32_t* shaderSpv = kIsPendulum ? kSwarmPendulumSpv : kSwarmAcrobotSpv;
    const size_t shaderLen = kIsPendulum ? sizeof(kSwarmPendulumSpv) : sizeof(kSwarmAcrobotSpv);
    auto pipe = vk.createPipe(shaderSpv, shaderLen, 5, sizeof(PC));
    const std::vector<VkBuffer> ssbos = {weightsBuf.buffer, stateBuf.buffer, noiseBuf.buffer, initBuf.buffer, outBuf.buffer};
    const uint32_t groups = (N_TOTAL + 63) / 64;

    std::mt19937 rng(1);
    std::normal_distribution<float> gauss(0.f, 1.f);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> stateHost(N_TOTAL * SSTRIDE), initHost(N_TOTAL * SD), noiseHost(N_TOTAL * A),
            weightsHost(WCOUNT), outHost(N_TOTAL * TRANS);
    std::uniform_int_distribution<int> phase(0, Env::kEpisodeSteps - 1);
    auto seedStates = [&] {
        for (int e = 0; e < N_TOTAL; ++e) {
            Env::packState(Env::sampleInitial(rng), &stateHost[e * SSTRIDE]);
            stateHost[e * SSTRIDE + SD] = static_cast<float>(phase(rng));
        }
    };
    auto seedInitials = [&] {
        for (int e = 0; e < N_TOTAL; ++e) Env::packState(Env::sampleInitial(rng), &initHost[e * SD]);
    };
    seedStates();
    vk.upload(stateBuf.buffer, stateHost.data(), stateHost.size() * sizeof(float));

    Trainer trainer(1);

    // ---- GPU<->CPU parity self-test (deterministic) ---------------------------
    trainer.extractWeights(weightsHost.data());
    vk.upload(weightsBuf.buffer, weightsHost.data(), WCOUNT * sizeof(float));
    std::fill(noiseHost.begin(), noiseHost.end(), 0.f);
    vk.upload(noiseBuf.buffer, noiseHost.data(), noiseHost.size() * sizeof(float));
    seedInitials();
    vk.upload(initBuf.buffer, initHost.data(), initHost.size() * sizeof(float));
    {
        PC pc = makePC(0, 0);
        vk.beginFrame();
        vk.dispatch(pipe, ssbos, &pc, sizeof(pc), groups, 1, 1);
        vk.endFrame();
        vk.readback(outBuf.buffer, outHost.data(), outHost.size() * sizeof(float));
        float maxDiff = 0.f;
        for (int e = 0; e < N_TOTAL; ++e) {
            const float* obs = &outHost[e * TRANS];
            const float gpuA = outHost[e * TRANS + O];// first action
            maxDiff = std::fmax(maxDiff, std::fabs(gpuA - forwardDet(weightsHost.data(), obs)));
        }
        std::printf("[parity:%s] GPU rollout vs CPU forward: max |diff| = %.2e  ->  %s\n",
                    Env::kName, maxDiff, maxDiff < 1e-4f ? "PARITY ok" : "MISMATCH");
    }
    seedStates();// the parity dispatch advanced state; reset for training
    vk.upload(stateBuf.buffer, stateHost.data(), stateHost.size() * sizeof(float));

    // ---- dedicated learner thread (decoupled from render/rollout) -------------
    // The render thread does GPU rollout + readback + viz; it hands each transition
    // batch to this thread, which runs SAC gradient steps flat-out on its own core
    // (no render contention, no per-frame budget) -> ~2x gradient throughput and a
    // smooth frame rate. ts stays owned by this single thread (addTransitions+update).
    std::atomic<bool> running{true};
    std::mutex batchMutex;
    bool batchReady = false;
    std::vector<float> pObs(N_TOTAL * O), pAct(N_TOTAL * A), pRew(N_TOTAL), pNObs(N_TOTAL * O);
    std::vector<unsigned char> pTrunc(N_TOTAL);
    std::vector<float> returnHistory;

    std::thread learnerThread([&] {
        std::vector<float> lObs(N_TOTAL * O), lAct(N_TOTAL * A), lRew(N_TOTAL), lNObs(N_TOTAL * O);
        std::vector<unsigned char> lTrunc(N_TOTAL);
        while (running.load()) {
            bool got = false;
            {
                std::lock_guard<std::mutex> lock(batchMutex);
                if (batchReady) {
                    lObs = pObs;
                    lAct = pAct;
                    lRew = pRew;
                    lNObs = pNObs;
                    lTrunc = pTrunc;
                    batchReady = false;
                    got = true;
                }
            }
            if (got) trainer.addTransitions(lObs.data(), lAct.data(), lRew.data(), lNObs.data(), lTrunc.data(), N_TOTAL);
            if (!trainer.done()) trainer.update(8);// a chunk, then re-check for fresh data
            else std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    float gradPerSec = 0.f;
    long lastGrad = 0;
    double spsTimer = 0.0;
    long tick = 0;
    std::mt19937 evalRng(99);

    // optional deterministic "policy view" showcase: a separate field driven by the
    // no-noise policy (policyActionsDet, chunked by kNumEnvs). The GPU training rollout
    // never pauses; this only changes what is drawn. Synchronized episodes.
    bool policyView = false;
    std::vector<typename Env::State> showStates(N_VIS);
    std::vector<float> showObs(Trainer::kNumEnvs * O), showAct(Trainer::kNumEnvs * A);
    int showEp = 0;
    std::mt19937 showRng(123);
    for (int i = 0; i < N_VIS; ++i) showStates[i] = Env::sampleInitial(showRng);

    auto deterministicEval = [&]() -> float {
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
        return std::accumulate(ret.begin(), ret.end(), 0.f) / E;
    };

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        const auto vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos({vp->WorkPos.x + vp->WorkSize.x - 10, vp->WorkPos.y + 10}, ImGuiCond_Always, {1, 0});
        ImGui::SetNextWindowSize({340, 0}, ImGuiCond_Always);
        ImGui::Begin("RLtools - GPU rollout swarm", nullptr, ImGuiWindowFlags_NoCollapse);
        ImGui::TextWrapped("Task: %s.  %d environments are stepped in ONE Vulkan compute dispatch (observe -> "
                           "actor MLP -> sample -> step). Transitions feed one CPU SAC learner.", Env::kName, N_TOTAL);
        ImGui::Checkbox("Policy view (deterministic - no exploration noise)", &policyView);
        ImGui::Separator();
        const long sc = trainer.gradientSteps();
        const long lim = Trainer::updateBudget();
        ImGui::ProgressBar(lim > 0 ? static_cast<float>(sc) / lim : 0.f, {-1, 0},
                           (std::to_string(sc) + " / " + std::to_string(lim) + " grad steps").c_str());
        ImGui::Text("Learner: %.0f grad/s   |   %ld transitions", gradPerSec, trainer.transitionsCollected());
        if (trainer.done()) ImGui::TextColored(ImVec4(0.22f, 0.83f, 0.29f, 1.f), "Converged.");
        else if (!trainer.pastWarmup()) ImGui::TextColored(ImVec4(1.f, 0.62f, 0.11f, 1.f), "Warmup");
        else ImGui::TextColored(ImVec4(1.f, 0.82f, 0.25f, 1.f), "Learning...");
        if (!returnHistory.empty())
            ImGui::PlotLines("##ret", returnHistory.data(), static_cast<int>(returnHistory.size()), 0,
                             "deterministic policy score (up=better)", FLT_MAX, FLT_MAX, {-1, 70});
        ImGui::End();
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Quaternion q;
    Clock clock;
    canvas.animate([&] {
        const float dt = clock.getDelta();

        // 1. upload the latest policy + per-tick noise/initials
        trainer.extractWeights(weightsHost.data());
        const bool warm = !trainer.pastWarmup();
        for (int e = 0; e < N_TOTAL * A; ++e) noiseHost[e] = warm ? uni(rng) : gauss(rng);
        seedInitials();
        vk.upload(weightsBuf.buffer, weightsHost.data(), WCOUNT * sizeof(float));
        vk.upload(noiseBuf.buffer, noiseHost.data(), noiseHost.size() * sizeof(float));
        vk.upload(initBuf.buffer, initHost.data(), initHost.size() * sizeof(float));

        // 2. GPU rollout dispatch (all envs) + readback transitions & states
        PC pc = makePC(1, warm ? 1u : 0u);
        vk.beginFrame();
        vk.dispatch(pipe, ssbos, &pc, sizeof(pc), groups, 1, 1);
        vk.endFrame();
        vk.readback2(outBuf.buffer, outHost.data(), outHost.size() * sizeof(float),
                     stateBuf.buffer, stateHost.data(), stateHost.size() * sizeof(float));

        // 3. hand the batch to the learner thread (it does add + grad steps on its own core)
        {
            std::lock_guard<std::mutex> lock(batchMutex);
            for (int e = 0; e < N_TOTAL; ++e) {
                const float* t = &outHost[e * TRANS];
                for (int j = 0; j < O; ++j) pObs[e * O + j] = t[j];
                for (int j = 0; j < A; ++j) pAct[e * A + j] = t[O + j];
                pRew[e] = t[O + A];
                for (int j = 0; j < O; ++j) pNObs[e * O + j] = t[O + A + 1 + j];
                pTrunc[e] = t[O + A + 1 + O] > 0.5f ? 1 : 0;
            }
            batchReady = true;
        }

        // 4b. when in policy view, step the deterministic showcase field (the GPU
        //     training rollout above is unaffected — it always feeds the learner).
        if (policyView) {
            for (int base = 0; base < N_VIS; base += Trainer::kNumEnvs) {
                const int m = std::min(Trainer::kNumEnvs, N_VIS - base);
                for (int i = 0; i < m; ++i) Env::observe(showStates[base + i], &showObs[i * O]);
                trainer.policyActionsDet(showObs.data(), showAct.data(), m);
                for (int i = 0; i < m; ++i) showStates[base + i] = Env::step(showStates[base + i], &showAct[i * A]);
            }
            if (++showEp >= Env::kEpisodeSteps) {
                showEp = 0;
                for (int i = 0; i < N_VIS; ++i) showStates[i] = Env::sampleInitial(showRng);
            }
        }

        // 5. visualize: the live GPU training rollout, or the deterministic showcase
        const Quaternion qi;// identity (spheres don't rotate)
        for (int r = 0; r < R; ++r) {
            for (int i = 0; i < N_VIS; ++i) {
                const Env::State st = policyView ? showStates[i] : Env::unpackState(&stateHost[i * SSTRIDE]);
                float x0, y0, x1, y1;
                Env::rod(st, r, x0, y0, x1, y1);
                const float ax = cellX[i] + x0, ay = baseY + y0;
                const float bx = cellX[i] + x1, by = baseY + y1;
                const float ddx = bx - ax, ddy = by - ay;
                const float len = std::sqrt(ddx * ddx + ddy * ddy);
                q.setFromAxisAngle({0.f, 0.f, 1.f}, std::atan2(-ddx, ddy));
                Matrix4 m;
                m.compose({(ax + bx) / 2.f, (ay + by) / 2.f, cellZ[i]}, q, {1.f, std::max(1e-3f, len), 1.f});
                links[r]->setMatrixAt(i, m);

                // sphere at the distal joint, shown in the rest- or goal-colored mesh
                const bool solved = Env::upright(st);
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

        // 6. metrics + console learning curve
        spsTimer += dt;
        if (spsTimer >= 0.4) {
            const long sc = trainer.gradientSteps();
            gradPerSec = static_cast<float>(sc - lastGrad) / static_cast<float>(spsTimer);
            lastGrad = sc;
            spsTimer = 0.0;
        }
        if (++tick % 20 == 0) {
            const float r = deterministicEval();
            returnHistory.push_back(r);
            if (returnHistory.size() > 300) returnHistory.erase(returnHistory.begin());
            std::printf("[train:%s] grad=%5ld  trans=%8ld  detScore=%8.1f  %s\n",
                        Env::kName, trainer.gradientSteps(), trainer.transitionsCollected(), r,
                        trainer.done() ? "DONE" : (trainer.pastWarmup() ? "learning" : "warmup"));
        }

        const long sc = trainer.gradientSteps();
        const long lim = Trainer::updateBudget();
        stepHud->setText("Grad steps  " + std::to_string(sc) + " / " + std::to_string(lim) +
                         "    |    " + std::to_string(N_TOTAL) + " envs on GPU/tick");
        statusHud->setText(trainer.done() ? "TRAINED" : (trainer.pastWarmup() ? "LEARNING (GPU rollout -> CPU SAC)" : "WARMUP"));

        renderer.render(*scene, *camera);
        ui.render();
    });

    running.store(false);
    if (learnerThread.joinable()) learnerThread.join();
    return 0;
}
