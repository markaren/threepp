// RLtools x threepp — GPU rollout swarm (Stage 2, Vulkan compute).
//
// The ENTIRE rollout runs on the GPU: a single fused compute shader steps N
// pendulum environments in parallel (one thread per env), running the actor MLP
// forward + sample_and_squash + dynamics, and emits a transition each. Those
// transitions are read back and fed to the SAME CPU SAC learner (RLSwarmTrainer)
// from Stage 1 — the learner never changed; only the rollout moved to the GPU.
// Cross-vendor (Vulkan), reusing the RF-DETR VkInfer harness.
//
// Prints a self-test (GPU action == CPU forward) and a live learning curve to the
// console, so the pipeline is verifiable without watching the window.

#include "VkInfer.hpp"// rfdetr/VkInfer.hpp (added to include path)

#include "RLSwarmTrainer.hpp"
#include "pendulum_env.hpp"

#include "threepp/extras/imgui/ImguiContext.hpp"
#include "threepp/objects/TextSprite.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"
#include "threepp/threepp.hpp"

#include "threepp/renderers/vulkan/shaders/swarm_rollout.comp.spv.h"// kSwarmRolloutSpv

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
    constexpr int N_TOTAL = 256;// environments stepped on the GPU per dispatch
    constexpr int N_VIS = 256;  // visualized (instanced)
    constexpr int kCols = 16, kRows = 16;
    constexpr int WCOUNT = RLSwarmTrainer::kWeightCount;
    constexpr int H = RLSwarmTrainer::kHidden;
    constexpr float kPivotY = 0.65f, kArmLen = rlenv::kL, kDX = 1.6f, kDZ = 1.6f;

    struct PC {
        uint32_t n;
        float dt, maxTorque, maxSpeed, g, l, m;
        uint32_t episodeLimit, stochastic, warmup;
    };

    // CPU reference forward (deterministic) — for the GPU parity self-test.
    float forwardDet(const float* w, const float* obs) {
        const float* W1 = w;
        const float* b1 = w + H * 3;
        const float* W2 = b1 + H;
        const float* b2 = W2 + H * H;
        const float* W3 = b2 + H;
        const float* b3 = W3 + 2 * H;
        float h1[H], h2[H];
        for (int o = 0; o < H; ++o) {
            float s = b1[o];
            for (int i = 0; i < 3; ++i) s += W1[o * 3 + i] * obs[i];
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
        return PC{(uint32_t) N_TOTAL, rlenv::kDt, rlenv::kMaxTorque, rlenv::kMaxSpeed,
                  rlenv::kG, rlenv::kL, rlenv::kM, (uint32_t) rlenv::kEpisodeSteps, stochastic, warmup};
    }
}// namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);// unbuffered so console validation shows up live
    Canvas canvas("RLtools x threepp - GPU swarm (Vulkan compute)", {{"vsync", true}, {"size", WindowSize{1280, 800}}});
    VulkanRenderer renderer(canvas);
    renderer.outputColorSpace = ColorSpace::sRGB;

    auto scene = Scene::create();
    scene->background = Color(0x20262e);
    auto camera = PerspectiveCamera::create(50, canvas.aspect(), 0.1f, 400);
    camera->position.set(0.f, 11.f, 30.f);
    OrbitControls controls{*camera, canvas};
    controls.target.set(0.f, 0.f, 0.f);
    controls.update();

    scene->add(HemisphereLight::create(0xbfd4ff, 0x202830, 1.2f));
    auto sun = DirectionalLight::create(0xffffff, 1.3f);
    sun->position.set(6.f, 10.f, 8.f);
    scene->add(sun);
    auto grid = GridHelper::create(46, 46, 0x3a4654, 0x2c343d);
    grid->position.y = -0.6f;
    scene->add(grid);

    // instanced field
    auto armMat = MeshStandardMaterial::create({{"color", Color(0xced6e0)}, {"roughness", 0.5f}, {"metalness", 0.3f}});
    auto bobMat = MeshStandardMaterial::create({{"color", Color(0x6fd0ff)}, {"roughness", 0.35f}, {"metalness", 0.1f}});
    auto instArm = InstancedMesh::create(CylinderGeometry::create(0.02f, 0.02f, kArmLen, 8), armMat, N_VIS);
    auto instBob = InstancedMesh::create(SphereGeometry::create(0.09f, 12, 8), bobMat, N_VIS);
    instArm->instanceMatrix()->setUsage(DrawUsage::Dynamic);
    instBob->instanceMatrix()->setUsage(DrawUsage::Dynamic);
    scene->add(instArm);
    scene->add(instBob);

    std::vector<Matrix4> pivotT(N_VIS);
    for (int r = 0; r < kRows; ++r)
        for (int c = 0; c < kCols; ++c) {
            const int i = r * kCols + c;
            if (i >= N_VIS) break;
            pivotT[i].makeTranslation((c - (kCols - 1) / 2.f) * kDX, kPivotY, (r - (kRows - 1) / 2.f) * kDZ);
        }
    Matrix4 armLocal, bobLocal;
    armLocal.makeTranslation(0.f, kArmLen / 2.f, 0.f);
    bobLocal.makeTranslation(0.f, kArmLen, 0.f);

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
    titleHud->setText("RLtools x threepp - GPU rollout swarm (Vulkan compute)");
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
    auto stateBuf = vk.createOwned({(uint32_t) (N_TOTAL * 3)});
    auto noiseBuf = vk.createOwned({(uint32_t) N_TOTAL});
    auto initBuf = vk.createOwned({(uint32_t) (N_TOTAL * 2)});
    auto outBuf = vk.createOwned({(uint32_t) (N_TOTAL * 9)});
    auto pipe = vk.createPipe(kSwarmRolloutSpv, sizeof(kSwarmRolloutSpv), 5, sizeof(PC));
    const std::vector<VkBuffer> ssbos = {weightsBuf.buffer, stateBuf.buffer, noiseBuf.buffer, initBuf.buffer, outBuf.buffer};
    const uint32_t groups = (N_TOTAL + 63) / 64;

    std::mt19937 rng(1);
    std::normal_distribution<float> gauss(0.f, 1.f);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> stateHost(N_TOTAL * 3), initHost(N_TOTAL * 2), noiseHost(N_TOTAL), weightsHost(WCOUNT), outHost(N_TOTAL * 9);
    std::uniform_int_distribution<int> phase(0, rlenv::kEpisodeSteps - 1);
    auto seedStates = [&] {
        for (int e = 0; e < N_TOTAL; ++e) {
            auto s = rlenv::sampleInitial(rng);
            stateHost[e * 3 + 0] = s.theta;
            stateHost[e * 3 + 1] = s.thetaDot;
            stateHost[e * 3 + 2] = static_cast<float>(phase(rng));
        }
    };
    seedStates();
    vk.upload(stateBuf.buffer, stateHost.data(), stateHost.size() * sizeof(float));

    RLSwarmTrainer trainer(1);

    // ---- GPU<->CPU parity self-test (deterministic) ---------------------------
    trainer.extractWeights(weightsHost.data());
    vk.upload(weightsBuf.buffer, weightsHost.data(), WCOUNT * sizeof(float));
    std::fill(noiseHost.begin(), noiseHost.end(), 0.f);
    vk.upload(noiseBuf.buffer, noiseHost.data(), noiseHost.size() * sizeof(float));
    for (int e = 0; e < N_TOTAL; ++e) { initHost[e * 2] = rlenv::kPi; initHost[e * 2 + 1] = 0.f; }
    vk.upload(initBuf.buffer, initHost.data(), initHost.size() * sizeof(float));
    {
        PC pc = makePC(0, 0);
        vk.beginFrame();
        vk.dispatch(pipe, ssbos, &pc, sizeof(pc), groups, 1, 1);
        vk.endFrame();
        vk.readback(outBuf.buffer, outHost.data(), outHost.size() * sizeof(float));
        float maxDiff = 0.f;
        for (int e = 0; e < N_TOTAL; ++e) {
            const float* obs = &outHost[e * 9];
            const float gpuA = outHost[e * 9 + 3];
            maxDiff = std::fmax(maxDiff, std::fabs(gpuA - forwardDet(weightsHost.data(), obs)));
        }
        std::printf("[parity] GPU rollout vs CPU forward: max |diff| = %.2e  ->  %s\n",
                    maxDiff, maxDiff < 1e-4f ? "PARITY ok" : "MISMATCH");
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
    std::vector<float> pObs(N_TOTAL * 3), pAct(N_TOTAL), pRew(N_TOTAL), pNObs(N_TOTAL * 3);
    std::vector<unsigned char> pTrunc(N_TOTAL);
    std::vector<float> returnHistory;

    std::thread learnerThread([&] {
        std::vector<float> lObs(N_TOTAL * 3), lAct(N_TOTAL), lRew(N_TOTAL), lNObs(N_TOTAL * 3);
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

    auto deterministicEval = [&]() -> float {
        constexpr int E = 8;
        std::vector<rlenv::State> s(E);
        for (auto& st : s) st = rlenv::sampleInitial(evalRng);
        std::vector<float> o(E * 3), a(E), ret(E, 0.f);
        for (int k = 0; k < rlenv::kEpisodeSteps; ++k) {
            for (int i = 0; i < E; ++i) rlenv::observe(s[i], &o[i * 3]);
            trainer.policyActionsDet(o.data(), a.data(), E);
            for (int i = 0; i < E; ++i) {
                ret[i] += rlenv::reward(s[i], a[i]);
                s[i] = rlenv::step(s[i], a[i]);
            }
        }
        return std::accumulate(ret.begin(), ret.end(), 0.f) / E;
    };

    ImguiFunctionalContext ui(canvas, renderer, [&] {
        const auto vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos({vp->WorkPos.x + vp->WorkSize.x - 10, vp->WorkPos.y + 10}, ImGuiCond_Always, {1, 0});
        ImGui::SetNextWindowSize({340, 0}, ImGuiCond_Always);
        ImGui::Begin("RLtools - GPU rollout swarm", nullptr, ImGuiWindowFlags_NoCollapse);
        ImGui::TextWrapped("%d environments are stepped in ONE Vulkan compute dispatch (observe -> actor MLP "
                           "-> sample -> step). Transitions feed one CPU SAC learner.", N_TOTAL);
        ImGui::Separator();
        const long sc = trainer.gradientSteps();
        const long lim = RLSwarmTrainer::updateBudget();
        ImGui::ProgressBar(lim > 0 ? static_cast<float>(sc) / lim : 0.f, {-1, 0},
                           (std::to_string(sc) + " / " + std::to_string(lim) + " grad steps").c_str());
        ImGui::Text("Learner: %.0f grad/s   |   %ld transitions", gradPerSec, trainer.transitionsCollected());
        if (trainer.done()) ImGui::TextColored(ImVec4(0.22f, 0.83f, 0.29f, 1.f), "Converged.");
        else if (!trainer.pastWarmup()) ImGui::TextColored(ImVec4(1.f, 0.62f, 0.11f, 1.f), "Warmup");
        else ImGui::TextColored(ImVec4(1.f, 0.82f, 0.25f, 1.f), "Learning...");
        if (!returnHistory.empty())
            ImGui::PlotLines("##ret", returnHistory.data(), static_cast<int>(returnHistory.size()), 0,
                             "deterministic policy score (up=better)", -1700.f, 0.f, {-1, 70});
        ImGui::End();
    });

    canvas.onWindowResize([&](WindowSize size) {
        camera->aspect = size.aspect();
        camera->updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&] {
        const float dt = clock.getDelta();

        // 1. upload the latest policy + per-tick noise/initials
        trainer.extractWeights(weightsHost.data());
        const bool warm = !trainer.pastWarmup();
        for (int e = 0; e < N_TOTAL; ++e) noiseHost[e] = warm ? uni(rng) : gauss(rng);
        for (int e = 0; e < N_TOTAL; ++e) {
            auto s = rlenv::sampleInitial(rng);
            initHost[e * 2] = s.theta;
            initHost[e * 2 + 1] = s.thetaDot;
        }
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
                pObs[e * 3] = outHost[e * 9 + 0];
                pObs[e * 3 + 1] = outHost[e * 9 + 1];
                pObs[e * 3 + 2] = outHost[e * 9 + 2];
                pAct[e] = outHost[e * 9 + 3];
                pRew[e] = outHost[e * 9 + 4];
                pNObs[e * 3] = outHost[e * 9 + 5];
                pNObs[e * 3 + 1] = outHost[e * 9 + 6];
                pNObs[e * 3 + 2] = outHost[e * 9 + 7];
                pTrunc[e] = outHost[e * 9 + 8] > 0.5f ? 1 : 0;
            }
            batchReady = true;
        }

        // 5. visualize the swarm from the read-back GPU states
        for (int i = 0; i < N_VIS; ++i) {
            const float theta = stateHost[i * 3];
            Matrix4 r, pr, mm;
            r.makeRotationZ(-theta);
            pr.multiplyMatrices(pivotT[i], r);
            mm.multiplyMatrices(pr, armLocal);
            instArm->setMatrixAt(i, mm);
            mm.multiplyMatrices(pr, bobLocal);
            instBob->setMatrixAt(i, mm);
        }
        instArm->instanceMatrix()->needsUpdate();
        instBob->instanceMatrix()->needsUpdate();

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
            std::printf("[train] grad=%5ld  trans=%8ld  detScore=%8.1f  %s\n",
                        trainer.gradientSteps(), trainer.transitionsCollected(), r,
                        trainer.done() ? "DONE" : (trainer.pastWarmup() ? "learning" : "warmup"));
        }

        const long sc = trainer.gradientSteps();
        const long lim = RLSwarmTrainer::updateBudget();
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
