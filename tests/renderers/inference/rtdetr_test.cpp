// RT-DETR-L WebGPU kernel test suite.
// Covers milestones 2–11 from examples/wgpu/rtdetr_tests.cpp, rewritten
// as proper Catch2 TEST_CASEs sharing a single GPU context.
//
// Prerequisites (optional — tests SKIP gracefully if absent):
//   weights : set RTDETR_WEIGHTS env var or place at <DATA_FOLDER>/../scripts/rtdetr_l.weights
//   ref data: set RTDETR_REF    env var or place at <DATA_FOLDER>/../scripts/rtdetr_l_ref.bin
//
// Run only the no-weights milestone:
//   ./rtdetr_test "[milestone-9]"
// Run all weight-dependent milestones:
//   ./rtdetr_test "[requires-weights]"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "threepp/threepp.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"

#include "rtdetr/RtDetr.hpp"
#include "rtdetr/WeightLoader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace threepp;

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------
namespace {

std::filesystem::path weightsPath() {
    return std::filesystem::path(DATA_FOLDER) / ".." / "scripts" / "rtdetr_l.weights";
}

std::filesystem::path refPath() {
    return std::filesystem::path(DATA_FOLDER) / ".." / "scripts" / "rtdetr_l_ref.bin";
}

// Print a one-time prerequisites banner so users on fresh machines understand
// what to do when most tests SKIP. Shown on stdout during first TestCtx init.
void printSetupBanner(const std::filesystem::path& wp, bool wOk,
                      const std::filesystem::path& rp, bool rOk) {
    if (wOk && rOk) return;
    std::cout
        << "\n============================================================\n"
        << "  RT-DETR test prerequisites\n"
        << "============================================================\n"
        << "  weights : " << (wOk ? "FOUND   " : "MISSING ") << wp.string() << "\n"
        << "  ref data: " << (rOk ? "FOUND   " : "MISSING ") << rp.string() << "\n"
        << "\n"
        << "  Tests that need the missing files will SKIP (not fail).\n"
        << "\n"
        << "  To enable them:\n"
        << "    1) Install ultralytics:    pip install ultralytics\n"
        << "    2) Export the weights:     python scripts/export_rtdetr_weights.py\n"
        << "                               (produces rtdetr_l.weights)\n"
        << "    3) Capture activations:    python scripts/capture_rtdetr_activations.py\n"
        << "                               (produces rtdetr_l_ref.bin)\n"
        << "============================================================\n\n";
}

// ---------------------------------------------------------------------------
// Session-level GPU context — constructed once, shared across all TEST_CASEs.
// ---------------------------------------------------------------------------
struct TestCtx {
    Canvas        canvas;
    WgpuRenderer  renderer;
    rtdetr::RtDetr model;
    bool weightsLoaded = false;

    TestCtx()
        : canvas(Canvas::Parameters() .title("rtdetr_test").size(640, 640).headless(true))
        , renderer(canvas)
        , model(renderer)
    {
        auto wp = weightsPath();
        auto rp = refPath();
        const bool wOk = std::filesystem::exists(wp);
        const bool rOk = std::filesystem::exists(rp);
        printSetupBanner(wp, wOk, rp, rOk);
        if (wOk) {
            model.loadWeights(wp.string());
            weightsLoaded = true;
        }
    }

    static TestCtx& get() {
        static TestCtx ctx;
        return ctx;
    }
};

// ---------------------------------------------------------------------------
// CPU scalar helpers (shared across milestones)
// ---------------------------------------------------------------------------

bool allInBounds(int y, int x, int inH, int inW, int kH, int kW, int s, int p) {
    int by = y * s - p, bx = x * s - p;
    return by >= 0 && by + kH <= inH && bx >= 0 && bx + kW <= inW;
}

std::vector<float> cpuConv(const rtdetr::RtDetr& m,
                           const std::vector<float>& vIn,
                           const std::string& wKey, const std::string& bKey,
                           bool relu)
{
    const auto& w  = m.cpuWeight(wKey);
    const auto& b  = m.cpuWeight(bKey);
    const auto& sh = m.weightShape(wKey);
    uint32_t OC = sh[0], IC = sh[1], kArea = sh[2] * sh[3];
    std::vector<float> out(OC);
    for (uint32_t oc = 0; oc < OC; ++oc) {
        float s = b[oc];
        for (uint32_t ic = 0; ic < IC; ++ic) {
            const float* wp = &w[oc * IC * kArea + ic * kArea];
            for (uint32_t i = 0; i < kArea; ++i) s += vIn[ic] * wp[i];
        }
        out[oc] = relu ? std::max(0.f, s) : s;
    }
    return out;
}

std::vector<float> cpuDwConv(const rtdetr::RtDetr& m,
                             const std::vector<float>& vIn,
                             const std::string& wKey, const std::string& bKey,
                             bool relu)
{
    const auto& w  = m.cpuWeight(wKey);
    const auto& b  = m.cpuWeight(bKey);
    const auto& sh = m.weightShape(wKey);
    uint32_t C = sh[0], kArea = sh[2] * sh[3];
    std::vector<float> out(C);
    for (uint32_t c = 0; c < C; ++c) {
        float ks = 0.f;
        for (uint32_t i = 0; i < kArea; ++i) ks += w[c * kArea + i];
        float s = vIn[c] * ks + b[c];
        out[c] = relu ? std::max(0.f, s) : s;
    }
    return out;
}

std::vector<float> cpuLightConv(const rtdetr::RtDetr& m,
                                const std::vector<float>& vIn,
                                const std::string& prefix)
{
    auto mid = cpuConv(m, vIn, prefix + ".conv1.fused.weight",
                                prefix + ".conv1.fused.bias", false);
    return cpuDwConv(m, mid,   prefix + ".conv2.fused.weight",
                                prefix + ".conv2.fused.bias", true);
}

std::vector<float> cpuHGBlock(const rtdetr::RtDetr& m,
                              const std::vector<float>& vIn,
                              const std::string& prefix,
                              int n, bool shortcut, bool lightconv)
{
    std::vector<std::vector<float>> branches;
    branches.reserve(n);
    const std::vector<float>* prev = &vIn;
    for (int i = 0; i < n; ++i) {
        std::string mp = prefix + ".m." + std::to_string(i);
        if (lightconv)
            branches.push_back(cpuLightConv(m, *prev, mp));
        else
            branches.push_back(cpuConv(m, *prev, mp + ".fused.weight",
                                                  mp + ".fused.bias", true));
        prev = &branches.back();
    }
    uint32_t cIn = static_cast<uint32_t>(vIn.size());
    uint32_t cBr = static_cast<uint32_t>(branches[0].size());
    std::vector<float> cat(cIn + n * cBr);
    std::copy(vIn.begin(), vIn.end(), cat.begin());
    for (int i = 0; i < n; ++i)
        std::copy(branches[i].begin(), branches[i].end(),
                  cat.begin() + cIn + i * cBr);
    auto sc = cpuConv(m, cat, prefix + ".sc.fused.weight", prefix + ".sc.fused.bias", true);
    auto ec = cpuConv(m, sc,  prefix + ".ec.fused.weight", prefix + ".ec.fused.bias", true);
    if (shortcut)
        for (size_t i = 0; i < ec.size(); ++i) ec[i] += vIn[i];
    return ec;
}

// Per-element diff against a PyTorch reference tensor.
// Returns number of elements exceeding (atol + rtol * |ref|).
struct DiffResult { size_t fail; float maxAbs; float maxRel; };

DiffResult tensorDiff(const std::vector<float>& got, const std::vector<float>& exp,
                      float atol, float rtol)
{
    DiffResult r{0, 0.f, 0.f};
    if (got.size() != exp.size()) { r.fail = got.size() + exp.size(); return r; }
    for (size_t i = 0; i < got.size(); ++i) {
        float err = std::abs(got[i] - exp[i]);
        float tol = atol + rtol * std::abs(exp[i]);
        if (err > tol) ++r.fail;
        r.maxAbs = std::max(r.maxAbs, err);
        r.maxRel = std::max(r.maxRel, err / std::max(1e-6f, std::abs(exp[i])));
    }
    return r;
}

// Interior-pixel mismatch counter for convolution output tensors.
struct InteriorResult { size_t checked; size_t mismatch; float maxAbsErr; };

InteriorResult interiorCheck(const std::vector<float>& gpu,
                             uint32_t OC, uint32_t OH, uint32_t OW,
                             int inH, int inW, int kH, int kW, int stride, int pad,
                             const std::vector<float>& expectedPerChannel,
                             float relTol)
{
    InteriorResult r{0, 0, 0.f};
    for (uint32_t c = 0; c < OC; ++c) {
        for (uint32_t y = 0; y < OH; ++y) {
            for (uint32_t x = 0; x < OW; ++x) {
                if (!allInBounds(int(y), int(x), inH, inW, kH, kW, stride, pad)) continue;
                float got = gpu[c * OH * OW + y * OW + x];
                float exp = expectedPerChannel[c];
                float err = std::abs(got - exp);
                r.maxAbsErr = std::max(r.maxAbsErr, err);
                ++r.checked;
                if (err > relTol * std::max(1.f, std::abs(exp))) ++r.mismatch;
            }
        }
    }
    return r;
}

} // namespace

// ===========================================================================
// Milestone 9 — MSDeformAttn analytical (no weights needed)
// ===========================================================================
TEST_CASE("MSDeformAttn analytical", "[rtdetr][milestone-9]") {
    auto& ctx = TestCtx::get();

    const uint32_t H = 2, d = 2, D = H * d;
    const uint32_t L = 2, P = 1, Nq = 1;
    const uint32_t totalTokens = 5;  // level0: 2×2=4, level1: 1×1=1

    std::vector<float> valData(totalTokens * D);
    for (uint32_t i = 0; i < totalTokens * D; ++i) valData[i] = float(i + 1);
    auto value  = ctx.model.uploadTensor({totalTokens, D}, valData.data());

    std::vector<std::pair<uint32_t,uint32_t>> shapes = {{2,2}, {1,1}};

    std::vector<float> refData = {0.5f, 0.5f};
    auto refPts = ctx.model.uploadTensor({Nq, 2u}, refData.data());

    std::vector<float> offData(Nq * H * L * P * 2, 0.0f);
    auto offs = ctx.model.uploadTensor({Nq, H * L * P * 2}, offData.data());

    std::vector<float> awData(Nq * H * L * P, 0.5f);
    auto aw = ctx.model.uploadTensor({Nq, H * L * P}, awData.data());

    auto out    = ctx.model.msDeformAttn_(value, shapes, refPts, offs, aw, H);
    auto outCpu = ctx.model.readback(out.buffer(), out.numel());

    // With ref=(0.5,0.5), zero offsets, level0 2×2 bilinear avg of tokens 1–4 per dim,
    // level1 1×1 token at top-left. Output[c] = 0.5*(c+7) + 0.5*(c+17) = c+12.
    std::vector<float> expected = {12.0f, 13.0f, 14.0f, 15.0f};
    REQUIRE(outCpu.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        INFO("element " << i << ": got=" << outCpu[i] << " exp=" << expected[i]);
        CHECK(std::abs(outCpu[i] - expected[i]) <= 1e-4f);
    }
}

// ===========================================================================
// Milestone 2 — DWConv analytical (requires weights)
// ===========================================================================
TEST_CASE("DWConv analytical", "[rtdetr][requires-weights][milestone-2]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found — set RTDETR_WEIGHTS or run export script");

    const std::string wKey = "model.2.fused.weight";
    const std::string bKey = "model.2.fused.bias";
    const auto& wShape = ctx.model.weightShape(wKey);  // [128, 1, 3, 3]
    const uint32_t C = wShape[0];
    const int kH = int(wShape[2]), kW = int(wShape[3]);
    const int sH = 2, pH = 1;

    const uint32_t inH = 16, inW = 16;
    auto xInput = ctx.model.allocFilled({C, inH, inW}, 1.0f);
    auto yOut   = ctx.model.dwConv_(xInput, wKey, bKey, sH, sH, pH, pH, rtdetr::Activation::None);

    const uint32_t outH = yOut.H(), outW = yOut.W();
    std::vector<float> gpu = ctx.model.readback(yOut.buffer(), yOut.numel());

    const auto& cpuW = ctx.model.cpuWeight(wKey);
    const auto& cpuB = ctx.model.cpuWeight(bKey);
    std::vector<float> expected(C);
    for (uint32_t c = 0; c < C; ++c) {
        float s = 0.f;
        for (int i = 0; i < kH * kW; ++i) s += cpuW[c * kH * kW + i];
        expected[c] = s + cpuB[c];
    }

    auto r = interiorCheck(gpu, C, outH, outW, int(inH), int(inW),
                           kH, kW, sH, pH, expected, 1e-2f);
    INFO("interior checked=" << r.checked << " mismatch=" << r.mismatch
         << " maxAbsErr=" << r.maxAbsErr);
    CHECK(r.mismatch == 0);
}

// ===========================================================================
// Milestone 3a — Conv2D analytical (requires weights)
// ===========================================================================
TEST_CASE("Conv2D analytical", "[rtdetr][requires-weights][milestone-3a]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");

    const std::string wKey = "model.0.stem1.fused.weight";
    const std::string bKey = "model.0.stem1.fused.bias";
    const auto& sh = ctx.model.weightShape(wKey);  // [32, 3, 3, 3]
    const uint32_t OC = sh[0], IC = sh[1];
    const int kH = int(sh[2]), kW = int(sh[3]);
    const int sH = 2, pH = 1;

    const uint32_t inH = 32, inW = 32;
    auto cIn  = ctx.model.allocFilled({IC, inH, inW}, 1.0f);
    auto cOut = ctx.model.conv_(cIn, wKey, bKey, sH, sH, pH, pH, rtdetr::Activation::None);

    const uint32_t outH = cOut.H(), outW = cOut.W();
    std::vector<float> gpu = ctx.model.readback(cOut.buffer(), cOut.numel());

    const auto& cpuW = ctx.model.cpuWeight(wKey);
    const auto& cpuB = ctx.model.cpuWeight(bKey);
    std::vector<float> expected(OC);
    for (uint32_t c = 0; c < OC; ++c) {
        float s = cpuB[c];
        uint32_t base = c * IC * uint32_t(kH * kW);
        for (uint32_t i = 0; i < IC * uint32_t(kH * kW); ++i) s += cpuW[base + i];
        expected[c] = s;
    }

    auto r = interiorCheck(gpu, OC, outH, outW, int(inH), int(inW),
                           kH, kW, sH, pH, expected, 1e-2f);
    INFO("interior checked=" << r.checked << " mismatch=" << r.mismatch
         << " maxAbsErr=" << r.maxAbsErr);
    CHECK(r.mismatch == 0);
}

// ===========================================================================
// Milestone 3b — MaxPool2D (requires weights: just needs model GPU context)
// ===========================================================================
TEST_CASE("MaxPool2D no padding", "[rtdetr][requires-weights][milestone-3b]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");

    const uint32_t mC = 4, mH = 8, mW = 8;
    std::vector<float> buf(mC * mH * mW);
    for (uint32_t c = 0; c < mC; ++c)
        for (uint32_t y = 0; y < mH; ++y)
            for (uint32_t x = 0; x < mW; ++x)
                buf[c * mH * mW + y * mW + x] = float(y * mW + x);
    auto mIn  = ctx.model.uploadTensor({mC, mH, mW}, buf.data());
    auto mOut = ctx.model.maxPool_(mIn, 2, 2, 1, 1, 0, 0, 0, 0);
    auto gpu  = ctx.model.readback(mOut.buffer(), mOut.numel());

    const uint32_t oH = mOut.H(), oW = mOut.W();
    size_t miss = 0;
    for (uint32_t c = 0; c < mC; ++c)
        for (uint32_t y = 0; y < oH; ++y)
            for (uint32_t x = 0; x < oW; ++x) {
                float exp = float((y + 1) * mW + (x + 1));
                if (std::abs(gpu[c * oH * oW + y * oW + x] - exp) > 1e-5f) ++miss;
            }
    CHECK(miss == 0);
}

TEST_CASE("MaxPool2D asymmetric padding", "[rtdetr][requires-weights][milestone-3b]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");

    const uint32_t mC = 4, mH = 8, mW = 8;
    std::vector<float> buf(mC * mH * mW);
    for (uint32_t c = 0; c < mC; ++c)
        for (uint32_t y = 0; y < mH; ++y)
            for (uint32_t x = 0; x < mW; ++x)
                buf[c * mH * mW + y * mW + x] = float(y * mW + x);
    auto mIn   = ctx.model.uploadTensor({mC, mH, mW}, buf.data());
    auto mOut2 = ctx.model.maxPool_(mIn, 2, 2, 1, 1, 0, 0, 1, 1);
    auto gpu2  = ctx.model.readback(mOut2.buffer(), mOut2.numel());

    // Asymmetric pad [0,0,1,1] → output dims match input dims.
    CHECK(mOut2.H() == mH);
    CHECK(mOut2.W() == mW);
    float brGot = gpu2[0 * mH * mW + (mH - 1) * mW + (mW - 1)];
    float brExp = float((mH - 1) * mW + (mW - 1));
    CHECK(std::abs(brGot - brExp) < 1e-5f);
}

// ===========================================================================
// Milestone 3c — ConcatC (requires weights: just needs GPU context)
// ===========================================================================
TEST_CASE("ConcatC channel axis", "[rtdetr][requires-weights][milestone-3c]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");

    const uint32_t Ah = 4, Aw = 4;
    std::vector<float> aBuf(3 * Ah * Aw), bBuf(2 * Ah * Aw);
    for (uint32_t c = 0; c < 3; ++c)
        for (uint32_t i = 0; i < Ah * Aw; ++i) aBuf[c * Ah * Aw + i] = 10.f + float(c);
    for (uint32_t c = 0; c < 2; ++c)
        for (uint32_t i = 0; i < Ah * Aw; ++i) bBuf[c * Ah * Aw + i] = 100.f + float(c);

    auto aT = ctx.model.uploadTensor({3, Ah, Aw}, aBuf.data());
    auto bT = ctx.model.uploadTensor({2, Ah, Aw}, bBuf.data());
    auto cT = ctx.model.concatC_(aT, bT);
    auto g  = ctx.model.readback(cT.buffer(), cT.numel());

    REQUIRE(cT.C() == 5);
    const float expVals[5] = {10, 11, 12, 100, 101};
    for (uint32_t c = 0; c < 5; ++c)
        for (uint32_t i = 0; i < Ah * Aw; ++i)
            CHECK(std::abs(g[c * Ah * Aw + i] - expVals[c]) <= 1e-5f);
}

// ===========================================================================
// Milestone 3d — HGStem end-to-end (requires weights)
// ===========================================================================
TEST_CASE("HGStem end-to-end", "[rtdetr][requires-weights][milestone-3d]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");

    auto& m = ctx.model;
    std::vector<float> v0(3, 1.0f);
    auto v1  = cpuConv(m, v0,  "model.0.stem1.fused.weight",  "model.0.stem1.fused.bias",  true);
    auto v2a = cpuConv(m, v1,  "model.0.stem2a.fused.weight", "model.0.stem2a.fused.bias", true);
    auto v2b = cpuConv(m, v2a, "model.0.stem2b.fused.weight", "model.0.stem2b.fused.bias", true);
    std::vector<float> vCat(v1.size() + v2b.size());
    std::copy(v1.begin(), v1.end(), vCat.begin());
    std::copy(v2b.begin(), v2b.end(), vCat.begin() + v1.size());
    auto v3   = cpuConv(m, vCat, "model.0.stem3.fused.weight", "model.0.stem3.fused.bias", true);
    auto vExp = cpuConv(m, v3,   "model.0.stem4.fused.weight", "model.0.stem4.fused.bias", true);

    const uint32_t inSz = 64;
    auto xIn  = m.allocFilled({3, inSz, inSz}, 1.0f);
    auto yOut = m.hgStem_(xIn);
    auto gpu  = m.readback(yOut.buffer(), yOut.numel());
    uint32_t OC = yOut.C(), OH = yOut.H(), OW = yOut.W();

    const uint32_t margin = 2;
    size_t mismatch = 0;
    float maxErr = 0.f;
    for (uint32_t c = 0; c < OC; ++c)
        for (uint32_t y = margin; y + margin < OH; ++y)
            for (uint32_t x = margin; x + margin < OW; ++x) {
                float err = std::abs(gpu[c * OH * OW + y * OW + x] - vExp[c]);
                maxErr = std::max(maxErr, err);
                if (err > 5e-2f * std::max(1.f, std::abs(vExp[c]))) ++mismatch;
            }
    INFO("mismatch=" << mismatch << " maxErr=" << maxErr);
    CHECK(mismatch == 0);
}

// ===========================================================================
// Milestone 4 — HGBlock end-to-end (requires weights)
// ===========================================================================
TEST_CASE("HGBlock end-to-end", "[rtdetr][requires-weights][milestone-4]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");

    auto& m = ctx.model;
    std::vector<float> v0(48, 1.0f);
    auto vExp = cpuHGBlock(m, v0, "model.1", 6, false, false);

    const uint32_t inSz = 24;
    auto xIn  = m.allocFilled({48, inSz, inSz}, 1.0f);
    auto yOut = m.hgBlock_(xIn, "model.1", 6, 3, false);
    auto gpu  = m.readback(yOut.buffer(), yOut.numel());
    uint32_t OC = yOut.C(), OH = yOut.H(), OW = yOut.W();

    const uint32_t margin = 7;
    size_t mismatch = 0; float maxErr = 0.f;
    for (uint32_t c = 0; c < OC; ++c)
        for (uint32_t y = margin; y + margin < OH; ++y)
            for (uint32_t x = margin; x + margin < OW; ++x) {
                float err = std::abs(gpu[c * OH * OW + y * OW + x] - vExp[c]);
                maxErr = std::max(maxErr, err);
                if (err > 1e-1f * std::max(1.f, std::abs(vExp[c]))) ++mismatch;
            }
    INFO("mismatch=" << mismatch << " maxErr=" << maxErr);
    CHECK(mismatch == 0);
}

// ===========================================================================
// Milestone 5a — LightConv isolated (requires weights)
// ===========================================================================
TEST_CASE("LightConv isolated", "[rtdetr][requires-weights][milestone-5a]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");

    auto& m = ctx.model;
    const auto& w1sh = m.weightShape("model.5.m.0.conv1.fused.weight");  // [192, 512, 1, 1]
    uint32_t IC = w1sh[1];

    std::vector<float> vIn(IC, 1.0f);
    auto vExp = cpuLightConv(m, vIn, "model.5.m.0");

    const uint32_t inSz = 20;
    auto xIn  = m.allocFilled({IC, inSz, inSz}, 1.0f);
    auto yOut = m.lightConv_(xIn, "model.5.m.0", 5);
    auto gpu  = m.readback(yOut.buffer(), yOut.numel());
    uint32_t OC = yOut.C(), OH = yOut.H(), OW = yOut.W();

    const uint32_t margin = 3;
    size_t mismatch = 0; float maxErr = 0.f;
    for (uint32_t c = 0; c < OC; ++c)
        for (uint32_t y = margin; y + margin < OH; ++y)
            for (uint32_t x = margin; x + margin < OW; ++x) {
                float err = std::abs(gpu[c * OH * OW + y * OW + x] - vExp[c]);
                maxErr = std::max(maxErr, err);
                if (err > 5e-2f * std::max(1.f, std::abs(vExp[c]))) ++mismatch;
            }
    INFO("mismatch=" << mismatch << " maxErr=" << maxErr);
    CHECK(mismatch == 0);
}

// ===========================================================================
// Milestone 5b — HGBlockLight (requires weights)
// ===========================================================================
static size_t runHGBlockLight(rtdetr::RtDetr& m, const std::string& prefix,
                              uint32_t inC, uint32_t inSz, int n, bool shortcut,
                              uint32_t margin)
{
    std::vector<float> vIn(inC, 1.0f);
    auto vExp = cpuHGBlock(m, vIn, prefix, n, shortcut, true);

    auto xIn  = m.allocFilled({inC, inSz, inSz}, 1.0f);
    auto y    = m.hgBlock_(xIn, prefix, n, 5, shortcut, true);
    auto gpu  = m.readback(y.buffer(), y.numel());
    uint32_t OC = y.C(), OH = y.H(), OW = y.W();

    size_t mismatch = 0;
    for (uint32_t c = 0; c < OC; ++c)
        for (uint32_t yy = margin; yy + margin < OH; ++yy)
            for (uint32_t xx = margin; xx + margin < OW; ++xx) {
                float err = std::abs(gpu[c * OH * OW + yy * OW + xx] - vExp[c]);
                if (err > 1e-1f * std::max(1.f, std::abs(vExp[c]))) ++mismatch;
            }
    return mismatch;
}

TEST_CASE("HGBlockLight model.5 no shortcut", "[rtdetr][requires-weights][milestone-5b]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");
    CHECK(runHGBlockLight(ctx.model, "model.5", 512, 32, 6, false, 13) == 0);
}

TEST_CASE("HGBlockLight model.6 with shortcut", "[rtdetr][requires-weights][milestone-5b]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");
    CHECK(runHGBlockLight(ctx.model, "model.6", 1024, 32, 6, true, 13) == 0);
}

// ===========================================================================
// Milestone 6a — Linear/GEMM (requires weights)
// ===========================================================================
TEST_CASE("Linear GEMM", "[rtdetr][requires-weights][milestone-6a]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");

    auto& m = ctx.model;
    const std::string wKey = "model.11.fc1.weight";
    const std::string bKey = "model.11.fc1.bias";
    const auto& wSh = m.weightShape(wKey);
    uint32_t N = wSh[0], K = wSh[1];

    std::vector<float> xOnes(K, 1.0f);
    auto xIn = m.uploadTensor({1, K}, xOnes.data());
    auto y   = m.linear_(xIn, wKey, bKey);
    auto yg  = m.readback(y.buffer(), y.numel());

    const auto& wCpu = m.cpuWeight(wKey);
    const auto& bCpu = m.cpuWeight(bKey);
    size_t fail = 0; float maxErr = 0.f;
    for (uint32_t n = 0; n < N; ++n) {
        double s = bCpu[n];
        for (uint32_t k = 0; k < K; ++k) s += wCpu[n * K + k];
        float err = std::abs(yg[n] - float(s));
        maxErr = std::max(maxErr, err);
        if (err > 1e-3f * std::max(1.f, std::abs(float(s)))) ++fail;
    }
    INFO("fail=" << fail << " maxErr=" << maxErr);
    CHECK(fail == 0);
}

// ===========================================================================
// Milestone 6b — LayerNorm (requires weights)
// ===========================================================================
TEST_CASE("LayerNorm all-ones input", "[rtdetr][requires-weights][milestone-6b]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");

    auto& m = ctx.model;
    const std::string wKey = "model.11.norm1.weight";
    const std::string bKey = "model.11.norm1.bias";
    uint32_t D = m.weightShape(wKey)[0];

    std::vector<float> xOnes(D, 1.0f);
    auto xIn = m.uploadTensor({1, D}, xOnes.data());
    auto y   = m.layerNorm_(xIn, wKey, bKey);
    auto yg  = m.readback(y.buffer(), y.numel());

    // All-ones input → mean=1, subtract → zero → output = bias.
    const auto& bCpu = m.cpuWeight(bKey);
    size_t fail = 0; float maxErr = 0.f;
    for (uint32_t c = 0; c < D; ++c) {
        float err = std::abs(yg[c] - bCpu[c]);
        maxErr = std::max(maxErr, err);
        if (err > 1e-5f) ++fail;
    }
    INFO("fail=" << fail << " maxErr=" << maxErr);
    CHECK(fail == 0);
}

// ===========================================================================
// Milestone 6c — Softmax (requires weights: just needs GPU context)
// ===========================================================================
TEST_CASE("Softmax uniform input", "[rtdetr][requires-weights][milestone-6c]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");

    const uint32_t M = 3, N = 400;
    std::vector<float> xOnes(M * N, 1.0f);
    auto xIn = ctx.model.uploadTensor({M, N}, xOnes.data());
    auto y   = ctx.model.softmaxLast_(xIn);
    auto yg  = ctx.model.readback(y.buffer(), y.numel());

    float expected = 1.f / float(N);
    size_t fail = 0; float maxErr = 0.f;
    for (float v : yg) {
        float err = std::abs(v - expected);
        maxErr = std::max(maxErr, err);
        if (err > 1e-6f) ++fail;
    }
    INFO("fail=" << fail << " maxErr=" << maxErr);
    CHECK(fail == 0);
}

// ===========================================================================
// Milestone 6d — GELU at known values (requires weights: just needs GPU context)
// ===========================================================================
TEST_CASE("GELU known values", "[rtdetr][requires-weights][milestone-6d]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");

    std::vector<float> xv  = {-2.f, -1.f,  0.f,         1.f,         2.f};
    std::vector<float> ref = {-0.04550026f, -0.15865525f, 0.f, 0.84134475f, 1.95449974f};

    auto xIn = ctx.model.uploadTensor({uint32_t(xv.size())}, xv.data());
    auto y   = ctx.model.gelu_(xIn);
    auto yg  = ctx.model.readback(y.buffer(), y.numel());

    for (size_t i = 0; i < xv.size(); ++i) {
        INFO("gelu(" << xv[i] << "): got=" << yg[i] << " ref=" << ref[i]);
        CHECK(std::abs(yg[i] - ref[i]) <= 2e-6f);
    }
}

// ===========================================================================
// Milestones 5d / 6e / 7 / 8a / 8b / 8c — strict PyTorch reference
// ===========================================================================
TEST_CASE("Backbone strict PyTorch reference", "[rtdetr][requires-ref][milestone-5d]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");
    auto rp = refPath();
    if (!std::filesystem::exists(rp)) SKIP("reference file not found — run capture_rtdetr_activations.py");

    auto& m  = ctx.model;
    auto ref = rtdetr::parseWeightBinary(rp.string());

    const auto& inData  = ref.data.at("input");
    const auto& inShape = ref.shapes.at("input");
    auto xIn = m.uploadTensor({inShape[0], inShape[1], inShape[2]}, inData.data());
    auto bb  = m.backbone_(xIn);

    auto p3g = m.readback(bb.p3.buffer(), bb.p3.numel());
    auto p4g = m.readback(bb.p4.buffer(), bb.p4.numel());
    auto p5g = m.readback(bb.p5.buffer(), bb.p5.numel());

    auto p3d = tensorDiff(p3g, ref.data.at("model.3"),  1e-2f, 1e-2f);
    auto p4d = tensorDiff(p4g, ref.data.at("model.7"),  1e-2f, 1e-2f);
    auto p5d = tensorDiff(p5g, ref.data.at("model.10"), 1e-2f, 1e-2f);

    INFO("P3: fail=" << p3d.fail << " maxAbs=" << p3d.maxAbs);
    INFO("P4: fail=" << p4d.fail << " maxAbs=" << p4d.maxAbs);
    INFO("P5: fail=" << p5d.fail << " maxAbs=" << p5d.maxAbs);
    CHECK(p3d.fail == 0);
    CHECK(p4d.fail == 0);
    CHECK(p5d.fail == 0);
}

TEST_CASE("AIFI strict PyTorch reference", "[rtdetr][requires-ref][milestone-6e]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");
    auto rp = refPath();
    if (!std::filesystem::exists(rp)) SKIP("reference file not found");

    auto& m  = ctx.model;
    auto ref = rtdetr::parseWeightBinary(rp.string());

    const auto& sh = ref.shapes.at("model.10");
    auto xIn = m.uploadTensor({sh[0], sh[1], sh[2]}, ref.data.at("model.10").data());
    auto yOut = m.aifi_(xIn);
    auto yg   = m.readback(yOut.buffer(), yOut.numel());

    auto d = tensorDiff(yg, ref.data.at("model.11"), 1e-2f, 1e-2f);
    INFO("AIFI: fail=" << d.fail << " maxAbs=" << d.maxAbs);
    CHECK(d.fail == 0);
}

TEST_CASE("CCFM neck strict PyTorch reference", "[rtdetr][requires-ref][milestone-7]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");
    auto rp = refPath();
    if (!std::filesystem::exists(rp)) SKIP("reference file not found");

    auto& m  = ctx.model;
    auto ref = rtdetr::parseWeightBinary(rp.string());

    auto upload = [&](const std::string& key) {
        const auto& sh = ref.shapes.at(key);
        return m.uploadTensor({sh[0], sh[1], sh[2]}, ref.data.at(key).data());
    };
    auto p3in = upload("model.3");
    auto p4in = upload("model.7");
    auto f5in = upload("model.11");

    auto neck = m.ccfm_(p3in, p4in, f5in);
    auto s3g  = m.readback(neck.s3.buffer(), neck.s3.numel());
    auto s4g  = m.readback(neck.s4.buffer(), neck.s4.numel());
    auto s5g  = m.readback(neck.s5.buffer(), neck.s5.numel());

    auto d3 = tensorDiff(s3g, ref.data.at("model.21"), 2e-2f, 2e-2f);
    auto d4 = tensorDiff(s4g, ref.data.at("model.24"), 2e-2f, 2e-2f);
    auto d5 = tensorDiff(s5g, ref.data.at("model.27"), 2e-2f, 2e-2f);
    INFO("S3: fail=" << d3.fail << " maxAbs=" << d3.maxAbs);
    INFO("S4: fail=" << d4.fail << " maxAbs=" << d4.maxAbs);
    INFO("S5: fail=" << d5.fail << " maxAbs=" << d5.maxAbs);
    CHECK(d3.fail == 0);
    CHECK(d4.fail == 0);
    CHECK(d5.fail == 0);
}

TEST_CASE("Decoder weight inventory preflight", "[rtdetr][requires-ref][milestone-8a]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");
    auto rp = refPath();
    if (!std::filesystem::exists(rp)) SKIP("reference file not found");

    auto& m  = ctx.model;
    size_t miss = 0, badShape = 0;

    auto check = [&](const std::string& key, std::vector<uint32_t> expected) {
        if (!m.hasWeight(key)) { ++miss; UNSCOPED_INFO("MISSING: " << key); return; }
        const auto& sh = m.weightShape(key);
        if (sh != expected) { ++badShape; UNSCOPED_INFO("BAD SHAPE: " << key); }
    };

    // RT-DETR-L decoder: d=256, heads=8, levels=3, points=4, queries=300, classes=80
    for (int l = 0; l < 6; ++l) {
        std::string pfx = "model.28.decoder.layers." + std::to_string(l);
        check(pfx + ".self_attn.in_proj_weight",        {768, 256});
        check(pfx + ".self_attn.in_proj_bias",          {768});
        check(pfx + ".self_attn.out_proj.weight",       {256, 256});
        check(pfx + ".cross_attn.sampling_offsets.weight", {192, 256});
        check(pfx + ".cross_attn.attention_weights.weight",{96, 256});
        check(pfx + ".cross_attn.value_proj.weight",    {256, 256});
        check(pfx + ".cross_attn.output_proj.weight",   {256, 256});
        check(pfx + ".linear1.weight",                  {1024, 256});
        check(pfx + ".linear2.weight",                  {256, 1024});
        check(pfx + ".norm1.weight",                    {256});
        check(pfx + ".norm2.weight",                    {256});
        check(pfx + ".norm3.weight",                    {256});
    }
    check("model.28.dec_score_head.5.weight", {80, 256});
    check("model.28.dec_bbox_head.5.layers.0.weight", {256, 256});

    INFO("missing=" << miss << " badShape=" << badShape);
    CHECK(miss == 0);
    CHECK(badShape == 0);
}

TEST_CASE("Decoder input_proj and enc_output strict ref", "[rtdetr][requires-ref][milestone-8bc]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");
    auto rp = refPath();
    if (!std::filesystem::exists(rp)) SKIP("reference file not found");

    auto& m  = ctx.model;
    auto ref = rtdetr::parseWeightBinary(rp.string());

    if (!ref.data.count("decoder.input_proj.0")) SKIP("ref missing decoder.input_proj tensors");

    auto uploadSh = [&](const std::string& key) {
        const auto& sh = ref.shapes.at(key);
        return m.uploadTensor({sh[0], sh[1], sh[2]}, ref.data.at(key).data());
    };

    // 8b: input_proj per scale
    {
        const char* srcKeys[3] = {"model.21", "model.24", "model.27"};
        const char* dstKeys[3] = {"decoder.input_proj.0","decoder.input_proj.1","decoder.input_proj.2"};
        for (int i = 0; i < 3; ++i) {
            auto x   = uploadSh(srcKeys[i]);
            auto y   = m.inputProj_(x, i);
            auto yg  = m.readback(y.buffer(), y.numel());
            auto d   = tensorDiff(yg, ref.data.at(dstKeys[i]), 2e-2f, 2e-2f);
            INFO("input_proj." << i << " fail=" << d.fail << " maxAbs=" << d.maxAbs);
            CHECK(d.fail == 0);
        }
    }

    if (!ref.data.count("decoder.memory")) SKIP("ref missing decoder.memory tensor");

    // 8c: memory assembly + enc_output
    {
        auto ip0 = uploadSh("decoder.input_proj.0");
        auto ip1 = uploadSh("decoder.input_proj.1");
        auto ip2 = uploadSh("decoder.input_proj.2");
        auto mem = m.buildMemory_(ip0, ip1, ip2);
        auto mg  = m.readback(mem.buffer(), mem.numel());
        auto dm  = tensorDiff(mg, ref.data.at("decoder.memory"), 1e-4f, 1e-4f);
        INFO("memory fail=" << dm.fail << " maxAbs=" << dm.maxAbs);
        CHECK(dm.fail == 0);

        auto lin = m.linear_(mem, "model.28.enc_output.0.weight", "model.28.enc_output.0.bias");
        auto ln  = m.layerNorm_(lin, "model.28.enc_output.1.weight", "model.28.enc_output.1.bias");
        auto lng = m.readback(ln.buffer(), ln.numel());
        auto de  = tensorDiff(lng, ref.data.at("decoder.enc_output"), 2e-3f, 2e-3f);
        INFO("enc_output fail=" << de.fail << " maxAbs=" << de.maxAbs);
        CHECK(de.fail == 0);
    }
}

// ===========================================================================
// Milestone 10 — Full decoder forward pass smoke test (requires weights)
// ===========================================================================
TEST_CASE("Full decoder forward smoke test", "[rtdetr][requires-weights][milestone-10]") {
    auto& ctx = TestCtx::get();
    if (!ctx.weightsLoaded) SKIP("weight file not found");

    auto& m   = ctx.model;
    auto input = m.allocFilled({3, 640, 640}, 0.5f);
    auto bp    = m.backbone_(input);
    auto f5    = m.aifi_(bp.p5);
    auto neck  = m.ccfm_(bp.p3, bp.p4, f5);
    auto ip0   = m.inputProj_(neck.s3, 0);
    auto ip1   = m.inputProj_(neck.s4, 1);
    auto ip2   = m.inputProj_(neck.s5, 2);
    auto mem   = m.buildMemory_(ip0, ip1, ip2);
    auto lin   = m.linear_(mem, "model.28.enc_output.0.weight", "model.28.enc_output.0.bias");
    auto enc   = m.layerNorm_(lin, "model.28.enc_output.1.weight", "model.28.enc_output.1.bias");

    std::vector<std::pair<uint32_t,uint32_t>> shapes = {
        {ip0.H(), ip0.W()}, {ip1.H(), ip1.W()}, {ip2.H(), ip2.W()}
    };
    auto result = m.decoder_(mem, enc, shapes);

    REQUIRE(result.scores.size() == 300 * 80);
    REQUIRE(result.bboxes.size() == 300 * 4);

    size_t oob = 0;
    for (float v : result.bboxes)
        if (v <= 0.f || v >= 1.f) ++oob;
    INFO("bbox values out of (0,1): " << oob);
    CHECK(oob == 0);
}
