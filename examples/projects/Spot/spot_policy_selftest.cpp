// Parity self-test for SpotPolicy: loads the exported weights (spot_policy.tpnn)
// and the golden obs->action pairs (spot_policy_ref.bin, from the torch forward),
// runs the C++ forward on each obs, and reports the max abs error vs torch. This
// proves the native path reproduces the trained policy before it ever drives a
// robot. Pure std + SpotPolicy.hpp — no torch, no threepp, no GL/Vulkan.
//
//   spot_policy_selftest [dir]   # dir holds the two files (default: cwd)
//   exit 0 = parity within tolerance, 1 = mismatch / load failure.

#include "SpotPolicy.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

    std::uint32_t readU32(std::ifstream& f) {
        std::uint32_t v = 0;
        f.read(reinterpret_cast<char*>(&v), 4);
        return v;
    }

}// namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? std::string(argv[1]) : std::string(".");
    const std::string tpnn = dir + "/spot_policy.tpnn";
    const std::string refp = dir + "/spot_policy_ref.bin";

    spot::SpotPolicy policy;
    try {
        policy = spot::SpotPolicy::load(tpnn);
    } catch (const std::exception& e) {
        std::cerr << "load failed: " << e.what() << "\n";
        return 1;
    }
    std::cout << "[selftest] loaded " << tpnn << "  in=" << policy.inputDim()
              << " out=" << policy.outputDim() << "\n";

    std::ifstream f(refp, std::ios::binary);
    if (!f) {
        std::cerr << "cannot open " << refp << "\n";
        return 1;
    }
    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "TPR1") {
        std::cerr << "bad ref magic\n";
        return 1;
    }
    const std::uint32_t n = readU32(f), inDim = readU32(f), outDim = readU32(f);
    if (inDim != policy.inputDim() || outDim != policy.outputDim()) {
        std::cerr << "ref/policy dim mismatch\n";
        return 1;
    }

    double maxErr = 0.0;
    double sumErr = 0.0;
    std::size_t count = 0;
    std::vector<float> obs(inDim), ref(outDim);
    for (std::uint32_t s = 0; s < n; ++s) {
        f.read(reinterpret_cast<char*>(obs.data()), static_cast<std::streamsize>(inDim * sizeof(float)));
        f.read(reinterpret_cast<char*>(ref.data()), static_cast<std::streamsize>(outDim * sizeof(float)));
        const std::vector<float> got = policy.act(obs);
        for (std::uint32_t o = 0; o < outDim; ++o) {
            const double e = std::abs(static_cast<double>(got[o]) - static_cast<double>(ref[o]));
            maxErr = e > maxErr ? e : maxErr;
            sumErr += e;
            ++count;
        }
    }
    if (!f) {
        std::cerr << "truncated ref file\n";
        return 1;
    }

    const double meanErr = count ? sumErr / static_cast<double>(count) : 0.0;
    std::cout << "[selftest] samples=" << n << "  max_abs_err=" << maxErr
              << "  mean_abs_err=" << meanErr << "\n";

    // ELU uses expm1 vs torch's exp(x)-1; with 4 fp32 matmuls the accumulated
    // difference stays well under 1e-4 on randn inputs.
    const double tol = 1e-4;
    if (maxErr > tol) {
        std::cout << "[selftest] FAIL (max_abs_err " << maxErr << " > tol " << tol << ")\n";
        return 1;
    }
    std::cout << "[selftest] PASS\n";
    return 0;
}
