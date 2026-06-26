// SpotPolicy — a tiny, dependency-free feed-forward policy network for native
// C++ deployment of a trained Spot (or any MLP) locomotion policy.
//
// The Python training/deploy path runs the policy through torch, which is fine
// for the GPU rollout but pins deployment to a Python + torch runtime. This
// header re-implements just the forward pass (the only thing deployment needs)
// in plain C++ over std:: only — no torch, no threepp, no GL/Vulkan. The same
// object therefore drives Spot identically under the GL or the Vulkan renderer;
// the renderer only draws the scene, it never touches inference.
//
// Weights come from a self-describing flat binary written by
// export_spot_policy.py (format below), so this loader is NOT hard-wired to the
// Isaac Spot shape (48->512->256->128->12, ELU) — it reads the layer list from
// the file and works for any exported MLP.
//
//   .tpnn  layout (little-endian):
//     char[4]  magic   = "TPN1"
//     u32      numLayers
//     repeated numLayers times:
//       u32    inDim
//       u32    outDim
//       u32    act         // 0 = none, 1 = ELU(alpha=1)
//       f32[outDim*inDim]  weight   (row-major [out, in], as torch nn.Linear.weight)
//       f32[outDim]        bias
//
// A forward pass on the Spot net is ~90k MACs — microseconds on one core — so
// there is no reason to put it on the GPU for single-robot play. (Batched many-
// robot inference is the case where a Vulkan-compute port pays off; this is the
// CPU path it would share weights with.)

#ifndef THREEPP_EXAMPLES_SPOT_POLICY_HPP
#define THREEPP_EXAMPLES_SPOT_POLICY_HPP

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace spot {

    class SpotPolicy {
    public:
        enum Act : std::uint32_t { None = 0, ELU = 1 };

        struct Layer {
            std::uint32_t in = 0, out = 0, act = None;
            std::vector<float> w;// [out*in], row-major
            std::vector<float> b;// [out]
        };

        // Load an exported policy from a .tpnn file. Throws std::runtime_error on
        // a bad magic, a truncated file, or an inconsistent layer chain.
        static SpotPolicy load(const std::string& path) {
            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("SpotPolicy: cannot open " + path);

            char magic[4];
            f.read(magic, 4);
            if (f.gcount() != 4 || magic[0] != 'T' || magic[1] != 'P' || magic[2] != 'N' || magic[3] != '1')
                throw std::runtime_error("SpotPolicy: bad magic in " + path + " (expected TPN1)");

            SpotPolicy p;
            const std::uint32_t numLayers = readU32(f);
            if (numLayers == 0 || numLayers > 64)
                throw std::runtime_error("SpotPolicy: implausible numLayers in " + path);
            p.layers_.resize(numLayers);
            for (std::uint32_t li = 0; li < numLayers; ++li) {
                Layer& L = p.layers_[li];
                L.in = readU32(f);
                L.out = readU32(f);
                L.act = readU32(f);
                if (L.in == 0 || L.out == 0)
                    throw std::runtime_error("SpotPolicy: zero-sized layer in " + path);
                if (li > 0 && L.in != p.layers_[li - 1].out)
                    throw std::runtime_error("SpotPolicy: layer dim mismatch in " + path);
                L.w.resize(static_cast<std::size_t>(L.out) * L.in);
                L.b.resize(L.out);
                f.read(reinterpret_cast<char*>(L.w.data()),
                        static_cast<std::streamsize>(L.w.size() * sizeof(float)));
                f.read(reinterpret_cast<char*>(L.b.data()),
                        static_cast<std::streamsize>(L.b.size() * sizeof(float)));
            }
            if (!f) throw std::runtime_error("SpotPolicy: truncated file " + path);
            return p;
        }

        [[nodiscard]] std::uint32_t inputDim() const { return layers_.empty() ? 0 : layers_.front().in; }
        [[nodiscard]] std::uint32_t outputDim() const { return layers_.empty() ? 0 : layers_.back().out; }

        // Deterministic forward pass (the policy MEAN action — what deployment
        // uses; the Gaussian exploration noise is training-only). `obs` must hold
        // exactly inputDim() floats. Returns outputDim() raw action values; the
        // caller applies action_scale and the joint-order remap.
        [[nodiscard]] std::vector<float> act(const float* obs, std::size_t n) const {
            if (layers_.empty()) throw std::runtime_error("SpotPolicy: no layers loaded");
            if (n != layers_.front().in)
                throw std::runtime_error("SpotPolicy: obs size " + std::to_string(n) +
                                         " != input dim " + std::to_string(layers_.front().in));
            std::vector<float> cur(obs, obs + n), nxt;
            for (const Layer& L : layers_) {
                nxt.assign(L.out, 0.0f);
                for (std::uint32_t o = 0; o < L.out; ++o) {
                    const float* wrow = &L.w[static_cast<std::size_t>(o) * L.in];
                    float acc = L.b[o];
                    for (std::uint32_t i = 0; i < L.in; ++i) acc += wrow[i] * cur[i];
                    if (L.act == ELU) acc = acc > 0.0f ? acc : std::expm1(acc);
                    nxt[o] = acc;
                }
                cur.swap(nxt);
            }
            return cur;
        }

        [[nodiscard]] std::vector<float> act(const std::vector<float>& obs) const {
            return act(obs.data(), obs.size());
        }

    private:
        static std::uint32_t readU32(std::ifstream& f) {
            std::uint32_t v = 0;
            f.read(reinterpret_cast<char*>(&v), 4);
            return v;// host is little-endian (x86/ARM); file is written little-endian
        }

        std::vector<Layer> layers_;
    };

}// namespace spot

#endif// THREEPP_EXAMPLES_SPOT_POLICY_HPP
