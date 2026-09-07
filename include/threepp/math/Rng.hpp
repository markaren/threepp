
#ifndef THREEPP_RNG_HPP
#define THREEPP_RNG_HPP

#include <cmath>
#include <cstdint>

namespace threepp::math {

    // A seedable random generator whose output is a pure function of its
    // inputs: same seed, same draw sequence, same bits — on every platform,
    // every standard library, every thread.
    //
    // This is the one blessed answer to a problem the codebase has solved
    // privately several times over (Flock, SplatData, FireEffect, PerchIndex,
    // and a dozen seeded mt19937 locals). The private solutions exist for two
    // reasons, and both are design constraints here:
    //
    //   1. math::randFloat() is seeded from std::random_device per thread and
    //      cannot replay. It stays as the convenience for code that wants
    //      throwaway variety; nothing reproducible can be built on it.
    //   2. A seeded std::mt19937 is NOT enough: std::uniform_real_distribution
    //      is free to map engine output differently between standard
    //      libraries. Reproducibility ends at the toolchain boundary exactly
    //      where a paper or a golden test needs it not to. So every float
    //      here comes from an explicit bit-level conversion, never from a
    //      <random> distribution.
    //
    // There is deliberately no global instance and no global seed. Streams
    // are per-component: a component owns an Rng (or forks one per element),
    // so adding a tree cannot shift the numbers a flock draws afterwards.
    // The sensor-audit property "changing one seed flips exactly that
    // component's output" depends on this shape; a seedable global cannot
    // provide it.
    //
    // State is 16 bytes and copying is well-defined: a copy replays the
    // original's future draws. That is a feature (checkpoint/replay), not an
    // accident.
    class Rng {

    public:
        explicit Rng(std::uint64_t seed = 0): seed_(seed), state_(seed) {}

        // ── Sequential draws (the SplatGenerator idiom) ──────────────────

        std::uint32_t nextUint() {

            return static_cast<std::uint32_t>(nextUint64() >> 32);
        }

        // [0, 1). 24 explicit mantissa bits: exact in fp32, never reaches 1,
        // and identical on every platform because no distribution object is
        // involved (constraint 2 above).
        float nextFloat() {

            return static_cast<float>(nextUint64() >> 40) * (1.f / 16777216.f);
        }

        // [lo, hi). Matches the randFloat(min, max) shape.
        float nextFloat(float lo, float hi) {

            return lo + (hi - lo) * nextFloat();
        }

        // [-range/2, range/2). Matches the randFloatSpread shape.
        float nextFloatSpread(float range) {

            return range * (nextFloat() - 0.5f);
        }

        // [0, 1) with 53 bits, for accumulators where fp32 granularity shows.
        double nextDouble() {

            return static_cast<double>(nextUint64() >> 11) * (1.0 / 9007199254740992.0);
        }

        // Integer in [lo, hi], both ends inclusive. Multiply-shift mapping:
        // no modulo bias spike, no rejection loop; the residual bias is
        // < 2^-32 of the range, which procedural generation cannot observe.
        // hi < lo is answered with lo rather than UB.
        int nextInt(int lo, int hi) {

            if (hi <= lo) return lo;
            const auto range = static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo) + 1u;
            const std::uint64_t m = static_cast<std::uint64_t>(nextUint()) * range;
            return lo + static_cast<int>(m >> 32);
        }

        // Standard normal via Marsaglia's polar method: sqrt is IEEE-exact,
        // and there is no sin/cos. The one honest caveat: std::log is libm,
        // so gaussian draws are bit-stable per binary but may differ in the
        // last ulps across toolchains — the same boundary the flock documents
        // for its trig. Uniform draws carry no such caveat.
        // The pair's second value is discarded so the object needs no cache:
        // copy/fork semantics stay trivial, at the cost of one extra draw.
        float nextGaussian() {

            while (true) {
                const float u = nextFloat(-1.f, 1.f);
                const float v = nextFloat(-1.f, 1.f);
                const float s = u * u + v * v;
                if (s > 0.f && s < 1.f) {
                    return u * std::sqrt(-2.f * std::log(s) / s);
                }
            }
        }

        float nextGaussian(float mean, float stddev) {

            return mean + stddev * nextGaussian();
        }

        // Fisher-Yates. std::shuffle is NOT a substitute: its algorithm is
        // implementation-defined, so the same seed deals different orders on
        // different standard libraries — the distribution problem again,
        // wearing an algorithm's name.
        template<class It>
        void shuffle(It first, It last) {

            const auto n = static_cast<int>(last - first);
            for (int i = n - 1; i > 0; --i) {
                const int j = nextInt(0, i);
                auto tmp = first[i];
                first[i] = first[j];
                first[j] = tmp;
            }
        }

        // ── Substreams ───────────────────────────────────────────────────

        // An independent stream, a pure function of (constructor seed,
        // stream). It does not advance this generator, and it does not
        // depend on how many draws happened before the fork — so per-element
        // streams are stable no matter what order elements are visited in,
        // or whether some other element drew more this run.
        //
        //     Rng perTree = forest.fork(treeIndex);
        [[nodiscard]] Rng fork(std::uint64_t stream) const {

            return Rng(mix(seed_ + 0x9e3779b97f4a7c15ull * (stream + 1u)));
        }

        // ── Stateless counter draws (the Flock idiom) ────────────────────
        //
        // A draw as a pure function of (seed, slot, stream): independent of
        // frame ordering, of how many frames a bake took, and of wall time.
        // `slot` is the element (bird, grain, probe), `stream` its private
        // sequence counter. No object, no state, no ODR concerns in headers.

        // The raw 64-bit mixer (one splitmix64 step), for hash-table slots and
        // key scrambling. Not a stream: no state, no sequence — one number in,
        // one well-mixed number out.
        [[nodiscard]] static std::uint64_t mixBits(std::uint64_t x) {

            return mix(x + 0x9e3779b97f4a7c15ull);
        }

        [[nodiscard]] static std::uint32_t hashUint(std::uint64_t seed, std::uint64_t slot, std::uint64_t stream) {

            return static_cast<std::uint32_t>(
                    mix(seed + 0x9e3779b97f4a7c15ull * slot + 0xd1b54a32d192ed03ull * stream) >> 32);
        }

        [[nodiscard]] static float hash01(std::uint64_t seed, std::uint64_t slot, std::uint64_t stream) {

            return static_cast<float>(hashUint(seed, slot, stream) >> 8) * (1.f / 16777216.f);
        }

        // Single-key form for callers that build their own composite key
        // (spatial cell hashes and the like).
        [[nodiscard]] static float hash01(std::uint64_t key) {

            return static_cast<float>(mixBits(key) >> 40) * (1.f / 16777216.f);
        }

        // The seed this generator (and all its forks) descends from.
        [[nodiscard]] std::uint64_t seed() const { return seed_; }

    private:
        std::uint64_t seed_;
        std::uint64_t state_;

        // splitmix64 (Steele/Lea/Flood; public-domain constants). Chosen over
        // the in-house 32-bit hashes because the 64-bit finalizer passes the
        // usual statistical batteries even for pathological seeds (0, 1, 2…,
        // which are exactly the seeds humans type), while staying two
        // multiplies and three xors — no table, no branch.
        static std::uint64_t mix(std::uint64_t z) {

            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
            return z ^ (z >> 31);
        }

        std::uint64_t nextUint64() {

            state_ += 0x9e3779b97f4a7c15ull;
            return mix(state_);
        }
    };

}// namespace threepp::math

#endif// THREEPP_RNG_HPP
