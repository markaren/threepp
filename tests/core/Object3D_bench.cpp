// CPU-only microbenchmark for the Object3D transform hierarchy.
//
// Not a ctest — run manually to compare builds (e.g. heap-allocated vs
// value-stored matrix/matrixWorld). Everything is deterministic (fixed seed,
// no renderer), so run-to-run variance is low; still, interleave runs of the
// two binaries when A/B-ing.
//
// Phases:
//   build      — construct an N-node tree (allocation cost)
//   static     — updateMatrixWorld() with nothing dirty (early-out + locality)
//   dynamic5   — mutate 5% of node positions, then updateMatrixWorld()
//   readAll    — read matrixWorld translation of every node (scene-prep style)
//   teardown   — destroy the tree (free cost)
//
// Each phase runs in two heap layouts: "clustered" (nodes allocated
// back-to-back, the fresh-load case) and "fragmented" (junk allocations
// interleaved during build then freed — the long-lived-scene case, which is
// where per-node heap matrices scatter furthest from their objects).
//
// Usage: Object3D_bench [nodeCount]   (default 100000)

#include "threepp/core/Object3D.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

using threepp::Object3D;

namespace {

    using Clock = std::chrono::steady_clock;

    double msSince(Clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }

    struct Tree {
        std::shared_ptr<Object3D> root;
        std::vector<Object3D*> flat;// all nodes, traversal order
    };

    // Build an N-node tree, branching factor 8, randomized transforms.
    // When fragment=true, interleave short-lived junk allocations of varying
    // size between node constructions so node/matrix allocations don't land
    // adjacent, then free the junk (leaving holes).
    Tree buildTree(std::size_t n, bool fragment, double& buildMs) {

        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> dist(-10.f, 10.f);
        std::uniform_int_distribution<std::size_t> junkSize(48, 1024);

        std::vector<void*> junk;
        if (fragment) junk.reserve(n);

        const auto t0 = Clock::now();

        Tree tree;
        tree.root = Object3D::create();
        tree.flat.reserve(n);
        tree.flat.push_back(tree.root.get());

        std::size_t next = 0;// index of the node receiving the next child
        while (tree.flat.size() < n) {

            if (fragment) junk.push_back(std::malloc(junkSize(rng)));

            auto node = Object3D::create();
            node->position.set(dist(rng), dist(rng), dist(rng));
            node->rotateX(dist(rng) * 0.1f);
            node->rotateY(dist(rng) * 0.1f);
            const float s = 1.f + dist(rng) * 0.01f;
            node->scale.set(s, s, s);

            Object3D* raw = node.get();
            tree.flat[next]->add(node);
            tree.flat.push_back(raw);

            if (tree.flat[next]->children.size() >= 8) ++next;
        }

        buildMs = msSince(t0);

        for (void* p : junk) std::free(p);
        return tree;
    }

    double median(std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    }

    struct PhaseResult {
        double medianMs;
        double minMs;
    };

    template<class F>
    PhaseResult runPhase(int reps, F&& fn) {
        std::vector<double> samples;
        samples.reserve(reps);
        for (int i = 0; i < reps; ++i) {
            const auto t0 = Clock::now();
            fn(i);
            samples.push_back(msSince(t0));
        }
        return {median(samples), *std::min_element(samples.begin(), samples.end())};
    }

}// namespace

int main(int argc, char** argv) {

    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 100000;
    const int reps = 100;

    std::printf("Object3D_bench  nodes=%zu  reps=%d  sizeof(Object3D)=%zu\n",
                n, reps, sizeof(Object3D));

    // Defeats dead-code elimination; also a cross-build equivalence check —
    // the checksum must match between the two binaries being compared.
    double checksum = 0.0;

    for (const bool fragment : {false, true}) {

        const char* layout = fragment ? "fragmented" : "clustered ";

        double buildMs = 0.0;
        auto tree = buildTree(n, fragment, buildMs);

        // Prime: first full update composes everything.
        tree.root->updateMatrixWorld();

        // static: nothing dirty, pure early-out traversal
        const auto stat = runPhase(reps, [&](int) {
            tree.root->updateMatrixWorld();
        });

        // dynamic5: mutate 5% of nodes, then update
        std::vector<Object3D*> dirty;
        for (std::size_t i = 0; i < tree.flat.size(); i += 20) dirty.push_back(tree.flat[i]);
        const auto dyn = runPhase(reps, [&](int i) {
            const float d = (i % 2 == 0) ? 0.001f : -0.001f;
            for (auto* o : dirty) o->position.x += d;
            tree.root->updateMatrixWorld();
        });

        // readAll: scene-prep style read of every node's world translation
        const auto read = runPhase(reps, [&](int) {
            double acc = 0.0;
            for (const auto* o : tree.flat) {
                const auto& e = o->matrixWorld->elements;
                acc += e[12] + e[13] + e[14];
            }
            checksum += acc;
        });

        // teardown
        const auto t0 = Clock::now();
        tree.flat.clear();
        tree.root.reset();
        const double teardownMs = msSince(t0);

        std::printf("[%s] build    %8.3f ms\n", layout, buildMs);
        std::printf("[%s] static   %8.3f ms  (min %.3f)\n", layout, stat.medianMs, stat.minMs);
        std::printf("[%s] dynamic5 %8.3f ms  (min %.3f)\n", layout, dyn.medianMs, dyn.minMs);
        std::printf("[%s] readAll  %8.3f ms  (min %.3f)\n", layout, read.medianMs, read.minMs);
        std::printf("[%s] teardown %8.3f ms\n", layout, teardownMs);
    }

    std::printf("checksum %.6e\n", checksum);
    return 0;
}
