// Dropping force=true from OverlayPass::record's scene.updateMatrixWorld is
// only safe if the non-forced call yields the SAME matrixWorld for every node.
//
// OverlayPass reads nothing but matrixWorld off the graph, so if the two calls
// agree node-for-node the pass cannot tell them apart — which is a stronger and
// far more stable statement than any pixel comparison (the overlay capture path
// lags a state change under MAILBOX present mode).
//
// The interesting case is a HUD scene rendered via render(hudScene, orthoCam):
// it never passes through ensureSceneBuilt, so OverlayPass' own call is the
// only thing making its matrices current. updateMatrix() polls position/
// quaternion/scale against a cached snapshot and raises matrixWorldNeedsUpdate
// on any change, so mutations must still propagate without force.

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Matrix4.hpp"

#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <memory>
#include <vector>

using namespace threepp;

namespace {

    // Deep chains + branching, mirroring a real HUD/scene graph.
    std::shared_ptr<Object3D> buildGraph(int groups, int depth) {
        auto root = Object3D::create();
        for (int g = 0; g < groups; ++g) {
            Object3D* cursor = root.get();
            for (int d = 0; d < depth; ++d) {
                auto n = Object3D::create();
                n->position.set(0.1f * float(d), 0.2f * float(g), 0.05f * float(g + d));
                n->rotation.y = 0.01f * float(g * depth + d);
                n->scale.set(1.f, 1.f + 0.01f * float(d), 1.f);
                auto* raw = n.get();
                cursor->add(n);
                cursor = raw;
            }
        }
        return root;
    }

    void forEachNode(Object3D& o, const std::function<void(Object3D&)>& fn) {
        fn(o);
        for (auto& c : o.children) forEachNode(*c, fn);
    }

    std::vector<Matrix4> snapshotWorlds(Object3D& root) {
        std::vector<Matrix4> out;
        forEachNode(root, [&](Object3D& o) { out.push_back(*o.matrixWorld); });
        return out;
    }

    // Same mutation applied to two structurally identical graphs.
    void mutate(Object3D& root, int nth) {
        int i = 0;
        forEachNode(root, [&](Object3D& o) {
            if (i % nth == 0) {
                o.position.x += 1.5f;
                o.rotation.z += 0.25f;
            }
            ++i;
        });
    }

}// namespace

TEST_CASE("updateMatrixWorld(false) matches force=true after mutation", "[core]") {
    auto forced = buildGraph(40, 8);
    auto polled = buildGraph(40, 8);

    // Frame 1: both start cold.
    forced->updateMatrixWorld(true);
    polled->updateMatrixWorld(false);
    REQUIRE(snapshotWorlds(*forced) == snapshotWorlds(*polled));

    // Frame 2: no mutation at all — the non-forced pass must not go stale.
    forced->updateMatrixWorld(true);
    polled->updateMatrixWorld(false);
    REQUIRE(snapshotWorlds(*forced) == snapshotWorlds(*polled));

    // Frame 3: mutate interior nodes. This is the propagation case — a parent
    // moves and every descendant's matrixWorld must follow without force.
    mutate(*forced, 5);
    mutate(*polled, 5);
    forced->updateMatrixWorld(true);
    polled->updateMatrixWorld(false);
    REQUIRE(snapshotWorlds(*forced) == snapshotWorlds(*polled));

    // Frame 4: mutate ONLY the root, so the change must cascade the whole depth.
    forced->position.y -= 3.f;
    polled->position.y -= 3.f;
    forced->updateMatrixWorld(true);
    polled->updateMatrixWorld(false);
    REQUIRE(snapshotWorlds(*forced) == snapshotWorlds(*polled));
}

TEST_CASE("updateMatrixWorld(false) propagates externally-driven matrices", "[core]") {
    // matrixAutoUpdate == false is the loader/helper path: `matrix` is written
    // directly. updateMatrixWorld polls the matrix bytes for this case.
    auto forced = buildGraph(6, 4);
    auto polled = buildGraph(6, 4);
    forced->updateMatrixWorld(true);
    polled->updateMatrixWorld(false);

    auto driveFirstChild = [](Object3D& root) {
        auto& mid              = *root.children.front();
        mid.matrixAutoUpdate   = false;
        mid.matrix->makeTranslation(4.f, -2.f, 1.f);
    };
    driveFirstChild(*forced);
    driveFirstChild(*polled);

    forced->updateMatrixWorld(true);
    polled->updateMatrixWorld(false);
    REQUIRE(snapshotWorlds(*forced) == snapshotWorlds(*polled));
}
