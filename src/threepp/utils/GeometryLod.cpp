#include "threepp/utils/GeometryLod.hpp"

#include <meshoptimizer.h>

#include <algorithm>

namespace threepp::geometrylod {

    namespace {
        constexpr size_t kMaxLevels = 4;
        constexpr size_t kMinIndices = 384;// 128 triangles
        constexpr float kTargetErrorRel = 0.05f;
        constexpr size_t kRefuseNumerator = 85;// stop if new count > 85% of previous
    }// namespace

    std::vector<Level> generateChain(const float* positions, size_t vertexCount,
                                      const uint32_t* indices, size_t indexCount) {
        std::vector<Level> chain;
        if (!positions || !indices || vertexCount < 3 || indexCount < kMinIndices) return chain;

        // Converts meshopt's mesh-extent-relative error into absolute
        // object-space units once; every level's result_error is scaled by
        // the same factor (see meshopt_simplifyScale's doc comment).
        const float scale = meshopt_simplifyScale(positions, vertexCount, 3 * sizeof(float));

        std::vector<uint32_t> src(indices, indices + indexCount);
        float prevErrorAbs = 0.f;
        constexpr unsigned options = meshopt_SimplifyLockBorder;

        for (size_t level = 0; level < kMaxLevels; ++level) {
            const size_t targetCount = (src.size() / 2) / 3 * 3;// ~50% of previous, tri-aligned
            if (targetCount < kMinIndices) break;

            // destination must hold up to src.size() elements (meshopt's
            // worst case is the INPUT count, not the target).
            std::vector<uint32_t> dst(src.size());
            float resultErrorRel = 0.f;
            const size_t newCount = meshopt_simplify(
                    dst.data(), src.data(), src.size(),
                    positions, vertexCount, 3 * sizeof(float),
                    targetCount, kTargetErrorRel, options, &resultErrorRel);

            if (newCount == 0) break;// degenerate result — stop the chain here
            if (newCount * 100 > src.size() * kRefuseNumerator) break;// refuses to reduce meaningfully

            dst.resize(newCount);
            // meshopt's result_error is measured against THIS call's input —
            // i.e. against the PREVIOUS level, not the original mesh. Chained
            // deviations compound, so the honest bound vs LOD0 is the running
            // SUM (a max() clamp under-reports deep levels ~N× and silently
            // breaks the sub-pixel-switch guarantee the selection threshold
            // assumes). Additive is conservative — it can only make selection
            // pick finer levels sooner, never admit a visible pop.
            const float errorAbs = prevErrorAbs + resultErrorRel * scale;
            prevErrorAbs = errorAbs;

            chain.push_back(Level{std::move(dst), errorAbs});
            src = chain.back().indices;// next level simplifies further from here
        }

        return chain;
    }

}// namespace threepp::geometrylod
