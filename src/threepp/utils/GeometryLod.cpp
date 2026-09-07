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
                                      const uint32_t* indices, size_t indexCount,
                                      bool sparse, const float* normals,
                                      float normalWeight) {
        std::vector<Level> chain;
        if (!positions || !indices || vertexCount < 3 || indexCount < kMinIndices) return chain;

        // Converts meshopt's mesh-extent-relative error into absolute
        // object-space units once; every level's result_error is scaled by
        // the same factor (see meshopt_simplifyScale's doc comment).
        const float scale = meshopt_simplifyScale(positions, vertexCount, 3 * sizeof(float));

        std::vector<uint32_t> src(indices, indices + indexCount);
        float prevErrorAbs = 0.f;
        // Sparse applies to EVERY level of a soup chain — the referenced
        // subset stays sparse relative to the original vertex buffer all the
        // way down (see the header's parameter doc for why it's required).
        const unsigned options = meshopt_SimplifyLockBorder |
                                 (sparse ? meshopt_SimplifySparse : 0u);

        for (size_t level = 0; level < kMaxLevels; ++level) {
            const size_t targetCount = (src.size() / 2) / 3 * 3;// ~50% of previous, tri-aligned
            if (targetCount < kMinIndices) break;

            // destination must hold up to src.size() elements (meshopt's
            // worst case is the INPUT count, not the target).
            std::vector<uint32_t> dst(src.size());
            float resultErrorRel = 0.f;
            // With normals, the collapse metric AND result_error charge normal
            // deviation (simplifier.cpp folds the attribute quadric into
            // c.error), so glossy curved surfaces stop simplifying / report
            // honest bounds instead of flattening their shading for free.
            const bool useAttrs = normals && normalWeight > 0.f;
            const float nw[3] = {normalWeight, normalWeight, normalWeight};
            const size_t newCount = useAttrs
                    ? meshopt_simplifyWithAttributes(
                              dst.data(), src.data(), src.size(),
                              positions, vertexCount, 3 * sizeof(float),
                              normals, 3 * sizeof(float), nw, 3,
                              /*vertex_lock=*/nullptr,
                              targetCount, kTargetErrorRel, options, &resultErrorRel)
                    : meshopt_simplify(
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

    std::vector<uint32_t> buildCanonicalIndices(const float* positions,
                                                 const float* normals,
                                                 const float* uvs,
                                                 size_t vertexCount) {
        std::vector<uint32_t> out;
        const size_t indexCount = vertexCount / 3 * 3;// whole triangles only
        if (!positions || indexCount < 3) return out;

        meshopt_Stream streams[3];
        size_t streamCount = 0;
        streams[streamCount++] = {positions, 3 * sizeof(float), 3 * sizeof(float)};
        if (normals) streams[streamCount++] = {normals, 3 * sizeof(float), 3 * sizeof(float)};
        if (uvs) streams[streamCount++] = {uvs, 2 * sizeof(float), 2 * sizeof(float)};

        // NULL indices = unindexed input (per meshopt's doc); remap[v] is the
        // dense unique-slot id every binary-identical vertex shares.
        std::vector<uint32_t> remap(vertexCount);
        meshopt_generateVertexRemapMulti(remap.data(), nullptr, vertexCount,
                                         vertexCount, streams, streamCount);

        // One representative ORIGINAL vid per unique slot (first occurrence),
        // so the output indexes the caller's untouched soup vertex buffer —
        // no compacted/remapped vertex stream is ever materialized.
        std::vector<uint32_t> representative(vertexCount, ~0u);
        out.resize(indexCount);
        for (size_t i = 0; i < indexCount; ++i) {
            const uint32_t slot = remap[i];
            if (representative[slot] == ~0u) representative[slot] = static_cast<uint32_t>(i);
            out[i] = representative[slot];
        }
        return out;
    }

}// namespace threepp::geometrylod
