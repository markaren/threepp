#ifndef THREEPP_WGPUPATHTRACERGEOMETRY_HPP
#define THREEPP_WGPUPATHTRACERGEOMETRY_HPP

// Private header — packs per-frame triangle / material / mesh-matrix buffers
// from the expanded RtMeshEntry list. Output buffers are sampled by the WGSL
// path tracer as paged textures.

#include "threepp/renderers/wgpu/pathtracer/WgpuPathTracerTypes.hpp"

#include <unordered_map>
#include <vector>

namespace threepp {
    class Texture;
}

namespace threepp::wgpu_pt {

    struct RtMeshEntry;

    /// Compute the flat index into a paged triBuffer for triangle `ti`, row `row`.
    /// The texture is TEX_PAGE_WIDTH columns wide; each triangle occupies
    /// TRI_TEX_HEIGHT rows. Used both during build and when GPU triCoord is
    /// translated back to a CPU index (e.g. for emissive triangle harvesting).
    inline int pagedIdx(int ti, int row) {
        return ((ti / TEX_PAGE_WIDTH * TRI_TEX_HEIGHT + row) * TEX_PAGE_WIDTH + ti % TEX_PAGE_WIDTH) * 4;
    }

    /// Pack triangles, per-mesh world / normal matrices, and per-mesh material
    /// rows for `entries` into the four output buffers. Caller is responsible
    /// for resizing buffers up-front; this function honours `maxTris`,
    /// `maxMats`, `maxMeshes` and stops cleanly at those caps.
    /// In append mode (any offset > 0) existing content is preserved.
    /// Returns the new total triangle count.
    ///
    /// If `entryTriRanges` is non-null it is resized to entries.size() and
    /// populated with (triStart, triCount) pairs per entry (into triBuffer /
    /// rawObjTriBuf). Entries skipped due to material/mesh/tri caps get a
    /// zero-count range. Used by the TLAS/BLAS builder to locate each
    /// entry's triangle slice for per-mesh BLAS construction.
    int buildGeometryBuffers(
            const std::vector<RtMeshEntry>& entries,
            const std::unordered_map<Texture*, int>& texSlotMap,
            std::vector<float>& triBuffer,
            std::vector<float>& matBuffer,
            std::vector<float>& rawObjTriBuf,
            std::vector<float>& matrixBuf,
            int maxTris, int maxMats, int maxMeshes,
            int triOffset = 0, int matOffset = 0, int meshOffset = 0,
            std::vector<std::pair<int, int>>* entryTriRanges = nullptr);

}// namespace threepp::wgpu_pt

#endif//THREEPP_WGPUPATHTRACERGEOMETRY_HPP
