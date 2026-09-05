#include "VulkanCoreImpl.hpp"
#include "VulkanCpuPhaseProf.hpp"

#include "threepp/core/AttributeView.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace {

    // CPU-side equivalents of GLSL packSnorm2x16 / packUnorm2x16 / packUnorm4x8.
    // The shaders decode with the matching unpack* built-ins, so the rounding
    // conventions must match the GLSL spec: round(clamp(v,...) * scale).
    uint32_t packSnorm2x16(float a, float b) {
        auto cvt = [](float v) -> uint32_t {
            const auto s = static_cast<int32_t>(std::lround(std::clamp(v, -1.f, 1.f) * 32767.f));
            return static_cast<uint32_t>(s) & 0xFFFFu;
        };
        return cvt(a) | (cvt(b) << 16);
    }

    uint32_t packUnorm2x16(float a, float b) {
        auto cvt = [](float v) -> uint32_t {
            return static_cast<uint32_t>(std::lround(std::clamp(v, 0.f, 1.f) * 65535.f));
        };
        return cvt(a) | (cvt(b) << 16);
    }

    uint32_t packUnorm4x8(float r, float g, float b, float a) {
        auto cvt = [](float v) -> uint32_t {
            return static_cast<uint32_t>(std::lround(std::clamp(v, 0.f, 1.f) * 255.f));
        };
        return cvt(r) | (cvt(g) << 8) | (cvt(b) << 16) | (cvt(a) << 24);
    }

    bool allWithin(const float* v, size_t n, float lo, float hi) {
        for (size_t i = 0; i < n; ++i) {
            if (!(v[i] >= lo && v[i] <= hi)) return false;// NaN also fails here
        }
        return true;
    }

    // Octahedral encode, matching probe_common.glsl's octEncode/octDecode pair
    // exactly (including the signNotZero convention) — the attribute decoders
    // in gbuffer_indirect.vert / deferred_shade / probe_update use the same
    // fold, so encode and decode must agree bit-for-bit on the wrap rule.
    std::pair<float, float> octEncode(float x, float y, float z) {
        const float len = std::abs(x) + std::abs(y) + std::abs(z);
        if (len < 1e-12f) return {0.f, 0.f};// degenerate → decodes to +Z
        x /= len;
        y /= len;
        z /= len;
        if (z < 0.f) {
            const float sx = x >= 0.f ? 1.f : -1.f;
            const float sy = y >= 0.f ? 1.f : -1.f;
            const float ox = (1.f - std::abs(y)) * sx;
            const float oy = (1.f - std::abs(x)) * sy;
            return {ox, oy};
        }
        return {x, y};
    }

    // BufferGeometry::drawRange resolved against the geometry's real element
    // total — index count when indexed, vertex count otherwise, same units as
    // GL's drawRange. elems is floored at 3 whenever the geometry is non-empty:
    // an AS cannot be "emptied" by a range (a mesh is hidden via visibility,
    // not drawRange), and a zero-primitive build range is driver roulette, so
    // a degenerate range keeps one triangle alive instead.
    struct DrawSpan {
        uint32_t first = 0;// element index the range starts at
        uint32_t elems = 0;// elements in the range (multiple-of-3 not enforced)
    };
    DrawSpan drawSpanOf(const threepp::BufferGeometry& geom, uint32_t totalElems) {
        const auto& dr = geom.drawRange;
        const auto start = dr.start > 0 ? static_cast<uint32_t>(dr.start) : 0u;
        const auto count = dr.count > 0 ? static_cast<uint32_t>(dr.count) : 0u;
        DrawSpan s;
        s.first = std::min(start, totalElems);
        s.elems = std::min(totalElems - s.first, count);
        if (s.elems < 3u && totalElems >= 3u) {
            s.first = std::min(s.first, totalElems - 3u);
            s.elems = 3u;
        }
        return s;
    }

}// namespace

namespace threepp {

std::unique_ptr<VulkanRenderer::Impl::BlasRecord> VulkanRenderer::Impl::buildBlasFor(const BufferGeometry& geom, bool allowPacked) {
            // Escape hatch for A/B triage: THREEPP_NO_PACK=1 forces every
            // attribute buffer back to tightly-packed float, same binary.
            static const bool noPack = std::getenv("THREEPP_NO_PACK") != nullptr;
            allowPacked = allowPacked && !noPack;
            // Per-geometry override, and deliberately HERE rather than at the
            // call sites: this is the one funnel every record passes through, so
            // a future caller cannot forget it. A geometry marked by
            // enableVertexInterop must come back tightly-packed float, because
            // that is the only layout a foreign device producer can be expected
            // to write — see forceUnpackedGeoms_ for the packed→unpacked handoff
            // this closes. Scoped to that one geometry: the rest of the scene
            // keeps its packed normals/uv/color, so this is NOT THREEPP_NO_PACK
            // by another name.
            // (Also read below: an interop-marked geometry's BLAS storage is
            // sized for both build-flag lineages, not just FAST_TRACE.)
            const bool interopTarget = forceUnpackedGeoms_.count(&geom) != 0;
            if (allowPacked && interopTarget) allowPacked = false;

            auto* posAttr = geom.getAttribute<float>("position");
            if (!posAttr) return nullptr;
            // Normal/uv/color may be narrowed (compressAttributes); the view
            // widens them once here, at upload, so the device buffers — which
            // the shaders index as tightly-packed float and the raster prepass
            // binds as float vertex input — keep their layout. Positions are
            // never narrowed and stay a direct typed read.
            FloatAttributeView normals(geom.getAttribute("normal"));
            if (!normals) return nullptr;// the RT path requires per-vertex normals
            const auto& positions = posAttr->array();
            const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
            if (vertexCount < 3) return nullptr;
            if (normals.count() != static_cast<int>(vertexCount)) return nullptr;

            const auto* idxAttr = geom.getIndex();
            const bool indexed = idxAttr != nullptr;
            const uint32_t primitiveCount = indexed
                    ? static_cast<uint32_t>(idxAttr->count() / 3)
                    : vertexCount / 3;
            if (primitiveCount == 0) return nullptr;

            // BLAS build is undefined behavior on NaN/Inf positions or
            // out-of-bounds indices — drivers respond with device-lost
            // (vkQueueWaitIdle returns VK_ERROR_DEVICE_LOST). Validate
            // before submission so a bad asset (typical of Assimp-loaded
            // URDF .dae meshes with degenerate triangles) is skipped with
            // a warning rather than killing the whole renderer.
            for (size_t i = 0; i < positions.size(); ++i) {
                if (!std::isfinite(positions[i])) {
                    std::cerr << "[VulkanRenderer] buildBlasFor: skipping geometry - "
                              << "position[" << i << "] is non-finite ("
                              << positions[i] << "), vertexCount=" << vertexCount << '\n';
                    return nullptr;
                }
            }
            if (indexed) {
                const auto& indices = idxAttr->array();
                for (size_t i = 0; i < indices.size(); ++i) {
                    if (indices[i] >= vertexCount) {
                        std::cerr << "[VulkanRenderer] buildBlasFor: skipping geometry - "
                                  << "index[" << i << "]=" << indices[i]
                                  << " >= vertexCount=" << vertexCount << '\n';
                        return nullptr;
                    }
                }
            }

            // Hybrid: raster G-buffer pre-pass binds these same allocations
            // directly as vertex / index buffers — no duplication, no extra
            // upload, and the raster prepass + RT shadow rays warm the same
            // cache lines. TRANSFER_SRC_BIT for displaced meshes that need
            // to vkCmdCopyBuffer the current vertex into prev each frame;
            // TRANSFER_DST_BIT for graduated per-frame dynamic records whose
            // new positions arrive by GPU copy from staging instead of a
            // host memcpy (recordDynamicGeomRefits).
            const VkBufferUsageFlags geomUsage =
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;

            auto rec = std::make_unique<BlasRecord>();

            const VkDeviceSize vbBytes = positions.size() * sizeof(float);
            rec->vertex = createBuffer(
                    ctx->allocator(), ctx->device(), vbBytes,
                    geomUsage, VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            uploadHostVisible(ctx->allocator(), rec->vertex, positions.data(), vbBytes);

            // Normals: octahedral snorm16x2, ONE uint per vertex (12 → 4 bytes)
            // when packing is allowed. Every consumer of normalAddress is a
            // software fetch (gbuffer_indirect.vert vertex-pull, deferred_shade
            // ray-query hits, probe_update) — the fixed-input gbuffer pipeline
            // exists but is never bound — so no fixed-function format concerns.
            const bool packNrm = allowPacked;
            std::vector<uint32_t> packedNrm;
            const void* nrmSrc = normals.data();
            VkDeviceSize nbBytes = normals.size() * sizeof(float);
            if (packNrm) {
                packedNrm.resize(vertexCount);
                for (uint32_t v = 0; v < vertexCount; ++v) {
                    const auto [ox, oy] = octEncode(normals[v * 3 + 0],
                                                    normals[v * 3 + 1],
                                                    normals[v * 3 + 2]);
                    packedNrm[v] = packSnorm2x16(ox, oy);
                }
                nrmSrc = packedNrm.data();
                nbBytes = packedNrm.size() * sizeof(uint32_t);
                rec->packedMask |= 1u;
            }
            rec->normal = createBuffer(
                    ctx->allocator(), ctx->device(), nbBytes,
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            uploadHostVisible(ctx->allocator(), rec->normal, nrmSrc, nbBytes);

            // Optional UV attribute (TEXCOORD_0). closest_hit interpolates and
            // samples albedo with these; absent → bindless texture is ignored.
            if (FloatAttributeView uvs{geom.getAttribute("uv")}) {
                if (uvs.count() == static_cast<int>(vertexCount) &&
                    uvs.size() == vertexCount * 2) {
                    // UVs: unorm16x2, one uint per vertex (8 → 4 bytes) — but
                    // only when every coordinate sits in [0,1]; tiled/atlas UVs
                    // would clamp onto the texture edge, so those stay float.
                    const bool packUv = allowPacked &&
                                        allWithin(uvs.data(), uvs.size(), 0.f, 1.f);
                    std::vector<uint32_t> packedUv;
                    const void* uvSrc = uvs.data();
                    VkDeviceSize uvBytes = uvs.size() * sizeof(float);
                    if (packUv) {
                        packedUv.resize(vertexCount);
                        for (uint32_t v = 0; v < vertexCount; ++v) {
                            packedUv[v] = packUnorm2x16(uvs[v * 2 + 0], uvs[v * 2 + 1]);
                        }
                        uvSrc = packedUv.data();
                        uvBytes = packedUv.size() * sizeof(uint32_t);
                        rec->packedMask |= 2u;
                    }
                    rec->uv = createBuffer(
                            ctx->allocator(), ctx->device(), uvBytes,
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VMA_MEMORY_USAGE_AUTO,
                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                    uploadHostVisible(ctx->allocator(), rec->uv, uvSrc, uvBytes);
                }
            }

            // Optional per-vertex color (material.vertexColors). closest_hit and
            // the raster gbuffer interpolate this and modulate albedo. Stored as
            // tightly-packed vec3 (the shaders fetch 3 floats/vertex); itemSize-4
            // colors are repacked to RGB, dropping the alpha. Whether it actually
            // applies is decided per-instance from the material's vertexColors
            // flag when GeometryDesc / DrawInfo are filled — the buffer is always
            // uploaded if present so a shared geometry works under either material.
            if (FloatAttributeView cols{geom.getAttribute("color")}) {
                const int itemSize = cols.itemSize();
                if (cols.count() == static_cast<int>(vertexCount) &&
                    (itemSize == 3 || itemSize == 4)) {
                    // Colors: unorm8x4, one uint per vertex (12 → 4 bytes) when
                    // packing is allowed and the values are LDR; HDR vertex
                    // colors (> 1) keep the float path. Alpha slot is unused by
                    // the shaders (RGB modulate only) — packed as 1.
                    const bool packCol = allowPacked &&
                                         allWithin(cols.data(), cols.size(), 0.f, 1.f);
                    if (packCol) {
                        std::vector<uint32_t> packedCol(vertexCount);
                        for (uint32_t v = 0; v < vertexCount; ++v) {
                            packedCol[v] = packUnorm4x8(cols[v * itemSize + 0],
                                                        cols[v * itemSize + 1],
                                                        cols[v * itemSize + 2], 1.f);
                        }
                        const VkDeviceSize cbBytes = packedCol.size() * sizeof(uint32_t);
                        rec->color = createBuffer(
                                ctx->allocator(), ctx->device(), cbBytes,
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                VMA_MEMORY_USAGE_AUTO,
                                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                        uploadHostVisible(ctx->allocator(), rec->color, packedCol.data(), cbBytes);
                        rec->packedMask |= 4u;
                    } else {
                    const VkDeviceSize cbBytes = static_cast<VkDeviceSize>(vertexCount) * 3 * sizeof(float);
                    rec->color = createBuffer(
                            ctx->allocator(), ctx->device(), cbBytes,
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VMA_MEMORY_USAGE_AUTO,
                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                    void* mapped = nullptr;
                    vmaMapMemory(ctx->allocator(), rec->color.alloc, &mapped);
                    if (itemSize == 3) {
                        std::memcpy(mapped, cols.data(), cbBytes);
                    } else {
                        auto* dst = static_cast<float*>(mapped);
                        for (uint32_t v = 0; v < vertexCount; ++v) {
                            dst[v * 3 + 0] = cols[v * 4 + 0];
                            dst[v * 3 + 1] = cols[v * 4 + 1];
                            dst[v * 3 + 2] = cols[v * 4 + 2];
                        }
                    }
                    flushHostWrites(ctx->allocator(), rec->color.alloc, 0, cbBytes);
                    vmaUnmapMemory(ctx->allocator(), rec->color.alloc);
                    }
                }
            }

            // Indices: uint16 (half the bytes, half the fetch bandwidth) when
            // every index fits — LOSSLESS, unlike the attribute packing above.
            // bit3 is set by vertex count alone so LOD levels of a non-indexed
            // soup record (which ARE indexed) can pack against the same rule;
            // any level's max index is bounded by the record's vertex count.
            const bool packIdx = allowPacked && vertexCount <= 65536u;
            if (packIdx) rec->packedMask |= 8u;

            if (indexed) {
                const auto& indices = idxAttr->array();
                if (packIdx) {
                    // Word-align the allocation: the shader-side fetch reads
                    // whole uints and extracts 16-bit halves.
                    std::vector<uint16_t> idx16(indices.size() + (indices.size() & 1u), 0);
                    for (size_t k = 0; k < indices.size(); ++k) {
                        idx16[k] = static_cast<uint16_t>(indices[k]);
                    }
                    const VkDeviceSize ibBytes = idx16.size() * sizeof(uint16_t);
                    rec->index = createBuffer(
                            ctx->allocator(), ctx->device(), ibBytes,
                            geomUsage, VMA_MEMORY_USAGE_AUTO,
                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                    uploadHostVisible(ctx->allocator(), rec->index, idx16.data(), ibBytes);
                } else {
                    const VkDeviceSize ibBytes = indices.size() * sizeof(unsigned int);
                    rec->index = createBuffer(
                            ctx->allocator(), ctx->device(), ibBytes,
                            geomUsage, VMA_MEMORY_USAGE_AUTO,
                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                    uploadHostVisible(ctx->allocator(), rec->index, indices.data(), ibBytes);
                }
            }

            VkAccelerationStructureGeometryTrianglesDataKHR triData{};
            triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triData.vertexData.deviceAddress = rec->vertex.address;
            triData.vertexStride = 3 * sizeof(float);
            triData.maxVertex = vertexCount - 1;
            if (indexed) {
                triData.indexType = packIdx ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
                triData.indexData.deviceAddress = rec->index.address;
            } else {
                triData.indexType = VK_INDEX_TYPE_NONE_KHR;
            }

            VkAccelerationStructureGeometryKHR blasGeom{};
            blasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            blasGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            blasGeom.geometry.triangles = triData;
            // No OPAQUE flag — that would suppress any-hit invocation per
            // geometry regardless of the ray flags, breaking alpha-test. The
            // any-hit shader short-circuits cheaply for materials with
            // alphaCutoff <= 0, so the cost on truly opaque meshes is one
            // buffer read + compare per hit candidate.
            blasGeom.flags = 0;

            VkAccelerationStructureBuildGeometryInfoKHR blasBuild{};
            blasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            blasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            // ALLOW_UPDATE so refitTlas's MODE_UPDATE on the TLAS still
            // resolves to a valid build chain even if a future BLAS-level
            // refit lands. PREFER_FAST_BUILD is intentionally not set —
            // BLAS is still built once per geometry and queried-many.
            blasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            blasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            blasBuild.geometryCount = 1;
            blasBuild.pGeometries = &blasGeom;

            VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
            blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(),
                    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &blasBuild, &primitiveCount, &blasSizes);

            // An interop record's per-frame rebuilds take PREFER_FAST_BUILD
            // (recordDynamicGeomRefits), and the two flag lineages are
            // different BVH formats with different footprints: on NVIDIA a
            // FAST_BUILD structure over the same geometry is ~5% LARGER
            // (measured 345856 vs 329600 B for a 5120-triangle indexed
            // mesh). Building the larger lineage into storage sized for the
            // smaller overruns the structure — VUID-vkCmdBuildAcceleration-
            // StructuresKHR-pInfos-10126 with the layers on, heap corruption
            // surfacing as VK_ERROR_DEVICE_LOST without — so size the
            // storage for the max of both lineages up front. Scoped to
            // interop-marked geometries: everything else builds FAST_TRACE
            // for life and should not pay the padding.
            VkDeviceSize asBytes = blasSizes.accelerationStructureSize;
            if (interopTarget) {
                VkAccelerationStructureBuildGeometryInfoKHR fbBuild = blasBuild;
                fbBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR |
                                VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
                VkAccelerationStructureBuildSizesInfoKHR fbSizes{};
                fbSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
                ctx->rt().getAccelerationStructureBuildSizes(
                        ctx->device(),
                        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                        &fbBuild, &primitiveCount, &fbSizes);
                asBytes = std::max(asBytes, fbSizes.accelerationStructureSize);
                rec->storageFitsFastBuild = true;
            }

            rec->storage = createBuffer(
                    ctx->allocator(), ctx->device(), asBytes,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);

            VkAccelerationStructureCreateInfoKHR blasCreate{};
            blasCreate.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            blasCreate.buffer = rec->storage.handle;
            blasCreate.size = asBytes;
            blasCreate.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            check(ctx->rt().createAccelerationStructure(ctx->device(), &blasCreate, nullptr, &rec->as),
                  "vkCreateAccelerationStructureKHR(BLAS)");

            Buffer scratch = createAsScratchBuffer(ctx->allocator(), ctx->device(), blasSizes.buildScratchSize);

            blasBuild.dstAccelerationStructure = rec->as;
            blasBuild.scratchData.deviceAddress = scratch.address;

            // The initial build honours BufferGeometry::drawRange: the AS and
            // scratch above were sized for FULL capacity (the size query used
            // primitiveCount), which the spec allows to be built with any
            // smaller count — so only [0, start+count) is built. NOT the exact
            // [start, start+count) sub-span: the chit/ray-query fetches resolve
            // gl_PrimitiveID against the BUFFER BASE (GeometryDesc carries no
            // offset), so a range with firstVertex/primitiveOffset would shade
            // the wrong vertices. start is 0 in practice (over-allocated
            // dynamic buffers); a start > 0 merely leaves the prefix visible
            // to rays. Refits compare against blasBuiltPrims and force
            // MODE_BUILD when the count moves (updating with a different
            // count is invalid).
            const DrawSpan span = drawSpanOf(
                    geom, indexed ? static_cast<uint32_t>(idxAttr->count()) : vertexCount);
            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = (span.first + span.elems) / 3u;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

            // Per-vertex previous-pose buffer for the chit's per-vertex motion
            // vector path. Skinned / displaced / morphed meshes own their own
            // prev buffer in their state struct; plain meshes that go through
            // refreshGeomBlas (PhysX soft bodies, ParticleSystems updating
            // vertices in place) need one here too — without it, the chit reads
            // gdesc.prevVertexAddress == vertexAddress, motion is zero, and
            // freshly-spawned dynamic bodies look noisy because the temporal
            // accumulator blends mismatched history. Static meshes get a free
            // copy (prev == current forever) — chit motion vector resolves to
            // zero, matching the previous fallback behavior.
            rec->prevVertex = createBuffer(
                    ctx->allocator(), ctx->device(), vbBytes,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);

            VkCommandBuffer cb = beginOneShot();
            // Seed prevVertex with the initial (current) vertex data — first
            // refresh would otherwise copy garbage into prev. Fold into the
            // same one-shot as the BLAS build to avoid an extra submit.
            VkBufferCopy seedCopy{};
            seedCopy.size = vbBytes;
            vkCmdCopyBuffer(cb, rec->vertex.handle, rec->prevVertex.handle, 1, &seedCopy);
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &blasBuild, &pRange);
            endAndSubmitOneShot(cb, "buildBlasFor");
            // Deferred when a batch is open (the shared submit hasn't run yet, so
            // the scratch is still in use); freed immediately otherwise.
            destroyBufferMaybeBatched(scratch);

            VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
            addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
            addrInfo.accelerationStructure = rec->as;
            rec->address = ctx->rt().getAccelerationStructureDeviceAddress(ctx->device(), &addrInfo);

            rec->geomVersion = geomVersionOf(geom);
            rec->vertexCount = vertexCount;
            rec->indexCount  = indexed ? static_cast<uint32_t>(idxAttr->count()) : 0u;
            rec->vbBytes     = vbBytes;// for the prevVertex re-sync copy
            rec->blasBuiltPrims = range.primitiveCount;
            rec->blasBuiltFlags = blasBuild.flags;
            rec->lastDrawStart  = geom.drawRange.start;
            rec->lastDrawCount  = geom.drawRange.count;

            return rec;
        }

// Creates one auto-LOD chain level's resources: a new index buffer + a
// static BLAS built against `rec`'s EXISTING vertex buffer (positions are
// never touched by simplification — every index in `level.indices` still
// refers into rec->vertex, same as LOD0; for welded soup the canonical
// index VALUES are original vertex ids, so the same holds). Subset of
// buildBlasFor's shape restricted to index-only geometry input; no normal/
// uv/color buffers (those are shared with LOD0 via the same vertexAddress-
// keyed lookup the shaders already do). The build itself is DEFERRED into
// `pending` — drainLodResults batches a whole frame's builds into one
// one-shot submit via flushLodLevelBuilds.
bool VulkanRenderer::Impl::buildLodLevelFor(BlasRecord& rec,
                                                     const geometrylod::Level& level,
                                                     BlasRecord::LodLevel& out,
                                                     std::vector<LodPendingBuild>& pending) {
            if (level.indices.empty()) return false;
            if (rec.vertex.handle == VK_NULL_HANDLE || rec.vertexCount == 0) return false;
            const uint32_t primitiveCount = static_cast<uint32_t>(level.indices.size() / 3);
            if (primitiveCount == 0) return false;

            const VkBufferUsageFlags idxUsage =
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

            // Levels pack to uint16 exactly when the base record does (bit 3):
            // every level index refers into rec->vertex, so base-fits implies
            // level-fits. GeometryDesc/DrawInfo carry the same bit per record,
            // and a LOD switch only repoints indexAddress — the flag holds.
            const bool packIdx = (rec.packedMask & 8u) != 0u;
            if (packIdx) {
                std::vector<uint16_t> idx16(level.indices.size() + (level.indices.size() & 1u), 0);
                for (size_t k = 0; k < level.indices.size(); ++k) {
                    idx16[k] = static_cast<uint16_t>(level.indices[k]);
                }
                const VkDeviceSize ibBytes = idx16.size() * sizeof(uint16_t);
                out.index = createBuffer(
                        ctx->allocator(), ctx->device(), ibBytes,
                        idxUsage, VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                uploadHostVisible(ctx->allocator(), out.index, idx16.data(), ibBytes);
            } else {
                const VkDeviceSize ibBytes = level.indices.size() * sizeof(uint32_t);
                out.index = createBuffer(
                        ctx->allocator(), ctx->device(), ibBytes,
                        idxUsage, VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                uploadHostVisible(ctx->allocator(), out.index, level.indices.data(), ibBytes);
            }

            // Size query needs a fully-specified build info; a LOCAL struct is
            // fine here (only the final batched cmdBuildAccelerationStructures
            // needs pointer-stable storage — flushLodLevelBuilds rebuilds it).
            VkAccelerationStructureGeometryTrianglesDataKHR triData{};
            triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triData.vertexData.deviceAddress = rec.vertex.address;
            triData.vertexStride = 3 * sizeof(float);
            triData.maxVertex = rec.vertexCount - 1;
            triData.indexType = packIdx ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
            triData.indexData.deviceAddress = out.index.address;

            VkAccelerationStructureGeometryKHR blasGeom{};
            blasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            blasGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            blasGeom.geometry.triangles = triData;
            // Same any-hit-visible flags as LOD0 (0, not OPAQUE) — a
            // simplified level of an alpha-tested surface must still run
            // the any-hit alpha cutout, not just its opaque triangles.
            blasGeom.flags = 0;

            VkAccelerationStructureBuildGeometryInfoKHR blasBuild{};
            blasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            blasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            // No ALLOW_UPDATE: LOD levels are static once built (their vertex
            // source never refits in place — a geometry-version bump instead
            // destroys the whole chain, see the geomVersion-changed rebuild
            // path in VulkanCoreScene.cpp), so PREFER_FAST_TRACE alone is
            // strictly better (smaller build, faster trace).
            blasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            blasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            blasBuild.geometryCount = 1;
            blasBuild.pGeometries = &blasGeom;

            VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
            blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(),
                    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &blasBuild, &primitiveCount, &blasSizes);

            out.storage = createBuffer(
                    ctx->allocator(), ctx->device(), blasSizes.accelerationStructureSize,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);

            VkAccelerationStructureCreateInfoKHR blasCreate{};
            blasCreate.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            blasCreate.buffer = out.storage.handle;
            blasCreate.size = blasSizes.accelerationStructureSize;
            blasCreate.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            check(ctx->rt().createAccelerationStructure(ctx->device(), &blasCreate, nullptr, &out.as),
                  "vkCreateAccelerationStructureKHR(LOD level BLAS)");

            // The AS device address is a property of the storage binding —
            // valid immediately at creation, before the build executes.
            VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
            addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
            addrInfo.accelerationStructure = out.as;
            out.address = ctx->rt().getAccelerationStructureDeviceAddress(ctx->device(), &addrInfo);

            LodPendingBuild build{};
            build.as = out.as;
            build.vertexAddress = rec.vertex.address;
            build.indexAddress = out.index.address;
            build.maxVertex = rec.vertexCount - 1;
            build.primitiveCount = primitiveCount;
            build.packedIdx = packIdx;
            // Per-build scratch — concurrent builds recorded into one command
            // buffer must not alias scratch memory (same rule as
            // refreshGeomBlasBatch's per-record persistent scratch).
            build.scratch = createAsScratchBuffer(ctx->allocator(), ctx->device(), blasSizes.buildScratchSize);
            pending.push_back(build);

            out.indexCount  = static_cast<uint32_t>(level.indices.size());
            out.errorBound  = level.error;
            return true;
        }

void VulkanRenderer::Impl::flushLodLevelBuilds(std::vector<LodPendingBuild>& pending) {
            if (pending.empty()) return;
            const uint32_t N = static_cast<uint32_t>(pending.size());
            // The build-info structs must stay pointer-stable until the GPU
            // executes them (pGeometries / rangePtrs are read at submit), so
            // they live in N-sized vectors that outlast the submit-wait —
            // same shape as refreshGeomBlasBatch's Phase D.
            std::vector<VkAccelerationStructureGeometryTrianglesDataKHR> triDatas(N);
            std::vector<VkAccelerationStructureGeometryKHR>              blasGeoms(N);
            std::vector<VkAccelerationStructureBuildGeometryInfoKHR>     blasBuilds(N);
            std::vector<VkAccelerationStructureBuildRangeInfoKHR>        ranges(N);
            std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePtrs(N);
            for (uint32_t k = 0; k < N; ++k) {
                const auto& b = pending[k];
                triDatas[k].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                triDatas[k].vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                triDatas[k].vertexData.deviceAddress = b.vertexAddress;
                triDatas[k].vertexStride = 3 * sizeof(float);
                triDatas[k].maxVertex = b.maxVertex;
                triDatas[k].indexType = b.packedIdx ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
                triDatas[k].indexData.deviceAddress = b.indexAddress;

                blasGeoms[k].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
                blasGeoms[k].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                blasGeoms[k].geometry.triangles = triDatas[k];
                blasGeoms[k].flags = 0;

                blasBuilds[k].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
                blasBuilds[k].type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                blasBuilds[k].flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
                blasBuilds[k].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
                blasBuilds[k].geometryCount = 1;
                blasBuilds[k].pGeometries = &blasGeoms[k];
                blasBuilds[k].dstAccelerationStructure = b.as;
                blasBuilds[k].scratchData.deviceAddress = b.scratch.address;

                ranges[k].primitiveCount = b.primitiveCount;
                rangePtrs[k] = &ranges[k];
            }

            VkCommandBuffer cb = beginOneShot();
            ctx->rt().cmdBuildAccelerationStructures(cb, N, blasBuilds.data(), rangePtrs.data());
            endAndSubmitOneShot(cb, "auto-LOD level BLAS batch");

            for (auto& b : pending) destroyBuffer(ctx->allocator(), b.scratch);
            pending.clear();
        }

// Drains finished chains up to the per-frame budget (16 geometries OR 8 MiB
// of new level resources, whichever hits first — see the declaration) and
// turns them into GPU resources. All of the frame's level BLAS builds are
// recorded into ONE one-shot submit at the end. Discards stale results: the
// record may have been evicted (geometry destroyed) or its geomVersion may
// have moved on (the user mutated vertices in place after the job was
// snapshotted) since the job was enqueued — either way the simplified data
// no longer matches anything live, so it's dropped; lodState resets to None
// so a fresh job gets queued next time the (now-different) geometry is seen
// as eligible. Stale/failed results don't count against the budget — they
// create no resources.
void VulkanRenderer::Impl::drainLodResults() {
            // Per-frame finalize budget. Deliberately small: every drain that
            // builds anything ends in flushLodLevelBuilds, whose one-shot submit
            // WAITS on the graphics queue — so the budget is not a throughput
            // knob, it is the size of a stall landing inside render(). At 16
            // geoms / 8 MiB a scene that brings a lot of geometry into view at
            // once (spot_slam's 46 vegetation chains) paid ~40 ms hitches every
            // time a batch landed. Two geometries / 1 MiB spreads the same total
            // work over more frames, trading a few frames of latency before a
            // chain becomes selectable for a stall small enough to stay inside
            // a 60 Hz budget.
            constexpr uint32_t kMaxGeomsPerFrame = 2;
            constexpr uint64_t kMaxNewBytesPerFrame = 1ull * 1024ull * 1024ull;

            uint32_t finalizedGeoms = 0;
            uint64_t newBytes = 0;
            std::vector<LodPendingBuild> pending;

            while (finalizedGeoms < kMaxGeomsPerFrame && newBytes < kMaxNewBytesPerFrame) {
                LodResult result;
                {
                    std::lock_guard<std::mutex> lk(lodResultMutex_);
                    if (lodResultQueue_.empty()) break;
                    result = std::move(lodResultQueue_.front());
                    lodResultQueue_.pop_front();
                }
                --lodChainsQueuedCount_;

                auto it = blasCache.find(result.geom);
                if (it == blasCache.end()) continue;// record evicted while the job was in flight
                BlasRecord& rec = *it->second;
                if (rec.geomVersion != result.geomVersion) {
                    // Stale — drop WITHOUT touching lodState. Every version-
                    // advance path already reset it (refreshGeomBlasBatch's
                    // destroyBlasLodLevels, record eviction), and a job for
                    // the CURRENT version may be Queued right now — resetting
                    // to None here would let selection enqueue a duplicate,
                    // whose result would then append a second copy of every
                    // level onto the Ready chain.
                    continue;
                }
                if (rec.lodState != BlasRecord::LodState::Queued) {
                    // Version matches but no job is outstanding for it —
                    // this is the second result of a double-enqueue (or the
                    // state moved on some other way). Appending would
                    // duplicate the chain's levels and double-count stats.
                    continue;
                }
                if (result.levels.empty()) {
                    rec.lodState = BlasRecord::LodState::Failed;
                    continue;
                }

                rec.lodLevels.reserve(result.levels.size());
                for (const auto& lvl : result.levels) {
                    // HARD budget enforcement at allocation time. The enqueue
                    // gate only sees RESIDENT bytes — a burst of jobs queued
                    // while still under budget would otherwise finalize far
                    // past the cap once their results return. Stopping
                    // mid-chain keeps the FINER levels already built (a valid,
                    // conservative chain — errors stay monotonic); levels that
                    // didn't fit are simply never selectable.
                    if (lodIndexBytes_ + lodBlasBytes_ >= kLodByteBudget) {
                        if (!lodBudgetWarned_) {
                            std::cerr << "[VulkanRenderer] auto-LOD: 256 MiB byte budget reached - "
                                         "truncating chains; no further levels will be built this session\n";
                            lodBudgetWarned_ = true;
                        }
                        break;
                    }
                    BlasRecord::LodLevel out{};
                    if (!buildLodLevelFor(rec, lvl, out, pending)) continue;
                    lodIndexBytes_ += out.index.size;
                    lodBlasBytes_  += out.storage.size;
                    newBytes       += out.index.size + out.storage.size;
                    rec.lodLevels.push_back(out);
                }
                if (rec.lodLevels.empty()) {
                    // Includes the over-budget-before-first-level case: Failed
                    // (not None) so selection doesn't spin re-enqueuing while
                    // the cap holds. Deliberately never retried this session.
                    rec.lodState = BlasRecord::LodState::Failed;
                } else {
                    rec.lodState = BlasRecord::LodState::Ready;
                    ++lodChainsReadyCount_;
                    ++finalizedGeoms;
                }
            }

            flushLodLevelBuilds(pending);
        }

void VulkanRenderer::Impl::refreshSkinnedBlas(SkinnedMesh& sm, SkinnedMeshState& st) {
            if (!st.blas || !sm.skeleton || st.boneCount == 0) return;

            // Flatten the pose first. The per-frame deformer scan already calls
            // update() for every skinned mesh it has a state for, but
            // ensureSkinnedBlas reaches here for a mesh the scan has never seen
            // — its boneMatrices is still the constructor's zero-filled vector,
            // and adopting that below would collapse the character onto the
            // origin for its first frame, in the ray-traced shadows and
            // reflections too. update() is idempotent, and its bone-texture
            // branch is dead on this backend (boneTexture is only ever built by
            // GLRenderer).
            sm.skeleton->update();

            // Per-bone matrix = bones[b]->matrixWorld * boneInverse[b] — which
            // is precisely the array update() just flattened, so take it whole
            // rather than rebuilding it a bone at a time. Upload into the
            // bones[..] section of the host-visible boneMatrices buffer
            // (skipping the [bindMat, bindInv] prefix written once in
            // ensureSkinnedBlas).
            const auto& skel = *sm.skeleton;
            // Advance the ring FIRST: this write must not land in the slot an
            // in-flight dispatch is still reading (see
            // SkinnedMeshState::boneMatrices). recordCommandBuffer dispatches
            // with skinDescSet[boneSlot], so the two stay in step.
            st.boneSlot = (st.boneSlot + 1u) % SkinnedMeshState::kBoneSlots;
            auto& slot = st.boneMatrices[st.boneSlot];
            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), slot.alloc, &mapped);
            // mats[1] = bindMatrixInverse is NOT constant in attached bind mode:
            // SkinnedMesh::updateMatrixWorld recomputes it to matrixWorld^-1 every
            // frame so that (TLAS instance transform · bindMatrixInverse) collapses
            // to a single world application. ensureSkinnedBlas wrote it once at the
            // bind pose (≈identity at the origin), so a *moving* skinned mesh gets
            // its world transform applied twice — invisible at the origin, wrong
            // once it translates/rotates. Re-upload it every frame.
            std::memcpy(static_cast<char*>(mapped) + 16 * sizeof(float),
                        sm.bindMatrixInverse.elements.data(), 16 * sizeof(float));
            char* dst = static_cast<char*>(mapped) + 32 * sizeof(float);
            // The admission test is on the BONE count, not the float count:
            // computeBoneTexture() swaps boneMatrices for a power-of-two padded
            // vector, so a float-count test would pass happily while the copy
            // ran off the end of the real bones into padding zeros. Both
            // conditions hold for every skeleton Skeleton::create builds; the
            // bone-at-a-time fallback covers a skeleton whose bone list was
            // mutated behind this state's back.
            //
            // A null bone slot now uploads boneInverses[b] where the old loop
            // uploaded identity — that is what Skeleton has always flattened
            // into boneMatrices, and what the GL path and three.js both use, so
            // this closes a divergence rather than opening one.
            if (skel.bones.size() >= st.boneCount &&
                skel.boneMatrices.size() >= size_t(st.boneCount) * 16u) {
                std::memcpy(dst, skel.boneMatrices.data(),
                            size_t(st.boneCount) * 16u * sizeof(float));
            } else {
                for (uint32_t b = 0; b < st.boneCount; ++b) {
                    Matrix4 m;
                    if (b < skel.bones.size() && skel.bones[b]) {
                        m.multiplyMatrices(*skel.bones[b]->matrixWorld, skel.boneInverses[b]);
                    }
                    std::memcpy(dst + b * 16 * sizeof(float),
                                m.elements.data(), 16 * sizeof(float));
                }
            }
            flushHostWrites(ctx->allocator(), slot.alloc);
            vmaUnmapMemory(ctx->allocator(), slot.alloc);

            // Cache the canonical bone matrices for next-frame dirty detection
            // (still uses Skeleton::boneMatrices since that's the host-visible
            // change signal the dirty path reads).
            const auto& bm = skel.boneMatrices;
            if (st.prevBoneMats.size() == bm.size()) {
                std::memcpy(st.prevBoneMats.data(), bm.data(), bm.size() * sizeof(float));
            } else {
                st.prevBoneMats = bm;
            }

            // Queue for per-frame skinning compute + BLAS rebuild. The actual
            // GPU work is recorded into the main command buffer at the start
            // of recordCommandBuffer — no blocking submit here.
            pendingSkinnedRebuilds_.push_back(&st);
        }

void VulkanRenderer::Impl::refreshTetBlas(Mesh& m, TetMeshState& st) {
            if (!st.blas) return;
            // Zero-copy interop: the registered CUDA device→device copy writes the
            // deformed tet positions straight into the exported binding-6 buffer —
            // no host readback, no DataTexture, no map/memcpy. The external buffer
            // is single-instance (bound in every ring slot), so unlike the CPU
            // path below it can still overlap a still-in-flight prior frame's
            // dispatch; ring-buffering it would need one exported allocation +
            // CUDA import per slot.
            if (st.tetPosExt.handle != VK_NULL_HANDLE) {
                if (st.tetPosExternalCopy) st.tetPosExternalCopy();
                pendingTetRebuilds_.push_back(&st);
                return;
            }
            auto mat = m.material();
            if (!mat || !mat->tetTexture) return;
            const auto& tetImg = mat->tetTexture->image().data<float>();
            const VkDeviceSize bytes = std::min<VkDeviceSize>(
                    st.tetPosBytes, static_cast<VkDeviceSize>(tetImg.size()) * sizeof(float));
            if (bytes == 0) return;
            // Advance the ring so this write never lands in a buffer an in-flight
            // frame's dispatch still reads (which showed the same physics state on
            // consecutive frames — visible stutter at a steady frame rate).
            st.tetPosSlot = (st.tetPosSlot + 1) % TetMeshState::kTetPosSlots;
            uploadHostVisible(ctx->allocator(), st.tetPos[st.tetPosSlot], tetImg.data(), bytes);
            pendingTetRebuilds_.push_back(&st);
        }

void VulkanRenderer::Impl::rewriteTetPosBinding(TetMeshState& st, VkBuffer buf) {
            VkDescriptorBufferInfo bi{};
            bi.buffer = buf;
            bi.offset = 0;
            bi.range  = VK_WHOLE_SIZE;
            for (auto ds : st.tetDescSet) {
                if (ds == VK_NULL_HANDLE) continue;
                VkWriteDescriptorSet wr{};
                wr.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wr.dstSet          = ds;
                wr.dstBinding      = 6;
                wr.descriptorCount = 1;
                wr.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                wr.pBufferInfo     = &bi;
                vkUpdateDescriptorSets(ctx->device(), 1, &wr, 0, nullptr);
            }
        }

void VulkanRenderer::Impl::disableSoftBodyInterop(const Mesh& mesh) {
            auto it = tetMeshStates.find(&mesh);
            if (it == tetMeshStates.end()) return;
            auto& st = *it->second;
            if (st.tetPosExt.handle == VK_NULL_HANDLE) return;
            check(vkDeviceWaitIdle(ctx->device()), "vkDeviceWaitIdle (softbody interop disable)");
            // Recreate the ring and point each slot's set back at its own buffer
            // (rewriteTetPosBinding writes ONE buffer into all slots — interop
            // only; here every slot must get its distinct ring buffer back).
            for (uint32_t s = 0; s < TetMeshState::kTetPosSlots; ++s) {
                st.tetPos[s] = createBuffer(
                        ctx->allocator(), ctx->device(), st.tetPosBytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                if (st.tetDescSet[s] == VK_NULL_HANDLE) continue;
                VkDescriptorBufferInfo bi{};
                bi.buffer = st.tetPos[s].handle;
                bi.offset = 0;
                bi.range  = VK_WHOLE_SIZE;
                VkWriteDescriptorSet wr{};
                wr.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                wr.dstSet          = st.tetDescSet[s];
                wr.dstBinding      = 6;
                wr.descriptorCount = 1;
                wr.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                wr.pBufferInfo     = &bi;
                vkUpdateDescriptorSets(ctx->device(), 1, &wr, 0, nullptr);
            }
            vulkan::destroyExternalBuffer(ctx->device(), st.tetPosExt);
            st.tetPosExternalCopy = nullptr;
        }

// ── Zero-copy MESH VERTEX interop (plan W3) ─────────────────────────────────
// The record a mesh's positions live in, or null. Only plain cached geometry
// qualifies: every other flavour (skinned / tet / displaced / grass / morphed)
// already has a per-frame producer writing the very buffers the foreign API
// would write, so arming interop on one is a silent fight over the memory
// rather than a feature. Refuse it at the door.
vulkan::impl::BlasRecord* VulkanRenderer::Impl::interopRecordFor(const Mesh& mesh) {
            // dynamic_cast rather than a pointer-value probe of each map: the
            // maps are keyed by DIFFERENT static types, and casting a Mesh* to
            // one of them just to call count() would be a lie whenever the mesh
            // is not that type (and would compare an address that means nothing).
            if (dynamic_cast<const SkinnedMesh*>(&mesh) ||
                dynamic_cast<const DisplacedMesh*>(&mesh) ||
                dynamic_cast<const GrassMesh*>(&mesh)) return nullptr;
            if (tetMeshStates.count(&mesh) || morphedMeshStates.count(&mesh)) return nullptr;
            const auto& geom = mesh.geometry();
            if (!geom) return nullptr;
            auto it = blasCache.find(geom.get());
            return it == blasCache.end() ? nullptr : it->second.get();
        }

VulkanRenderer::VertexInteropHandle
VulkanRenderer::Impl::enableVertexInterop(const Mesh& mesh, std::function<void()> deviceCopy,
                                          bool validate, bool stableCorrespondence) {
            // Every failure is a NULL HANDLE, never a throw. The record only
            // exists after the mesh's first render (buildBlasFor runs in
            // ensureSceneBuilt), so "not yet" is the expected answer to a call
            // made at setup time and the caller is documented to poll — the
            // same contract the soft-body glue's needsVkInteropRegister() loop
            // was built around.
            BlasRecord* recPtr = interopRecordFor(mesh);
            if (!recPtr || !ctx->externalMemorySupported()) return {};
            if (recPtr->vertex.handle == VK_NULL_HANDLE ||
                recPtr->normal.handle == VK_NULL_HANDLE || recPtr->vertexCount == 0u) return {};
            // ── The packed → unpacked handoff ────────────────────────────────
            // A packed record's normal buffer is oct-snorm16x2, not float xyz
            // (packedMask bit 0), and uv/color are narrowed likewise. No foreign
            // producer can be expected to write that encoding, so interop needs
            // an UNPACKED record. And this is the COMMON case, not a corner one:
            // every ordinary mesh reaches blasCache through
            // buildBlasFor(geom, allowPacked=true) (VulkanCoreScene.cpp, the
            // missing-record branch), so a plain static mesh — precisely the
            // interop target — is packed by default. Rejecting here made the
            // whole API unreachable outside THREEPP_NO_PACK=1.
            //
            // The mark is what makes the rebuild — and every LATER rebuild of this
            // geometry — come back unpacked; buildBlasFor is the single funnel
            // that reads it. It outlives this call deliberately: a subsequent
            // geomVersion-mismatch rebuild that re-packed would silently break
            // the producer's normal writes.
            //
            // The rebuild itself happens HERE, behind a device drain, rather than
            // being deferred to the next frame's structural pass. The deferred
            // shape was the first design and it is the safer-looking one, but it
            // makes arming take TWO polls instead of one (record-exists, then
            // record-unpacked), and callers written to the documented contract —
            // "render once, then enable" — arm once and take the fallback branch
            // on the null return. Registration-time teardown behind
            // vkDeviceWaitIdle is not new ground: enableSoftBodyInterop does
            // exactly this, for exactly this reason (a rare registration call, in
            // exchange for not having to reason about in-flight readers).
            //
            // The drain retires every in-flight reader of the old buffers; the
            // stale ADDRESSES that outlive them live only in host-side arrays
            // (geomDescs / drawInfos / TLAS instances). lodChangedThisFrame_ plus
            // the interop record's own unconditional per-frame enqueue (which sets
            // the TLAS-refit gate) republish all three through the ordinary
            // machinery before the next submit, so nothing here hand-rolls a
            // descriptor update — and, unlike a forced structural rebuild, the
            // interop enqueue still runs on that frame.
            //
            // The same rebuild also runs for a record that is ALREADY unpacked
            // (THREEPP_NO_PACK=1, or geometry whose attributes never packed)
            // but whose BLAS storage predates the interop mark: interop refits
            // build with PREFER_FAST_BUILD, and buildBlasFor only sizes the
            // storage for that lineage once the geometry is in
            // forceUnpackedGeoms_ (storageFitsFastBuild). Undersized storage
            // is not cosmetic — a FAST_BUILD build into it overruns the
            // structure (VUID 10126 / device-lost) — and replacing the whole
            // record through this branch reuses the drain + republish
            // machinery instead of hand-rolling an AS-only swap whose new
            // device address the TLAS instances would have to chase.
            if (recPtr->packedMask != 0u || !recPtr->storageFitsFastBuild) {
                const auto geomSp = mesh.geometry();
                forceUnpackedGeoms_.insert(geomSp.get());
                check(vkDeviceWaitIdle(ctx->device()),
                      "vkDeviceWaitIdle (vertex interop unpacked rebuild)");
                flushRetireQueue();// device idle ⇒ reclaim now, don't carry across
                auto fresh = buildBlasFor(*geomSp, /*allowPacked=*/true);// mark forces false
                if (!fresh) {
                    std::cerr << "[VulkanRenderer] enableVertexInterop: unpacked rebuild of the "
                                 "geometry failed - the mesh keeps its CPU attribute path.\n";
                    return {};
                }
                fresh->liveCheck = geomSp;
                auto cIt = blasCache.find(geomSp.get());
                if (cIt == blasCache.end()) return {};// cannot happen: rec came from here
                // The LOD chain simplified the PACKED record's data and its level
                // BLASes reference buffers about to die. destroyBlasLodLevels, not
                // a hand-rolled free — it keeps the lodBlasBytes_ accounting.
                destroyBlasLodLevels(*cIt->second);
                destroyBlasRecord(*cIt->second);
                cIt->second = std::move(fresh);
                lodChangedThisFrame_ = true;// the effective LOD level just became 0
                recPtr = cIt->second.get();

                // The swap just freed every buffer the entries-indexed
                // GeometryDesc mirror names for this geometry, and the ordinary
                // machinery does NOT republish those slots: the TLAS instance
                // fill and the indirect DrawInfo build re-read the record, but
                // geomDescsCached_ is only ever patched per-entry on a LOD
                // switch — indexAddress alone, and only for a record that HAS
                // a chain, which this one no longer does. Left stale, every RT
                // hit on this mesh (shadow/GI/reflection/probe/lidar) fetches
                // indices and vertices through the freed device addresses.
                // That reads as last frame's data for as long as the allocator
                // keeps the old block bytes — and turns into wild indices, a
                // fetch gigabytes off the heap and VK_ERROR_DEVICE_LOST the
                // moment the region is recycled (first observed via the per-
                // frame refit's scratch regrowth under a deforming producer).
                // Rewrite the slots in place, exactly as the full rebuild
                // fills them, and stage the change for every frame in flight.
                const auto lodSel0 = selectLodGeom(*recPtr, 0);
                for (size_t i = 0; i < lastVisibleEntries_.size(); ++i) {
                    const MeshEntry& en = lastVisibleEntries_[i];
                    if (en.isOverlay || !en.mesh) continue;
                    if (en.mesh->geometry().get() != geomSp.get()) continue;
                    if (i >= geomDescsCached_.size()) continue;
                    auto& gd = geomDescsCached_[i];
                    gd.vertexAddress = recPtr->vertex.address;
                    gd.normalAddress = recPtr->normal.address;
                    gd.indexAddress  = lodSel0.indexAddress;
                    gd.uvAddress     = recPtr->uv.address;
                    gd.foamAddress   = recPtr->isOceanSurface ? 1ull : 0ull;
                    // interopWorldStatic is not set on the fresh record yet;
                    // the caller's flag is what it is about to become.
                    gd.prevVertexAddress =
                            (recPtr->prevVertex.handle != VK_NULL_HANDLE &&
                             stableCorrespondence)
                                    ? recPtr->prevVertex.address
                                    : recPtr->vertex.address;
                    const auto mat = en.mesh->material();
                    gd.colorAddress = (recPtr->color.handle != VK_NULL_HANDLE &&
                                       mat && mat->vertexColors)
                                              ? recPtr->color.address
                                              : 0ull;
                    gd.indexed = lodSel0.indexed ? 1u : 0u;
                    // Preserve bit 0 (moved-sticky); the fresh record is
                    // unpacked, so the packed bits above it all clear.
                    gd.flags = (gd.flags & 1u) | (recPtr->packedMask << 1);
                    markGeomDescsDirty(static_cast<uint32_t>(i));
                }
                // The DrawInfo skip signature cannot see this swap either — on
                // an otherwise-static scene it would keep serving the raster
                // vertex-pull the freed addresses verbatim.
                ++drawInputsVersion_;
            }
            // Bound only now: the packed branch above may have replaced the record.
            auto& rec = *recPtr;

            if (rec.interop) {// already enabled — re-arm the copy, same allocations
                rec.externalCopy = std::move(deviceCopy);
                // A re-enable may turn validation ON after a first call left it
                // off, so the set is allocated here too rather than only on the
                // cold path. (Turning it off just stops dispatching; the set is
                // kept, because the next re-enable is one bool away.)
                if (validate && rec.sanitizeDS == VK_NULL_HANDLE) {
                    if (!vertexSanitize_)
                        vertexSanitize_ = std::make_unique<vulkan::VertexSanitizePipeline>(*ctx);
                    rec.sanitizeDS = vertexSanitize_->allocateRecordDescriptorSet(rec.posExt.handle);
                }
                rec.interopValidate = validate && rec.sanitizeDS != VK_NULL_HANDLE;
                rec.interopWorldStatic = !stableCorrespondence;
                return {vulkan::takeOsHandle(ctx->device(), rec.posExt),
                        static_cast<size_t>(rec.posExt.size),
                        vulkan::takeOsHandle(ctx->device(), rec.nrmExt),
                        static_cast<size_t>(rec.nrmExt.size)};
            }

            // STORAGE | TRANSFER_SRC and deliberately NOT SHADER_DEVICE_ADDRESS:
            // createExternalBuffer's dedicated allocation carries no
            // VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, and a buffer that asks for
            // an address on memory not allocated for one is invalid. TRANSFER_SRC
            // is what the renderer actually needs (these are copy sources);
            // STORAGE is what the sanitize dispatch binds. Same reasoning, same
            // flags as ParticleFieldPass::kExternalPositionUsage.
            constexpr VkBufferUsageFlags kExtUsage =
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            // NO device drain at enable, and that is structural rather than an
            // oversight — the same argument ParticleFieldPass::enableInterop
            // makes. enableSoftBodyInterop drains because it SWAPS a buffer
            // in-flight dispatches are already reading and rewrites the
            // descriptor sets naming it. Here nothing is swapped and nothing is
            // rebound: the exports are brand-new allocations no submitted
            // command buffer can name, and rec.vertex/rec.normal keep every
            // address they had. There is nothing in flight to wait for.
            try {
                rec.posExt = vulkan::createExternalBuffer(ctx->physicalDevice(), ctx->device(),
                                                          rec.vertex.size, kExtUsage);
                rec.nrmExt = vulkan::createExternalBuffer(ctx->physicalDevice(), ctx->device(),
                                                          rec.normal.size, kExtUsage);
            } catch (const std::exception& e) {
                vulkan::destroyExternalBuffer(ctx->device(), rec.posExt);
                vulkan::destroyExternalBuffer(ctx->device(), rec.nrmExt);
                std::cerr << "[VulkanRenderer] enableVertexInterop: export failed (" << e.what()
                          << ") - the mesh keeps its CPU attribute path.\n";
                return {};
            }

            if (validate) {
                if (!vertexSanitize_)
                    vertexSanitize_ = std::make_unique<vulkan::VertexSanitizePipeline>(*ctx);
                rec.sanitizeDS = vertexSanitize_->allocateRecordDescriptorSet(rec.posExt.handle);
                if (rec.sanitizeDS == VK_NULL_HANDLE) {
                    std::cerr << "[VulkanRenderer] enableVertexInterop: sanitize descriptor pool "
                                 "exhausted (VertexSanitizePipeline::kMaxRecords) - this mesh runs "
                                 "interop WITHOUT the finiteness guard.\n";
                }
            }
            rec.interopValidate = validate && rec.sanitizeDS != VK_NULL_HANDLE;
            rec.externalCopy = std::move(deviceCopy);
            rec.interop = true;
            rec.interopWorldStatic = !stableCorrespondence;

            // Force the graduated per-frame residency rather than waiting for
            // kDynamicGraduationStreak dirty frames that will never arrive:
            // under interop the CPU never touches the attributes
            // again, so the streak counter can't graduate this record. The
            // staging ring the normal graduation allocates is NOT allocated here
            // — under interop the copy source is posExt/nrmExt and the host pack
            // that fills dynStaging is skipped entirely (see
            // recordDynamicGeomRefits).
            rec.perFrameDynamic = true;
            rec.prevVertexResyncPending = false;// settles through the frame cb now
            // Pin the geometry unpacked for life: this record is unpacked now,
            // but a later geomVersion-mismatch rebuild goes through
            // buildBlasFor(geom, allowPacked=true) and would re-pack it,
            // silently breaking the producer's normal writes. Idempotent — the
            // packed branch above already inserted it on the path that got here
            // by rebuilding.
            forceUnpackedGeoms_.insert(mesh.geometry().get());
            //
            // NO structural rebuild is requested here, and that is load-bearing
            // rather than an omission. The culls read MeshEntry::isVertexInterop,
            // and the obvious way to refresh it is to force the entry list to be
            // re-expanded — but forcing the fullRebuild path on the frame right
            // after arming makes the fluid demo render an EMPTY BASIN from then
            // on. ensureSceneBuilt's structural path does not run the
            // unconditional interop enqueue (that lives on the incremental,
            // structurally-same branch), so the arming frame publishes a TLAS,
            // a set of GeometryDescs and a G-buffer draw built from the record's
            // untouched CPU-side vertex buffer, and the producer's first write
            // never gets folded in. The flag is instead refreshed IN PLACE every
            // frame by that same enqueue loop in VulkanCoreScene.cpp, which is
            // both cheaper and impossible to leave stale.
            std::cerr << "[VulkanRenderer] vertex interop armed on a " << rec.vertexCount
                      << "-vertex geometry (" << rec.vertex.size << " + " << rec.normal.size
                      << " bytes exported, sanitize "
                      << (rec.interopValidate ? "on" : "off") << ")\n";

            return {vulkan::takeOsHandle(ctx->device(), rec.posExt),
                    static_cast<size_t>(rec.posExt.size),
                    vulkan::takeOsHandle(ctx->device(), rec.nrmExt),
                    static_cast<size_t>(rec.nrmExt.size)};
        }

// Release the exports and hand the mesh back to the CPU-driven path. Mirrors
// disableSoftBodyInterop: the caller is responsible for having stopped the
// foreign writes first — nothing here can wait for a CUDA stream.
//
// perFrameDynamic is intentionally left set. It is one-way for a record's
// lifetime (see the comment on BlasRecord), and the graduated
// path is strictly better than the drained one for a record whose attributes
// were being rewritten every frame a moment ago. The record simply resumes
// taking its data from the host arrays, gated on BufferAttribute::version like
// any other graduated mesh — which means an application that disables interop
// must call needsUpdate() again for its edits to land.
void VulkanRenderer::Impl::disableVertexInterop(const Mesh& mesh) {
            BlasRecord* recPtr = interopRecordFor(mesh);
            if (!recPtr || !recPtr->interop) return;
            auto& rec = *recPtr;
            rec.interop = false;
            rec.externalCopy = nullptr;
            rec.interopValidate = false;
            // MeshEntry::isVertexInterop goes back to false on its own: the
            // enqueue loop in ensureSceneBuilt rewrites it from the record every
            // frame, for every plain entry, clearing as well as setting. Nothing
            // to invalidate here. (The forceUnpackedGeoms_ mark deliberately
            // STAYS — re-packing the geometry would only buy back a few bytes
            // and would break a re-arm.)
            // Retire, don't destroy: an in-flight frame's head-of-frame copy may
            // still be READING these as a transfer source. destroyExternalBuffer
            // frees the VkDeviceMemory immediately, and there is no VMA-style
            // deferred-free list for external allocations — so this one place
            // does pay the drain enable skipped. The sanitize descriptor set is
            // freed AFTER the drain for the same reason: the last frame's
            // sanitize dispatch may still name it (VUID-vkFreeDescriptorSets-
            // pDescriptorSets-00309 at teardown otherwise).
            check(vkDeviceWaitIdle(ctx->device()), "vkDeviceWaitIdle (vertex interop disable)");
            if (rec.sanitizeDS != VK_NULL_HANDLE && vertexSanitize_) {
                vertexSanitize_->freeRecordDescriptorSet(rec.sanitizeDS);
                rec.sanitizeDS = VK_NULL_HANDLE;
            }
            vulkan::destroyExternalBuffer(ctx->device(), rec.posExt);
            vulkan::destroyExternalBuffer(ctx->device(), rec.nrmExt);
            // The staging ring the graduated path needs was never allocated (the
            // interop route skips the host pack), so allocate it now or the next
            // host-driven refit copies from a null buffer.
            if (rec.dynStaging.handle == VK_NULL_HANDLE && rec.vbBytes > 0) {
                rec.dynStagingSlotBytes = rec.vbBytes + rec.normal.size;
                rec.dynStaging = createBuffer(
                        ctx->allocator(), ctx->device(),
                        rec.dynStagingSlotBytes * kFramesInFlight,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            }
        }

void VulkanRenderer::Impl::refreshGeomBlasBatch(const std::vector<VulkanRenderer::Impl::GeomRefreshOp>& ops) {
            if (ops.empty()) return;

            // Phase A — validate positions are finite. Non-finite values cause
            // VK_ERROR_DEVICE_LOST during BLAS build on NVIDIA. Skip individual
            // bad ops rather than throwing the whole batch out.
            std::vector<size_t> liveOps;
            liveOps.reserve(ops.size());
            for (size_t k = 0; k < ops.size(); ++k) {
                auto* posAttr = ops[k].geom->getAttribute<float>("position");
                if (!posAttr) continue;
                // Float-typed on purpose: this is the per-frame deforming
                // refresh, and only float geometry ever reaches it (physics /
                // morph / displacement all write float). Narrow attributes are
                // a static-geometry feature — see buildBlasFor.
                if (!ops[k].geom->getAttribute<float>("normal")) continue;
                bool ok = true;
                const auto& p = posAttr->array();
                for (size_t i = 0; i < p.size(); ++i) {
                    if (!std::isfinite(p[i])) {
                        std::cerr << "[VulkanRenderer] refreshGeomBlasBatch: skipping geom - "
                                  << "position[" << i << "] is non-finite (" << p[i] << ")\n";
                        ok = false;
                        break;
                    }
                }
                if (ok) liveOps.push_back(k);
            }
            if (liveOps.empty()) return;

            // Phase B — snapshot current vertex into prevVertex for the chit's
            // per-vertex motion vector. Recorded into one shared cmdbuf; submit
            // + wait once. Must run before the host memcpy of new positions
            // below or we'd snapshot the new state, not last frame's.
            {
                bool any = false;
                VkCommandBuffer cb = beginOneShot();
                for (size_t k : liveOps) {
                    auto& rec = *ops[k].rec;
                    if (rec.prevVertex.handle == VK_NULL_HANDLE) continue;
                    auto* posAttr = ops[k].geom->getAttribute<float>("position");
                    VkBufferCopy region{};
                    region.size = posAttr->array().size() * sizeof(float);
                    vkCmdCopyBuffer(cb, rec.vertex.handle, rec.prevVertex.handle, 1, &region);
                    // This snapshot is the PRE-update geometry. It is correct for
                    // the change frame's motion vector, but must be re-synced to
                    // the new positions on the next clean frame (see the resync
                    // pass in ensureSceneBuilt) or the mesh shakes forever.
                    rec.prevVertexResyncPending = true;
                    any = true;
                }
                if (any) {
                    endAndSubmitOneShot(cb, "refreshGeomBlasBatch (prevVertex)");
                } else {
                    // No prev buffers attached — skip the empty submit so we
                    // don't round-trip the queue for zero work.
                    vkEndCommandBuffer(cb);
                    vkFreeCommandBuffers(ctx->device(), cmdPool, 1, &cb);
                }
            }

            // Phase C — host writes into the now-snapshotted vertex/normal/uv/
            // index buffers (flushed for non-coherent portability); the
            // implicit submit barrier on the next phase makes these visible
            // to the BLAS build.
            // Vertex-range of an attribute's pending edit, honouring
            // BufferAttribute::updateRange (offset/count in SCALAR elements,
            // count == -1 ⇒ whole array — the same contract GLAttributes has
            // always implemented). Without this a geometry preallocated at
            // capacity and updated in place — a growing reconstruction surface,
            // a dynamic trail — re-uploaded the ENTIRE allocation and, for
            // packed normals, re-encoded every vertex, so the cost tracked the
            // capacity instead of the edit (measured: a 600k-vertex capacity
            // holding ~150k live vertices spent ~15 ms in the frame that
            // touched it). Snapped to whole vertices so the packed encoders
            // never see a partial vertex.
            auto vertexRangeOf = [](const FloatBufferAttribute& a, uint32_t itemSize)
                    -> std::pair<uint32_t, uint32_t> {
                const auto total = static_cast<uint32_t>(a.array().size() / itemSize);
                const auto& r = a.updateRange;
                if (r.count < 0) return {0u, total};
                const uint32_t first = static_cast<uint32_t>(r.offset) / itemSize;
                const uint32_t last  = (static_cast<uint32_t>(r.offset + r.count) + itemSize - 1u) / itemSize;
                if (first >= total) return {0u, 0u};
                return {first, std::min(last, total) - first};
            };

            for (size_t k : liveOps) {
                const auto& geom = *ops[k].geom;
                auto& rec = *ops[k].rec;
                auto* posAttr = geom.getAttribute<float>("position");
                auto* nrmAttr = geom.getAttribute<float>("normal");

                const auto [pFirst, pCount] = vertexRangeOf(*posAttr, 3u);
                if (pCount > 0) {
                    uploadHostVisible(ctx->allocator(), rec.vertex,
                                      posAttr->array().data() + static_cast<size_t>(pFirst) * 3u,
                                      static_cast<VkDeviceSize>(pCount) * 3u * sizeof(float),
                                      static_cast<VkDeviceSize>(pFirst) * 3u * sizeof(float));
                }
                // Consume the range, exactly as GLAttributes::updateBuffer does:
                // a later needsUpdate() that sets no range must mean "all of
                // it", and without clearing it here the two backends would
                // disagree on that. updateRange is upload bookkeeping rather
                // than geometry content — the const_cast is because
                // GeomRefreshOp holds the geometry by const pointer (nothing
                // else in this path mutates it), not because this is unsafe.
                const_cast<FloatBufferAttribute*>(posAttr)->updateRange.count = -1;

                // A STATIC geometry whose attributes were edited (needsUpdate)
                // also lands here, and its buffers may hold PACKED data — the
                // re-upload must match the buffer's format or it overflows the
                // smaller packed allocation.
                const auto [nFirst, nCount] = vertexRangeOf(*nrmAttr, 3u);
                if (rec.packedMask & 1u) {
                    if (nCount > 0) {
                        const auto& nrm = nrmAttr->array();
                        std::vector<uint32_t> packed(nCount);
                        for (uint32_t v = 0; v < nCount; ++v) {
                            const size_t s = (static_cast<size_t>(nFirst) + v) * 3u;
                            const auto [ox, oy] = octEncode(nrm[s + 0], nrm[s + 1], nrm[s + 2]);
                            packed[v] = packSnorm2x16(ox, oy);
                        }
                        uploadHostVisible(ctx->allocator(), rec.normal, packed.data(),
                                          packed.size() * sizeof(uint32_t),
                                          static_cast<VkDeviceSize>(nFirst) * sizeof(uint32_t));
                    }
                } else if (nCount > 0) {
                    uploadHostVisible(ctx->allocator(), rec.normal,
                                      nrmAttr->array().data() + static_cast<size_t>(nFirst) * 3u,
                                      static_cast<VkDeviceSize>(nCount) * 3u * sizeof(float),
                                      static_cast<VkDeviceSize>(nFirst) * 3u * sizeof(float));
                }
                const_cast<FloatBufferAttribute*>(nrmAttr)->updateRange.count = -1;

                // uv/index keep the whole-array re-upload: they are static for
                // every in-place-edited geometry threepp ships (positions and
                // normals are what a dynamic surface rewrites), so there is no
                // measured win to justify the extra range plumbing here.
                const uint32_t vtxCount = static_cast<uint32_t>(posAttr->count());
                if (auto* uvAttr = geom.getAttribute<float>("uv");
                    uvAttr && rec.uv.handle != VK_NULL_HANDLE) {
                    if (rec.packedMask & 2u) {
                        const auto& uv = uvAttr->array();
                        std::vector<uint32_t> packed(vtxCount);
                        for (uint32_t v = 0; v < vtxCount; ++v) {
                            packed[v] = packUnorm2x16(uv[v * 2 + 0], uv[v * 2 + 1]);
                        }
                        uploadHostVisible(ctx->allocator(), rec.uv, packed.data(),
                                          packed.size() * sizeof(uint32_t));
                    } else {
                        uploadHostVisible(ctx->allocator(), rec.uv, uvAttr->array().data(),
                                          uvAttr->array().size() * sizeof(float));
                    }
                }

                if (auto* idxAttr = geom.getIndex();
                    idxAttr && rec.index.handle != VK_NULL_HANDLE) {
                    if (rec.packedMask & 8u) {
                        const auto& src = idxAttr->array();
                        std::vector<uint16_t> idx16(src.size() + (src.size() & 1u), 0);
                        for (size_t k2 = 0; k2 < src.size(); ++k2) {
                            idx16[k2] = static_cast<uint16_t>(src[k2]);
                        }
                        uploadHostVisible(ctx->allocator(), rec.index, idx16.data(),
                                          idx16.size() * sizeof(uint16_t));
                    } else {
                        uploadHostVisible(ctx->allocator(), rec.index, idxAttr->array().data(),
                                          idxAttr->array().size() * sizeof(unsigned int));
                    }
                }
            }

            // Phase D — refit each BLAS. The build-info structs need to outlive
            // the cmdBuildAccelerationStructures call until the GPU executes
            // them, so we hold them in vectors that live until after the submit-
            // wait below. Each op uses its own (persistent) scratch buffer so
            // concurrent builds within the cmdbuf don't alias.
            const uint32_t N = static_cast<uint32_t>(liveOps.size());
            std::vector<VkAccelerationStructureGeometryTrianglesDataKHR> triDatas(N);
            std::vector<VkAccelerationStructureGeometryKHR>              blasGeoms(N);
            std::vector<VkAccelerationStructureBuildGeometryInfoKHR>     blasBuilds(N);
            std::vector<VkAccelerationStructureBuildRangeInfoKHR>        ranges(N);
            std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePtrs(N);

            for (uint32_t kk = 0; kk < N; ++kk) {
                const size_t k = liveOps[kk];
                const auto& geom = *ops[k].geom;
                auto& rec = *ops[k].rec;
                auto* posAttr = geom.getAttribute<float>("position");
                const auto* idxAttr = geom.getIndex();
                const bool indexed = idxAttr != nullptr;
                const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
                // Only [0, drawRange start+count) is built — same clamp and
                // same no-offset rationale as buildBlasFor.
                const DrawSpan span = drawSpanOf(
                        geom, indexed ? static_cast<uint32_t>(idxAttr->count()) : vertexCount);
                const uint32_t primitiveCount = (span.first + span.elems) / 3u;

                triDatas[kk].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                triDatas[kk].vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                triDatas[kk].vertexData.deviceAddress = rec.vertex.address;
                triDatas[kk].vertexStride = 3 * sizeof(float);
                triDatas[kk].maxVertex = vertexCount - 1;
                if (indexed) {
                    triDatas[kk].indexType = (rec.packedMask & 8u) ? VK_INDEX_TYPE_UINT16
                                                                   : VK_INDEX_TYPE_UINT32;
                    triDatas[kk].indexData.deviceAddress = rec.index.address;
                } else {
                    triDatas[kk].indexType = VK_INDEX_TYPE_NONE_KHR;
                }

                blasGeoms[kk].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
                blasGeoms[kk].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                blasGeoms[kk].geometry.triangles = triDatas[kk];
                blasGeoms[kk].flags = 0;

                // Periodic MODE_BUILD every kBlasFullRebuildInterval frames
                // keeps the BVH balanced under sustained vertex motion (soft
                // body collapse, particle swarms) where pure refits drift the
                // tree away from optimal. A count change (drawRange moved since
                // the AS was last built) also forces BUILD — MODE_UPDATE
                // against a different primitive count is invalid — as does a
                // flags-lineage change (an update must carry its source
                // build's flags; a record that just left the interop route
                // was last built with PREFER_FAST_BUILD).
                const VkBuildAccelerationStructureFlagsKHR wantFlags =
                        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
                const bool fullRebuild = primitiveCount != rec.blasBuiltPrims ||
                        wantFlags != rec.blasBuiltFlags ||
                        rec.blasRefitCounter >= BlasRecord::kBlasFullRebuildInterval;
                rec.blasRefitCounter = fullRebuild ? 0u : (rec.blasRefitCounter + 1u);
                rec.blasBuiltPrims = primitiveCount;
                rec.blasBuiltFlags = wantFlags;

                blasBuilds[kk].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
                blasBuilds[kk].type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                blasBuilds[kk].flags = wantFlags;
                blasBuilds[kk].mode = fullRebuild
                        ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
                        : VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
                blasBuilds[kk].geometryCount = 1;
                blasBuilds[kk].pGeometries = &blasGeoms[kk];
                blasBuilds[kk].srcAccelerationStructure = fullRebuild ? VK_NULL_HANDLE : rec.as;
                blasBuilds[kk].dstAccelerationStructure = rec.as;

                VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
                blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
                ctx->rt().getAccelerationStructureBuildSizes(
                        ctx->device(),
                        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                        &blasBuilds[kk], &primitiveCount, &blasSizes);

                // Persistent scratch: size to max(buildScratch, updateScratch)
                // on first use (buildScratch is always >= updateScratch per
                // spec), then reuse across frames. Reallocate only if a future
                // topology change grew the required size — current refresh
                // path is fixed-topology (caller gates on counts).
                const VkDeviceSize neededScratch = blasSizes.buildScratchSize;
                if (rec.blasScratch.handle == VK_NULL_HANDLE || rec.blasScratchSize < neededScratch) {
                    destroyBuffer(ctx->allocator(), rec.blasScratch);
                    rec.blasScratch = createAsScratchBuffer(
                            ctx->allocator(), ctx->device(), neededScratch);
                    rec.blasScratchSize = neededScratch;
                }
                blasBuilds[kk].scratchData.deviceAddress = rec.blasScratch.address;

                ranges[kk].primitiveCount = primitiveCount;
                rangePtrs[kk] = &ranges[kk];

                rec.geomVersion = geomVersionOf(geom);
            }

            VkCommandBuffer cb = beginOneShot();
            ctx->rt().cmdBuildAccelerationStructures(cb, N, blasBuilds.data(), rangePtrs.data());
            endAndSubmitOneShot(cb, "refreshGeomBlasBatch (BLAS)");
        }

void VulkanRenderer::Impl::recordDynamicGeomRefits(VkCommandBuffer cb) {
            if (pendingDynamicGeomRefits_.empty() && pendingDynamicPrevResyncs_.empty()) return;
            THREEPP_CPUPROF("frame.dynGeomRefit");

            // Scratch that outlives the call. Every one of these used to be a
            // fresh vector per frame, and packedNormals a fresh one per RECORD:
            // a 200k-vertex graduated deformer paid a demand-zeroed 800 KB
            // malloc + free every frame for an array the very next line
            // overwrote element by element. Holding the storage across frames
            // costs one high-water allocation and nothing after that.
            //
            // Per-thread rather than per-renderer because recording is
            // single-threaded per renderer and none of these carries anything
            // across a call — each is resized and refilled before it is read —
            // so two renderers sharing a thread cannot observe each other. The
            // build-info arrays take clear() + resize(), never a bare resize:
            // Phase 6 leaves several fields of
            // VkAccelerationStructureBuildGeometryInfoKHR unassigned and relies
            // on the value-initialisation a fresh vector used to hand it.
            static thread_local std::vector<size_t>   liveOps;
            static thread_local std::vector<uint32_t> packedNormals;
            static thread_local std::vector<VkAccelerationStructureGeometryTrianglesDataKHR> triDatas;
            static thread_local std::vector<VkAccelerationStructureGeometryKHR>              blasGeoms;
            static thread_local std::vector<VkAccelerationStructureBuildGeometryInfoKHR>     blasBuilds;
            static thread_local std::vector<VkAccelerationStructureBuildRangeInfoKHR>        ranges;
            static thread_local std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePtrs;

            // Phase 0 — same finite-position contract as refreshGeomBlasBatch's
            // Phase A: a NaN reaching the BLAS build is VK_ERROR_DEVICE_LOST on
            // NVIDIA. A bad op is dropped (its geomVersion stays stale, so a
            // later structural rebuild picks it up) and last frame's geometry
            // stays on screen.
            //
            // INTEROP RECORDS SKIP THIS SCAN. The array it walks is the host
            // `position` attribute, and under interop a foreign device producer
            // owns the positions — the host copy is whatever was there when the
            // geometry was built, i.e. stale or empty. Scanning it would prove
            // nothing about the bytes the BLAS is going to read. The guard is not
            // dropped, it MOVES to the GPU: the sanitize dispatch below runs over
            // the exported buffer, which is the memory that actually reaches the
            // build. (Same reason interop records never fail the attribute test:
            // a producer-owned geometry need not carry host arrays at all.)
            liveOps.clear();
            liveOps.reserve(pendingDynamicGeomRefits_.size());
            for (size_t k = 0; k < pendingDynamicGeomRefits_.size(); ++k) {
                const auto& geom = *pendingDynamicGeomRefits_[k].geom;
                if (pendingDynamicGeomRefits_[k].rec->interop) {
                    liveOps.push_back(k);
                    continue;
                }
                auto* posAttr = geom.getAttribute<float>("position");
                if (!posAttr || !geom.getAttribute<float>("normal")) continue;
                bool ok = true;
                const auto& p = posAttr->array();
                for (size_t i = 0; i < p.size(); ++i) {
                    if (!std::isfinite(p[i])) {
                        std::cerr << "[VulkanRenderer] recordDynamicGeomRefits: skipping geom - "
                                  << "position[" << i << "] is non-finite (" << p[i] << ")\n";
                        ok = false;
                        break;
                    }
                }
                if (ok) liveOps.push_back(k);
            }

            // Phase 1 — host: pack this frame's positions/normals into slot
            // `currentFrame` of each record's staging ring. We are RECORDING,
            // i.e. past the inFlight[currentFrame] fence wait, so this slot's
            // previous consumer (frame currentFrame − kFramesInFlight) has
            // fully executed its copies — the memcpy cannot race the GPU.
            // That fence is the entire reason the staging ring exists: writing
            // rec.vertex from the host directly is exactly the mid-flight
            // mutation the drain-based path pays vkDeviceWaitIdle to avoid.
            //
            // ── AND THE ONE BRANCH THIS WHOLE FEATURE TURNS ON ──────────────
            // An interop record must NOT take the pack. Doing so memcpys the
            // stale host attribute arrays into the staging slot and Phase 4 then
            // copies them straight over rec.vertex/rec.normal — silently
            // clobbering everything the foreign producer just wrote, with no
            // error and no visual clue beyond "the mesh never moves". That is
            // exactly the trap a caller falls into by reaching for needsUpdate()
            // to make the renderer notice a CUDA write.
            //
            // The replacement is the producer's own callback, invoked HERE:
            // post-fence (recordCommandBuffer runs past this slot's fence),
            // pre-record of the copies that consume it, pre-submit, and never
            // inside a render pass. It is contractually SYNCHRONOUS, so when it
            // returns the exported allocation holds this frame's data and the
            // command buffer that will copy it has not been submitted. That host
            // ordering is the entire synchronisation story — the same one the tet
            // path (refreshTetBlas) and ParticleField F6 already run on, and the
            // same standing decision not to introduce a shared timeline semaphore.
            bool anySanitize = false;
            for (size_t k : liveOps) {
                const auto& geom = *pendingDynamicGeomRefits_[k].geom;
                auto& rec = *pendingDynamicGeomRefits_[k].rec;
                if (rec.interop) {
                    if (rec.externalCopy) rec.externalCopy();
                    if (rec.interopValidate && vertexSanitize_) anySanitize = true;
                    // Stamp the version even though nothing on the host moved it.
                    // A structural rebuild elsewhere in the frame compares
                    // geomVersionOf() against this field and DESTROYS the record
                    // on a mismatch (VulkanCoreScene's ensureSceneBuilt), taking
                    // the exported allocation — which a foreign API has already
                    // imported — with it. Keeping the stamp current means a
                    // caller who also calls needsUpdate() out of habit gets a
                    // no-op instead of a teardown.
                    rec.geomVersion = geomVersionOf(geom);
                    continue;
                }
                auto* posAttr = geom.getAttribute<float>("position");
                auto* nrmAttr = geom.getAttribute<float>("normal");
                const VkDeviceSize slotOff = rec.dynStagingSlotBytes * currentFrame;
                const VkDeviceSize posBytes = rec.vbBytes;
                uploadHostVisible(ctx->allocator(), rec.dynStaging,
                                  posAttr->array().data(), posBytes, slotOff);
                if (rec.packedMask & 1u) {
                    const auto& nrm = nrmAttr->array();
                    // Bare resize, not clear() + resize(): the loop assigns
                    // every element of [0, vertexCount), so the zero-fill a
                    // fresh vector performed was dead stores. size() is still
                    // exactly vertexCount, which is what the upload measures.
                    packedNormals.resize(rec.vertexCount);
                    for (uint32_t v = 0; v < rec.vertexCount; ++v) {
                        const auto [ox, oy] = octEncode(nrm[v * 3u + 0], nrm[v * 3u + 1], nrm[v * 3u + 2]);
                        packedNormals[v] = packSnorm2x16(ox, oy);
                    }
                    uploadHostVisible(ctx->allocator(), rec.dynStaging, packedNormals.data(),
                                      packedNormals.size() * sizeof(uint32_t), slotOff + posBytes);
                } else {
                    uploadHostVisible(ctx->allocator(), rec.dynStaging, nrmAttr->array().data(),
                                      nrmAttr->array().size() * sizeof(float), slotOff + posBytes);
                }
                // The whole array travels every frame on this path — consume
                // updateRange exactly like the host-write path does, so "no
                // range set later" keeps meaning "all of it" on both routes.
                const_cast<FloatBufferAttribute*>(posAttr)->updateRange.count = -1;
                const_cast<FloatBufferAttribute*>(nrmAttr)->updateRange.count = -1;
                // Version bookkeeping here, not at enqueue: if a topology
                // change elsewhere aborts to the structural rebuild after the
                // enqueue, the stale version makes that rebuild re-admit this
                // geometry from its current CPU data instead of trusting a
                // buffer the cleared pending list never filled.
                rec.geomVersion = geomVersionOf(geom);
            }

            const bool anyRefit = !liveOps.empty();
            if (!anyRefit && pendingDynamicPrevResyncs_.empty()) {
                pendingDynamicGeomRefits_.clear();
                return;
            }

            // Everything from here down is RECORDED work, so it is what the
            // bracket should cover — the host pack above is CPU time and belongs
            // to THREEPP_CPUPROF("frame.dynGeomRefit"), not to a GPU column.
            gpuTimings_->begin(cb, vulkan::TP_DynGeomRefit, currentFrame);

            // NO cross-frame WAR fence here, on purpose — and it was not an
            // oversight the first time either. The prior frame may still be
            // reading vertex/normal/prevVertex when these copies execute; a
            // barrier wide enough to cover every reader (deferred_shade's
            // ray-query fetches are COMPUTE, so the mask would have to fence
            // compute) also fences the previous frame's entire post chain —
            // measured +6 ms/frame, handing back everything the drain removal
            // bought. The skinned/tet/displaced/grass deformers have always
            // rewritten their BLAS vertex buffers here under exactly this
            // exposure: the frame model (present-block at frame end, deforms
            // recorded first) keeps the window closed in practice, and this
            // path deliberately matches their contract rather than inventing
            // a stricter one.

            // Live vertex rows under BufferGeometry::drawRange, consumed by the
            // interop sanitize/snapshot/copies below so a mostly-empty capacity
            // buffer stops costing full-capacity bandwidth every frame.
            // Non-indexed only: an index can reference ANY vertex, so indexed
            // interop keeps the full-capacity copies.
            const auto interopLiveVerts = [](const BufferGeometry& g,
                                             const BlasRecord& r) -> uint32_t {
                if (r.indexCount != 0u) return r.vertexCount;
                const DrawSpan s = drawSpanOf(g, r.vertexCount);
                return std::min(r.vertexCount, s.first + s.elems);
            };

            // Phase 2 — GPU sanitize (interop only). Runs over the EXPORTED
            // positions, before anything copies them into rec.vertex, so the
            // buffer the BLAS build reads is finite by construction. This is the
            // stand-in for the Phase 0 host scan these records skip; see
            // shaders/vertex_sanitize.comp for the repair rule. Only the live
            // rows are dispatched: the tail is never copied out of the export,
            // so nothing downstream can read whatever garbage it holds.
            if (anySanitize) {
                for (size_t k : liveOps) {
                    auto& rec = *pendingDynamicGeomRefits_[k].rec;
                    if (!rec.interop || !rec.interopValidate) continue;
                    vertexSanitize_->recordDispatch(
                            cb, rec.sanitizeDS,
                            interopLiveVerts(*pendingDynamicGeomRefits_[k].geom, rec));
                }
            }

            // Phase 3 — motion snapshot: vertex → prevVertex. For refit ops
            // that is frame N−1's positions (the change frame's motion base);
            // for resync-only ops vertex already holds the settled positions,
            // so prevVertex == vertex and motion collapses back to zero — the
            // frame-cb twin of the prevVertexResyncPending pass.
            for (size_t k : liveOps) {
                auto& rec = *pendingDynamicGeomRefits_[k].rec;
                if (rec.prevVertex.handle == VK_NULL_HANDLE) continue;
                // World-static interop (re-triangulated soups): the selection
                // sites never read prevVertex for this record, so the snapshot
                // is dead work — same skip as the ocean's adaptive warp.
                if (rec.interopWorldStatic) continue;
                VkBufferCopy region{};
                region.size = rec.vbBytes;
                if (rec.interop) {
                    // Live rows only. A row revealed by a GROWING drawRange gets
                    // one frame of stale prevVertex — a triangle that just
                    // appeared has no valid history either way (the full-size
                    // copy would hand it last frame's unrelated positions).
                    region.size = VkDeviceSize(interopLiveVerts(
                                          *pendingDynamicGeomRefits_[k].geom, rec)) *
                                  3u * sizeof(float);
                }
                vkCmdCopyBuffer(cb, rec.vertex.handle, rec.prevVertex.handle, 1, &region);
            }
            for (auto* rec : pendingDynamicPrevResyncs_) {
                if (rec->prevVertex.handle == VK_NULL_HANDLE) continue;
                VkBufferCopy region{};
                region.size = rec->vbBytes;
                vkCmdCopyBuffer(cb, rec->vertex.handle, rec->prevVertex.handle, 1, &region);
            }

            if (anyRefit) {
                // Phase 4 — the snapshot READ vertex; the staging copy is about
                // to WRITE it. Another execution-only WAR fence. When a sanitize
                // dispatch ran it also carries the real compute-write → transfer-
                // read dependency for posExt (hence the non-zero srcAccessMask on
                // that path: the WAR half needs no access scope, the sanitize
                // half does).
                {
                    VkMemoryBarrier2 mb{};
                    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    mb.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
                    mb.srcAccessMask = 0;
                    mb.dstStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
                    mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    if (anySanitize) {
                        mb.srcStageMask  |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                        mb.srcAccessMask |= VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                        mb.dstAccessMask |= VK_ACCESS_2_TRANSFER_READ_BIT;
                    }
                    VkDependencyInfo dep{};
                    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep.memoryBarrierCount = 1;
                    dep.pMemoryBarriers    = &mb;
                    vkCmdPipelineBarrier2(cb, &dep);
                }
                for (size_t k : liveOps) {
                    auto& rec = *pendingDynamicGeomRefits_[k].rec;
                    if (rec.interop) {
                        // Same two copies, different source: the exported
                        // allocations the producer just filled, instead of this
                        // frame's staging slot. Sizes come from the BLAS buffers
                        // rather than from posExt.size — the export is the
                        // ALLOCATION size, which the driver rounds up past the
                        // destination's capacity (see ExternalBuffer::size) —
                        // and are then trimmed to the drawRange's live rows:
                        // nothing reads past them (the raster clamps its draw,
                        // the BLAS builds only the live span), so the tail of an
                        // over-allocated capacity buffer costs no bandwidth.
                        const uint32_t live = interopLiveVerts(
                                *pendingDynamicGeomRefits_[k].geom, rec);
                        VkBufferCopy pr{};
                        pr.size = std::min<VkDeviceSize>(
                                rec.vertex.size, VkDeviceSize(live) * 3u * sizeof(float));
                        vkCmdCopyBuffer(cb, rec.posExt.handle, rec.vertex.handle, 1, &pr);
                        VkBufferCopy nr{};
                        nr.size = std::min<VkDeviceSize>(
                                rec.normal.size,
                                rec.vertexCount ? rec.normal.size / rec.vertexCount *
                                                          VkDeviceSize(live)
                                                : rec.normal.size);
                        vkCmdCopyBuffer(cb, rec.nrmExt.handle, rec.normal.handle, 1, &nr);
                        continue;
                    }
                    const VkDeviceSize slotOff = rec.dynStagingSlotBytes * currentFrame;
                    VkBufferCopy pr{};
                    pr.srcOffset = slotOff;
                    pr.size      = rec.vbBytes;
                    vkCmdCopyBuffer(cb, rec.dynStaging.handle, rec.vertex.handle, 1, &pr);
                    VkBufferCopy nr{};
                    nr.srcOffset = slotOff + rec.vbBytes;
                    nr.size      = rec.dynStagingSlotBytes - rec.vbBytes;
                    vkCmdCopyBuffer(cb, rec.dynStaging.handle, rec.normal.handle, 1, &nr);
                }
            }

            // Phase 5 — publish the copies: the BLAS refit reads vertex as
            // build input, the raster prepass reads vertex/normal/prevVertex
            // as vertex attributes, chit/probe fetch them as storage.
            {
                VkMemoryBarrier2 mb{};
                mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                mb.srcStageMask  = VK_PIPELINE_STAGE_2_COPY_BIT;
                mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                mb.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                   VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
                                   VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR |
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                   VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                VkDependencyInfo dep{};
                dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.memoryBarrierCount = 1;
                dep.pMemoryBarriers    = &mb;
                vkCmdPipelineBarrier2(cb, &dep);
            }

            if (anyRefit) {
                // Phase 6 — batched BLAS refit, the recorded twin of
                // refreshGeomBlasBatch's Phase D. The build-info structs are
                // consumed by the record below and never outlive it, so the
                // frame scratch declared at the top of the function carries
                // them; clear() before resize() so every element is
                // value-initialised exactly as a freshly constructed vector
                // would be.
                const uint32_t N = static_cast<uint32_t>(liveOps.size());
                triDatas.clear();
                triDatas.resize(N);
                blasGeoms.clear();
                blasGeoms.resize(N);
                blasBuilds.clear();
                blasBuilds.resize(N);
                ranges.clear();
                ranges.resize(N);
                rangePtrs.clear();
                rangePtrs.resize(N);
                for (uint32_t kk = 0; kk < N; ++kk) {
                    auto& rec = *pendingDynamicGeomRefits_[liveOps[kk]].rec;
                    const bool indexed = rec.indexCount != 0u;
                    // Only [0, drawRange start+count) is built — same clamp
                    // and same no-offset rationale as buildBlasFor.
                    const DrawSpan span = drawSpanOf(
                            *pendingDynamicGeomRefits_[liveOps[kk]].geom,
                            indexed ? rec.indexCount : rec.vertexCount);
                    const uint32_t primitiveCount = (span.first + span.elems) / 3u;

                    triDatas[kk].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                    triDatas[kk].vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                    triDatas[kk].vertexData.deviceAddress = rec.vertex.address;
                    triDatas[kk].vertexStride = 3 * sizeof(float);
                    triDatas[kk].maxVertex = rec.vertexCount - 1;
                    if (indexed) {
                        triDatas[kk].indexType = (rec.packedMask & 8u) ? VK_INDEX_TYPE_UINT16
                                                                       : VK_INDEX_TYPE_UINT32;
                        triDatas[kk].indexData.deviceAddress = rec.index.address;
                    } else {
                        triDatas[kk].indexType = VK_INDEX_TYPE_NONE_KHR;
                    }

                    blasGeoms[kk].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
                    blasGeoms[kk].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
                    blasGeoms[kk].geometry.triangles = triDatas[kk];
                    blasGeoms[kk].flags = 0;

                    // Interop records rebuild nearly every frame (a producer-
                    // owned surface whose triangle count moves forces
                    // MODE_BUILD), so they take PREFER_FAST_BUILD — the
                    // standard choice for per-frame rebuilt geometry.
                    // CPU-graduated records keep FAST_TRACE: their fixed
                    // topology refits via MODE_UPDATE 63 frames of 64.
                    //
                    // FAST_BUILD is only legal because the storage was sized
                    // for it: the two lineages are different BVH formats with
                    // different footprints, and building the one the storage
                    // was not sized for overruns the structure — VUID-
                    // vkCmdBuildAccelerationStructuresKHR-pInfos-10126 with
                    // the layers on, heap corruption surfacing as
                    // VK_ERROR_DEVICE_LOST without. buildBlasFor sizes an
                    // interop-marked record's storage for the max of BOTH
                    // lineages' size queries (storageFitsFastBuild), and
                    // enableVertexInterop rebuilds any record armed before
                    // that sizing existed, so the fits-check below passes for
                    // every armed record.
                    //
                    // It is still made against a LIVE query rather than
                    // trusted from the flag. The sizing guarantee rests on
                    // the driver answering the same size query with the same
                    // number at creation and at refit — which the spec
                    // implies but which the one observed overrun (345856
                    // required vs a 329600 FAST_TRACE allocation for the same
                    // 5120-triangle mesh, 2026-08-27) could never be
                    // reproduced against with byte-identical queries. So the
                    // denial path stays: it turns the unexplained case into a
                    // benign FAST_TRACE frame, and reports itself once per
                    // record instead of corrupting the heap silently. The
                    // query is cheap — a driver-side estimate, no allocation.
                    bool fastBuildFits = false;
                    if (rec.interop) {
                        VkAccelerationStructureBuildGeometryInfoKHR q{};
                        q.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
                        q.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                        q.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR |
                                  VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
                        q.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
                        q.geometryCount = 1;
                        q.pGeometries = &blasGeoms[kk];
                        VkAccelerationStructureBuildSizesInfoKHR s{};
                        s.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
                        ctx->rt().getAccelerationStructureBuildSizes(
                                ctx->device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                &q, &primitiveCount, &s);
                        fastBuildFits = s.accelerationStructureSize <= rec.storage.size;
                        if (!fastBuildFits && !rec.fastBuildDeniedWarned) {
                            rec.fastBuildDeniedWarned = true;
                            std::cerr << "[VulkanRenderer] interop BLAS refit: FAST_BUILD needs "
                                      << s.accelerationStructureSize << " B but storage is "
                                      << rec.storage.size << " B ("
                                      << (rec.storageFitsFastBuild
                                          ? "max-of-lineages sized at creation - the driver's "
                                            "answer moved between creation and refit"
                                          : "storage predates the max-of-lineages sizing")
                                      << ") - this record keeps the FAST_TRACE lineage.\n";
                        }
                    }
                    const VkBuildAccelerationStructureFlagsKHR wantFlags =
                            ((rec.interop && fastBuildFits)
                            ? VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR
                            : VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR) |
                            VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
                    // A count change forces MODE_BUILD: refitting against a
                    // different primitive count than the AS was built with is
                    // invalid. For a marching-cubes surface whose triangle
                    // count moves every frame this makes most frames full
                    // builds — of the LIVE count, which is cheaper than
                    // refitting a padded capacity worth of degenerates. A
                    // flags-lineage change (a record entering or leaving the
                    // interop route) forces BUILD for the same
                    // update-must-match-its-source reason.
                    const bool fullRebuild = primitiveCount != rec.blasBuiltPrims ||
                            wantFlags != rec.blasBuiltFlags ||
                            rec.blasRefitCounter >= BlasRecord::kBlasFullRebuildInterval;
                    rec.blasRefitCounter = fullRebuild ? 0u : (rec.blasRefitCounter + 1u);
                    rec.blasBuiltPrims = primitiveCount;
                    rec.blasBuiltFlags = wantFlags;
                    ++dynGeomStats_.refitsRecorded;
                    if (fullRebuild) ++dynGeomStats_.fullRebuilds;

                    blasBuilds[kk].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
                    blasBuilds[kk].type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                    blasBuilds[kk].flags = wantFlags;
                    blasBuilds[kk].mode = fullRebuild
                            ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
                            : VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
                    blasBuilds[kk].geometryCount = 1;
                    blasBuilds[kk].pGeometries = &blasGeoms[kk];
                    blasBuilds[kk].srcAccelerationStructure = fullRebuild ? VK_NULL_HANDLE : rec.as;
                    blasBuilds[kk].dstAccelerationStructure = rec.as;

                    VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
                    blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
                    ctx->rt().getAccelerationStructureBuildSizes(
                            ctx->device(),
                            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                            &blasBuilds[kk], &primitiveCount, &blasSizes);
                    // Scratch grows via retire(), never destroyBuffer: a still-
                    // in-flight frame's refit may be reading the old one. Cold
                    // path — graduation follows ≥3 drained refreshes, which
                    // already sized it.
                    if (rec.blasScratch.handle == VK_NULL_HANDLE ||
                        rec.blasScratchSize < blasSizes.buildScratchSize) {
                        retire(std::move(rec.blasScratch));
                        rec.blasScratch = createAsScratchBuffer(
                                ctx->allocator(), ctx->device(), blasSizes.buildScratchSize);
                        rec.blasScratchSize = blasSizes.buildScratchSize;
                    }
                    blasBuilds[kk].scratchData.deviceAddress = rec.blasScratch.address;

                    ranges[kk].primitiveCount = primitiveCount;
                    rangePtrs[kk] = &ranges[kk];
                }
                ctx->rt().cmdBuildAccelerationStructures(cb, N, blasBuilds.data(), rangePtrs.data());

                // Phase 7 — AS write → AS read: the TLAS refit recorded later
                // in recordDeformAndTlas consumes these BLASes this same frame.
                {
                    VkMemoryBarrier2 mb{};
                    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                    mb.srcStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
                    mb.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
                    mb.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                       VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                    mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
                    VkDependencyInfo dep{};
                    dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                    dep.memoryBarrierCount = 1;
                    dep.pMemoryBarriers    = &mb;
                    vkCmdPipelineBarrier2(cb, &dep);
                }
            }

            gpuTimings_->end(cb, vulkan::TP_DynGeomRefit, currentFrame);

            pendingDynamicGeomRefits_.clear();
            pendingDynamicPrevResyncs_.clear();
        }

void VulkanRenderer::Impl::refreshMorphedBlas(Mesh& mesh, MorphedMeshState& st) {
            cpuMorphBlend(mesh, st.blendedPositions, st.blendedNormals);
            if (st.blendedPositions.empty() || !st.blas) return;

            uploadHostVisible(ctx->allocator(), st.blas->vertex, st.blendedPositions.data(),
                              st.blendedPositions.size() * sizeof(float));
            uploadHostVisible(ctx->allocator(), st.blas->normal, st.blendedNormals.data(),
                              st.blendedNormals.size() * sizeof(float));

            auto* posAttr = mesh.geometry()->getAttribute<float>("position");
            auto* idxAttr = mesh.geometry()->getIndex();
            const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
            const bool indexed = idxAttr != nullptr;
            const uint32_t primitiveCount = indexed
                    ? static_cast<uint32_t>(idxAttr->count() / 3)
                    : vertexCount / 3;

            VkAccelerationStructureGeometryTrianglesDataKHR triData{};
            triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triData.vertexData.deviceAddress = st.blas->vertex.address;
            triData.vertexStride = 3 * sizeof(float);
            triData.maxVertex = vertexCount - 1;
            if (indexed) {
                triData.indexType = VK_INDEX_TYPE_UINT32;
                triData.indexData.deviceAddress = st.blas->index.address;
            } else {
                triData.indexType = VK_INDEX_TYPE_NONE_KHR;
            }

            VkAccelerationStructureGeometryKHR blasGeom{};
            blasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            blasGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            blasGeom.geometry.triangles = triData;
            blasGeom.flags = 0;

            VkAccelerationStructureBuildGeometryInfoKHR blasBuild{};
            blasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            blasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            blasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            blasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            blasBuild.geometryCount = 1;
            blasBuild.pGeometries = &blasGeom;
            blasBuild.dstAccelerationStructure = st.blas->as;

            VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
            blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(),
                    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &blasBuild, &primitiveCount, &blasSizes);

            Buffer scratch = createAsScratchBuffer(ctx->allocator(), ctx->device(), blasSizes.buildScratchSize);
            blasBuild.scratchData.deviceAddress = scratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = primitiveCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

            VkCommandBuffer cb = beginOneShot();
            // Motion-vector prev-pose: buildBlasFor seeded prevVertex with the
            // REST (unmorphed) positions, but we just overwrote vertex with the
            // BLENDED positions. Left as-is, the gbuffer VS / chit read
            // prevPos = rest while pos = blended → a permanent rest→blend motion
            // vector on an otherwise static morph, so TAA reprojects history
            // from the wrong pixel every frame and the surface shakes. Sync
            // prevVertex := vertex here so the morph reports zero per-vertex
            // motion (transform/camera motion still flows through motionMat),
            // exactly like the static-mesh prevPosAddr == posAddr fallback.
            if (st.blas->prevVertex.handle != VK_NULL_HANDLE) {
                VkBufferCopy region{};
                region.size = st.blendedPositions.size() * sizeof(float);
                vkCmdCopyBuffer(cb, st.blas->vertex.handle,
                                st.blas->prevVertex.handle, 1, &region);
            }
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &blasBuild, &pRange);
            endAndSubmitOneShot(cb);
            destroyBuffer(ctx->allocator(), scratch);

            auto* morphObj = mesh.as<ObjectWithMorphTargetInfluences>();
            if (morphObj) st.prevInfluences = morphObj->morphTargetInfluences();
        }

void VulkanRenderer::Impl::recordDisplacedDeform(VkCommandBuffer cb, DisplacedMesh& dm, DisplacedMeshState& st, float elapsedSeconds, bool timed) {

            // (0) Live sea state. The Phillips h0 pass is normally one-shot,
            // but windSpeed/windTheta/fetch are plain Params fields — when
            // they drift from what the cascades were baked with, rewrite each
            // cascade's spectrum params and drop the phillipsRecorded latch so
            // the loop below re-dispatches h0 this frame. The persistent per-
            // cascade noise images keep successive h0 fields phase-correlated,
            // so the sea MORPHS into the new state over a few swell periods
            // instead of popping. (Mapped-UBO rewrite mid-flight follows the
            // same convention as DynamicSpectrum's per-frame time update.)
            if (dm.params.windSpeed != st.appliedWindSpeed ||
                dm.params.windTheta != st.appliedWindTheta ||
                dm.params.fetch     != st.appliedFetch) {
                for (uint32_t i = 0; i < 3; ++i) {
                    if (!(st.cascadeMask & (1u << i))) continue;
                    // Cascade 1's sample domain is rotated; keep its world
                    // propagation on windTheta (same compensation as setup).
                    const float theta = dm.params.windTheta +
                            (i == 1 ? kOceanCascade1RotTheta : 0.f);
                    st.cascades[i].phillips->updateSeaState(theta, dm.params.windSpeed,
                                                            dm.params.fetch);
                    st.cascades[i].phillipsRecorded = false;
                }
                st.appliedWindSpeed = dm.params.windSpeed;
                st.appliedWindTheta = dm.params.windTheta;
                st.appliedFetch     = dm.params.fetch;
            }

            // (1)..(3) Run each enabled cascade's FFT chain in turn. Phillips
            // is one-shot per cascade. DynamicSpectrum re-runs each frame.
            // IFFT calls are sequential on the same queue so they can share
            // the single scratch image. Cascades dispatch back-to-back; the
            // Vulkan command buffer recording order plus the IFFT's internal
            // image-layout barriers serialize the work correctly.
            if (timed) gpuTimings_->begin(cb, vulkan::TP_OceanFFT, currentFrame);
            // WAR hazard across frames: the PREVIOUS frame's height-field
            // readback copy reads each cascade's `ht` image at the TRANSFER
            // stage, and the only barriers in the chain below are
            // COMPUTE→COMPUTE, so this frame's dynamic-spectrum dispatch was
            // free to overwrite `ht` while that copy was still in flight — the
            // copy then landed a mix of two frames' fields in its readback
            // slot. Invisible at real frame sizes (the copy is long done before
            // the next frame's compute starts); at the 128×96 frames the
            // regression tests render, adjacent frames overlap and
            // sample_height() returned the NEXT frame's (or a torn) height for
            // 1-2 frames in 40, run to run. One execution dependency closes it;
            // only recorded when a copy can exist (the sticky readback opt-in).
            if (dm.wantsHeightReadback) {
                VkMemoryBarrier mb{};
                mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                mb.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(cb,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, &mb, 0, nullptr, 0, nullptr);
            }
            for (uint32_t i = 0; i < 3; ++i) {
                if (!(st.cascadeMask & (1u << i))) continue;
                auto& c = st.cascades[i];
                if (!c.phillipsRecorded) {
                    c.phillips->recordCompute(cb);
                    c.phillipsRecorded = true;
                }
                c.dyn->recordCompute(cb, elapsedSeconds);
                water::OceanImage ht  = c.dyn->ht();
                water::OceanImage dsp = c.dyn->displacement();
                ht.currentLayout  = VK_IMAGE_LAYOUT_GENERAL;
                dsp.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
                // scratchA.currentLayout is deliberately NOT forced: it is
                // created UNDEFINED and recordApply's cmdTransitionToGeneral
                // relies on the tracked value to record the one first-use
                // UNDEFINED→GENERAL barrier (pre-setting GENERAL here skipped
                // that barrier and tripped submit-time layout validation on
                // the first FFT chain).
                c.ifft->recordApply(cb, ht,  st.scratchA);
                c.ifft->recordApply(cb, dsp, st.scratchA);

                // Copy the spatial-domain height image into THIS FRAME's slot
                // of the host-mapped readback ring for this cascade (slot =
                // currentFrame; DisplacedMeshState::heightReadback says why it
                // is a ring and which slot the mirror reads). On the one-shot
                // first build the buffer is filled by the time
                // endAndSubmitOneShot returns; on the per-frame path when
                // inFlight[currentFrame] next signals.
                // Gated on the sticky sampleHeight() opt-in — scenes that
                // never query CPU wave height skip the copies (and their
                // barriers) entirely and never allocate the ring.
                Buffer* rb = nullptr;
                if (dm.wantsHeightReadback) {
                    if (st.heightReadback[0][0].handle == VK_NULL_HANDLE) {
                        // First opted-in record: allocate the whole ring —
                        // every enabled cascade × every slot — so it is
                        // either complete or absent. Cascade 0 is always
                        // enabled, so its slot 0 is the "allocated" probe.
                        for (uint32_t c = 0; c < 3; ++c) {
                            const uint32_t dim = st.heightReadbackDim[c];
                            if (dim == 0) continue;
                            const VkDeviceSize bytes =
                                    VkDeviceSize(dim) * VkDeviceSize(dim) * 8u;
                            for (uint32_t s = 0; s < kFramesInFlight; ++s) {
                                auto& b = st.heightReadback[c][s];
                                b = createBuffer(
                                        ctx->allocator(), ctx->device(), bytes,
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                        VMA_MEMORY_USAGE_AUTO,
                                        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
                                ctx->setObjectName(b.handle,
                                        (std::string("ocean.heightReadback") + std::to_string(c) +
                                         "." + std::to_string(s)).c_str());
                            }
                        }
                    }
                    rb = &st.heightReadback[i][currentFrame];
                }
                if (rb && rb->handle != VK_NULL_HANDLE) {
                    st.heightReadbackWritten[currentFrame] = true;
                    VkImageMemoryBarrier imb{};
                    imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    imb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    imb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    imb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    imb.image = c.dyn->ht().image;
                    imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    vkCmdPipelineBarrier(cb,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, nullptr, 0, nullptr, 1, &imb);

                    VkBufferImageCopy bic{};
                    bic.bufferOffset = 0;
                    bic.bufferRowLength = 0;
                    bic.bufferImageHeight = 0;
                    bic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                    bic.imageOffset = {0, 0, 0};
                    bic.imageExtent = {st.heightReadbackDim[i], st.heightReadbackDim[i], 1};
                    vkCmdCopyImageToBuffer(cb, c.dyn->ht().image, VK_IMAGE_LAYOUT_GENERAL,
                                           rb->handle, 1, &bic);

                    VkBufferMemoryBarrier bmb{};
                    bmb.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                    bmb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    bmb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    bmb.buffer = rb->handle;
                    bmb.size   = VK_WHOLE_SIZE;
                    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    vkCmdPipelineBarrier(cb,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_HOST_BIT,
                            0, 0, nullptr, 1, &bmb, 0, nullptr);
                }
            }
            if (timed) gpuTimings_->end(cb, vulkan::TP_OceanFFT, currentFrame);

            // (3.5) Per-vertex motion: copy current vertex positions into
            // prevVertex BEFORE water_displace overwrites them. Hybrid-only
            // (raster gbuffer reads prevVertex at attribute 3).
            //
            // Skipped when the adaptive warp is active: the warp re-centres
            // the grid on the vessel every frame, so vertex i is a different
            // world point each frame and prevVertex (indexed by vertex id)
            // can't track a stable surface point. The motion-vector consumers
            // are pointed at the current vertex buffer instead (see the
            // prevVertexAddress / prevPosAddr selection sites), making the
            // ocean reproject as world-static — so the copy is dead work and
            // skipping it also saves a full vertex-buffer transfer per frame.
            const bool warpActive = dm.warp.halfRange > 0.0f;
            if (st.blas->prevVertex.handle != VK_NULL_HANDLE && !warpActive) {
                VkBufferCopy region{};
                region.size = VkDeviceSize(st.vertexCount) * 3u * sizeof(float);
                vkCmdCopyBuffer(cb, st.blas->vertex.handle,
                                st.blas->prevVertex.handle, 1, &region);
                // Barrier: copy WRITE → vertex shader READ (raster prepass)
                // and the immediate water_displace.comp WRITE on vertex.
                std::array<VkBufferMemoryBarrier, 1> bmb{};
                bmb[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                bmb[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                bmb[0].dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                                       VK_ACCESS_SHADER_READ_BIT;
                bmb[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                bmb[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                bmb[0].buffer = st.blas->prevVertex.handle;
                bmb[0].size   = VK_WHOLE_SIZE;
                vkCmdPipelineBarrier(cb,
                        VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 0, nullptr, uint32_t(bmb.size()), bmb.data(), 0, nullptr);
            }

            // (4) Dispatch water_displace.comp → writes positions + normals
            // into the BLAS vertex/normal buffers.

            // (4a) Foam-disturbance SSBO upload. Lazy-allocate the host-
            // mapped buffer on first use, then memcpy the (possibly empty)
            // user-supplied list each frame. Capped at kMaxFoamDisturbances;
            // extras are dropped silently. The fence wait at frame start
            // guarantees the GPU is done reading last frame's disturbances
            // before we overwrite, so no double-buffering is needed.
            const uint32_t kFoamDisturbStride = 16u;  // 4 floats per entry
            const uint32_t kFoamDisturbBytes  =
                    DisplacedMeshState::kMaxFoamDisturbances * kFoamDisturbStride;
            if (st.foamDisturbBuffer.handle == VK_NULL_HANDLE) {
                st.foamDisturbBuffer = createBuffer(
                        ctx->allocator(), ctx->device(), kFoamDisturbBytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                ctx->setObjectName(st.foamDisturbBuffer.handle, "ocean.foamDisturb");
            }
            // Foam dispatch interval: THREEPP_OCEAN_FOAM_INTERVAL=N runs the
            // foam_world accumulator every Nth frame (default 1 = every
            // frame). The decay push constant is dt-aware, so a skipped frame
            // just widens the next dispatch's decay step. Disturbance stamps
            // supplied on skipped frames are latched in st.foamDisturbCarry
            // and uploaded as a union with the next dispatch's list — the
            // shader combines stamps with max(), so re-stamping a source that
            // persists across frames is idempotent. The buffer's only consumer
            // is foam_world (water_displace no longer carries a disturbAddr at
            // all), so a deferred upload changes nothing else.
            static const uint32_t kFoamInterval = [] {
                const char* e = std::getenv("THREEPP_OCEAN_FOAM_INTERVAL");
                const long v = e ? std::atol(e) : 2L;
                return static_cast<uint32_t>(std::max(v, 1L));
            }();
            const bool runFoam = (st.foamTick++ % kFoamInterval) == 0u;

            static_assert(sizeof(DisplacedMeshState::FoamDisturbCarry) ==
                                  sizeof(DisplacedMesh::FoamDisturbance),
                          "carry layout must mirror FoamDisturbance (memcpy'd)");
            auto latchDisturbances = [&] {
                for (const auto& d : dm.foamDisturbances)
                    st.foamDisturbCarry.push_back(
                            {d.worldX, d.worldZ, d.radius, d.intensity});
                if (st.foamDisturbCarry.size() >
                    DisplacedMeshState::kMaxFoamDisturbances)
                    st.foamDisturbCarry.erase(
                            st.foamDisturbCarry.begin(),
                            st.foamDisturbCarry.end() -
                                    DisplacedMeshState::kMaxFoamDisturbances);
            };
            uint32_t disturbCount = 0;
            if (!runFoam) {
                latchDisturbances();
            } else if (!st.foamDisturbCarry.empty()) {
                latchDisturbances();// union: carried + this frame's list
                disturbCount = static_cast<uint32_t>(st.foamDisturbCarry.size());
                uploadHostVisible(ctx->allocator(), st.foamDisturbBuffer,
                                  st.foamDisturbCarry.data(),
                                  disturbCount * kFoamDisturbStride);
                st.foamDisturbCarry.clear();
            } else {
                disturbCount = static_cast<uint32_t>(std::min<size_t>(
                        dm.foamDisturbances.size(),
                        DisplacedMeshState::kMaxFoamDisturbances));
                if (disturbCount > 0u) {
                    uploadHostVisible(ctx->allocator(), st.foamDisturbBuffer,
                                      dm.foamDisturbances.data(),
                                      disturbCount * kFoamDisturbStride);
                }
            }

            // (4b) Wake-trail SSBO upload. Same pattern as disturbance buffer:
            // lazy-allocate, memcpy newest-first list, drop the tail beyond
            // kMaxWakeSamples. Each entry is 32 B = DisplacedMesh::WakeSample.
            const uint32_t kWakeSampleStride = 32u;
            const uint32_t kWakeTrailBytes   =
                    DisplacedMeshState::kMaxWakeSamples * kWakeSampleStride;
            if (st.wakeTrailBuffer.handle == VK_NULL_HANDLE) {
                st.wakeTrailBuffer = createBuffer(
                        ctx->allocator(), ctx->device(), kWakeTrailBytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                ctx->setObjectName(st.wakeTrailBuffer.handle, "ocean.wakeTrail");
            }
            const uint32_t wakeSampleCount = static_cast<uint32_t>(std::min<size_t>(
                    dm.wake.trail.size(),
                    DisplacedMeshState::kMaxWakeSamples));
            if (wakeSampleCount > 0u) {
                uploadHostVisible(ctx->allocator(), st.wakeTrailBuffer,
                                  dm.wake.trail.data(),
                                  wakeSampleCount * kWakeSampleStride);
            }

            vulkan::WaterDisplacePipeline::PushConstants pc{};
            pc.posOut       = st.blas->vertex.address;
            pc.normOut      = st.blas->normal.address;
            pc.vertexCount  = st.vertexCount;
            pc.gridDimX     = st.gridDimX;
            pc.gridDimZ     = st.gridDimZ;
            pc.planeSizeX   = st.planeSizeX;
            pc.planeSizeZ   = st.planeSizeZ;
            pc.tileSize0    = dm.params.tileSize0;
            pc.tileSize1    = dm.params.tileSize1;
            pc.tileSize2    = dm.params.tileSize2;
            pc.waveScale    = dm.params.waveScale;
            pc.choppiness   = dm.params.choppiness;
            pc.cascadeMask  = st.cascadeMask;
            pc.hullCenterX    = dm.hullExclusion.centerX;
            pc.hullCenterZ    = dm.hullExclusion.centerZ;
            pc.hullHalfLength = dm.hullExclusion.halfLength;
            pc.hullHalfBeam   = dm.hullExclusion.halfBeam;
            pc.hullSinYaw     = dm.hullExclusion.sinYaw;
            pc.hullCosYaw     = dm.hullExclusion.cosYaw;
            pc.forwardSpeed   = dm.wake.enabled ? dm.wake.forwardSpeed : 0.0f;
            pc.warpCenterX    = dm.warp.centerX;
            pc.warpCenterZ    = dm.warp.centerZ;
            pc.warpHalfRange  = dm.warp.halfRange;
            pc.warpCoefA      = dm.warp.coefA;
            pc.wakeTrailAddr  = st.wakeTrailBuffer.address;
            pc.wakeTrailCount = wakeSampleCount;
            // Waterline plane of the excluded patch (0/0/0 = ocean rest plane,
            // the historical behaviour) — see DisplacedMesh::HullExclusion.
            pc.hullCenterY    = dm.hullExclusion.centerY;
            pc.hullPitch      = dm.hullExclusion.pitch;
            pc.hullRoll       = dm.hullExclusion.roll;
            if (timed) gpuTimings_->begin(cb, vulkan::TP_OceanDisplace, currentFrame);
            waterDisplace_->recordDispatch(cb, st.displaceDS, pc);
            if (timed) gpuTimings_->end(cb, vulkan::TP_OceanDisplace, currentFrame);

            // (4c) World-space foam pass — same inputs as water_displace
            // (cascades, disturbances, hull/wake state) but evaluated per
            // foam-texel rather than per mesh vertex. Replaces the per-
            // vertex foam buffer that water_displace used to write. Run
            // AFTER water_displace finishes so we share the descriptor
            // pool's cascade-image bindings without a layout flip; the
            // cascades stay in GENERAL throughout. Skipped entirely on
            // off-interval frames: the accumulator keeps last dispatch's
            // content (reads need no barrier, layout stays GENERAL) and the
            // next dispatch's dt-aware decay covers the gap.
            if (runFoam) {
                // Wall-clock foam persistence. The old fixed 0.992/frame tied
                // the foam half-life to frame rate (≈1.4 s at 60 fps, half
                // that at 120) — same bug class as the TAA temporal constants.
                // τ = 2 s reproduces the old look at 60 fps. dt clamped so an
                // alt-tab / loading stall can't wipe the accumulator in one
                // frame.
                const double nowSec = frameNowSec();
                float foamDecay = 0.992f;// first frame: no dt reference yet
                if (st.foamPrevTimeSec >= 0.0) {
                    const float dt = std::clamp(
                            static_cast<float>(nowSec - st.foamPrevTimeSec),
                            0.0f, 0.25f);
                    foamDecay = std::exp(-dt / 2.0f);
                }
                st.foamPrevTimeSec = nowSec;

                vulkan::FoamWorldPipeline::PushConstants fpc{};
                fpc.disturbAddr    = st.foamDisturbBuffer.address;
                fpc.foamRes        = st.foamRes;
                fpc.foamTileSize   = st.foamTileSize;
                fpc.tileSize0      = dm.params.tileSize0;
                fpc.tileSize1      = dm.params.tileSize1;
                fpc.tileSize2      = dm.params.tileSize2;
                fpc.waveScale      = dm.params.waveScale;
                fpc.choppiness     = dm.params.choppiness;
                fpc.cascadeMask    = st.cascadeMask;
                fpc.hullCenterX    = dm.hullExclusion.centerX;
                fpc.hullCenterZ    = dm.hullExclusion.centerZ;
                fpc.hullHalfLength = dm.hullExclusion.halfLength;
                fpc.hullHalfBeam   = dm.hullExclusion.halfBeam;
                fpc.hullSinYaw     = dm.hullExclusion.sinYaw;
                fpc.hullCosYaw     = dm.hullExclusion.cosYaw;
                fpc.forwardSpeed   = dm.wake.enabled ? dm.wake.forwardSpeed : 0.0f;
                fpc.disturbCount   = disturbCount;
                fpc.decay          = foamDecay;
                fpc.wakeTrailAddr  = st.wakeTrailBuffer.address;
                fpc.wakeTrailCount = wakeSampleCount;
                fpc.natFoamScale   = std::clamp(dm.params.foamAmount, 0.0f, 1.0f);
                if (timed) gpuTimings_->begin(cb, vulkan::TP_OceanFoam, currentFrame);
                foamWorld_->recordDispatch(cb, st.foamWorldDS, fpc);
                if (timed) gpuTimings_->end(cb, vulkan::TP_OceanFoam, currentFrame);

                // Barrier: compute WRITE → ray-trace shader READ on the foam
                // image. chit (binding 44) samples it via a combined image-
                // sampler in the same frame.
                VkImageMemoryBarrier fmb{};
                fmb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                fmb.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                fmb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                fmb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                fmb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                fmb.image = st.foamImage.image;
                fmb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                fmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                fmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                vkCmdPipelineBarrier(cb,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                        0, 0, nullptr, 0, nullptr, 1, &fmb);
            }

            // Buffer barrier: compute write → AS-build read on the vertex/normal buffers.
            VkBufferMemoryBarrier bbs[2]{};
            bbs[0].sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            bbs[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            bbs[0].dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            bbs[0].buffer        = st.blas->vertex.handle;
            bbs[0].size          = VK_WHOLE_SIZE;
            bbs[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bbs[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bbs[1] = bbs[0];
            bbs[1].buffer = st.blas->normal.handle;
            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                0, 0, nullptr, 2, bbs, 0, nullptr);

            // (5) Rebuild the BLAS in-place (same buffers, same AS handle).
            // Mirrors refreshSkinnedBlas — extracted into a helper would be
            // nicer but the duplicated 70-line block is fine for a first cut.
            auto* posAttr = dm.geometry()->getAttribute<float>("position");
            auto* idxAttr = dm.geometry()->getIndex();
            const uint32_t vc = static_cast<uint32_t>(posAttr->count());
            const bool indexed = idxAttr != nullptr;
            const uint32_t primCount = indexed ? static_cast<uint32_t>(idxAttr->count() / 3)
                                               : vc / 3;

            VkAccelerationStructureGeometryTrianglesDataKHR triData{};
            triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triData.vertexData.deviceAddress = st.blas->vertex.address;
            triData.vertexStride = 3 * sizeof(float);
            triData.maxVertex = vc - 1;
            if (indexed) {
                triData.indexType = VK_INDEX_TYPE_UINT32;
                triData.indexData.deviceAddress = st.blas->index.address;
            } else {
                triData.indexType = VK_INDEX_TYPE_NONE_KHR;
            }

            VkAccelerationStructureGeometryKHR blasGeom{};
            blasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            blasGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            blasGeom.geometry.triangles = triData;
            blasGeom.flags = 0;

            // Refit after the initial buildBlasFor; periodic full rebuild
            // keeps the BVH balanced over time. buildBlasFor itself counts as
            // the first build, so blasRefitCounter starts at 0 and the very
            // first refreshDisplacedBlas takes the UPDATE path.
            const bool fullRebuild =
                    st.blasRefitCounter >= DisplacedMeshState::kBlasFullRebuildInterval;
            st.blasRefitCounter = fullRebuild ? 0u : (st.blasRefitCounter + 1u);

            VkAccelerationStructureBuildGeometryInfoKHR build{};
            build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            build.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                          VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            build.mode  = fullRebuild
                    ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
                    : VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
            build.geometryCount = 1;
            build.pGeometries = &blasGeom;
            build.srcAccelerationStructure = fullRebuild ? VK_NULL_HANDLE : st.blas->as;
            build.dstAccelerationStructure = st.blas->as;

            VkAccelerationStructureBuildSizesInfoKHR sizes{};
            sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &build, &primCount, &sizes);

            // Persistent scratch (lazy, sized to buildScratchSize which always
            // covers updateScratchSize) — same pattern as refreshGrassBlas. A
            // per-call transient scratch would be freed while the batched frame
            // command buffer still references it.
            if (st.blas->blasScratch.handle == VK_NULL_HANDLE ||
                st.blas->blasScratchSize < sizes.buildScratchSize) {
                if (st.blas->blasScratch.handle != VK_NULL_HANDLE)
                    destroyBuffer(ctx->allocator(), st.blas->blasScratch);
                st.blas->blasScratch = createAsScratchBuffer(
                        ctx->allocator(), ctx->device(), sizes.buildScratchSize);
                st.blas->blasScratchSize = sizes.buildScratchSize;
            }
            build.scratchData.deviceAddress = st.blas->blasScratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = primCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
            if (timed) gpuTimings_->begin(cb, vulkan::TP_OceanBlas, currentFrame);
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &build, &pRange);
            if (timed) gpuTimings_->end(cb, vulkan::TP_OceanBlas, currentFrame);
        }

void VulkanRenderer::Impl::mirrorDisplacedHeightfields(DisplacedMesh& dm, DisplacedMeshState& st) {
            // Nothing has ever called sampleHeight() on this mesh → the GPU
            // copies were skipped and there is nothing (correct) to mirror.
            if (!dm.wantsHeightReadback) return;
            // Reads slot currentFrame of the readback ring: the copy recorded
            // by this slot's previous occupant kFramesInFlight frames ago —
            // or, on the first-build path, by the one-shot that just ran.
            // The caller guarantees that copy is complete (frame-begin path:
            // beginDeferredFrame calls this right after
            // vkWaitForFences(inFlight[currentFrame]); one-shot:
            // endAndSubmitOneShot waited the queue), and nothing writes this
            // slot again until this frame records its own copy, after the
            // mirror. So the field is whole and a FIXED kFramesInFlight
            // frames old however the host is paced. The unwritten-slot guard
            // covers the first opted-in frames: the opt-in flips at
            // sampleHeight() time, but a slot only holds real data once a
            // command buffer has recorded the copies into it (the ring is
            // allocated on that first record); until then keep the last
            // good field rather than hand buoyancy uninitialized memory.
            const uint32_t slot = currentFrame;
            if (!st.heightReadbackWritten[slot]) return;
            const float tileSizes[3] = {dm.params.tileSize0, dm.params.tileSize1, dm.params.tileSize2};
            for (int ci = 0; ci < 3; ++ci) {
                auto& cf = dm.heightFields[ci];
                Buffer& buf = st.heightReadback[ci][slot];
                const uint32_t dim = st.heightReadbackDim[ci];
                if (buf.handle != VK_NULL_HANDLE && dim > 0) {
                    const size_t cells = size_t(dim) * size_t(dim);
                    const size_t bytes = cells * 2 * sizeof(float);
                    if (cf.data.size() != cells * 2)
                        cf.data.assign(cells * 2, 0.f);
                    void* mapped = nullptr;
                    vmaMapMemory(ctx->allocator(), buf.alloc, &mapped);
                    invalidateHostReads(ctx->allocator(), buf.alloc, 0, bytes);
                    std::memcpy(cf.data.data(), mapped, bytes);
                    vmaUnmapMemory(ctx->allocator(), buf.alloc);
                    cf.dim      = dim;
                    cf.tileSize = tileSizes[ci];
                }
            }
        }

void VulkanRenderer::Impl::refreshDisplacedBlas(DisplacedMesh& dm, DisplacedMeshState& st, float elapsedSeconds) {
            VkCommandBuffer cb = beginOneShot();
            recordDisplacedDeform(cb, dm, st, elapsedSeconds);
            endAndSubmitOneShot(cb);
            mirrorDisplacedHeightfields(dm, st);
        }

void VulkanRenderer::Impl::recordGrassDeform(VkCommandBuffer cb, GrassMesh& gm, GrassMeshState& st) {
            vulkan::GrassWindPipeline::PushConstants pc{};
            pc.posOut       = st.blas->vertex.address;
            pc.restIn       = st.restPos.address;
            pc.hfracIn      = st.heightFrac.address;
            pc.vertexCount  = st.vertexCount;
            pc.time         = gm.params.time;
            pc.windStrength = gm.params.windStrength;
            pc.windDirX     = gm.params.windDir.x;
            pc.windDirZ     = gm.params.windDir.y;
            grassWind_->recordDispatch(cb, pc);

            // Compute write → AS-build read (BLAS refit) + vertex-attribute read
            // (raster G-buffer prepass reads the deformed positions). Mirrors the
            // skinned path's post-dispatch barrier.
            {
                VkMemoryBarrier2 mb{};
                mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
                mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                mb.dstStageMask  = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                                   VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT |
                                   VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                mb.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR |
                                   VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                VkDependencyInfo dep{};
                dep.sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
                dep.memoryBarrierCount = 1;
                dep.pMemoryBarriers    = &mb;
                vkCmdPipelineBarrier2(cb, &dep);
            }

            // Refit the BLAS in place (periodic full rebuild keeps it balanced).
            auto* posAttr = gm.geometry()->getAttribute<float>("position");
            auto* idxAttr = gm.geometry()->getIndex();
            const uint32_t vc = static_cast<uint32_t>(posAttr->count());
            const bool indexed = idxAttr != nullptr;
            const uint32_t primCount = indexed ? static_cast<uint32_t>(idxAttr->count() / 3)
                                               : vc / 3;

            VkAccelerationStructureGeometryTrianglesDataKHR triData{};
            triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triData.vertexData.deviceAddress = st.blas->vertex.address;
            triData.vertexStride = 3 * sizeof(float);
            triData.maxVertex = vc - 1;
            if (indexed) {
                triData.indexType = VK_INDEX_TYPE_UINT32;
                triData.indexData.deviceAddress = st.blas->index.address;
            } else {
                triData.indexType = VK_INDEX_TYPE_NONE_KHR;
            }

            VkAccelerationStructureGeometryKHR blasGeom{};
            blasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            blasGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            blasGeom.geometry.triangles = triData;
            blasGeom.flags = 0;

            const bool fullRebuild =
                    st.blasRefitCounter >= GrassMeshState::kBlasFullRebuildInterval;
            st.blasRefitCounter = fullRebuild ? 0u : (st.blasRefitCounter + 1u);

            VkAccelerationStructureBuildGeometryInfoKHR build{};
            build.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            build.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                          VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            build.mode  = fullRebuild
                    ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
                    : VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
            build.geometryCount = 1;
            build.pGeometries = &blasGeom;
            build.srcAccelerationStructure = fullRebuild ? VK_NULL_HANDLE : st.blas->as;
            build.dstAccelerationStructure = st.blas->as;

            VkAccelerationStructureBuildSizesInfoKHR sizes{};
            sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &build, &primCount, &sizes);

            // Persistent scratch (sized once; buildScratchSize ≥ updateScratchSize).
            if (st.blas->blasScratch.handle == VK_NULL_HANDLE ||
                st.blas->blasScratchSize < sizes.buildScratchSize) {
                if (st.blas->blasScratch.handle != VK_NULL_HANDLE)
                    destroyBuffer(ctx->allocator(), st.blas->blasScratch);
                st.blas->blasScratch = createAsScratchBuffer(
                        ctx->allocator(), ctx->device(), sizes.buildScratchSize);
                st.blas->blasScratchSize = sizes.buildScratchSize;
            }
            build.scratchData.deviceAddress = st.blas->blasScratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = primCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &build, &pRange);
        }

void VulkanRenderer::Impl::refreshGrassBlas(GrassMesh& gm, GrassMeshState& st, float /*elapsedSeconds*/) {
            VkCommandBuffer cb = beginOneShot();
            recordGrassDeform(cb, gm, st);
            endAndSubmitOneShot(cb, "grass prime");
        }

void VulkanRenderer::Impl::buildTlas(const std::vector<VkAccelerationStructureInstanceKHR>& instances) {
            const uint32_t instanceCount = static_cast<uint32_t>(instances.size());
            const VkDeviceSize instBytes = std::max<VkDeviceSize>(
                    instanceCount * sizeof(VkAccelerationStructureInstanceKHR),
                    sizeof(VkAccelerationStructureInstanceKHR));// keep buf non-empty

            // (Re)allocate all in-flight instance buffers and seed every slot
            // with the current instances, so whichever slot the next per-frame
            // refit overwrites is already a valid build input.
            for (uint32_t s = 0; s < kFramesInFlight; ++s) {
                destroyBuffer(ctx->allocator(), tlasInstancesBuffers[s]);
                tlasInstancesBuffers[s] = createBuffer(
                        ctx->allocator(), ctx->device(), instBytes,
                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                if (instanceCount > 0) {
                    uploadHostVisible(ctx->allocator(), tlasInstancesBuffers[s], instances.data(),
                                      instanceCount * sizeof(VkAccelerationStructureInstanceKHR));
                }
            }

            VkAccelerationStructureGeometryInstancesDataKHR instData{};
            instData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
            instData.arrayOfPointers = VK_FALSE;
            instData.data.deviceAddress = tlasInstancesBuffers[0].address;

            VkAccelerationStructureGeometryKHR tlasGeom{};
            tlasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            tlasGeom.geometry.instances = instData;
            tlasGeom.flags = 0;// per-instance opaque is on VkAccelerationStructureInstanceKHR.flags, not here

            VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{};
            tlasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            tlasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            tlasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            tlasBuild.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            tlasBuild.geometryCount = 1;
            tlasBuild.pGeometries = &tlasGeom;

            VkAccelerationStructureBuildSizesInfoKHR tlasSizes{};
            tlasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(),
                    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &tlasBuild, &instanceCount, &tlasSizes);

            tlasBuffer = createBuffer(
                    ctx->allocator(), ctx->device(), tlasSizes.accelerationStructureSize,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);

            VkAccelerationStructureCreateInfoKHR tlasCreate{};
            tlasCreate.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            tlasCreate.buffer = tlasBuffer.handle;
            tlasCreate.size = tlasSizes.accelerationStructureSize;
            tlasCreate.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            check(ctx->rt().createAccelerationStructure(ctx->device(), &tlasCreate, nullptr, &tlas),
                  "vkCreateAccelerationStructureKHR(TLAS)");

            Buffer scratch = createAsScratchBuffer(ctx->allocator(), ctx->device(), tlasSizes.buildScratchSize);

            tlasBuild.dstAccelerationStructure = tlas;
            tlasBuild.scratchData.deviceAddress = scratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = instanceCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

            VkCommandBuffer cb = beginOneShot();
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &tlasBuild, &pRange);
            // Site 1 of 2: the structural build. Always MODE_BUILD — this path
            // recreates the TLAS object itself, so instance ordering here is
            // whatever the fresh build chooses.
            ++tlasStats_.fullRebuilds;
            tlasStats_.instances = instanceCount;
            endAndSubmitOneShot(cb, "buildTlas");
            destroyBuffer(ctx->allocator(), scratch);
            tlasBuiltInstanceCount_ = instanceCount;
        }

void VulkanRenderer::Impl::recordTlasRefit(VkCommandBuffer cb,
                             const std::vector<VkAccelerationStructureInstanceKHR>& instances,
                             bool fullBuild) {
            const uint32_t instanceCount = static_cast<uint32_t>(instances.size());
            if (instanceCount == 0 || tlas == VK_NULL_HANDLE) return;

            // A MODE_UPDATE must carry exactly the instance count of the TLAS's
            // last full build — anything else is invalid and corrupts traversal
            // (ray-query hang → TDR). Membership changes are classified
            // structural upstream, so this firing means some path slipped
            // through: promote to a full in-place build rather than corrupt.
            if (!fullBuild && instanceCount != tlasBuiltInstanceCount_) {
                std::cerr << "[VulkanRenderer] TLAS refit count " << instanceCount
                          << " != built count " << tlasBuiltInstanceCount_
                          << "; promoting to full build\n";
                fullBuild = true;
            }

            Buffer& instBuf = tlasInstancesBuffers[currentFrame];
            const VkDeviceSize instBytes =
                    instanceCount * sizeof(VkAccelerationStructureInstanceKHR);
            if (instBytes > instBuf.size) {
                // Instance buffers are sized by the last structural buildTlas;
                // writing past them corrupts whatever VMA placed next. A stale
                // TLAS for one frame is invisible next to that.
                std::cerr << "[VulkanRenderer] TLAS refit skipped: instance data "
                          << instBytes << " B exceeds staged buffer "
                          << instBuf.size << " B\n";
                return;
            }
            // 64 B/instance, straight into the buffer the AS builder reads as its
            // host-side build input (no staging hop). Distinct from
            // scene.7_tlasRefitFill, which is the CPU FILL of the same array back
            // inside ensureSceneBuilt; this is the copy to the device. Scoped to
            // the copy ALONE so it stays a sibling of frame.L_tlasRefitRec below.
            {
                THREEPP_CPUPROF("frame.H_uploadTlasInst");
                uploadHostVisible(ctx->allocator(), instBuf, instances.data(), instBytes);
            }
            // The build-size query is a per-frame driver round-trip at N instances
            // and the refit record is one command — separated from the copy so a
            // large L can be attributed to one or the other.
            THREEPP_CPUPROF("frame.L_tlasRefitRec");

            VkAccelerationStructureGeometryInstancesDataKHR instData{};
            instData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
            instData.arrayOfPointers = VK_FALSE;
            instData.data.deviceAddress = instBuf.address;

            VkAccelerationStructureGeometryKHR tlasGeom{};
            tlasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            tlasGeom.geometry.instances = instData;
            tlasGeom.flags = 0;

            const auto mode = fullBuild
                    ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
                    : VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;

            VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{};
            tlasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            tlasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            tlasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            tlasBuild.mode = mode;
            tlasBuild.srcAccelerationStructure = fullBuild ? VK_NULL_HANDLE : tlas;
            tlasBuild.dstAccelerationStructure = tlas;
            tlasBuild.geometryCount = 1;
            tlasBuild.pGeometries = &tlasGeom;

            VkAccelerationStructureBuildSizesInfoKHR sizes{};
            sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(),
                    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &tlasBuild, &instanceCount, &sizes);

            // Persistent scratch (sized once; build ≥ update). The TLAS build is
            // ordered after the prior frame's by submit order, so REUSE is safe.
            // GROWTH is not: this runs mid-record, and the previous frame's
            // cmdBuildAccelerationStructures may still be in flight reading the
            // old buffer at the address it captured. Destroying it here is a
            // use-after-free (a device-lost / TDR class of multi-second hitch on
            // a scene whose instance count is climbing). Hand it to the retire
            // queue instead, which holds it until its frame serial has provably
            // retired. retire() no-ops on a null handle and zeroes the source.
            const VkDeviceSize need = std::max(sizes.buildScratchSize, sizes.updateScratchSize);
            if (tlasRefitScratch_.handle == VK_NULL_HANDLE || tlasRefitScratchSize_ < need) {
                retire(std::move(tlasRefitScratch_));
                tlasRefitScratch_ = createAsScratchBuffer(ctx->allocator(), ctx->device(), need);
                tlasRefitScratchSize_ = need;
            }
            tlasBuild.scratchData.deviceAddress = tlasRefitScratch_.address;

            // A promoted/full build reuses the existing TLAS object, whose
            // storage was sized for the last structural build's count. Skip if
            // the fresh build wouldn't fit — one stale-TLAS frame, not a
            // heap stomp; the structural path re-sizes on its next pass.
            if (fullBuild && sizes.accelerationStructureSize > tlasBuffer.size) {
                std::cerr << "[VulkanRenderer] TLAS refit skipped: full build needs "
                          << sizes.accelerationStructureSize << " B, storage is "
                          << tlasBuffer.size << " B\n";
                return;
            }

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = instanceCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &tlasBuild, &pRange);
            // Site 2 of 2: the per-frame refit. `fullBuild` here is either the
            // caller's request or the count-mismatch promotion above, and it is
            // the same MODE_BUILD the structural path uses — so it counts as a
            // rebuild, not an update. The early returns above (count 0, no TLAS,
            // oversize instance data, oversize promoted build) record nothing,
            // deliberately: they record no command either.
            ++(fullBuild ? tlasStats_.fullRebuilds : tlasStats_.updates);
            tlasStats_.instances = instanceCount;
            if (fullBuild) tlasBuiltInstanceCount_ = instanceCount;
        }


void VulkanRenderer::Impl::lodWorkerMain() {
            for (;;) {
                LodJob job;
                {
                    std::unique_lock<std::mutex> lk(lodJobMutex_);
                    lodJobCv_.wait(lk, [this] { return lodWorkerStop_ || !lodJobQueue_.empty(); });
                    // Exit immediately on stop, even with jobs still queued —
                    // shutdown must be bounded; nothing will ever drain their
                    // results after this. A job already popped and mid-
                    // simplify (the loop body below, no lock held) still
                    // runs to completion — can't interrupt meshopt mid-call,
                    // and its result is simply never drained.
                    if (lodWorkerStop_) return;
                    job = std::move(lodJobQueue_.front());
                    lodJobQueue_.pop_front();
                }
                const size_t vertexCount = job.positions.size() / 3;
                // Soup input: weld first (identical-attribute duplicates →
                // canonical indices over the ORIGINAL vertex ids), then
                // simplify those. An empty weld result flows through as an
                // empty chain ⇒ LodState::Failed at drain, same as any other
                // degenerate geometry.
                std::vector<uint32_t> canonical;
                const uint32_t* idxData = job.indices.data();
                size_t idxCount = job.indices.size();
                if (job.indices.empty()) {
                    canonical = geometrylod::buildCanonicalIndices(
                            job.positions.data(),
                            job.normals.empty() ? nullptr : job.normals.data(),
                            job.uvs.empty() ? nullptr : job.uvs.data(),
                            vertexCount);
                    idxData = canonical.data();
                    idxCount = canonical.size();
                }
                // sparse=true for welded soup: the canonical indices reference
                // only ~1/6 of the soup vertex buffer (the representatives),
                // and meshopt needs SimplifySparse to keep the unreferenced
                // duplicates out of its wedge/seam analysis — see the
                // parameter doc in GeometryLod.hpp.
                auto levels = geometrylod::generateChain(
                        job.positions.data(), vertexCount, idxData, idxCount,
                        /*sparse=*/job.indices.empty(),
                        job.normals.empty() ? nullptr : job.normals.data(),
                        job.normalWeight);
                std::lock_guard<std::mutex> lk(lodResultMutex_);
                lodResultQueue_.push_back({job.geom, job.geomVersion, std::move(levels)});
            }
        }

void VulkanRenderer::Impl::destroyBlasLodLevels(BlasRecord& rec) {
            for (auto& lvl : rec.lodLevels) {
                if (lvl.as) ctx->rt().destroyAccelerationStructure(ctx->device(), lvl.as, nullptr);
                lodBlasBytes_  -= std::min<uint64_t>(lodBlasBytes_,  lvl.storage.size);
                lodIndexBytes_ -= std::min<uint64_t>(lodIndexBytes_, lvl.index.size);
                destroyBuffer(ctx->allocator(), lvl.storage);
                destroyBuffer(ctx->allocator(), lvl.index);
            }
            if (rec.lodState == BlasRecord::LodState::Ready && !rec.lodLevels.empty()) {
                lodChainsReadyCount_ = lodChainsReadyCount_ > 0 ? lodChainsReadyCount_ - 1 : 0;
            }
            rec.lodLevels.clear();
            rec.lodState = BlasRecord::LodState::None;
        }

VulkanRenderer::Impl::SkinnedMeshState* VulkanRenderer::Impl::ensureSkinnedBlas(SkinnedMesh& sm) {
            auto it = skinnedMeshStates.find(&sm);
            if (it != skinnedMeshStates.end()) return it->second.get();

            // Deforming paths (skinned / tet / morph / displaced) stay
            // float-typed by design: the skinning compute rewrites these very
            // buffers every frame, so upload-time widening would be a per-frame
            // tax and the narrow source array would be dead weight. A narrowed
            // normal here means someone ran compressAttributes() on a skinned
            // mesh — warn instead of silently dropping it from the scene.
            auto* posAttr     = sm.geometry()->getAttribute<float>("position");
            auto* nrmAttr     = sm.geometry()->getAttribute<float>("normal");
            auto* skinIdxAttr = sm.geometry()->getAttribute<float>("skinIndex");
            auto* skinWAttr   = sm.geometry()->getAttribute<float>("skinWeight");
            if (!nrmAttr && sm.geometry()->hasAttribute("normal")) {
                static bool warned = false;
                if (!warned) {
                    warned = true;
                    std::cerr << "[VulkanRenderer] skinned mesh has a non-float 'normal' "
                                 "attribute - deforming geometry must keep float attributes "
                                 "(do not compressAttributes() skinned meshes). Skipping.\n";
                }
            }
            if (!posAttr || !nrmAttr || !skinIdxAttr || !skinWAttr) return nullptr;
            if (!sm.skeleton || sm.skeleton->bones.empty()) return nullptr;

            // Build BLAS with the bind-pose positions/normals first. The
            // BLAS buffers are then re-written each frame by the skinning
            // compute shader (binding 5/6) and rebuilt in-place.
            auto rec = buildBlasFor(*sm.geometry());
            if (!rec) return nullptr;
            rec->liveCheck = sm.geometry();

            // Per-vertex previous-pose buffer. Used for two purposes:
            // (1) Hybrid raster motion-vector source (existing).
            // (2) Per-vertex prev-world-position reproject (2026-05-13):
            //     the deferred shade's ray-query hit handling reads via
            //     gdesc.prevVertexAddress, interpolates, and derives the
            //     hit's prevWorldPos, which feeds the motionMat reproject.
            //     The per-frame vkCmdCopyBuffer pushes current→prev before
            //     the skinning compute writes new positions.
            // buildBlasFor already allocated + seeded rec->prevVertex with
            // exactly the size/usage this path needs — re-creating it here
            // orphaned that buffer (VUID 05137 leak at device destroy).

            auto state = std::make_unique<SkinnedMeshState>();
            state->blas = std::move(rec);
            state->liveCheck = sm.geometry();
            const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
            const uint32_t boneCount   = static_cast<uint32_t>(sm.skeleton->bones.size());
            state->vertexCount = vertexCount;
            state->boneCount   = boneCount;
            state->prevBoneMats.assign(boneCount * 16, 0.f);

            auto* idxAttr = sm.geometry()->getIndex();
            state->indexed = idxAttr != nullptr;
            state->primitiveCount = state->indexed
                    ? static_cast<uint32_t>(idxAttr->count() / 3)
                    : vertexCount / 3;

            // ── GPU-skinning input buffers. Uploaded once, reused every frame.
            auto allocAndUpload = [&](Buffer& dst, const void* src,
                                      VkDeviceSize bytes) {
                dst = createBuffer(
                        ctx->allocator(), ctx->device(), bytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                uploadHostVisible(ctx->allocator(), dst, src, bytes);
            };
            allocAndUpload(state->baseVertex, posAttr->array().data(),
                           vertexCount * 3 * sizeof(float));
            allocAndUpload(state->baseNormal, nrmAttr->array().data(),
                           vertexCount * 3 * sizeof(float));
            allocAndUpload(state->skinIndex, skinIdxAttr->array().data(),
                           vertexCount * 4 * sizeof(float));
            allocAndUpload(state->skinWeight, skinWAttr->array().data(),
                           vertexCount * 4 * sizeof(float));

            // Bone matrices buffer: [bindMatrix, bindMatrixInverse, bones...].
            // bindMatrix is constant. bindMatrixInverse is NOT (attached bind mode
            // ties it to the current matrixWorld) — refreshSkinnedBlas re-uploads
            // it every frame; seed both here.
            // One per ring slot (see SkinnedMeshState::boneMatrices): the host
            // write lands before this frame's fence wait, so the slot an
            // earlier frame's dispatch is reading must not be the one being
            // written.
            const VkDeviceSize matsBytes = (2 + boneCount) * 16 * sizeof(float);
            for (auto& slot : state->boneMatrices) {
                slot = createBuffer(
                        ctx->allocator(), ctx->device(), matsBytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), slot.alloc, &mapped);
                std::memcpy(static_cast<char*>(mapped),
                            sm.bindMatrix.elements.data(),
                            16 * sizeof(float));
                std::memcpy(static_cast<char*>(mapped) + 16 * sizeof(float),
                            sm.bindMatrixInverse.elements.data(),
                            16 * sizeof(float));
                // The bones[..] region is left zeroed in every slot, as the
                // single buffer was. No dispatch ever reads it: a state only
                // reaches pendingSkinnedRebuilds_ through refreshSkinnedBlas,
                // which advances the slot and fills it immediately before the
                // dispatch that names it.
                std::memset(static_cast<char*>(mapped) + 32 * sizeof(float),
                            0, boneCount * 16 * sizeof(float));
                flushHostWrites(ctx->allocator(), slot.alloc);
                vmaUnmapMemory(ctx->allocator(), slot.alloc);
            }

            // Descriptor sets — one per ring slot, identical except for
            // binding 4 (the bone matrices).
            std::array<VkDescriptorBufferInfo, 7> bi{};
            std::array<VkWriteDescriptorSet, 7> wr{};
            for (uint32_t s = 0; s < SkinnedMeshState::kBoneSlots; ++s) {
                state->skinDescSet[s] = skinning_->allocateMeshDescriptorSet();

                const Buffer* bufs[7] = {
                        &state->baseVertex, &state->baseNormal,
                        &state->skinIndex,  &state->skinWeight,
                        &state->boneMatrices[s],
                        &state->blas->vertex, &state->blas->normal,
                };
                for (uint32_t i = 0; i < 7; ++i) {
                    bi[i].buffer        = bufs[i]->handle;
                    bi[i].offset        = 0;
                    bi[i].range         = VK_WHOLE_SIZE;
                    wr[i]               = {};
                    wr[i].sType         = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wr[i].dstSet        = state->skinDescSet[s];
                    wr[i].dstBinding    = i;
                    wr[i].descriptorCount = 1;
                    wr[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wr[i].pBufferInfo     = &bi[i];
                }
                vkUpdateDescriptorSets(ctx->device(),
                                       static_cast<uint32_t>(wr.size()),
                                       wr.data(), 0, nullptr);
            }

            // BLAS rebuild scratch buffer (persistent — sized once, reused).
            VkAccelerationStructureGeometryTrianglesDataKHR triData{};
            triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triData.vertexData.deviceAddress = state->blas->vertex.address;
            triData.vertexStride = 3 * sizeof(float);
            triData.maxVertex    = vertexCount - 1;
            if (state->indexed) {
                triData.indexType = VK_INDEX_TYPE_UINT32;
                triData.indexData.deviceAddress = state->blas->index.address;
            } else {
                triData.indexType = VK_INDEX_TYPE_NONE_KHR;
            }
            VkAccelerationStructureGeometryKHR blasGeom{};
            blasGeom.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            blasGeom.geometryType  = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            blasGeom.geometry.triangles = triData;
            blasGeom.flags         = 0;
            VkAccelerationStructureBuildGeometryInfoKHR blasBuild{};
            blasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            blasBuild.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            blasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            blasBuild.mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            blasBuild.geometryCount = 1;
            blasBuild.pGeometries   = &blasGeom;
            blasBuild.dstAccelerationStructure = state->blas->as;
            VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
            blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(),
                    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &blasBuild, &state->primitiveCount, &blasSizes);
            state->blasScratchSize = blasSizes.buildScratchSize;
            state->blasScratch = createAsScratchBuffer(
                    ctx->allocator(), ctx->device(), state->blasScratchSize);

            auto* raw = state.get();
            skinnedMeshStates.emplace(&sm, std::move(state));
            // First-frame refresh: upload bones + queue a rebuild so the BLAS
            // reflects the current pose, not bind pose, when the next frame
            // records.
            refreshSkinnedBlas(sm, *raw);
            return raw;
        }

VulkanRenderer::Impl::TetMeshState* VulkanRenderer::Impl::ensureTetBlas(Mesh& m) {
            auto found = tetMeshStates.find(&m);
            if (found != tetMeshStates.end()) return found->second.get();

            auto geom = m.geometry();
            if (!geom) return nullptr;
            auto* posAttr = geom->getAttribute<float>("position");
            auto* nrmAttr = geom->getAttribute<float>("normal");
            auto* tiAttr  = geom->getAttribute<float>("tetIndex");
            auto* twAttr  = geom->getAttribute<float>("tetWeight");
            auto* r0Attr  = geom->getAttribute<float>("tetRestInv0");
            auto* r1Attr  = geom->getAttribute<float>("tetRestInv1");
            auto* r2Attr  = geom->getAttribute<float>("tetRestInv2");
            auto mat = m.material();
            if (!posAttr || !nrmAttr || !tiAttr || !twAttr || !r0Attr || !r1Attr || !r2Attr) return nullptr;
            if (!mat || !mat->tetTexture) return nullptr;

            // BLAS built from the rest positions; the tet_skinning compute then
            // rewrites the vertex/normal buffers each frame and the BLAS is refit.
            auto rec = buildBlasFor(*geom);
            if (!rec) return nullptr;
            rec->liveCheck = geom;

            // Previous-frame vertex buffer for per-vertex motion vectors (same
            // role as the skinned path; copied current->prev before each
            // compute). buildBlasFor already allocated + seeded rec->prevVertex
            // — re-creating it here orphaned that buffer (VUID 05137).

            auto state = std::make_unique<TetMeshState>();
            state->blas = std::move(rec);
            state->liveCheck = geom;
            const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
            state->vertexCount = vertexCount;
            auto* idxAttr = geom->getIndex();
            state->indexed = idxAttr != nullptr;
            state->primitiveCount = state->indexed
                    ? static_cast<uint32_t>(idxAttr->count() / 3)
                    : vertexCount / 3;

            auto allocAndUpload = [&](Buffer& dst, const void* src, VkDeviceSize bytes) {
                dst = createBuffer(
                        ctx->allocator(), ctx->device(), bytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                uploadHostVisible(ctx->allocator(), dst, src, bytes);
            };
            allocAndUpload(state->tetIndex,   tiAttr->array().data(),  vertexCount * 4 * sizeof(float));
            allocAndUpload(state->tetWeight,  twAttr->array().data(),  vertexCount * 4 * sizeof(float));
            allocAndUpload(state->baseNormal, nrmAttr->array().data(), vertexCount * 3 * sizeof(float));
            allocAndUpload(state->restInv0,   r0Attr->array().data(),  vertexCount * 3 * sizeof(float));
            allocAndUpload(state->restInv1,   r1Attr->array().data(),  vertexCount * 3 * sizeof(float));
            allocAndUpload(state->restInv2,   r2Attr->array().data(),  vertexCount * 3 * sizeof(float));

            // Per-frame tet positions ring, sized to the tet texture image (one
            // RGBA32F texel per collision-tet vertex). Filled by refreshTetBlas,
            // one slot per frame (see TetMeshState::tetPos).
            const auto& tetImg = mat->tetTexture->image().data<float>();
            state->tetPosBytes = static_cast<VkDeviceSize>(tetImg.size()) * sizeof(float);
            for (auto& slot : state->tetPos) {
                slot = createBuffer(
                        ctx->allocator(), ctx->device(), state->tetPosBytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            }

            // Descriptor sets — 9 storage buffers each (see tet_skinning.comp);
            // one set per ring slot, identical except binding 6 (tetPos).
            for (uint32_t s = 0; s < TetMeshState::kTetPosSlots; ++s) {
                state->tetDescSet[s] = tetSkinning_->allocateMeshDescriptorSet();
                std::array<VkDescriptorBufferInfo, 9> bi{};
                const Buffer* bufs[9] = {
                        &state->tetIndex, &state->tetWeight, &state->baseNormal,
                        &state->restInv0, &state->restInv1, &state->restInv2,
                        &state->tetPos[s],
                        &state->blas->vertex, &state->blas->normal,
                };
                std::array<VkWriteDescriptorSet, 9> wr{};
                for (uint32_t i = 0; i < 9; ++i) {
                    bi[i].buffer          = bufs[i]->handle;
                    bi[i].offset          = 0;
                    bi[i].range           = VK_WHOLE_SIZE;
                    wr[i]                 = {};
                    wr[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wr[i].dstSet          = state->tetDescSet[s];
                    wr[i].dstBinding      = i;
                    wr[i].descriptorCount = 1;
                    wr[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wr[i].pBufferInfo     = &bi[i];
                }
                vkUpdateDescriptorSets(ctx->device(),
                                       static_cast<uint32_t>(wr.size()), wr.data(), 0, nullptr);
            }

            // Persistent BLAS-rebuild scratch (sized once, reused every frame).
            VkAccelerationStructureGeometryTrianglesDataKHR triData{};
            triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triData.vertexData.deviceAddress = state->blas->vertex.address;
            triData.vertexStride = 3 * sizeof(float);
            triData.maxVertex    = vertexCount - 1;
            if (state->indexed) {
                triData.indexType = VK_INDEX_TYPE_UINT32;
                triData.indexData.deviceAddress = state->blas->index.address;
            } else {
                triData.indexType = VK_INDEX_TYPE_NONE_KHR;
            }
            VkAccelerationStructureGeometryKHR blasGeom{};
            blasGeom.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            blasGeom.geometryType  = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            blasGeom.geometry.triangles = triData;
            blasGeom.flags         = 0;
            VkAccelerationStructureBuildGeometryInfoKHR blasBuild{};
            blasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            blasBuild.type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            blasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                              VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
            blasBuild.mode  = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            blasBuild.geometryCount = 1;
            blasBuild.pGeometries   = &blasGeom;
            blasBuild.dstAccelerationStructure = state->blas->as;
            VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
            blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
            ctx->rt().getAccelerationStructureBuildSizes(
                    ctx->device(),
                    VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    &blasBuild, &state->primitiveCount, &blasSizes);
            state->blasScratchSize = blasSizes.buildScratchSize;
            state->blasScratch = createAsScratchBuffer(
                    ctx->allocator(), ctx->device(), state->blasScratchSize);

            auto* raw = state.get();
            tetMeshStates.emplace(&m, std::move(state));
            // First-frame refresh so the BLAS reflects the current deformation.
            refreshTetBlas(m, *raw);
            return raw;
        }

VulkanRenderer::Impl::DisplacedMeshState* VulkanRenderer::Impl::ensureDisplacedState(DisplacedMesh& dm) {
            auto it = displacedStates.find(&dm);
            if (it != displacedStates.end()) return it->second.get();

            auto* posAttr = dm.geometry()->getAttribute<float>("position");
            if (!posAttr) return nullptr;
            const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
            // Grid topology: explicit gridWidth/gridDepth hints when set
            // (Ocean::create always fills them — required for rectangular
            // grids), else the historical square derivation from
            // sqrt(vertexCount). PlaneGeometry(w, h, segX, segY) produces
            // (segX+1)·(segY+1) verts laid out row-major, X fastest.
            uint32_t gridDimX = dm.gridWidth;
            uint32_t gridDimZ = dm.gridDepth;
            if (gridDimX == 0 || gridDimZ == 0) {
                const uint32_t gridDim = static_cast<uint32_t>(std::round(std::sqrt(double(vertexCount))));
                gridDimX = gridDimZ = gridDim;
            }
            if (gridDimX < 2 || gridDimZ < 2 || gridDimX * gridDimZ != vertexCount) return nullptr;

            // Plane extents: derive from the rest-position bbox (the displace
            // pass reconstructs rest positions from grid index + these).
            float xMin = std::numeric_limits<float>::infinity();
            float xMax = -std::numeric_limits<float>::infinity();
            float zMin = std::numeric_limits<float>::infinity();
            float zMax = -std::numeric_limits<float>::infinity();
            for (uint32_t i = 0; i < vertexCount; ++i) {
                const float x = posAttr->getX(i);
                const float z = posAttr->getZ(i);
                if (x < xMin) xMin = x;
                if (x > xMax) xMax = x;
                if (z < zMin) zMin = z;
                if (z > zMax) zMax = z;
            }
            const float planeSizeX = xMax - xMin;
            const float planeSizeZ = zMax - zMin;
            if (!(planeSizeX > 0.f) || !(planeSizeZ > 0.f)) return nullptr;

            auto blas = buildBlasFor(*dm.geometry());
            if (!blas) return nullptr;
            blas->liveCheck = dm.geometry();

            // FFT-displaced ocean mesh: mark it so the chit / deferred passes
            // apply world-space foam + thin-shell water shading. Foam itself
            // lives in a world-space texture built by foam_world.comp, so the
            // only per-mesh state needed here is this "is water" marker (it
            // used to be a zero-filled per-vertex foam buffer, read solely for
            // its non-null address — the contents were dead).
            blas->isOceanSurface = true;

            // Per-vertex previous-pose buffer for hybrid raster motion vec —
            // buildBlasFor already allocated + seeded blas->prevVertex (same
            // size R32G32B32 SFLOAT × vertexCount, same usage); it is filled
            // GPU-side via vkCmdCopyBuffer before each water_displace
            // dispatch. Re-creating it here orphaned the buildBlasFor buffer
            // (the one VUID 05137 leak that survived renderer teardown).

            auto state = std::make_unique<DisplacedMeshState>();
            state->blas = std::move(blas);
            state->vertexCount = vertexCount;
            state->gridDimX = gridDimX;
            state->gridDimZ = gridDimZ;
            state->planeSizeX = planeSizeX;
            state->planeSizeZ = planeSizeZ;
            state->liveCheck = dm.geometry();

            // FFT cascades — one Phillips/Dynamic/IFFT chain per non-zero
            // `tileSize` in DisplacedMesh::Params. Cascades are band-passed
            // by k so each covers a disjoint wavenumber range:
            //   cascade 0 (largest tile): 0 → kNyq of cascade 1
            //   cascade 1 (middle):       kNyq of cascade 1 → kNyq of cascade 2
            //   cascade 2 (smallest):     kNyq of cascade 2 → ∞
            // where kNyq_i = π·N / tileSize_i. Cascade 0 is required;
            // tileSize1/tileSize2 == 0 disable the corresponding band.
            const float tileSizes[3] = {
                    dm.params.tileSize0,
                    dm.params.tileSize1,
                    dm.params.tileSize2,
            };
            const uint32_t textureSizes[3] = {
                    dm.params.textureSize0,
                    dm.params.textureSize1,
                    dm.params.textureSize2,
            };
            // Hand-off k between adjacent cascades = the SMALLER tile's lowest
            // natural k = 2π / tileSize_(smaller). Each cascade then covers
            // wavelengths between its own tile and the next-smaller tile:
            //   cascade 0: λ ∈ [tileSize_1, tileSize_0]
            //   cascade 1: λ ∈ [tileSize_2, tileSize_1]
            //   cascade 2: λ ∈ [<tileSize_2]  (down to its own Nyquist)
            //
            // Why not split at the larger tile's Nyquist (k = π·N / L_larger)?
            // Because that boundary is at the mesh's resolving limit — it
            // hands all the mesh-resolvable wavelengths to cascade 0 alone,
            // and cascades 1 and 2 emit only sub-mesh wavelengths that
            // alias as displacement noise. The 2π/L_smaller scheme reserves
            // a real, mesh-displayable band for each intermediate cascade.
            constexpr float kTwoPi = 6.28318530717958647692f;
            const float kHandoff01 = (tileSizes[0] > 0.f && tileSizes[1] > 0.f)
                    ? kTwoPi / tileSizes[1] : 0.f;
            const float kHandoff12 = (tileSizes[1] > 0.f && tileSizes[2] > 0.f)
                    ? kTwoPi / tileSizes[2] : 0.f;
            for (uint32_t i = 0; i < 3; ++i) {
                if (!(tileSizes[i] > 0.f)) continue;
                const uint32_t texSize = textureSizes[i];
                water::PhillipsSpectrum::Settings ps{};
                ps.textureSize = texSize;
                ps.tileSize    = tileSizes[i];
                // Cascade 1's SAMPLE DOMAIN is rotated by kOceanCascade1RotTheta
                // (repeat-lattice break — see vulkan_shared.h). A domain-space
                // wave at angle θd propagates in world at θd − θrot, so add the
                // rotation here to keep the rendered propagation direction at
                // the caller's windTheta for every cascade.
                ps.windTheta   = dm.params.windTheta +
                                 (i == 1 ? kOceanCascade1RotTheta : 0.f);
                ps.windSpeed   = dm.params.windSpeed;
                ps.fetch       = dm.params.fetch;
                if (i == 0) {
                    ps.kMin = 0.f;
                    ps.kMax = kHandoff01; // 0 if no cascade 1 → no upper bound
                } else if (i == 1) {
                    ps.kMin = kHandoff01;
                    ps.kMax = kHandoff12; // 0 if no cascade 2 → no upper bound
                } else {
                    ps.kMin = kHandoff12;
                    ps.kMax = 0.f;
                }
                // Short-wave roll-off. Two regimes:
                //  • OPEN band (kMax == 0, the finest enabled cascade): suppress
                //    wavelengths shorter than ~5× the sample spacing, or
                //    Phillips puts energy into bands the FFT can't resolve and
                //    the crests alias into spikes. Resolution-tied by design.
                //  • BAND-LIMITED (kMax > 0): the hard kMax already guarantees
                //    nothing unresolvable is emitted, so the roll-off only
                //    tapers the band edge. Keyed to the BAND, not the
                //    resolution: the old 5·tile/texSize formula would wipe out
                //    the entire band content when the band-passed cascades run
                //    at their right-sized (small) resolutions.
                //    The per-cascade ratios REPRODUCE the attenuation the old
                //    formula gave at the historical resolutions ({1024, 512}):
                //    0.05·λmin ≈ 0.91 at cascade 0's edge, 0.12·λmin ≈ 0.55 at
                //    cascade 1's. The stronger cascade-1 taper is load-bearing:
                //    a first cut used 0.05 for both, and the extra ~9–12 m
                //    energy it left in cascade 1 aliased into a checkerboard
                //    moiré at grazing distances where the (warp-thinned) mesh
                //    spacing approaches those wavelengths' Nyquist.
                ps.smallWaveCutoff = (ps.kMax > 0.f)
                        ? (i == 0 ? 0.05f : 0.12f) * (kTwoPi / ps.kMax)
                        : 5.f * tileSizes[i] / float(texSize);
                auto& c = state->cascades[i];
                c.tileSize = tileSizes[i];
                c.phillips = std::make_unique<water::PhillipsSpectrum>(*ctx, ps);
                c.dyn      = std::make_unique<water::DynamicSpectrum>(
                        *ctx, *c.phillips, texSize, tileSizes[i]);
                c.ifft     = std::make_unique<water::IFFT>(*ctx, texSize);
                state->cascadeMask |= (1u << i);
            }
            if (state->cascadeMask == 0u) return nullptr; // no cascades → invalid setup
            state->appliedWindSpeed = dm.params.windSpeed;
            state->appliedWindTheta = dm.params.windTheta;
            state->appliedFetch     = dm.params.fetch;

            // Per-cascade height readback DIMENSIONS. Each cascade can run at
            // a different FFT resolution, so each readback buffer is sized to
            // its own dim²·8 bytes (RG32F). The buffers themselves — a ring of
            // kFramesInFlight per cascade, see DisplacedMeshState — are
            // allocated by recordDisplacedDeform on the first frame that
            // records the copies (sampleHeight()'s sticky opt-in); a scene
            // that never queries CPU wave height never pays for them.
            state->heightReadbackDim[0] = textureSizes[0];
            if (dm.params.tileSize1 > 0.f) state->heightReadbackDim[1] = textureSizes[1];
            if (dm.params.tileSize2 > 0.f) state->heightReadbackDim[2] = textureSizes[2];

            // Scratch image for IFFT ping-pong (RG32F). Cascades dispatch
            // back-to-back on the same queue and share this scratch, so size
            // it to the largest enabled cascade. Smaller cascades' IFFT runs
            // only touch their own extent within the scratch — the unused
            // tail is harmless.
            uint32_t scratchDim = 0;
            for (uint32_t i = 0; i < 3; ++i) {
                if (tileSizes[i] > 0.f) scratchDim = std::max(scratchDim, textureSizes[i]);
            }
            {
                VkImageCreateInfo ici{};
                ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                ici.imageType     = VK_IMAGE_TYPE_2D;
                ici.format        = VK_FORMAT_R32G32_SFLOAT;
                ici.extent        = {scratchDim, scratchDim, 1};
                ici.mipLevels     = 1;
                ici.arrayLayers   = 1;
                ici.samples       = VK_SAMPLE_COUNT_1_BIT;
                ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
                ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT |
                                    VK_IMAGE_USAGE_SAMPLED_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
                ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                VmaAllocationCreateInfo aci{};
                aci.usage = VMA_MEMORY_USAGE_AUTO;
                check(vmaCreateImage(ctx->allocator(), &ici, &aci,
                                     &state->scratchA.image, &state->scratchA.alloc, nullptr),
                      "vmaCreateImage(displaceScratch)");
                state->scratchA.format = VK_FORMAT_R32G32_SFLOAT;
                state->scratchA.width  = scratchDim;
                state->scratchA.height = scratchDim;
                VkImageViewCreateInfo vci{};
                vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                vci.image = state->scratchA.image;
                vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                vci.format = VK_FORMAT_R32G32_SFLOAT;
                vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                check(vkCreateImageView(ctx->device(), &vci, nullptr, &state->scratchA.view),
                      "vkCreateImageView(displaceScratch)");
                ctx->setObjectName(state->scratchA.image, "ocean.scratchA (IFFT ping-pong)");
                ctx->setObjectName(state->scratchA.view,  "ocean.scratchA (IFFT ping-pong)");
            }

            // World-space foam image. Coverage equals the cascade-0 tile
            // (matches the FFT periodicity, so REPEAT-sampling at any
            // world XZ folds back into the same texture cell). Resolution
            // targets ~1 m per texel — capped at 2048² and floored at 256²
            // so a small pond isn't handed a giant accumulator. The target
            // was 0.5 m/texel (1 km tile → 2048², 16 MB) until the 2026-08
            // pass bench: foam_world was the ocean's single biggest GPU item
            // at 0.74 ms/frame, 1 m/texel cuts it to 0.28 ms, and A/B
            // captures at wind 10/16 (whitecap field, wake-trail sheet,
            // vista) were indistinguishable — foam is a decayed accumulator
            // sampled bilinearly, so the detail floor is the stamp widths,
            // not the texel grid. THREEPP_OCEAN_FOAM_TEXEL=0.5 restores the
            // old density. R32F storage so both compute imageLoad/Store and
            // chit linear sampling work without format conversions.
            {
                // THREEPP_OCEAN_FOAM_RES caps the foam accumulator's edge (power
                // of two). Raise it to 2048 to restore the pre-cap density on a
                // large ocean; see the cost note below.
                static const uint32_t kFoamResCap = [] {
                    const char* e = std::getenv("THREEPP_OCEAN_FOAM_RES");
                    const long v = e ? std::atol(e) : 1024L;
                    return static_cast<uint32_t>(std::clamp(v, 256L, 2048L));
                }();
                static const float kFoamTexelTarget = [] {
                    const char* e = std::getenv("THREEPP_OCEAN_FOAM_TEXEL");
                    const float v = e ? static_cast<float>(std::atof(e)) : 1.0f;
                    return v > 0.f ? v : 1.0f;
                }();
                // Cap by COST, not by texture size. The loop below targets a
                // fixed texel SIZE, so the dispatch grows with tile AREA: the
                // fjord's 3200 m cascade-0 tile hit the old 2048 ceiling at
                // 4.2 M texels/frame (2.5-3.0 ms) while a 500 m ocean bought
                // the same visual density for ~1/40th. 2048 read as "largest
                // sane texture"; what actually matters is the dispatch cost,
                // and 1024^2 = 1 M texels lands at ~0.9 ms. Small oceans never
                // reach the cap, so this changes nothing for them.
                uint32_t foamRes = 256u;
                while (foamRes < kFoamResCap && float(foamRes) * kFoamTexelTarget < dm.params.tileSize0)
                    foamRes *= 2u;
                state->foamRes      = foamRes;
                state->foamTileSize = dm.params.tileSize0;
                // Attach the accumulator config to any run that is profiling.
                // Foam cost is set by texel COUNT, so a foam number without its
                // foamRes is ambiguous — an unattached 1024-vs-2048 assumption
                // manufactured a phantom 3.4x per-texel anomaly once already.
                if (vulkan::cpuprof::Registry::get().on) {
                    std::fprintf(stderr,
                                 "[foam] foamRes=%u tileSize0=%.1f texelTarget=%.2f "
                                 "-> %.2f m/texel, %.2fM texels\n",
                                 foamRes, double(dm.params.tileSize0), double(kFoamTexelTarget),
                                 double(dm.params.tileSize0) / double(foamRes),
                                 double(foamRes) * double(foamRes) / 1.0e6);
                }
                VkImageCreateInfo ici{};
                ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                ici.imageType     = VK_IMAGE_TYPE_2D;
                ici.format        = VK_FORMAT_R32_SFLOAT;
                ici.extent        = {state->foamRes, state->foamRes, 1};
                ici.mipLevels     = 1;
                ici.arrayLayers   = 1;
                ici.samples       = VK_SAMPLE_COUNT_1_BIT;
                ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
                ici.usage         = VK_IMAGE_USAGE_STORAGE_BIT |
                                    VK_IMAGE_USAGE_SAMPLED_BIT |
                                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                ici.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
                ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                VmaAllocationCreateInfo aci{};
                aci.usage = VMA_MEMORY_USAGE_AUTO;
                check(vmaCreateImage(ctx->allocator(), &ici, &aci,
                                     &state->foamImage.image, &state->foamImage.alloc, nullptr),
                      "vmaCreateImage(foamWorld)");
                state->foamImage.format = VK_FORMAT_R32_SFLOAT;
                state->foamImage.width  = state->foamRes;
                state->foamImage.height = state->foamRes;
                VkImageViewCreateInfo vci{};
                vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                vci.image = state->foamImage.image;
                vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                vci.format = VK_FORMAT_R32_SFLOAT;
                vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                check(vkCreateImageView(ctx->device(), &vci, nullptr, &state->foamImage.view),
                      "vkCreateImageView(foamWorld)");
                ctx->setObjectName(state->foamImage.image, "ocean.foamWorld (R32F)");
                ctx->setObjectName(state->foamImage.view,  "ocean.foamWorld (R32F)");

                // Initial clear to zero + layout transition to GENERAL so the
                // first foam_world dispatch's imageLoad reads 0 (no foam yet)
                // and the chit's linear sampler reads from a defined image.
                VkCommandBuffer cb = beginOneShot();
                {
                    VkImageMemoryBarrier imb{};
                    imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    imb.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                    imb.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    imb.srcAccessMask = 0;
                    imb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    imb.image = state->foamImage.image;
                    imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    vkCmdPipelineBarrier(cb,
                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            0, 0, nullptr, 0, nullptr, 1, &imb);
                }
                VkClearColorValue cc{};
                cc.float32[0] = 0.0f;
                VkImageSubresourceRange sub{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cb, state->foamImage.image,
                                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                     &cc, 1, &sub);
                {
                    VkImageMemoryBarrier imb{};
                    imb.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    imb.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                    imb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    imb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    imb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                    imb.image = state->foamImage.image;
                    imb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                    imb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    imb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    vkCmdPipelineBarrier(cb,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                    VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                            0, 0, nullptr, 0, nullptr, 1, &imb);
                }
                endAndSubmitOneShot(cb);
                state->foamImage.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
            }

            // Allocate this mesh's displace descriptor set + write bindings.
            state->displaceDS = waterDisplace_->allocateMeshDescriptorSet();

            // Bind each enabled cascade's spatial images to its (height, displace)
            // slot pair. Disabled cascades are filled with cascade 0's images
            // so the shader's combined-image-sampler bindings are always valid;
            // the shader gates which slots are actually sampled via cascadeMask.
            std::array<VkDescriptorImageInfo, 6> imageInfos{};
            for (uint32_t i = 0; i < 3; ++i) {
                const uint32_t srcCascade = (state->cascadeMask & (1u << i)) ? i : 0u;
                const auto& c = state->cascades[srcCascade];
                imageInfos[i * 2 + 0].sampler     = waterDisplace_->sampler();
                imageInfos[i * 2 + 0].imageView   = c.dyn->ht().view;
                imageInfos[i * 2 + 0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                imageInfos[i * 2 + 1].sampler     = waterDisplace_->sampler();
                imageInfos[i * 2 + 1].imageView   = c.dyn->displacement().view;
                imageInfos[i * 2 + 1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
            std::array<VkWriteDescriptorSet, 6> ws{};
            for (uint32_t i = 0; i < 6; ++i) {
                ws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                ws[i].dstSet = state->displaceDS;
                ws[i].dstBinding = i;
                ws[i].descriptorCount = 1;
                ws[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                ws[i].pImageInfo = &imageInfos[i];
            }
            vkUpdateDescriptorSets(ctx->device(), uint32_t(ws.size()), ws.data(), 0, nullptr);

            // Foam-world descriptor set — same cascade bindings 0..5 as the
            // displace set, plus binding 6 = the storage image foam target.
            state->foamWorldDS = foamWorld_->allocateMeshDescriptorSet();
            std::array<VkDescriptorImageInfo, 6> foamCascadeInfos{};
            for (uint32_t i = 0; i < 3; ++i) {
                const uint32_t srcCascade = (state->cascadeMask & (1u << i)) ? i : 0u;
                const auto& c = state->cascades[srcCascade];
                foamCascadeInfos[i * 2 + 0].sampler     = foamWorld_->sampler();
                foamCascadeInfos[i * 2 + 0].imageView   = c.dyn->ht().view;
                foamCascadeInfos[i * 2 + 0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                foamCascadeInfos[i * 2 + 1].sampler     = foamWorld_->sampler();
                foamCascadeInfos[i * 2 + 1].imageView   = c.dyn->displacement().view;
                foamCascadeInfos[i * 2 + 1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            }
            VkDescriptorImageInfo foamStorageInfo{};
            foamStorageInfo.sampler     = VK_NULL_HANDLE;
            foamStorageInfo.imageView   = state->foamImage.view;
            foamStorageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            std::array<VkWriteDescriptorSet, 7> fws{};
            for (uint32_t i = 0; i < 6; ++i) {
                fws[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                fws[i].dstSet = state->foamWorldDS;
                fws[i].dstBinding = i;
                fws[i].descriptorCount = 1;
                fws[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                fws[i].pImageInfo = &foamCascadeInfos[i];
            }
            fws[6].sType          = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            fws[6].dstSet         = state->foamWorldDS;
            fws[6].dstBinding     = 6;
            fws[6].descriptorCount= 1;
            fws[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            fws[6].pImageInfo     = &foamStorageInfo;
            vkUpdateDescriptorSets(ctx->device(), uint32_t(fws.size()), fws.data(), 0, nullptr);

            // Hand the smallest enabled cascade's height image to closest_hit
            // (binding 32) for sub-mesh-resolution normal perturbation. Picks
            // the highest enabled bit — the cascade with the smallest tileSize
            // and therefore the finest spatial resolution. The cascade VkImage
            // handle is stable until the DisplacedMesh is destroyed, so we
            // only rewrite the descriptor on first init (not per frame).
            {
                uint32_t fineIdx = 0;
                float fineTile   = 0.f;
                for (uint32_t i = 0; i < 3; ++i) {
                    if (state->cascadeMask & (1u << i)) {
                        fineIdx  = i;
                        fineTile = state->cascades[i].tileSize;
                    }
                }
                if (fineTile > 0.f) {
                    oceanFineHeightView = state->cascades[fineIdx].dyn->ht().view;
                    oceanFineTileSize   = fineTile;
                    rewriteDeferredDescriptors();
                }
            }

            // Hand this mesh's world-foam view to closest_hit (binding 33) so
            // the chit can sample foam at arbitrary world XZ during shading.
            // Like oceanFineHeight above, the foam VkImage handle is stable
            // until the DisplacedMesh is destroyed, so we only rewrite once.
            oceanFoamView     = state->foamImage.view;
            oceanFoamTileSize = state->foamTileSize;
            rewriteDeferredDescriptors();

            auto* raw = state.get();
            displacedStates.emplace(&dm, std::move(state));
            return raw;
        }

VulkanRenderer::Impl::GrassMeshState* VulkanRenderer::Impl::ensureGrassState(GrassMesh& gm) {
            auto it = grassStates.find(&gm);
            if (it != grassStates.end()) return it->second.get();

            auto* posAttr = gm.geometry()->getAttribute<float>("position");
            auto* hfAttr  = gm.geometry()->getAttribute<float>("heightFrac");
            if (!posAttr || !hfAttr) return nullptr;
            const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
            if (vertexCount == 0 || static_cast<uint32_t>(hfAttr->count()) != vertexCount) return nullptr;

            auto blas = buildBlasFor(*gm.geometry());
            if (!blas) return nullptr;
            blas->liveCheck = gm.geometry();

            auto state = std::make_unique<GrassMeshState>();
            state->vertexCount = vertexCount;
            state->liveCheck = gm.geometry();

            // Immutable rest positions — the shader reads these and writes the
            // displaced result into the (separate) BLAS vertex buffer.
            const VkDeviceSize posBytes = VkDeviceSize(vertexCount) * 3u * sizeof(float);
            state->restPos = createBuffer(
                    ctx->allocator(), ctx->device(), posBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            uploadHostVisible(ctx->allocator(), state->restPos, posAttr->array().data(), posBytes);

            const VkDeviceSize hfBytes = VkDeviceSize(vertexCount) * sizeof(float);
            state->heightFrac = createBuffer(
                    ctx->allocator(), ctx->device(), hfBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            uploadHostVisible(ctx->allocator(), state->heightFrac, hfAttr->array().data(), hfBytes);

            state->blas = std::move(blas);
            auto* raw = state.get();
            grassStates.emplace(&gm, std::move(state));
            return raw;
        }

bool VulkanRenderer::Impl::enqueueLodJob(const BufferGeometry* geomPtr, unsigned int geomVersion, BufferGeometry& geom,
                           float normalWeight) {
            auto* posAttr = geom.getAttribute<float>("position");
            if (!posAttr) return false;
            LodJob job;
            job.geom = geomPtr;
            job.geomVersion = geomVersion;
            job.normalWeight = normalWeight;
            const auto& pos = posAttr->array();
            job.positions.assign(pos.begin(), pos.end());
            const auto vtxCount = posAttr->count();
            // Normals feed the simplifier's ATTRIBUTE metric on every path
            // (indexed and soup) — without them the position-only quadric
            // flattens smooth glossy surfaces' shading for free (the
            // CarConcept regression; see generateChain's header doc). On the
            // soup path they additionally drive the weld's seam preservation.
            if (FloatAttributeView nrm{geom.getAttribute("normal")};
                nrm && nrm.count() == vtxCount &&
                nrm.size() == static_cast<size_t>(vtxCount) * 3) {
                job.normals.assign(nrm.data(), nrm.data() + nrm.size());
            }
            if (const auto* idxAttr = geom.getIndex()) {
                const auto& idx = idxAttr->array();
                job.indices.assign(idx.begin(), idx.end());
            } else {
                // Soup weld only: UVs join the binary-equality test so
                // genuine UV seams stay split.
                if (FloatAttributeView uv{geom.getAttribute("uv")};
                    uv && uv.count() == vtxCount &&
                    uv.size() == static_cast<size_t>(vtxCount) * 2) {
                    job.uvs.assign(uv.data(), uv.data() + uv.size());
                }
            }
            ensureLodWorkerStarted();
            {
                std::lock_guard<std::mutex> lk(lodJobMutex_);
                lodJobQueue_.push_back(std::move(job));
            }
            lodJobCv_.notify_one();
            ++lodChainsQueuedCount_;
            return true;
        }

void VulkanRenderer::Impl::cpuMorphBlend(Mesh& mesh,
                                  std::vector<float>& outPos,
                                  std::vector<float>& outNorm) {
            const auto& geom = *mesh.geometry();
            auto* posAttr = geom.getAttribute<float>("position");
            auto* nrmAttr = geom.getAttribute<float>("normal");
            if (!posAttr) return;

            const int vtxCount = posAttr->count();
            const auto& basePos = posAttr->array();
            outPos.assign(basePos.begin(), basePos.end());

            if (nrmAttr) {
                const auto& baseNrm = nrmAttr->array();
                outNorm.assign(baseNrm.begin(), baseNrm.end());
            } else {
                outNorm.assign(vtxCount * 3, 0.f);
            }

            const auto& morphAttrsMap = geom.getMorphAttributes();
            auto posIt = morphAttrsMap.find("position");
            if (posIt == morphAttrsMap.end()) return;
            const auto& morphPos = posIt->second;

            const std::vector<std::shared_ptr<BufferAttribute>>* morphNrm = nullptr;
            auto nrmIt = morphAttrsMap.find("normal");
            if (nrmIt != morphAttrsMap.end()) morphNrm = &nrmIt->second;

            auto* morphObj = mesh.as<ObjectWithMorphTargetInfluences>();
            if (!morphObj) return;
            const auto& influences = morphObj->morphTargetInfluences();

            const bool relative = geom.morphTargetsRelative;
            const size_t numTargets = morphPos.size();

            for (size_t t = 0; t < numTargets && t < influences.size(); ++t) {
                const float w = influences[t];
                if (w == 0.f) continue;

                auto* tAttr = dynamic_cast<TypedBufferAttribute<float>*>(morphPos[t].get());
                if (!tAttr || tAttr->count() != vtxCount) continue;
                const auto& tData = tAttr->array();

                if (relative) {
                    for (int v = 0; v < vtxCount; ++v) {
                        outPos[v * 3 + 0] += w * tData[v * 3 + 0];
                        outPos[v * 3 + 1] += w * tData[v * 3 + 1];
                        outPos[v * 3 + 2] += w * tData[v * 3 + 2];
                    }
                } else {
                    for (int v = 0; v < vtxCount; ++v) {
                        outPos[v * 3 + 0] += w * (tData[v * 3 + 0] - basePos[v * 3 + 0]);
                        outPos[v * 3 + 1] += w * (tData[v * 3 + 1] - basePos[v * 3 + 1]);
                        outPos[v * 3 + 2] += w * (tData[v * 3 + 2] - basePos[v * 3 + 2]);
                    }
                }

                if (morphNrm && t < morphNrm->size()) {
                    auto* nAttr = dynamic_cast<TypedBufferAttribute<float>*>((*morphNrm)[t].get());
                    if (nAttr && nAttr->count() == vtxCount) {
                        const auto& nData = nAttr->array();
                        if (relative) {
                            for (int v = 0; v < vtxCount; ++v) {
                                outNorm[v * 3 + 0] += w * nData[v * 3 + 0];
                                outNorm[v * 3 + 1] += w * nData[v * 3 + 1];
                                outNorm[v * 3 + 2] += w * nData[v * 3 + 2];
                            }
                        } else if (nrmAttr) {
                            const auto& baseNrm = nrmAttr->array();
                            for (int v = 0; v < vtxCount; ++v) {
                                outNorm[v * 3 + 0] += w * (nData[v * 3 + 0] - baseNrm[v * 3 + 0]);
                                outNorm[v * 3 + 1] += w * (nData[v * 3 + 1] - baseNrm[v * 3 + 1]);
                                outNorm[v * 3 + 2] += w * (nData[v * 3 + 2] - baseNrm[v * 3 + 2]);
                            }
                        }
                    }
                }
            }

            // Renormalize normals.
            for (int v = 0; v < vtxCount; ++v) {
                float& nx = outNorm[v * 3 + 0];
                float& ny = outNorm[v * 3 + 1];
                float& nz = outNorm[v * 3 + 2];
                const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (len > 0.f) { const float inv = 1.f / len; nx *= inv; ny *= inv; nz *= inv; }
            }
        }
}// namespace threepp
