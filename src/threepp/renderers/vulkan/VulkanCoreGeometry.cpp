#include "VulkanCoreImpl.hpp"

#include "threepp/core/AttributeView.hpp"

#include <algorithm>
#include <cmath>
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

}// namespace

namespace threepp {

std::unique_ptr<VulkanRendererCore::CoreImpl::BlasRecord> VulkanRendererCore::CoreImpl::buildBlasFor(const BufferGeometry& geom, bool allowPacked) {
            // Escape hatch for A/B triage: THREEPP_NO_PACK=1 forces every
            // attribute buffer back to tightly-packed float, same binary.
            static const bool noPack = std::getenv("THREEPP_NO_PACK") != nullptr;
            allowPacked = allowPacked && !noPack;

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
                    std::cerr << "[VulkanRenderer] buildBlasFor: skipping geometry — "
                              << "position[" << i << "] is non-finite ("
                              << positions[i] << "), vertexCount=" << vertexCount << '\n';
                    return nullptr;
                }
            }
            if (indexed) {
                const auto& indices = idxAttr->array();
                for (size_t i = 0; i < indices.size(); ++i) {
                    if (indices[i] >= vertexCount) {
                        std::cerr << "[VulkanRenderer] buildBlasFor: skipping geometry — "
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
            // to vkCmdCopyBuffer the current vertex into prev each frame.
            const VkBufferUsageFlags geomUsage =
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

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
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
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

            if (indexed) {
                const auto& indices = idxAttr->array();
                const VkDeviceSize ibBytes = indices.size() * sizeof(unsigned int);
                rec->index = createBuffer(
                        ctx->allocator(), ctx->device(), ibBytes,
                        geomUsage, VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                uploadHostVisible(ctx->allocator(), rec->index, indices.data(), ibBytes);
            }

            VkAccelerationStructureGeometryTrianglesDataKHR triData{};
            triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triData.vertexData.deviceAddress = rec->vertex.address;
            triData.vertexStride = 3 * sizeof(float);
            triData.maxVertex = vertexCount - 1;
            if (indexed) {
                triData.indexType = VK_INDEX_TYPE_UINT32;
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

            rec->storage = createBuffer(
                    ctx->allocator(), ctx->device(), blasSizes.accelerationStructureSize,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);

            VkAccelerationStructureCreateInfoKHR blasCreate{};
            blasCreate.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            blasCreate.buffer = rec->storage.handle;
            blasCreate.size = blasSizes.accelerationStructureSize;
            blasCreate.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            check(ctx->rt().createAccelerationStructure(ctx->device(), &blasCreate, nullptr, &rec->as),
                  "vkCreateAccelerationStructureKHR(BLAS)");

            Buffer scratch = createAsScratchBuffer(ctx->allocator(), ctx->device(), blasSizes.buildScratchSize);

            blasBuild.dstAccelerationStructure = rec->as;
            blasBuild.scratchData.deviceAddress = scratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = primitiveCount;
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
bool VulkanRendererCore::CoreImpl::buildLodLevelFor(BlasRecord& rec,
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

            const VkDeviceSize ibBytes = level.indices.size() * sizeof(uint32_t);
            out.index = createBuffer(
                    ctx->allocator(), ctx->device(), ibBytes,
                    idxUsage, VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            uploadHostVisible(ctx->allocator(), out.index, level.indices.data(), ibBytes);

            // Size query needs a fully-specified build info; a LOCAL struct is
            // fine here (only the final batched cmdBuildAccelerationStructures
            // needs pointer-stable storage — flushLodLevelBuilds rebuilds it).
            VkAccelerationStructureGeometryTrianglesDataKHR triData{};
            triData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            triData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            triData.vertexData.deviceAddress = rec.vertex.address;
            triData.vertexStride = 3 * sizeof(float);
            triData.maxVertex = rec.vertexCount - 1;
            triData.indexType = VK_INDEX_TYPE_UINT32;
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
            // Per-build scratch — concurrent builds recorded into one command
            // buffer must not alias scratch memory (same rule as
            // refreshGeomBlasBatch's per-record persistent scratch).
            build.scratch = createAsScratchBuffer(ctx->allocator(), ctx->device(), blasSizes.buildScratchSize);
            pending.push_back(build);

            out.indexCount  = static_cast<uint32_t>(level.indices.size());
            out.errorBound  = level.error;
            return true;
        }

void VulkanRendererCore::CoreImpl::flushLodLevelBuilds(std::vector<LodPendingBuild>& pending) {
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
                triDatas[k].indexType = VK_INDEX_TYPE_UINT32;
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
void VulkanRendererCore::CoreImpl::drainLodResults() {
            constexpr uint32_t kMaxGeomsPerFrame = 16;
            constexpr uint64_t kMaxNewBytesPerFrame = 8ull * 1024ull * 1024ull;

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
                    rec.lodState = BlasRecord::LodState::None;// stale — re-enqueue will pick up the new version
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
                            std::cerr << "[VulkanRenderer] auto-LOD: 256 MiB byte budget reached — "
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

void VulkanRendererCore::CoreImpl::refreshSkinnedBlas(SkinnedMesh& sm, SkinnedMeshState& st) {
            if (!st.blas || !sm.skeleton || st.boneCount == 0) return;

            // Recompute per-bone matrix = bones[b]->matrixWorld * boneInverse[b].
            // Mirrors what cpuSkin used to do on host; cheap (a few dozen
            // matrix multiplies). Upload into the bones[..] section of the
            // host-visible boneMatrices buffer (skipping the [bindMat, bindInv]
            // prefix written once in ensureSkinnedBlas).
            const auto& skel = *sm.skeleton;
            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), st.boneMatrices.alloc, &mapped);
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
            for (uint32_t b = 0; b < st.boneCount; ++b) {
                Matrix4 m;
                if (b < skel.bones.size() && skel.bones[b]) {
                    m.multiplyMatrices(*skel.bones[b]->matrixWorld, skel.boneInverses[b]);
                }
                std::memcpy(dst + b * 16 * sizeof(float),
                            m.elements.data(), 16 * sizeof(float));
            }
            flushHostWrites(ctx->allocator(), st.boneMatrices.alloc);
            vmaUnmapMemory(ctx->allocator(), st.boneMatrices.alloc);

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

void VulkanRendererCore::CoreImpl::refreshTetBlas(Mesh& m, TetMeshState& st) {
            if (!st.blas) return;
            // Zero-copy interop: the registered CUDA device→device copy writes the
            // deformed tet positions straight into the exported binding-6 buffer —
            // no host readback, no DataTexture, no map/memcpy. Runs at the SAME
            // frame point as the CPU upload below, so the (pre-existing, benign)
            // overlap with a still-in-flight prior frame's dispatch is unchanged.
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
            uploadHostVisible(ctx->allocator(), st.tetPos, tetImg.data(), bytes);
            pendingTetRebuilds_.push_back(&st);
        }

void VulkanRendererCore::CoreImpl::rewriteTetPosBinding(TetMeshState& st, VkBuffer buf) {
            VkDescriptorBufferInfo bi{};
            bi.buffer = buf;
            bi.offset = 0;
            bi.range  = VK_WHOLE_SIZE;
            VkWriteDescriptorSet wr{};
            wr.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr.dstSet          = st.tetDescSet;
            wr.dstBinding      = 6;
            wr.descriptorCount = 1;
            wr.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr.pBufferInfo     = &bi;
            vkUpdateDescriptorSets(ctx->device(), 1, &wr, 0, nullptr);
        }

void VulkanRendererCore::CoreImpl::disableSoftBodyInterop(const Mesh& mesh) {
            auto it = tetMeshStates.find(&mesh);
            if (it == tetMeshStates.end()) return;
            auto& st = *it->second;
            if (st.tetPosExt.handle == VK_NULL_HANDLE) return;
            check(vkDeviceWaitIdle(ctx->device()), "vkDeviceWaitIdle (softbody interop disable)");
            st.tetPos = createBuffer(
                    ctx->allocator(), ctx->device(), st.tetPosBytes,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            rewriteTetPosBinding(st, st.tetPos.handle);
            vulkan::destroyExternalBuffer(ctx->device(), st.tetPosExt);
            st.tetPosExternalCopy = nullptr;
        }

void VulkanRendererCore::CoreImpl::refreshGeomBlasBatch(const std::vector<VulkanRendererCore::CoreImpl::GeomRefreshOp>& ops) {
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
                        std::cerr << "[VulkanRenderer] refreshGeomBlasBatch: skipping geom — "
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
            for (size_t k : liveOps) {
                const auto& geom = *ops[k].geom;
                auto& rec = *ops[k].rec;
                auto* posAttr = geom.getAttribute<float>("position");
                auto* nrmAttr = geom.getAttribute<float>("normal");

                uploadHostVisible(ctx->allocator(), rec.vertex, posAttr->array().data(),
                                  posAttr->array().size() * sizeof(float));

                // A STATIC geometry whose attributes were edited (needsUpdate)
                // also lands here, and its buffers may hold PACKED data — the
                // re-upload must match the buffer's format or it overflows the
                // smaller packed allocation.
                const uint32_t vtxCount = static_cast<uint32_t>(posAttr->count());
                if (rec.packedMask & 1u) {
                    const auto& nrm = nrmAttr->array();
                    std::vector<uint32_t> packed(vtxCount);
                    for (uint32_t v = 0; v < vtxCount; ++v) {
                        const auto [ox, oy] = octEncode(nrm[v * 3 + 0], nrm[v * 3 + 1], nrm[v * 3 + 2]);
                        packed[v] = packSnorm2x16(ox, oy);
                    }
                    uploadHostVisible(ctx->allocator(), rec.normal, packed.data(),
                                      packed.size() * sizeof(uint32_t));
                } else {
                    uploadHostVisible(ctx->allocator(), rec.normal, nrmAttr->array().data(),
                                      nrmAttr->array().size() * sizeof(float));
                }

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
                    uploadHostVisible(ctx->allocator(), rec.index, idxAttr->array().data(),
                                      idxAttr->array().size() * sizeof(unsigned int));
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
                const uint32_t primitiveCount = indexed
                        ? static_cast<uint32_t>(idxAttr->count() / 3)
                        : vertexCount / 3;

                triDatas[kk].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
                triDatas[kk].vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
                triDatas[kk].vertexData.deviceAddress = rec.vertex.address;
                triDatas[kk].vertexStride = 3 * sizeof(float);
                triDatas[kk].maxVertex = vertexCount - 1;
                if (indexed) {
                    triDatas[kk].indexType = VK_INDEX_TYPE_UINT32;
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
                // tree away from optimal.
                const bool fullRebuild =
                        rec.blasRefitCounter >= BlasRecord::kBlasFullRebuildInterval;
                rec.blasRefitCounter = fullRebuild ? 0u : (rec.blasRefitCounter + 1u);

                blasBuilds[kk].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
                blasBuilds[kk].type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                blasBuilds[kk].flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                                       VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
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

void VulkanRendererCore::CoreImpl::refreshMorphedBlas(Mesh& mesh, MorphedMeshState& st) {
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

void VulkanRendererCore::CoreImpl::recordDisplacedDeform(VkCommandBuffer cb, DisplacedMesh& dm, DisplacedMeshState& st, float elapsedSeconds) {

            // (1)..(3) Run each enabled cascade's FFT chain in turn. Phillips
            // is one-shot per cascade. DynamicSpectrum re-runs each frame.
            // IFFT calls are sequential on the same queue so they can share
            // the single scratch image. Cascades dispatch back-to-back; the
            // Vulkan command buffer recording order plus the IFFT's internal
            // image-layout barriers serialize the work correctly.
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
                st.scratchA.currentLayout = VK_IMAGE_LAYOUT_GENERAL;
                c.ifft->recordApply(cb, ht,  st.scratchA);
                c.ifft->recordApply(cb, dsp, st.scratchA);

                // Copy the spatial-domain height image into the host-mapped
                // readback buffer for this cascade. By the time
                // endAndSubmitOneShot returns, the buffer is filled.
                Buffer* rb = (i == 0) ? &st.heightReadback
                           : (i == 1) ? &st.heightReadback1
                                      : &st.heightReadback2;
                if (rb->handle != VK_NULL_HANDLE) {
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
            }
            const uint32_t disturbCount = static_cast<uint32_t>(std::min<size_t>(
                    dm.foamDisturbances.size(),
                    DisplacedMeshState::kMaxFoamDisturbances));
            if (disturbCount > 0u) {
                uploadHostVisible(ctx->allocator(), st.foamDisturbBuffer,
                                  dm.foamDisturbances.data(),
                                  disturbCount * kFoamDisturbStride);
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
            pc.disturbAddr  = st.foamDisturbBuffer.address;
            pc.vertexCount  = st.vertexCount;
            pc.gridDim      = st.gridDim;
            pc.planeSize    = st.planeSize;
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
            pc.disturbCount   = disturbCount;
            pc.warpCenterX    = dm.warp.centerX;
            pc.warpCenterZ    = dm.warp.centerZ;
            pc.warpHalfRange  = dm.warp.halfRange;
            pc.warpCoefA      = dm.warp.coefA;
            pc.wakeTrailAddr  = st.wakeTrailBuffer.address;
            pc.wakeTrailCount = wakeSampleCount;
            waterDisplace_->recordDispatch(cb, st.displaceDS, pc);

            // (4c) World-space foam pass — same inputs as water_displace
            // (cascades, disturbances, hull/wake state) but evaluated per
            // foam-texel rather than per mesh vertex. Replaces the per-
            // vertex foam buffer that water_displace used to write. Run
            // AFTER water_displace finishes so we share the descriptor
            // pool's cascade-image bindings without a layout flip; the
            // cascades stay in GENERAL throughout.
            {
                // Wall-clock foam persistence. The old fixed 0.992/frame tied
                // the foam half-life to frame rate (≈1.4 s at 60 fps, half
                // that at 120) — same bug class as the TAA temporal constants.
                // τ = 2 s reproduces the old look at 60 fps. dt clamped so an
                // alt-tab / loading stall can't wipe the accumulator in one
                // frame.
                const double nowSec = glfwGetTime();
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
                fpc._pad           = 0;
                foamWorld_->recordDispatch(cb, st.foamWorldDS, fpc);

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
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &build, &pRange);
        }

void VulkanRendererCore::CoreImpl::mirrorDisplacedHeightfields(DisplacedMesh& dm, DisplacedMeshState& st) {
            struct { Buffer* buf; float tileSize; } cascades[] = {
                {&st.heightReadback,  dm.params.tileSize0},
                {&st.heightReadback1, dm.params.tileSize1},
                {&st.heightReadback2, dm.params.tileSize2},
            };
            for (int ci = 0; ci < 3; ++ci) {
                auto& cf = dm.heightFields[ci];
                const uint32_t dim = st.heightReadbackDim[ci];
                if (cascades[ci].buf->handle != VK_NULL_HANDLE && dim > 0) {
                    const size_t cells = size_t(dim) * size_t(dim);
                    const size_t bytes = cells * 2 * sizeof(float);
                    if (cf.data.size() != cells * 2)
                        cf.data.assign(cells * 2, 0.f);
                    void* mapped = nullptr;
                    vmaMapMemory(ctx->allocator(), cascades[ci].buf->alloc, &mapped);
                    invalidateHostReads(ctx->allocator(), cascades[ci].buf->alloc, 0, bytes);
                    std::memcpy(cf.data.data(), mapped, bytes);
                    vmaUnmapMemory(ctx->allocator(), cascades[ci].buf->alloc);
                    cf.dim      = dim;
                    cf.tileSize = cascades[ci].tileSize;
                }
            }
        }

void VulkanRendererCore::CoreImpl::refreshDisplacedBlas(DisplacedMesh& dm, DisplacedMeshState& st, float elapsedSeconds) {
            VkCommandBuffer cb = beginOneShot();
            recordDisplacedDeform(cb, dm, st, elapsedSeconds);
            endAndSubmitOneShot(cb);
            mirrorDisplacedHeightfields(dm, st);
        }

void VulkanRendererCore::CoreImpl::recordGrassDeform(VkCommandBuffer cb, GrassMesh& gm, GrassMeshState& st) {
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

void VulkanRendererCore::CoreImpl::refreshGrassBlas(GrassMesh& gm, GrassMeshState& st, float /*elapsedSeconds*/) {
            VkCommandBuffer cb = beginOneShot();
            recordGrassDeform(cb, gm, st);
            endAndSubmitOneShot(cb, "grass prime");
        }

void VulkanRendererCore::CoreImpl::buildTlas(const std::vector<VkAccelerationStructureInstanceKHR>& instances) {
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
            endAndSubmitOneShot(cb, "buildTlas");
            destroyBuffer(ctx->allocator(), scratch);
        }

void VulkanRendererCore::CoreImpl::recordTlasRefit(VkCommandBuffer cb,
                             const std::vector<VkAccelerationStructureInstanceKHR>& instances,
                             bool fullBuild) {
            const uint32_t instanceCount = static_cast<uint32_t>(instances.size());
            if (instanceCount == 0 || tlas == VK_NULL_HANDLE) return;

            Buffer& instBuf = tlasInstancesBuffers[currentFrame];
            uploadHostVisible(ctx->allocator(), instBuf, instances.data(),
                              instanceCount * sizeof(VkAccelerationStructureInstanceKHR));

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

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = instanceCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &tlasBuild, &pRange);
        }

}// namespace threepp
