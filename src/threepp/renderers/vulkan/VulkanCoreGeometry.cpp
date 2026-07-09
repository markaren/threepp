#include "VulkanCoreImpl.hpp"

namespace threepp {

std::unique_ptr<VulkanRendererCore::CoreImpl::BlasRecord> VulkanRendererCore::CoreImpl::buildBlasFor(const BufferGeometry& geom) {
            auto* posAttr = geom.getAttribute<float>("position");
            if (!posAttr) return nullptr;
            auto* normAttr = geom.getAttribute<float>("normal");
            if (!normAttr) return nullptr;// the RT path requires per-vertex normals
            const auto& positions = posAttr->array();
            const auto& normals = normAttr->array();
            const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
            if (vertexCount < 3) return nullptr;
            if (normAttr->count() != static_cast<int>(vertexCount)) return nullptr;

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
            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), rec->vertex.alloc, &mapped);
            std::memcpy(mapped, positions.data(), vbBytes);
            vmaUnmapMemory(ctx->allocator(), rec->vertex.alloc);

            const VkDeviceSize nbBytes = normals.size() * sizeof(float);
            rec->normal = createBuffer(
                    ctx->allocator(), ctx->device(), nbBytes,
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            vmaMapMemory(ctx->allocator(), rec->normal.alloc, &mapped);
            std::memcpy(mapped, normals.data(), nbBytes);
            vmaUnmapMemory(ctx->allocator(), rec->normal.alloc);

            // Optional UV attribute (TEXCOORD_0). closest_hit interpolates and
            // samples albedo with these; absent → bindless texture is ignored.
            if (auto* uvAttr = geom.getAttribute<float>("uv")) {
                const auto& uvs = uvAttr->array();
                if (uvAttr->count() == static_cast<int>(vertexCount) &&
                    uvs.size() == vertexCount * 2) {
                    const VkDeviceSize uvBytes = uvs.size() * sizeof(float);
                    rec->uv = createBuffer(
                            ctx->allocator(), ctx->device(), uvBytes,
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VMA_MEMORY_USAGE_AUTO,
                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                    vmaMapMemory(ctx->allocator(), rec->uv.alloc, &mapped);
                    std::memcpy(mapped, uvs.data(), uvBytes);
                    vmaUnmapMemory(ctx->allocator(), rec->uv.alloc);
                }
            }

            // Optional per-vertex color (material.vertexColors). closest_hit and
            // the raster gbuffer interpolate this and modulate albedo. Stored as
            // tightly-packed vec3 (the shaders fetch 3 floats/vertex); itemSize-4
            // colors are repacked to RGB, dropping the alpha. Whether it actually
            // applies is decided per-instance from the material's vertexColors
            // flag when GeometryDesc / DrawInfo are filled — the buffer is always
            // uploaded if present so a shared geometry works under either material.
            if (auto* colAttr = geom.getAttribute<float>("color")) {
                const int itemSize = colAttr->itemSize();
                if (colAttr->count() == static_cast<int>(vertexCount) &&
                    (itemSize == 3 || itemSize == 4)) {
                    const auto& cols = colAttr->array();
                    const VkDeviceSize cbBytes = static_cast<VkDeviceSize>(vertexCount) * 3 * sizeof(float);
                    rec->color = createBuffer(
                            ctx->allocator(), ctx->device(), cbBytes,
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VMA_MEMORY_USAGE_AUTO,
                            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
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
                    vmaUnmapMemory(ctx->allocator(), rec->color.alloc);
                }
            }

            if (indexed) {
                const auto& indices = idxAttr->array();
                const VkDeviceSize ibBytes = indices.size() * sizeof(unsigned int);
                rec->index = createBuffer(
                        ctx->allocator(), ctx->device(), ibBytes,
                        geomUsage, VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                vmaMapMemory(ctx->allocator(), rec->index.alloc, &mapped);
                std::memcpy(mapped, indices.data(), ibBytes);
                vmaUnmapMemory(ctx->allocator(), rec->index.alloc);
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
            destroyBuffer(ctx->allocator(), scratch);

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
            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), st.tetPos.alloc, &mapped);
            std::memcpy(mapped, tetImg.data(), bytes);
            vmaUnmapMemory(ctx->allocator(), st.tetPos.alloc);
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
            // index buffers. Each is host-coherent (HOST_ACCESS_SEQUENTIAL_WRITE
            // at buildBlasFor time); the implicit submit barrier on the next
            // phase makes these visible to the BLAS build.
            for (size_t k : liveOps) {
                const auto& geom = *ops[k].geom;
                auto& rec = *ops[k].rec;
                auto* posAttr = geom.getAttribute<float>("position");
                auto* nrmAttr = geom.getAttribute<float>("normal");

                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), rec.vertex.alloc, &mapped);
                std::memcpy(mapped, posAttr->array().data(),
                            posAttr->array().size() * sizeof(float));
                vmaUnmapMemory(ctx->allocator(), rec.vertex.alloc);

                vmaMapMemory(ctx->allocator(), rec.normal.alloc, &mapped);
                std::memcpy(mapped, nrmAttr->array().data(),
                            nrmAttr->array().size() * sizeof(float));
                vmaUnmapMemory(ctx->allocator(), rec.normal.alloc);

                if (auto* uvAttr = geom.getAttribute<float>("uv");
                    uvAttr && rec.uv.handle != VK_NULL_HANDLE) {
                    vmaMapMemory(ctx->allocator(), rec.uv.alloc, &mapped);
                    std::memcpy(mapped, uvAttr->array().data(),
                                uvAttr->array().size() * sizeof(float));
                    vmaUnmapMemory(ctx->allocator(), rec.uv.alloc);
                }

                if (auto* idxAttr = geom.getIndex();
                    idxAttr && rec.index.handle != VK_NULL_HANDLE) {
                    vmaMapMemory(ctx->allocator(), rec.index.alloc, &mapped);
                    std::memcpy(mapped, idxAttr->array().data(),
                                idxAttr->array().size() * sizeof(unsigned int));
                    vmaUnmapMemory(ctx->allocator(), rec.index.alloc);
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

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), st.blas->vertex.alloc, &mapped);
            std::memcpy(mapped, st.blendedPositions.data(),
                        st.blendedPositions.size() * sizeof(float));
            vmaUnmapMemory(ctx->allocator(), st.blas->vertex.alloc);

            vmaMapMemory(ctx->allocator(), st.blas->normal.alloc, &mapped);
            std::memcpy(mapped, st.blendedNormals.data(),
                        st.blendedNormals.size() * sizeof(float));
            vmaUnmapMemory(ctx->allocator(), st.blas->normal.alloc);

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
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), st.foamDisturbBuffer.alloc, &mapped);
                std::memcpy(mapped, dm.foamDisturbances.data(),
                            disturbCount * kFoamDisturbStride);
                vmaUnmapMemory(ctx->allocator(), st.foamDisturbBuffer.alloc);
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
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), st.wakeTrailBuffer.alloc, &mapped);
                std::memcpy(mapped, dm.wake.trail.data(),
                            wakeSampleCount * kWakeSampleStride);
                vmaUnmapMemory(ctx->allocator(), st.wakeTrailBuffer.alloc);
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
                    void* mapped = nullptr;
                    vmaMapMemory(ctx->allocator(), tlasInstancesBuffers[s].alloc, &mapped);
                    std::memcpy(mapped, instances.data(),
                                instanceCount * sizeof(VkAccelerationStructureInstanceKHR));
                    vmaUnmapMemory(ctx->allocator(), tlasInstancesBuffers[s].alloc);
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
            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), instBuf.alloc, &mapped);
            std::memcpy(mapped, instances.data(),
                        instanceCount * sizeof(VkAccelerationStructureInstanceKHR));
            vmaUnmapMemory(ctx->allocator(), instBuf.alloc);

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
            // ordered after the prior frame's by submit order, so reuse is safe.
            const VkDeviceSize need = std::max(sizes.buildScratchSize, sizes.updateScratchSize);
            if (tlasRefitScratch_.handle == VK_NULL_HANDLE || tlasRefitScratchSize_ < need) {
                if (tlasRefitScratch_.handle != VK_NULL_HANDLE)
                    destroyBuffer(ctx->allocator(), tlasRefitScratch_);
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
