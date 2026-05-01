// VulkanRenderer — Phase 4b (scene ingestion: per-Mesh BLAS, multi-instance TLAS).
//
// On the first render() call we walk the scene, build one BLAS per unique
// BufferGeometry, then build a TLAS containing one instance per Mesh whose
// transform is its world matrix. Subsequent frames reuse the AS — for now
// the scene is captured statically; dynamic mutations / animation are not
// re-ingested (a later phase will refit / rebuild on demand). Phase 5 adds
// shading (Lambert + GGX); for now closest-hit still returns barycentrics.

#define VMA_IMPLEMENTATION
#include "vulkan/VulkanContext.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/canvas/Canvas.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/renderers/VulkanRenderer.hpp"

#include "threepp/renderers/vulkan/shaders/raygen.rgen.spv.h"
#include "threepp/renderers/vulkan/shaders/miss.rmiss.spv.h"
#include "threepp/renderers/vulkan/shaders/closest_hit.rchit.spv.h"

#include <GLFW/glfw3.h>

#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp {

    using vulkan::VulkanContext;

    namespace {
        constexpr uint32_t kFramesInFlight = 2;

        void check(VkResult r, const char* what) {
            if (r != VK_SUCCESS) {
                throw std::runtime_error(std::string("[VulkanRenderer] ") + what + " failed: " + std::to_string(r));
            }
        }

        // Round x up to the nearest multiple of `align` (align must be POT).
        uint32_t alignUp(uint32_t x, uint32_t align) {
            return (x + align - 1) & ~(align - 1);
        }

        struct Buffer {
            VkBuffer       handle = VK_NULL_HANDLE;
            VmaAllocation  alloc  = VK_NULL_HANDLE;
            VkDeviceSize   size   = 0;
            VkDeviceAddress address = 0;
        };

        Buffer createBuffer(VmaAllocator alloc, VkDevice device,
                            VkDeviceSize size, VkBufferUsageFlags usage,
                            VmaMemoryUsage memoryUsage,
                            VmaAllocationCreateFlags flags = 0) {
            Buffer b{};
            b.size = size;

            VkBufferCreateInfo bci{};
            bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bci.size = size;
            bci.usage = usage;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo aci{};
            aci.usage = memoryUsage;
            aci.flags = flags;

            check(vmaCreateBuffer(alloc, &bci, &aci, &b.handle, &b.alloc, nullptr),
                  "vmaCreateBuffer");

            if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
                VkBufferDeviceAddressInfo dai{};
                dai.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
                dai.buffer = b.handle;
                b.address = vkGetBufferDeviceAddress(device, &dai);
            }
            return b;
        }

        void destroyBuffer(VmaAllocator alloc, Buffer& b) {
            if (b.handle) vmaDestroyBuffer(alloc, b.handle, b.alloc);
            b = {};
        }
    }// namespace

    struct VulkanRenderer::Impl {
        Canvas& canvas;
        WindowSize size;
        float pixelRatio = 1.f;
        Color clearColor{0.f, 0.f, 0.f};
        float clearAlpha = 1.f;
        Vector4 viewport;
        Vector4 scissor;
        bool scissorTest = false;

        std::unique_ptr<VulkanContext> ctx;

        // Per-geometry BLAS + the buffers that back its build inputs. Vertex /
        // index buffers are kept alive past the build so later phases can
        // sample them in the closest-hit shader (normals, UVs, ...).
        struct BlasRecord {
            VkAccelerationStructureKHR as = VK_NULL_HANDLE;
            Buffer storage;
            Buffer vertex;
            Buffer index;// .handle == VK_NULL_HANDLE for non-indexed geometry
            VkDeviceAddress address = 0;
        };
        std::unordered_map<const BufferGeometry*, std::unique_ptr<BlasRecord>> blasCache;

        // Single TLAS over all mesh instances in the scene.
        VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
        Buffer tlasBuffer;
        Buffer tlasInstancesBuffer;
        bool sceneBuilt_ = false;

        // Ray-tracing pipeline.
        VkDescriptorSetLayout rtDsLayout = VK_NULL_HANDLE;
        VkPipelineLayout      rtPipelineLayout = VK_NULL_HANDLE;
        VkPipeline            rtPipeline = VK_NULL_HANDLE;

        // Shader Binding Table (one record per group: rgen / miss / hit).
        Buffer sbtBuffer;
        VkStridedDeviceAddressRegionKHR rgenRegion{};
        VkStridedDeviceAddressRegionKHR missRegion{};
        VkStridedDeviceAddressRegionKHR hitRegion{};
        VkStridedDeviceAddressRegionKHR callRegion{};// unused

        // Per-frame-in-flight camera UBO (viewInverse + projInverse).
        // 2 mat4 packed back-to-back, std140 layout.
        std::array<Buffer, kFramesInFlight> cameraUbos{};

        // Descriptor pool + sets indexed by [frame * imageCount + image] so
        // each set can reference both a per-frame UBO and a per-image view.
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> descriptorSets;
        uint32_t imageCount_ = 0;

        // Per-frame command resources.
        VkCommandPool                                cmdPool = VK_NULL_HANDLE;
        std::array<VkCommandBuffer, kFramesInFlight> cmdBuffers{};
        std::array<VkSemaphore,     kFramesInFlight> imageAvailable{};
        std::array<VkSemaphore,     kFramesInFlight> renderFinished{};
        std::array<VkFence,         kFramesInFlight> inFlight{};

        uint32_t currentFrame = 0;
        bool needsResize = false;

        explicit Impl(Canvas& c) : canvas(c), size(c.size()) {
            ctx = std::make_unique<VulkanContext>(
                    static_cast<GLFWwindow*>(canvas.windowPtr()),
                    /*enableRayTracing*/ true);

            // The scene-dependent AS build runs lazily on the first render()
            // call. Everything below is scene-independent and safe at ctor time.
            createCommandResources();
            createCameraUbos();
            createRtPipeline();
            createShaderBindingTable();
            createDescriptorPool();
        }

        ~Impl() {
            if (!ctx) return;
            VkDevice d = ctx->device();
            vkDeviceWaitIdle(d);

            for (auto s : imageAvailable) if (s) vkDestroySemaphore(d, s, nullptr);
            for (auto s : renderFinished) if (s) vkDestroySemaphore(d, s, nullptr);
            for (auto f : inFlight) if (f) vkDestroyFence(d, f, nullptr);
            if (cmdPool) vkDestroyCommandPool(d, cmdPool, nullptr);

            if (descriptorPool) vkDestroyDescriptorPool(d, descriptorPool, nullptr);

            destroyBuffer(ctx->allocator(), sbtBuffer);
            if (rtPipeline)       vkDestroyPipeline(d, rtPipeline, nullptr);
            if (rtPipelineLayout) vkDestroyPipelineLayout(d, rtPipelineLayout, nullptr);
            if (rtDsLayout)       vkDestroyDescriptorSetLayout(d, rtDsLayout, nullptr);

            if (tlas) ctx->rt().destroyAccelerationStructure(d, tlas, nullptr);
            destroyBuffer(ctx->allocator(), tlasBuffer);
            destroyBuffer(ctx->allocator(), tlasInstancesBuffer);

            for (auto& [_, rec] : blasCache) {
                if (rec->as) ctx->rt().destroyAccelerationStructure(d, rec->as, nullptr);
                destroyBuffer(ctx->allocator(), rec->storage);
                destroyBuffer(ctx->allocator(), rec->vertex);
                destroyBuffer(ctx->allocator(), rec->index);
            }
            blasCache.clear();

            for (auto& b : cameraUbos) destroyBuffer(ctx->allocator(), b);
        }

        void createCommandResources() {
            VkCommandPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pci.queueFamilyIndex = ctx->queueFamilies().graphics;
            check(vkCreateCommandPool(ctx->device(), &pci, nullptr, &cmdPool),
                  "vkCreateCommandPool");

            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = cmdPool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = kFramesInFlight;
            check(vkAllocateCommandBuffers(ctx->device(), &ai, cmdBuffers.data()),
                  "vkAllocateCommandBuffers");

            VkSemaphoreCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            VkFenceCreateInfo fci{};
            fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            for (uint32_t i = 0; i < kFramesInFlight; ++i) {
                check(vkCreateSemaphore(ctx->device(), &sci, nullptr, &imageAvailable[i]), "vkCreateSemaphore A");
                check(vkCreateSemaphore(ctx->device(), &sci, nullptr, &renderFinished[i]), "vkCreateSemaphore B");
                check(vkCreateFence(ctx->device(), &fci, nullptr, &inFlight[i]), "vkCreateFence");
            }
        }

        // Allocate, begin, return a one-shot command buffer.
        VkCommandBuffer beginOneShot() {
            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = cmdPool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            VkCommandBuffer cb = VK_NULL_HANDLE;
            check(vkAllocateCommandBuffers(ctx->device(), &ai, &cb), "alloc one-shot cb");

            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(vkBeginCommandBuffer(cb, &bi), "begin one-shot cb");
            return cb;
        }

        void endAndSubmitOneShot(VkCommandBuffer cb) {
            check(vkEndCommandBuffer(cb), "end one-shot cb");
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cb;
            check(vkQueueSubmit(ctx->graphicsQueue(), 1, &si, VK_NULL_HANDLE), "submit one-shot");
            check(vkQueueWaitIdle(ctx->graphicsQueue()), "wait one-shot");
            vkFreeCommandBuffers(ctx->device(), cmdPool, 1, &cb);
        }

        // Build a single BLAS for the given geometry. Vertex / index buffers
        // are uploaded host-mapped, then the AS is built into freshly allocated
        // device storage. The temporary scratch buffer is destroyed on exit.
        std::unique_ptr<BlasRecord> buildBlasFor(const BufferGeometry& geom) {
            auto* posAttr = geom.getAttribute<float>("position");
            if (!posAttr) return nullptr;
            const auto& positions = posAttr->array();
            const uint32_t vertexCount = static_cast<uint32_t>(posAttr->count());
            if (vertexCount < 3) return nullptr;

            const auto* idxAttr = geom.getIndex();
            const bool indexed = idxAttr != nullptr;
            const uint32_t primitiveCount = indexed
                    ? static_cast<uint32_t>(idxAttr->count() / 3)
                    : vertexCount / 3;
            if (primitiveCount == 0) return nullptr;

            const VkBufferUsageFlags geomUsage =
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

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
            blasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

            VkAccelerationStructureBuildGeometryInfoKHR blasBuild{};
            blasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            blasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
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

            Buffer scratch = createBuffer(
                    ctx->allocator(), ctx->device(), blasSizes.buildScratchSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);

            blasBuild.dstAccelerationStructure = rec->as;
            blasBuild.scratchData.deviceAddress = scratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = primitiveCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

            VkCommandBuffer cb = beginOneShot();
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &blasBuild, &pRange);
            endAndSubmitOneShot(cb);
            destroyBuffer(ctx->allocator(), scratch);

            VkAccelerationStructureDeviceAddressInfoKHR addrInfo{};
            addrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
            addrInfo.accelerationStructure = rec->as;
            rec->address = ctx->rt().getAccelerationStructureDeviceAddress(ctx->device(), &addrInfo);

            return rec;
        }

        // Build a TLAS over the supplied instance descriptors. Empty input is
        // legal — produces an empty TLAS that always misses (Phase 2 fallback).
        void buildTlas(const std::vector<VkAccelerationStructureInstanceKHR>& instances) {
            const uint32_t instanceCount = static_cast<uint32_t>(instances.size());
            const VkDeviceSize instBytes = std::max<VkDeviceSize>(
                    instanceCount * sizeof(VkAccelerationStructureInstanceKHR),
                    sizeof(VkAccelerationStructureInstanceKHR));// keep buf non-empty

            tlasInstancesBuffer = createBuffer(
                    ctx->allocator(), ctx->device(), instBytes,
                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            if (instanceCount > 0) {
                void* mapped = nullptr;
                vmaMapMemory(ctx->allocator(), tlasInstancesBuffer.alloc, &mapped);
                std::memcpy(mapped, instances.data(),
                            instanceCount * sizeof(VkAccelerationStructureInstanceKHR));
                vmaUnmapMemory(ctx->allocator(), tlasInstancesBuffer.alloc);
            }

            VkAccelerationStructureGeometryInstancesDataKHR instData{};
            instData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
            instData.arrayOfPointers = VK_FALSE;
            instData.data.deviceAddress = tlasInstancesBuffer.address;

            VkAccelerationStructureGeometryKHR tlasGeom{};
            tlasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
            tlasGeom.geometry.instances = instData;
            tlasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

            VkAccelerationStructureBuildGeometryInfoKHR tlasBuild{};
            tlasBuild.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            tlasBuild.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
            tlasBuild.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
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

            Buffer scratch = createBuffer(
                    ctx->allocator(), ctx->device(), tlasSizes.buildScratchSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO);

            tlasBuild.dstAccelerationStructure = tlas;
            tlasBuild.scratchData.deviceAddress = scratch.address;

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = instanceCount;
            const VkAccelerationStructureBuildRangeInfoKHR* pRange = &range;

            VkCommandBuffer cb = beginOneShot();
            ctx->rt().cmdBuildAccelerationStructures(cb, 1, &tlasBuild, &pRange);
            endAndSubmitOneShot(cb);
            destroyBuffer(ctx->allocator(), scratch);
        }

        // Walk the scene once, build per-geometry BLASes, then a single TLAS
        // with one instance per Mesh whose transform is its world matrix.
        // Phase 4b is static: subsequent scene mutations are not re-ingested.
        void ensureSceneBuilt(Object3D& scene) {
            if (sceneBuilt_) return;

            scene.updateMatrixWorld(true);

            std::vector<Mesh*> meshes;
            scene.traverse([&](Object3D& o) {
                auto* m = dynamic_cast<Mesh*>(&o);
                if (!m || !m->visible) return;
                auto geom = m->geometry();
                if (!geom || !geom->hasAttribute("position")) return;
                meshes.push_back(m);
            });

            std::vector<VkAccelerationStructureInstanceKHR> instances;
            instances.reserve(meshes.size());

            for (size_t i = 0; i < meshes.size(); ++i) {
                Mesh* m = meshes[i];
                const BufferGeometry* geomKey = m->geometry().get();

                auto it = blasCache.find(geomKey);
                if (it == blasCache.end()) {
                    auto rec = buildBlasFor(*m->geometry());
                    if (!rec) continue;// degenerate / empty geometry
                    it = blasCache.emplace(geomKey, std::move(rec)).first;
                }

                VkAccelerationStructureInstanceKHR inst{};
                // VkTransformMatrixKHR is row-major 3x4; threepp Matrix4 is
                // column-major 4x4 (elements[c*4 + r]).
                const auto& e = m->matrixWorld->elements;
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        inst.transform.matrix[r][c] = e[c * 4 + r];
                    }
                }
                inst.instanceCustomIndex = static_cast<uint32_t>(i);
                inst.mask = 0xFFu;
                inst.instanceShaderBindingTableRecordOffset = 0;
                inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                inst.accelerationStructureReference = it->second->address;
                instances.push_back(inst);
            }

            buildTlas(instances);
            allocateAndUpdateDescriptors();
            sceneBuilt_ = true;
        }

        void createCameraUbos() {
            for (auto& b : cameraUbos) {
                b = createBuffer(
                        ctx->allocator(), ctx->device(),
                        /*size*/ 2 * 16 * sizeof(float),// viewInverse + projInverse
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VMA_MEMORY_USAGE_AUTO,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
            }
        }

        void updateCameraUbo(uint32_t frame, Camera& camera) {
            camera.updateMatrixWorld(true);

            float data[32];
            std::memcpy(data + 0,  camera.matrixWorld->elements.data(),            64);
            std::memcpy(data + 16, camera.projectionMatrixInverse.elements.data(), 64);

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), cameraUbos[frame].alloc, &mapped);
            std::memcpy(mapped, data, sizeof(data));
            vmaUnmapMemory(ctx->allocator(), cameraUbos[frame].alloc);
        }

        void createRtPipeline() {
            // Descriptor set layout: 0=TLAS, 1=storage image, 2=camera UBO.
            std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
            bindings[0].binding = 0;
            bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            bindings[0].descriptorCount = 1;
            bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            bindings[1].binding = 1;
            bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            bindings[1].descriptorCount = 1;
            bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            bindings[2].binding = 2;
            bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            bindings[2].descriptorCount = 1;
            bindings[2].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

            VkDescriptorSetLayoutCreateInfo dlci{};
            dlci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            dlci.bindingCount = static_cast<uint32_t>(bindings.size());
            dlci.pBindings = bindings.data();
            check(vkCreateDescriptorSetLayout(ctx->device(), &dlci, nullptr, &rtDsLayout),
                  "vkCreateDescriptorSetLayout(RT)");

            VkPipelineLayoutCreateInfo plci{};
            plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount = 1;
            plci.pSetLayouts = &rtDsLayout;
            check(vkCreatePipelineLayout(ctx->device(), &plci, nullptr, &rtPipelineLayout),
                  "vkCreatePipelineLayout(RT)");

            auto loadModule = [this](const uint32_t* code, size_t size) {
                VkShaderModuleCreateInfo smci{};
                smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
                smci.codeSize = size;
                smci.pCode = code;
                VkShaderModule m = VK_NULL_HANDLE;
                check(vkCreateShaderModule(ctx->device(), &smci, nullptr, &m),
                      "vkCreateShaderModule(RT)");
                return m;
            };

            VkShaderModule rgenMod = loadModule(kRaygenRgenSpv, sizeof(kRaygenRgenSpv));
            VkShaderModule missMod = loadModule(kMissRmissSpv, sizeof(kMissRmissSpv));
            VkShaderModule chitMod = loadModule(kClosestHitRchitSpv, sizeof(kClosestHitRchitSpv));

            std::array<VkPipelineShaderStageCreateInfo, 3> stages{};
            for (auto& s : stages) {
                s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                s.pName = "main";
            }
            stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            stages[0].module = rgenMod;
            stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
            stages[1].module = missMod;
            stages[2].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            stages[2].module = chitMod;

            std::array<VkRayTracingShaderGroupCreateInfoKHR, 3> groups{};
            for (auto& g : groups) {
                g.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
                g.generalShader = VK_SHADER_UNUSED_KHR;
                g.closestHitShader = VK_SHADER_UNUSED_KHR;
                g.anyHitShader = VK_SHADER_UNUSED_KHR;
                g.intersectionShader = VK_SHADER_UNUSED_KHR;
            }
            groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            groups[0].generalShader = 0;
            groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
            groups[1].generalShader = 1;
            groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
            groups[2].closestHitShader = 2;

            VkRayTracingPipelineCreateInfoKHR rci{};
            rci.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
            rci.stageCount = static_cast<uint32_t>(stages.size());
            rci.pStages = stages.data();
            rci.groupCount = static_cast<uint32_t>(groups.size());
            rci.pGroups = groups.data();
            rci.maxPipelineRayRecursionDepth = 1;
            rci.layout = rtPipelineLayout;

            check(ctx->rt().createRayTracingPipelines(
                          ctx->device(), VK_NULL_HANDLE, VK_NULL_HANDLE,
                          1, &rci, nullptr, &rtPipeline),
                  "vkCreateRayTracingPipelinesKHR");

            vkDestroyShaderModule(ctx->device(), rgenMod, nullptr);
            vkDestroyShaderModule(ctx->device(), missMod, nullptr);
            vkDestroyShaderModule(ctx->device(), chitMod, nullptr);
        }

        void createShaderBindingTable() {
            const auto& props = ctx->rtPipelineProperties();
            const uint32_t handleSize = props.shaderGroupHandleSize;
            const uint32_t handleAlignment = props.shaderGroupHandleAlignment;
            const uint32_t baseAlignment = props.shaderGroupBaseAlignment;
            const uint32_t handleSizeAligned = alignUp(handleSize, handleAlignment);

            // 3 groups (rgen, miss, hit), 1 record per group.
            constexpr uint32_t groupCount = 3;
            const uint32_t handlesDataSize = groupCount * handleSize;
            std::vector<uint8_t> handles(handlesDataSize);
            check(ctx->rt().getRayTracingShaderGroupHandles(
                          ctx->device(), rtPipeline, 0, groupCount,
                          handlesDataSize, handles.data()),
                  "vkGetRayTracingShaderGroupHandlesKHR");

            // SBT layout: 1 record per region, region base aligned to
            // shaderGroupBaseAlignment, stride = aligned handle size.
            const uint32_t regionStride = alignUp(handleSizeAligned, baseAlignment);
            const VkDeviceSize sbtSize = static_cast<VkDeviceSize>(regionStride) * 3;

            sbtBuffer = createBuffer(
                    ctx->allocator(), ctx->device(), sbtSize,
                    VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                            VMA_ALLOCATION_CREATE_MAPPED_BIT);

            void* mapped = nullptr;
            vmaMapMemory(ctx->allocator(), sbtBuffer.alloc, &mapped);
            std::memset(mapped, 0, sbtSize);
            uint8_t* dst = static_cast<uint8_t*>(mapped);
            for (uint32_t i = 0; i < groupCount; ++i) {
                std::memcpy(dst + i * regionStride, handles.data() + i * handleSize, handleSize);
            }
            vmaUnmapMemory(ctx->allocator(), sbtBuffer.alloc);

            const VkDeviceAddress base = sbtBuffer.address;
            rgenRegion.deviceAddress = base + 0 * regionStride;
            rgenRegion.stride = regionStride;
            rgenRegion.size   = regionStride;
            missRegion.deviceAddress = base + 1 * regionStride;
            missRegion.stride = regionStride;
            missRegion.size   = regionStride;
            hitRegion.deviceAddress = base + 2 * regionStride;
            hitRegion.stride = regionStride;
            hitRegion.size   = regionStride;
            callRegion = {};
        }

        void createDescriptorPool() {
            imageCount_ = static_cast<uint32_t>(ctx->swapchainImages().size());
            const uint32_t totalSets = imageCount_ * kFramesInFlight;
            std::array<VkDescriptorPoolSize, 3> ps{};
            ps[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            ps[0].descriptorCount = totalSets;
            ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            ps[1].descriptorCount = totalSets;
            ps[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            ps[2].descriptorCount = totalSets;

            VkDescriptorPoolCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            ci.maxSets = totalSets;
            ci.poolSizeCount = static_cast<uint32_t>(ps.size());
            ci.pPoolSizes = ps.data();
            check(vkCreateDescriptorPool(ctx->device(), &ci, nullptr, &descriptorPool),
                  "vkCreateDescriptorPool(RT)");
        }

        void allocateAndUpdateDescriptors() {
            const uint32_t totalSets = imageCount_ * kFramesInFlight;
            std::vector<VkDescriptorSetLayout> layouts(totalSets, rtDsLayout);
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = descriptorPool;
            ai.descriptorSetCount = totalSets;
            ai.pSetLayouts = layouts.data();
            descriptorSets.resize(totalSets);
            check(vkAllocateDescriptorSets(ctx->device(), &ai, descriptorSets.data()),
                  "vkAllocateDescriptorSets(RT)");

            VkWriteDescriptorSetAccelerationStructureKHR asWrite{};
            asWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
            asWrite.accelerationStructureCount = 1;
            asWrite.pAccelerationStructures = &tlas;

            for (uint32_t f = 0; f < kFramesInFlight; ++f) {
                for (uint32_t i = 0; i < imageCount_; ++i) {
                    const uint32_t idx = f * imageCount_ + i;

                    VkWriteDescriptorSet wAS{};
                    wAS.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wAS.pNext = &asWrite;
                    wAS.dstSet = descriptorSets[idx];
                    wAS.dstBinding = 0;
                    wAS.descriptorCount = 1;
                    wAS.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

                    VkDescriptorImageInfo imgInfo{};
                    imgInfo.imageView = ctx->swapchainImageViews()[i];
                    imgInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    VkWriteDescriptorSet wImg{};
                    wImg.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wImg.dstSet = descriptorSets[idx];
                    wImg.dstBinding = 1;
                    wImg.descriptorCount = 1;
                    wImg.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    wImg.pImageInfo = &imgInfo;

                    VkDescriptorBufferInfo bufInfo{};
                    bufInfo.buffer = cameraUbos[f].handle;
                    bufInfo.offset = 0;
                    bufInfo.range = VK_WHOLE_SIZE;
                    VkWriteDescriptorSet wUbo{};
                    wUbo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    wUbo.dstSet = descriptorSets[idx];
                    wUbo.dstBinding = 2;
                    wUbo.descriptorCount = 1;
                    wUbo.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    wUbo.pBufferInfo = &bufInfo;

                    std::array<VkWriteDescriptorSet, 3> writes{wAS, wImg, wUbo};
                    vkUpdateDescriptorSets(ctx->device(),
                                           static_cast<uint32_t>(writes.size()),
                                           writes.data(), 0, nullptr);
                }
            }
        }

        void recreateSwapchainAndDescriptors() {
            ctx->recreateSwapchain();
            vkDestroyDescriptorPool(ctx->device(), descriptorPool, nullptr);
            descriptorPool = VK_NULL_HANDLE;
            descriptorSets.clear();
            createDescriptorPool();
            allocateAndUpdateDescriptors();
            size = WindowSize{static_cast<int>(ctx->swapchainExtent().width),
                              static_cast<int>(ctx->swapchainExtent().height)};
        }

        void recordCommandBuffer(VkCommandBuffer cb, uint32_t imageIndex) {
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            check(vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");

            const VkImage img = ctx->swapchainImages()[imageIndex];

            // UNDEFINED -> GENERAL for ray-gen storage write.
            VkImageMemoryBarrier2 toGeneral{};
            toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            toGeneral.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            toGeneral.srcAccessMask = 0;
            toGeneral.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            toGeneral.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toGeneral.image = img;
            toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            toGeneral.subresourceRange.levelCount = 1;
            toGeneral.subresourceRange.layerCount = 1;

            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.imageMemoryBarrierCount = 1;
            dep.pImageMemoryBarriers = &toGeneral;
            vkCmdPipelineBarrier2(cb, &dep);

            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline);
            const uint32_t setIdx = currentFrame * imageCount_ + imageIndex;
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                                    rtPipelineLayout, 0, 1,
                                    &descriptorSets[setIdx], 0, nullptr);

            const VkExtent2D ext = ctx->swapchainExtent();
            ctx->rt().cmdTraceRays(cb, &rgenRegion, &missRegion, &hitRegion, &callRegion,
                                   ext.width, ext.height, 1);

            VkImageMemoryBarrier2 toPresent{};
            toPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            toPresent.srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            toPresent.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            toPresent.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            toPresent.dstAccessMask = 0;
            toPresent.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            toPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toPresent.image = img;
            toPresent.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            toPresent.subresourceRange.levelCount = 1;
            toPresent.subresourceRange.layerCount = 1;
            dep.pImageMemoryBarriers = &toPresent;
            vkCmdPipelineBarrier2(cb, &dep);

            check(vkEndCommandBuffer(cb), "vkEndCommandBuffer");
        }

        void renderFrame(Camera& camera) {
            VkDevice d = ctx->device();
            vkWaitForFences(d, 1, &inFlight[currentFrame], VK_TRUE, UINT64_MAX);

            uint32_t imageIndex = 0;
            VkResult acq = vkAcquireNextImageKHR(d, ctx->swapchain(), UINT64_MAX,
                                                 imageAvailable[currentFrame], VK_NULL_HANDLE, &imageIndex);
            if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapchainAndDescriptors();
                return;
            }
            if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
                check(acq, "vkAcquireNextImageKHR");
            }

            updateCameraUbo(currentFrame, camera);

            vkResetFences(d, 1, &inFlight[currentFrame]);
            vkResetCommandBuffer(cmdBuffers[currentFrame], 0);
            recordCommandBuffer(cmdBuffers[currentFrame], imageIndex);

            VkSemaphoreSubmitInfo waitInfo{};
            waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            waitInfo.semaphore = imageAvailable[currentFrame];
            waitInfo.stageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

            VkSemaphoreSubmitInfo signalInfo{};
            signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
            signalInfo.semaphore = renderFinished[currentFrame];
            signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

            VkCommandBufferSubmitInfo cbInfo{};
            cbInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
            cbInfo.commandBuffer = cmdBuffers[currentFrame];

            VkSubmitInfo2 submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
            submit.waitSemaphoreInfoCount = 1;
            submit.pWaitSemaphoreInfos = &waitInfo;
            submit.commandBufferInfoCount = 1;
            submit.pCommandBufferInfos = &cbInfo;
            submit.signalSemaphoreInfoCount = 1;
            submit.pSignalSemaphoreInfos = &signalInfo;
            check(vkQueueSubmit2(ctx->graphicsQueue(), 1, &submit, inFlight[currentFrame]),
                  "vkQueueSubmit2");

            VkPresentInfoKHR pi{};
            pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = &renderFinished[currentFrame];
            VkSwapchainKHR sc = ctx->swapchain();
            pi.swapchainCount = 1;
            pi.pSwapchains = &sc;
            pi.pImageIndices = &imageIndex;

            VkResult pr = vkQueuePresentKHR(ctx->presentQueue(), &pi);
            if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR || needsResize) {
                needsResize = false;
                recreateSwapchainAndDescriptors();
            } else if (pr != VK_SUCCESS) {
                check(pr, "vkQueuePresentKHR");
            }

            currentFrame = (currentFrame + 1) % kFramesInFlight;
        }
    };

    VulkanRenderer::VulkanRenderer(Canvas& canvas) {
        canvas.initWindow(GraphicsAPI::Vulkan);
        pimpl_ = std::make_unique<Impl>(canvas);
    }

    VulkanRenderer::~VulkanRenderer() = default;

    void VulkanRenderer::render(Object3D& scene, Camera& camera) {
        const auto cur = pimpl_->canvas.size();
        if (cur.width() != pimpl_->size.width() || cur.height() != pimpl_->size.height()) {
            pimpl_->needsResize = true;
        }
        pimpl_->ensureSceneBuilt(scene);
        pimpl_->renderFrame(camera);
    }

    WindowSize VulkanRenderer::size() const { return pimpl_->size; }

    void VulkanRenderer::setSize(const std::pair<int, int>& s) {
        pimpl_->size = WindowSize{s.first, s.second};
        pimpl_->needsResize = true;
    }

    float VulkanRenderer::getTargetPixelRatio() const { return pimpl_->pixelRatio; }
    void VulkanRenderer::setPixelRatio(float v) { pimpl_->pixelRatio = v; }

    void VulkanRenderer::setViewport(const Vector4& v) { pimpl_->viewport = v; }
    void VulkanRenderer::setViewport(int x, int y, int w, int h) {
        pimpl_->viewport.set(static_cast<float>(x), static_cast<float>(y),
                             static_cast<float>(w), static_cast<float>(h));
    }

    void VulkanRenderer::setScissor(const Vector4& v) { pimpl_->scissor = v; }
    void VulkanRenderer::setScissor(int x, int y, int w, int h) {
        pimpl_->scissor.set(static_cast<float>(x), static_cast<float>(y),
                            static_cast<float>(w), static_cast<float>(h));
    }
    void VulkanRenderer::setScissorTest(bool b) { pimpl_->scissorTest = b; }

    void VulkanRenderer::setClearColor(const Color& c, float a) {
        pimpl_->clearColor = c;
        pimpl_->clearAlpha = a;
    }
    void VulkanRenderer::getClearColor(Color& target) const { target = pimpl_->clearColor; }
    float VulkanRenderer::getClearAlpha() const { return pimpl_->clearAlpha; }
    void VulkanRenderer::setClearAlpha(float a) { pimpl_->clearAlpha = a; }

    void VulkanRenderer::clear(bool, bool, bool) {}

    RenderTarget* VulkanRenderer::getRenderTarget() { return nullptr; }
    void VulkanRenderer::setRenderTarget(RenderTarget*, int, int) {}

    std::vector<unsigned char> VulkanRenderer::readRGBPixels() { return {}; }

    void VulkanRenderer::dispose() { pimpl_.reset(); }

}// namespace threepp
