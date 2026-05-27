#pragma once

// Minimal Vulkan compute harness for the YOLOv8n inference port.
//
// Mirrors the role of threepp's WgpuComputePipeline + WgpuBuffer on the Vulkan
// side, but stays entirely example-local: it is constructed from the native
// handles a VulkanRenderer already exposes (VkDevice / VkPhysicalDevice /
// graphics-capable VkQueue / queue family), so no core renderer change is
// needed. Buffers use raw VkDeviceMemory; every op's parameters travel as push
// constants (all param structs are <= 64 bytes) so descriptor sets carry
// storage buffers only.
//
// Execution model: a whole inference is recorded into ONE command buffer
// (beginFrame() -> many dispatch() -> endFrame()) and submitted once, like the
// WGPU path — not one submit per op. Each dispatch gets its own descriptor set
// (from a per-frame pool) and a leading storage barrier so layer N+1 sees layer
// N's writes. Intermediate activations live in a per-inference arena owned by
// VkInfer (freed by resetArena() after readback), so a recorded buffer is never
// destroyed while the single submit still references it. Weights and the
// detection buffers are persistent owning VkTensors, separate from the arena.

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <utility>
#include <vector>

namespace rtdetr {

    /// Owning GPU buffer (move-only RAII): weights, detection buffers, the dummy,
    /// and the arena's backing storage.
    struct VkTensor {
        std::vector<uint32_t> shape;
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize capacity = 0;
        VkDevice device = VK_NULL_HANDLE;

        VkTensor() = default;
        ~VkTensor() { reset(); }
        VkTensor(const VkTensor&) = delete;
        VkTensor& operator=(const VkTensor&) = delete;
        VkTensor(VkTensor&& o) noexcept { moveFrom(o); }
        VkTensor& operator=(VkTensor&& o) noexcept {
            if (this != &o) { reset(); moveFrom(o); }
            return *this;
        }
        void reset() {
            if (device && buffer) vkDestroyBuffer(device, buffer, nullptr);
            if (device && memory) vkFreeMemory(device, memory, nullptr);
            buffer = VK_NULL_HANDLE;
            memory = VK_NULL_HANDLE;
            capacity = 0;
        }
        [[nodiscard]] uint32_t C() const { return shape.empty() ? 1 : shape[0]; }
        [[nodiscard]] uint32_t H() const { return shape.size() > 1 ? shape[1] : 1; }
        [[nodiscard]] uint32_t W() const { return shape.size() > 2 ? shape[2] : 1; }

    private:
        void moveFrom(VkTensor& o) {
            shape = std::move(o.shape);
            buffer = o.buffer;
            memory = o.memory;
            capacity = o.capacity;
            device = o.device;
            o.buffer = VK_NULL_HANDLE;
            o.memory = VK_NULL_HANDLE;
            o.capacity = 0;
        }
    };

    /// Non-owning view into a buffer (arena- or persistently-owned). Copyable;
    /// carries just the handle + NCHW shape. This is what op helpers pass around.
    struct Tensor {
        VkBuffer buffer = VK_NULL_HANDLE;
        std::vector<uint32_t> shape;

        [[nodiscard]] bool valid() const { return buffer != VK_NULL_HANDLE; }
        [[nodiscard]] uint32_t numel() const {
            uint32_t n = 1;
            for (auto s : shape) n *= s;
            return n;
        }
        [[nodiscard]] size_t bytes() const { return static_cast<size_t>(numel()) * sizeof(float); }
        [[nodiscard]] uint32_t C() const { return shape.empty() ? 1 : shape[0]; }
        [[nodiscard]] uint32_t H() const { return shape.size() > 1 ? shape[1] : 1; }
        [[nodiscard]] uint32_t W() const { return shape.size() > 2 ? shape[2] : 1; }
    };

    struct VkPipe {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout dsLayout = VK_NULL_HANDLE;
        uint32_t nSSBO = 0;
        uint32_t pushBytes = 0;
    };

    class VkInfer {
    public:
        VkInfer(VkDevice device, VkPhysicalDevice phys, VkQueue queue, uint32_t queueFamily);
        ~VkInfer();

        VkInfer(const VkInfer&) = delete;
        VkInfer& operator=(const VkInfer&) = delete;

        // ── persistent owning buffers (weights, detection buffers) ─────────
        VkTensor createOwned(const std::vector<uint32_t>& shape);// f32, sized to numel
        VkTensor createOwnedRaw(VkDeviceSize bytes);

        // ── per-inference arena buffers (activations) ──────────────────────
        Tensor createTensor(std::initializer_list<uint32_t> shape);
        Tensor createTensorV(const std::vector<uint32_t>& shape);
        Tensor createRaw(VkDeviceSize bytes);// shapeless (e.g. packed input)
        void resetArena();                   // free all arena buffers

        // ── data movement (one-shot submit+wait; safe any time) ────────────
        void upload(VkBuffer dst, const void* data, VkDeviceSize bytes);
        void readback(VkBuffer src, void* dst, VkDeviceSize bytes);
        // Two device->host copies in a single submit (one GPU round-trip).
        void readback2(VkBuffer srcA, void* dstA, VkDeviceSize bytesA,
                       VkBuffer srcB, void* dstB, VkDeviceSize bytesB);
        void zero(VkBuffer buf);

        [[nodiscard]] VkBuffer dummy() const { return dummy_.buffer; }

        // ── pipelines ──────────────────────────────────────────────────────
        VkPipe createPipe(const uint32_t* spv, size_t spvByteCount, uint32_t nSSBO, uint32_t pushBytes);
        void destroyPipe(VkPipe& p);

        // ── batched dispatch ───────────────────────────────────────────────
        void beginFrame();// reset per-frame descriptor pool + start command buffer
        void recordFill(VkBuffer dst, VkDeviceSize bytes);// zero-fill recorded into the frame
        void dispatch(const VkPipe& pipe, const std::vector<VkBuffer>& ssbos,
                      const void* push, uint32_t pushBytes,
                      uint32_t gx, uint32_t gy, uint32_t gz);
        void endFrame();// end + submit + wait

        // GPU timestamp between dispatches (no-op unless RTDETR_PROFILE is set).
        // endFrame() prints per-mark GPU deltas, isolating GPU time from CPU
        // command-recording overhead.
        void markTimestamp(const char* label);

        [[nodiscard]] VkDevice device() const { return device_; }

    private:
        uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const;
        VkTensor allocBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags props);
        // Persistent staging buffers (grown on demand, reused) so uploads/readbacks
        // don't vkAllocateMemory/vkFreeMemory every call. uploadStaging_ is
        // HOST_COHERENT (fast writes); readbackStaging_ prefers HOST_CACHED (fast
        // device->host reads), falling back to HOST_COHERENT.
        void ensureUploadStaging(VkDeviceSize bytes);
        void ensureReadbackStaging(VkDeviceSize bytes);
        VkBuffer acquireArena(VkDeviceSize bytes);// pooled activation buffer (reused across frames)
        template<typename Fn>
        void oneShot(Fn&& fn);

        VkDevice device_;
        VkPhysicalDevice phys_;
        VkQueue queue_;
        uint32_t queueFamily_;

        VkCommandPool cmdPool_ = VK_NULL_HANDLE;    // frame command buffer
        VkCommandPool oneShotPool_ = VK_NULL_HANDLE;// transient upload/readback/zero buffers
        VkDescriptorPool descPool_ = VK_NULL_HANDLE;

        VkCommandBuffer frameCb_ = VK_NULL_HANDLE;
        bool recording_ = false;

        // Activation buffers, pooled + reused across inferences (the op sequence
        // is identical every frame), so steady state does zero buffer allocation.
        std::vector<VkTensor> arenaSlots_;
        size_t arenaCursor_ = 0;
        VkTensor dummy_;
        VkTensor uploadStaging_;  // persistent host-visible upload scratch
        VkTensor readbackStaging_;// persistent host-cached readback scratch

        // Profiling timestamps (enabled by RTDETR_PROFILE).
        VkQueryPool tsPool_ = VK_NULL_HANDLE;
        float tsPeriodNs_ = 1.0f;
        bool tsEnabled_ = false;
        uint32_t tsCount_ = 0;
        std::vector<const char*> tsLabels_;
    };

}// namespace rtdetr
