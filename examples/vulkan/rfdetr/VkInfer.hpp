#pragma once

// Minimal Vulkan compute harness for the RF-DETR inference port.
//
// Plays the role of a thin compute-pipeline + buffer wrapper on the Vulkan
// side, but stays entirely example-local: it is constructed from the native
// handles a VulkanRenderer already exposes (VkDevice / VkPhysicalDevice /
// graphics-capable VkQueue / queue family), so no core renderer change is
// needed. Buffers use raw VkDeviceMemory; every op's parameters travel as push
// constants (all param structs are <= 96 bytes) so descriptor sets carry
// storage buffers only.
//
// Execution model: a whole inference is recorded into ONE command buffer
// (beginFrame() -> many dispatch() -> endFrame()), submitted once and waited on
// with one fence — not one submit per op. Host input and output travel inside
// that command buffer too (recordUpload() / recordReadback(), through
// persistently mapped staging), so an inference makes no other round trip.
// Each dispatch binds a descriptor set cached by (pipeline, buffers) and carries
// a leading storage barrier so layer N+1 sees layer N's writes. Intermediate
// activations live in a per-inference arena owned by VkInfer (rewound by
// resetArena() after readback), so a recorded buffer is never destroyed while
// the submit still references it. Weights are persistent owning VkTensors,
// separate from the arena.

#include <vulkan/vulkan.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <map>
#include <utility>
#include <vector>

namespace rfdetr {

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
        const char* name = "";// the per-kernel profile's label
    };

    /// One host->device copy for uploadMany (batches N uploads into a single submit).
    struct UploadItem {
        VkBuffer dst = VK_NULL_HANDLE;
        const void* data = nullptr;
        VkDeviceSize bytes = 0;
    };

    class VkInfer {
    public:
        VkInfer(VkInstance instance, VkDevice device, VkPhysicalDevice phys, VkQueue queue, uint32_t queueFamily);
        ~VkInfer();

        // Clustered subgroup operations in compute shaders (GL_KHR_shader_subgroup_clustered),
        // subgroups of at least 8.
        [[nodiscard]] bool subgroupClustered() const { return subgroupClustered_; }
        // Streaming multiprocessors (NVIDIA) / compute units (AMD), for sizing GEMM
        // grids; 32 where the device does not say.
        [[nodiscard]] uint32_t smCount() const { return smCount_; }
        // fp16 tensor-core matrix multiplies (VK_KHR_cooperative_matrix: 16x16x16, fp16 A/B,
        // fp32 accumulation, subgroup scope, 32-wide subgroups), with the device features
        // VulkanContext enables for them.
        [[nodiscard]] bool coopMatF16() const { return coopMatF16_; }

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
        // N host->device copies in a single submit+wait (one staging pack, one round-trip)
        // — collapses what would be N separate upload() drains into one.
        void uploadMany(const std::vector<UploadItem>& items);
        void readback(VkBuffer src, void* dst, VkDeviceSize bytes, VkDeviceSize srcOffset = 0);
        // Two device->host copies in a single submit (one GPU round-trip).
        void readback2(VkBuffer srcA, void* dstA, VkDeviceSize bytesA,
                       VkBuffer srcB, void* dstB, VkDeviceSize bytesB);
        void zero(VkBuffer buf);

        // ── host I/O inside the frame's one submit ─────────────────────────
        // Size the frame staging before beginFrame(): the totals the frame will
        // upload and read back (copies are padded to 16 bytes; 256 bytes of
        // slack are added for that).
        void reserveFrameIO(VkDeviceSize uploadBytes, VkDeviceSize readbackBytes);
        // Host data -> dst: staged now, copied at this point of the command buffer.
        // Returns its offset in the frame staging (for frameUploadData()).
        VkDeviceSize recordUpload(VkBuffer dst, const void* data, VkDeviceSize bytes);
        // The staging a recordUpload() of the recorded frame copies from, to refill
        // before replayFrame().
        [[nodiscard]] void* frameUploadData(VkDeviceSize offset);
        // src -> host, copied after the work recorded so far. Returns the offset
        // to pass to frameReadback() once endFrame() has returned.
        VkDeviceSize recordReadback(VkBuffer src, VkDeviceSize bytes);
        // What a recordReadback() of this frame copied; valid until the next beginFrame().
        [[nodiscard]] const void* frameReadback(VkDeviceSize offset) const;

        [[nodiscard]] VkBuffer dummy() const { return dummy_.buffer; }

        // ── pipelines ──────────────────────────────────────────────────────
        VkPipe createPipe(const uint32_t* spv, size_t spvByteCount, uint32_t nSSBO, uint32_t pushBytes,
                          const char* name = "");
        void destroyPipe(VkPipe& p);

        // Drop all cached descriptor sets (call when weight buffers are replaced,
        // e.g. on loadWeights, since cached sets reference the old buffer handles).
        void resetDescriptorCache();

        // ── batched dispatch ───────────────────────────────────────────────
        // Rewind the arena + frame I/O, start the command buffer. With a replayKey the
        // frame can be submitted again by replayFrame() while replayable(replayKey)
        // holds: nothing it was recorded against (arena slots, frame staging,
        // descriptor sets) has been replaced since, and no other frame was recorded.
        void beginFrame(uint64_t replayKey = 0);
        [[nodiscard]] bool replayable(uint64_t key) const {
            return key != 0 && key == replayKey_ && layoutGen_ == replayGen_;
        }
        void replayFrame();// the recorded frame again: submit + fence wait
        void recordFill(VkBuffer dst, VkDeviceSize bytes);// zero-fill recorded into the frame
        // Device→device copy recorded into the frame (no submit/wait); leading
        // barrier makes prior compute writes to src visible. Lets a value stay
        // GPU-resident across frames without a host round-trip.
        void recordCopy(VkBuffer dst, VkBuffer src, VkDeviceSize bytes);
        void dispatch(const VkPipe& pipe, const std::vector<VkBuffer>& ssbos,
                      const void* push, uint32_t pushBytes,
                      uint32_t gx, uint32_t gy, uint32_t gz);
        void endFrame();// end + submit + fence wait

        // GPU timestamp between dispatches (no-op unless RFDETR_PROFILE is set).
        // endFrame() prints per-mark GPU deltas, isolating GPU time from CPU
        // command-recording overhead, and the GPU time of every dispatch summed
        // by kernel (each dispatch is timestamped too while profiling), and the host
        // time from beginFrame() to the submit. On NVIDIA under Windows a dispatch's
        // timestamp can bill its time to a neighbour (other processes' GPU work
        // time-slices in), so trust frame totals and marks over single kernels.
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

        // Descriptor-set cache keyed by (pipeline, bound buffers). The op sequence and
        // its arena buffers are identical every inference, so each dispatch's set is
        // allocated+written once and reused — eliminating ~250 vkAllocateDescriptorSets
        // + vkUpdateDescriptorSets per inference in steady state.
        std::map<std::pair<VkPipeline, std::vector<VkBuffer>>, VkDescriptorSet> dsCache_;

        VkCommandBuffer frameCb_ = VK_NULL_HANDLE;
        bool recording_ = false;

        // Activation buffers, pooled + reused across inferences (the op sequence
        // is identical every frame), so steady state does zero buffer allocation.
        std::vector<VkTensor> arenaSlots_;
        size_t arenaCursor_ = 0;
        VkTensor dummy_;
        VkTensor uploadStaging_;  // persistent host-visible upload scratch
        VkTensor readbackStaging_;// persistent host-cached readback scratch

        // Frame I/O staging, mapped for their whole life (separate from the
        // one-shot staging above, which a one-shot call may reuse mid-frame).
        VkTensor frameUpload_, frameReadback_;
        void* frameUploadMapped_ = nullptr;
        void* frameReadbackMapped_ = nullptr;
        VkDeviceSize frameUploadCursor_ = 0, frameReadbackCursor_ = 0;
        bool frameHasReadback_ = false;
        VkFence frameFence_ = VK_NULL_HANDLE;

        bool subgroupClustered_ = false;
        bool coopMatF16_ = false;
        void detectCoopMat_(VkInstance instance);
        uint32_t smCount_ = 32;

        // Profiling timestamps (enabled by RFDETR_PROFILE): marks and, after
        // every dispatch, one per dispatch labelled with its kernel.
        static constexpr uint32_t kMaxTimestamps = 1024;
        VkQueryPool tsPool_ = VK_NULL_HANDLE;
        float tsPeriodNs_ = 1.0f;
        bool tsEnabled_ = false;
        // Bumped whenever a buffer or descriptor set a recorded frame may use is replaced.
        uint64_t layoutGen_ = 1;
        uint64_t recordKey_ = 0, replayKey_ = 0, replayGen_ = 0;
        void submitAndWait_();// the frame command buffer, then the profile printout
        std::chrono::steady_clock::time_point hostBegin_;// beginFrame(), for the host-side split
        uint32_t tsCount_ = 0;
        std::vector<const char*> tsLabels_;
        std::vector<bool> tsIsDispatch_;
    };

}// namespace rfdetr
