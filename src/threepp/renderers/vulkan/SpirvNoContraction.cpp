#include "threepp/renderers/vulkan/SpirvNoContraction.hpp"

#include "threepp/renderers/vulkan/NoContraction.hpp"

#include "threepp/renderers/vulkan/EnvKnob.hpp"
#include "threepp/renderers/vulkan/VulkanContext.hpp"
#include "threepp/renderers/vulkan/VulkanLog.hpp"

#include <atomic>
#include <mutex>
#include <optional>
#include <unordered_set>

// VulkanContext.hpp routes every vkCreateShaderModule of the backend here; this
// file makes the one call of the real entry point.
#undef vkCreateShaderModule

namespace threepp::vulkan {

    namespace {

        // The SPIR-V opcodes and the decoration this reads and writes.
        enum : uint32_t {
            OpSourceContinued = 2,
            OpSource = 3,
            OpSourceExtension = 4,
            OpName = 5,
            OpMemberName = 6,
            OpString = 7,
            OpExtension = 10,
            OpExtInstImport = 11,
            OpMemoryModel = 14,
            OpEntryPoint = 15,
            OpExecutionMode = 16,
            OpCapability = 17,
            OpTypeFloat = 22,
            OpTypeVector = 23,
            OpTypeMatrix = 24,
            OpDecorate = 71,
            OpMemberDecorate = 72,
            OpDecorationGroup = 73,
            OpGroupDecorate = 74,
            OpGroupMemberDecorate = 75,
            OpFNegate = 127,
            OpFAdd = 129,
            OpFSub = 131,
            OpFMul = 133,
            OpFDiv = 136,
            OpFRem = 140,
            OpFMod = 141,
            OpVectorTimesScalar = 142,
            OpMatrixTimesScalar = 143,
            OpVectorTimesMatrix = 144,
            OpMatrixTimesVector = 145,
            OpMatrixTimesMatrix = 146,
            OpDot = 148,
            OpModuleProcessed = 330,
            OpExecutionModeId = 331,
            OpDecorateId = 332,
            OpDecorateString = 5632,
            OpMemberDecorateString = 5633,
        };
        constexpr uint32_t DecNoContraction = 42;
        constexpr uint32_t kMagic = 0x07230203u;
        constexpr size_t kHeaderWords = 5;

        bool isFloatArithmetic(uint32_t op) {
            switch (op) {
                case OpFNegate:
                case OpFAdd:
                case OpFSub:
                case OpFMul:
                case OpFDiv:
                case OpFRem:
                case OpFMod:
                case OpVectorTimesScalar:
                case OpMatrixTimesScalar:
                case OpVectorTimesMatrix:
                case OpMatrixTimesVector:
                case OpMatrixTimesMatrix:
                case OpDot:
                    return true;
                default:
                    return false;
            }
        }

        // Capabilities, extensions, imports, memory model, entry points,
        // execution modes, debug and annotation instructions: the sections a
        // decoration may sit in or after. The first instruction outside them
        // starts the type declarations.
        bool isPreamble(uint32_t op) {
            switch (op) {
                case OpSourceContinued:
                case OpSource:
                case OpSourceExtension:
                case OpName:
                case OpMemberName:
                case OpString:
                case OpExtension:
                case OpExtInstImport:
                case OpMemoryModel:
                case OpEntryPoint:
                case OpExecutionMode:
                case OpCapability:
                case OpDecorate:
                case OpMemberDecorate:
                case OpDecorationGroup:
                case OpGroupDecorate:
                case OpGroupMemberDecorate:
                case OpModuleProcessed:
                case OpExecutionModeId:
                case OpDecorateId:
                case OpDecorateString:
                case OpMemberDecorateString:
                    return true;
                default:
                    return false;
            }
        }

        // The option, fixed at the first shader module: -1 until then, else 0 or 1.
        std::atomic<int> gFixed{-1};
        std::mutex gMutex;
        std::optional<bool> gRequested;// setNoContraction, before the module

        bool fromEnvironment() {
            return env::flag("THREEPP_VK_NO_CONTRACTION");
        }

        bool fixOption() {
            const int fixed = gFixed.load(std::memory_order_acquire);
            if (fixed >= 0) return fixed != 0;
            std::lock_guard lock(gMutex);
            if (gFixed.load(std::memory_order_relaxed) < 0) {
                const bool on = gRequested.value_or(fromEnvironment());
                if (on) {
                    THREEPP_LOG(kLogDevice, Info) << "shader float arithmetic: NoContraction on every module "
                                                     "(THREEPP_VK_NO_CONTRACTION or vulkan::setNoContraction)";
                }
                gFixed.store(on ? 1 : 0, std::memory_order_release);
            }
            return gFixed.load(std::memory_order_relaxed) != 0;
        }

    }// namespace

    std::vector<uint32_t> addNoContraction(const uint32_t* w, size_t n, NoContractionStats* stats) {
        std::vector<uint32_t> out(w, w + n);
        if (n < kHeaderWords || w[0] != kMagic) return out;

        std::unordered_set<uint32_t> floatTypes, decorated;
        std::vector<uint32_t> targets;
        size_t insertAt = kHeaderWords;
        bool inPreamble = true;
        for (size_t i = kHeaderWords; i < n;) {
            const uint32_t wc = w[i] >> 16, op = w[i] & 0xFFFFu;
            if (wc == 0 || wc > n - i) return out;// malformed: leave it to the driver as it is
            if (inPreamble) {
                if (isPreamble(op)) insertAt = i + wc;
                else inPreamble = false;
            }
            if (op == OpDecorate && wc >= 3 && w[i + 2] == DecNoContraction) {
                decorated.insert(w[i + 1]);
            } else if (op == OpTypeFloat && wc >= 2) {
                floatTypes.insert(w[i + 1]);
            } else if ((op == OpTypeVector || op == OpTypeMatrix) && wc >= 4 && floatTypes.count(w[i + 2])) {
                floatTypes.insert(w[i + 1]);
            } else if (isFloatArithmetic(op) && wc >= 4 && floatTypes.count(w[i + 1])) {
                targets.push_back(w[i + 2]);
            }
            i += wc;
        }

        NoContractionStats st;
        std::vector<uint32_t> add;
        add.reserve(targets.size() * 3);
        for (const uint32_t id : targets) {
            ++st.floatResults;
            if (!decorated.insert(id).second) {
                ++st.alreadyDecorated;
                continue;
            }
            add.push_back((3u << 16) | OpDecorate);
            add.push_back(id);
            add.push_back(DecNoContraction);
            ++st.added;
        }
        if (!add.empty()) out.insert(out.begin() + static_cast<std::ptrdiff_t>(insertAt), add.begin(), add.end());
        if (stats) *stats = st;
        return out;
    }

    bool setNoContraction(bool enabled) {
        std::lock_guard lock(gMutex);
        const int fixed = gFixed.load(std::memory_order_relaxed);
        if (fixed >= 0) {
            if ((fixed != 0) == enabled) return true;
            THREEPP_LOG(kLogRenderer, Warn) << "setNoContraction(" << (enabled ? "true" : "false")
                                            << ") ignored: shader modules were already created with it "
                                            << (fixed ? "on" : "off");
            return false;
        }
        gRequested = enabled;
        return true;
    }

    bool noContraction() {
        std::lock_guard lock(gMutex);
        const int fixed = gFixed.load(std::memory_order_relaxed);
        return fixed >= 0 ? fixed != 0 : gRequested.value_or(fromEnvironment());
    }

}// namespace threepp::vulkan

VkResult threeppVkCreateShaderModule(VkDevice device, const VkShaderModuleCreateInfo* createInfo,
                                     const VkAllocationCallbacks* allocator, VkShaderModule* module) {
    using namespace threepp::vulkan;
    if (!fixOption() || !createInfo || !createInfo->pCode || createInfo->codeSize % 4 != 0) {
        return vkCreateShaderModule(device, createInfo, allocator, module);
    }
    const auto words = addNoContraction(createInfo->pCode, createInfo->codeSize / 4);
    VkShaderModuleCreateInfo decorated = *createInfo;
    decorated.codeSize = words.size() * sizeof(uint32_t);
    decorated.pCode = words.data();
    return vkCreateShaderModule(device, &decorated, allocator, module);
}
