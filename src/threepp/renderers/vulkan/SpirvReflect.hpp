// SpirvReflect — the descriptor bindings a SPIR-V entry point actually uses.
//
// Reads a module compiled for SPIR-V 1.4 or later (the backend compiles with
// --target-env vulkan1.3, i.e. SPIR-V 1.6), where OpEntryPoint lists every
// global variable the entry point statically uses — not only its inputs and
// outputs. For each used descriptor variable it reports the set, binding,
// descriptor kind, array size, and whether the shader can read and/or write
// it (readonly / writeonly qualifiers: NonWritable / NonReadable).
//
// This is what lets a render-graph pass derive its resource declarations from
// the pipeline it dispatches instead of a hand-written list that drifts.

#ifndef THREEPP_VULKAN_SPIRV_REFLECT_HPP
#define THREEPP_VULKAN_SPIRV_REFLECT_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    struct SpirvBinding {
        enum class Kind : uint8_t {
            SampledImage,     // texture2D etc. (separate image)
            CombinedSampler,  // sampler2D etc.
            Sampler,          // sampler
            StorageImage,     // image2D etc.
            UniformTexelBuffer,
            StorageTexelBuffer,
            UniformBuffer,
            StorageBuffer,
            AccelerationStructure,
            Unknown,
        };
        uint32_t set = 0;
        uint32_t binding = 0;
        uint32_t arraySize = 1;// 0 = runtime-sized
        Kind     kind = Kind::Unknown;
        bool     read = true;
        bool     write = false;
    };

    // Bindings used by the module's first entry point, sorted by (set, binding).
    // Empty on a malformed module.
    std::vector<SpirvBinding> reflectSpirvBindings(const uint32_t* words, size_t wordCount);

    template<size_t N>
    std::vector<SpirvBinding> reflectSpirvBindings(const uint32_t (&spv)[N]) {
        return reflectSpirvBindings(spv, N);
    }

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_SPIRV_REFLECT_HPP
