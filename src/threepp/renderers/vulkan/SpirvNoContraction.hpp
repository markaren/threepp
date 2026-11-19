// SpirvNoContraction — the SPIR-V rewrite behind vulkan::setNoContraction
// (threepp/renderers/vulkan/NoContraction.hpp).
//
// Decorates NoContraction on the result of every float arithmetic instruction
// glslang decorates for a `precise` expression: OpFNegate, OpFAdd, OpFSub,
// OpFMul, OpFDiv, OpFRem, OpFMod, OpVectorTimesScalar, OpMatrixTimesScalar,
// OpVectorTimesMatrix, OpMatrixTimesVector, OpMatrixTimesMatrix and OpDot,
// when the result type is a float scalar, vector or matrix. Integer arithmetic,
// extended instructions (GLSL.std.450) and OpOuterProduct are left alone, as
// glslang leaves them. Ids that already carry the decoration (the shaders' own
// `precise`) are not decorated twice.
//
// The new OpDecorate instructions are inserted at the end of the module's
// annotation section, before its first type declaration. No id is added, so
// the header's bound is unchanged.

#ifndef THREEPP_VULKAN_SPIRV_NO_CONTRACTION_HPP
#define THREEPP_VULKAN_SPIRV_NO_CONTRACTION_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace threepp::vulkan {

    struct NoContractionStats {
        uint32_t floatResults = 0;    // results of the instructions above with a float type
        uint32_t alreadyDecorated = 0;// of those, already NoContraction
        uint32_t added = 0;           // decorations inserted
    };

    // The module with its float arithmetic decorated NoContraction. A module
    // that is not SPIR-V or does not parse comes back unchanged.
    std::vector<uint32_t> addNoContraction(const uint32_t* words, size_t wordCount,
                                           NoContractionStats* stats = nullptr);

    template<size_t N>
    std::vector<uint32_t> addNoContraction(const uint32_t (&spv)[N], NoContractionStats* stats = nullptr) {
        return addNoContraction(spv, N, stats);
    }

}// namespace threepp::vulkan

#endif//THREEPP_VULKAN_SPIRV_NO_CONTRACTION_HPP
