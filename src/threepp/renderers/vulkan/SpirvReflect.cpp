#include "threepp/renderers/vulkan/SpirvReflect.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace threepp::vulkan {

    namespace {

        // The SPIR-V opcodes, decorations and storage classes this reads.
        enum : uint32_t {
            OpEntryPoint = 15,
            OpTypeImage = 25,
            OpTypeSampler = 26,
            OpTypeSampledImage = 27,
            OpTypeArray = 28,
            OpTypeRuntimeArray = 29,
            OpTypeStruct = 30,
            OpTypePointer = 32,
            OpConstant = 43,
            OpVariable = 59,
            OpDecorate = 71,
            OpMemberDecorate = 72,
            OpTypeAccelerationStructureKHR = 5341,
        };
        enum : uint32_t {
            DecBlock = 2,
            DecBufferBlock = 3,
            DecNonWritable = 24,
            DecNonReadable = 25,
            DecBinding = 33,
            DecDescriptorSet = 34,
        };
        enum : uint32_t {
            ScUniformConstant = 0,
            ScUniform = 2,
            ScStorageBuffer = 12,
        };
        constexpr uint32_t kDimBuffer = 5;

        struct Decorations {
            uint32_t set = 0, binding = UINT32_MAX;
            bool nonWritable = false, nonReadable = false;
            bool block = false, bufferBlock = false;
        };
        struct MemberFlags {
            uint32_t members = 0, nonWritable = 0, nonReadable = 0;
        };

    }// namespace

    std::vector<SpirvBinding> reflectSpirvBindings(const uint32_t* words, size_t wordCount) {
        std::vector<SpirvBinding> out;
        if (!words || wordCount < 5 || words[0] != 0x07230203u) return out;

        std::unordered_map<uint32_t, Decorations> dec;
        std::unordered_map<uint32_t, MemberFlags> memberDec;
        std::unordered_map<uint32_t, uint32_t> structMembers;         // struct id -> member count
        std::unordered_map<uint32_t, uint32_t> pointee;               // pointer type -> pointee
        std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> arrays;// array type -> (elem, length id); length UINT32_MAX = runtime
        std::unordered_map<uint32_t, uint32_t> constants;             // constant id -> value (scalar 32-bit)
        std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> images;// image type -> (dim, sampled)
        std::unordered_set<uint32_t> samplers, sampledImages, accels;
        struct Var { uint32_t type, storage; };
        std::unordered_map<uint32_t, Var> vars;
        std::vector<uint32_t> interfaceIds;
        bool haveEntry = false;

        for (size_t i = 5; i < wordCount;) {
            const uint32_t op = words[i] & 0xFFFFu;
            const uint32_t n  = words[i] >> 16;
            if (n == 0 || i + n > wordCount) return {};
            const uint32_t* w = words + i;
            switch (op) {
                case OpEntryPoint: {
                    if (haveEntry) break;// the first entry point
                    haveEntry = true;
                    // model, id, name (nul-terminated, padded to words), interface...
                    size_t k = 3;
                    while (k < n) {
                        const uint32_t word = w[k++];
                        if ((word & 0xFF000000u) == 0 || (word & 0x00FF0000u) == 0 ||
                            (word & 0x0000FF00u) == 0 || (word & 0x000000FFu) == 0) break;
                    }
                    for (; k < n; ++k) interfaceIds.push_back(w[k]);
                    break;
                }
                case OpDecorate: {
                    auto& d = dec[w[1]];
                    switch (w[2]) {
                        case DecDescriptorSet: if (n > 3) d.set = w[3]; break;
                        case DecBinding: if (n > 3) d.binding = w[3]; break;
                        case DecNonWritable: d.nonWritable = true; break;
                        case DecNonReadable: d.nonReadable = true; break;
                        case DecBlock: d.block = true; break;
                        case DecBufferBlock: d.bufferBlock = true; break;
                        default: break;
                    }
                    break;
                }
                case OpMemberDecorate: {
                    auto& m = memberDec[w[1]];
                    if (w[3] == DecNonWritable) m.nonWritable |= 1u << std::min(w[2], 31u);
                    if (w[3] == DecNonReadable) m.nonReadable |= 1u << std::min(w[2], 31u);
                    break;
                }
                case OpTypeStruct: structMembers[w[1]] = n - 2; break;
                case OpTypePointer: pointee[w[1]] = w[3]; break;
                case OpTypeArray: arrays[w[1]] = {w[2], w[3]}; break;
                case OpTypeRuntimeArray: arrays[w[1]] = {w[2], UINT32_MAX}; break;
                case OpConstant: if (n == 4) constants[w[2]] = w[3]; break;
                case OpTypeImage: images[w[1]] = {w[3], w[7]}; break;
                case OpTypeSampler: samplers.insert(w[1]); break;
                case OpTypeSampledImage: sampledImages.insert(w[1]); break;
                case OpTypeAccelerationStructureKHR: accels.insert(w[1]); break;
                case OpVariable: vars[w[2]] = {w[1], w[3]}; break;
                default: break;
            }
            i += n;
        }
        if (!haveEntry) return out;

        for (const uint32_t id : interfaceIds) {
            const auto v = vars.find(id);
            if (v == vars.end()) continue;
            const uint32_t sc = v->second.storage;
            if (sc != ScUniformConstant && sc != ScUniform && sc != ScStorageBuffer) continue;
            const auto d = dec.find(id);
            if (d == dec.end() || d->second.binding == UINT32_MAX) continue;

            SpirvBinding b;
            b.set     = d->second.set;
            b.binding = d->second.binding;

            uint32_t type = pointee.count(v->second.type) ? pointee[v->second.type] : 0;
            // Peel one array level (descriptor arrays).
            if (const auto a = arrays.find(type); a != arrays.end()) {
                b.arraySize = a->second.second == UINT32_MAX
                                      ? 0u
                                      : (constants.count(a->second.second) ? constants[a->second.second] : 1u);
                type = a->second.first;
            }

            const auto& vd = d->second;
            if (sc == ScUniformConstant) {
                if (sampledImages.count(type)) {
                    b.kind = SpirvBinding::Kind::CombinedSampler;
                } else if (samplers.count(type)) {
                    b.kind = SpirvBinding::Kind::Sampler;
                } else if (accels.count(type)) {
                    b.kind = SpirvBinding::Kind::AccelerationStructure;
                } else if (const auto im = images.find(type); im != images.end()) {
                    const bool buffer  = im->second.first == kDimBuffer;
                    const bool storage = im->second.second == 2;
                    if (buffer) {
                        b.kind = storage ? SpirvBinding::Kind::StorageTexelBuffer
                                         : SpirvBinding::Kind::UniformTexelBuffer;
                    } else {
                        b.kind = storage ? SpirvBinding::Kind::StorageImage
                                         : SpirvBinding::Kind::SampledImage;
                    }
                    if (storage) {
                        b.read  = !vd.nonReadable;
                        b.write = !vd.nonWritable;
                    }
                }
            } else {
                const auto& td = dec[type];
                const bool ssbo = sc == ScStorageBuffer || td.bufferBlock;
                b.kind = ssbo ? SpirvBinding::Kind::StorageBuffer : SpirvBinding::Kind::UniformBuffer;
                if (ssbo) {
                    // readonly / writeonly on a block land on its members (or,
                    // from some front ends, on the variable).
                    const uint32_t members = structMembers.count(type) ? structMembers[type] : 0u;
                    const uint32_t all = members >= 32 ? 0xFFFFFFFFu : ((1u << members) - 1u);
                    const auto m = memberDec.find(type);
                    const bool allNonWritable = vd.nonWritable ||
                                                (m != memberDec.end() && members > 0 && (m->second.nonWritable & all) == all);
                    const bool allNonReadable = vd.nonReadable ||
                                                (m != memberDec.end() && members > 0 && (m->second.nonReadable & all) == all);
                    b.read  = !allNonReadable;
                    b.write = !allNonWritable;
                }
            }
            out.push_back(b);
        }
        std::sort(out.begin(), out.end(), [](const SpirvBinding& a, const SpirvBinding& c) {
            return a.set != c.set ? a.set < c.set : a.binding < c.binding;
        });
        return out;
    }

}// namespace threepp::vulkan
