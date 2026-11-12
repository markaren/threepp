// the extended ocean's master switch (THREEPP_OCEAN_EXTENDED, default on; =0 off).
//
// Off, the renderer takes the base renderer's own code paths for the clouds, the
// ocean/water, foam and fog: the shaders branch on the specialization
// constants of ocean_extended.glsl, whose off side is upstream's code, and the
// host skips the extra uploads and passes and uses upstream's values. Every
// other the extended ocean switch (THREEPP_OCEAN_*, ...) is ignored while it is
// off: read them through oceanFeature(), never std::getenv alone.
//
// Not cached (EnvKnob.hpp's rule): tests set variables between renderers.

#ifndef THREEPP_VULKAN_OCEAN_EXTENDED_HPP
#define THREEPP_VULKAN_OCEAN_EXTENDED_HPP

#include "threepp/renderers/vulkan/EnvKnob.hpp"

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>

namespace threepp::vulkan {

    inline bool oceanExtended() { return env::flag("THREEPP_OCEAN_EXTENDED", true); }

    // An extended ocean switch: its own value under the master switch, off without it.
    inline bool oceanFeature(const char* name, bool dflt) { return oceanExtended() && env::flag(name, dflt); }

    // The water G-buffer normal (THREEPP_OCEAN_WATER_GBUFFER_NORMAL, default on).
    inline bool oceanWaterGbufferNormal() { return oceanFeature("THREEPP_OCEAN_WATER_GBUFFER_NORMAL", true); }

    // ocean_extended.glsl's specialization constants.
    constexpr uint32_t kOceanExtConstantId        = 21u;
    constexpr uint32_t kWaterGbufNormalConstantId = 22u;

    // The two entries, for a pipeline that has no specialization of its own:
    // info() is nullptr with both off, so that pipeline is built exactly as
    // upstream builds it. Values read at construction.
    struct OceanSpec {
        std::array<uint32_t, 2> values{};
        std::array<VkSpecializationMapEntry, 2> entries{};
        VkSpecializationInfo si{};
        uint32_t count = 0;

        OceanSpec() {
            values = {oceanExtended() ? 1u : 0u, oceanWaterGbufferNormal() ? 1u : 0u};
            const uint32_t ids[2] = {kOceanExtConstantId, kWaterGbufNormalConstantId};
            for (uint32_t i = 0; i < 2; ++i) {
                if (values[i] == 0u) continue;
                entries[count++] = {ids[i], i * static_cast<uint32_t>(sizeof(uint32_t)), sizeof(uint32_t)};
            }
            si.mapEntryCount = count;
            si.pMapEntries   = entries.data();
            si.dataSize      = sizeof(values);
            si.pData         = values.data();
        }
        OceanSpec(const OceanSpec&) = delete;
        OceanSpec& operator=(const OceanSpec&) = delete;

        [[nodiscard]] const VkSpecializationInfo* info() const { return count ? &si : nullptr; }

        // Appends the entries to a pipeline's own specialization: `vals` gets
        // the values at word `at` and `at + 1`, `ents` the entries from `n`.
        template<size_t NV, size_t NE>
        void appendTo(std::array<uint32_t, NV>& vals, uint32_t at, std::array<VkSpecializationMapEntry, NE>& ents,
                      uint32_t& n, uint32_t& dataWords) const {
            const uint32_t ids[2] = {kOceanExtConstantId, kWaterGbufNormalConstantId};
            for (uint32_t i = 0; i < 2; ++i) {
                if (values[i] == 0u) continue;
                vals[at + i] = values[i];
                ents[n++] = {ids[i], (at + i) * static_cast<uint32_t>(sizeof(uint32_t)), sizeof(uint32_t)};
                if (at + i + 1u > dataWords) dataWords = at + i + 1u;
            }
        }
    };

}// namespace threepp::vulkan

#endif// THREEPP_VULKAN_OCEAN_EXTENDED_HPP
