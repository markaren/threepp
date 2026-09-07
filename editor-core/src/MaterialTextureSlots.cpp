
#include "threepp/extras/editor/MaterialTextureSlots.hpp"

#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>
#include <array>
#include <cctype>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // Suffixes seen on textures exported by the usual authoring tools. Longest
    // first within each slot so "_normalgl" cannot be shadowed by "_normal".
    struct SlotAliases {
        const char* slot;
        std::array<const char*, 6> suffixes;
    };

    constexpr std::array<SlotAliases, 5> aliases{{
            {"normalMap", {"normalgl", "normaldx", "normal", "norm", "nrm", "nor"}},
            {"roughnessMap", {"roughness", "rough", "rgh", nullptr, nullptr, nullptr}},
            {"metalnessMap", {"metalness", "metallic", "metal", "mtl", nullptr, nullptr}},
            {"aoMap", {"occlusion", "ambientocclusion", "ao", nullptr, nullptr, nullptr}},
            {"emissiveMap", {"emissive", "emission", "emit", nullptr, nullptr, nullptr}},
    }};

    std::string lowered(const std::string& text) {

        std::string out = text;
        std::transform(out.begin(), out.end(), out.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return out;
    }

    // The trailing word of a name split on the usual separators: "brick_wall-NRM"
    // -> "nrm". Matching the last token only is what keeps a file that merely
    // contains "normal" somewhere in a directory-ish name from being claimed.
    std::string lastToken(const std::string& stem) {

        const auto text = lowered(stem);
        const auto cut = text.find_last_of("_-. ");
        return cut == std::string::npos ? text : text.substr(cut + 1);
    }

}// namespace


std::vector<MaterialTextureSlot> threepp::editor::textureSlotsOf(Material& material) {

    std::vector<MaterialTextureSlot> slots;

    if (auto* m = dynamic_cast<MaterialWithMap*>(&material)) {
        slots.push_back({"map", [m](const std::shared_ptr<Texture>& t) { m->map = t; }, m->map, true});
    }
    if (auto* m = dynamic_cast<MaterialWithNormalMap*>(&material)) {
        slots.push_back({"normalMap", [m](const std::shared_ptr<Texture>& t) { m->normalMap = t; },
                         m->normalMap, false});
    }
    if (auto* m = dynamic_cast<MaterialWithRoughness*>(&material)) {
        slots.push_back({"roughnessMap", [m](const std::shared_ptr<Texture>& t) { m->roughnessMap = t; },
                         m->roughnessMap, false});
    }
    if (auto* m = dynamic_cast<MaterialWithMetalness*>(&material)) {
        slots.push_back({"metalnessMap", [m](const std::shared_ptr<Texture>& t) { m->metalnessMap = t; },
                         m->metalnessMap, false});
    }
    if (auto* m = dynamic_cast<MaterialWithAoMap*>(&material)) {
        slots.push_back({"aoMap", [m](const std::shared_ptr<Texture>& t) { m->aoMap = t; },
                         m->aoMap, false});
    }
    if (auto* m = dynamic_cast<MaterialWithEmissive*>(&material)) {
        slots.push_back({"emissiveMap", [m](const std::shared_ptr<Texture>& t) { m->emissiveMap = t; },
                         m->emissiveMap, true});
    }

    return slots;
}

std::string threepp::editor::textureSlotFromFilename(const std::string& stem) {

    const auto token = lastToken(stem);
    if (token.empty()) return {};

    for (const auto& entry : aliases) {
        for (const char* suffix : entry.suffixes) {
            if (!suffix) break;
            if (token == suffix) return entry.slot;
        }
    }

    return {};
}
