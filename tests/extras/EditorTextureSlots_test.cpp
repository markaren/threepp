// Which texture slot a material has, and which one a file name is asking for.
//
// Both feed the editor's file-drop handler. Before this list existed a dropped
// image had nowhere to go but `map`, and it was loaded as sRGB regardless -
// which silently mis-decodes a normal or roughness map.

#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/editor/MaterialTextureSlots.hpp"

#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/textures/Texture.hpp"

#include <algorithm>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    const MaterialTextureSlot* find(const std::vector<MaterialTextureSlot>& slots, const std::string& name) {

        const auto it = std::find_if(slots.begin(), slots.end(),
                                     [&name](const MaterialTextureSlot& slot) { return slot.name == name; });
        return it == slots.end() ? nullptr : &*it;
    }

}// namespace


TEST_CASE("A material reports the texture slots it actually has") {

    SECTION("a standard material has the full PBR set") {

        auto material = MeshStandardMaterial::create();
        const auto slots = textureSlotsOf(*material);

        // The base colour map first: it is what an unrecognised drop falls back
        // to, and that fallback is positional.
        REQUIRE_FALSE(slots.empty());
        CHECK(slots.front().name == "map");

        for (const auto* name : {"map", "normalMap", "roughnessMap", "metalnessMap", "aoMap", "emissiveMap"}) {
            INFO("slot " << name);
            CHECK(find(slots, name) != nullptr);
        }
    }

    SECTION("colour maps decode from sRGB, data maps must not") {

        auto material = MeshStandardMaterial::create();
        const auto slots = textureSlotsOf(*material);

        CHECK(find(slots, "map")->srgb);
        CHECK(find(slots, "emissiveMap")->srgb);

        // The bug this guards: a normal map decoded as sRGB shades wrong, and
        // it looks like a renderer problem rather than an import one.
        CHECK_FALSE(find(slots, "normalMap")->srgb);
        CHECK_FALSE(find(slots, "roughnessMap")->srgb);
        CHECK_FALSE(find(slots, "metalnessMap")->srgb);
        CHECK_FALSE(find(slots, "aoMap")->srgb);
    }

    SECTION("a basic material offers only what it supports") {

        auto material = MeshBasicMaterial::create();
        const auto slots = textureSlotsOf(*material);

        CHECK(find(slots, "map") != nullptr);
        CHECK(find(slots, "normalMap") == nullptr);
        CHECK(find(slots, "roughnessMap") == nullptr);
    }

    SECTION("the setter writes the slot it names") {

        auto material = MeshStandardMaterial::create();
        const auto slots = textureSlotsOf(*material);

        auto texture = Texture::create();
        find(slots, "normalMap")->set(texture);

        CHECK(material->normalMap == texture);
        CHECK(material->map == nullptr);
    }
}


TEST_CASE("A texture file name suggests the slot it belongs in") {

    SECTION("the usual authoring-tool suffixes are recognised") {

        CHECK(textureSlotFromFilename("brick_normal") == "normalMap");
        CHECK(textureSlotFromFilename("brick_nrm") == "normalMap");
        CHECK(textureSlotFromFilename("brick-NormalGL") == "normalMap");
        CHECK(textureSlotFromFilename("wood_roughness") == "roughnessMap");
        CHECK(textureSlotFromFilename("wood.rough") == "roughnessMap");
        CHECK(textureSlotFromFilename("steel_metallic") == "metalnessMap");
        CHECK(textureSlotFromFilename("crate_ao") == "aoMap");
        CHECK(textureSlotFromFilename("sign_emissive") == "emissiveMap");
    }

    SECTION("case and separator do not matter") {

        CHECK(textureSlotFromFilename("Brick_NORMAL") == "normalMap");
        CHECK(textureSlotFromFilename("brick-normal") == "normalMap");
        CHECK(textureSlotFromFilename("brick normal") == "normalMap");
    }

    SECTION("anything else is not guessed at") {

        // No match means the caller falls back to the base colour map, which is
        // the old behaviour and the right default for a plain diffuse texture.
        CHECK(textureSlotFromFilename("checker").empty());
        CHECK(textureSlotFromFilename("brick_diffuse").empty());
        CHECK(textureSlotFromFilename("").empty());
    }

    SECTION("only the trailing token counts") {

        // "normal" here is part of the subject, not a channel tag. Matching
        // anywhere in the name would claim this for normalMap.
        CHECK(textureSlotFromFilename("normal_brick").empty());
        CHECK(textureSlotFromFilename("perfectly_normal_wall").empty());
    }
}
