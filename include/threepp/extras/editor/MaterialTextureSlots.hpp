// The texture slots the editor exposes on a material, in one place.
//
// Two things need this list and they must not drift: the inspector, which draws
// a row per slot, and the file-drop handler, which has to work out which slot a
// dropped image belongs in. Before this existed only the inspector knew, so a
// dropped texture had nowhere to go but `map`.
//
// Each entry carries the colour-space the slot needs. That is not cosmetic:
// a normal or roughness map tagged sRGB is decoded on sampling and the shading
// comes out wrong, so anything writing a slot has to honour it.

#ifndef THREEPP_EDITOR_MATERIALTEXTURESLOTS_HPP
#define THREEPP_EDITOR_MATERIALTEXTURESLOTS_HPP

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace threepp {

    class Material;
    class Texture;

}// namespace threepp

namespace threepp::editor {

    struct MaterialTextureSlot {

        // Stable identifier — the inspector's label, the undo entry's name, and
        // what filename matching compares against.
        std::string name;

        std::function<void(const std::shared_ptr<Texture>&)> set;
        std::shared_ptr<Texture> current;

        // True for slots holding colour (base colour, emissive), false for the
        // ones holding data (normals, roughness, metalness, occlusion).
        bool srgb = true;
    };

    // Every slot `material` actually has, in the order the inspector shows them.
    [[nodiscard]] std::vector<MaterialTextureSlot> textureSlotsOf(Material& material);

    // The slot a file named like `stem` is probably meant for — "brick_normal"
    // -> "normalMap". Empty when nothing matches, which means the caller should
    // fall back to the base colour map rather than guess.
    //
    // Deliberately suffix-driven and conservative: a wrong guess is undoable and
    // logged, but a guess nobody asked for is still worse than no guess at all.
    [[nodiscard]] std::string textureSlotFromFilename(const std::string& stem);

}// namespace threepp::editor

#endif//THREEPP_EDITOR_MATERIALTEXTURESLOTS_HPP
