// Procedural tree authoring, stored on the object itself.
//
// A tree is a Group carrying `userData["tree"]` — the encoded TreeParams the
// vegetation generator takes. Under it sit TWO derived children tagged
// `userData["treeDerived"]`, "trunk" and "leaves", because the two halves want
// different materials: opaque bark, and alpha-tested double-sided foliage. They
// are real document nodes like the spline's tube, so a saved scene renders (and
// collides, and can carry physics) with no editor present; the config is only
// needed to EDIT the tree again, and it travels in the same file.
//
// What makes a Group a tree is the presence of the entry. There is no "enabled"
// flag and write() never erases, for SplineConfig's reason: the entry IS the
// tree.
//
// Storage is the flat `key=value;…` string the Config family shares, since
// userData round-trips scalars only (see ObjectExporter::writeUserData). It is a
// long one — TreeParams has forty-odd knobs — but every key is the C++ field
// name, so a saved document reads as the parameter block it is. Unknown keys are
// ignored on read, so a document written by a newer editor still loads.
//
// GEOMETRY AND TEXTURES ARE ONE UNIT OF WORK. Both meshes come off a single
// skeleton (build() below), because growing it twice would give the trunk and
// the leaves different trees. The bark and leaf atlases are procedural too, and
// they are a function of only a handful of the params — textureKey() names that
// subset, so changing the trunk height does not redraw the bark, and two trees
// of the same species SHARE the atlases rather than each embedding their own
// copy in the saved file.

#ifndef THREEPP_EDITOR_TREECONFIG_HPP
#define THREEPP_EDITOR_TREECONFIG_HPP

#include "threepp/extras/vegetation/TreeGenerator.hpp"

#include <memory>
#include <optional>
#include <string>

namespace threepp {

    class BufferGeometry;
    class DataTexture;
    class Material;
    class MeshStandardMaterial;
    class Object3D;

}// namespace threepp

namespace threepp::editor {

    struct TreeConfig {

        // Which half of the tree a derived child is. They are separate meshes
        // because they are separately materialled, not because they are
        // separately generated — see build().
        enum class Part {
            Trunk,
            Leaves
        };

        vegetation::TreeParams params;

        static constexpr const char* userDataKey = "tree";
        // Marks a generated child. The value is the part token, so the sync
        // pass can tell the two apart without depending on child order.
        static constexpr const char* derivedKey = "treeDerived";

        // --- species presets -------------------------------------------------
        // vegetation::applyPreset's four, by the same index. A preset RESETS
        // every field but the seed (see its note), which is why this is an
        // action the inspector offers rather than a stored mode: after applying
        // one, the params alone describe the tree.
        static constexpr int presetCount = 4;
        static const char* presetLabel(int preset);

        // --- document round trip ---------------------------------------------
        [[nodiscard]] std::string encode() const;
        [[nodiscard]] static std::optional<TreeConfig> decode(const std::string& text);

        // nullopt when the object is not a tree.
        [[nodiscard]] static std::optional<TreeConfig> read(const Object3D& object);

        // Always writes the entry — see the header note.
        void write(Object3D& object) const;
        static void erase(Object3D& object);

        // Carrying the entry is the whole definition.
        [[nodiscard]] static bool isTree(const Object3D& object);

        // --- derived children -------------------------------------------------
        [[nodiscard]] static bool isDerived(const Object3D& object);
        static void markDerived(Object3D& object, Part part);
        // The generated mesh for `part`, or nullptr. First tagged match wins;
        // the editor keeps there being exactly one of each.
        [[nodiscard]] static Object3D* derivedPart(const Object3D& tree, Part part);

        static const char* partToken(Part part);
        static const char* label(Part part);

        // --- generation --------------------------------------------------------
        struct Geometries {
            std::shared_ptr<BufferGeometry> trunk;
            std::shared_ptr<BufferGeometry> leaves;
        };

        // Both skins off ONE skeleton — the trunk and its foliage have to be
        // the same tree. Deterministic in params.seed, so the same config
        // always builds the same geometry.
        [[nodiscard]] Geometries build() const;

        // --- procedural textures ------------------------------------------------
        struct Textures {
            std::shared_ptr<DataTexture> barkAlbedo;
            std::shared_ptr<DataTexture> barkNormal;
            std::shared_ptr<DataTexture> leaf;
        };

        // Exactly the params the atlases are drawn from. Two configs with the
        // same key share textures; a config whose key moved needs them redrawn.
        [[nodiscard]] std::string textureKey() const;

        // Cached on textureKey(), weakly: a forest of one species draws three
        // atlases, and the exporter emits one image entry per distinct texture,
        // so sharing is what keeps a saved scene from carrying a base64 PNG per
        // tree. Entries die with the last material holding them.
        [[nodiscard]] Textures textures() const;

        // Fresh materials for a new tree: bark opaque, leaves alpha-tested and
        // double-sided with the baked canopy occlusion read from vertex colours.
        [[nodiscard]] std::shared_ptr<MeshStandardMaterial> makeBarkMaterial() const;
        [[nodiscard]] std::shared_ptr<MeshStandardMaterial> makeLeafMaterial() const;

        // Re-point an EXISTING material's maps at this config's atlases,
        // leaving everything else the user set on it alone. No-op on a material
        // that is not a MeshStandardMaterial — a tree whose bark the user
        // replaced outright is theirs, not ours to overwrite.
        static void applyTextures(Material* bark, Material* leaves, const TreeConfig& config);

        bool operator==(const TreeConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_TREECONFIG_HPP
