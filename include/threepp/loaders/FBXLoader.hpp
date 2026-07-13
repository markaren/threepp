#pragma once

#include <filesystem>
#include <memory>

namespace threepp {

    class Group;

    /**
     * Static-mesh FBX importer (built on OpenFBX).
     *
     * Imports mesh geometry, per-material textures, and lights, baking each
     * node's world transform into its mesh. It is deliberately limited to
     * static scene content: node hierarchy, animation, skinning/bones,
     * blend-shape morphs, cameras, and geometry instancing are NOT imported
     * (parent-child transforms are flattened into world space).
     *
     * For rigged, animated, or instance-heavy assets, convert to glTF and load
     * with GLTFLoader instead — see the scripts/ folder (mixamo_to_glb.py,
     * prop_to_glb.py) for the Blender-based conversion pipeline. glTF is the
     * fully-supported runtime path.
     */
    class FBXLoader {
    public:
        // Controls how the FBX SPECULAR texture slot is interpreted.
        // FBX has no standard way to mark a texture as a traditional specular
        // map versus a PBR ORM pack (R=AO, G=Roughness, B=Metalness), so the
        // loader must choose. Auto guesses from the filename; Phong/PBR force it.
        enum class MaterialMode {
            Auto, ///< Guess per-material from the SPECULAR texture filename (default).
            Phong,///< Always treat the SPECULAR slot as a traditional specular map (MeshPhongMaterial).
            PBR,  ///< Always treat the SPECULAR slot as ORM-packed roughness/metalness (MeshPhysicalMaterial).
        };

        MaterialMode materialMode = MaterialMode::Auto;

        // Multiplier applied to every emissive material's intensity. FBX files
        // commonly author emissive factors far below what reads as light in a
        // path tracer (the Amazon Bistro's Falcor scene multiplies them by 1000),
        // so this lets callers boost emitters without editing the asset.
        // 1.0 keeps the file's authored values.
        float emissiveScale = 1.0f;

        FBXLoader();
        ~FBXLoader();

        std::shared_ptr<Group> load(const std::filesystem::path& path);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp
