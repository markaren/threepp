#pragma once

#include <filesystem>
#include <memory>

namespace threepp {

    class Group;

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
