#ifndef THREEPP_WGPUPATHTRACERATLAS_HPP
#define THREEPP_WGPUPATHTRACERATLAS_HPP

// Private header — texture-atlas builder and material-extraction helpers used
// by the path tracer's per-frame scene rebuild. Not part of the public API.

#include "threepp/math/Color.hpp"
#include "threepp/math/Matrix4.hpp"

#include <cstdint>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace threepp {
    class Material;
    class Mesh;
    class Texture;
}

namespace threepp::wgpu_pt {

    /// Per-instance entry. Each InstancedMesh produces N entries (one per
    /// instance), each with its own effective world matrix; regular meshes
    /// produce a single entry. Populated by expandMeshEntries().
    struct RtMeshEntry {
        Mesh* mesh;
        Matrix4 worldMatrix;
    };

    /// Expand a flat mesh list so InstancedMesh becomes N separate entries
    /// (one per instance), each with its own effective world matrix. Regular
    /// meshes pass through as a single entry.
    std::vector<RtMeshEntry> expandMeshEntries(const std::vector<Mesh*>& meshes);

    /// Build the texture atlas sized to actual slot usage (not MAX_TEX_SLOTS).
    /// Returns {pixel data, layer count, columns-per-layer, tile size}.
    /// Populates `texSlotMap` so material extraction can later resolve textures.
    std::tuple<std::vector<unsigned char>, int, int, int> buildAtlas(
            const std::vector<Mesh*>& meshes,
            std::unordered_map<Texture*, int>& texSlotMap,
            int tileSize);

    /// Encode an atlas slot index plus wrapS/wrapT modes into a single float.
    /// Layout: slot * 16 + wrapS * 4 + wrapT  (wrap: 0=repeat, 1=clamp, 2=mirror)
    float encodeSlotWrap(int slot, const Texture* tex);

    /// Material parameters extracted from a threepp Material via dynamic_cast
    /// against the various material interfaces. All physically-based fields are
    /// pre-clamped/pre-squared into the form the path tracer expects.
    struct ExtractedMaterial {
        Color albedo{0.8f, 0.8f, 0.8f};
        float shininess = 8.f;
        float metalness = 0.f;
        Color emissive{0.f, 0.f, 0.f};
        float transmission = 0.f;
        float ior = 1.5f;
        float alphaTest = 0.f;
        Color attenuationColor{1.f, 1.f, 1.f};
        float attenuationDistance = 0.f;
        float clearcoat = 0.f;
        float clearcoatRoughness = 0.f;
        Color sheenColor{0.f, 0.f, 0.f};
        float sheenRoughness = 0.f;
        float specularIntensity = 1.f;
        Color specularColor{1.f, 1.f, 1.f};
        float dispersion = 0.f;
        float thickness = 0.f;
    };

    /// Extract path-tracer material parameters from a threepp Material.
    /// MeshBasicMaterial is signalled as unlit by setting shininess = -1.
    ExtractedMaterial extractMaterial(const Material* mat);

}// namespace threepp::wgpu_pt

#endif//THREEPP_WGPUPATHTRACERATLAS_HPP
