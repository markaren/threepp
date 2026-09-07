// Terrain authoring, stored on the mesh itself.
//
// A terrain is an ordinary Mesh carrying `userData["terrain"]` — the flat
// `key=value;…` encoding of `threepp::terrain::TerrainParams` plus an `eroded`
// flag recording whether the stored bake ran the (slow) erosion pass. The
// geometry is BAKED into the document like Text's glyphs and the tree's trunk,
// so a saved scene renders, collides and plays with no editor present. The
// config exists so the terrain can be RE-EDITED. Opening a scene never
// regenerates — the triangles in the file are the truth.
//
// Carrying the entry is the whole definition; there is no "enabled" flag and
// write() never erases, for SplineConfig's reason: the entry IS the terrain.
//
// ── Delta recovery — the load-bearing idea ─────────────────────────────────
//
// Generation is deterministic per seed (noise and erosion both), so the user's
// SCULPT layer never needs separate storage. It is the difference between the
// mesh as it stands and the mesh the stored config would have produced:
//
//     delta      = currentHeights − base(configBefore)
//     newHeights = base(configAfter) + delta
//
// Recovered from the BEFORE config, never the after — the after has not been
// baked yet, and subtracting it would fold the parameter change into the
// "sculpt". On a resolution change the delta grid is bilinearly resampled.
// Nobody loses an hour of sculpting because they touched a noise slider.
//
// Vertex grid and height lattice are index-identical (dim = resolution+1,
// vertex iz*dim+ix — TerrainGenerator's displaceTo already relies on
// PlaneGeometry's vertex order). Only Y is ever rewritten; X and Z are the
// immutable grid.

#ifndef THREEPP_EDITOR_TERRAINCONFIG_HPP
#define THREEPP_EDITOR_TERRAINCONFIG_HPP

#include "threepp/extras/terrain/TerrainGenerator.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace threepp {

    class BufferGeometry;
    class Object3D;
    class Texture;

}// namespace threepp

namespace threepp::editor {

    struct TerrainConfig {

        terrain::TerrainParams params;
        // Whether the stored bake ran erode(). Kept out of TerrainParams because
        // it is not a generator knob but a record of what was actually done:
        // erosion is a ~1 s pass that runs behind a button, so a slider drag
        // re-bakes RAW and this drops to 0 until Generate is pressed again.
        bool eroded = false;

        static constexpr const char* userDataKey = "terrain";

        // --- presets ---------------------------------------------------------
        // terrain::applyPreset's four, by the same index. A preset gives the
        // LANDSCAPE CHARACTER; the editor keeps its own scale (see
        // applyPreset below) — the hero presets are 1200 m across and would
        // dwarf a scene of 1 m boxes.
        static constexpr int presetCount = 4;
        static const char* presetLabel(int preset);
        void applyPreset(int preset);

        // Editor-scale starting terrain: frames in the default camera, reads as
        // ground rather than as a wall. NOT the generator's hero defaults.
        [[nodiscard]] static TerrainConfig makeDefault();

        // --- document round trip ---------------------------------------------
        [[nodiscard]] std::string encode() const;
        [[nodiscard]] static std::optional<TerrainConfig> decode(const std::string& text);

        // nullopt when the object is not a terrain.
        [[nodiscard]] static std::optional<TerrainConfig> read(const Object3D& object);

        // Always writes the entry — see the header note.
        void write(Object3D& object) const;

        [[nodiscard]] static bool isTerrain(const Object3D& object);

        // --- the deterministic bake -------------------------------------------
        // dim = resolution+1: one lattice cell per mesh vertex, one albedo texel
        // per mesh vertex.
        [[nodiscard]] int dim() const;

        struct Bake {
            std::shared_ptr<BufferGeometry> geometry;
            // Y per vertex, in vertex order. The sculpt layer's coordinate space.
            std::vector<float> heights;
            // Normalised [0,1] field the splat bake reads.
            std::vector<float> field;
            std::vector<unsigned char> albedo;// dim*dim*4, sRGB
            int dim = 0;
        };

        // buildField + (erode iff eroded) + geometry + splat. THE path — every
        // regeneration and every delta recovery replays exactly this, which is
        // what makes `base(config)` a well-defined quantity.
        [[nodiscard]] Bake bake() const;

        // Splat colours for an arbitrary field (i.e. after sculpting). Cheap:
        // no noise, no erosion — just the slope/altitude bands.
        [[nodiscard]] std::vector<unsigned char> bakeAlbedo(const std::vector<float>& field) const;

        // --- height lattice helpers -------------------------------------------
        [[nodiscard]] static std::vector<float> heightsOf(const BufferGeometry& geometry);
        static void setHeights(BufferGeometry& geometry, const std::vector<float>& heights);
        // Y → normalised field, inverting displaceTo (see TerrainGenerator's
        // baseSink note). Needs the geometry for the X/Z of each vertex.
        [[nodiscard]] std::vector<float> fieldOf(const BufferGeometry& geometry,
                                                 const std::vector<float>& heights) const;
        // Bilinear resample of a square lattice. Used on the sculpt delta when
        // the resolution changes under it.
        [[nodiscard]] static std::vector<float> resample(const std::vector<float>& src,
                                                         int srcDim, int dstDim);

        // --- albedo texture ----------------------------------------------------
        // The material's map. A freshly baked terrain's is a DataTexture; a
        // reloaded one's is whatever the loader made of the embedded image, so
        // this is deliberately the base type.
        [[nodiscard]] static std::shared_ptr<Texture> albedoTexture(const Object3D& object);
        // Uploads `albedo` into the node's map, creating (or replacing) the
        // texture when the dimensions moved. No-op off a MeshStandardMaterial —
        // a terrain the user re-materialled is theirs.
        static void applyAlbedo(Object3D& object, const std::vector<unsigned char>& albedo, int dim);

        // --- regeneration -------------------------------------------------------
        // Delta recovery + re-bake + geometry swap ON THE SAME NODE (uuid,
        // material and userData preserved). Returns the resulting geometry, or
        // nullptr when `object` is not a Mesh. Headless — no GL, no imgui.
        static std::shared_ptr<BufferGeometry> rebuild(Object3D& object,
                                                       const TerrainConfig& before,
                                                       const TerrainConfig& after);

        bool operator==(const TerrainConfig&) const = default;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_TERRAINCONFIG_HPP
