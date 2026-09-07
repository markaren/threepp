// https://github.com/mrdoob/three.js/blob/r129/src/core/Object3D.js (toJSON)

#ifndef THREEPP_OBJECTEXPORTER_HPP
#define THREEPP_OBJECTEXPORTER_HPP

#include <filesystem>
#include <string>
#include <vector>

namespace threepp {

    class Object3D;
    class ZipWriter;

    // How a texture's pixels reach the document.
    enum class ImageStorage {

        // Base64 PNG data-URI in the `images` array. The document stands on its
        // own — it renders with no other file present — at the cost of being
        // roughly 4/3 the size of the source images and of re-encoding every
        // texture to PNG on each save.
        Embed,

        // A path to the file the texture was loaded from, relative to the
        // document. Cheap to write and to read, but the document is only
        // complete alongside its textures. Falls back to Embed (with a warning)
        // for any texture that has no `sourceFile` — a procedural texture, or
        // one that lived inside a .glb/.fbx and never existed as its own file.
        Reference,

        // No image entry at all. For callers that keep the live Texture objects
        // themselves and only need the graph — see editor::SceneSnapshot.
        Omit,
    };

    // How an imported model's geometry reaches the document.
    enum class ModelStorage {

        // Every vertex is written into the `geometries` array. Self-contained,
        // and by far the largest thing in a document holding real models.
        Embed,

        // Subtrees marked with assetSource() are written as a reference to the
        // source .glb/.fbx/.urdf plus a table of per-node edits; ObjectLoader
        // re-imports the file and replays the edits. Orders of magnitude
        // smaller and faster for imported models, but the document is only
        // complete alongside the assets it points at.
        //
        // Objects with no assetSource() are unaffected and always inline —
        // primitives and edited meshes have no file to point back at.
        Reference,
    };

    // Which container save() writes the document into.
    enum class DocumentFormat {

        // Archive when the target is named ".tpz", plain JSON otherwise.
        Auto,

        // The JSON document alone. Whatever `images`/`models` ask for.
        Json,

        // One .tpz file: an uncompressed zip holding scene.json next to the
        // images and the geometry, so the document is self-contained AND cheap.
        // Images go in as their ORIGINAL encoded bytes (no base64, no PNG
        // re-encode) and geometry as raw little-endian binary, which is what
        // makes it cheap; both override `images`, except for ImageStorage::Omit.
        // toJson() has nowhere to put the extra members, so this only affects
        // save().
        Archive,
    };

    struct ObjectExporterOptions {

        ImageStorage images = ImageStorage::Embed;

        ModelStorage models = ModelStorage::Embed;

        DocumentFormat format = DocumentFormat::Auto;

        // Directory that Reference paths are written relative to. save() fills
        // this in from the output file when it is left empty; toJson() has no
        // file to infer it from, so set it there or get absolute paths.
        std::filesystem::path resourcePath;

        bool prettyPrint = false;
    };

    // Writes a scene graph as three.js' "Object" JSON scene format
    // (metadata.version 4.5) — the counterpart of ObjectLoader.
    //
    // Any Object3D may be the root: exporting a child Group writes that subtree
    // plus exactly the geometries/materials/textures/skeletons/clips it
    // references. Output is deterministic — the same scene always produces a
    // byte-identical document, so autosaves diff cleanly.
    class ObjectExporter {

    public:
        [[nodiscard]] std::string toJson(Object3D& object, const ObjectExporterOptions& options = {});

        // Writes the JSON document, or — for a ".tpz" target, or when the
        // options say so — the single-file archive. The archive is written
        // through a temp file and renamed over the target, so a crash mid-save
        // cannot destroy the archive that was already there.
        void save(Object3D& object, const std::filesystem::path& path, const ObjectExporterOptions& options = {});

        // Everything the last toJson()/save() could not represent (a texture
        // without CPU-side pixels, a userData entry of an unsupported type,
        // ShaderMaterial uniforms, ...). Also written to std::cerr. Cleared at
        // the start of every export.
        [[nodiscard]] const std::vector<std::string>& warnings() const;

    private:
        // The one export routine. `archive` non-null routes images and geometry
        // into it as their own members, leaving urls behind in the JSON.
        std::string write(Object3D& object, const ObjectExporterOptions& options, ZipWriter* archive);

        std::vector<std::string> warnings_;
    };

}// namespace threepp

#endif//THREEPP_OBJECTEXPORTER_HPP
