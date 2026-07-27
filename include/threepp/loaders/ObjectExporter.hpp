// https://github.com/mrdoob/three.js/blob/r129/src/core/Object3D.js (toJSON)

#ifndef THREEPP_OBJECTEXPORTER_HPP
#define THREEPP_OBJECTEXPORTER_HPP

#include <filesystem>
#include <string>
#include <vector>

namespace threepp {

    class Object3D;

    struct ObjectExporterOptions {

        // Embed every texture image as a base64 PNG data-URI in the `images`
        // array. When false, textures whose image data is CPU-side are still
        // referenced but their image entry is omitted, so the JSON stays small
        // at the cost of not being self-contained.
        bool embedImages = true;

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

        void save(Object3D& object, const std::filesystem::path& path, const ObjectExporterOptions& options = {});

        // Everything the last toJson()/save() could not represent (a texture
        // without CPU-side pixels, a userData entry of an unsupported type,
        // ShaderMaterial uniforms, ...). Also written to std::cerr. Cleared at
        // the start of every export.
        [[nodiscard]] const std::vector<std::string>& warnings() const;

    private:
        std::vector<std::string> warnings_;
    };

}// namespace threepp

#endif//THREEPP_OBJECTEXPORTER_HPP
