// https://github.com/mrdoob/three.js/blob/r129/src/loaders/ObjectLoader.js

#ifndef THREEPP_OBJECTLOADER_HPP
#define THREEPP_OBJECTLOADER_HPP

#include "threepp/utils/ZipReader.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace threepp {

    class Object3D;
    class SplatCloud;

    // Reads three.js' "Object" JSON scene format (metadata.version 4.5) —
    // geometries, materials, textures, images, skeletons, animations and the
    // object tree. Unknown entries are reported and skipped, never fatal.
    //
    // Every serialized uuid is adopted verbatim (objects, geometries, materials,
    // textures, skeletons, clips), so a document round-trips without changing
    // identity.
    class ObjectLoader {

    public:
        // Returns nullptr when the document is malformed or carries no `object`.
        std::shared_ptr<Object3D> parse(const std::string& jsonText);

        // A JSON document, or a .tpz archive holding one — decided by sniffing
        // the file, not by its name. The archive's images/ and buffers/ take the
        // place of the directory the loose document would resolve urls against;
        // the JSON inside is identical either way.
        std::shared_ptr<Object3D> load(const std::filesystem::path& path);

        // Base directory for image urls that are plain relative paths rather
        // than data-URIs. Defaults to the directory of the file passed to load().
        void setResourcePath(const std::filesystem::path& path);

        // Everything the last parse()/load() could not represent (unknown types,
        // dangling uuid references, undecodable images, ...). Also written to
        // std::cerr. Cleared at the start of every parse.
        [[nodiscard]] const std::vector<std::string>& warnings() const;

        // A splat cloud is written as a reference to its file, and reading a
        // scan back is seconds and a gigabyte. A caller that still HOLDS the
        // cloud the document describes — the editor's play snapshot, which
        // restores a scene it captured a moment ago — installs this to hand it
        // back by uuid instead; the loader then applies the document's
        // placement, look and userData to that object as if it had loaded it.
        // Return nullptr to fall through to the file.
        using SplatCloudResolver = std::function<std::shared_ptr<SplatCloud>(const std::string& uuid)>;
        void setSplatCloudResolver(SplatCloudResolver resolver);

    private:
        std::filesystem::path resourcePath_;
        SplatCloudResolver splatResolver_;
        // Set only for the duration of an archive load(); parse() on its own has
        // nothing but the text it was handed. The path travels with the reader
        // because a subtree re-imported out of the archive has to record which
        // archive that was, not just that it was one.
        std::shared_ptr<ZipReader> archive_;
        std::filesystem::path archivePath_;
        std::vector<std::string> warnings_;
    };

}// namespace threepp

#endif//THREEPP_OBJECTLOADER_HPP
