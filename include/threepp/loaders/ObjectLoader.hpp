// https://github.com/mrdoob/three.js/blob/r129/src/loaders/ObjectLoader.js

#ifndef THREEPP_OBJECTLOADER_HPP
#define THREEPP_OBJECTLOADER_HPP

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace threepp {

    class Object3D;

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

        std::shared_ptr<Object3D> load(const std::filesystem::path& path);

        // Base directory for image urls that are plain relative paths rather
        // than data-URIs. Defaults to the directory of the file passed to load().
        void setResourcePath(const std::filesystem::path& path);

        // Everything the last parse()/load() could not represent (unknown types,
        // dangling uuid references, undecodable images, ...). Also written to
        // std::cerr. Cleared at the start of every parse.
        [[nodiscard]] const std::vector<std::string>& warnings() const;

    private:
        std::filesystem::path resourcePath_;
        std::vector<std::string> warnings_;
    };

}// namespace threepp

#endif//THREEPP_OBJECTLOADER_HPP
