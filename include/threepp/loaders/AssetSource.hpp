// Which file on disk a subtree was imported from.
//
// An imported model is the one thing in a scene document that already exists,
// in better form, somewhere else. Inlining it means writing every vertex out as
// JSON numbers on save and parsing them all back on load — for a real asset
// that is tens of megabytes and most of the time spent on both ends.
//
// So an importer may stamp the root of what it built with the path it came
// from. ObjectExporter under ModelStorage::Reference then writes that path
// instead of the subtree, and ObjectLoader re-imports the file. The mark is a
// plain userData string, so it survives an ordinary (inlined) round trip too —
// a document saved with everything embedded can still be re-saved as
// references later, and the editor can tell you what a subtree is linked to.
//
// Marking a subtree does not by itself change anything: with the default
// ModelStorage::Embed the geometry is still written out in full.

#ifndef THREEPP_ASSETSOURCE_HPP
#define THREEPP_ASSETSOURCE_HPP

#include <filesystem>

namespace threepp {

    class Object3D;

    // Records `path` on `object`. An empty path clears the mark.
    void setAssetSource(Object3D& object, const std::filesystem::path& path);

    // Empty when `object` is not the root of an imported subtree.
    [[nodiscard]] std::filesystem::path assetSource(const Object3D& object);

    void clearAssetSource(Object3D& object);

    // userData key, exposed so tools can recognise the mark without linking
    // against this translation unit.
    inline constexpr const char* assetSourceKey = "assetSource";

}// namespace threepp

#endif//THREEPP_ASSETSOURCE_HPP
