// Where a splat cloud came from, and what the importer did to it on the way in.
//
// A SplatCloud is NOT serialized yet — ObjectExporter skips it and says so, and
// the object is lost on save and on Play-restore. This mark exists anyway, and
// costs nothing now: it is the hook the serialization pass will read. When that
// pass lands, "re-load this file and re-apply these operations" is the whole of
// what a document needs to store for an imported scan, because a .ply on disk
// is a far better container for a million Gaussians than three.js JSON is.
//
// Only what CANNOT be recovered from the live object is recorded. The splat
// count and the SH degree are already on the cloud (splatCount(), data()), and
// duplicating them here would just create two answers that can disagree.
//
// Deliberately NOT setAssetSource(). That mark means "ObjectLoader can rebuild
// this subtree by re-importing the file", which is not true of a splat yet —
// ModelLoader does not dispatch on .ply, so a ModelStorage::Reference save
// would write a reference nothing can load back. Stamping it would trade a
// visible gap for a silent one.
//
// Storage follows RobotConfig: two userData entries rather than one packed
// string, because a Windows path contains the characters the flat key=value
// format uses as delimiters.
//
//   userData["splatSource"]     C:\scans\atlas\scene.ply   - verbatim
//   userData["splatImportOps"]  cull=1;removed=812;flipX=1 - what was applied

#ifndef THREEPP_EDITOR_SPLATIMPORTCONFIG_HPP
#define THREEPP_EDITOR_SPLATIMPORTCONFIG_HPP

#include <cstddef>
#include <optional>
#include <string>

namespace threepp {

    class Object3D;

}

namespace threepp::editor {

    struct SplatImportConfig {

        // The .ply the cloud was loaded from, stored verbatim.
        std::string source;

        // SplatData::removeOutliers() was run, and how many splats it dropped.
        // Recorded together: a cull that removed nothing and no cull at all are
        // the same picture but not the same document.
        bool culled = false;
        std::size_t removed = 0;

        // The conventional half-turn about X. Photogrammetry rigs (COLMAP, and
        // the 3DGS pipelines on top of it) put +Y down, so a scan arrives
        // upside down in a +Y-up scene. The flip is applied to the NODE, not to
        // the data, so it also lives in the object's own rotation — this flag
        // is what says the rotation was the importer's doing rather than the
        // user's, which is what a future re-import has to know.
        bool flippedX = false;

        static constexpr const char* sourceKey = "splatSource";
        static constexpr const char* opsKey = "splatImportOps";

        // nullopt when the object carries no splat import mark.
        [[nodiscard]] static std::optional<SplatImportConfig> read(const Object3D& object);

        void write(Object3D& object) const;

        static void erase(Object3D& object);
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SPLATIMPORTCONFIG_HPP
