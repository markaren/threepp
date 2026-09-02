// Where a splat cloud came from, and what the importer did to it on the way in.
//
// This mark is what ObjectExporter serializes a SplatCloud FROM: it writes a
// `threeppSplat` block holding the source path (relative to the document when
// it can be) and the ops to replay, and ObjectLoader re-imports the file and
// replays them on open. "Re-load this file and re-apply these operations" is
// the whole of what a document needs to store for an imported scan, because a
// .ply on disk is a far better container for a million Gaussians than three.js
// JSON is. A cloud with no source (a procedural one) gets a sidecar .ply next
// to the document, or a member of the .tpz. The keys are read by name in the
// library (ObjectJsonConstants.hpp's splatSourceKey / splatOpsKey), so they
// must not change here without changing there.
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
        //
        // TRUE for a SOG scan as well, despite SOG v2 declaring itself +Y up:
        // that declaration is about the container, not about which way the
        // capture inside it was reconstructed, and splat-transform re-encodes a
        // 3DGS .ply without reorienting it. A COLMAP-derived SOG is therefore
        // +Y down exactly like the .ply it came from.
        bool flippedX = false;

        // Which detail level was read, for a SOG asset that declares several.
        // -1 for a .ply and for a lone SOG chunk, neither of which has levels —
        // a re-import has to be able to tell "level 0" from "no levels at all",
        // and 0 cannot say both.
        int lod = -1;

        // The file held a colour-only point cloud (no f_dc_0), imported
        // through SplatLoader::loadPointCloudPly: one isotropic degree-0
        // Gaussian per point, sized from the median neighbour spacing. A
        // re-import has to take that loader again rather than the splat one.
        bool pointCloud = false;

        static constexpr const char* sourceKey = "splatSource";
        static constexpr const char* opsKey = "splatImportOps";

        // nullopt when the object carries no splat import mark.
        [[nodiscard]] static std::optional<SplatImportConfig> read(const Object3D& object);

        void write(Object3D& object) const;

        static void erase(Object3D& object);
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SPLATIMPORTCONFIG_HPP
