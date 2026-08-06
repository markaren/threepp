// What the editor will import, in one place.
//
// This list used to be written out three times — the Import dialog's filter,
// the drag-and-drop handler and the asset browser — and they drifted: USD was
// supported by ModelLoader but offered nowhere, and FBX was accepted by a drop
// but hidden in the file dialog. One definition, no drift.

#ifndef THREEPP_EDITOR_IMPORTFORMATS_HPP
#define THREEPP_EDITOR_IMPORTFORMATS_HPP

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace threepp::editor::formats {

    // Lower-cased extension of `path`, dot included ("" when there is none).
    inline std::string extensionOf(const std::filesystem::path& path) {

        auto extension = path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return extension;
    }

    // Everything ModelLoader dispatches on.
    inline const std::vector<std::string>& meshes() {

        static const std::vector<std::string> list{
                ".obj", ".dae", ".gltf", ".glb", ".stl", ".fbx",
                ".usd", ".usda", ".usdc", ".usdz"};
        return list;
    }

    // Robot descriptions, which go to URDFLoader instead and come back as an
    // articulated Robot rather than a plain Group.
    inline const std::vector<std::string>& robots() {

        static const std::vector<std::string> list{".urdf", ".xacro"};
        return list;
    }

    // 3D Gaussian Splat scans, which go to SplatLoader or SogLoader and come
    // back as a single SplatCloud rather than a Group of meshes.
    //
    // Its own category rather than an entry in meshes() because the extension
    // does not settle it: Turk's PLY carries triangle soup, laser point clouds
    // and 3DGS output alike, and the same file name means all three. The
    // extension only gets a file as far as the import; which loader runs is
    // decided by sniffing — f_dc_0 in the header for a PLY
    // (SplatLoader::isSplatPly), the meta.json for a SOG (SogLoader::isSog).
    //
    // ".zip" earns its place the same way and is the widest net here, because a
    // SOG scan downloaded from superspl.at arrives under exactly that name —
    // the spec's bundled ".sog" is the same container, and neither extension is
    // reliable. Nothing is claimed on the strength of the name: a zip holding
    // anything else is refused by the import with a message that says so, which
    // is a better answer than the drop being silently ignored.
    //
    // A SOG asset can also be a DIRECTORY, which has no extension at all. That
    // case is handled at the drop and dialog sites rather than here, since a
    // list of extensions cannot express it.
    inline const std::vector<std::string>& splats() {

        static const std::vector<std::string> list{".ply", ".sog", ".zip"};
        return list;
    }

    // Behaviour scripts. Not importable — a script is attached to an existing
    // object rather than added to the scene, so it is its own category.
    inline const std::vector<std::string>& scripts() {

        static const std::vector<std::string> list{".py"};
        return list;
    }

    // Audio files. Exactly the three decoders the vendored miniaudio builds in
    // (MA_NO_WAV / MA_NO_MP3 / MA_NO_FLAC are never defined) — offering
    // anything else would mean a file dialog that accepts what the loader
    // throws on. Not importable either: a sound is attached to an existing
    // object, like a script.
    inline const std::vector<std::string>& sounds() {

        static const std::vector<std::string> list{".wav", ".mp3", ".flac"};
        return list;
    }

    // Textures — anything ImageLoader decodes into a material map. Everything up to
    // ".gif" is stb_image's doing; ".webp" is the vendored libwebp decoder
    // (src/external/libwebp), which ImageLoader dispatches to by sniffing the RIFF
    // magic rather than by extension. The name only has to get the file as far as a
    // file dialog or a drop; the loader decides what it actually is.
    inline const std::vector<std::string>& images() {

        static const std::vector<std::string> list{
                ".png", ".jpg", ".jpeg", ".bmp", ".tga", ".gif", ".webp"};
        return list;
    }

    // HDR images, which become the scene environment rather than a material map.
    // Both decode to linear float RGBA equirects (RGBELoader / EXRLoader), so
    // everything downstream treats them identically.
    inline const std::vector<std::string>& environments() {

        static const std::vector<std::string> list{".hdr", ".exr"};
        return list;
    }

    // Everything the Import action accepts: mesh, robot and splat alike.
    inline const std::vector<std::string>& importable() {

        static const std::vector<std::string> list = [] {
            std::vector<std::string> all = meshes();
            all.insert(all.end(), robots().begin(), robots().end());
            all.insert(all.end(), splats().begin(), splats().end());
            return all;
        }();
        return list;
    }

    inline bool contains(const std::vector<std::string>& list, const std::string& extension) {

        return std::find(list.begin(), list.end(), extension) != list.end();
    }

    inline bool isRobot(const std::filesystem::path& path) {

        return contains(robots(), extensionOf(path));
    }

    // The extension only — it says the file MIGHT be a splat scan. The header
    // is what says it is.
    inline bool isSplatCandidate(const std::filesystem::path& path) {

        return contains(splats(), extensionOf(path));
    }

    inline bool isImportable(const std::filesystem::path& path) {

        return contains(importable(), extensionOf(path));
    }

    inline bool isEnvironment(const std::string& extension) {

        return contains(environments(), extension);
    }

    inline bool isImage(const std::string& extension) {

        return contains(images(), extension);
    }

}// namespace threepp::editor::formats

#endif//THREEPP_EDITOR_IMPORTFORMATS_HPP
