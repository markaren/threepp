// A scene frozen as three.js JSON, plus the live textures it referenced.
//
// This is how Play mode is undone. Pressing Play hands the scene to arbitrary
// runtime code (physics, scripts, anything a PlaySession wants to do); pressing
// Stop must put every last transform, material and hierarchy change back. Rather
// than asking each runtime to be reversible, the editor captures the whole scene
// before it starts and rebuilds it afterwards.
//
// The document format is exactly the one used for File ▸ Save (ObjectExporter /
// ObjectLoader), so anything that round-trips a saved scene round-trips a play
// session, and vice versa. Fixing one fixes both.
//
// Images are NOT embedded. Base64-encoding every texture on each Play press
// would cost hundreds of milliseconds and a lot of memory, so the snapshot keeps
// the live std::shared_ptr<Texture> objects by uuid instead and re-binds them
// into the restored materials. The GPU-side texture is never re-uploaded.

#ifndef THREEPP_EDITOR_SCENESNAPSHOT_HPP
#define THREEPP_EDITOR_SCENESNAPSHOT_HPP

#include <memory>
#include <string>
#include <unordered_map>

namespace threepp {

    class Object3D;
    class Scene;
    class SplatCloud;
    class Texture;

}// namespace threepp

namespace threepp::editor {

    class SceneSnapshot {

    public:
        // Serialize `scene`. Any editor-only object must already be detached —
        // SceneDocument does that for its callers.
        bool capture(Scene& scene, std::string* error = nullptr);

        // Parse back into a fresh Scene with the captured textures re-bound.
        // Object, geometry, material and texture uuids are all preserved, so the
        // caller can re-resolve a selection by uuid. Returns nullptr on failure.
        [[nodiscard]] std::shared_ptr<Scene> restore(std::string* error = nullptr) const;

        [[nodiscard]] bool valid() const { return !json_.empty(); }

        [[nodiscard]] const std::string& json() const { return json_; }

        void clear();

        // Walk a subtree and collect every texture reachable from its materials,
        // keyed by uuid. Exposed because it is the other half of a "keep the GPU
        // resources, replace the graph" operation and is useful on its own.
        static void collectTextures(Object3D& root,
                                    std::unordered_map<std::string, std::shared_ptr<Texture>>& out);

        // Replace every texture in `root`'s materials whose uuid appears in
        // `textures` with the mapped instance. Slots not present are left alone.
        static void rebindTextures(Object3D& root,
                                   const std::unordered_map<std::string, std::shared_ptr<Texture>>& textures);

    private:
        std::string json_;
        std::unordered_map<std::string, std::shared_ptr<Texture>> textures_;

        // The splat clouds, kept live for the same reason as the textures and
        // a bigger one: the document holds a reference to a file, and reading
        // a scan back is seconds and a gigabyte of host memory plus a GPU
        // re-upload. restore() hands these to the loader by uuid
        // (ObjectLoader::setSplatCloudResolver); the loader re-applies the
        // captured placement and look to the same object.
        std::unordered_map<std::string, std::shared_ptr<SplatCloud>> splats_;
    };

}// namespace threepp::editor

#endif//THREEPP_EDITOR_SCENESNAPSHOT_HPP
