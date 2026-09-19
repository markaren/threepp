#ifndef THREEPP_GLTFLOADER_HPP
#define THREEPP_GLTFLOADER_HPP

#include "threepp/animation/AnimationClip.hpp"
#include "threepp/loaders/MaterialVariants.hpp"
#include "threepp/threepp.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace threepp {

    struct GLTFResult {
        std::shared_ptr<Group> scene;                          ///< Root node of the loaded model
        std::vector<std::shared_ptr<Group>> scenes;            ///< All scenes in the file
        std::vector<std::shared_ptr<AnimationClip>> animations;///< All animations in the file
        MaterialVariants variants;                             ///< Named material variants (empty if none)
    };

    class GLTFLoader {
    public:
        // Keep integer-typed vertex attributes in their source width instead of
        // widening to float. Today that covers COLOR_0 stored as normalized
        // uint8/uint16 (the glTF-recommended encoding): it loads as a
        // Uint8/Uint16BufferAttribute at 1/4 the memory. Both renderers consume
        // these natively; user code that reads colors should go through
        // FloatAttributeView (or getAttribute<float> + a narrow fallback)
        // rather than assuming float. Set to false to restore the old
        // widen-everything behaviour.
        bool preserveNarrowAttributes = true;

        std::optional<GLTFResult> load(const std::filesystem::path& path);

        // Load a model already held in memory, e.g. fetched over the network.
        // `data` is the whole .glb or .gltf file; which of the two is detected
        // from the GLB magic. External resources (a .gltf's .bin and image
        // files) are resolved against `basePath`; with the default they resolve
        // against the working directory, so only self-contained data (a GLB, or
        // a .gltf using data: URIs) loads without one.
        std::optional<GLTFResult> load(const std::vector<uint8_t>& data, const std::filesystem::path& basePath = {});

    private:
        struct Impl;
    };

} // namespace threepp

#endif
