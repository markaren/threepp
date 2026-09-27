// Shared by the two translation units of ObjectLoader: ObjectLoaderResources.cpp
// builds the document's resource tables, ObjectLoader.cpp the object tree that
// references them. Internal to the library.

#ifndef THREEPP_OBJECTJSONPARSE_HPP
#define THREEPP_OBJECTJSONPARSE_HPP

#include "threepp/animation/AnimationClip.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/textures/Image.hpp"
#include "threepp/textures/Texture.hpp"
#include "threepp/utils/ZipReader.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace threepp::objectjson {

    // Everything the document asks for but threepp cannot represent is
    // collected here as well as logged, so an editor can surface it.
    struct Warnings {

        std::vector<std::string> messages;

        void add(std::string message) {

            std::cerr << "[ObjectLoader] " << message << std::endl;
            messages.push_back(std::move(message));
        }
    };

    using GeometryMap = std::unordered_map<std::string, std::shared_ptr<BufferGeometry>>;
    using MaterialMap = std::unordered_map<std::string, std::shared_ptr<Material>>;
    using TextureMap = std::unordered_map<std::string, std::shared_ptr<Texture>>;
    using ImageMap = std::unordered_map<std::string, std::vector<Image>>;
    using AnimationMap = std::unordered_map<std::string, std::shared_ptr<AnimationClip>>;
    using SkeletonMap = std::unordered_map<std::string, std::shared_ptr<Skeleton>>;

    template<class T>
    T value(const nlohmann::json& j, const char* key, T fallback) {

        if (!j.contains(key) || j.at(key).is_null()) return fallback;
        return j.at(key).get<T>();
    }

    // ------------------------------------------------------------ resolution
    //
    // Where a `url` in the document resolves from. A document is either a file
    // in a directory next to its images/ and buffers/, or a .tpz archive whose
    // members carry those same names — the JSON is identical either way, which
    // is the whole design: the archive is packaging, not a second format.
    //
    // Only the archive side answers here. A directory-backed image stays on
    // ImageLoader's path-taking overload, which sniffs its own file (see
    // decodeImage): handing it bytes instead would change behaviour for every
    // document that ever referenced an image by path.
    struct DocumentSource {

        std::filesystem::path resourcePath;
        const ZipReader* archive{nullptr};

        [[nodiscard]] bool has(const std::string& url) const {

            if (archive) return archive->has(url);

            std::error_code ec;
            return !resourcePath.empty() && std::filesystem::is_regular_file(resourcePath / url, ec);
        }

        [[nodiscard]] std::optional<std::vector<unsigned char>> read(const std::string& url) const {

            if (archive) {

                if (!archive->has(url)) return std::nullopt;
                return archive->read(url);
            }

            const auto path = resourcePath.empty() ? std::filesystem::path(url) : resourcePath / url;

            std::ifstream in(path, std::ios::binary | std::ios::ate);
            if (!in) return std::nullopt;

            const std::streamoff end = in.tellg();
            if (end < 0) return std::nullopt;

            std::vector<unsigned char> bytes(static_cast<std::size_t>(end));
            in.seekg(0, std::ios::beg);
            if (end > 0) {

                in.read(reinterpret_cast<char*>(bytes.data()), end);
                if (in.gcount() != end) return std::nullopt;
            }

            return bytes;
        }
    };

    // The binary sections geometry attributes point into, read once each. A
    // geometry's attributes and its index all name the same section, so without
    // this a mesh with five attributes would read (and, in an archive, copy out
    // of the loaded archive) the same megabytes five times over.
    //
    // A section that cannot be read is remembered as absent, so the warning is
    // one per section rather than one per attribute.
    class BufferCache {

    public:
        explicit BufferCache(const DocumentSource& source): source_(source) {}

        const std::vector<unsigned char>* get(const std::string& url, Warnings& warnings) {

            const auto it = sections_.find(url);
            if (it != sections_.end()) return it->second ? &*it->second : nullptr;

            auto bytes = source_.read(url);
            if (!bytes) warnings.add("cannot read binary section '" + url + "'");

            const auto inserted = sections_.emplace(url, std::move(bytes)).first;
            return inserted->second ? &*inserted->second : nullptr;
        }

    private:
        const DocumentSource& source_;
        std::unordered_map<std::string, std::optional<std::vector<unsigned char>>> sections_;
    };

    inline Matrix4 matrixFrom(const nlohmann::json& j) {

        Matrix4 m;
        if (j.is_array() && j.size() == 16) {
            for (size_t i = 0; i < 16; ++i) m.elements[i] = j[i].get<float>();
        }
        return m;
    }

    inline Vector2 vector2From(const nlohmann::json& j, const Vector2& fallback = {}) {

        if (j.is_array() && j.size() >= 2) return {j[0].get<float>(), j[1].get<float>()};
        return fallback;
    }

    inline Color colorFrom(const nlohmann::json& j) {

        Color c;
        c.setHex(j.get<unsigned int>());
        return c;
    }

    inline void applyUserData(Object3D& object, const nlohmann::json& j, Warnings& warnings) {

        if (!j.is_object()) return;

        for (auto it = j.begin(); it != j.end(); ++it) {
            const auto& v = it.value();
            if (v.is_boolean()) {
                object.userData[it.key()] = v.get<bool>();
            } else if (v.is_number_integer() || v.is_number_unsigned()) {
                // JSON has no signed/unsigned distinction, so whole numbers come
                // back as int unless they do not fit.
                const auto n = v.get<std::int64_t>();
                if (n >= std::numeric_limits<int>::min() && n <= std::numeric_limits<int>::max()) {
                    object.userData[it.key()] = static_cast<int>(n);
                } else {
                    object.userData[it.key()] = n;
                }
            } else if (v.is_number_float()) {
                object.userData[it.key()] = v.get<double>();
            } else if (v.is_string()) {
                object.userData[it.key()] = v.get<std::string>();
            } else {
                warnings.add("skipping userData entry '" + it.key() + "': unsupported JSON type");
            }
        }
    }

    inline void applyLayers(Object3D& object, unsigned int mask) {

        object.layers.disableAll();
        for (unsigned int channel = 0; channel < 32; ++channel) {
            if (mask & (1u << channel)) object.layers.enable(channel);
        }
    }

    // ------------------------------------------------ resource tables
    //
    // Defined in ObjectLoaderResources.cpp.

    ImageMap parseImages(const nlohmann::json& j, const DocumentSource& source, Warnings& warnings);

    TextureMap parseTextures(const nlohmann::json& j, const ImageMap& images);

    MaterialMap parseMaterials(const nlohmann::json& j, const TextureMap& textures, Warnings& warnings);

    GeometryMap parseGeometries(const nlohmann::json& j, BufferCache& buffers, Warnings& warnings);

    AnimationMap parseAnimations(const nlohmann::json& j, Warnings& warnings);

}// namespace threepp::objectjson

#endif//THREEPP_OBJECTJSONPARSE_HPP
