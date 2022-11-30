
#ifndef THREEPP_MTLLOADER_HPP
#define THREEPP_MTLLOADER_HPP

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>

#include "threepp/constants.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/math/Vector2.hpp"


namespace threepp {

    struct TexParams {
        Vector2 scale;
        Vector2 offset;
        std::string url;
    };

    struct MaterialOptions {

        std::string name;
        int side = FrontSide;
        int wrap = RepeatWrapping;
        bool normalizeRGB = false;
        bool ignoreZeroRGBs = false;
        bool invertTrProperty = false;
    };

    typedef std::variant<std::string, std::vector<float>> MatVariant;
    typedef std::unordered_map<std::string, std::unordered_map<std::string, MatVariant>> MaterialsInfo;

    class MaterialCreator {

    private:
        std::filesystem::path baseUrl;
        std::optional<MaterialOptions> options;

        int side;
        int wrap;

        std::unordered_map<std::string, std::shared_ptr<Material>> materials;
        std::vector<std::shared_ptr<Material>> materialsArray;
        MaterialsInfo materialsInfo;
        std::unordered_map<std::string, int> nameLookup;

    public:
        explicit MaterialCreator(
                std::filesystem::path baseUrl = "",
                std::optional<MaterialOptions> options = std::nullopt)
            : baseUrl(std::move(baseUrl)),
              options(std::move(options)) {

            side = this->options ? this->options->side : FrontSide;
            wrap = this->options ? this->options->wrap : RepeatWrapping;
        }

        MaterialsInfo convert(const MaterialsInfo &mi);

        MaterialCreator &preload() {
            for (auto &[mn, _] : materialsInfo) {
                create(mn);
            }
            return *this;
        }

        void setMaterials(const MaterialsInfo &mi) {
            this->materialsInfo = convert(mi);
        }

        std::shared_ptr<Material> create(const std::string &materialName) {

            if (materials.find(materialName) == materials.end()) {
                createMaterial(materialName);
            }

            return materials.at(materialName);
        }


    private:
        void createMaterial(const std::string &materialName);

        std::shared_ptr<Texture> loadTexture(const std::filesystem::path &path, std::optional<int> mapping = std::nullopt);
    };


    class MTLLoader {

    private:
        std::optional<std::filesystem::path> path_;
        std::optional<std::filesystem::path> resourcePath_;

    public:
        std::optional<MaterialOptions> materialOptions;

        void setPath(const std::filesystem::path &path) {
            path_ = path;
        }

        void setResourcePath(const std::filesystem::path &resourcePath) {
            resourcePath_ = resourcePath;
        }

        MaterialCreator load(const std::filesystem::path &path);
    };

}// namespace threepp

#endif//THREEPP_MTLLOADER_HPP
