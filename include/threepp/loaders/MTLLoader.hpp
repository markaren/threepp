
#ifndef THREEPP_MTLLOADER_HPP
#define THREEPP_MTLLOADER_HPP

#include <optional>
#include <string>
#include <utility>
#include <filesystem>
#include <unordered_map>
#include <variant>
#include <array>

#include "threepp/constants.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/utils/StringUtils.hpp"


namespace threepp {


    struct TexParams {
        Vector2 scale;
        Vector2 offset;
        std::string url;
    };

    struct MaterialParams {

        std::string name;
        int side;
        float bumpScale;

        std::shared_ptr<Texture>& operator [](const std::string& mapType) {
            return params_[mapType];
        }

    private:
        std::unordered_map<std::string, std::shared_ptr<Texture>> params_;

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

        MaterialsInfo convert(const MaterialsInfo& materialsInfo) {

            if (!options) return materialsInfo;

            MaterialsInfo converted = MaterialsInfo{};

            for (auto& [key, mat]: materialsInfo) {

                auto covMat = std::unordered_map<std::string, MatVariant>{};

                converted[key] = covMat;

                for (auto& [prop, value] : mat) {

                    bool save = true;
                    auto lprop = utils::toLower(prop);

                    if (lprop == "kd" || lprop == "ka" || lprop == "ks") {

                        if (options->normalizeRGB){
                            auto& v = const_cast<std::vector<float>&>(std::get<std::vector<float>>(value));

                            v[0] /= 255;
                            v[1] /= 255;
                            v[2] /= 255;

                        }

                    }


                }

            }

            return converted;

        }

        void setMaterials(const MaterialsInfo& materialsInfo) {
            this->materialsInfo = convert(materialsInfo);
        }

        size_t getIndex(const std::string& materialName) {
            return nameLookup.at(materialName);
        }

    private:

        void createMaterial(const std::string& materialName);

        TexParams getTextureParams(const std::string& value, MaterialParams& params);

        std::shared_ptr<Texture> loadTexture(const std::filesystem::path& path, std::optional<int> mapping = std::nullopt);


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

        MaterialCreator load(std::filesystem::path& path);

    };

}// namespace threepp

#endif//THREEPP_MTLLOADER_HPP
