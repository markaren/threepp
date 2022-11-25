
#ifndef THREEPP_MTLLOADER_HPP
#define THREEPP_MTLLOADER_HPP

#include <optional>
#include <string>
#include <utility>
#include <unordered_map>

#include "threepp/constants.hpp"
#include "threepp/math/Vector2.hpp"
#include "threepp/materials/Material.hpp"

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

    class MaterialCreator {

    private:
        std::string baseUrl;
        std::optional<MaterialOptions> options;

        int side;
        int wrap;

        std::unordered_map<std::string, std::shared_ptr<Material>> materials;
        std::vector<std::shared_ptr<Material>> materialsArray;
        std::unordered_map<std::string, int> nameLookup;

    public:
        explicit MaterialCreator(
                std::string  baseUrl = "",
                std::optional<MaterialOptions> options = std::nullopt)
            : baseUrl(std::move(baseUrl)),
              options(std::move(options)) {

            side = options ? options->side : FrontSide;
            wrap = options ? options->wrap : RepeatWrapping;
        }

    };


    class MTLoader {

    private:
        std::string path;
        std::string resourcePath;
    };

}// namespace threepp

#endif//THREEPP_MTLLOADER_HPP
