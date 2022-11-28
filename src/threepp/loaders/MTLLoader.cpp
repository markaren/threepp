
#include "threepp/loaders/MTLLoader.hpp"

#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/utils/StringUtils.hpp"
#include "threepp/loaders/TextureLoader.hpp"

#include <fstream>

using namespace threepp;

MaterialCreator MTLLoader::load(std::filesystem::path &path) {

    std::ifstream in(path);

    std::unordered_map<std::string, MatVariant> info;
    std::regex delimiterPattern{"\\s+"};
    MaterialsInfo materialsInfo;

    std::string line;
    while (std::getline(in, line)) {

        utils::trimInplace(line);

        if (line.empty() || line.front() == '#') {
            continue;
        }

        auto pos = line.find(' ');

        auto key = (pos != std::string::npos) ? line.substr(0, pos) : line;
        utils::toLowerInplace(key);

        auto value = (pos != std::string::npos) ? line.substr(pos+1) : "";
        utils::trimInplace(value);

        if (key == "newmtl") {

            info = std::unordered_map<std::string, MatVariant> {{"name", value}};
            materialsInfo[value] = info;

        } else {

            if (key == "ka" || key == "kd" || key == "ks" || key == "ke") {

                auto ss = utils::split(value, ' ');
                info[key] = std::vector<float>{std::stof(ss[0]), std::stof(ss[1]), std::stof(ss[2])};

            } else {

                info[key] = value;

            }

        }

    }

    MaterialCreator materialCreator(resourcePath_ ? *resourcePath_ : path, materialOptions);
    materialCreator.setMaterials(materialsInfo);
    return materialCreator;
}

TexParams MaterialCreator::getTextureParams(const std::string &value, MaterialParams &params) {
    TexParams texParams{Vector2(1, 1), Vector2(0, 0)};

    auto items = utils::regexSplit(value, std::regex(" \\s+ "));
    auto pos = std::find(items.begin(), items.end(), "-bm");

    if (pos != items.end()) {

        params.bumpScale = std::stof(*(pos+1));
        items.erase(pos, pos+2);

    }

    pos = std::find(items.begin(), items.end(), "-s");

    if (pos != items.end()) {

        texParams.scale.set(std::stof(*(pos+1)), std::stof(*(pos+2)));
        items.erase(pos, pos+4);

    }

    pos = std::find(items.begin(), items.end(), "-o");

    if (pos != items.end()) {

        texParams.offset.set(std::stof(*(pos+1)), std::stof(*(pos+2)));
        items.erase(pos, pos+4);

    }

    texParams.url = utils::trim(utils::join(items, ' '));
    return texParams;
}


std::shared_ptr<Texture> MaterialCreator::loadTexture(const std::filesystem::path &path, std::optional<int> mapping) {

    auto texture = TextureLoader().loadTexture(path);
    texture->mapping = mapping;
    return texture;
}

void MaterialCreator::createMaterial(const std::string &materialName) {

    auto mat = materialsInfo.at(materialName);
    MaterialParams params;
    params.name = materialName;
    params.side = side;

    std::function<void(const std::string &, const std::string &)> setMapForType = [&](auto &mapType, auto &value) {
        if (params[mapType]) return;

        auto texParams = getTextureParams(value, params);
        auto map = loadTexture(baseUrl.string() + texParams.url);

        map->repeat.copy(texParams.scale);
        map->offset.copy(texParams.offset);
        map->wrapS = wrap;
        map->wrapT = wrap;
        params[mapType] = map;

    };

    for (auto& [key, value] : mat) {



    }

}

