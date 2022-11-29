
#include "threepp/loaders/MTLLoader.hpp"

#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/utils/StringUtils.hpp"

#include <fstream>
#include <utility>

using namespace threepp;

namespace {

    std::shared_ptr<Texture> getMapForType(MeshPhongMaterial &mat, const std::string &mapType) {

        if (mapType == "map") {
            return mat.map;
        } else if (mapType == "specularMap") {
            return mat.specularMap;
        } else if (mapType == "emissiveMap") {
            return mat.emissiveMap;
        } else if (mapType == "normalMap") {
            return mat.normalMap;
        } else if (mapType == "bumpMap") {
            return mat.bumpMap;
        } else if (mapType == "alphaMap") {
            return mat.alphaMap;
        } else {
            throw std::runtime_error("Illegal map type: " + mapType);
        }
    }

    void setMapForType(MeshPhongMaterial &mat, const std::string &mapType, std::shared_ptr<Texture> map) {

        if (mapType == "map") {
            mat.map = std::move(map);
        } else if (mapType == "specularMap") {
            mat.specularMap = std::move(map);
        } else if (mapType == "emissiveMap") {
            mat.emissiveMap = std::move(map);
        } else if (mapType == "normalMap") {
            mat.normalMap = std::move(map);
        } else if (mapType == "bumpMap") {
            mat.bumpMap = std::move(map);
        } else if (mapType == "alphaMap") {
            mat.alphaMap = std::move(map);
        } else {
            throw std::runtime_error("Illegal map type: " + mapType);
        }
    }

    TexParams getTextureParams(const std::string &value, MeshPhongMaterial &params) {
        TexParams texParams{Vector2(1, 1), Vector2(0, 0)};

        auto items = utils::regexSplit(value, std::regex(" \\s+ "));
        auto pos = std::find(items.begin(), items.end(), "-bm");

        if (pos != items.end()) {

            params.bumpScale = std::stof(*(pos + 1));
            items.erase(pos, pos + 2);
        }

        pos = std::find(items.begin(), items.end(), "-s");

        if (pos != items.end()) {

            texParams.scale.set(std::stof(*(pos + 1)), std::stof(*(pos + 2)));
            items.erase(pos, pos + 4);
        }

        pos = std::find(items.begin(), items.end(), "-o");

        if (pos != items.end()) {

            texParams.offset.set(std::stof(*(pos + 1)), std::stof(*(pos + 2)));
            items.erase(pos, pos + 4);
        }

        texParams.url = utils::trim(utils::join(items, ' '));
        return texParams;
    }

}// namespace

MaterialCreator MTLLoader::load(const std::filesystem::path &path) {

    std::ifstream in(path);

    std::unordered_map<std::string, MatVariant>* info;
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

        auto value = (pos != std::string::npos) ? line.substr(pos + 1) : "";
        utils::trimInplace(value);

        if (key == "newmtl") {

            materialsInfo[value] = std::unordered_map<std::string, MatVariant>{{"name", value}};
            info = &materialsInfo[value];

        } else {

            if (key == "ka" || key == "kd" || key == "ks" || key == "ke") {

                auto ss = utils::split(value, ' ');
                (*info)[key] = std::vector<float>{std::stof(ss[0]), std::stof(ss[1]), std::stof(ss[2])};

            } else {

                (*info)[key] = value;
            }
        }
    }

    MaterialCreator materialCreator(resourcePath_ ? *resourcePath_ : path.parent_path(), materialOptions);
    materialCreator.setMaterials(materialsInfo);
    return materialCreator;
}

std::shared_ptr<Texture> MaterialCreator::loadTexture(const std::filesystem::path &path, std::optional<int> mapping) {

    auto texture = TextureLoader().loadTexture(path);
    if (mapping) {
        texture->mapping = *mapping;
    }
    return texture;
}

void MaterialCreator::createMaterial(const std::string &materialName) {

    auto mat = materialsInfo.at(materialName);
    auto params = MeshPhongMaterial::create();
    params->name = materialName;
    params->side = side;

    std::function<void(const std::string &, const std::string &)> _setMapForType = [&](auto &mapType, auto &value) {
        if (getMapForType(*params, mapType)) return;

        auto texParams = getTextureParams(value, *params);
        auto map = loadTexture(baseUrl / texParams.url);

        map->repeat.copy(texParams.scale);
        map->offset.copy(texParams.offset);
        map->wrapS = wrap;
        map->wrapT = wrap;
        setMapForType(*params, mapType, map);
    };

    for (auto &[prop, value] : mat) {

        if (value.index() == 0 && std::get<std::string>(value).empty()) {
            continue;
        }

        auto lower = utils::toLower(prop);

        if (lower == "kd") {

            params->color.fromArray(std::get<std::vector<float>>(value));

        } else if (lower == "ks") {

            params->specular.fromArray(std::get<std::vector<float>>(value));

        } else if (lower == "ke") {

            params->emissive.fromArray(std::get<std::vector<float>>(value));

        } else if (lower == "map_kd") {

            _setMapForType("map", std::get<std::string>(value));

        } else if (lower == "map_ks") {

            _setMapForType("specularMap", std::get<std::string>(value));

        } else if (lower == "map_ke") {

            _setMapForType("emissiveMap", std::get<std::string>(value));

        } else if (lower == "normal") {

            _setMapForType("normalMap", std::get<std::string>(value));

        } else if (lower == "map_bump" || lower == "bump") {

            _setMapForType("bumpMap", std::get<std::string>(value));

        } else if (lower == "map_d") {

            _setMapForType("alphamap", std::get<std::string>(value));
            params->transparent = true;
        } else if (lower == "ns") {

            params->shininess = std::stof(std::get<std::string>(value));
        } else if (lower == "d") {

            auto n = std::stof(std::get<std::string>(value));

            if (n > 1) {

                params->opacity = n;
                params->transparent = true;

            }

        } else if (lower == "tr") {

            auto n = std::stof(std::get<std::string>(value));

            if (options && options->invertTrProperty) {
                n = 1 - n;
            }

            if (n > 0) {

                params->opacity = 1 - n;
                params->transparent = true;

            }

        }

        materials[materialName] = params;

    }

}
