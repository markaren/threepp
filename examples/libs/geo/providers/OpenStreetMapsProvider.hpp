// https://github.com/tentone/geo-three/blob/master/source/providers/OpenStreetMapsProvider.ts

#ifndef THREEPP_OPENSTREETMAPSPROVIDER_HPP
#define THREEPP_OPENSTREETMAPSPROVIDER_HPP

#include "../../utility/URLFetcher.hpp"
#include "../providers/MapProvider.hpp"

#include "threepp/loaders/ImageLoader.hpp"

#include <fstream>
#include <sstream>
#include <utility>

namespace threepp {

    class OpenStreetMapProvider: public MapProvider {

    public:
        explicit OpenStreetMapProvider(std::string address = "https://a.tile.openstreetmap.org/")
            : address(std::move(address)) {

            this->maxZoom = 19;
        }

        Image fetchTile(int zoom, int x, int y) override {

            std::stringstream ss;
            ss << address << zoom << '/' << x << '/' << y << '.' << format;
            const auto url = ss.str();

            std::vector<unsigned char> data;
            std::string cacheFilePath = ".cache/openstreetmaps/" + std::to_string(zoom) + "_" + std::to_string(x) + "_" + std::to_string(y) + "." + format;

            if (std::filesystem::exists(cacheFilePath)) {
                // Load from cache file
                std::ifstream file(cacheFilePath, std::ios::binary);
                data = std::vector<unsigned char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            } else if (urlFetcher.fetch(url, data)) {
                // Save to cache file
                std::filesystem::create_directories(".cache/openstreetmaps/");
                std::ofstream file(cacheFilePath, std::ios::binary);
                file.write(reinterpret_cast<const char*>(data.data()), data.size());
            }

            return *loader.load(data, format == "png" ? 4 : 3, true);
        }

    private:
        std::string address;
        std::string format = "png";

        ImageLoader loader;
        utils::UrlFetcher urlFetcher;

        std::unordered_map<std::string, std::vector<unsigned char>> cache_{};
    };

}// namespace threepp

#endif//THREEPP_OPENSTREETMAPSPROVIDER_HPP
