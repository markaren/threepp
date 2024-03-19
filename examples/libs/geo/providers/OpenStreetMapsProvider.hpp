// https://github.com/tentone/geo-three/blob/master/source/providers/OpenStreetMapsProvider.ts

#ifndef THREEPP_OPENSTREETMAPSPROVIDER_HPP
#define THREEPP_OPENSTREETMAPSPROVIDER_HPP

#include "geo/providers/MapProvider.hpp"
#include "utility/URLFetcher.hpp"

#include "threepp/loaders/ImageLoader.hpp"

#include <fstream>
#include <sstream>
#include <utility>

namespace threepp {

    class OpenStreetMapProvider: public MapProvider {

    public:
        explicit OpenStreetMapProvider() {

            this->maxZoom = 19;
        }

        Image fetchTile(int zoom, int x, int y) override {

            std::stringstream ss;
            ss << address << zoom << '/' << x << '/' << y << '.' << format;
            const auto url = ss.str();

            std::vector<unsigned char> data;
            std::string cacheFilePath = cacheDir + std::to_string(zoom) + "_" + std::to_string(x) + "_" + std::to_string(y) + "." + format;

            if (std::filesystem::exists(cacheFilePath)) {
                // Load from cache file
                std::ifstream file(cacheFilePath, std::ios::binary);
                data = std::vector<unsigned char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            } else if (urlFetcher.fetch(url, data)) {
                // Save to cache file
                std::filesystem::create_directories(cacheDir);
                std::ofstream file(cacheFilePath, std::ios::binary);
                file.write(reinterpret_cast<const char*>(data.data()), data.size());
            }

            return *loader.load(data, 4, true);
        }

    private:
        ImageLoader loader;
        utils::UrlFetcher urlFetcher;

        std::string format{"png"};
        std::string cacheDir{".cache/openstreetmaps/"};
        std::string address{"https://a.tile.openstreetmap.org/"};
    };

}// namespace threepp

#endif//THREEPP_OPENSTREETMAPSPROVIDER_HPP
