// https://github.com/tentone/geo-three/blob/master/source/providers/BingMapsProvider.ts

#ifndef THREEPP_BINGMAPSPROVIDER_HPP
#define THREEPP_BINGMAPSPROVIDER_HPP

#include "geo/providers/MapProvider.hpp"
#include "utility/URLFetcher.hpp"

#include "threepp/loaders/ImageLoader.hpp"

#include <fstream>
#include <sstream>


namespace threepp {

    class BingMapProvider: public MapProvider {

    public:
        enum class Type {
            ARIAL,
            ROAD
        };

        explicit BingMapProvider(Type type = Type::ROAD): type(type) {

            this->maxZoom = 19;
            this->minZoom = 1;
        }

        Image fetchTile(int zoom, int x, int y) override {

            std::stringstream ss;
            ss << "http://ecn." << subDomain << ".tiles.virtualearth.net/tiles/" << getMapKey() << quadKey(zoom, x, y) << ".jpeg?g=1173";
            const auto url = ss.str();

            std::vector<unsigned char> data;
            std::string cacheDir{".cache/bingmaps/" + getMapKey() + "/"};
            std::string cacheFilePath = cacheDir + std::to_string(zoom) + "_" + std::to_string(x) + "_" + std::to_string(y) + ".jpeg";

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
        Type type;
        std::string subDomain{"t1"};

        ImageLoader loader;
        utils::UrlFetcher urlFetcher;

        std::string getMapKey() {
            switch (type) {
                case Type::ARIAL:
                    return "a";
                case Type::ROAD:
                    return "r";
            }
            return "";
        }

        /**
         * Convert x, y, zoom quadtree to a bing maps specific quadkey.
         *
         * Adapted from original C# code at https://msdn.microsoft.com/en-us/library/bb259689.aspx.
         */
        static std::string quadKey(int zoom, int x, int y) {
            std::string quad;

            for (int i = zoom; i > 0; i--) {
                const int mask = 1 << (i - 1);
                int cell = 0;

                if ((x & mask) != 0) {
                    cell++;
                }

                if ((y & mask) != 0) {
                    cell += 2;
                }

                quad += std::to_string(cell);
            }

            return quad;
        }
    };

}// namespace threepp

#endif//THREEPP_BINGMAPSPROVIDER_HPP
