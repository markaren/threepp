// https://github.com/tentone/geo-three/blob/master/source/providers/OpenStreetMapsProvider.ts

#ifndef THREEPP_OPENSTREETMAPSPROVIDER_HPP
#define THREEPP_OPENSTREETMAPSPROVIDER_HPP

#include "../../utility/URLFetcher.hpp"
#include "../providers/MapProvider.hpp"

#include "threepp/loaders/ImageLoader.hpp"

#include <iostream>
#include <mutex>
#include <sstream>
#include <utility>

namespace threepp {

    class OpenStreetMapProvider: public MapProvider {

    public:
        explicit OpenStreetMapProvider(std::string address = "https://a.tile.openstreetmap.org/")
            : address(std::move(address)) {

            this->maxZoom = 19;
        }

        Image fetchTile(float zoom, float x, float y) override {

            std::stringstream ss;
            ss << address << zoom << '/' << x << '/' << y << '.' << format;
            const auto url = ss.str();

            std::vector<unsigned char> data;

            if (cache_.count(url)) {
                data = cache_.at(url);

            } else if (urlFetcher.fetch(url, data)) {

                cache_[url] = data;
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
