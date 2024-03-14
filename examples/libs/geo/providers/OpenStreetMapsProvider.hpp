
#ifndef THREEPP_OPENSTREETMAPSPROVIDER_HPP
#define THREEPP_OPENSTREETMAPSPROVIDER_HPP

#include "geo/providers/MapProvider.hpp"

#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/utils/URLFetcher.hpp"

#include <sstream>
#include <utility>

namespace threepp {

    class OpenStreetMapProvider: public MapProvider {
    public:
        explicit OpenStreetMapProvider(std::string address = "https://a.tile.openstreetmap.org/")
            : address(std::move(address)) {}

        [[nodiscard]] std::string name() const override {
            return "OpenStreetMapProvider";
        }

        Image fetchTile(float zoom, float x, float y) override {

            std::stringstream ss;
            ss << address << zoom << '/' << x << '/' << y << '.' << format;

            std::vector<unsigned char> data;
            urlFetcher.fetch(ss.str(), data);

            return *loader.load(data, format == "png" ? 4 : 3, true);
        }

    private:
        std::string address;
        std::string format = "png";

        float maxZoom = 19;

        utils::UrlFetcher urlFetcher;
        ImageLoader loader;
    };

}// namespace threepp

#endif//THREEPP_OPENSTREETMAPSPROVIDER_HPP
