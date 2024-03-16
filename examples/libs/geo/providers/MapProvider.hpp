// https://github.com/tentone/geo-three/blob/master/source/providers/MapProvider.ts

#ifndef THREEPP_MAPPROVIDER_HPP
#define THREEPP_MAPPROVIDER_HPP

#include <optional>
#include <string>

#include "threepp/textures/Image.hpp"

namespace threepp {

    class MapProvider {

    public:
        int minZoom = 0;
        int maxZoom = 20;

        virtual Image fetchTile(int zoom, int x, int y) = 0;

        virtual ~MapProvider() = default;
    };

}// namespace threepp

#endif//THREEPP_MAPPROVIDER_HPP
