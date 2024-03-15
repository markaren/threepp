// https://github.com/tentone/geo-three/blob/master/source/providers/MapProvider.ts

#ifndef THREEPP_MAPPROVIDER_HPP
#define THREEPP_MAPPROVIDER_HPP

#include <optional>
#include <string>

#include "threepp/textures/Image.hpp"

namespace threepp {

    class MapProvider {

    public:
        float minZoom = 0;
        float maxZoom = 20;

        virtual Image fetchTile(float zoom, float x, float y) = 0;

        virtual ~MapProvider() = default;
    };

}// namespace threepp

#endif//THREEPP_MAPPROVIDER_HPP
