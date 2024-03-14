
#ifndef THREEPP_MAPVIEW_HPP
#define THREEPP_MAPVIEW_HPP

#include "threepp/objects/Mesh.hpp"

#include "geo/lod/LODControl.hpp"
#include "geo/lod/LODRaycast.hpp"
#include "geo/providers/MapProvider.hpp"

namespace threepp {

    class MapView: public Mesh {

    public:
        std::unique_ptr<LODControl> lod;
        std::unique_ptr<MapProvider> provider;

        MapView(std::unique_ptr<MapProvider> provider): provider(std::move(provider)) {

            onBeforeRender = [](void*, Object3D*, Camera*, BufferGeometry*, Material*, std::optional<GeometryGroup>) {

            };

            lod = std::make_unique<LODRaycast>();
        }

        float minZoom() {
            return this->provider->minZoom;
        }

        float maxZoom() {
            return this->provider->maxZoom;
        }


    private:
    };

}// namespace threepp

#endif//THREEPP_MAPVIEW_HPP
