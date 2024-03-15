// https://github.com/tentone/geo-three/blob/master/source/MapView.ts

#ifndef THREEPP_MAPVIEW_HPP
#define THREEPP_MAPVIEW_HPP

#include "threepp/objects/Mesh.hpp"

#include "geo/lod/LODControl.hpp"
#include "nodes/MapPlaneNode.hpp"
#include "providers/MapProvider.hpp"

namespace threepp {

    class MapView: public Mesh {

    public:

        explicit MapView(std::unique_ptr<MapProvider> provider, std::unique_ptr<LODControl> lod);

        void preSubDivide();

        [[nodiscard]] float minZoom() const {
            return this->provider->minZoom;
        }

        [[nodiscard]] float maxZoom() const {
            return this->provider->maxZoom;
        }

        MapProvider* getProvider() const;

        void raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) override {}

    private:

        std::unique_ptr<MapPlaneNode> root;
        std::unique_ptr<LODControl> lod;
        std::unique_ptr<MapProvider> provider;
    };


}// namespace threepp

#endif//THREEPP_MAPVIEW_HPP
