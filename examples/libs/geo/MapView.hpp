// https://github.com/tentone/geo-three/blob/master/source/MapView.ts

#ifndef THREEPP_MAPVIEW_HPP
#define THREEPP_MAPVIEW_HPP

#include "threepp/objects/Mesh.hpp"

#include "geo/lod/LODControl.hpp"
#include "geo/nodes/MapPlaneNode.hpp"
#include "geo/providers/MapProvider.hpp"

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

        void setProvider(std::unique_ptr<MapProvider> provider) {
            this->provider = std::move(provider);
            clear();
        }

        MapProvider* getProvider() const;

        void clear() {
            traverse([&](Object3D& object) {
                if (auto node = object.as<MapNode>()) {
                    node->initialize();
                }
            });
        }

        void raycast(const Raycaster& raycaster, std::vector<Intersection>& intersects) override {}

    private:
        std::unique_ptr<MapPlaneNode> root;
        std::unique_ptr<LODControl> lod;
        std::unique_ptr<MapProvider> provider;
    };


}// namespace threepp

#endif//THREEPP_MAPVIEW_HPP
