
#ifndef THREEPP_MAPNODE_HPP
#define THREEPP_MAPNODE_HPP

#include "threepp/objects/Mesh.hpp"

#include "MapView.hpp"

namespace threepp {

    struct QuadTreePosition {

        static int root;
        static int topLeft;
        static int topRight;
        static int bottomLeft;
        static int bottomRight;
    };

    int QuadTreePosition::root = 1;
    int QuadTreePosition::topLeft = 0;
    int QuadTreePosition::topRight = 1;
    int QuadTreePosition::bottomLeft = 2;
    int QuadTreePosition::bottomRight = 3;

    class MapNode: public Mesh {

    public:
        MapNode* parentNode = nullptr;

        MapNode(MapNode* parentNode, const std::shared_ptr<MapView>& mapView, int location = QuadTreePosition::root)
            : parentNode(parentNode), mapView(mapView) {
        }

        virtual void initialize() = 0;

        virtual void createChildNodes();

        void subdivide() {
            const auto maxZoom = this->mapView->maxZoom();
            if (this->children.size() > 0 || this->level + 1 > maxZoom || this->parentNode && this->parentNode->nodesLoaded < MapNode.childrens) {
                return;
            }

            this->createChildNodes();

            this->subdivided = true;
        }

        void simplify() {
            const auto minZoom = this->mapView->minZoom();
            if (this->level - 1 < minZoom) {
                return;
            }

            // Clear children and reset flags
            this->subdivided = false;
            this->clear();
            this->nodesLoaded = 0;
        }


        [[nodiscard]] int getLevel() const {

            return level;
        }

        ~MapNode() override = default;

    private:
        std::shared_ptr<MapView> mapView;


        int location;
        int level;

        float x;
        float y;

        bool subdivided = false;
        bool disposed = false;

        int nodesLoaded = 0;

        std::shared_ptr<BufferGeometry> baseGeometry;

        int childrens = 4;
    };

}// namespace threepp

#endif//THREEPP_MAPNODE_HPP
