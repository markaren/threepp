// https://github.com/tentone/geo-three/blob/master/source/nodes/MapNode.ts

#ifndef THREEPP_MAPNODE_HPP
#define THREEPP_MAPNODE_HPP

#include "threepp/objects/Mesh.hpp"

namespace threepp {

    class MapView;

    struct QuadTreePosition {

        static inline int root = -1;
        static inline int topLeft = 0;
        static inline int topRight = 1;
        static inline int bottomLeft = 2;
        static inline int bottomRight = 3;
    };

    class MapNode: public Mesh {

    public:
        MapNode* parentNode = nullptr;
        bool remove = false;

        MapNode(MapNode* parentNode, MapView* mapView,
                int location = QuadTreePosition::root,
                int level = 0, int x = 0, int y = 0,
                const std::shared_ptr<BufferGeometry>& geometry = nullptr,
                const std::shared_ptr<Material>& material = nullptr);

        virtual void initialize() = 0;

        virtual void createChildNodes() = 0;

        void subdivide();

        void simplify();

        void loadData();

        void nodeReady();

        [[nodiscard]] virtual Vector3 baseScale() const = 0;

        virtual std::shared_ptr<BufferGeometry> baseGeometry() = 0;

        [[nodiscard]] int getLevel() const {

            return level;
        }

        ~MapNode() override = default;

    protected:
        MapView* mapView;

        int location;
        int level;

        int x;
        int y;

        bool subdivided = false;
        bool disposed = false;

        int nodesLoaded = 0;

        inline static int childrens = 4;
    };

}// namespace threepp

#endif//THREEPP_MAPNODE_HPP
