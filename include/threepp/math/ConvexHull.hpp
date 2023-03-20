// https://github.com/mrdoob/three.js/blob/r150/examples/jsm/math/ConvexHull.js

#ifndef THREEPP_CONVEXHULL_HPP
#define THREEPP_CONVEXHULL_HPP

#include "threepp/math/Vector3.hpp"

#include <vector>


namespace threepp {

    struct Face;


    // A vertex as a double linked list node.

    struct VertexNode {

        Vector3 point;
        VertexNode* prev = nullptr;
        VertexNode* next = nullptr;
        std::shared_ptr<Face> face = nullptr;

        explicit VertexNode(const Vector3& point): point(point) {}
    };

    // A double linked list that contains vertex nodes.

    struct VertexList {

        VertexNode* head = nullptr;
        VertexNode* tail = nullptr;

        VertexNode* first() const;

        VertexNode* last() const;

        VertexList& clear();

        // Inserts a vertex before the target vertex

        VertexList& insertBefore(VertexNode* target, VertexNode* vertex);

        // Inserts a vertex after the target vertex

        VertexList& insertAfter(VertexNode* target, VertexNode* vertex);

        // Appends a vertex to the end of the linked list

        VertexList& append(VertexNode* vertex);

        // Appends a chain of vertices where 'vertex' is the head.

        VertexList& appendChain(VertexNode* vertex);

        // Removes a vertex from the linked list

        VertexList& remove(VertexNode* vertex);

        // Removes a list of vertices whose 'head' is 'a' and whose 'tail' is b

        VertexList& removeSubList(VertexNode* a, VertexNode* b);

        [[nodiscard]] bool isEmpty() const;
    };

    struct HalfEdge;

    struct Face {

        Vector3 normal;
        Vector3 midpoint;
        float area = 0;

        float constant = 0;
        VertexNode* outside = nullptr;
        int mark = 0;
        std::shared_ptr<HalfEdge> edge = nullptr;

        static std::shared_ptr<Face> create(VertexNode* a, VertexNode* b, VertexNode* c);

        [[nodiscard]] std::shared_ptr<HalfEdge> getEdge(unsigned int i);

        void compute();

        [[nodiscard]] float distanceToPoint(const Vector3& point) const;

    private:
        Face();
    };

    struct HalfEdge {

        VertexNode* vertex;
        std::shared_ptr<HalfEdge> prev = nullptr;
        std::shared_ptr<HalfEdge> next = nullptr;
        HalfEdge* twin = nullptr;
        std::shared_ptr<Face> face = nullptr;

        HalfEdge(VertexNode* vertex, std::shared_ptr<Face> face);

        [[nodiscard]] VertexNode* head() const;

        [[nodiscard]] VertexNode* tail() const;

        [[nodiscard]] float length() const;

        [[nodiscard]] float lengthSquared() const;

        HalfEdge& setTwin(const std::shared_ptr<HalfEdge>& edge);
    };

    class ConvexHull {

    public:
        float tolerance = -1;

        ConvexHull();

        ConvexHull& setFromPoints(const std::vector<Vector3>& points);

        ConvexHull& makeEmpty();

    private:
        std::vector<Face> faces;   // the generated faces of the convex hull
        std::vector<Face> newFaces;// this array holds the faces that are generated within a single iteration

        std::vector<float> vertices;// vertices of the hull (internal representation of given geometry data)
    };

}// namespace threepp

#endif//THREEPP_CONVEXHULL_HPP
