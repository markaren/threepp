// https://github.com/mrdoob/three.js/blob/r150/examples/jsm/math/ConvexHull.js

#ifndef THREEPP_CONVEXHULL_HPP
#define THREEPP_CONVEXHULL_HPP

#include "threepp/math/Vector3.hpp"

#include <memory>
#include <vector>

namespace threepp {

    struct Face;

    // A vertex as a double linked list node.

    struct VertexNode {

        Vector3 point;
        VertexNode* prev = nullptr;
        VertexNode* next = nullptr;
        Face* face = nullptr;

        explicit VertexNode(const Vector3& point): point(point) {}
    };

    // A double linked list that contains vertex nodes.

    struct VertexList {

        VertexNode* head = nullptr;
        VertexNode* tail = nullptr;

        [[nodiscard]] VertexNode* first() const;

        [[nodiscard]] VertexNode* last() const;

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
        int mark;
        std::shared_ptr<HalfEdge> edge = nullptr;

        static std::shared_ptr<Face> create(VertexNode* a, VertexNode* b, VertexNode* c);

        [[nodiscard]] std::shared_ptr<HalfEdge> getEdge(int i);

        void compute();

        [[nodiscard]] float distanceToPoint(const Vector3& point) const;

    private:
        Face();
    };

    struct HalfEdge: std::enable_shared_from_this<HalfEdge> {

        VertexNode* vertex;
        std::shared_ptr<HalfEdge> prev = nullptr;
        std::shared_ptr<HalfEdge> next = nullptr;
        std::shared_ptr<HalfEdge> twin = nullptr;
        Face* face = nullptr;

        HalfEdge(VertexNode* vertex, Face* face);

        [[nodiscard]] VertexNode* head() const;

        [[nodiscard]] VertexNode* tail() const;

        [[nodiscard]] float length() const;

        void setTwin(const std::shared_ptr<HalfEdge>& edge);
    };

    class ConvexHull {

    public:
        float tolerance = -1;

        std::vector<std::shared_ptr<Face>> faces;// the generated faces of the convex hull

        ConvexHull& setFromPoints(const std::vector<Vector3>& points);

        bool containsPoint(const Vector3& point);

        ConvexHull& makeEmpty();

    private:
        std::vector<std::shared_ptr<Face>> newFaces;// this array holds the faces that are generated within a single iteration

        VertexList assigned;
        VertexList unassigned;

        std::vector<VertexNode> vertices;// vertices of the hull (internal representation of given geometry data)


        // Adds a vertex to the 'assigned' list of vertices and assigns it to the given face

        ConvexHull& addVertexToFace(VertexNode* vertex, Face* face);

        // Removes a vertex from the 'assigned' list of vertices and from the given face

        ConvexHull& removeVertexFromFace(VertexNode* vertex, Face* face);

        // Removes all the visible vertices that a given face is able to see which are stored in the 'assigned' vertex list

        VertexNode* removeAllVerticesFromFace(Face* face);

        // Removes all the visible vertices that 'face' is able to see

        ConvexHull& deleteFaceVertices(Face* face, Face* absorbingFace);

        // Reassigns as many vertices as possible from the unassigned list to the new faces

        ConvexHull& resolveUnassignedPoints(std::vector<std::shared_ptr<Face>>& newFaces);

        // Computes the extremes of a simplex which will be the initial hull

        std::pair<std::vector<VertexNode*>, std::vector<VertexNode*>> computeExtremes();

        // Computes the initial simplex assigning to its faces all the points
        // that are candidates to form part of the hull

        ConvexHull& computeInitialHull();

        // Removes inactive faces

        ConvexHull& reindexFaces();

        // Finds the next vertex to create faces with the current hull

        VertexNode* nextVertexToAdd();

        // Computes a chain of half edges in CCW order called the 'horizon'.
        // For an edge to be part of the horizon it must join a face that can see
        // 'eyePoint' and a face that cannot see 'eyePoint'.

        ConvexHull& computeHorizon(const Vector3& eyePoint, std::shared_ptr<HalfEdge>& crossEdge, Face* face, std::vector<std::shared_ptr<HalfEdge>>& horizon);

        // Creates a face with the vertices 'eyeVertex.point', 'horizonEdge.tail' and 'horizonEdge.head' in CCW order

        std::shared_ptr<HalfEdge> addAdjoiningFace(VertexNode* eyeVertex, const std::shared_ptr<HalfEdge>& horizonEdge);

        //  Adds 'horizon.length' faces to the hull, each face will be linked with the
        //  horizon opposite face and the face on the left/right

        ConvexHull& addNewFaces(VertexNode* eyeVertex, std::vector<std::shared_ptr<HalfEdge>>& horizon);

        // Adds a vertex to the hull

        ConvexHull& addVertexToHull(VertexNode* eyeVertex);

        ConvexHull& cleanup();

        ConvexHull& compute();
    };

}// namespace threepp

#endif//THREEPP_CONVEXHULL_HPP
